"""Greedy engine for Qwen3-4B-Instruct-2507 on a single H100.

Decode is memory-bound (~8 GB of bf16 weights streamed per step), so the
whole design is about not wasting the ~2.4 ms that one step costs:

  * StaticCache, so the KV cache is never reallocated or copied.
  * A CUDA-graph-captured decode step (torch.compile reduce-overhead), so
    Python dispatch over 36 layers is not on the critical path.
  * Static shapes everywhere -- the attention mask is a full-length buffer
    written in place, never grown with torch.cat.
  * The device->host sync for each token is overlapped with the launch of
    the next step, instead of draining the pipeline every token.
  * Prefill keeps only the last position's logits.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, StaticCache

from triton_rmsnorm import patch_qwen3_rmsnorm

# Round the cache up to a multiple of this, so that a range of prompt and
# generation lengths reuses one set of compiled graphs.
CACHE_BUCKET = 256


class Engine:
    def __init__(
        self,
        model_path: str,
        compile: bool = True,
        triton_rmsnorm: bool = False,
        warmup: tuple[int, int] | None = None,
    ) -> None:
        # The custom op is opaque to inductor, so it blocks the fusions that
        # make the compiled path fast. Only patch it in when running eager.
        if triton_rmsnorm and not compile:
            patch_qwen3_rmsnorm()

        self.device = torch.device("cuda")
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=self.device,
        )
        self.model.eval()

        # Any id works for left padding: those positions are masked out and
        # the cache slots they occupy are never attended to.
        self.pad_id = self.config.eos_token_id
        if isinstance(self.pad_id, (list, tuple)):
            self.pad_id = self.pad_id[0]

        self.compiled = compile
        self._step = self._decode_step
        if compile:
            self._step = torch.compile(
                self._decode_step, mode="reduce-overhead", fullgraph=True, dynamic=False
            )

        self.cache: StaticCache | None = None
        self.cache_len = 0
        self.mask: torch.Tensor | None = None

        if warmup is not None:
            self.warmup(*warmup)

    # -- buffers ---------------------------------------------------------

    def _ensure_capacity(self, batch: int, total_len: int) -> None:
        """(Re)allocate the cache and mask buffer, bucketed so shapes are stable."""
        need = -(-total_len // CACHE_BUCKET) * CACHE_BUCKET
        if self.cache is not None and self.cache_len == need and self.mask.shape[0] == batch:
            self.cache.reset()
            self.mask.zero_()
            return
        self.cache = StaticCache(config=self.config, max_cache_len=need)
        self.cache_len = need
        self.mask = torch.zeros((batch, need), dtype=torch.long, device=self.device)

    # -- the hot path ----------------------------------------------------

    def _decode_step(self, tok, mask, position_ids, cache_position):
        """One token for the whole batch. Captured by CUDA graphs when compiled.

        Returns the argmax on device -- never a Python value, so the caller
        decides when to pay for the sync.
        """
        out = self.model(
            input_ids=tok,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        return out.logits[:, -1, :].argmax(dim=-1)

    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        batch = len(input_ids)
        prompt_len = max(len(seq) for seq in input_ids)
        self._ensure_capacity(batch, prompt_len + max_new_tokens)

        ids = torch.full((batch, prompt_len), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(input_ids):
            ids[i, prompt_len - len(seq):] = torch.tensor(seq, dtype=torch.long)
            self.mask[i, prompt_len - len(seq):prompt_len] = 1
        ids = ids.to(self.device, non_blocking=True)

        # Left padding shifts absolute positions, so pass them explicitly.
        prompt_mask = self.mask[:, :prompt_len]
        position_ids = (prompt_mask.cumsum(-1) - 1).clamp(min=0)
        cache_position = torch.arange(prompt_len, device=self.device)

        out = self.model(
            input_ids=ids,
            attention_mask=self.mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,  # (B, S, 151936) for the whole prompt is pure waste
        )
        tok = out.logits[:, -1, :].argmax(dim=-1)
        next_pos = position_ids[:, -1:] + 1
        cache_position = torch.tensor([prompt_len], device=self.device)

        for i in range(max_new_tokens):
            # Launch step i+1 before syncing on step i, so the device->host
            # copy overlaps with real work instead of draining the pipeline.
            # Nothing is launched on the last iteration -- that forward pass
            # would produce a token no one consumes.
            if i + 1 < max_new_tokens:
                self.mask[:, prompt_len + i] = 1
                nxt = self._step(tok.unsqueeze(-1), self.mask, next_pos, cache_position)
                next_pos = next_pos + 1
                cache_position = cache_position + 1

            yield tok.tolist()

            if i + 1 < max_new_tokens:
                tok = nxt

    # -- keep compilation off the timed path -----------------------------

    def warmup(self, batch: int, total_len: int) -> None:
        """Compile and capture graphs ahead of time.

        Call this with the harness's batch size and (prompt + generated)
        length before any timed run, or the first measured request pays for
        inductor compilation and CUDA graph capture.
        """
        prompt = [[self.pad_id] * max(1, total_len - 8)] * batch
        for _ in self.generate(prompt, 8):
            pass
        torch.cuda.synchronize()
