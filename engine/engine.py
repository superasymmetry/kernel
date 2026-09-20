"""Greedy engine for Qwen3-4B-Instruct-2507 on a single H100.

Decode is memory-bound (~8 GB of bf16 weights per step), so the only thing
that matters is keeping the GPU fed. Three things do that here:

  * StaticCache             -- no per-step torch.cat on 36 layers of K/V
  * preallocated masks      -- every decode input keeps a constant shape
  * torch.compile(reduce-overhead) -- captures the step as a CUDA graph, so
                               ~900 kernel launches collapse into one replay

Tokens stay on the GPU and are copied back one chunk at a time, so the decode
loop never blocks on a device->host sync.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, StaticCache

# Device->host transfers per SYNC_CHUNK steps instead of per step. The
# generator still yields one list[int] per step, in order; it just refills its
# buffer in bursts rather than stalling the pipeline 256 times.
SYNC_CHUNK = 32

# StaticCache length must be stable across calls or torch.compile recaptures.
CACHE_GRANULARITY = 256


class Engine:
    def __init__(self, model_path: str, use_triton_rmsnorm: bool = False) -> None:
        # The hand-written Triton RMSNorm is a graph break under fullgraph
        # compile, and at 8 rows x 2560 the launch costs more than the work.
        # Inductor fuses RMSNorm itself. Off by default; flip to compare.
        if use_triton_rmsnorm:
            from triton_rmsnorm import patch_qwen3_rmsnorm

            patch_qwen3_rmsnorm()

        self.device = torch.device("cuda")
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.model.to(self.device)
        self.model.eval()

        # Any id works for left padding: those positions are masked out and
        # the cache slots they occupy are never attended to.
        self.pad_id = self.config.eos_token_id
        if isinstance(self.pad_id, (list, tuple)):
            self.pad_id = self.pad_id[0]

        # Prefill keeps dynamic shapes (prompt length varies); only the decode
        # step is compiled, and it is the one that runs max_new_tokens times.
        self.decode_step = torch.compile(
            self.model, mode="reduce-overhead", fullgraph=True
        )

        self._cache: StaticCache | None = None
        self._cache_key: tuple[int, int] | None = None

    def _get_cache(self, batch: int, cache_len: int) -> StaticCache:
        # StaticCache binds its batch size on first use, so a call with a
        # different batch needs a fresh cache -- reset() alone would leave
        # buffers shaped for the previous batch.
        key = (batch, cache_len)
        if self._cache is None or self._cache_key != key:
            self._cache = StaticCache(config=self.config, max_cache_len=cache_len)
            self._cache_key = key
        else:
            self._cache.reset()
        return self._cache

    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        batch = len(input_ids)
        max_len = max(len(seq) for seq in input_ids)

        total = max_len + max_new_tokens
        cache_len = -(-total // CACHE_GRANULARITY) * CACHE_GRANULARITY
        past = self._get_cache(batch, cache_len)

        # Build the prompt on the host, then one transfer for each tensor.
        ids = torch.full((batch, max_len), self.pad_id, dtype=torch.long)
        prompt_mask = torch.zeros((batch, max_len), dtype=torch.long)
        for i, seq in enumerate(input_ids):
            ids[i, max_len - len(seq):] = torch.tensor(seq, dtype=torch.long)
            prompt_mask[i, max_len - len(seq):] = 1
        ids = ids.to(self.device, non_blocking=True)

        # One full-width mask, allocated once: decode flips a single column per
        # step instead of reallocating via torch.cat, which keeps the decode
        # input shape constant and lets the CUDA graph be reused.
        mask = torch.zeros((batch, cache_len), dtype=torch.long, device=self.device)
        mask[:, :max_len] = prompt_mask.to(self.device, non_blocking=True)

        # Left padding shifts absolute positions, so pass them explicitly. Note
        # position_ids (for RoPE) and cache_position (the physical cache slot)
        # deliberately differ under left padding.
        position_ids = (mask[:, :max_len].cumsum(-1) - 1).clamp(min=0)
        cache_position = torch.arange(max_len, device=self.device)

        out = self.model(
            input_ids=ids,
            attention_mask=mask[:, :max_len],
            position_ids=position_ids,
            past_key_values=past,
            cache_position=cache_position,
            use_cache=True,
            # Prefill only needs the last position; without this we run a
            # (batch*max_len) x 151936 GEMM and throw all but one row away.
            logits_to_keep=1,
        )
        next_tok = out.logits[:, -1, :].argmax(dim=-1)
        next_pos = position_ids[:, -1:] + 1
        cache_position = torch.tensor([max_len], device=self.device)

        # Tokens accumulate on device; nothing here forces a sync.
        tokens = torch.empty(
            (batch, max_new_tokens), dtype=torch.long, device=self.device
        )

        chunk_start = 0
        chunk_cpu = None
        for step in range(max_new_tokens):
            tokens[:, step] = next_tok

            # Flush a completed chunk: one transfer, then hand out its rows.
            if step - chunk_start + 1 == SYNC_CHUNK or step == max_new_tokens - 1:
                chunk_cpu = tokens[:, chunk_start:step + 1].t().tolist()
                for row in chunk_cpu:
                    yield row
                chunk_start = step + 1

            if step == max_new_tokens - 1:
                break

            mask[:, max_len + step] = 1
            out = self.decode_step(
                input_ids=next_tok.unsqueeze(-1),
                attention_mask=mask,
                position_ids=next_pos,
                past_key_values=past,
                cache_position=cache_position,
                use_cache=True,
            )
            # reduce-overhead hands back graph-owned storage that the next
            # replay overwrites, so take a copy before looping.
            next_tok = out.logits[:, -1, :].argmax(dim=-1).clone()
            next_pos = next_pos + 1
            cache_position = cache_position + 1
