"""Baseline greedy engine for Qwen3-4B-Instruct-2507 on a single H100.

Left-pads the batch, prefills once, then decodes greedily with the HF KV
cache. RMSNorm is served by a fused Triton kernel (triton_rmsnorm.py).
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from triton_rmsnorm import patch_qwen3_rmsnorm


class Engine:
    def __init__(self, model_path: str) -> None:
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

    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        batch = len(input_ids)
        max_len = max(len(seq) for seq in input_ids)

        ids = torch.full((batch, max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((batch, max_len), dtype=torch.long)
        for i, seq in enumerate(input_ids):
            ids[i, max_len - len(seq):] = torch.tensor(seq, dtype=torch.long)
            mask[i, max_len - len(seq):] = 1
        ids = ids.to(self.device)
        mask = mask.to(self.device)

        # Left padding shifts absolute positions, so pass them explicitly.
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)

        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1)
        next_pos = position_ids[:, -1:] + 1

        for _ in range(max_new_tokens):
            yield next_tok.tolist()

            mask = torch.cat(
                [mask, torch.ones((batch, 1), dtype=mask.dtype, device=self.device)],
                dim=-1,
            )
            out = self.model(
                input_ids=next_tok.unsqueeze(-1),
                attention_mask=mask,
                position_ids=next_pos,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1)
            next_pos = next_pos + 1
