"""Checkpointed, chunked policy scoring without materializing [B, T, V] logits."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class QwenPolicyLogprobs(torch.nn.Module):
    """Expose causal token_logprobs while retaining a standard HF model for export.

    Qwen3.5 has no stochastic layers when attention_dropout is zero. Enable its
    checkpointing layer wrappers during differentiable scoring even though GRPO
    evaluates the policy with dropout disabled.
    """

    def __init__(self, policy, chunk_size: int = 128) -> None:
        super().__init__()
        from transformers import Qwen3_5ForCausalLM

        if not isinstance(policy, Qwen3_5ForCausalLM):
            raise TypeError("Chunked policy scoring currently requires Qwen3.5 text")
        if policy.config.attention_dropout != 0:
            raise ValueError("Chunked GRPO scoring requires attention_dropout=0")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.policy = policy
        self.chunk_size = chunk_size

    def get_input_embeddings(self):
        return self.policy.get_input_embeddings()

    def get_output_embeddings(self):
        return self.policy.get_output_embeddings()

    def forward(self, input_ids, attention_mask):
        width = input_ids.shape[1]
        attended = attention_mask.bool().any(0).nonzero().flatten()
        if not attended.numel():
            raise ValueError("Policy input must contain attended tokens")
        length = int(attended[-1]) + 1
        ids, attention = input_ids[:, :length], attention_mask[:, :length]
        layers = self.policy.model.layers
        modes = [layer.training for layer in layers]
        if torch.is_grad_enabled():
            for layer in layers:
                if layer.gradient_checkpointing:
                    layer.training = True
        try:
            hidden = self.policy.model(
                input_ids=ids,
                attention_mask=attention,
                use_cache=False,
            ).last_hidden_state
        finally:
            for layer, mode in zip(layers, modes, strict=True):
                layer.training = mode
        valid = attention[:, 1:].bool() & attention[:, :-1].bool()
        states = hidden[:, :-1][valid]
        targets = ids[:, 1:][valid].long()

        def score_chunk(states, weight, targets):
            logits = F.linear(states, weight).float()
            return F.log_softmax(logits, -1).gather(-1, targets[:, None]).squeeze(-1)

        values = []
        for start in range(0, len(targets), self.chunk_size):
            args = (
                states[start : start + self.chunk_size],
                self.policy.lm_head.weight,
                targets[start : start + self.chunk_size],
            )
            values.append(
                checkpoint(score_chunk, *args, use_reentrant=False)
                if torch.is_grad_enabled()
                else score_chunk(*args)
            )
        scores = torch.zeros_like(valid, dtype=torch.float32).masked_scatter(
            valid, torch.cat(values)
        )
        return SimpleNamespace(token_logprobs=F.pad(scores, (1, width - length)))
