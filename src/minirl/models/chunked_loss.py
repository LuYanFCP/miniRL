"""Memory-bounded causal cross entropy for the Qwen3.5 text policy."""

from functools import wraps
from types import MethodType

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast


def enable_chunked_loss(model, chunk_size: int) -> None:
    """Checkpoint LM-head/CE chunks; export remains a standard HF text model.

    Only labelled forwards omit full logits. Unlabelled inference/generation uses
    the original forward. Loss is a mean over all non-ignored next-token labels.
    """
    from transformers import Qwen3_5ForCausalLM

    if not isinstance(model, Qwen3_5ForCausalLM):
        raise TypeError("loss_chunk_size currently supports Qwen3_5ForCausalLM only")
    original = model.forward.__func__

    @wraps(original)
    def forward(self, *args, **kwargs):
        labels = kwargs.get("labels")
        if labels is None:
            return original(self, *args, **kwargs)
        if args:
            raise ValueError("Chunked labelled forwards require keyword arguments")
        kwargs = dict(kwargs)
        kwargs.pop("labels")
        if kwargs.pop("logits_to_keep", 0) != 0:
            raise ValueError("Chunked loss requires all labelled token positions")
        if kwargs.pop("return_dict", True) is False:
            raise ValueError("Chunked loss requires return_dict=True")
        kwargs["use_cache"] = False
        outputs = self.model(**kwargs)
        hidden = outputs.last_hidden_state[:, :-1].reshape(-1, self.config.hidden_size)
        targets = labels[:, 1:].reshape(-1)
        count = (targets != -100).sum()
        if count == 0:
            raise ValueError("Chunked loss requires at least one supervised token")

        def chunk_loss(states, weight, expected):
            logits = F.linear(states, weight)
            return F.cross_entropy(
                logits.float(), expected, ignore_index=-100, reduction="sum"
            )

        loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, targets.numel(), chunk_size):
            arguments = (
                hidden[start : start + chunk_size],
                self.lm_head.weight,
                targets[start : start + chunk_size],
            )
            loss = loss + (
                checkpoint(chunk_loss, *arguments, use_reentrant=False)
                if torch.is_grad_enabled()
                else chunk_loss(*arguments)
            )
        return CausalLMOutputWithPast(
            loss=loss / count,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = MethodType(forward, model)
