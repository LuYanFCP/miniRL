"""Outcome-supervised GRPO with explicit prompt groups and causal token alignment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .base import Algorithm, Losses, ModelMap, TensorBatch
from .functional import binary_mask, check_response_spans, evaluating, fixed_targets


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 8
    clip_ratio: float = 0.2
    kl_coef: float = 0.0
    advantage_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError("group_size must be an integer >= 2")
        if not math.isfinite(self.clip_ratio) or not 0 <= self.clip_ratio < 1:
            raise ValueError("clip_ratio must be finite and in [0, 1)")
        if not math.isfinite(self.kl_coef) or self.kl_coef < 0:
            raise ValueError("kl_coef must be finite and non-negative")
        if not math.isfinite(self.advantage_epsilon) or self.advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be finite and positive")


class GRPO(Algorithm):
    """Vanilla sequence-mean GRPO, without a value model.

    `rewards` and integer `group_ids` are [B]; token fields are [B, T].
    Response position t is scored using logits[t-1]. Include EOS and exclude
    prompt/padding. A group contains exactly G samples of the same prompt.
    Advantages use population std + epsilon and are computed BEFORE minibatching.
    Sampling must use temperature=1, top_p=1, top_k=0, without logit processors
    that change the policy distribution. The reference, when used, stays fixed.
    """

    MODEL_NAMES = ("actor",)
    TRAINABLE_MODELS = ("actor",)

    def __init__(self, config: GRPOConfig) -> None:
        self.config = config
        if config.kl_coef:
            self.MODEL_NAMES = ("actor", "reference")
        self.last_rollout_metrics: dict[str, float] = {}

    def _response_mask(self, batch: TensorBatch) -> torch.Tensor:
        ids = batch["input_ids"]
        if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] < 2:
            raise ValueError("input_ids must have non-empty shape [B, T] with T >= 2")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        attention = binary_mask("attention_mask", batch["attention_mask"], ids)
        response = binary_mask("response_mask", batch["response_mask"], ids)
        predictable = attention & F.pad(attention[:, :-1], (1, 0), value=False)
        if (response & ~predictable).any():
            raise ValueError("Each response token must have an attended predecessor")
        check_response_spans(response)
        return response

    @staticmethod
    def _policy_logits(model, input_ids, attention_mask):
        with evaluating(model):
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = getattr(output, "logits", None)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != input_ids.shape
            or not logits.is_floating_point()
        ):
            raise ValueError(
                "The actor/reference must return floating logits [B, T, V]"
            )
        return logits

    def per_token_logprobs(self, model, input_ids, attention_mask):
        """Score the unmodified policy in eval mode, restoring all module modes."""
        logits = self._policy_logits(model, input_ids, attention_mask)
        values = F.log_softmax(logits[:, :-1].float(), dim=-1)
        selected = values.gather(-1, input_ids[:, 1:].long().unsqueeze(-1)).squeeze(-1)
        return F.pad(selected, (1, 0))

    @torch.no_grad()
    def prepare(self, models: ModelMap, batch: TensorBatch) -> TensorBatch:
        mask = self._response_mask(batch)
        rewards, groups = batch["rewards"], batch["group_ids"]
        if rewards.shape != mask.shape[:1] or rewards.device != mask.device:
            raise ValueError("rewards must have shape [B] on the input device")
        if not rewards.is_floating_point() or not torch.isfinite(rewards).all():
            raise ValueError("rewards must be finite floating-point scalars")
        if (
            groups.shape != rewards.shape
            or groups.device != mask.device
            or groups.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("group_ids must be integer [B] on the input device")
        unique, counts = groups.unique(return_counts=True)
        if not (counts == self.config.group_size).all():
            raise ValueError("Each group must contain exactly group_size responses")
        rewards = rewards.detach().float()
        advantages = torch.zeros_like(rewards)
        stds = []
        attention = batch["attention_mask"].bool()
        starts = mask.long().argmax(-1)
        for group in unique:
            rows = (groups == group).nonzero().flatten().tolist()
            prefixes = [
                batch["input_ids"][i, : starts[i]][attention[i, : starts[i]]]
                for i in rows
            ]
            if any(not torch.equal(prefixes[0], p) for p in prefixes[1:]):
                raise ValueError(
                    "Responses in a group must share the same prompt tokens"
                )
            values = rewards[rows]
            std = values.std(unbiased=False)
            advantages[rows] = (values - values.mean()) / (
                std + self.config.advantage_epsilon
            )
            stds.append(std)
        stds = torch.stack(stds)
        if not torch.isfinite(stds).all() or not torch.isfinite(advantages).all():
            raise ValueError("Group reward normalization must remain finite in FP32")
        self.last_rollout_metrics = {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)),
            "group_reward_std_mean": float(stds.mean()),
            "nonzero_advantage_group_fraction": float((stds > 0).float().mean()),
            "response_tokens_mean": float(mask.sum(-1).float().mean()),
        }
        result = {
            **batch,
            "rewards": rewards,
            "old_logprobs": fixed_targets("old_logprobs", batch["old_logprobs"], mask),
            "advantages": advantages[:, None].expand_as(mask).masked_fill(~mask, 0),
        }
        if self.config.kl_coef:
            reference = self.per_token_logprobs(
                models["reference"], batch["input_ids"], batch["attention_mask"]
            )
            result["reference_logprobs"] = fixed_targets(
                "reference_logprobs", reference, mask
            )
        return result

    def rollout_metrics(self) -> dict[str, float]:
        return dict(self.last_rollout_metrics)

    def compute_losses(self, models: ModelMap, batch: TensorBatch) -> Losses:
        mask = self._response_mask(batch)
        ids = batch["input_ids"]
        old = fixed_targets("old_logprobs", batch["old_logprobs"], mask)[mask]
        advantages = fixed_targets("advantages", batch["advantages"], mask)[mask]
        logits = self._policy_logits(models["actor"], ids, batch["attention_mask"])
        # Select active predictors first: masked NaNs must never enter reductions.
        distribution = F.log_softmax(logits[:, :-1][mask[:, 1:]].float(), dim=-1)
        current = distribution.gather(-1, ids[mask].long().unsqueeze(-1)).squeeze(-1)
        ratios = (current - old).exp()
        if not torch.isfinite(ratios).all():
            raise ValueError("GRPO probability ratios must be finite")
        clipped = ratios.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
        token_loss = -torch.minimum(ratios * advantages, clipped * advantages)
        if self.config.kl_coef:
            reference = fixed_targets(
                "reference_logprobs", batch["reference_logprobs"], mask
            )[mask]
            delta = reference - current
            token_loss = token_loss + self.config.kl_coef * (delta.expm1() - delta)
        # Original GRPO: mean over each response's tokens, then over responses.
        per_row = torch.zeros_like(mask, dtype=torch.float32).masked_scatter(
            mask, token_loss
        )
        loss = (per_row.sum(-1) / mask.sum(-1)).mean()
        if not torch.isfinite(loss):
            raise ValueError("GRPO loss must be finite")
        return {"actor": loss}

    def state_dict(self) -> dict[str, Any]:
        return {"batch_format": 1, "config": asdict(self.config)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state != self.state_dict():
            raise ValueError("GRPO checkpoint configuration or token alignment differs")
