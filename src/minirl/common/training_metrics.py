"""Aggregate inexpensive scalar statistics across complete optimizer updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from minirl.algorithm.base import TensorBatch


@dataclass
class TrainingMetrics:
    samples: int = 0
    tokens: int = 0
    supervised_tokens: int = 0
    updates: int = 0
    microbatches: int = 0
    loss_sum: float = 0.0
    nll_sum: float = 0.0
    cells: int = 0
    sequence_length_max: int = 0
    seconds: float = 0.0
    grad_norm_sum: float = 0.0
    grad_norm_max: float = 0.0
    clipped: int = 0
    lr: float = 0.0
    peak_allocated: int = 0
    peak_reserved: int = 0

    def observe_batch(self, batch: TensorBatch, loss: float) -> None:
        ids = batch["input_ids"]
        self.samples += len(ids)
        self.loss_sum += loss
        self.microbatches += 1
        # Generic SFTTrainer can also be used for regression/custom algorithms.
        if ids.ndim != 2 or ids.is_floating_point():
            return
        mask = batch.get("attention_mask", torch.ones_like(ids)).bool()
        lengths = mask.sum(-1).tolist()
        self.sequence_length_max = max(self.sequence_length_max, max(lengths))
        self.tokens += sum(lengths)
        self.cells += ids.numel()
        labels = batch.get("labels")
        if (
            labels is not None
            and labels.shape == ids.shape
            and not labels.is_floating_point()
        ):
            count = int(((labels[:, 1:] != -100) & mask[:, 1:]).sum())
            self.supervised_tokens += count
            self.nll_sum += loss * count

    def observe_update(
        self, gradients: dict[str, float], seconds: float, device: torch.device
    ) -> None:
        self.updates += 1
        self.seconds += seconds
        self.grad_norm_sum += gradients["grad_norm"]
        self.grad_norm_max = max(self.grad_norm_max, gradients["grad_norm"])
        self.clipped += int(gradients["grad_clipped"])
        self.lr = gradients["lr"]
        if device.type == "cuda":
            self.peak_allocated = max(
                self.peak_allocated, torch.cuda.max_memory_allocated(device)
            )
            self.peak_reserved = max(
                self.peak_reserved, torch.cuda.max_memory_reserved(device)
            )

    def metrics(self) -> dict[str, int | float]:
        seconds = max(self.seconds, 1e-9)
        result = {
            "loss": self.loss_sum / self.microbatches,
            "lr": self.lr,
            "train/grad_norm": self.grad_norm_sum / self.updates,
            "train/grad_norm_max": self.grad_norm_max,
            "train/grad_clip_fraction": self.clipped / self.updates,
            "train/step_seconds": seconds / self.updates,
            "train/samples_per_second": self.samples / seconds,
            "train/samples": self.samples,
            "train/updates": self.updates,
        }
        if self.cells:
            result.update(
                {
                    "train/tokens_per_second": self.tokens / seconds,
                    "train/supervised_tokens_per_second": self.supervised_tokens
                    / seconds,
                    "train/tokens": self.tokens,
                    "train/supervised_tokens": self.supervised_tokens,
                    "train/padding_fraction": 1 - self.tokens / self.cells,
                    "train/sequence_length_mean": self.tokens / self.samples,
                    "train/sequence_length_max": self.sequence_length_max,
                }
            )
        if self.supervised_tokens:
            result["train/token_weighted_loss"] = self.nll_sum / self.supervised_tokens
        return result

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)
