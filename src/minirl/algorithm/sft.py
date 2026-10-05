"""Supervised fine-tuning using a model-provided loss."""

from __future__ import annotations

import torch

from .base import Algorithm, Losses, ModelMap, TensorBatch


class SFT(Algorithm):
    MODEL_NAMES = ("policy",)
    TRAINABLE_MODELS = ("policy",)

    def compute_losses(self, models: ModelMap, batch: TensorBatch) -> Losses:
        # The model owns label shifting, ignore indices, and loss reduction.
        output = models["policy"](**batch)
        loss = getattr(output, "loss", None)
        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                "SFT requires a model output with a Tensor loss; supply labels"
            )
        if loss.ndim != 0 or not loss.is_floating_point():
            raise ValueError("SFT requires a scalar floating-point loss")
        if not torch.isfinite(loss):
            raise ValueError(
                "SFT loss is not finite; check labels and their ignore mask"
            )
        return {"policy": loss}
