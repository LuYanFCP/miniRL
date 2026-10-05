"""Training objectives with a shared model-name-to-loss interface."""

from .base import Algorithm, Losses, ModelMap, TensorBatch
from .grpo import GRPO, GRPOConfig
from .sft import SFT

__all__ = [
    "GRPO",
    "SFT",
    "Algorithm",
    "GRPOConfig",
    "Losses",
    "ModelMap",
    "TensorBatch",
]
