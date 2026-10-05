"""Training objectives with a shared model-name-to-loss interface."""

from .base import Algorithm, Losses, ModelMap, TensorBatch
from .sft import SFT

__all__ = ["SFT", "Algorithm", "Losses", "ModelMap", "TensorBatch"]
