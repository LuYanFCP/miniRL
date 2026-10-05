"""Loss objectives and rollout preparation, independent of training execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

type TensorBatch = dict[str, torch.Tensor]
type ModelMap = Mapping[str, torch.nn.Module]
type Losses = dict[str, torch.Tensor]


class Algorithm(ABC):
    """Define model roles, fixed rollout targets, and differentiable objectives.

    Trainers own optimizers, scheduling, devices, data loading, and checkpoints.
    Rollout callables own data generation. Model names are algorithm-specific.
    """

    #: Required model roles and the subset for which trainers create optimizers.
    MODEL_NAMES: tuple[str, ...] = ()
    TRAINABLE_MODELS: tuple[str, ...] = ()

    def prepare(self, models: ModelMap, batch: TensorBatch) -> TensorBatch:
        """Prepare targets once per rollout, before minibatching or repeated updates.

        Every output field must retain the same leading batch dimension.
        RLTrainer runs this hook without gradients and snapshots its result.
        """
        return batch

    def rollout_metrics(self) -> Mapping[str, float]:
        """Return statistics for the prepared rollout, reused for its updates."""
        return {}

    @abstractmethod
    def compute_losses(self, models: ModelMap, batch: TensorBatch) -> Losses:
        """Return scalar losses keyed by trainable model name, including SFT.

        Trainers backpropagate all losses together before updating parameters.
        Losses may share a graph; registered models must not share parameters.
        Merge auxiliary objectives into the corresponding model's loss.
        """

    def state_dict(self) -> dict[str, Any]:
        """Return state compatible with torch.load(weights_only=True)."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError(
                f"{type(self).__name__} must implement load_state_dict for its state"
            )

    def on_iteration_end(self, models: ModelMap, iteration: int) -> None:
        """Update optional algorithm state after a completed rollout iteration."""
