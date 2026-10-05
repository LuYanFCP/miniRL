"""Task-independent context passed to user supplied evaluation functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch

from .collector import MetricValue

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class EvaluationContext:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    samples: Sequence[Mapping[str, Any]]
    prompts: Sequence[list[dict[str, str]]]
    device: torch.device
    step: int
    output_dir: Path


class EvaluationPlugin(Protocol):
    def __call__(
        self, context: EvaluationContext, **kwargs: Any
    ) -> Mapping[str, MetricValue]: ...
