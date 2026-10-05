"""Compute teacher-forced validation loss and dispatch task evaluation plugins."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from minirl.common import MetricValue
from minirl.common.evaluation import EvaluationContext, EvaluationPlugin
from minirl.common.plugins import PluginConfig
from minirl.data.sft import ChatSFTDataset, SFTCollator


class SFTEvaluator:
    """Executed by SFTTrainer under eval/no_grad/autocast with training RNG restored."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        dataset: ChatSFTDataset,
        output_dir: Path,
        *,
        batch_size: int,
        plugins: list[tuple[PluginConfig, EvaluationPlugin]],
        seed: int = 42,
    ) -> None:
        self.model, self.tokenizer, self.dataset = model, tokenizer, dataset
        self.output_dir, self.plugins = output_dir, plugins
        self.seed = seed
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=SFTCollator(tokenizer.pad_token_id),
            generator=torch.Generator().manual_seed(0),
        )

    def __call__(self, step: int) -> dict[str, MetricValue]:
        device = next(self.model.parameters()).device
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.random.default_generator.manual_seed(self.seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(self.seed)
        nll, tokens = 0.0, 0
        for batch in self.dataloader:
            count = int((batch["labels"][:, 1:] != -100).sum())
            loss = self.model(
                **{key: value.to(device) for key, value in batch.items()}
            ).loss
            if loss is None or loss.numel() != 1 or not torch.isfinite(loss):
                raise ValueError("Validation model must return a finite scalar loss")
            nll += float(loss) * count
            tokens += count
        if not tokens:
            raise ValueError("Validation data has no supervised tokens")
        result: dict[str, MetricValue] = {
            "eval/loss": nll / tokens,
            "eval/supervised_tokens": tokens,
            "eval/samples": len(self.dataset),
        }
        for config, callback in self.plugins:
            directory = self.output_dir / "eval" / config.name / f"step_{step:08d}"
            directory.mkdir(parents=True, exist_ok=True)
            context = EvaluationContext(
                self.model,
                self.tokenizer,
                self.dataset.records,
                self.dataset.prompts,
                device,
                step,
                directory,
            )
            try:
                values = callback(context, **config.kwargs)
                if not isinstance(values, Mapping):
                    raise TypeError(
                        "Evaluation plugins must return a mapping of scalar metrics"
                    )
                for name, value in values.items():
                    key = f"eval/{config.name}/{name}"
                    if (
                        not isinstance(name, str)
                        or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_./-]*", name)
                        or len(key) > 255
                    ):
                        raise ValueError(f"Invalid plugin metric name: {name!r}")
                    if isinstance(value, torch.Tensor):
                        if value.numel() != 1:
                            raise ValueError(f"Plugin metric {name!r} must be scalar")
                        value = value.detach().item()
                    if not isinstance(value, Real) or not math.isfinite(value):
                        raise ValueError(
                            f"Plugin metric {name!r} must be a finite number"
                        )
                    result[key] = (
                        int(value) if isinstance(value, Integral) else float(value)
                    )
            except Exception as error:
                error.add_note(
                    f"Evaluation plugin {config.name!r} ({config.entrypoint}) at step {step}"
                )
                raise
        return result
