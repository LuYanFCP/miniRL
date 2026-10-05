from __future__ import annotations

import json
import logging
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence, Sized
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler

from .algorithm import SFT
from .algorithm.base import Algorithm, TensorBatch
from .common import DataCollector, MetricValue
from .common.training_metrics import TrainingMetrics
from .config import MuonConfig, RLConfig, SFTConfig, TrainerConfig
from .optim import build_optimizer

logger = logging.getLogger(__name__)


def _resolve_device(requested: str | None) -> torch.device:
    """Resolve automatic selection once, pinning CUDA to a concrete device index."""
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise ValueError(f"Requested device {requested!r}, but CUDA is unavailable")

    # Validate the original ordinal: torch.device can narrow large ordinals.
    index = (
        int(requested.split(":", 1)[1])
        if ":" in requested
        else torch.cuda.current_device()
    )
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise ValueError(
            f"CUDA device index {index} is outside the {count} visible devices"
        )
    return torch.device("cuda", index)


@runtime_checkable
class _Stateful(Protocol):
    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class _ResumableBatchSampler(Sampler[list[int]]):
    """Commit the cursor only after an optimizer update, never while yielding data."""

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch = 0
        self.next_batch = 0
        self.order: list[int] = []

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        if not self.order:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            self.order = torch.randperm(self.size, generator=generator).tolist()
        for batch in range(self.next_batch, len(self)):
            start = batch * self.batch_size
            yield self.order[start : start + self.batch_size]

    def advance(self, batches: int) -> None:
        self.next_batch += batches
        if self.next_batch == len(self):
            self.epoch += 1
            self.next_batch = 0
            self.order = []

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "epoch": self.epoch,
            "next_batch": self.next_batch,
            "order": self.order,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["size"] != self.size:
            raise ValueError("Checkpoint dataset size differs from the current dataset")
        epoch, next_batch, order = state["epoch"], state["next_batch"], state["order"]
        if epoch < 0 or not 0 <= next_batch < len(self):
            raise ValueError("Invalid checkpoint data cursor")
        if order and (len(order) != self.size or set(order) != set(range(self.size))):
            raise ValueError("Invalid checkpoint sample permutation")
        if next_batch and not order:
            raise ValueError("A partial epoch requires its sample permutation")
        self.epoch, self.next_batch, self.order = epoch, next_batch, list(order)


class Trainer:
    def __init__(
        self, config: TrainerConfig, *, collector: DataCollector | None = None
    ) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.models: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, Optimizer] = {}
        self.schedulers: dict[str, LambdaLR] = {}
        self.global_step = 0
        self.collector = collector if collector is not None else DataCollector()
        self._checkpoint_ready = True
        self._failed = False
        self._grad_metrics: dict[str, dict[str, float]] = {}
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    def register_model[ModelT: torch.nn.Module](
        self, name: str, model: ModelT, trainable: bool = True, lr: float | None = None
    ) -> ModelT:
        if name in self.models:
            raise ValueError(f"Model {name!r} is already registered")
        existing = {
            id(parameter)
            for registered in self.models.values()
            for parameter in registered.parameters()
        }
        if any(id(parameter) in existing for parameter in model.parameters()):
            raise ValueError(
                "Models must not share parameter objects; register a shared backbone only once"
            )
        model = model.to(self.device)
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if trainable and not parameters:
            raise ValueError(f"Trainable model {name!r} has no trainable parameters")
        if not trainable:
            model.requires_grad_(False)
            model.eval()
            for parameter in model.parameters():
                parameter.grad = None
        self.models[name] = model
        if trainable:
            self.optimizers[name] = build_optimizer(model, self.config, lr=lr)
        return model

    def register_scheduler(self, name: str, total_steps: int) -> None:
        warmup = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0.0, 1.0 - progress)

        self.schedulers[name] = LambdaLR(self.optimizers[name], lr_lambda)

    # bf16 amp
    def amp(self) -> torch.autocast | nullcontext[None]:
        if self.config.bf16:
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def grad_step(self, name: str) -> None:
        model = self.models[name]
        optimizer = self.optimizers[name]
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
        )
        self._grad_metrics[name] = {
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": float(norm),
            "grad_clipped": float(norm > self.config.max_grad_norm),
        }
        optimizer.step()
        if name in self.schedulers:
            self.schedulers[name].step()
        optimizer.zero_grad(set_to_none=True)

    def log(self, metrics: Mapping[str, MetricValue]) -> None:
        self.collector.record(metrics, step=self.global_step)

    def close(self) -> None:
        """Close the collector after the run (including an injected collector)."""
        self.collector.close()

    @property
    def _ckpt_path(self) -> Path:
        return Path(self.config.output_dir) / "trainer_state.pt"

    def _loop_state_dict(self) -> dict[str, Any]:
        return {}

    def _load_loop_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("Unexpected trainer loop state")

    def _rng_state(self) -> dict[str, Any]:
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
        }

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                numpy_state[0],
                np.array(numpy_state[1], dtype=np.uint32),
                *numpy_state[2:],
            )
        )
        torch.set_rng_state(state["torch"])
        if self.device.type == "cuda" and state["cuda"] is not None:
            torch.cuda.set_rng_state(state["cuda"], self.device)

    def save_checkpoint(self) -> None:
        """Save at a completed SFT accumulation window or RL iteration boundary."""
        if not self._checkpoint_ready or self._failed:
            raise RuntimeError(
                "Cannot checkpoint an incomplete update; restore the last checkpoint after a failure"
            )
        config = asdict(self.config)
        state = {
            "format_version": 1,
            "trainer_type": type(self).__name__,
            "config": config,
            "models": {n: m.state_dict() for n, m in self.models.items()},
            "optimizers": {n: o.state_dict() for n, o in self.optimizers.items()},
            "schedulers": {n: s.state_dict() for n, s in self.schedulers.items()},
            "global_step": self.global_step,
            "run_id": self.collector.run_id,
            "loop": self._loop_state_dict(),
            "rng": self._rng_state(),
        }
        self._ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        # A failed write must leave the previous checkpoint intact.
        fd, temporary = tempfile.mkstemp(dir=self._ckpt_path.parent, suffix=".pt.tmp")
        os.close(fd)
        try:
            torch.save(state, temporary)
            os.replace(temporary, self._ckpt_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        (self._ckpt_path.parent / "config.json").write_text(
            json.dumps(config, indent=2)
        )
        logger.info("Checkpoint saved to %s", self._ckpt_path)

    def load_checkpoint(self) -> None:
        # RNG tensors must stay on CPU; optimizer.load_state_dict moves its own state.
        state = torch.load(
            self._ckpt_path, map_location="cpu", weights_only=True, mmap=True
        )
        if state.get("format_version") != 1:
            raise ValueError("Legacy checkpoint has no resumable loop/RNG state")
        if state["trainer_type"] != type(self).__name__:
            raise ValueError("Checkpoint trainer type differs")
        ignored = {"output_dir", "device", "log_steps", "save_steps"}
        current_config = asdict(self.config)
        saved_config = dict(state["config"])
        # Checkpoints written before optimizer selection always used AdamW.
        saved_config.setdefault("optimizer", "adamw")
        saved_config.setdefault("muon", asdict(MuonConfig()))
        changed = [
            key
            for key, value in saved_config.items()
            if key not in ignored and current_config.get(key) != value
        ]
        if changed:
            raise ValueError(
                f"Checkpoint training configuration differs: {', '.join(changed)}"
            )
        for key, registered in (
            ("models", self.models),
            ("optimizers", self.optimizers),
            ("schedulers", self.schedulers),
        ):
            if set(state[key]) != set(registered):
                raise ValueError(f"Checkpoint {key} differ from registered {key}")
        for n, m in self.models.items():
            m.load_state_dict(state["models"][n])
        for n, o in self.optimizers.items():
            o.load_state_dict(state["optimizers"][n])
        for n, s in self.schedulers.items():
            s.load_state_dict(state["schedulers"][n])
        self.global_step = state["global_step"]
        self.collector.run_id = state.get("run_id", self.collector.run_id)
        self._load_loop_state_dict(state["loop"])
        self._restore_rng_state(state["rng"])
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        self._checkpoint_ready = True
        self._failed = False

    def train(self) -> None:
        if self._failed:
            raise RuntimeError(
                "Previous training failed; load a checkpoint before retrying"
            )
        try:
            self._train()
        except BaseException as training_error:
            self._failed = True
            try:
                self.collector.flush()
            except BaseException as flush_error:
                training_error.add_note(
                    f"DataCollector.flush also failed: {flush_error!r}"
                )
                logger.warning(
                    "Collector flush failed while handling a training error",
                    exc_info=True,
                )
            raise
        else:
            self.collector.flush()

    def _train(self) -> None:
        raise NotImplementedError


class SFTTrainer[SampleT](Trainer):
    config: SFTConfig

    def __init__(
        self,
        config: SFTConfig,
        model: torch.nn.Module,
        train_dataset: Dataset[SampleT] | Sequence[SampleT],
        collate_fn: Callable[[list[SampleT]], TensorBatch] | None,
        algorithm: Algorithm | None = None,
        *,
        collector: DataCollector | None = None,
        evaluator: Callable[[int], Mapping[str, MetricValue]] | None = None,
        eval_steps: int = 0,
        eval_at_start: bool = True,
    ) -> None:
        super().__init__(config, collector=collector)
        if (
            isinstance(eval_steps, bool)
            or not isinstance(eval_steps, int)
            or eval_steps < 0
        ):
            raise ValueError("eval_steps must be a non-negative integer")
        self.evaluator = evaluator
        self.eval_steps = eval_steps
        self.eval_at_start = eval_at_start
        self._last_eval_step: int | None = None
        self._metrics = TrainingMetrics()
        self._totals = {"samples": 0, "tokens": 0, "supervised_tokens": 0}
        self.algorithm: Algorithm = algorithm or SFT()
        self.register_model("policy", model)
        if not isinstance(train_dataset, Sized) or len(train_dataset) == 0:
            raise ValueError("SFT requires a non-empty, sized dataset")
        self._batch_sampler = _ResumableBatchSampler(
            len(train_dataset), config.batch_size, config.seed
        )
        # DataLoader also accepts indexable sequences at runtime, including the example's list.
        self.dataloader: DataLoader[SampleT] = DataLoader(
            cast(Dataset[SampleT], train_dataset),
            batch_sampler=self._batch_sampler,
            collate_fn=collate_fn,
            # DataLoader iterator creation must not advance the model/dropout RNG on resume.
            generator=torch.Generator().manual_seed(config.seed),
        )
        self.register_scheduler("policy", self._total_steps())

    def _total_steps(self) -> int:
        return self.config.num_epochs * math.ceil(
            len(self.dataloader) / self.config.gradient_accumulation_steps
        )

    def _loop_state_dict(self) -> dict[str, Any]:
        return {
            "sampler": self._batch_sampler.state_dict(),
            "algorithm": self.algorithm.state_dict(),
            "metrics": self._metrics.state_dict(),
            "totals": self._totals,
            "last_eval_step": self._last_eval_step,
        }

    def _load_loop_state_dict(self, state: dict[str, Any]) -> None:
        self._batch_sampler.load_state_dict(state["sampler"])
        if self._batch_sampler.epoch > self.config.num_epochs:
            raise ValueError("Checkpoint epoch exceeds num_epochs")
        self.algorithm.load_state_dict(state["algorithm"])
        self._metrics = TrainingMetrics(**state.get("metrics", {}))
        self._totals.update(state.get("totals", {}))
        self._last_eval_step = state.get("last_eval_step")

    def _evaluate(self) -> None:
        if self.evaluator is None or self._last_eval_step == self.global_step:
            return
        rng = self._rng_state()
        modes = {
            module: module.training
            for model in self.models.values()
            for module in model.modules()
        }
        start = time.perf_counter()
        try:
            for model in self.models.values():
                model.eval()
            with torch.no_grad(), self.amp():
                metrics = dict(self.evaluator(self.global_step))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            metrics["eval/seconds"] = time.perf_counter() - start
            self.log(metrics)
            self._last_eval_step = self.global_step
        finally:
            for module, training in modes.items():
                module.training = training
            self._restore_rng_state(rng)

    def _begin_update(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return time.perf_counter()

    def _log_training_metrics(self) -> None:
        sampler = self._batch_sampler
        metrics = self._metrics.metrics()
        metrics["epoch"] = sampler.epoch + sampler.next_batch / len(sampler)
        metrics.update(
            {f"train/total_{key}": value for key, value in self._totals.items()}
        )
        metrics["train/eta_seconds"] = (
            self._metrics.seconds
            / self._metrics.updates
            * (self._total_steps() - self.global_step)
        )
        if self.device.type == "cuda":
            metrics.update(
                {
                    "gpu/allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "gpu/reserved_bytes": torch.cuda.memory_reserved(self.device),
                    "gpu/peak_allocated_bytes": self._metrics.peak_allocated,
                    "gpu/peak_reserved_bytes": self._metrics.peak_reserved,
                }
            )
        self.log(metrics)
        self._metrics = TrainingMetrics()

    def _train(self) -> None:
        self.models["policy"].train()
        self.optimizers["policy"].zero_grad(set_to_none=True)
        accum = self.config.gradient_accumulation_steps
        sampler = self._batch_sampler
        batch: TensorBatch

        if (
            self.global_step == 0
            and sampler.epoch < self.config.num_epochs
            and self.eval_at_start
        ):
            self._evaluate()
        started = self._begin_update()

        while sampler.epoch < self.config.num_epochs:
            micro = 0
            window_size = min(accum, len(sampler) - sampler.next_batch)
            for batch in self.dataloader:
                self._checkpoint_ready = False
                cpu_batch = batch
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with self.amp():
                    losses = self.algorithm.compute_losses(self.models, batch)
                loss = losses["policy"] / window_size
                loss.backward()
                previous = {key: getattr(self._metrics, key) for key in self._totals}
                self._metrics.observe_batch(
                    cpu_batch, float(losses["policy"].detach().item())
                )
                for key in self._totals:
                    self._totals[key] += getattr(self._metrics, key) - previous[key]
                micro += 1

                if micro == window_size:
                    self.grad_step("policy")
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    self._metrics.observe_update(
                        self._grad_metrics["policy"],
                        time.perf_counter() - started,
                        self.device,
                    )
                    self.global_step += 1
                    sampler.advance(micro)
                    self._checkpoint_ready = True
                    finished = sampler.epoch == self.config.num_epochs
                    if self.global_step % self.config.log_steps == 0 or finished:
                        self._log_training_metrics()
                    if finished or (
                        self.eval_steps and self.global_step % self.eval_steps == 0
                    ):
                        self._evaluate()
                    micro = 0
                    window_size = min(accum, len(sampler) - sampler.next_batch)
                    if (
                        self.config.save_steps > 0
                        and self.global_step % self.config.save_steps == 0
                    ):
                        self.save_checkpoint()
                    started = self._begin_update()

        self.save_checkpoint()


class RLTrainer[PromptT](Trainer):
    """Prepare a rollout once, update on shuffled minibatches, then checkpoint.

    Rollouts supply detached policy samples and their behavior log probabilities.
    A callable with state_dict/load_state_dict also restores its sampling cursor.
    Checkpoints resume at completed rollout boundaries, including RNG state.
    """

    config: RLConfig

    def __init__(
        self,
        config: RLConfig,
        algorithm: Algorithm,
        models: Mapping[str, torch.nn.Module],
        rollout_fn: Callable[[list[PromptT]], TensorBatch],
        prompts: list[PromptT] | None = None,
        *,
        collector: DataCollector | None = None,
    ) -> None:
        super().__init__(config, collector=collector)
        self.algorithm = algorithm
        self.rollout_fn = rollout_fn
        self.prompts: list[PromptT] = prompts or []
        self.iteration = 0

        for name in algorithm.MODEL_NAMES:
            self.register_model(
                name, models[name], trainable=name in algorithm.TRAINABLE_MODELS
            )

    def _minibatches(self, batch: TensorBatch, size: int) -> Iterator[TensorBatch]:
        B = batch["input_ids"].shape[0]
        if B == 0 or any(
            value.ndim == 0 or value.shape[0] != B for value in batch.values()
        ):
            raise ValueError(
                "Rollout fields must have the same non-empty batch dimension"
            )
        idx = torch.randperm(B, device=self.device)
        for start in range(0, B, size):
            sel = idx[start : start + size]
            yield {k: v[sel] for k, v in batch.items()}

    def _loop_state_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "algorithm": self.algorithm.state_dict(),
            "rollout": self.rollout_fn.state_dict()
            if isinstance(self.rollout_fn, _Stateful)
            else None,
        }

    def _load_loop_state_dict(self, state: dict[str, Any]) -> None:
        if not 0 <= state["iteration"] <= self.config.num_iterations:
            raise ValueError("Invalid checkpoint iteration")
        self.iteration = state["iteration"]
        self.algorithm.load_state_dict(state["algorithm"])
        if state["rollout"] is not None:
            if not isinstance(self.rollout_fn, _Stateful):
                raise ValueError("Checkpoint requires a stateful rollout callable")
            self.rollout_fn.load_state_dict(state["rollout"])

    def _train(self) -> None:
        for name, model in self.models.items():
            model.train(name in self.algorithm.TRAINABLE_MODELS)
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        while self.iteration < self.config.num_iterations:
            it = self.iteration
            self._checkpoint_ready = False
            with torch.no_grad(), self.amp():
                batch = self.rollout_fn(self.prompts)
                batch = {k: v.detach().to(self.device) for k, v in batch.items()}
                batch = self.algorithm.prepare(self.models, batch)
                # A returned parameter/view must not change as optimizers update the models.
                batch = {k: v.detach().clone() for k, v in batch.items()}
            rollout_metrics = {
                f"rollout/{key}": value
                for key, value in self.algorithm.rollout_metrics().items()
            }

            for _ in range(self.config.update_epochs):
                for mb in self._minibatches(batch, self.config.minibatch_size):
                    for optimizer in self.optimizers.values():
                        optimizer.zero_grad(set_to_none=True)
                    with self.amp():
                        losses = self.algorithm.compute_losses(self.models, mb)
                    if not losses or set(losses) - self.optimizers.keys():
                        raise ValueError(
                            "RL losses must be a non-empty mapping keyed by registered optimizer names"
                        )
                    if any(
                        loss.numel() != 1 or not loss.requires_grad
                        for loss in losses.values()
                    ):
                        raise ValueError(
                            "RL losses must be differentiable scalar tensors"
                        )
                    # Backpropagate all roots together before any parameter is mutated.
                    torch.autograd.backward(tuple(losses.values()))
                    for name in losses:
                        self.grad_step(name)
                    self.global_step += 1

                    if self.global_step % self.config.log_steps == 0:
                        self.log(
                            {
                                "iter": it,
                                **rollout_metrics,
                                **losses,
                                **{
                                    f"{name}/{key}": value
                                    for name in losses
                                    for key, value in self._grad_metrics[name].items()
                                },
                            }
                        )

            self.algorithm.on_iteration_end(self.models, it)
            self.iteration += 1
            self._checkpoint_ready = True
            if self.config.save_steps > 0 and (it + 1) % self.config.save_steps == 0:
                self.save_checkpoint()

        self.save_checkpoint()
