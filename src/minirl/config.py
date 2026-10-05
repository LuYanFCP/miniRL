import math
from dataclasses import dataclass, field
from typing import Literal


def _positive_integer(name: str, value: int, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass
class MuonConfig:
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    adjust_lr_fn: Literal["original", "match_rms_adamw", "spectral_unclamped"] = (
        "match_rms_adamw"
    )
    # None shares TrainerConfig.learning_rate with the Muon parameter group.
    adamw_lr: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.momentum, bool) or not 0 <= self.momentum < 1:
            raise ValueError("muon.momentum must be in [0, 1)")
        if type(self.nesterov) is not bool:
            raise ValueError("muon.nesterov must be a boolean")
        _positive_integer("muon.ns_steps", self.ns_steps)
        if self.ns_steps >= 100:
            raise ValueError("muon.ns_steps must be smaller than 100")
        if self.adjust_lr_fn not in (
            "original",
            "match_rms_adamw",
            "spectral_unclamped",
        ):
            raise ValueError("Unsupported muon.adjust_lr_fn")
        if self.adamw_lr is not None and (
            isinstance(self.adamw_lr, bool)
            or not math.isfinite(self.adamw_lr)
            or self.adamw_lr < 0
        ):
            raise ValueError("muon.adamw_lr must be finite and non-negative")


@dataclass
class TrainerConfig:
    output_dir: str = "checkpoints"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    bf16: bool = False
    log_steps: int = 1
    save_steps: int = 0
    seed: int = 42
    device: str | None = None
    optimizer: Literal["adamw", "muon"] = field(default="adamw", kw_only=True)
    muon: MuonConfig = field(default_factory=MuonConfig, kw_only=True)

    def __post_init__(self) -> None:
        if self.optimizer not in ("adamw", "muon"):
            raise ValueError("optimizer must be 'adamw' or 'muon'")
        if not isinstance(self.muon, MuonConfig):
            raise TypeError("muon must be a MuonConfig")
        for name in ("learning_rate", "weight_decay", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_grad_norm == 0:
            raise ValueError("max_grad_norm must be positive")
        for name in ("gradient_accumulation_steps", "log_steps"):
            _positive_integer(name, getattr(self, name))
        for name in ("warmup_steps", "save_steps", "seed"):
            _positive_integer(name, getattr(self, name), allow_zero=True)
        if self.seed >= 2**32:
            raise ValueError("seed must be smaller than 2**32")


@dataclass
class SFTConfig(TrainerConfig):
    num_epochs: int = 3
    batch_size: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("num_epochs", "batch_size"):
            _positive_integer(name, getattr(self, name))


@dataclass
class RLConfig(TrainerConfig):
    """One optimizer update per minibatch, with a constant learning rate.

    log_steps counts optimizer updates; save_steps counts complete rollouts.
    """

    num_iterations: int = 100
    update_epochs: int = 2
    minibatch_size: int = 4
    microbatch_size: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("num_iterations", "update_epochs", "minibatch_size"):
            _positive_integer(name, getattr(self, name))
        if self.microbatch_size is not None:
            _positive_integer("microbatch_size", self.microbatch_size)
        if self.gradient_accumulation_steps != 1:
            raise ValueError("RL gradient_accumulation_steps must be 1")
        if self.warmup_steps != 0:
            raise ValueError("RL warmup_steps must be 0; no LR schedule is configured")
