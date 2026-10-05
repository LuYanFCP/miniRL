import math
from dataclasses import dataclass


def _positive_integer(name: str, value: int, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


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

    def __post_init__(self) -> None:
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
