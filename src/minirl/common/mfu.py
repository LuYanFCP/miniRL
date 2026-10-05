"""The 6N training FLOPs proxy; excludes attention and activation recomputation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MFU6NEstimator:
    """Estimate useful model throughput using global, non-padding input tokens.

    The caller supplies the active dense parameter count and *dense* hardware
    peak at the training precision. This proxy is not an operator FLOPs count,
    especially for hybrid attention, shared weights, or partially frozen models.
    Timing should cover synchronized optimizer updates, excluding evaluation and
    checkpoint saves. Recomputed forward passes do not increase the numerator.
    """

    parameters: int
    peak_tflops_per_gpu: float
    num_gpus: int = 1

    def __post_init__(self) -> None:
        for name in ("parameters", "num_gpus"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.peak_tflops_per_gpu, bool)
            or not math.isfinite(self.peak_tflops_per_gpu)
            or self.peak_tflops_per_gpu <= 0
        ):
            raise ValueError("peak_tflops_per_gpu must be finite and positive")

    def estimate(self, tokens: int, seconds: float) -> dict[str, float]:
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("tokens must be a positive integer")
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be finite and positive")
        tflops = 6 * self.parameters * tokens / seconds / self.num_gpus / 1e12
        # Do not clip at 100%: a bad peak/count/timing assumption should be visible.
        return {
            "perf/mfu_6n_estimate_pct": 100 * tflops / self.peak_tflops_per_gpu,
            "perf/model_tflops_6n_estimate_per_gpu": tflops,
        }
