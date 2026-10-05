"""Tensor validation helpers shared by algorithm implementations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def evaluating(model: torch.nn.Module) -> Iterator[None]:
    """Disable dropout and buffer updates while retaining parameter gradients."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def check_shape(name: str, tensor: torch.Tensor, like: torch.Tensor) -> None:
    if tensor.shape != like.shape or tensor.device != like.device:
        raise ValueError(f"{name} must have shape {tuple(like.shape)} on {like.device}")


def binary_mask(name: str, tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    check_shape(name, tensor, like)
    if not ((tensor == 0) | (tensor == 1)).all():
        raise ValueError(f"{name} must contain only zero and one")
    return tensor.bool()


def fixed_targets(name: str, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    check_shape(name, tensor, mask)
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating-point")
    if not torch.isfinite(tensor[mask]).all():
        raise ValueError(f"{name} must be finite at response positions")
    return tensor.detach().float().masked_fill(~mask, 0)


def check_response_spans(mask: torch.Tensor) -> None:
    # One contiguous response per row; padding and prompt tokens stay outside it.
    starts = mask & ~F.pad(mask[:, :-1], (1, 0), value=False)
    if not (starts.sum(-1) == 1).all():
        raise ValueError("Each row must contain exactly one non-empty response span")
