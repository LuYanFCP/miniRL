"""AdamW with FP32 master weights for low-precision trainable parameters."""

from __future__ import annotations

import torch
from torch.optim import AdamW


class MasterAdamW(AdamW):
    """Keep small updates and Adam moments in FP32; run the model in BF16.

    Master parameters are optimizer-owned, so ordinary AdamW checkpoint loading
    preserves their FP32 moments. The master values are saved explicitly too.
    """

    def __init__(self, parameters, **kwargs):
        self.model_parameters = list(parameters)
        self.master_parameters = [
            torch.nn.Parameter(p.detach().float().clone(), requires_grad=True)
            if p.dtype in (torch.bfloat16, torch.float16)
            else p
            for p in self.model_parameters
        ]
        # Avoid a parameter-sized foreach temporary on a single training GPU.
        super().__init__(self.master_parameters, foreach=False, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for parameter, master in zip(self.model_parameters, self.master_parameters):
            if parameter is not master:
                master.grad = None if parameter.grad is None else parameter.grad.float()
                # Gradients have already been clipped by Trainer. Release the
                # BF16 copy as FP32 gradients are created to bound peak memory.
                parameter.grad = None
        super().step()
        for parameter, master in zip(self.model_parameters, self.master_parameters):
            if parameter is not master:
                parameter.copy_(master)
        return loss

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for parameter, master in zip(self.model_parameters, self.master_parameters):
            if parameter is not master and parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.detach_()
                    parameter.grad.zero_()

    def state_dict(self):
        state = super().state_dict()
        state["master_weights"] = [
            master.detach() if parameter is not master else None
            for parameter, master in zip(self.model_parameters, self.master_parameters)
        ]
        return state

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        weights = state_dict.get("master_weights")
        if weights is None or len(weights) != len(self.master_parameters):
            raise ValueError("Checkpoint is missing compatible FP32 master weights")
        for parameter, master, saved in zip(
            self.model_parameters, self.master_parameters, weights
        ):
            if parameter is not master:
                if (
                    saved is None
                    or saved.shape != master.shape
                    or saved.dtype != torch.float32
                ):
                    raise ValueError("Checkpoint FP32 master weight is incompatible")
                master.copy_(saved)
                parameter.copy_(master)
            elif saved is not None:
                raise ValueError("Checkpoint master parameter layout differs")
        super().load_state_dict(
            {k: v for k, v in state_dict.items() if k != "master_weights"}
        )
