"""AdamW and optional Muon with FP32 master weights for BF16/FP16 training."""

from __future__ import annotations

import torch
from torch.optim import AdamW, Muon, Optimizer

from .config import TrainerConfig


class _MasterWeights:
    """Keep small updates and optimizer state in FP32; run the model in BF16.

    Master parameters are optimizer-owned, so ordinary optimizer checkpoint loading
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
        super().__init__(self.master_parameters, **kwargs)

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


class MasterAdamW(_MasterWeights, AdamW):
    """AdamW with FP32 master weights and moments, including on checkpoint load."""

    def __init__(self, parameters, **kwargs):
        # Avoid a parameter-sized foreach temporary on a single training GPU.
        super().__init__(parameters, foreach=False, **kwargs)


class MasterMuon(_MasterWeights, Muon):
    """Native Muon with FP32 master weights and momentum; BF16 NS iterations."""


def muon_parameter_groups(
    model: torch.nn.Module,
) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Keep embeddings/heads and non-matrix parameters on auxiliary AdamW.

    Resolve exclusions by parameter identity so tied embedding/output weights
    appear exactly once. HF output-embedding accessors and conventional module
    names also cover untied heads. Convolution tensors stay on AdamW: native
    torch.optim.Muon accepts two-dimensional matrices only.
    """
    excluded = set()
    for name, module in model.named_modules():
        if isinstance(
            module,
            (
                torch.nn.Embedding,
                torch.nn.EmbeddingBag,
                torch.nn.LayerNorm,
                torch.nn.RMSNorm,
            ),
        ) or (
            name.rsplit(".", 1)[-1]
            in {"lm_head", "head", "classifier", "output", "output_layer"}
        ):
            excluded.update(id(parameter) for parameter in module.parameters())
    for name in ("get_input_embeddings", "get_output_embeddings"):
        accessor = getattr(model, name, None)
        if callable(accessor):
            module = accessor()
            if module is not None:
                excluded.update(id(parameter) for parameter in module.parameters())
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {"muon": [], "adamw": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            kind = (
                "muon"
                if parameter.ndim == 2
                and id(parameter) not in excluded
                and name.rsplit(".", 1)[-1] != "bias"
                else "adamw"
            )
            groups[kind].append((name, parameter))
    return groups


class MuonAdamW(Optimizer):
    """Expose Muon + auxiliary AdamW as one scheduled, checkpointable optimizer.

    Delegate math to native PyTorch optimizers. Both child parameter groups are
    shared with this facade so LambdaLR updates both learning rates. FP32 master
    weights are enabled independently for each low-precision parameter group.
    """

    def __init__(self, model: torch.nn.Module, config: TrainerConfig, *, lr: float):
        groups = muon_parameter_groups(model)
        self.parameter_names = {
            kind: [name for name, _ in named] for kind, named in groups.items()
        }
        self.optimizers = {}
        options = config.muon
        for kind, named in groups.items():
            if not named:
                continue
            parameters = [parameter for _, parameter in named]
            low_precision = any(
                p.dtype in (torch.bfloat16, torch.float16) for p in parameters
            )
            if kind == "muon":
                optimizer_type = MasterMuon if low_precision else Muon
                optimizer = optimizer_type(
                    parameters,
                    lr=lr,
                    weight_decay=config.weight_decay,
                    momentum=options.momentum,
                    nesterov=options.nesterov,
                    ns_steps=options.ns_steps,
                    adjust_lr_fn=options.adjust_lr_fn,
                )
            else:
                optimizer_type = MasterAdamW if low_precision else AdamW
                kwargs = {} if low_precision else {"foreach": False}
                optimizer = optimizer_type(
                    parameters,
                    lr=lr if options.adamw_lr is None else options.adamw_lr,
                    weight_decay=config.weight_decay,
                    **kwargs,
                )
            for group in optimizer.param_groups:
                group["optimizer_kind"] = kind
            self.optimizers[kind] = optimizer
        self._building_groups = True
        super().__init__(self._child_groups(), {"lr": lr})
        self._building_groups = False

    def add_param_group(self, param_group):
        if not self._building_groups:
            raise RuntimeError(
                "Muon/AdamW parameter groups are fixed; rebuild the optimizer"
            )
        super().add_param_group(param_group)

    def _child_groups(self):
        return [
            group
            for optimizer in self.optimizers.values()
            for group in optimizer.param_groups
        ]

    def _refresh_state(self):
        # Child load_state_dict replaces its dictionaries; rebind the facade too.
        self.param_groups = self._child_groups()
        self.state.clear()
        for optimizer in self.optimizers.values():
            self.state.update(optimizer.state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers.values():
            optimizer.step()
        self._refresh_state()
        return loss

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "optimizer_type": "muon_adamw",
            "format_version": 1,
            "parameter_names": self.parameter_names,
            "optimizers": {
                kind: optimizer.state_dict()
                for kind, optimizer in self.optimizers.items()
            },
        }

    def load_state_dict(self, state_dict):
        if (
            state_dict.get("optimizer_type") != "muon_adamw"
            or state_dict.get("format_version") != 1
        ):
            raise ValueError("Checkpoint is not a compatible Muon/AdamW optimizer")
        if state_dict.get("parameter_names") != self.parameter_names or set(
            state_dict.get("optimizers", {})
        ) != set(self.optimizers):
            raise ValueError("Checkpoint Muon/AdamW parameter assignment differs")
        for kind, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][kind])
        self._refresh_state()


def build_optimizer(
    model: torch.nn.Module, config: TrainerConfig, *, lr: float | None = None
) -> Optimizer:
    """Select AdamW by default, or opt into mixed Muon/AdamW parameter groups."""
    learning_rate = config.learning_rate if lr is None else lr
    if config.optimizer == "muon":
        return MuonAdamW(model, config, lr=learning_rate)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_type = (
        MasterAdamW
        if any(p.dtype in (torch.bfloat16, torch.float16) for p in parameters)
        else AdamW
    )
    return optimizer_type(
        parameters, lr=learning_rate, weight_decay=config.weight_decay
    )
