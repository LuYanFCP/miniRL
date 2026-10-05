"""Load explicitly configured Python metric functions without a global registry."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PluginConfig:
    name: str
    entrypoint: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", self.name):
            raise ValueError(
                "Plugin name must start with a letter and contain letters, digits, _ or -"
            )
        split_entrypoint(self.entrypoint)


def split_entrypoint(entrypoint: str) -> tuple[str, str]:
    source, separator, function = entrypoint.rpartition(":")
    if not separator or not source or not function.isidentifier():
        raise ValueError(
            "Plugin entrypoint must be 'package.module:function' or 'file.py:function'"
        )
    return source, function


def resolve_entrypoint(entrypoint: str, base: Path) -> str:
    source, function = split_entrypoint(entrypoint)
    if source.endswith(".py"):
        source = str((base / Path(source).expanduser()).resolve())
        if not Path(source).is_file():
            raise ValueError(f"Plugin file does not exist: {source}")
    return f"{source}:{function}"


def load_plugin(config: PluginConfig) -> Callable[..., Any]:
    """Import trusted code only when a run starts; config validation never imports it."""
    source, function = split_entrypoint(config.entrypoint)
    if source.endswith(".py"):
        path = Path(source).resolve()
        name = "_minirl_plugin_" + hashlib.sha256(str(path).encode()).hexdigest()
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load plugin: {path}")
        module = importlib.util.module_from_spec(spec)
        # Dataclasses and other introspection require a registered module.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    else:
        module = importlib.import_module(source)
    callback = getattr(module, function, None)
    if not callable(callback) or inspect.iscoroutinefunction(callback):
        raise ValueError(f"Plugin {config.entrypoint!r} must be a synchronous callable")
    # Fail before loading model weights for unknown or missing keyword arguments.
    inspect.signature(callback).bind(object(), **config.kwargs)
    return callback


def plugin_fingerprint(config: PluginConfig) -> str:
    """Fingerprint the entrypoint source for recipe resume checks."""
    source, _ = split_entrypoint(config.entrypoint)
    if source.endswith(".py"):
        path = Path(source)
    else:
        spec = importlib.util.find_spec(source)
        if spec is None or spec.origin is None:
            raise ValueError(f"Cannot find plugin module: {source}")
        path = Path(spec.origin)
    return hashlib.sha256(path.read_bytes()).hexdigest()
