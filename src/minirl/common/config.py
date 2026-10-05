"""Load safe YAML into typed dataclasses with strict field validation."""

from __future__ import annotations

import math
import re
import types
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml
from yaml.nodes import MappingNode


class ConfigError(ValueError):
    """A configuration cannot be read or does not match its declared schema."""


class _ConfigLoader(yaml.SafeLoader):
    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[str, Any]:
        self.flatten_mapping(node)
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError(
                    f"Config keys must be strings at {key_node.start_mark}"
                )
            if key in result:
                raise ConfigError(f"Duplicate key {key!r} at {key_node.start_mark}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


# PyYAML's default float resolver does not recognize common values such as 2e-5.
_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def _convert(value: object, annotation: Any, path: str) -> Any:
    if annotation is Any:
        # Plugin options may contain JSON-like values, never arbitrary YAML objects.
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if isinstance(value, list):
            return [_convert(item, Any, f"{path}[{i}]") for i, item in enumerate(value)]
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return {
                key: _convert(item, Any, f"{path}.{key}") for key, item in value.items()
            }
        raise ConfigError(f"{path}: expected finite JSON-compatible values")
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in arguments:
            return None
        for option in arguments:
            if option is not type(None):
                try:
                    return _convert(value, option, path)
                except ConfigError:
                    pass
        raise ConfigError(f"{path}: expected {annotation}, got {value!r}")
    if origin is Literal:
        if any(type(value) is type(choice) and value == choice for choice in arguments):
            return value
        raise ConfigError(f"{path}: expected one of {arguments}, got {value!r}")
    if is_dataclass(annotation):
        return _parse_dataclass(value, annotation, path)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list")
        return [
            _convert(item, arguments[0], f"{path}[{i}]") for i, item in enumerate(value)
        ]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping")
        return {
            _convert(key, arguments[0], path): _convert(
                item, arguments[1], f"{path}.{key}"
            )
            for key, item in value.items()
        }
    if annotation is float:
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ConfigError(f"{path}: expected a finite number, got {value!r}")
        return float(value)
    if annotation in (str, bool, int):
        if type(value) is not annotation:
            raise ConfigError(f"{path}: expected {annotation.__name__}, got {value!r}")
        return value
    raise TypeError(f"Unsupported config annotation at {path}: {annotation}")


def _parse_dataclass(value: object, schema: Any, path: str) -> Any:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a mapping")
    declared = {field.name: field for field in fields(schema) if field.init}
    unknown = value.keys() - declared.keys()
    if unknown:
        raise ConfigError(f"{path}: unknown fields: {', '.join(sorted(unknown))}")
    hints = get_type_hints(schema)
    kwargs = {}
    for name, field in declared.items():
        if name in value:
            kwargs[name] = _convert(value[name], hints[name], f"{path}.{name}")
        elif field.default is MISSING and field.default_factory is MISSING:
            raise ConfigError(f"{path}.{name}: required field is missing")
    try:
        return schema(**kwargs)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{path}: {error}") from error


def load_yaml_config[T](path: str | Path, schema: type[T]) -> T:
    """Read one YAML document and validate it against a nested dataclass schema.

    Unknown/duplicate fields, unsafe YAML tags, missing required values, and
    implicit type coercions are rejected. Defaults and __post_init__ validators
    come from the schema. File paths inside the document remain schema-owned.
    """
    if not isinstance(schema, type) or not is_dataclass(schema):
        raise TypeError("schema must be a dataclass type")
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=_ConfigLoader)
        return _parse_dataclass(value, schema, "config")
    except (OSError, UnicodeError, yaml.YAMLError, ConfigError) as error:
        raise ConfigError(f"{path}: {error}") from error


def dump_yaml_config(config: object) -> str:
    """Serialize a resolved dataclass configuration without Python-specific tags."""
    if isinstance(config, type) or not is_dataclass(config):
        raise TypeError("config must be a dataclass instance")
    return yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True)
