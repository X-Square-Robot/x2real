"""Shared YAML/JSON helpers for manaenv writable code.

This module centralizes the small subset of Arena IO helpers used by
``Translator`` and ``ManaPolicyCFG`` while enforcing a stricter contract:
YAML roots consumed through the public loader must be mappings.
"""

from __future__ import annotations

import json
import types
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


def load_dict_from_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """Load a YAML file whose root node must be a mapping."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if path.stat().st_size == 0:
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def convert_yaml_value(field_type: type, value: Any) -> Any:
    """Convert YAML value to the appropriate type based on field annotation."""
    if field_type == Path or (
        hasattr(field_type, "__origin__") and field_type.__origin__ is Union and Path in field_type.__args__
    ):
        if isinstance(value, str):
            return Path(value)
        return value

    if hasattr(field_type, "__origin__") and field_type.__origin__ is tuple:
        if isinstance(value, list):
            return tuple(value)
        return value

    if field_type in (str, int, float, bool):
        return field_type(value)

    if isinstance(field_type, str):
        scalar_type = {"str": str, "int": int, "float": float, "bool": bool}.get(field_type)
        if scalar_type is not None:
            return scalar_type(value)

    return value


def create_config_from_dict_strict[  # noqa: C901
    ConfigType
](data: dict[str, Any], config_class: type[ConfigType], *, context: str = "",) -> ConfigType:
    """Create a dataclass config from YAML data with exact field-name parity.

    This helper is intentionally small: it validates that every payload key
    exists as a dataclass field, applies the same field-local YAML value
    conversion used by the legacy loader, and then instantiates the CFG.
    It does not rename fields or infer domain semantics.
    """
    if not isinstance(data, dict):
        label = f"{context}: " if context else ""
        raise TypeError(f"{label}{config_class.__name__} payload must be a mapping, got {type(data).__name__}")

    # Registry providers may resolve a YAML key to a plain class; fail with
    # context instead of the bare ``fields()`` TypeError.
    if not is_dataclass(config_class):
        label = f"{context}: " if context else ""
        raise TypeError(
            f"{label}{config_class.__module__}.{config_class.__qualname__} resolved as a CFG "
            "but is not a dataclass; reflective YAML keys must map to dataclass CFGs"
        )

    try:
        resolved_types = get_type_hints(config_class)
    except Exception:
        resolved_types = {}
    field_types = {field.name: resolved_types.get(field.name, field.type) for field in fields(config_class)}
    unknown_fields = sorted(set(data) - set(field_types))
    if unknown_fields:
        label = f"{context}: " if context else ""
        available = ", ".join(sorted(field_types))
        unknown = ", ".join(unknown_fields)
        raise ValueError(
            f"{label}{config_class.__name__} got unknown field(s): {unknown}. "
            f"YAML keys must match CFG dataclass fields exactly. Available fields: {available}"
        )

    def nested_dataclass_type(field_type: Any) -> type | None:
        if isinstance(field_type, type) and is_dataclass(field_type):
            return field_type
        origin = get_origin(field_type)
        if origin in (Union, types.UnionType):
            dataclass_args = [arg for arg in get_args(field_type) if isinstance(arg, type) and is_dataclass(arg)]
            if len(dataclass_args) == 1:
                return dataclass_args[0]
        return None

    def strict_element_dataclass_type(annotation: Any) -> type | None:
        """Resolve a container element annotation to a dataclass, dicts not allowed.

        Only ``SomeCFG`` and ``SomeCFG | None`` qualify. Unions that
        explicitly admit non-dataclass payloads (e.g. ``CFG | dict``)
        keep elements raw so authors retain that escape hatch.
        """
        if isinstance(annotation, type) and is_dataclass(annotation):
            return annotation
        origin = get_origin(annotation)
        if origin in (Union, types.UnionType):
            args = get_args(annotation)
            dataclass_args = [arg for arg in args if isinstance(arg, type) and is_dataclass(arg)]
            other_args = [arg for arg in args if arg not in dataclass_args and arg is not type(None)]
            if len(dataclass_args) == 1 and not other_args:
                return dataclass_args[0]
        return None

    def construct_element(item: Any, element_type: type, item_context: str) -> Any:
        if isinstance(item, element_type):
            return item
        if isinstance(item, dict) and len(item) == 1:
            # Explicitly CFG-named element: a single-key wrapper whose key was
            # already reflected by the registry into the declared element type.
            sole_key, sole_value = next(iter(item.items()))
            if isinstance(sole_value, element_type):
                return sole_value
            if is_dataclass(sole_value):
                raise TypeError(
                    f"{item_context}.{sole_key}: reflected to {type(sole_value).__name__}, "
                    f"but the declared element type is {element_type.__name__}"
                )
        if not isinstance(item, dict):
            raise TypeError(
                f"{item_context}: expected {element_type.__name__} payload mapping, got {type(item).__name__}"
            )
        return create_config_from_dict_strict(item, element_type, context=item_context)

    def convert_field_value(field_name: str, field_type: Any, value: Any) -> Any:
        field_context = f"{context}.{field_name}" if context else field_name
        nested_type = nested_dataclass_type(field_type)
        if nested_type is not None:
            if isinstance(value, nested_type):
                return value
            if value is None:
                value = {}
            if not isinstance(value, dict):
                return convert_yaml_value(field_type, value)
            if len(value) == 1:
                # Explicitly CFG-named payload: a single-key wrapper whose key was
                # already reflected by the registry into a subclass of the declared
                # field type (polymorphic fields such as ``implementation:``).
                sole_key, sole_value = next(iter(value.items()))
                if isinstance(sole_value, nested_type):
                    return sole_value
                if is_dataclass(sole_value) and not isinstance(sole_value, type):
                    raise TypeError(
                        f"{field_context}.{sole_key}: reflected to {type(sole_value).__name__}, "
                        f"but the declared field type is {nested_type.__name__}"
                    )
            return create_config_from_dict_strict(value, nested_type, context=field_context)

        origin = get_origin(field_type)
        type_args = get_args(field_type)
        if origin is list and len(type_args) == 1 and isinstance(value, list):
            element_type = strict_element_dataclass_type(type_args[0])
            if element_type is not None:
                return [
                    construct_element(item, element_type, f"{field_context}[{idx}]") for idx, item in enumerate(value)
                ]
        if origin is dict and len(type_args) == 2 and type_args[0] is str and isinstance(value, dict):
            value_type = strict_element_dataclass_type(type_args[1])
            if value_type is not None:
                return {
                    key: construct_element(item, value_type, f"{field_context}.{key}") for key, item in value.items()
                }
        return convert_yaml_value(field_type, value)

    converted_data: dict[str, Any] = {}
    for field_name, value in data.items():
        try:
            converted_data[field_name] = convert_field_value(field_name, field_types[field_name], value)
        except Exception as exc:
            label = f"{context}: " if context else ""
            raise ValueError(
                f"{label}failed to convert field {field_name!r} for {config_class.__name__}: {exc}"
            ) from exc

    try:
        return config_class(**converted_data)
    except Exception as exc:
        label = f"{context}: " if context else ""
        raise ValueError(f"{label}failed to create {config_class.__name__}: {exc}") from exc


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write mapping data to YAML, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write mapping data to JSON, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    """Read a JSON mapping if the file exists and parses successfully."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None
