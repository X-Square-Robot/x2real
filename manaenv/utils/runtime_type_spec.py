"""Runtime-type materializer for structured ``semantic_tags``.

Structured ``semantic_tags`` (for example
``- Openable: {openable_joint_name: ...}``) describe runtime
affordance mixins. The main path is pre-instantiation:

* parse structured tags before asset construction;
* synthesize a cooperative class such as
  ``type("ButtonWithTouchable", (Touchable, Button), attrs)`` when
  the asset class does not already inherit the affordance;
* pass the reflected affordance kwargs into the constructor so each
  affordance ``__init__`` runs in the normal MRO.

``materialize_runtime_types`` remains as a legacy validation helper
for already-instantiated assets, but it never mutates ``__class__``.

Affordance classes are resolved through the global lazy registry's
``affordance`` discovery domain. The yaml symbol therefore matches
the class symbol: ``Touchable`` -> ``manaenv.assets.affordances.touchable.Touchable``
or ``Openable`` -> ``isaaclab_arena.affordances.openable.Openable``.
Tests and custom extensions can still register explicit specs as
overrides without changing the production discovery path.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.registry.global_provider import global_registry
from manaenv.utils.semantic_tags import normalize_semantic_tags


@dataclass(frozen=True)
class RuntimeTypeSpec:
    """Mapping from a semantic-tag name to a runtime affordance class.

    Attributes:
        name: Tag name as written in yaml (matches ``semantic_tag_attrs``
            key).
        runtime_type: Affordance / mixin class to attach.
        required_attrs: Attribute names that must be present (supplied
            either by yaml or by the asset class) for materialization
            to succeed.
        default_attrs: Defaults to merge in when neither yaml nor the
            asset class supplies a value.
    """

    name: str
    runtime_type: type
    required_attrs: tuple[str, ...] = ()
    default_attrs: Mapping[str, Any] = field(default_factory=dict)
    allowed_attrs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeTypePreparation:
    """Pre-instantiation runtime type plan for one asset class."""

    asset_class: type
    init_kwargs: Mapping[str, Any] = field(default_factory=dict)
    materialized_names: tuple[str, ...] = ()


_OVERRIDES: dict[str, RuntimeTypeSpec] = {}
_DISCOVERED: dict[str, RuntimeTypeSpec] = {}


def register_runtime_type_spec(spec: RuntimeTypeSpec) -> None:
    """Register or replace a runtime-type spec override keyed on ``spec.name``."""
    _OVERRIDES[spec.name] = spec


def get_runtime_type_spec(name: str) -> RuntimeTypeSpec | None:
    """Return the registered spec for *name*, or ``None`` when missing."""
    override = _OVERRIDES.get(name)
    if override is not None:
        return override
    discovered = _DISCOVERED.get(name)
    if discovered is not None:
        return discovered
    runtime_type = _resolve_affordance_type(name)
    if runtime_type is None:
        return None
    spec = _spec_from_affordance_type(name, runtime_type)
    _DISCOVERED[name] = spec
    return spec


def list_runtime_type_specs() -> tuple[RuntimeTypeSpec, ...]:
    return tuple(_OVERRIDES.values()) + tuple(_DISCOVERED.values())


def clear_runtime_type_registry() -> None:
    """Test helper: drop registered override specs and discovered cache."""
    _OVERRIDES.clear()
    _DISCOVERED.clear()


def _resolve_affordance_type(name: str) -> type | None:
    """Resolve *name* through the lazy affordance discovery domain."""
    try:
        from manaenv.assets.affordances.discovery import discover_affordance
    except ImportError:
        return None

    spec = discover_affordance(name)
    if spec is None:
        return None
    global_registry.register(spec)
    try:
        runtime_type = global_registry.get_impl(name)
    except ModuleNotFoundError:
        return None
    if runtime_type is None:
        return None

    from isaaclab_arena.affordances.affordance_base import AffordanceBase

    if not issubclass(runtime_type, AffordanceBase):
        raise TypeError(f"Resolved affordance symbol {name!r} is not an AffordanceBase subclass.")
    return runtime_type


def _spec_from_affordance_type(name: str, runtime_type: type) -> RuntimeTypeSpec:
    """Derive yaml attrs schema from an affordance class constructor."""
    signature = inspect.signature(runtime_type.__init__)
    required_attrs: list[str] = []
    default_attrs: dict[str, Any] = {}
    allowed_attrs: list[str] = []
    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        allowed_attrs.append(param_name)
        if param.default is inspect.Parameter.empty:
            required_attrs.append(param_name)
        else:
            default_attrs[param_name] = param.default
    return RuntimeTypeSpec(
        name=name,
        runtime_type=runtime_type,
        required_attrs=tuple(required_attrs),
        default_attrs=default_attrs,
        allowed_attrs=tuple(allowed_attrs),
    )


def _read_class_data_attr(cls: type, name: str) -> Any:
    """Return a class-level data attr by name, ignoring methods / descriptors.

    Walks ``cls.__mro__`` and returns the first occurrence whose raw
    value is a plain Python data type. Methods, ``property``,
    ``classmethod``, ``staticmethod`` and other descriptors return
    ``None``.
    """
    for klass in cls.__mro__:
        if name in klass.__dict__:
            raw = klass.__dict__[name]
            if isinstance(raw, (int, float, str, bool, tuple, list, dict, type(None))):
                return raw
            return None
    return None


def _candidate_keys(spec: RuntimeTypeSpec) -> list[str]:
    return list(spec.allowed_attrs or ()) or list(spec.default_attrs or ()) + list(spec.required_attrs)


def _resolve_attrs_from_class(
    asset_cls: type,
    spec: RuntimeTypeSpec,
    yaml_attrs: Mapping[str, Any] | None,
    *,
    asset_name: str,
    validate_required: bool = True,
) -> dict[str, Any]:
    """Merge spec defaults / class data attrs / yaml attrs with yaml winning."""
    merged: dict[str, Any] = dict(spec.default_attrs or {})
    candidate_keys = _candidate_keys(spec)
    for k in candidate_keys:
        cls_value = _read_class_data_attr(asset_cls, k)
        if cls_value is not None:
            merged[k] = cls_value
    unknown = sorted(set(yaml_attrs or {}) - set(candidate_keys))
    if unknown:
        raise ValueError(
            f"prepare_runtime_asset_class: asset {asset_name!r} declares "
            f"semantic_tag {spec.name!r} with unsupported attr(s): {unknown!r}"
        )
    for k, v in (yaml_attrs or {}).items():
        merged[k] = v
    missing = [k for k in spec.required_attrs if merged.get(k) is None]
    if validate_required and missing:
        raise ValueError(
            f"prepare_runtime_asset_class: asset {asset_name!r} declares "
            f"semantic_tag {spec.name!r} but is missing required attr(s): {missing!r}"
        )
    return merged


def _resolve_attrs(
    asset: Any,
    spec: RuntimeTypeSpec,
    yaml_attrs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge defaults / class attrs / instance attrs / yaml attrs with yaml winning."""
    merged = _resolve_attrs_from_class(
        type(asset),
        spec,
        yaml_attrs,
        asset_name=getattr(asset, "name", type(asset).__name__),
        validate_required=False,
    )
    for k in _candidate_keys(spec):
        if k not in merged and hasattr(asset, k):
            inst_value = getattr(asset, k)
            if not callable(inst_value):
                merged[k] = inst_value
    missing = [k for k in spec.required_attrs if merged.get(k) is None]
    if missing:
        raise ValueError(
            f"materialize_runtime_types: asset {getattr(asset, 'name', '<unnamed>')!r} declares "
            f"semantic_tag {spec.name!r} but is missing required attr(s): {missing!r}"
        )
    return merged


def _safe_type_suffix(name: str) -> str:
    suffix = re.sub(r"[^0-9A-Za-z_]+", "_", name.title())
    return suffix.strip("_") or "Runtime"


def prepare_runtime_asset_class(asset_class: type, semantic_tags: Any) -> RuntimeTypePreparation:
    """Return the class / constructor kwargs needed for structured semantic tags.

    Only structured tags are materialized. Bare scalar tags remain
    metadata and are applied after construction by
    :func:`manaenv.utils.semantic_tags.apply_semantic_tags`.
    """
    tag_cfgs = normalize_semantic_tags(semantic_tags)
    if not tag_cfgs:
        return RuntimeTypePreparation(asset_class=asset_class)

    class_attrs: dict[str, Any] = {}
    init_kwargs: dict[str, Any] = {}
    missing_runtime_types: list[type] = []
    materialized_names: list[str] = []
    asset_name = getattr(asset_class, "name", asset_class.__name__)

    for tag_cfg in tag_cfgs:
        if not tag_cfg.structured and not tag_cfg.attrs:
            continue
        spec = get_runtime_type_spec(tag_cfg.name)
        if spec is None:
            raise ValueError(
                f"prepare_runtime_asset_class: asset {asset_name!r} declares semantic_tag "
                f"{tag_cfg.name!r} but no affordance type was discovered for that symbol."
            )
        attrs = _resolve_attrs_from_class(asset_class, spec, tag_cfg.attrs, asset_name=asset_name)
        class_attrs.update(attrs)
        materialized_names.append(spec.name)
        if not issubclass(asset_class, spec.runtime_type):
            missing_runtime_types.append(spec.runtime_type)
            init_kwargs.update(attrs)

    if not materialized_names:
        return RuntimeTypePreparation(asset_class=asset_class)

    if missing_runtime_types:
        suffix = "With" + "And".join(_safe_type_suffix(name) for name in materialized_names)
        prepared_class = type(
            f"{asset_class.__name__}{suffix}", tuple(missing_runtime_types) + (asset_class,), class_attrs
        )
    elif class_attrs:
        suffix = "With" + "And".join(_safe_type_suffix(name) for name in materialized_names)
        prepared_class = type(f"{asset_class.__name__}{suffix}", (asset_class,), class_attrs)
    else:
        prepared_class = asset_class

    return RuntimeTypePreparation(
        asset_class=prepared_class,
        init_kwargs=init_kwargs,
        materialized_names=tuple(materialized_names),
    )


def _materialize_one(asset: Any, spec: RuntimeTypeSpec, attrs: Mapping[str, Any]) -> None:
    """Apply a single spec to an already-compatible asset instance."""
    if not isinstance(asset, spec.runtime_type):
        raise TypeError(
            f"materialize_runtime_types: asset {getattr(asset, 'name', '<unnamed>')!r} declares "
            f"semantic_tag {spec.name!r}, but {type(asset).__name__} does not inherit "
            f"{spec.runtime_type.__name__}. Compose runtime types before asset construction."
        )
    for k, v in attrs.items():
        setattr(asset, k, v)


def materialize_runtime_types(asset: Any) -> list[str]:
    """Validate already-instantiated affordance declarations.

    Returns the list of spec names that were applied (whether
    ``isinstance`` already held). Names that have no registered /
    discovered affordance spec raise eagerly because structured
    semantic tags describe executable behavior.
    """
    payloads = getattr(asset, "semantic_tag_attrs", None)
    if not payloads:
        return []
    materialized: list[str] = []
    for name, attrs in payloads.items():
        spec = get_runtime_type_spec(name)
        if spec is None:
            raise ValueError(
                f"materialize_runtime_types: asset {getattr(asset, 'name', '<unnamed>')!r} declares "
                f"semantic_tag {name!r} but no affordance type was discovered for that symbol."
            )
        merged_attrs = _resolve_attrs(asset, spec, attrs)
        _materialize_one(asset, spec, merged_attrs)
        materialized.append(name)
    return materialized
