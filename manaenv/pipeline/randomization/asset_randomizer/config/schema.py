from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import ConfigError
from ..model import ApplyOptions, RandomizationRule, RandomizationSpec, TargetQuery

MATERIAL_SLOT_MODES = frozenset({"auto", "all", "explicit"})


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    raise ConfigError(f"expected string or list of strings, got {type(value).__name__}")


def _parse_material_slots(value: Any) -> tuple[str, tuple[str, ...]]:
    if value is None:
        return "auto", ()
    if isinstance(value, str):
        mode = value.strip()
        if mode in {"auto", "all"}:
            return mode, ()
        return "explicit", (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "explicit", tuple(str(item) for item in value)
    if isinstance(value, Mapping):
        mode = str(value.get("mode") or "auto").strip()
        if mode not in MATERIAL_SLOT_MODES:
            raise ConfigError(f"material_slots.mode must be one of {sorted(MATERIAL_SLOT_MODES)}, got {mode!r}")
        slots = _as_string_tuple(value.get("slots"))
        if mode == "explicit" and not slots:
            raise ConfigError("material_slots.mode explicit requires non-empty slots")
        if mode in {"auto", "all"} and slots:
            raise ConfigError(f"material_slots.mode {mode} must not define slots")
        return mode, slots
    raise ConfigError(f"material_slots must be string, list, or mapping, got {type(value).__name__}")


def parse_target_query(data: Mapping[str, Any] | None) -> TargetQuery:
    data = data or {}
    if not isinstance(data, Mapping):
        raise ConfigError("targets must be a mapping")
    material_slot_mode, material_slots = _parse_material_slots(data.get("material_slots"))
    return TargetQuery(
        asset_names=_as_string_tuple(data.get("asset_names")),
        prim_paths=_as_string_tuple(data.get("prim_paths")),
        semantic_tags=_as_string_tuple(data.get("semantic_tags")),
        material_slots=material_slots,
        material_slot_mode=material_slot_mode,
    )


def parse_apply_options(data: Mapping[str, Any] | None) -> ApplyOptions:
    data = data or {}
    if not isinstance(data, Mapping):
        raise ConfigError("apply must be a mapping")
    return ApplyOptions(
        clone_shared_materials=bool(data.get("clone_shared_materials", True)),
    )


def parse_randomization_spec(data: Mapping[str, Any]) -> RandomizationSpec:
    if not isinstance(data, Mapping):
        raise ConfigError("randomization spec must be a mapping")
    schema_version = int(data.get("schema_version", 1))
    payload = data.get("asset_randomization", data)
    if not isinstance(payload, Mapping):
        raise ConfigError("asset_randomization must be a mapping")
    rules_data = payload.get("rules", ())
    if not isinstance(rules_data, Sequence) or isinstance(rules_data, (str, bytes, bytearray)):
        raise ConfigError("asset_randomization.rules must be a list")
    rules: list[RandomizationRule] = []
    for index, item in enumerate(rules_data):
        if not isinstance(item, Mapping):
            raise ConfigError(f"rule {index} must be a mapping")
        name = str(item.get("name") or f"rule_{index}")
        policy = item.get("policy")
        if not isinstance(policy, Mapping):
            raise ConfigError(f"rule {name} requires a policy mapping")
        rules.append(
            RandomizationRule(
                name=name,
                targets=parse_target_query(item.get("targets")),
                policy=dict(policy),
                apply=parse_apply_options(item.get("apply")),
            )
        )
    return RandomizationSpec(
        enabled=bool(payload.get("enabled", True)),
        seed_scope=str(payload.get("seed_scope", "episode")),
        rules=tuple(rules),
        schema_version=schema_version,
    )
