"""Registry-backed YAML to CFG reflection helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from manaenv.pipeline.registry.global_provider import global_registry
from manaenv.utils.config_io import create_config_from_dict_strict

__all__ = ["reflect_yaml_dict"]


def _yaml_key_to_cfg_symbol(yaml_key: Any) -> str:
    key = str(yaml_key)
    if "_" not in key and any(char.isupper() for char in key):
        return key
    return "".join(part.capitalize() for part in key.split("_"))


def reflect_yaml_dict(
    data: Mapping[str, Any],
    *,
    context: str = "yaml",
    ignore_unknown: bool = False,
    skip_payload_keys: set[str] | frozenset[str] = frozenset(),
    extra_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reflect a YAML mapping into ``{yaml_key: CFG | list[CFG]}``.

    The function is intentionally registry-backed and domain-light:
    callers provide only the symbol convention. Discovery itself is owned
    by ``global_provider`` through ``global_registry``.
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"{context}: expected a mapping, got {type(data).__name__}")

    reflected: dict[str, Any] = {}
    for yaml_key, payload in data.items():
        symbol = _yaml_key_to_cfg_symbol(yaml_key)
        try:
            cfg_class = global_registry.get_cfg(symbol)
        except (AttributeError, ImportError, ModuleNotFoundError, KeyError):
            # KeyError: the registry's "did you mean 'pd'?" near-miss heuristic
            # (lazy.get_spec) raises when a case-variant symbol is already
            # registered by an earlier build sharing the process-global registry
            # (e.g. a persistent editor engine swapping scenes). Under best-effort
            # reflection over data-only nested trees (actuator_params groups like
            # stablepd/pd), that probe must stay order-independent — skip it exactly
            # as a clean not-found would, so a swapped scene behaves like a fresh load.
            if ignore_unknown:
                continue
            raise
        if cfg_class is None:
            if ignore_unknown:
                continue
            raise KeyError(f"{context}: YAML key {yaml_key!r} did not resolve to CFG symbol {symbol!r}")

        def construct_one(item: Any, item_context: str) -> Any:
            if isinstance(item, cfg_class):
                return item
            if item is None:
                item = {}
            if not isinstance(item, dict):
                raise TypeError(f"{item_context}: expected CFG payload mapping, got {type(item).__name__}")
            cfg_payload = {key: value for key, value in item.items() if key not in skip_payload_keys}
            if extra_payload:
                cfg_payload.update(extra_payload)
            return create_config_from_dict_strict(cfg_payload, cfg_class, context=item_context)

        entry_context = f"{context}.{yaml_key}"
        if isinstance(payload, list):
            reflected[yaml_key] = [construct_one(item, f"{entry_context}[{idx}]") for idx, item in enumerate(payload)]
        else:
            reflected[yaml_key] = construct_one(payload, entry_context)
    return reflected
