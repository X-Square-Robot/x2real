from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import permutations
from math import isfinite
from pathlib import Path
from typing import Any, Literal, overload

from manaenv.assets.resolver import is_asset_reference, resolve_asset_path

from ..common.diagnostics import error

DEFAULT_RUNTIME_TOPOLOGY = "pooled"
SUPPORTED_RUNTIME_TOPOLOGIES = frozenset({"active_only", "pooled"})


def _mapping(value: Any, *, target: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error("schema_vnext_mapping_required", "expected a mapping", target=target)
    return value


def _list(value: Any, *, target: str) -> list[Any]:
    if not isinstance(value, list):
        raise error("schema_vnext_list_required", "expected a list", target=target)
    return value


def _non_empty_str(value: Any, *, target: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise error("schema_vnext_id_required", "expected a non-empty string", target=target)
    return text


def _string_tuple(value: Any, *, target: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_non_empty_str(value, target=target),)
    values = _list(value, target=target)
    out = tuple(_non_empty_str(item, target=f"{target}[]") for item in values)
    if len(set(out)) != len(out):
        raise error("schema_vnext_duplicate_id", "ids must be unique", target=target)
    return out


def _string_sequence(value: Any, *, target: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_non_empty_str(value, target=target),)
    values = _list(value, target=target)
    return tuple(_non_empty_str(item, target=f"{target}[]") for item in values)


def _weight(value: Any, *, target: str, default: float = 1.0) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise error(
            "schema_vnext_weight_invalid", "weight must be a finite non-negative number", target=target
        ) from exc
    if not isfinite(parsed) or parsed < 0.0:
        raise error("schema_vnext_weight_invalid", "weight must be a finite non-negative number", target=target)
    return parsed


def _unique(ids: list[str], *, target: str) -> None:
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise error("schema_vnext_duplicate_id", "ids must be unique", target=target, ids=duplicates)


def _reject_unsupported_fields(payload: Mapping[str, Any], *, allowed: set[str], code: str, target: str) -> None:
    unsupported = sorted(str(key) for key in payload if str(key) not in allowed)
    if unsupported:
        raise error(code, "unsupported field", target=target, fields=unsupported)


def scene_runtime_topology_mode(scene_or_payload: Any) -> str:
    """Return the authored runtime-topology mode for a schema-vNext scene.

    Schema-vNext defaults to ``pooled``: every pool candidate within the
    object's ``cache_size`` window is prespawned, and a reset swaps which slot
    is active by writing pose/visibility.  That keeps the episode boundary
    inside one USD topology, so PhysX views, the render child and any worker
    physics survive it.  Scenes that need a fresh concrete asset per episode
    opt in with ``runtime_topology: active_only``, which costs a full
    scene-epoch rebuild (see ``AssembledEnvironment.rebuild_scene_epoch``).
    """
    source = getattr(scene_or_payload, "source", scene_or_payload)
    if not isinstance(source, Mapping):
        return ""
    definition = source.get("definition")
    if not isinstance(definition, Mapping):
        return DEFAULT_RUNTIME_TOPOLOGY
    metadata = definition.get("metadata")
    if not isinstance(metadata, Mapping):
        return DEFAULT_RUNTIME_TOPOLOGY
    return str(metadata.get("runtime_topology") or DEFAULT_RUNTIME_TOPOLOGY).strip().lower()


def scene_uses_active_only_topology(scene_or_payload: Any) -> bool:
    """Whether the scene must contain only its build-time active pool assets."""
    return scene_runtime_topology_mode(scene_or_payload) == "active_only"


def _candidate_weights_positive(items: list[Mapping[str, Any]], *, key: str, target: str) -> None:
    if not any(_weight(item.get("weight"), target=f"{target}.{item.get(key)}") > 0.0 for item in items):
        raise error("schema_vnext_all_weights_zero", "at least one candidate must have positive weight", target=target)


def _strict_ref(ref: str, known: set[str], *, target: str, kind: str) -> None:
    if ref not in known:
        raise error("schema_vnext_unknown_reference", f"unknown {kind} reference", target=target, ref=ref)


def _scalar_or_pair(value: Any, *, target: str) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        parsed = float(value)
        if not isfinite(parsed):
            raise error("schema_vnext_range_invalid", "range scalar must be finite", target=target)
        return (parsed, parsed)
    if isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
        left = float(value[0])
        right = float(value[1])
        if not isfinite(left) or not isfinite(right) or left > right:
            raise error("schema_vnext_range_invalid", "range must be finite [min, max]", target=target)
        return (left, right)
    raise error("schema_vnext_range_invalid", "expected scalar or [min, max]", target=target)


def _positive_range(value: Any, *, target: str) -> tuple[float, float]:
    left, right = _scalar_or_pair(value, target=target)
    if left <= 0.0 or right <= 0.0:
        raise error("schema_vnext_scale_invalid", "scale must be positive", target=target)
    return (left, right)


@overload
def _numeric_vec(value: Any, *, length: Literal[3], target: str) -> tuple[float, float, float]: ...  # noqa: E704


@overload
def _numeric_vec(value: Any, *, length: Literal[4], target: str) -> tuple[float, float, float, float]: ...  # noqa: E704


@overload
def _numeric_vec(value: Any, *, length: int, target: str) -> tuple[float, ...]: ...  # noqa: E704


def _numeric_vec(value: Any, *, length: int, target: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise error("schema_vnext_vector_invalid", f"expected {length} numeric values", target=target)
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise error("schema_vnext_vector_invalid", f"expected {length} numeric values", target=target) from exc
    if not all(isfinite(item) for item in parsed):
        raise error("schema_vnext_vector_invalid", f"expected {length} finite numeric values", target=target)
    return parsed


def _validate_ranges(value: Any, *, target: str) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        _reject_unsupported_fields(
            value,
            allowed={"x", "y"},
            code="schema_vnext_ranges_field_unsupported",
            target=target,
        )
        if "x" not in value or "y" not in value:
            raise error("schema_vnext_ranges_invalid", "ranges mapping requires x and y", target=target)
        _scalar_or_pair(value["x"], target=f"{target}.x")
        _scalar_or_pair(value["y"], target=f"{target}.y")
        return ()
    items = _list(value, target=target)
    ids: list[str] = []
    weighted_items: list[Mapping[str, Any]] = []
    for index, item in enumerate(items):
        data = _mapping(item, target=f"{target}[{index}]")
        _reject_unsupported_fields(
            data,
            allowed={"id", "x", "y", "weight"},
            code="schema_vnext_ranges_field_unsupported",
            target=f"{target}[{index}]",
        )
        item_id = _non_empty_str(data.get("id"), target=f"{target}[{index}].id")
        if "x" not in data or "y" not in data:
            raise error("schema_vnext_ranges_invalid", "range choice requires x and y", target=f"{target}.{item_id}")
        _scalar_or_pair(data["x"], target=f"{target}.{item_id}.x")
        _scalar_or_pair(data["y"], target=f"{target}.{item_id}.y")
        _weight(data.get("weight"), target=f"{target}.{item_id}.weight")
        ids.append(item_id)
        weighted_items.append(data)
    _unique(ids, target=target)
    _candidate_weights_positive(weighted_items, key="id", target=target)
    return tuple(ids)


def _validate_yaws(value: Any, *, target: str) -> tuple[str, ...]:
    if isinstance(value, (int, float)):
        _scalar_or_pair(value, target=target)
        return ()
    if isinstance(value, list) and (len(value) != 0) and all(not isinstance(item, Mapping) for item in value):
        _scalar_or_pair(value, target=target)
        return ()
    items = _list(value, target=target)
    ids: list[str] = []
    weighted_items: list[Mapping[str, Any]] = []
    for index, item in enumerate(items):
        data = _mapping(item, target=f"{target}[{index}]")
        _reject_unsupported_fields(
            data,
            allowed={"id", "value", "range", "weight"},
            code="schema_vnext_yaws_field_unsupported",
            target=f"{target}[{index}]",
        )
        item_id = _non_empty_str(data.get("id"), target=f"{target}[{index}].id")
        if "value" in data:
            _scalar_or_pair(data["value"], target=f"{target}.{item_id}.value")
        elif "range" in data:
            _scalar_or_pair(data["range"], target=f"{target}.{item_id}.range")
        else:
            raise error("schema_vnext_yaws_invalid", "yaw choice requires value or range", target=f"{target}.{item_id}")
        _weight(data.get("weight"), target=f"{target}.{item_id}.weight")
        ids.append(item_id)
        weighted_items.append(data)
    _unique(ids, target=target)
    _candidate_weights_positive(weighted_items, key="id", target=target)
    return tuple(ids)


@dataclass(frozen=True)
class PoolCandidateSpec:
    candidate_id: str
    assets: tuple[str, ...]
    asset_scales: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    scale_normalization: Mapping[str, float] = field(default_factory=dict)
    # Per-asset concrete ``scale`` multiplier declared in the pool JSON entries.
    # Applied BEFORE the candidate-level ``scale_normalization`` (which then only
    # clamps the already-scaled size). Keyed by materialized asset name.
    asset_scale_factors: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetSelectorSpec:
    selector_type: str
    asset_id: str | None = None
    tag: str | None = None
    candidates: tuple[PoolCandidateSpec, ...] = ()
    cache_size: int | None = None
    # Pool cache fill: ``shuffle`` (default) keeps slot 0 = materialized pick and
    # samples remaining slots randomly; ``authored`` maps candidate[i] → slot i
    # for tasks that bind materials/logic to fixed cache slot names (nut_on_screw).
    cache_slot_order: str = "shuffle"

    @classmethod
    def from_dict(cls, data: Any, *, target: str) -> AssetSelectorSpec:
        payload = _mapping(data, target=target)
        selector_type = _non_empty_str(payload.get("type"), target=f"{target}.type")
        if selector_type == "asset_id":
            _reject_unsupported_fields(
                payload, allowed={"type", "id"}, code="schema_vnext_asset_field_unsupported", target=target
            )
            return cls(selector_type=selector_type, asset_id=_non_empty_str(payload.get("id"), target=f"{target}.id"))
        if selector_type == "tag":
            _reject_unsupported_fields(
                payload, allowed={"type", "tag"}, code="schema_vnext_asset_field_unsupported", target=target
            )
            return cls(selector_type=selector_type, tag=_non_empty_str(payload.get("tag"), target=f"{target}.tag"))
        if selector_type == "pool":
            _reject_unsupported_fields(
                payload,
                # ``id`` is ignored; include-merge from an asset_id parent can leave it behind.
                allowed={"type", "candidates", "cache_size", "cache_slot_order", "id"},
                code="schema_vnext_asset_field_unsupported",
                target=target,
            )
            cache_size = _optional_positive_int(payload.get("cache_size"), target=f"{target}.cache_size")
            cache_slot_order = "shuffle"
            if "cache_slot_order" in payload:
                cache_slot_order = (
                    _non_empty_str(payload.get("cache_slot_order"), target=f"{target}.cache_slot_order").strip().lower()
                )
                if cache_slot_order not in {"shuffle", "authored"}:
                    raise error(
                        "schema_vnext_cache_slot_order_invalid",
                        "pool cache_slot_order must be shuffle or authored",
                        target=f"{target}.cache_slot_order",
                        cache_slot_order=cache_slot_order,
                    )
            candidates = []
            seen_assets: set[str] = set()
            for index, raw_candidate in enumerate(_list(payload.get("candidates"), target=f"{target}.candidates")):
                candidate = _mapping(raw_candidate, target=f"{target}.candidates[{index}]")
                _reject_unsupported_fields(
                    candidate,
                    allowed={"id", "assets", "from_file", "item_range", "bbox_filter", "scale_normalization"},
                    code="schema_vnext_pool_candidate_field_unsupported",
                    target=f"{target}.candidates[{index}]",
                )
                candidate_id = _non_empty_str(candidate.get("id"), target=f"{target}.candidates[{index}].id")
                inline_assets = _string_tuple(candidate.get("assets"), target=f"{target}.candidates[{index}].assets")
                file_assets, asset_scales, scale_normalization, asset_scale_factors = _file_pool_assets(
                    candidate,
                    target=f"{target}.candidates[{index}]",
                )
                assets = tuple([*inline_assets, *file_assets])
                if not assets:
                    raise error("schema_vnext_pool_assets_empty", "pool candidate requires assets", target=target)
                repeated = sorted(seen_assets.intersection(assets))
                if repeated:
                    raise error(
                        "schema_vnext_pool_asset_duplicate",
                        "concrete asset ids cannot appear in multiple candidate groups",
                        target=target,
                        assets=repeated,
                    )
                seen_assets.update(assets)
                candidates.append(
                    PoolCandidateSpec(
                        candidate_id=candidate_id,
                        assets=assets,
                        asset_scales=asset_scales,
                        scale_normalization=scale_normalization,
                        asset_scale_factors=asset_scale_factors,
                    )
                )
            _unique([candidate.candidate_id for candidate in candidates], target=f"{target}.candidates")
            if not candidates:
                raise error("schema_vnext_pool_empty", "asset pool requires candidates", target=target)
            return cls(
                selector_type=selector_type,
                candidates=tuple(candidates),
                cache_size=cache_size,
                cache_slot_order=cache_slot_order,
            )
        raise error(
            "schema_vnext_asset_type_unsupported", "unsupported asset selector type", target=target, type=selector_type
        )

    @property
    def candidate_ids(self) -> set[str]:
        return {candidate.candidate_id for candidate in self.candidates}


def _optional_positive_int(value: Any, *, target: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise error("schema_vnext_positive_int_invalid", "expected a positive integer", target=target, value=value)
    return int(value)


def _item_range(value: Any, *, target: str, length: int) -> tuple[int, int | None]:
    """Parse half-open ``[start, end)``; ``end`` may be null for the list end."""
    if value is None:
        return 0, None
    if not isinstance(value, list) or len(value) != 2:
        raise error(
            "schema_vnext_item_range_invalid",
            "item_range must be a two-element list [start, end]",
            target=target,
            value=value,
        )
    start_raw, end_raw = value
    if not isinstance(start_raw, int) or isinstance(start_raw, bool) or start_raw < 0:
        raise error(
            "schema_vnext_item_range_invalid",
            "item_range start must be a non-negative integer",
            target=target,
            value=value,
        )
    start = int(start_raw)
    if end_raw is None:
        end: int | None = None
    elif not isinstance(end_raw, int) or isinstance(end_raw, bool):
        raise error(
            "schema_vnext_item_range_invalid",
            "item_range end must be null or an integer",
            target=target,
            value=value,
        )
    else:
        end = int(end_raw)
        if end <= start:
            raise error(
                "schema_vnext_item_range_invalid",
                "item_range end must be greater than start",
                target=target,
                value=value,
            )
    if start > int(length):
        raise error(
            "schema_vnext_item_range_invalid",
            "item_range start exceeds pool length",
            target=target,
            value=value,
            length=int(length),
        )
    return start, end


def _file_pool_assets(candidate: Mapping[str, Any], *, target: str) -> tuple[
    tuple[str, ...],
    dict[str, tuple[float, float, float]],
    dict[str, float],
    dict[str, float],
]:
    path_value = candidate.get("from_file")
    scale_config = _scale_normalization(candidate.get("scale_normalization"), target=f"{target}.scale_normalization")
    if path_value is None:
        return (), {}, dict(scale_config or {}), {}
    path = _resolve_asset_pool_path(_non_empty_str(path_value, target=f"{target}.from_file"))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise error(
            "schema_vnext_asset_pool_file_missing", "asset pool file does not exist", target=target, path=str(path)
        ) from exc
    except json.JSONDecodeError as exc:
        raise error(
            "schema_vnext_asset_pool_file_invalid", "asset pool file is not valid JSON", target=target, path=str(path)
        ) from exc
    if not isinstance(payload, Mapping):
        raise error("schema_vnext_asset_pool_file_invalid", "asset pool file must contain a JSON object", target=target)
    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, list):
        raise error("schema_vnext_asset_pool_file_invalid", "asset pool file must contain assets list", target=target)
    start, end = _item_range(candidate.get("item_range"), target=f"{target}.item_range", length=len(raw_assets))
    raw_assets = raw_assets[start:end]
    if candidate.get("item_range") is not None and not raw_assets:
        raise error(
            "schema_vnext_item_range_empty",
            "item_range selects no assets from the pool file",
            target=target,
            item_range=candidate.get("item_range"),
            length=len(payload.get("assets") or []),
        )
    bbox_filter = _bbox_filter(candidate.get("bbox_filter"), target=f"{target}.bbox_filter")
    assets: list[str] = []
    asset_scales: dict[str, tuple[float, float, float]] = {}
    asset_scale_factors: dict[str, float] = {}
    for index, raw_asset in enumerate(raw_assets):
        asset_name, bbox, scale_factor = _asset_pool_file_entry(
            raw_asset, target=f"{target}.from_file.assets[{start + index}]"
        )
        if not asset_name:
            continue
        if asset_name in assets:
            continue
        assets.append(asset_name)
        if scale_factor is not None:
            asset_scale_factors[asset_name] = scale_factor
        if bbox is not None:
            longest = max(bbox)
            if not _bbox_in_range(longest, bbox_filter):
                assets.pop()
                asset_scale_factors.pop(asset_name, None)
                continue
            # A known bbox fully determines the scale at parse time, so record
            # one for every in-range asset (identity when no rescale is needed).
            # This keeps the runtime windowed cache from lazily opening the USD
            # stage just to measure a bbox we already have. The per-asset concrete
            # ``scale`` multiplier is applied first, the candidate
            # scale_normalization second.
            scale = compose_scale_factor(list(bbox), scale_factor, scale_config)
            asset_scales[asset_name] = scale if scale is not None else (1.0, 1.0, 1.0)
    if not assets:
        raise error("schema_vnext_asset_pool_file_empty", "asset pool file produced no usable assets", target=target)
    return tuple(assets), asset_scales, dict(scale_config or {}), asset_scale_factors


def _resolve_asset_pool_path(path_text: str) -> Path:
    if is_asset_reference(path_text):
        return resolve_asset_path(path_text)
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    package_root = Path(__file__).resolve().parents[4]
    candidates = [
        package_root / "defination" / path,
        package_root / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _bbox_filter(value: Any, *, target: str) -> dict[str, float | None]:
    if value is None:
        return {"min": None, "max": None}
    payload = _mapping(value, target=target)
    _reject_unsupported_fields(
        payload,
        allowed={"min_longest_edge", "max_longest_edge", "hard_min_edge", "hard_max_edge"},
        code="schema_vnext_bbox_filter_field_unsupported",
        target=target,
    )
    return {
        "min": _optional_non_negative_float(
            payload.get("min_longest_edge", payload.get("hard_min_edge")),
            target=f"{target}.min_longest_edge",
        ),
        "max": _optional_non_negative_float(
            payload.get("max_longest_edge", payload.get("hard_max_edge")),
            target=f"{target}.max_longest_edge",
        ),
    }


def _scale_normalization(value: Any, *, target: str) -> dict[str, float] | None:
    if value is None:
        return None
    payload = _mapping(value, target=target)
    _reject_unsupported_fields(
        payload,
        allowed={
            "target_longest_edge",
            "min_longest_edge",
            "max_longest_edge",
            "target_shortest_edge",
            "min_shortest_edge",
            "max_shortest_edge",
            "min_scale",
            "max_scale",
        },
        code="schema_vnext_scale_normalization_field_unsupported",
        target=target,
    )
    shortest_keys = ("target_shortest_edge", "min_shortest_edge", "max_shortest_edge")
    longest_keys = ("target_longest_edge", "min_longest_edge", "max_longest_edge")
    has_shortest = any(payload.get(key) is not None for key in shortest_keys)
    has_longest = any(payload.get(key) is not None for key in longest_keys)
    if has_shortest and has_longest:
        target_shortest = payload.get("target_shortest_edge")
        target_longest = payload.get("target_longest_edge")
        if target_shortest is not None and target_longest is not None:
            # Anisotropic dual-target: longest axis -> target_longest_edge, other
            # axes -> target_shortest_edge. Bound keys cannot mix with dual targets.
            bound_keys = ("min_shortest_edge", "max_shortest_edge", "min_longest_edge", "max_longest_edge")
            present_bounds = [key for key in bound_keys if payload.get(key) is not None]
            if present_bounds:
                raise error(
                    "schema_vnext_scale_normalization_invalid",
                    "dual target_longest_edge/target_shortest_edge cannot mix with edge min/max bounds",
                    target=target,
                    fields=present_bounds,
                )
            min_scale = _positive_float(payload.get("min_scale", 0.05), target=f"{target}.min_scale")
            max_scale = _positive_float(payload.get("max_scale", 20.0), target=f"{target}.max_scale")
            if min_scale > max_scale:
                raise error("schema_vnext_scale_normalization_invalid", "min_scale must be <= max_scale", target=target)
            return {
                "target_shortest_edge": _positive_float(target_shortest, target=f"{target}.target_shortest_edge"),
                "target_longest_edge": _positive_float(target_longest, target=f"{target}.target_longest_edge"),
                "min_scale": min_scale,
                "max_scale": max_scale,
            }
        mixed_unsupported = [
            key
            for key in ("target_shortest_edge", "target_longest_edge", "min_longest_edge")
            if payload.get(key) is not None
        ]
        if mixed_unsupported:
            raise error(
                "schema_vnext_scale_normalization_invalid",
                "mixed shortest/longest normalization supports shortest-edge min/max plus longest-edge max",
                target=target,
                fields=mixed_unsupported,
            )
        min_shortest = _optional_non_negative_float(
            payload.get("min_shortest_edge"), target=f"{target}.min_shortest_edge"
        )
        max_shortest = _optional_non_negative_float(
            payload.get("max_shortest_edge"), target=f"{target}.max_shortest_edge"
        )
        max_longest = _optional_non_negative_float(payload.get("max_longest_edge"), target=f"{target}.max_longest_edge")
        if max_shortest is None or max_longest is None:
            raise error(
                "schema_vnext_scale_normalization_invalid",
                "mixed shortest/longest normalization requires both maximum bounds",
                target=target,
            )
        if min_shortest is not None and min_shortest > max_shortest:
            raise error(
                "schema_vnext_scale_normalization_invalid",
                "min_shortest_edge must be <= max_shortest_edge",
                target=target,
            )
        out = {
            "max_shortest_edge": max_shortest,
            "max_longest_edge": max_longest,
        }
        if min_shortest is not None:
            out["min_shortest_edge"] = min_shortest
        if payload.get("min_scale") is not None:
            out["min_scale"] = _positive_float(payload["min_scale"], target=f"{target}.min_scale")
        if payload.get("max_scale") is not None:
            out["max_scale"] = _positive_float(payload["max_scale"], target=f"{target}.max_scale")
        if out.get("min_scale", 0.0) > out.get("max_scale", float("inf")):
            raise error("schema_vnext_scale_normalization_invalid", "min_scale must be <= max_scale", target=target)
        return out
    edge = "shortest" if has_shortest else "longest"
    target_value = payload.get(f"target_{edge}_edge")
    min_edge = _optional_non_negative_float(payload.get(f"min_{edge}_edge"), target=f"{target}.min_{edge}_edge")
    max_edge = _optional_non_negative_float(payload.get(f"max_{edge}_edge"), target=f"{target}.max_{edge}_edge")
    if target_value is None and min_edge is None and max_edge is None:
        raise error(
            "schema_vnext_scale_normalization_invalid",
            "scale_normalization requires a target edge or min/max edge bounds",
            target=target,
        )
    target_edge = (
        _positive_float(target_value, target=f"{target}.target_{edge}_edge") if target_value is not None else None
    )
    min_scale = _positive_float(payload.get("min_scale", 0.05), target=f"{target}.min_scale")
    max_scale = _positive_float(payload.get("max_scale", 20.0), target=f"{target}.max_scale")
    if min_scale > max_scale:
        raise error("schema_vnext_scale_normalization_invalid", "min_scale must be <= max_scale", target=target)
    out = {"min_scale": min_scale, "max_scale": max_scale}
    if target_edge is not None:
        out[f"target_{edge}_edge"] = target_edge
    if min_edge is not None:
        out[f"min_{edge}_edge"] = min_edge
    if max_edge is not None:
        out[f"max_{edge}_edge"] = max_edge
    return out


def _optional_non_negative_float(value: Any, *, target: str) -> float | None:
    if value is None:
        return None
    parsed = _float(value, target=target)
    if parsed < 0.0:
        raise error("schema_vnext_float_invalid", "value must be non-negative", target=target, value=value)
    return parsed


def _positive_float(value: Any, *, target: str) -> float:
    parsed = _float(value, target=target)
    if parsed <= 0.0:
        raise error("schema_vnext_float_invalid", "value must be positive", target=target, value=value)
    return parsed


def _float(value: Any, *, target: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise error("schema_vnext_float_invalid", "value must be a finite number", target=target, value=value) from exc
    if not isfinite(parsed):
        raise error("schema_vnext_float_invalid", "value must be a finite number", target=target, value=value)
    return parsed


def _asset_pool_file_entry(
    raw_asset: Any, *, target: str
) -> tuple[str, tuple[float, float, float] | None, float | None]:
    if isinstance(raw_asset, str):
        return _asset_id_to_name(raw_asset), None, None
    payload = _mapping(raw_asset, target=target)
    asset_name = str(
        payload.get("asset_name") or _asset_id_to_name(payload.get("id") or payload.get("asset_id"))
    ).strip()
    if not asset_name:
        raise error("schema_vnext_asset_pool_asset_id_missing", "asset pool file entry requires id", target=target)
    bbox = _bbox_tuple(payload.get("bbox"), target=f"{target}.bbox") if payload.get("bbox") is not None else None
    scale_factor = _pool_asset_scale_factor(payload.get("scale"), target=f"{target}.scale")
    return asset_name, bbox, scale_factor


def _pool_asset_scale_factor(value: Any, *, target: str) -> float | None:
    """Parse pool-file per-asset scale as a positive isotropic factor.

    Accepts a scalar or a length-3 list. Imported asset pools often emit
    isotropic triples like ``[s, s, s]``; collapse those to one factor. True
    anisotropic triples are rejected until asset_scale_factors supports axes.
    """
    if value is None:
        return None
    if isinstance(value, list):
        axes = _numeric_vec(value, length=3, target=target)
        if any(axis <= 0.0 for axis in axes):
            raise error("schema_vnext_float_invalid", "value must be positive", target=target, value=value)
        if max(axes) - min(axes) > 1e-6:
            raise error(
                "schema_vnext_pool_scale_anisotropic",
                "asset pool scale list must be isotropic [s,s,s] (or use a scalar)",
                target=target,
                value=value,
            )
        return float(axes[0])
    return _positive_float(value, target=target)


def compose_scale_factor(
    dims: list[float] | tuple[float, ...],
    json_scale_factor: float | None,
    yaml_scale_config: Mapping[str, float] | None,
) -> tuple[float, float, float] | None:
    """Compose the per-asset (JSON) multiplier and candidate (YAML) normalization.

    The concrete per-asset ``json_scale_factor`` is applied first; the candidate
    ``yaml_scale_config`` is then evaluated against the already-scaled size so it
    only clamps out-of-range results rather than overriding the per-asset intent.
    Returns a per-axis scale triple, or ``None`` when the composed factor is identity.
    Dual ``target_longest_edge`` + ``target_shortest_edge`` yields anisotropic axes.
    """
    if len(dims) != 3:
        return None
    base = (float(dims[0]), float(dims[1]), float(dims[2]))
    if any(value <= 0.0 for value in base):
        return None
    s_json = float(json_scale_factor) if json_scale_factor else 1.0
    scaled = (base[0] * s_json, base[1] * s_json, base[2] * s_json)
    yaml_axes = scale_axes_from_bbox_edges(scaled, yaml_scale_config) if yaml_scale_config else None
    if yaml_axes is None:
        total = (s_json, s_json, s_json)
    else:
        total = (s_json * yaml_axes[0], s_json * yaml_axes[1], s_json * yaml_axes[2])
    if all(abs(axis - 1.0) <= 1e-9 for axis in total):
        return None
    return total


def _asset_id_to_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    source, sep, asset_id = text.partition(":")
    if sep and source and asset_id:
        return f"{source}__{asset_id}"
    return text.replace(":", "__")


def _bbox_tuple(value: Any, *, target: str) -> tuple[float, float, float]:
    raw = _list(value, target=target)
    if len(raw) != 3:
        raise error("schema_vnext_asset_pool_bbox_invalid", "bbox must be [w, d, h]", target=target)
    bbox = tuple(_positive_float(item, target=f"{target}[{index}]") for index, item in enumerate(raw))
    return (bbox[0], bbox[1], bbox[2])


def _bbox_in_range(longest: float, bbox_filter: Mapping[str, float | None]) -> bool:
    min_edge = bbox_filter.get("min")
    max_edge = bbox_filter.get("max")
    if min_edge is not None and longest < float(min_edge):
        return False
    if max_edge is not None and longest > float(max_edge):
        return False
    return True


def edge_normalization_bounds(
    scale_config: Mapping[str, float],
) -> tuple[str, float | None, float | None, float | None]:
    """Resolve which bbox edge drives normalization and its target/bounds.

    Returns ``(kind, target, min_edge, max_edge)`` where *kind* is ``"shortest"``
    or ``"longest"``. Shortest-edge keys take precedence when present.
    """
    if any(key in scale_config for key in ("target_shortest_edge", "min_shortest_edge", "max_shortest_edge")):
        return (
            "shortest",
            scale_config.get("target_shortest_edge"),
            scale_config.get("min_shortest_edge"),
            scale_config.get("max_shortest_edge"),
        )
    return (
        "longest",
        scale_config.get("target_longest_edge"),
        scale_config.get("min_longest_edge"),
        scale_config.get("max_longest_edge"),
    )


def scale_from_edge(reference: float, scale_config: Mapping[str, float]) -> float | None:
    """Compute the isotropic scale that maps *reference* edge into the config."""
    _kind, target, min_edge, max_edge = edge_normalization_bounds(scale_config)
    if reference <= 0.0:
        return None
    if target is not None:
        scale = float(target) / float(reference)
    elif min_edge is not None and reference < float(min_edge):
        scale = float(min_edge) / float(reference)
    elif max_edge is not None and reference > float(max_edge):
        scale = float(max_edge) / float(reference)
    else:
        return None
    return max(float(scale_config.get("min_scale", 0.05)), min(float(scale_config.get("max_scale", 20.0)), scale))


def scale_from_bbox_edges(dims: list[float] | tuple[float, ...], scale_config: Mapping[str, float]) -> float | None:
    """Compute one isotropic scale from the shortest and longest bbox edges."""
    values = [float(value) for value in dims if float(value) > 0.0]
    if not values:
        return None
    if "max_shortest_edge" in scale_config and "max_longest_edge" in scale_config:
        shortest = min(values)
        lower_scale = (
            float(scale_config["min_shortest_edge"]) / shortest if "min_shortest_edge" in scale_config else 0.0
        )
        upper_scale = min(
            float(scale_config["max_shortest_edge"]) / min(values),
            float(scale_config["max_longest_edge"]) / max(values),
        )
        # Clamp the identity scale into the feasible interval. If the shortest
        # minimum and longest maximum conflict, the upper bound wins.
        scale = min(max(1.0, lower_scale), upper_scale)
        if scale == 1.0:
            return None
        if "min_scale" in scale_config:
            scale = max(float(scale_config["min_scale"]), scale)
        if "max_scale" in scale_config:
            scale = min(float(scale_config["max_scale"]), scale)
        return scale
    kind, _target, _min_edge, _max_edge = edge_normalization_bounds(scale_config)
    reference = min(values) if kind == "shortest" else max(values)
    return scale_from_edge(reference, scale_config)


def scale_axes_from_bbox_edges(
    dims: list[float] | tuple[float, ...], scale_config: Mapping[str, float]
) -> tuple[float, float, float] | None:
    """Compute per-axis scales from bbox edges.

    Dual ``target_longest_edge`` + ``target_shortest_edge`` maps the longest axis
    to the long target and every other axis to the short target (anisotropic).
    All other configs remain isotropic ``(s, s, s)``.
    """
    if (
        scale_config.get("target_longest_edge") is not None
        and scale_config.get("target_shortest_edge") is not None
        and len(dims) == 3
    ):
        values = (float(dims[0]), float(dims[1]), float(dims[2]))
        if any(value <= 0.0 for value in values):
            return None
        longest_idx = max(range(3), key=lambda index: values[index])
        target_longest = float(scale_config["target_longest_edge"])
        target_shortest = float(scale_config["target_shortest_edge"])
        min_scale = float(scale_config.get("min_scale", 0.05))
        max_scale = float(scale_config.get("max_scale", 20.0))
        axes: list[float] = []
        for index, value in enumerate(values):
            target = target_longest if index == longest_idx else target_shortest
            scale = max(min_scale, min(max_scale, target / value))
            axes.append(scale)
        if all(abs(axis - 1.0) <= 1e-9 for axis in axes):
            return None
        return (axes[0], axes[1], axes[2])
    scale = scale_from_bbox_edges(dims, scale_config)
    if scale is None:
        return None
    return (scale, scale, scale)


@dataclass(frozen=True)
class MaterialVariantSpec:
    variant_id: str
    path: str
    candidates: tuple[str, ...]
    channel: str = "base_color"
    selector: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScaleSpec:
    scale_range: tuple[float, float] | None = None
    axis_scale: tuple[float, float, float] | None = None

    @classmethod
    def from_value(cls, value: Any, *, target: str) -> ScaleSpec | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return cls(scale_range=_positive_range(value, target=target))
        if isinstance(value, Mapping):
            _reject_unsupported_fields(
                value,
                allowed={"range"},
                code="schema_vnext_scale_field_unsupported",
                target=target,
            )
            if "range" not in value:
                raise error("schema_vnext_scale_invalid", "scale mapping requires range", target=target)
            return cls(scale_range=_positive_range(value["range"], target=f"{target}.range"))
        if isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            return cls(scale_range=_positive_range(value, target=target))
        if isinstance(value, list) and len(value) == 3 and all(isinstance(item, (int, float)) for item in value):
            parsed = tuple(float(item) for item in value)
            if not all(isfinite(item) and item > 0.0 for item in parsed):
                raise error("schema_vnext_scale_invalid", "scale must be positive and finite", target=target)
            if max(parsed) - min(parsed) > 1e-9:
                # Non-uniform per-axis scale: keep the explicit vec3 rather than
                # collapsing to a single factor.
                return cls(axis_scale=parsed)
            return cls(scale_range=(parsed[0], parsed[0]))
        raise error("schema_vnext_scale_invalid", "expected positive number, [min, max], or vec3", target=target)


@dataclass(frozen=True)
class MassSpec:
    """Authoring-time mass in kg: fixed value or positive [min, max] range."""

    mass_range: tuple[float, float]

    @classmethod
    def from_value(cls, value: Any, *, target: str) -> MassSpec | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return cls(mass_range=_positive_range(value, target=target))
        if isinstance(value, Mapping):
            _reject_unsupported_fields(
                value,
                allowed={"range"},
                code="schema_vnext_spawn_mass_field_unsupported",
                target=target,
            )
            if "range" not in value:
                raise error("schema_vnext_spawn_mass_invalid", "spawn_mass mapping requires range", target=target)
            return cls(mass_range=_positive_range(value["range"], target=f"{target}.range"))
        if isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            return cls(mass_range=_positive_range(value, target=target))
        raise error(
            "schema_vnext_spawn_mass_invalid",
            "expected positive number, [min, max], or {range: [min, max]}",
            target=target,
        )


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    asset: AssetSelectorSpec
    semantic_tags: tuple[str, ...] = ()
    scale: ScaleSpec | None = None
    #: Optional base orientation (XYZW) composed under layout yaw / slot placement.
    #: Useful for slot groups where node placement is omitted and each slot only
    #: supplies surface ranges / yaws.
    orientation: tuple[float, float, float, float] | None = None
    spawn_mass: MassSpec | None = None
    #: Absolute rigid-body material override applied at spawn (Isaac RigidBodyMaterialCfg).
    spawn_static_friction: float | None = None
    spawn_dynamic_friction: float | None = None
    #: Spawn the rigid body with physics kinematicEnabled so it ignores external forces.
    kinematic: bool = False
    #: Optional post-spawn mesh collision approximation rewrite for ObjectsComponent
    #: (e.g. ``boundingCube`` / ``convexHull``). Needed when particle cloth must
    #: collide with an authored mesh that is not GPU-compatible.
    collision_approximation: str | None = None
    #: Clear authored child rotateXYZ Z under visual/collision before PhysX cook;
    #: bake that yaw into the object root pose in the scene YAML instead.
    strip_child_z_rotation: bool = False
    instances: Mapping[str, Any] | None = None
    anchors: Mapping[str, Any] = field(default_factory=dict)
    material_variants: Mapping[str, MaterialVariantSpec] = field(default_factory=dict)

    @property
    def instance_ids(self) -> tuple[str, ...]:
        return _instance_ids(self.instances)

    @property
    def materialized_ids(self) -> tuple[str, ...]:
        instances = self.instance_ids
        return instances or (self.object_id,)

    @classmethod
    def from_dict(cls, data: Any, *, target: str) -> ObjectSpec:
        payload = _mapping(data, target=target)
        object_id = _non_empty_str(payload.get("id"), target=f"{target}.id")
        if "name" in payload or "tags" in payload or "asset_pool" in payload:
            raise error("schema_vnext_retired_object_field", "retired object fields are not allowed", target=target)
        _reject_unsupported_fields(
            payload,
            allowed={
                "id",
                "asset",
                "semantic_tags",
                "scale",
                "orientation",
                "spawn_mass",
                "spawn_static_friction",
                "spawn_dynamic_friction",
                "kinematic",
                "collision_approximation",
                "strip_child_z_rotation",
                "instances",
                "anchors",
                "material_variants",
            },
            code="schema_vnext_object_field_unsupported",
            target=target,
        )
        orientation = None
        if "orientation" in payload:
            orientation = _numeric_vec(payload["orientation"], length=4, target=f"{target}.orientation")
        kinematic = payload.get("kinematic", False)
        if not isinstance(kinematic, bool):
            raise error("schema_vnext_kinematic_invalid", "kinematic must be a bool", target=f"{target}.kinematic")
        collision_approximation = payload.get("collision_approximation")
        if collision_approximation is not None and (
            not isinstance(collision_approximation, str) or not collision_approximation.strip()
        ):
            raise error(
                "schema_vnext_collision_approximation_invalid",
                "collision_approximation must be a non-empty string when set",
                target=f"{target}.collision_approximation",
            )
        if isinstance(collision_approximation, str):
            collision_approximation = collision_approximation.strip()
        strip_child_z_rotation = payload.get("strip_child_z_rotation", False)
        if not isinstance(strip_child_z_rotation, bool):
            raise error(
                "schema_vnext_strip_child_z_rotation_invalid",
                "strip_child_z_rotation must be a bool",
                target=f"{target}.strip_child_z_rotation",
            )
        instances = _parse_instances(payload.get("instances"), target=f"{target}.instances")
        anchors = _parse_anchors(payload.get("anchors"), target=f"{target}.anchors")
        material_variants = _parse_material_variants(
            payload.get("material_variants"), target=f"{target}.material_variants"
        )
        return cls(
            object_id=object_id,
            asset=AssetSelectorSpec.from_dict(payload.get("asset"), target=f"{target}.asset"),
            semantic_tags=_string_tuple(payload.get("semantic_tags"), target=f"{target}.semantic_tags"),
            scale=ScaleSpec.from_value(payload.get("scale"), target=f"{target}.scale"),
            orientation=orientation,
            spawn_mass=MassSpec.from_value(payload.get("spawn_mass"), target=f"{target}.spawn_mass"),
            spawn_static_friction=_optional_non_negative_float(
                payload.get("spawn_static_friction"), target=f"{target}.spawn_static_friction"
            ),
            spawn_dynamic_friction=_optional_non_negative_float(
                payload.get("spawn_dynamic_friction"), target=f"{target}.spawn_dynamic_friction"
            ),
            kinematic=kinematic,
            collision_approximation=collision_approximation,
            strip_child_z_rotation=strip_child_z_rotation,
            instances=instances,
            anchors=anchors,
            material_variants=material_variants,
        )


def _parse_material_variants(value: Any, *, target: str) -> Mapping[str, MaterialVariantSpec]:
    if value is None:
        return {}
    variants: list[MaterialVariantSpec] = []
    for index, raw_variant in enumerate(_list(value, target=target)):
        item_target = f"{target}[{index}]"
        payload = _mapping(raw_variant, target=item_target)
        _reject_unsupported_fields(
            payload,
            allowed={"id", "path", "candidates", "channel", "selector"},
            code="schema_vnext_material_variant_field_unsupported",
            target=item_target,
        )
        variant_id = _non_empty_str(payload.get("id"), target=f"{item_target}.id")
        path = _non_empty_str(payload.get("path"), target=f"{item_target}.path")
        candidates = _string_tuple(payload.get("candidates"), target=f"{item_target}.candidates")
        if not candidates:
            raise error(
                "schema_vnext_material_variant_candidates_empty",
                "material variant requires candidates",
                target=item_target,
            )
        channel = str(payload.get("channel") or "base_color").strip()
        if not channel:
            raise error(
                "schema_vnext_material_variant_channel_empty",
                "material variant channel cannot be empty",
                target=item_target,
            )
        selector = payload.get("selector")
        if selector is not None and not isinstance(selector, Mapping):
            raise error(
                "schema_vnext_material_variant_selector_invalid",
                "material variant selector must be a mapping",
                target=item_target,
            )
        variants.append(
            MaterialVariantSpec(
                variant_id=variant_id,
                path=path,
                candidates=candidates,
                channel=channel,
                selector=dict(selector or {}),
            )
        )
    _unique([variant.variant_id for variant in variants], target=target)
    return {variant.variant_id: variant for variant in variants}


def _parse_instances(value: Any, *, target: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    payload = _mapping(value, target=target)
    mode = _non_empty_str(payload.get("type"), target=f"{target}.type")
    if mode == "auto":
        _reject_unsupported_fields(
            payload,
            allowed={"type", "count", "template", "start_index"},
            code="schema_vnext_instances_field_unsupported",
            target=target,
        )
        forbidden = {"ids"}.intersection(payload)
        if forbidden:
            raise error("schema_vnext_instances_mode_conflict", "auto instances cannot define ids", target=target)
        count = payload.get("count")
        if not isinstance(count, int) or count <= 0:
            raise error("schema_vnext_instances_count_invalid", "auto instances require positive count", target=target)
        template = _non_empty_str(payload.get("template"), target=f"{target}.template")
        if "{i}" not in template:
            raise error("schema_vnext_instances_template_invalid", "auto template must contain {i}", target=target)
        if "zero_pad" in payload:
            raise error(
                "schema_vnext_instances_zero_pad_unsupported", "zero_pad is not part of schema vnext", target=target
            )
        start_index = payload.get("start_index", 0)
        if not isinstance(start_index, int):
            raise error("schema_vnext_instances_start_invalid", "start_index must be an integer", target=target)
        return dict(payload)
    if mode == "explicit":
        _reject_unsupported_fields(
            payload,
            allowed={"type", "ids"},
            code="schema_vnext_instances_field_unsupported",
            target=target,
        )
        forbidden = {"count", "template", "start_index", "zero_pad"}.intersection(payload)
        if forbidden:
            raise error(
                "schema_vnext_instances_mode_conflict", "explicit instances cannot define auto fields", target=target
            )
        ids = _string_tuple(payload.get("ids"), target=f"{target}.ids")
        if not ids:
            raise error("schema_vnext_instances_ids_empty", "explicit instances require ids", target=target)
        return dict(payload)
    raise error("schema_vnext_instances_type_unsupported", "unsupported instances type", target=target, type=mode)


def _parse_anchors(value: Any, *, target: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    payload = _mapping(value, target=target)
    unsupported = set(payload) - {"local_surfaces", "local_volumes"}
    if unsupported:
        raise error(
            "schema_vnext_anchors_field_unsupported",
            "unsupported anchors field",
            target=target,
            fields=sorted(unsupported),
        )
    for key in ("local_surfaces", "local_volumes"):
        if key in payload and not isinstance(payload[key], Mapping):
            raise error("schema_vnext_anchors_invalid", f"{key} must be a mapping", target=f"{target}.{key}")
    return dict(payload)


@dataclass(frozen=True)
class SurfaceMaterialPoolSpec:
    surface_id: str
    manifest: str
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackgroundSpec:
    background_id: str | None
    asset: AssetSelectorSpec
    pose_position: tuple[float, float, float]
    pose_orientation: tuple[float, float, float, float]
    required_surfaces: tuple[str, ...] = ()
    surface_catalog: str | None = None
    surface_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    surface_material_pool: Mapping[str, SurfaceMaterialPoolSpec] = field(default_factory=dict)
    pose_by_asset: Mapping[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = field(
        default_factory=dict
    )
    #: ManaEnv ``AssetBasedComponent`` preload module for task-local
    #: ``@register_asset`` assets. Owned here so every parse path accepts it.
    asset_module: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> BackgroundSpec:
        payload = _mapping(data, target="background")
        _reject_unsupported_fields(
            payload,
            allowed={
                "id",
                "asset",
                "pose",
                "pose_by_asset",
                "required_surfaces",
                "surface_catalog",
                "surface_overrides",
                "surface_material_pool",
                "asset_module",
            },
            code="schema_vnext_background_field_unsupported",
            target="background",
        )
        pose_position, pose_orientation = _parse_background_pose(payload.get("pose"))
        required_surfaces = _string_tuple(payload.get("required_surfaces"), target="background.required_surfaces")
        if payload.get("surface_catalog") is not None:
            surface_catalog_raw = _non_empty_str(payload.get("surface_catalog"), target="background.surface_catalog")
            # Resolve local catalogs through the public asset boundary; remote
            # schemes are rejected by ``resolve_asset_path``.
            surface_catalog = str(_resolve_asset_pool_path(surface_catalog_raw))
        else:
            surface_catalog = None
        surface_overrides = _parse_background_surface_overrides(payload.get("surface_overrides"))
        surface_material_pool = _parse_surface_material_pool(
            payload.get("surface_material_pool"),
            known_surfaces=set(required_surfaces).union(surface_overrides),
        )
        pose_by_asset = _parse_background_pose_by_asset(payload.get("pose_by_asset"))
        return cls(
            background_id=str(payload["id"]).strip() if payload.get("id") else None,
            asset=AssetSelectorSpec.from_dict(payload.get("asset"), target="background.asset"),
            pose_position=pose_position,
            pose_orientation=pose_orientation,
            required_surfaces=required_surfaces,
            surface_catalog=surface_catalog,
            surface_overrides=surface_overrides,
            surface_material_pool=surface_material_pool,
            pose_by_asset=pose_by_asset,
            asset_module=(
                _non_empty_str(payload.get("asset_module"), target="background.asset_module")
                if payload.get("asset_module") is not None
                else None
            ),
        )


def _parse_background_pose(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    if not isinstance(value, Mapping):
        raise error("schema_vnext_background_pose_required", "background.pose is required", target="background.pose")
    _reject_unsupported_fields(
        value,
        allowed={"position", "orientation"},
        code="schema_vnext_background_pose_field_unsupported",
        target="background.pose",
    )
    if "position" not in value or "orientation" not in value:
        raise error(
            "schema_vnext_background_pose_required",
            "background.pose requires position and orientation",
            target="background.pose",
        )
    return (
        _numeric_vec(value["position"], length=3, target="background.pose.position"),
        _numeric_vec(value["orientation"], length=4, target="background.pose.orientation"),
    )


def _parse_background_pose_by_asset(
    value: Any,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    """Optional per-asset background poses for ``background.asset.type=pool``.

    Accepts either an inline mapping ``{asset_id: {position, orientation}}`` or
    ``{from_file: path.json}`` where the JSON contains a ``poses`` mapping.
    """
    if value is None:
        return {}
    payload = _mapping(value, target="background.pose_by_asset")
    if "from_file" in payload:
        _reject_unsupported_fields(
            payload,
            allowed={"from_file"},
            code="schema_vnext_background_pose_by_asset_field_unsupported",
            target="background.pose_by_asset",
        )
        path = _resolve_asset_pool_path(
            _non_empty_str(payload.get("from_file"), target="background.pose_by_asset.from_file")
        )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise error(
                "schema_vnext_background_pose_by_asset_missing",
                "background pose_by_asset file does not exist",
                target="background.pose_by_asset.from_file",
                path=str(path),
            ) from exc
        except json.JSONDecodeError as exc:
            raise error(
                "schema_vnext_background_pose_by_asset_invalid",
                "background pose_by_asset file is not valid JSON",
                target="background.pose_by_asset.from_file",
                path=str(path),
            ) from exc
        if not isinstance(raw, Mapping):
            raise error(
                "schema_vnext_background_pose_by_asset_invalid",
                "background pose_by_asset file must contain a JSON object",
                target="background.pose_by_asset.from_file",
            )
        poses_payload = raw.get("poses", raw)
        if not isinstance(poses_payload, Mapping):
            raise error(
                "schema_vnext_background_pose_by_asset_invalid",
                "background pose_by_asset file requires a poses mapping",
                target="background.pose_by_asset.from_file",
            )
        payload = poses_payload
    out: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
    for asset_id, raw_pose in payload.items():
        key = _non_empty_str(asset_id, target="background.pose_by_asset")
        pose = _mapping(raw_pose, target=f"background.pose_by_asset.{key}")
        out[key] = _parse_background_pose(pose)
    return out


def _parse_background_surface_overrides(value: Any) -> dict[str, Mapping[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        raw_items = []
        for surface_id, raw_surface in value.items():
            data = dict(_mapping(raw_surface, target=f"background.surface_overrides.{surface_id}"))
            data.setdefault("id", surface_id)
            raw_items.append(data)
    else:
        raw_items = _list(value, target="background.surface_overrides")
    overrides: dict[str, Mapping[str, Any]] = {}
    for index, raw_surface in enumerate(raw_items):
        data = dict(_mapping(raw_surface, target=f"background.surface_overrides[{index}]"))
        surface_id = _non_empty_str(data.get("id"), target=f"background.surface_overrides[{index}].id")
        if surface_id in overrides:
            raise error(
                "schema_vnext_duplicate_id",
                "ids must be unique",
                target="background.surface_overrides",
                ids=[surface_id],
            )
        surface_type = _non_empty_str(data.get("type"), target=f"background.surface_overrides.{surface_id}.type")
        if surface_type != "manual":
            raise error(
                "schema_vnext_background_surface_override_type_unsupported",
                "background surface override must be a manual surface",
                target=f"background.surface_overrides.{surface_id}",
                type=surface_type,
            )
        shape = _non_empty_str(data.get("shape"), target=f"background.surface_overrides.{surface_id}.shape")
        if shape == "rectangle":
            _reject_unsupported_fields(
                data,
                allowed={"id", "type", "shape", "frame", "center", "x_axis", "y_axis", "normal", "bounds"},
                code="schema_vnext_background_surface_override_field_unsupported",
                target=f"background.surface_overrides.{surface_id}",
            )
            bounds = _mapping(data.get("bounds"), target=f"background.surface_overrides.{surface_id}.bounds")
            _scalar_or_pair(bounds.get("x"), target=f"background.surface_overrides.{surface_id}.bounds.x")
            _scalar_or_pair(bounds.get("y"), target=f"background.surface_overrides.{surface_id}.bounds.y")
            for key in ("frame", "center", "x_axis", "y_axis", "normal"):
                if key not in data:
                    raise error(
                        "schema_vnext_background_surface_override_field_missing",
                        "manual rectangle surface override missing field",
                        target=f"background.surface_overrides.{surface_id}.{key}",
                    )
        elif shape == "polygon":
            _reject_unsupported_fields(
                data,
                allowed={
                    "id",
                    "type",
                    "shape",
                    "frame",
                    "center",
                    "x_axis",
                    "y_axis",
                    "normal",
                    "outer",
                    "holes",
                    "margin",
                },
                code="schema_vnext_background_surface_override_field_unsupported",
                target=f"background.surface_overrides.{surface_id}",
            )
            if "outer" not in data:
                raise error(
                    "schema_vnext_background_surface_override_polygon_outer_missing",
                    "manual polygon surface override requires outer",
                    target=f"background.surface_overrides.{surface_id}",
                )
        else:
            raise error(
                "schema_vnext_surface_shape_unsupported",
                "unsupported manual surface shape",
                target=f"background.surface_overrides.{surface_id}",
                shape=shape,
            )
        overrides[surface_id] = data
    return overrides


def _parse_surface_material_pool(
    value: Any,
    *,
    known_surfaces: set[str],
) -> dict[str, SurfaceMaterialPoolSpec]:
    if value is None:
        return {}
    payload = _mapping(value, target="background.surface_material_pool")
    pools: dict[str, SurfaceMaterialPoolSpec] = {}
    for raw_surface_id, raw_pool in payload.items():
        surface_id = _non_empty_str(raw_surface_id, target="background.surface_material_pool")
        if known_surfaces and surface_id not in known_surfaces:
            raise error(
                "schema_vnext_surface_material_unknown_surface",
                "surface material pool references an unknown background surface",
                target=f"background.surface_material_pool.{surface_id}",
                surface=surface_id,
            )
        pool = _mapping(raw_pool, target=f"background.surface_material_pool.{surface_id}")
        _reject_unsupported_fields(
            pool,
            allowed={"manifest", "candidates"},
            code="schema_vnext_surface_material_pool_field_unsupported",
            target=f"background.surface_material_pool.{surface_id}",
        )
        pools[surface_id] = SurfaceMaterialPoolSpec(
            surface_id=surface_id,
            manifest=_non_empty_str(
                pool.get("manifest"), target=f"background.surface_material_pool.{surface_id}.manifest"
            ),
            candidates=_string_tuple(
                pool.get("candidates"),
                target=f"background.surface_material_pool.{surface_id}.candidates",
            ),
        )
    return pools


@dataclass(frozen=True)
class SurfaceAliasSpec:
    surface_id: str
    alias_type: str
    ref: str | None = None


@dataclass(frozen=True)
class NodePlacementSpec:
    placement_type: str
    surface: str | None = None
    volume: str | None = None
    base_orientation: tuple[float, float, float, float] | None = None
    range_candidate_ids: tuple[str, ...] = ()
    yaw_candidate_ids: tuple[str, ...] = ()
    pose_candidate_ids: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    object_id: str
    scale: ScaleSpec | None = None
    placement: NodePlacementSpec | None = None


@dataclass(frozen=True)
class SlotAssignmentSpec:
    assignment_id: str
    bindings: Mapping[str, str]
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    group_type: str
    nodes: tuple[str, ...]
    slot_ids: tuple[str, ...] = ()
    assignment_mode: str | None = None
    assignments: tuple[SlotAssignmentSpec, ...] = ()
    generated_assignment_ids: tuple[str, ...] = ()
    default_assignment: str | None = None
    base_orientation: tuple[float, float, float, float] | None = None
    range_candidate_ids: tuple[str, ...] = ()
    yaw_candidate_ids: tuple[str, ...] = ()
    pose_candidate_ids: tuple[str, ...] = ()

    @property
    def assignment_ids(self) -> set[str]:
        return {item.assignment_id for item in self.assignments}.union(self.generated_assignment_ids)


@dataclass(frozen=True)
class LayoutVNextSpec:
    seed: int | None
    surfaces: Mapping[str, SurfaceAliasSpec]
    volumes: Mapping[str, Mapping[str, Any]]
    nodes: Mapping[str, NodeSpec]
    groups: Mapping[str, GroupSpec]
    relations: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, data: Any, *, objects: Mapping[str, ObjectSpec]) -> LayoutVNextSpec:
        payload = _mapping(data, target="layout")
        forbidden = {"enabled", "surface_catalog_refs", "collision", "variants"}.intersection(payload)
        if forbidden:
            raise error(
                "schema_vnext_retired_layout_field",
                "retired layout fields are not allowed",
                target="layout",
                fields=sorted(forbidden),
            )
        _reject_unsupported_fields(
            payload,
            allowed={"seed", "surfaces", "volumes", "nodes", "groups", "relations"},
            code="schema_vnext_layout_field_unsupported",
            target="layout",
        )
        seed = payload.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise error("schema_vnext_seed_invalid", "layout.seed must be an integer", target="layout.seed")
        surfaces = _parse_surfaces(payload.get("surfaces"), target="layout.surfaces", objects=objects)
        volumes = _parse_volumes(payload.get("volumes", {}), target="layout.volumes", objects=objects)
        nodes = _parse_nodes(
            payload.get("nodes"),
            object_refs=_object_reference_ids(objects),
            surfaces=surfaces,
            volumes=volumes,
        )
        groups = _parse_groups(payload.get("groups", {}), nodes=nodes, surfaces=surfaces, volumes=volumes)
        _validate_placement_owners(nodes, groups)
        relations = _parse_relations(payload.get("relations", []), nodes=nodes)
        return cls(seed=seed, surfaces=surfaces, volumes=volumes, nodes=nodes, groups=groups, relations=relations)


def _parse_surfaces(value: Any, *, target: str, objects: Mapping[str, ObjectSpec]) -> dict[str, SurfaceAliasSpec]:
    payload = _mapping(value, target=target)
    surfaces: dict[str, SurfaceAliasSpec] = {}
    for surface_id, raw_surface in payload.items():
        sid = _non_empty_str(surface_id, target=target)
        data = _mapping(raw_surface, target=f"{target}.{sid}")
        surface_type = _non_empty_str(data.get("type"), target=f"{target}.{sid}.type")
        if surface_type == "ref":
            if "ref" not in data:
                raise error("schema_vnext_surface_ref_missing", "ref surface requires ref", target=f"{target}.{sid}")
            illegal = {"frame", "shape", "center", "x_axis", "y_axis", "normal", "bounds", "size"}.intersection(data)
            if illegal:
                raise error(
                    "schema_vnext_surface_ref_geometry_conflict",
                    "ref surface cannot define geometry",
                    target=f"{target}.{sid}",
                    fields=sorted(illegal),
                )
            _reject_unsupported_fields(
                data,
                allowed={"type", "ref"},
                code="schema_vnext_surface_field_unsupported",
                target=f"{target}.{sid}",
            )
            ref = _non_empty_str(data["ref"], target=f"{target}.{sid}.ref")
            _validate_surface_ref(ref, objects=objects, target=f"{target}.{sid}.ref")
            surfaces[sid] = SurfaceAliasSpec(surface_id=sid, alias_type=surface_type, ref=ref)
            continue
        if surface_type == "manual":
            shape = _non_empty_str(data.get("shape"), target=f"{target}.{sid}.shape")
            if shape == "rectangle":
                _reject_unsupported_fields(
                    data,
                    allowed={"type", "shape", "frame", "center", "x_axis", "y_axis", "normal", "bounds"},
                    code="schema_vnext_surface_field_unsupported",
                    target=f"{target}.{sid}",
                )
                bounds = _mapping(data.get("bounds"), target=f"{target}.{sid}.bounds")
                _scalar_or_pair(bounds.get("x"), target=f"{target}.{sid}.bounds.x")
                _scalar_or_pair(bounds.get("y"), target=f"{target}.{sid}.bounds.y")
                for key in ("frame", "center", "x_axis", "y_axis", "normal"):
                    if key not in data:
                        raise error(
                            "schema_vnext_surface_manual_field_missing",
                            "manual rectangle surface missing field",
                            target=f"{target}.{sid}.{key}",
                        )
            elif shape == "polygon":
                _reject_unsupported_fields(
                    data,
                    allowed={
                        "type",
                        "shape",
                        "frame",
                        "center",
                        "x_axis",
                        "y_axis",
                        "normal",
                        "outer",
                        "holes",
                        "margin",
                    },
                    code="schema_vnext_surface_field_unsupported",
                    target=f"{target}.{sid}",
                )
                if "outer" not in data:
                    raise error(
                        "schema_vnext_surface_polygon_outer_missing",
                        "manual polygon surface requires outer",
                        target=f"{target}.{sid}",
                    )
            else:
                raise error(
                    "schema_vnext_surface_shape_unsupported",
                    "unsupported manual surface shape",
                    target=f"{target}.{sid}",
                    shape=shape,
                )
            surfaces[sid] = SurfaceAliasSpec(surface_id=sid, alias_type=surface_type)
            continue
        raise error(
            "schema_vnext_surface_type_unsupported",
            "unsupported surface alias type",
            target=f"{target}.{sid}",
            type=surface_type,
        )
    return surfaces


def _parse_volumes(value: Any, *, target: str, objects: Mapping[str, ObjectSpec]) -> dict[str, Mapping[str, Any]]:
    payload = _mapping(value or {}, target=target)
    volumes: dict[str, Mapping[str, Any]] = {}
    for volume_id, raw_volume in payload.items():
        vid = _non_empty_str(volume_id, target=target)
        data = _mapping(raw_volume, target=f"{target}.{vid}")
        volume_type = _non_empty_str(data.get("type"), target=f"{target}.{vid}.type")
        if volume_type == "ref":
            if "ref" not in data:
                raise error("schema_vnext_volume_ref_missing", "ref volume requires ref", target=f"{target}.{vid}")
            illegal = {
                "frame",
                "shape",
                "center",
                "bounds",
                "size",
                "x",
                "y",
                "z",
                "radius",
                "inner_diameter",
                "height",
            }.intersection(data)
            if illegal:
                raise error(
                    "schema_vnext_volume_ref_geometry_conflict",
                    "ref volume cannot define geometry",
                    target=f"{target}.{vid}",
                    fields=sorted(illegal),
                )
            _reject_unsupported_fields(
                data,
                allowed={"type", "ref"},
                code="schema_vnext_volume_field_unsupported",
                target=f"{target}.{vid}",
            )
            _validate_volume_ref(
                _non_empty_str(data["ref"], target=f"{target}.{vid}.ref"), objects=objects, target=f"{target}.{vid}.ref"
            )
        elif volume_type == "manual":
            _reject_unsupported_fields(
                data,
                allowed={
                    "type",
                    "shape",
                    "frame",
                    "center",
                    "bounds",
                    "size",
                    "x",
                    "y",
                    "z",
                    "radius",
                    "inner_diameter",
                    "height",
                    "margin",
                },
                code="schema_vnext_volume_field_unsupported",
                target=f"{target}.{vid}",
            )
            shape = _non_empty_str(data.get("shape"), target=f"{target}.{vid}.shape")
            if shape not in {"box", "cylinder"}:
                raise error(
                    "schema_vnext_volume_shape_unsupported",
                    "unsupported manual volume shape",
                    target=f"{target}.{vid}",
                    shape=shape,
                )
            if shape == "box" and "bounds" not in data and "size" not in data:
                raise error(
                    "schema_vnext_volume_box_bounds_missing",
                    "manual box volume requires bounds or size",
                    target=f"{target}.{vid}",
                )
            if shape == "cylinder":
                has_radius = "radius" in data or "inner_diameter" in data
                has_height = "z" in data or "height" in data
                if not has_radius or not has_height:
                    raise error(
                        "schema_vnext_volume_cylinder_fields_missing",
                        "manual cylinder volume requires radius/inner_diameter and z/height",
                        target=f"{target}.{vid}",
                    )
        else:
            raise error(
                "schema_vnext_volume_type_unsupported",
                "unsupported volume alias type",
                target=f"{target}.{vid}",
                type=volume_type,
            )
        volumes[vid] = dict(data)
    return volumes


def _validate_surface_ref(ref: str, *, objects: Mapping[str, ObjectSpec], target: str) -> None:
    if "." not in ref:
        raise error(
            "schema_vnext_surface_ref_invalid",
            "surface ref must use background.surface_id or object_id.surface_id",
            target=target,
            ref=ref,
        )
    owner, surface_id = ref.split(".", 1)
    if owner == "background":
        return
    raw_object = _object_for_anchor_ref_owner(owner, objects=objects, target=target, ref=ref, kind="surface")
    _validate_object_anchor_name(
        raw_object,
        anchor_section="local_surfaces",
        anchor_name=surface_id,
        target=target,
        ref=ref,
        kind="surface",
    )


def _validate_volume_ref(ref: str, *, objects: Mapping[str, ObjectSpec], target: str) -> None:
    if "." not in ref:
        raise error(
            "schema_vnext_volume_ref_invalid", "volume ref must use object_id.volume_id", target=target, ref=ref
        )
    owner, volume_id = ref.split(".", 1)
    raw_object = _object_for_anchor_ref_owner(owner, objects=objects, target=target, ref=ref, kind="volume")
    _validate_object_anchor_name(
        raw_object,
        anchor_section="local_volumes",
        anchor_name=volume_id,
        target=target,
        ref=ref,
        kind="volume",
    )


def _object_for_anchor_ref_owner(
    owner: str,
    *,
    objects: Mapping[str, ObjectSpec],
    target: str,
    ref: str,
    kind: str,
) -> ObjectSpec:
    raw_object = objects.get(owner)
    if raw_object is not None:
        if raw_object.instance_ids:
            raise error(
                "schema_vnext_instance_base_anchor_ref_forbidden",
                "object-local anchor refs must use a concrete instance id when the object defines instances",
                target=target,
                ref=ref,
                object=owner,
                instances=list(raw_object.instance_ids),
            )
        return raw_object
    for candidate in objects.values():
        if owner in candidate.instance_ids:
            return candidate
    raise error(
        f"schema_vnext_{kind}_ref_unknown_owner",
        f"{kind} ref owner must be a known object or object instance",
        target=target,
        ref=ref,
    )


def _validate_object_anchor_name(
    raw_object: ObjectSpec,
    *,
    anchor_section: str,
    anchor_name: str,
    target: str,
    ref: str,
    kind: str,
) -> None:
    anchors = raw_object.anchors if isinstance(raw_object.anchors, Mapping) else {}
    raw_anchor_payload = anchors.get(anchor_section)
    anchor_payload = raw_anchor_payload if isinstance(raw_anchor_payload, Mapping) else {}
    if anchor_name not in anchor_payload:
        raise error(
            f"schema_vnext_{kind}_ref_unknown_anchor",
            f"{kind} ref points to an unknown object-local anchor",
            target=target,
            ref=ref,
            anchor=anchor_name,
        )


def _parse_nodes(
    value: Any,
    *,
    object_refs: set[str],
    surfaces: Mapping[str, SurfaceAliasSpec],
    volumes: Mapping[str, Mapping[str, Any]],
) -> dict[str, NodeSpec]:
    payload = _mapping(value, target="layout.nodes")
    nodes: dict[str, NodeSpec] = {}
    for node_id, raw_node in payload.items():
        nid = _non_empty_str(node_id, target="layout.nodes")
        data = _mapping(raw_node, target=f"layout.nodes.{nid}")
        _reject_unsupported_fields(
            data,
            allowed={"object", "scale", "placement"},
            code="schema_vnext_node_field_unsupported",
            target=f"layout.nodes.{nid}",
        )
        object_id = _non_empty_str(data.get("object", nid), target=f"layout.nodes.{nid}.object")
        _strict_ref(object_id, object_refs, target=f"layout.nodes.{nid}.object", kind="object")
        scale = ScaleSpec.from_value(data.get("scale"), target=f"layout.nodes.{nid}.scale")
        placement = None
        if "placement" in data:
            placement = _parse_node_placement(
                data["placement"],
                target=f"layout.nodes.{nid}.placement",
                surfaces=surfaces,
                volumes=volumes,
            )
        nodes[nid] = NodeSpec(node_id=nid, object_id=object_id, scale=scale, placement=placement)
    return nodes


def _parse_node_placement(
    data: Any,
    *,
    target: str,
    surfaces: Mapping[str, SurfaceAliasSpec],
    volumes: Mapping[str, Mapping[str, Any]],
) -> NodePlacementSpec:
    payload = _mapping(data, target=target)
    placement_type = _non_empty_str(payload.get("type"), target=f"{target}.type")
    if placement_type == "surface":
        _reject_unsupported_fields(
            payload,
            allowed={"type", "surface", "orientation", "ranges", "yaws"},
            code="schema_vnext_node_placement_field_unsupported",
            target=target,
        )
        surface = _non_empty_str(payload.get("surface"), target=f"{target}.surface")
        _strict_ref(surface, set(surfaces), target=f"{target}.surface", kind="surface")
        base_orientation = None
        if "orientation" in payload:
            base_orientation = _numeric_vec(payload["orientation"], length=4, target=f"{target}.orientation")
        range_ids = _validate_ranges(payload.get("ranges"), target=f"{target}.ranges")
        yaw_ids = _validate_yaws(payload.get("yaws", 0.0), target=f"{target}.yaws")
        return NodePlacementSpec(
            placement_type=placement_type,
            surface=surface,
            base_orientation=base_orientation,
            range_candidate_ids=range_ids,
            yaw_candidate_ids=yaw_ids,
        )
    if placement_type == "pose":
        _reject_unsupported_fields(
            payload,
            allowed={"type", "poses"},
            code="schema_vnext_node_placement_field_unsupported",
            target=target,
        )
        raw_poses = _list(payload.get("poses"), target=f"{target}.poses")
        pose_ids: list[str] = []
        weighted_items: list[Mapping[str, Any]] = []
        for index, raw_pose in enumerate(raw_poses):
            pose = _mapping(raw_pose, target=f"{target}.poses[{index}]")
            _reject_unsupported_fields(
                pose,
                allowed={"id", "position", "orientation", "weight"},
                code="schema_vnext_pose_field_unsupported",
                target=f"{target}.poses[{index}]",
            )
            pose_id = _non_empty_str(pose.get("id"), target=f"{target}.poses[{index}].id")
            if "position" not in pose or "orientation" not in pose:
                raise error(
                    "schema_vnext_pose_invalid",
                    "pose candidate requires position and orientation",
                    target=f"{target}.poses.{pose_id}",
                )
            _weight(pose.get("weight"), target=f"{target}.poses.{pose_id}.weight")
            pose_ids.append(pose_id)
            weighted_items.append(pose)
        _unique(pose_ids, target=f"{target}.poses")
        _candidate_weights_positive(weighted_items, key="id", target=f"{target}.poses")
        return NodePlacementSpec(placement_type=placement_type, pose_candidate_ids=tuple(pose_ids))
    if placement_type == "volume":
        _reject_unsupported_fields(
            payload,
            allowed={"type", "volume", "mode", "yaws"},
            code="schema_vnext_node_placement_field_unsupported",
            target=target,
        )
        volume = _non_empty_str(payload.get("volume"), target=f"{target}.volume")
        _strict_ref(volume, set(volumes), target=f"{target}.volume", kind="volume")
        params: dict[str, Any] = {}
        if "mode" in payload:
            mode = _non_empty_str(payload.get("mode"), target=f"{target}.mode").lower()
            if mode not in {"fit", "center"}:
                raise error("schema_vnext_in_volume_mode_unsupported", "unsupported node volume mode", target=target)
            params["mode"] = mode
        yaw_ids = _validate_yaws(payload["yaws"], target=f"{target}.yaws") if "yaws" in payload else ()
        return NodePlacementSpec(
            placement_type=placement_type,
            volume=volume,
            yaw_candidate_ids=yaw_ids,
            params=params,
        )
    raise error(
        "schema_vnext_placement_type_unsupported", "unsupported node placement type", target=target, type=placement_type
    )


def _parse_groups(
    value: Any,
    *,
    nodes: Mapping[str, NodeSpec],
    surfaces: Mapping[str, SurfaceAliasSpec],
    volumes: Mapping[str, Mapping[str, Any]],
) -> dict[str, GroupSpec]:
    payload = _mapping(value or {}, target="layout.groups")
    groups: dict[str, GroupSpec] = {}
    for group_id, raw_group in payload.items():
        gid = _non_empty_str(group_id, target="layout.groups")
        data = _mapping(raw_group, target=f"layout.groups.{gid}")
        group_type = _non_empty_str(data.get("type"), target=f"layout.groups.{gid}.type")
        if group_type == "slot":
            groups[gid] = _parse_slot_group(gid, data, nodes=nodes, surfaces=surfaces)
            continue
        group_nodes = _string_tuple(data.get("nodes"), target=f"layout.groups.{gid}.nodes")
        if not group_nodes:
            raise error(
                "schema_vnext_group_nodes_empty",
                "group requires at least one node",
                target=f"layout.groups.{gid}.nodes",
            )
        for node_id in group_nodes:
            _strict_ref(node_id, set(nodes), target=f"layout.groups.{gid}.nodes", kind="node")
        range_ids = ()
        yaw_ids = ()
        base_orientation = None
        if group_type in {"region", "scatter"}:
            surface = _non_empty_str(data.get("surface"), target=f"layout.groups.{gid}.surface")
            _strict_ref(surface, set(surfaces), target=f"layout.groups.{gid}.surface", kind="surface")
            if "ranges" in data:
                range_ids = _validate_ranges(data["ranges"], target=f"layout.groups.{gid}.ranges")
            if "yaws" in data:
                yaw_ids = _validate_yaws(data["yaws"], target=f"layout.groups.{gid}.yaws")
            if "orientation" in data:
                base_orientation = _numeric_vec(
                    data["orientation"], length=4, target=f"layout.groups.{gid}.orientation"
                )
        elif group_type == "volume":
            volume = _non_empty_str(data.get("volume"), target=f"layout.groups.{gid}.volume")
            _strict_ref(volume, set(volumes), target=f"layout.groups.{gid}.volume", kind="volume")
            if "yaws" in data:
                yaw_ids = _validate_yaws(data["yaws"], target=f"layout.groups.{gid}.yaws")
            if "orientation" in data:
                base_orientation = _numeric_vec(
                    data["orientation"], length=4, target=f"layout.groups.{gid}.orientation"
                )
        else:
            raise error(
                "schema_vnext_group_type_unsupported",
                "unsupported group type",
                target=f"layout.groups.{gid}",
                type=group_type,
            )
        groups[gid] = GroupSpec(
            group_id=gid,
            group_type=group_type,
            nodes=group_nodes,
            base_orientation=base_orientation,
            range_candidate_ids=range_ids,
            yaw_candidate_ids=yaw_ids,
        )
    return groups


def _parse_slot_group(
    group_id: str,
    data: Mapping[str, Any],
    *,
    nodes: Mapping[str, NodeSpec],
    surfaces: Mapping[str, SurfaceAliasSpec],
) -> GroupSpec:
    group_nodes = _string_tuple(data.get("nodes"), target=f"layout.groups.{group_id}.nodes")
    if not group_nodes:
        raise error(
            "schema_vnext_group_nodes_empty",
            "slot group requires at least one node",
            target=f"layout.groups.{group_id}.nodes",
        )
    for node_id in group_nodes:
        _strict_ref(node_id, set(nodes), target=f"layout.groups.{group_id}.nodes", kind="node")
    slots = _mapping(data.get("slots"), target=f"layout.groups.{group_id}.slots")
    slot_ids = tuple(_non_empty_str(slot_id, target=f"layout.groups.{group_id}.slots") for slot_id in slots)
    _unique(list(slot_ids), target=f"layout.groups.{group_id}.slots")
    if not slot_ids:
        raise error(
            "schema_vnext_slot_group_slots_empty",
            "slot group requires at least one slot",
            target=f"layout.groups.{group_id}.slots",
        )
    for slot_id, raw_slot in slots.items():
        slot = _mapping(raw_slot, target=f"layout.groups.{group_id}.slots.{slot_id}")
        slot_type = _non_empty_str(slot.get("type"), target=f"layout.groups.{group_id}.slots.{slot_id}.type")
        if slot_type != "surface":
            raise error(
                "schema_vnext_slot_type_unsupported",
                "slot only supports type=surface",
                target=f"layout.groups.{group_id}.slots.{slot_id}",
            )
        surface = _non_empty_str(slot.get("surface"), target=f"layout.groups.{group_id}.slots.{slot_id}.surface")
        _strict_ref(surface, set(surfaces), target=f"layout.groups.{group_id}.slots.{slot_id}.surface", kind="surface")
        _validate_ranges(slot.get("ranges"), target=f"layout.groups.{group_id}.slots.{slot_id}.ranges")
        _validate_yaws(slot.get("yaws"), target=f"layout.groups.{group_id}.slots.{slot_id}.yaws")

    assignments = _mapping(data.get("assignments"), target=f"layout.groups.{group_id}.assignments")
    mode = _non_empty_str(assignments.get("mode"), target=f"layout.groups.{group_id}.assignments.mode")
    if mode == "explicit":
        items = _list(assignments.get("items"), target=f"layout.groups.{group_id}.assignments.items")
        if not items:
            raise error(
                "schema_vnext_slot_assignments_empty",
                "explicit slot assignments require at least one item",
                target=f"layout.groups.{group_id}.assignments.items",
            )
        parsed_items: list[SlotAssignmentSpec] = []
        for index, raw_item in enumerate(items):
            item = _mapping(raw_item, target=f"layout.groups.{group_id}.assignments.items[{index}]")
            assignment_id = _non_empty_str(
                item.get("id"), target=f"layout.groups.{group_id}.assignments.items[{index}].id"
            )
            bindings = _mapping(
                item.get("bindings"), target=f"layout.groups.{group_id}.assignments.{assignment_id}.bindings"
            )
            if set(bindings) != set(group_nodes):
                raise error(
                    "schema_vnext_slot_assignment_incomplete",
                    "slot assignment must bind every group node exactly once",
                    target=f"layout.groups.{group_id}.assignments.{assignment_id}",
                )
            binding_values = tuple(
                _non_empty_str(value, target=f"layout.groups.{group_id}.assignments.{assignment_id}.bindings")
                for value in bindings.values()
            )
            if len(set(binding_values)) != len(binding_values):
                raise error(
                    "schema_vnext_slot_assignment_duplicate_slot",
                    "slot assignment cannot bind multiple nodes to one slot",
                    target=f"layout.groups.{group_id}.assignments.{assignment_id}",
                )
            for slot_ref in binding_values:
                _strict_ref(
                    slot_ref,
                    set(slot_ids),
                    target=f"layout.groups.{group_id}.assignments.{assignment_id}.bindings",
                    kind="slot",
                )
            params = item.get("params", {})
            if not isinstance(params, Mapping):
                raise error(
                    "schema_vnext_slot_assignment_params_invalid",
                    "assignment params must be a mapping",
                    target=f"layout.groups.{group_id}.assignments.{assignment_id}.params",
                )
            parsed_items.append(
                SlotAssignmentSpec(assignment_id=assignment_id, bindings=dict(bindings), params=dict(params))
            )
        _unique([item.assignment_id for item in parsed_items], target=f"layout.groups.{group_id}.assignments.items")
        default_assignment = data.get("default_assignment")
        if len(parsed_items) > 1:
            default = _non_empty_str(default_assignment, target=f"layout.groups.{group_id}.default_assignment")
            _strict_ref(
                default,
                {item.assignment_id for item in parsed_items},
                target=f"layout.groups.{group_id}.default_assignment",
                kind="assignment",
            )
        else:
            default = (
                str(default_assignment).strip()
                if default_assignment
                else (parsed_items[0].assignment_id if parsed_items else None)
            )
        return GroupSpec(
            group_id=group_id,
            group_type="slot",
            nodes=group_nodes,
            slot_ids=slot_ids,
            assignment_mode=mode,
            assignments=tuple(parsed_items),
            default_assignment=default,
        )
    if mode == "permutation":
        assignment_nodes = _string_tuple(assignments.get("nodes"), target=f"layout.groups.{group_id}.assignments.nodes")
        assignment_slots = _string_tuple(assignments.get("slots"), target=f"layout.groups.{group_id}.assignments.slots")
        if set(assignment_nodes) != set(group_nodes):
            raise error(
                "schema_vnext_slot_permutation_nodes_invalid",
                "permutation nodes must match group nodes",
                target=f"layout.groups.{group_id}.assignments.nodes",
            )
        if len(assignment_nodes) != len(assignment_slots):
            raise error(
                "schema_vnext_slot_permutation_size_mismatch",
                "permutation nodes and slots must have equal length",
                target=f"layout.groups.{group_id}.assignments",
            )
        if len(set(assignment_slots)) != len(assignment_slots):
            raise error(
                "schema_vnext_slot_permutation_slot_duplicate",
                "permutation slots must be unique",
                target=f"layout.groups.{group_id}.assignments.slots",
            )
        for slot_ref in assignment_slots:
            _strict_ref(slot_ref, set(slot_ids), target=f"layout.groups.{group_id}.assignments.slots", kind="slot")
        generated = tuple(f"perm_{index}" for index, _ in enumerate(permutations(assignment_slots)))
        return GroupSpec(
            group_id=group_id,
            group_type="slot",
            nodes=group_nodes,
            slot_ids=slot_ids,
            assignment_mode=mode,
            generated_assignment_ids=generated,
        )
    raise error(
        "schema_vnext_slot_assignment_mode_unsupported",
        "unsupported slot assignments mode",
        target=f"layout.groups.{group_id}.assignments.mode",
        mode=mode,
    )


def _validate_placement_owners(nodes: Mapping[str, NodeSpec], groups: Mapping[str, GroupSpec]) -> None:
    owned: dict[str, str] = {}
    for group in groups.values():
        for node_id in group.nodes:
            if node_id in owned:
                raise error(
                    "schema_vnext_node_multi_owner", "node cannot belong to multiple placement owners", target=node_id
                )
            owned[node_id] = group.group_id
            if nodes[node_id].placement is not None:
                raise error(
                    "schema_vnext_node_group_placement_conflict",
                    "group-owned node cannot define node-level placement",
                    target=node_id,
                )


def _parse_relations(value: Any, *, nodes: Mapping[str, NodeSpec]) -> tuple[Mapping[str, Any], ...]:
    items = _list(value or [], target="layout.relations")
    relations: list[Mapping[str, Any]] = []
    for index, raw_relation in enumerate(items):
        relation = _mapping(raw_relation, target=f"layout.relations[{index}]")
        relation_type = _non_empty_str(relation.get("type"), target=f"layout.relations[{index}].type")
        if relation_type != "relative_direction":
            raise error(
                "schema_vnext_relation_type_unsupported",
                "unsupported relation type",
                target=f"layout.relations[{index}]",
                type=relation_type,
            )
        for key in ("a", "b"):
            ref = _non_empty_str(relation.get(key), target=f"layout.relations[{index}].{key}")
            _strict_ref(ref, set(nodes), target=f"layout.relations[{index}].{key}", kind="node")
        relations.append(dict(relation))
    return tuple(relations)


@dataclass(frozen=True)
class VariantChoiceSpec:
    choice_id: str
    weight: float
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    section: str
    variant_type: str
    choices: tuple[VariantChoiceSpec, ...]
    group: str | None = None


@dataclass(frozen=True)
class VariantsSpec:
    materialization: tuple[VariantSpec, ...] = ()
    assignments: tuple[VariantSpec, ...] = ()


@dataclass(frozen=True)
class VNextSceneSpec:
    background: BackgroundSpec
    objects: Mapping[str, ObjectSpec]
    layout: LayoutVNextSpec
    variants: VariantsSpec
    source: Mapping[str, Any] = field(default_factory=dict)
    definition: Mapping[str, Any] = field(default_factory=dict)
    render_settings: Mapping[str, Any] = field(default_factory=dict)
    lighting: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VNextSceneSpec:
        payload = _mapping(data, target="scene")
        runtime_topology = scene_runtime_topology_mode(payload)
        if runtime_topology not in SUPPORTED_RUNTIME_TOPOLOGIES:
            raise error(
                "schema_vnext_runtime_topology_unsupported",
                "runtime_topology must be active_only or pooled",
                target="definition.metadata.runtime_topology",
                mode=runtime_topology,
            )
        objects = _parse_objects(payload.get("objects"))
        source = deepcopy(dict(payload))
        instance_aliases = _instance_aliases(objects)
        if isinstance(source.get("layout"), Mapping):
            source["layout"] = _rewrite_layout_instance_refs(source["layout"], instance_aliases=instance_aliases)
        background = BackgroundSpec.from_dict(source.get("background"))
        layout = LayoutVNextSpec.from_dict(source.get("layout"), objects=objects)
        variants = _parse_variants(source.get("variants"), background=background, objects=objects, layout=layout)
        return cls(
            definition=dict(source.get("definition") or {}),
            render_settings=dict(source.get("render_settings") or {}),
            background=background,
            lighting=dict(source.get("lighting") or {}),
            objects=objects,
            layout=layout,
            variants=variants,
            source=source,
        )

    def object_spec_for_id(self, object_id: str) -> ObjectSpec | None:
        direct = self.objects.get(object_id)
        if direct is not None:
            return direct
        for obj in self.objects.values():
            if object_id in obj.instance_ids:
                return obj
        return None


def _parse_objects(value: Any) -> dict[str, ObjectSpec]:
    items = _list(value, target="objects")
    objects: dict[str, ObjectSpec] = {}
    for index, raw_object in enumerate(items):
        obj = ObjectSpec.from_dict(raw_object, target=f"objects[{index}]")
        if obj.object_id in objects:
            raise error("schema_vnext_object_duplicate", "object ids must be unique", target=f"objects[{index}].id")
        objects[obj.object_id] = obj
    seen = set(objects)
    for obj in objects.values():
        for instance_id in obj.instance_ids:
            if instance_id in seen:
                raise error(
                    "schema_vnext_instance_id_duplicate",
                    "object instance ids must not duplicate object or instance ids",
                    target=obj.object_id,
                    id=instance_id,
                )
            seen.add(instance_id)
    return objects


def _instance_ids(instances: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not instances:
        return ()
    if instances.get("type") == "explicit":
        return tuple(str(item) for item in instances.get("ids", ()))
    if instances.get("type") == "auto":
        start = int(instances.get("start_index", 0))
        count = int(instances.get("count", 0))
        template = str(instances.get("template"))
        return tuple(template.format(i=i) for i in range(start, start + count))
    return ()


def _instance_aliases(objects: Mapping[str, ObjectSpec]) -> dict[str, tuple[str, ...]]:
    return {object_id: obj.instance_ids for object_id, obj in objects.items() if obj.instance_ids}


def _object_reference_ids(objects: Mapping[str, ObjectSpec]) -> set[str]:
    refs: set[str] = set()
    for obj in objects.values():
        refs.update(obj.materialized_ids)
    return refs


def _rewrite_layout_instance_refs(value: Any, *, instance_aliases: Mapping[str, tuple[str, ...]]) -> Mapping[str, Any]:
    layout = deepcopy(dict(_mapping(value, target="layout")))
    if not instance_aliases:
        return layout
    nodes = dict(_mapping(layout.get("nodes", {}), target="layout.nodes"))
    groups = dict(_mapping(layout.get("groups", {}), target="layout.groups"))
    instance_ids = {instance_id for ids in instance_aliases.values() for instance_id in ids}
    for group_id, raw_group in groups.items():
        group = dict(_mapping(raw_group, target=f"layout.groups.{group_id}"))
        if "nodes" in group:
            expanded_nodes = _expand_instance_refs(
                group["nodes"], aliases=instance_aliases, target=f"layout.groups.{group_id}.nodes"
            )
            group["nodes"] = list(expanded_nodes)
            _ensure_instance_nodes(nodes, expanded_nodes, instance_ids=instance_ids)
        assignments = group.get("assignments")
        if isinstance(assignments, Mapping):
            rewritten_assignments = dict(assignments)
            if "nodes" in rewritten_assignments:
                expanded_assignment_nodes = _expand_instance_refs(
                    rewritten_assignments["nodes"],
                    aliases=instance_aliases,
                    target=f"layout.groups.{group_id}.assignments.nodes",
                )
                rewritten_assignments["nodes"] = list(expanded_assignment_nodes)
                _ensure_instance_nodes(nodes, expanded_assignment_nodes, instance_ids=instance_ids)
            group["assignments"] = rewritten_assignments
        groups[group_id] = group
    layout["nodes"] = nodes
    layout["groups"] = groups
    return layout


def _expand_instance_refs(value: Any, *, aliases: Mapping[str, tuple[str, ...]], target: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return _expand_instance_ref_item(value, aliases=aliases, target=target)
    out: list[str] = []
    for index, item in enumerate(_list(value, target=target)):
        out.extend(_expand_instance_ref_item(item, aliases=aliases, target=f"{target}[{index}]"))
    return tuple(out)


def _expand_instance_ref_item(value: Any, *, aliases: Mapping[str, tuple[str, ...]], target: str) -> tuple[str, ...]:
    text = _non_empty_str(value, target=target)
    if not text.startswith("@instances:"):
        return (text,)
    object_id = _non_empty_str(text.split(":", 1)[1], target=target)
    instances = aliases.get(object_id)
    if not instances:
        raise error("schema_vnext_instance_ref_unknown", "unknown or empty instance reference", target=target, ref=text)
    return tuple(instances)


def _ensure_instance_nodes(nodes: dict[str, Any], node_ids: tuple[str, ...], *, instance_ids: set[str]) -> None:
    for node_id in node_ids:
        if node_id in instance_ids and node_id not in nodes:
            nodes[node_id] = {"object": node_id}


def _parse_variants(
    value: Any,
    *,
    background: BackgroundSpec,
    objects: Mapping[str, ObjectSpec],
    layout: LayoutVNextSpec,
) -> VariantsSpec:
    if value is None:
        return VariantsSpec()
    payload = _mapping(value, target="variants")
    _reject_unsupported_fields(
        payload,
        allowed={"materialization", "assignments"},
        code="schema_vnext_variants_field_unsupported",
        target="variants",
    )
    materialization = tuple(
        _parse_materialization_variant(item, index=index, background=background, objects=objects)
        for index, item in enumerate(_list(payload.get("materialization", []), target="variants.materialization"))
    )
    assignments = tuple(
        _parse_assignment_variant(item, index=index, layout=layout)
        for index, item in enumerate(_list(payload.get("assignments", []), target="variants.assignments"))
    )
    _unique([variant.variant_id for variant in materialization], target="variants.materialization")
    _unique([variant.variant_id for variant in assignments], target="variants.assignments")
    return VariantsSpec(materialization=materialization, assignments=assignments)


def _parse_materialization_variant(
    value: Any,
    *,
    index: int,
    background: BackgroundSpec,
    objects: Mapping[str, ObjectSpec],
) -> VariantSpec:
    payload = _mapping(value, target=f"variants.materialization[{index}]")
    _reject_unsupported_fields(
        payload,
        allowed={"id", "type", "choices"},
        code="schema_vnext_variant_field_unsupported",
        target=f"variants.materialization[{index}]",
    )
    variant_id = _non_empty_str(payload.get("id"), target=f"variants.materialization[{index}].id")
    variant_type = _non_empty_str(payload.get("type"), target=f"variants.materialization[{index}].type")
    raw_choices = _list(payload.get("choices"), target=f"variants.materialization.{variant_id}.choices")
    if not raw_choices:
        raise error("schema_vnext_variant_choices_empty", "variant choices must be non-empty", target=variant_id)
    choices: list[VariantChoiceSpec] = []
    if variant_type == "asset_pool_choice":
        for choice_index, raw_choice in enumerate(raw_choices):
            choice = _mapping(raw_choice, target=f"variants.materialization.{variant_id}.choices[{choice_index}]")
            _reject_unsupported_fields(
                choice,
                allowed={"id", "candidate_refs", "weight"},
                code="schema_vnext_variant_choice_field_unsupported",
                target=f"variants.materialization.{variant_id}.choices[{choice_index}]",
            )
            refs = _string_tuple(
                choice.get("candidate_refs"),
                target=f"variants.materialization.{variant_id}.choices[{choice_index}].candidate_refs",
            )
            if not refs:
                raise error(
                    "schema_vnext_variant_candidate_refs_empty",
                    "asset_pool_choice requires candidate_refs",
                    target=variant_id,
                )
            seen_objects: set[str] = set()
            for ref in refs:
                object_id, candidate_id = _split_candidate_ref(ref, target=variant_id)
                _strict_ref(object_id, set(objects), target=variant_id, kind="object")
                if object_id in seen_objects:
                    raise error(
                        "schema_vnext_variant_object_duplicate",
                        "choice cannot bind the same object twice",
                        target=variant_id,
                    )
                seen_objects.add(object_id)
                if objects[object_id].asset.selector_type != "pool":
                    raise error(
                        "schema_vnext_variant_object_not_pool",
                        "asset_pool_choice target object must use asset.type=pool",
                        target=variant_id,
                    )
                _strict_ref(
                    candidate_id, objects[object_id].asset.candidate_ids, target=variant_id, kind="asset pool candidate"
                )
            choices.append(
                VariantChoiceSpec(
                    choice_id=_choice_id(choice, choice_index),
                    weight=_weight(choice.get("weight"), target=variant_id),
                    refs=refs,
                )
            )
    elif variant_type == "background_pool_choice":
        if background.asset.selector_type != "pool":
            raise error(
                "schema_vnext_background_not_pool",
                "background_pool_choice requires background.asset.type=pool",
                target=variant_id,
            )
        for choice_index, raw_choice in enumerate(raw_choices):
            choice = _mapping(raw_choice, target=f"variants.materialization.{variant_id}.choices[{choice_index}]")
            _reject_unsupported_fields(
                choice,
                allowed={"id", "candidate_ref", "weight"},
                code="schema_vnext_variant_choice_field_unsupported",
                target=f"variants.materialization.{variant_id}.choices[{choice_index}]",
            )
            ref = _non_empty_str(
                choice.get("candidate_ref"),
                target=f"variants.materialization.{variant_id}.choices[{choice_index}].candidate_ref",
            )
            prefix, candidate_id = _split_candidate_ref(ref, target=variant_id)
            if prefix != "background":
                raise error(
                    "schema_vnext_background_candidate_ref_invalid",
                    "background candidate_ref must use background.candidate_id",
                    target=variant_id,
                )
            _strict_ref(candidate_id, background.asset.candidate_ids, target=variant_id, kind="background candidate")
            choices.append(
                VariantChoiceSpec(
                    choice_id=_choice_id(choice, choice_index),
                    weight=_weight(choice.get("weight"), target=variant_id),
                    refs=(ref,),
                )
            )
    else:
        raise error(
            "schema_vnext_variant_type_unsupported",
            "unsupported materialization variant type",
            target=variant_id,
            type=variant_type,
        )
    _unique([choice.choice_id for choice in choices], target=f"variants.materialization.{variant_id}.choices")
    if not any(choice.weight > 0.0 for choice in choices):
        raise error(
            "schema_vnext_variant_weights_all_zero",
            "variant choices require at least one positive weight",
            target=variant_id,
        )
    return VariantSpec(
        variant_id=variant_id, section="materialization", variant_type=variant_type, choices=tuple(choices)
    )


def _parse_assignment_variant(value: Any, *, index: int, layout: LayoutVNextSpec) -> VariantSpec:
    payload = _mapping(value, target=f"variants.assignments[{index}]")
    _reject_unsupported_fields(
        payload,
        allowed={"id", "type", "group", "choices"},
        code="schema_vnext_variant_field_unsupported",
        target=f"variants.assignments[{index}]",
    )
    variant_id = _non_empty_str(payload.get("id"), target=f"variants.assignments[{index}].id")
    variant_type = _non_empty_str(payload.get("type"), target=f"variants.assignments[{index}].type")
    if variant_type != "slot_assignment_choice":
        raise error(
            "schema_vnext_variant_type_unsupported",
            "unsupported assignment variant type",
            target=variant_id,
            type=variant_type,
        )
    group_id = _non_empty_str(payload.get("group"), target=f"variants.assignments.{variant_id}.group")
    _strict_ref(group_id, set(layout.groups), target=variant_id, kind="group")
    group = layout.groups[group_id]
    if group.group_type != "slot":
        raise error(
            "schema_vnext_assignment_variant_group_not_slot",
            "slot_assignment_choice requires a slot group",
            target=variant_id,
        )
    raw_choices = _list(payload.get("choices"), target=f"variants.assignments.{variant_id}.choices")
    if not raw_choices:
        raise error("schema_vnext_variant_choices_empty", "variant choices must be non-empty", target=variant_id)
    choices: list[VariantChoiceSpec] = []
    for choice_index, raw_choice in enumerate(raw_choices):
        choice = _mapping(raw_choice, target=f"variants.assignments.{variant_id}.choices[{choice_index}]")
        _reject_unsupported_fields(
            choice,
            allowed={"assignment", "weight"},
            code="schema_vnext_variant_choice_field_unsupported",
            target=f"variants.assignments.{variant_id}.choices[{choice_index}]",
        )
        assignment_id = _non_empty_str(
            choice.get("assignment"), target=f"variants.assignments.{variant_id}.choices[{choice_index}].assignment"
        )
        _strict_ref(assignment_id, group.assignment_ids, target=variant_id, kind="slot assignment")
        choices.append(
            VariantChoiceSpec(
                choice_id=assignment_id, weight=_weight(choice.get("weight"), target=variant_id), refs=(assignment_id,)
            )
        )
    if not any(choice.weight > 0.0 for choice in choices):
        raise error(
            "schema_vnext_variant_weights_all_zero",
            "variant choices require at least one positive weight",
            target=variant_id,
        )
    return VariantSpec(
        variant_id=variant_id, section="assignments", variant_type=variant_type, choices=tuple(choices), group=group_id
    )


def _choice_id(choice: Mapping[str, Any], index: int) -> str:
    return _non_empty_str(choice.get("id", f"choice_{index}"), target="choice.id")


def _split_candidate_ref(ref: str, *, target: str) -> tuple[str, str]:
    if "." not in ref:
        raise error(
            "schema_vnext_candidate_ref_invalid", "candidate ref must use prefix.candidate_id", target=target, ref=ref
        )
    left, right = ref.split(".", 1)
    return (_non_empty_str(left, target=target), _non_empty_str(right, target=target))


@dataclass(frozen=True)
class ScheduleBlockSpec:
    candidate_id: str
    count: int


@dataclass(frozen=True)
class SelectorSpec:
    selector_type: str
    shared_id: str | None = None
    distinct_id: str | None = None
    weights: Mapping[str, float] = field(default_factory=dict)
    default_weight: float = 0.0
    mode: str | None = None
    order: tuple[str, ...] = ()
    blocks: tuple[ScheduleBlockSpec, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)
    order_policy: str | None = None
    on_exhausted: str = "repeat"

    @classmethod
    def from_dict(cls, data: Any, *, target: str, candidate_ids: set[str] | None = None) -> SelectorSpec:
        payload = _mapping(data, target=target)
        selector_type = _non_empty_str(payload.get("type"), target=f"{target}.type")
        if selector_type == "uniform":
            _reject_unsupported_fields(
                payload,
                allowed={"type", "shared_id", "distinct_id"},
                code="schema_vnext_selector_field_unsupported",
                target=target,
            )
            shared_id, distinct_id = _selector_coupling_ids(payload, target=target)
            return cls(selector_type=selector_type, shared_id=shared_id, distinct_id=distinct_id)
        if selector_type == "weighted":
            _reject_unsupported_fields(
                payload,
                allowed={"type", "weights", "default_weight", "shared_id", "distinct_id"},
                code="schema_vnext_selector_field_unsupported",
                target=target,
            )
            raw_weights = _mapping(payload.get("weights"), target=f"{target}.weights")
            weights = {str(key): _weight(value, target=f"{target}.weights.{key}") for key, value in raw_weights.items()}
            default_weight = _weight(payload.get("default_weight"), target=f"{target}.default_weight", default=0.0)
            if candidate_ids is not None:
                unknown = sorted(set(weights) - candidate_ids)
                if unknown:
                    raise error(
                        "schema_vnext_selector_unknown_candidate",
                        "selector weights reference unknown candidates",
                        target=target,
                        ids=unknown,
                    )
                effective = [weights.get(candidate_id, default_weight) for candidate_id in candidate_ids]
                if not any(weight > 0.0 for weight in effective):
                    raise error(
                        "schema_vnext_selector_weights_all_zero",
                        "selector requires at least one positive effective weight",
                        target=target,
                    )
            shared_id, distinct_id = _selector_coupling_ids(payload, target=target)
            return cls(
                selector_type=selector_type,
                shared_id=shared_id,
                distinct_id=distinct_id,
                weights=weights,
                default_weight=default_weight,
            )
        if selector_type == "schedule":
            mode = _non_empty_str(payload.get("mode"), target=f"{target}.mode")
            on_exhausted = _schedule_on_exhausted(
                payload.get("on_exhausted"),
                target=f"{target}.on_exhausted",
                default="repeat" if mode == "cycle" else "stop",
            )
            if mode == "cycle":
                _reject_unsupported_fields(
                    payload,
                    allowed={"type", "mode", "order", "on_exhausted", "shared_id", "distinct_id"},
                    code="schema_vnext_selector_field_unsupported",
                    target=target,
                )
                order = _string_sequence(payload.get("order"), target=f"{target}.order")
                if not order:
                    raise error("schema_vnext_selector_schedule_empty", "cycle schedule requires order", target=target)
                if candidate_ids is not None:
                    unknown = sorted(set(order) - candidate_ids)
                    if unknown:
                        raise error(
                            "schema_vnext_selector_unknown_candidate",
                            "selector order references unknown candidates",
                            target=target,
                            ids=unknown,
                        )
                shared_id, distinct_id = _selector_coupling_ids(payload, target=target)
                return cls(
                    selector_type=selector_type,
                    shared_id=shared_id,
                    distinct_id=distinct_id,
                    mode=mode,
                    order=order,
                    on_exhausted=on_exhausted,
                )
            if mode == "blocks":
                _reject_unsupported_fields(
                    payload,
                    allowed={"type", "mode", "blocks", "on_exhausted", "shared_id", "distinct_id"},
                    code="schema_vnext_selector_field_unsupported",
                    target=target,
                )
                blocks = _schedule_blocks(payload.get("blocks"), target=f"{target}.blocks", candidate_ids=candidate_ids)
                shared_id, distinct_id = _selector_coupling_ids(payload, target=target)
                return cls(
                    selector_type=selector_type,
                    shared_id=shared_id,
                    distinct_id=distinct_id,
                    mode=mode,
                    blocks=blocks,
                    on_exhausted=on_exhausted,
                )
            if mode == "counts":
                _reject_unsupported_fields(
                    payload,
                    allowed={
                        "type",
                        "mode",
                        "counts",
                        "order",
                        "order_policy",
                        "on_exhausted",
                        "shared_id",
                        "distinct_id",
                    },
                    code="schema_vnext_selector_field_unsupported",
                    target=target,
                )
                counts = _schedule_counts(payload.get("counts"), target=f"{target}.counts", candidate_ids=candidate_ids)
                order_policy = str(payload.get("order_policy") or "shuffle_once").strip()
                if order_policy not in {"authored", "shuffle_once"}:
                    raise error(
                        "schema_vnext_selector_order_policy_unsupported",
                        "unsupported schedule order_policy",
                        target=target,
                        order_policy=order_policy,
                    )
                order = _string_tuple(payload.get("order"), target=f"{target}.order")
                if order_policy == "authored" and not order:
                    raise error(
                        "schema_vnext_selector_schedule_order_missing",
                        "authored counts schedule requires order",
                        target=target,
                    )
                if order and set(order) != set(counts):
                    raise error(
                        "schema_vnext_selector_schedule_order_mismatch",
                        "counts schedule order must match counts keys",
                        target=target,
                    )
                shared_id, distinct_id = _selector_coupling_ids(payload, target=target)
                return cls(
                    selector_type=selector_type,
                    shared_id=shared_id,
                    distinct_id=distinct_id,
                    mode=mode,
                    counts=counts,
                    order=order,
                    order_policy=order_policy,
                    on_exhausted=on_exhausted,
                )
            raise error(
                "schema_vnext_selector_schedule_mode_unsupported", "unsupported schedule mode", target=target, mode=mode
            )
        raise error(
            "schema_vnext_selector_type_unsupported", "unsupported selector type", target=target, type=selector_type
        )


def _optional_selector_shared_id(payload: Mapping[str, Any], *, target: str) -> str | None:
    if "shared_id" not in payload or payload.get("shared_id") is None:
        return None
    return _non_empty_str(payload.get("shared_id"), target=f"{target}.shared_id")


def _optional_selector_distinct_id(payload: Mapping[str, Any], *, target: str) -> str | None:
    if "distinct_id" not in payload or payload.get("distinct_id") is None:
        return None
    return _non_empty_str(payload.get("distinct_id"), target=f"{target}.distinct_id")


def _selector_coupling_ids(payload: Mapping[str, Any], *, target: str) -> tuple[str | None, str | None]:
    """Parse optional same-asset and distinct-asset selector groups."""
    shared_id = _optional_selector_shared_id(payload, target=target)
    distinct_id = _optional_selector_distinct_id(payload, target=target)
    return shared_id, distinct_id


def _schedule_on_exhausted(value: Any, *, target: str, default: str) -> str:
    mode = str(value or default).strip()
    if mode not in {"stop", "repeat", "hold_last"}:
        raise error(
            "schema_vnext_selector_on_exhausted_unsupported",
            "unsupported schedule on_exhausted",
            target=target,
            on_exhausted=mode,
        )
    return mode


def _positive_count(value: Any, *, target: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise error("schema_vnext_selector_count_invalid", "schedule count must be a positive integer", target=target)
    return value


def _schedule_blocks(value: Any, *, target: str, candidate_ids: set[str] | None) -> tuple[ScheduleBlockSpec, ...]:
    blocks: list[ScheduleBlockSpec] = []
    for index, raw_block in enumerate(_list(value, target=target)):
        block = _mapping(raw_block, target=f"{target}[{index}]")
        _reject_unsupported_fields(
            block,
            allowed={"id", "count"},
            code="schema_vnext_selector_block_field_unsupported",
            target=f"{target}[{index}]",
        )
        candidate_id = _non_empty_str(block.get("id"), target=f"{target}[{index}].id")
        if candidate_ids is not None and candidate_id not in candidate_ids:
            raise error(
                "schema_vnext_selector_unknown_candidate",
                "schedule block references unknown candidate",
                target=target,
                ids=[candidate_id],
            )
        blocks.append(
            ScheduleBlockSpec(
                candidate_id=candidate_id,
                count=_positive_count(block.get("count"), target=f"{target}.{candidate_id}.count"),
            )
        )
    if not blocks:
        raise error(
            "schema_vnext_selector_schedule_empty", "blocks schedule requires at least one block", target=target
        )
    return tuple(blocks)


def _schedule_counts(value: Any, *, target: str, candidate_ids: set[str] | None) -> Mapping[str, int]:
    payload = _mapping(value, target=target)
    counts = {
        str(candidate_id): _positive_count(count, target=f"{target}.{candidate_id}")
        for candidate_id, count in payload.items()
    }
    if not counts:
        raise error(
            "schema_vnext_selector_schedule_empty", "counts schedule requires at least one candidate", target=target
        )
    if candidate_ids is not None:
        unknown = sorted(set(counts) - candidate_ids)
        if unknown:
            raise error(
                "schema_vnext_selector_unknown_candidate",
                "schedule counts reference unknown candidates",
                target=target,
                ids=unknown,
            )
    return counts


@dataclass(frozen=True)
class PlacementRuntimeSpec:
    entry_id: str
    target_type: str
    target_id: str
    sample_type: str
    selectors: Mapping[str, SelectorSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetRuntimeSpec:
    entry_id: str
    target_type: str
    target_id: str
    selector: SelectorSpec
    cache_size: int = 3


@dataclass(frozen=True)
class RuntimeSampleSpec:
    runtime_id: str = "runtime"
    seed: int | None = None
    assets: tuple[AssetRuntimeSpec, ...] = ()
    placements: tuple[PlacementRuntimeSpec, ...] = ()
    materials: tuple[Mapping[str, Any], ...] = ()
    lighting: Mapping[str, Any] = field(default_factory=dict)
    parked_slot_collision: str = "disable"


@dataclass(frozen=True)
class VNextTaskSpec:
    definition: Mapping[str, Any]
    roles: Mapping[str, Any]
    instruction: Mapping[str, Any]
    runtime_sample: RuntimeSampleSpec
    template: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, scene: VNextSceneSpec | None = None) -> VNextTaskSpec:
        payload = _mapping(data, target="task")
        if "event" in payload:
            raise error(
                "schema_vnext_task_event_forbidden",
                "task event terms are not part of schema vnext",
                target="task.event",
            )
        if "template_payload" in payload:
            raise error(
                "schema_vnext_task_template_payload_forbidden",
                "task.template_payload is internal; use task.payload",
                target="task.template_payload",
            )
        task_class_sections = [
            key
            for key in payload
            if isinstance(key, str) and key.endswith("Task") and isinstance(payload[key], Mapping)
        ]
        if task_class_sections:
            raise error(
                "schema_vnext_task_template_section_forbidden",
                "schema vNext task must use template and payload fields",
                target="task",
                keys=task_class_sections,
            )
        template = str(payload.get("template") or "").strip()
        task_template_payload = _mapping(payload.get("payload", {}), target="task.payload")
        if task_template_payload and not template:
            raise error(
                "schema_vnext_task_template_missing", "task.payload requires task.template", target="task.template"
            )
        if template and not template.endswith("Task"):
            raise error(
                "schema_vnext_task_template_invalid",
                "task.template must be a registered *Task class name",
                target="task.template",
                template=template,
            )
        layout = _mapping(payload.get("layout"), target="task.layout")
        _reject_unsupported_fields(
            layout,
            allowed={"runtime_sample"},
            code="schema_vnext_task_layout_field_forbidden",
            target="task.layout",
        )
        runtime_payload = _mapping(layout.get("runtime_sample"), target="task.layout.runtime_sample")
        runtime_allowed = {"id", "seed", "targets", "global"}
        runtime_forbidden = sorted(str(key) for key in runtime_payload if str(key) not in runtime_allowed)
        if runtime_forbidden:
            raise error(
                "schema_vnext_runtime_field_forbidden",
                "unsupported runtime_sample field",
                target="task.layout.runtime_sample",
                fields=runtime_forbidden,
            )
        runtime_id = str(runtime_payload.get("id") or "runtime")
        seed = runtime_payload.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise error(
                "schema_vnext_seed_invalid",
                "runtime_sample.seed must be an integer",
                target="task.layout.runtime_sample.seed",
            )
        targets = _mapping(runtime_payload.get("targets", {}), target="task.layout.runtime_sample.targets")
        target_forbidden = sorted(str(key) for key in targets if str(key) not in {"assets", "placements", "materials"})
        if target_forbidden:
            raise error(
                "schema_vnext_runtime_targets_field_forbidden",
                "unsupported runtime_sample.targets field",
                target="task.layout.runtime_sample.targets",
                fields=target_forbidden,
            )
        assets = _parse_runtime_assets(targets.get("assets", []), scene=scene)
        placements = _parse_runtime_placements(targets.get("placements", []), scene=scene)
        materials = _parse_runtime_materials(targets.get("materials", []))
        global_payload = _mapping(runtime_payload.get("global", {}), target="task.layout.runtime_sample.global")
        _reject_unsupported_fields(
            global_payload,
            allowed={"lighting", "parked_slot_collision"},
            code="schema_vnext_runtime_global_field_forbidden",
            target="task.layout.runtime_sample.global",
        )
        parked_slot_collision = str(global_payload.get("parked_slot_collision", "disable") or "").strip().lower()
        if parked_slot_collision not in {"disable", "keep"}:
            raise error(
                "schema_vnext_runtime_parked_slot_collision_invalid",
                "parked_slot_collision must be 'disable' or 'keep'",
                target="task.layout.runtime_sample.global.parked_slot_collision",
                value=global_payload.get("parked_slot_collision"),
            )
        return cls(
            definition=dict(payload.get("definition") or {}),
            roles=dict(payload.get("roles") or {}),
            template=template,
            payload=dict(task_template_payload),
            instruction=dict(payload.get("instruction") or {}),
            source=dict(payload),
            runtime_sample=RuntimeSampleSpec(
                runtime_id=runtime_id,
                seed=seed,
                assets=assets,
                placements=placements,
                materials=materials,
                lighting=dict(global_payload.get("lighting") or {}),
                parked_slot_collision=parked_slot_collision,
            ),
        )


def _parse_runtime_assets(value: Any, *, scene: VNextSceneSpec | None) -> tuple[AssetRuntimeSpec, ...]:
    assets: list[AssetRuntimeSpec] = []
    occupied_targets: dict[str, str] = {}
    for index, raw_asset in enumerate(_list(value, target="task.layout.runtime_sample.targets.assets")):
        item = _mapping(raw_asset, target=f"task.layout.runtime_sample.targets.assets[{index}]")
        _reject_unsupported_fields(
            item,
            allowed={"id", "target", "selector", "cache_size"},
            code="schema_vnext_runtime_asset_field_forbidden",
            target=f"task.layout.runtime_sample.targets.assets[{index}]",
        )
        entry_id = _non_empty_str(item.get("id"), target=f"task.layout.runtime_sample.targets.assets[{index}].id")
        target = _mapping(item.get("target"), target=f"assets.{entry_id}.target")
        _reject_unsupported_fields(
            target,
            allowed={"type", "id"},
            code="schema_vnext_runtime_target_field_forbidden",
            target=f"assets.{entry_id}.target",
        )
        target_type = _non_empty_str(target.get("type"), target=f"assets.{entry_id}.target.type")
        if target_type != "node":
            raise error(
                "schema_vnext_runtime_asset_target_type_unsupported",
                "runtime asset target only supports node",
                target=entry_id,
                type=target_type,
            )
        target_id = _non_empty_str(target.get("id"), target=f"assets.{entry_id}.target.id")
        candidate_ids: set[str] | None = None
        object_id = target_id
        if scene is not None:
            _strict_ref(target_id, set(scene.layout.nodes), target=entry_id, kind="node")
            node = scene.layout.nodes[target_id]
            # Instance nodes (content_1) resolve to the parent pool object (content).
            obj = scene.object_spec_for_id(node.object_id)
            if obj is None:
                raise error(
                    "schema_vnext_runtime_asset_target_unknown_object",
                    "runtime asset target references unknown object",
                    target=entry_id,
                    object=node.object_id,
                )
            object_id = obj.object_id
            if obj.asset.selector_type != "pool":
                raise error(
                    "schema_vnext_runtime_asset_target_not_pool",
                    "runtime asset target object must use asset.type: pool",
                    target=entry_id,
                    object=object_id,
                )
            candidate_ids = set(obj.asset.candidate_ids)
        selector = SelectorSpec.from_dict(
            item.get("selector", {"type": "uniform"}),
            target=f"assets.{entry_id}.selector",
            candidate_ids=candidate_ids,
        )
        declared_cache_size = item.get("cache_size")
        if declared_cache_size is not None and (
            not isinstance(declared_cache_size, int)
            or isinstance(declared_cache_size, bool)
            or declared_cache_size <= 0
        ):
            raise error(
                "schema_vnext_runtime_asset_cache_size_invalid",
                "runtime asset cache_size must be a positive integer",
                target=entry_id,
                cache_size=declared_cache_size,
            )
        scene_cache_size: int | None = None
        if scene is not None:
            # Active-only scenes select the concrete asset before gym.make and
            # therefore expose exactly one physical slot, regardless of the
            # authored pool window.  This keeps runtime compilation aligned
            # with the topology that was actually materialized.
            if scene_uses_active_only_topology(scene):
                scene_cache_size = 1
            else:
                # Scene pool cache_size is authoritative; materialization defaults missing to 3.
                scene_cache_size = int(obj.asset.cache_size) if obj.asset.cache_size is not None else 3
        if scene_cache_size is not None:
            if declared_cache_size is not None and int(declared_cache_size) != scene_cache_size:
                raise error(
                    "schema_vnext_runtime_asset_cache_size_mismatch",
                    "runtime asset cache_size must match scene pool cache_size (or be omitted)",
                    target=entry_id,
                    cache_size=declared_cache_size,
                    scene_cache_size=scene_cache_size,
                    object=object_id,
                )
            cache_size = scene_cache_size
        elif declared_cache_size is not None:
            cache_size = int(declared_cache_size)
        else:
            raise error(
                "schema_vnext_runtime_asset_cache_size_missing",
                "runtime asset cache_size must be inherited from scene pool cache_size",
                target=entry_id,
            )
        if object_id in occupied_targets:
            raise error(
                "schema_vnext_runtime_asset_target_conflict",
                "multiple runtime asset entries target the same object",
                target=entry_id,
                object=object_id,
                entries=[occupied_targets[object_id], entry_id],
            )
        occupied_targets[object_id] = entry_id
        assets.append(
            AssetRuntimeSpec(
                entry_id=entry_id,
                target_type=target_type,
                target_id=target_id,
                selector=selector,
                cache_size=int(cache_size),
            )
        )
    _unique([entry.entry_id for entry in assets], target="task.layout.runtime_sample.targets.assets")
    return tuple(assets)


def _parse_runtime_placements(value: Any, *, scene: VNextSceneSpec | None) -> tuple[PlacementRuntimeSpec, ...]:
    placements: list[PlacementRuntimeSpec] = []
    for index, raw_placement in enumerate(_list(value, target="task.layout.runtime_sample.targets.placements")):
        placement = _mapping(raw_placement, target=f"task.layout.runtime_sample.targets.placements[{index}]")
        _reject_unsupported_fields(
            placement,
            allowed={"id", "target", "sample"},
            code="schema_vnext_runtime_placement_field_forbidden",
            target=f"task.layout.runtime_sample.targets.placements[{index}]",
        )
        entry_id = _non_empty_str(
            placement.get("id"), target=f"task.layout.runtime_sample.targets.placements[{index}].id"
        )
        target = _mapping(
            placement.get("target"), target=f"task.layout.runtime_sample.targets.placements.{entry_id}.target"
        )
        _reject_unsupported_fields(
            target,
            allowed={"type", "id"},
            code="schema_vnext_runtime_target_field_forbidden",
            target=f"task.layout.runtime_sample.targets.placements.{entry_id}.target",
        )
        target_type = _non_empty_str(
            target.get("type"), target=f"task.layout.runtime_sample.targets.placements.{entry_id}.target.type"
        )
        target_id = _non_empty_str(
            target.get("id"), target=f"task.layout.runtime_sample.targets.placements.{entry_id}.target.id"
        )
        sample = _mapping(
            placement.get("sample"), target=f"task.layout.runtime_sample.targets.placements.{entry_id}.sample"
        )
        sample_type = _non_empty_str(
            sample.get("type"), target=f"task.layout.runtime_sample.targets.placements.{entry_id}.sample.type"
        )
        if sample_type not in {"inherit", "override", "jitter"}:
            raise error(
                "schema_vnext_sample_type_unsupported", "unsupported sample type", target=entry_id, type=sample_type
            )
        _validate_runtime_sample_fields(
            sample, sample_type=sample_type, target=f"task.layout.runtime_sample.targets.placements.{entry_id}.sample"
        )
        candidate_lookup: dict[str, set[str]] = {}
        if scene is not None:
            candidate_lookup = _runtime_candidate_lookup(
                scene, target_type=target_type, target_id=target_id, sample_type=sample_type, target=entry_id
            )
        selectors = {
            str(name): SelectorSpec.from_dict(
                raw_selector,
                target=f"task.layout.runtime_sample.targets.placements.{entry_id}.selectors.{name}",
                candidate_ids=(None if sample_type == "override" else candidate_lookup.get(str(name))),
            )
            for name, raw_selector in _mapping(
                sample.get("selectors", {}),
                target=f"task.layout.runtime_sample.targets.placements.{entry_id}.sample.selectors",
            ).items()
        }
        if (
            scene is not None
            and target_type == "group"
            and "assignments" in selectors
            and target_id in {str(variant.group) for variant in scene.variants.assignments}
        ):
            raise error(
                "schema_vnext_assignment_variant_runtime_conflict",
                "a slot group can be controlled by scene variants.assignments or task runtime assignment selector, not both",
                target=entry_id,
                group=target_id,
            )
        placements.append(
            PlacementRuntimeSpec(
                entry_id=entry_id,
                target_type=target_type,
                target_id=target_id,
                sample_type=sample_type,
                selectors=selectors,
            )
        )
    _unique([placement.entry_id for placement in placements], target="task.layout.runtime_sample.targets.placements")
    return tuple(placements)


def _validate_runtime_sample_fields(sample: Mapping[str, Any], *, sample_type: str, target: str) -> None:
    if sample_type == "inherit":
        allowed = {"type", "selectors"}
    elif sample_type == "override":
        allowed = {"type", "overrides", "selectors"}
    elif sample_type == "jitter":
        allowed = {"type", "dx", "dy", "yaw_delta", "selectors"}
    else:
        allowed = {"type"}
    _reject_unsupported_fields(
        sample,
        allowed=allowed,
        code="schema_vnext_runtime_sample_field_forbidden",
        target=target,
    )


def _parse_runtime_materials(value: Any) -> tuple[Mapping[str, Any], ...]:
    items: list[Mapping[str, Any]] = []
    for index, raw_item in enumerate(_list(value, target="task.layout.runtime_sample.targets.materials")):
        item = _mapping(raw_item, target=f"task.layout.runtime_sample.targets.materials[{index}]")
        _reject_unsupported_fields(
            item,
            allowed={"id", "target", "policy", "selector", "apply"},
            code="schema_vnext_runtime_material_field_forbidden",
            target=f"task.layout.runtime_sample.targets.materials[{index}]",
        )
        items.append(dict(item))
    return tuple(items)


def _runtime_candidate_lookup(
    scene: VNextSceneSpec,
    *,
    target_type: str,
    target_id: str,
    sample_type: str,
    target: str,
) -> dict[str, set[str]]:
    if target_type == "node":
        _strict_ref(target_id, set(scene.layout.nodes), target=target, kind="node")
        node = scene.layout.nodes[target_id]
        if node.placement is None:
            raise error(
                "schema_vnext_runtime_target_not_owner",
                "target node is not a node-level placement owner",
                target=target,
            )
        return {
            "ranges": set(node.placement.range_candidate_ids),
            "yaws": set(node.placement.yaw_candidate_ids),
            "poses": set(node.placement.pose_candidate_ids),
        }
    if target_type == "group":
        _strict_ref(target_id, set(scene.layout.groups), target=target, kind="group")
        group = scene.layout.groups[target_id]
        if group.group_type == "slot" and sample_type != "inherit":
            raise error(
                "schema_vnext_slot_runtime_sample_type_invalid",
                "slot group runtime sampling only supports inherit",
                target=target,
                sample_type=sample_type,
            )
        return {
            "ranges": set(group.range_candidate_ids),
            "yaws": set(group.yaw_candidate_ids),
            "poses": set(group.pose_candidate_ids),
            "assignments": group.assignment_ids,
        }
    raise error(
        "schema_vnext_runtime_target_type_unsupported",
        "runtime placement target only supports node or group",
        target=target,
        type=target_type,
    )
