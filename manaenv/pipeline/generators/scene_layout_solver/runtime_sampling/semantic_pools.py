from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .meta import (
    META_PLAN_KIND,
    META_SEMANTIC_POOL_KIND,
    META_SOURCE_SCENE_FIELD,
    META_WEIGHT_SCOPE,
    META_WEIGHT_TOTAL,
    META_ZERO_WEIGHT_EXCLUDED,
    PLAN_VARIANT_CHOICE_POOL,
    POOL_POSE_CHOICE,
    POOL_RELATION_SURFACE_RANGE_CHOICE,
    POOL_SURFACE_RANGE_CHOICE,
    POOL_VARIANT_CHOICE,
    POOL_YAW_CHOICE,
)


def semantic_pool_artifacts(plan: Any) -> list[dict[str, Any]]:
    """Return a stable summary of semantic-weighted runtime pools.

    This is an inspection artifact only; sampler/coupling behavior continues to
    use domains, assignments, and conditional assignments directly.
    """

    pools: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for domain in getattr(plan, "domains", []) or []:
        pool = _domain_pool(domain)
        if pool is not None:
            key = (str(pool.get("kind") or ""), str(pool.get("id") or ""))
            if key not in seen:
                seen.add(key)
                pools.append(pool)
    for assignment in getattr(plan, "assignments", []) or []:
        pool = _assignment_pool(assignment, conditional=False)
        if pool is not None:
            pools.append(pool)
    for conditional in getattr(plan, "conditional_assignments", []) or []:
        pool = _assignment_pool(conditional, conditional=True)
        if pool is not None:
            pools.append(pool)
    return pools


def _domain_pool(domain: Any) -> dict[str, Any] | None:
    meta = _meta(domain)
    kind = _semantic_kind(meta)
    if kind == POOL_SURFACE_RANGE_CHOICE:
        weights = _weights_from_items(getattr(domain, "range_choices", ()) or ())
    elif kind == POOL_POSE_CHOICE:
        weights = _weights_from_items(getattr(domain, "poses", ()) or ())
    elif kind == POOL_YAW_CHOICE:
        weights = _weights_from_items(getattr(domain, "yaw_choices", ()) or ())
    else:
        return None
    pool_id = str(meta.get("yaw_choice_pool_id") or getattr(domain, "domain_id", ""))
    payload = _base_payload(kind=kind, pool_id=pool_id, meta=meta, weights=weights)
    payload.update(
        {
            "domain": str(getattr(domain, "domain_id", "")),
            "object": str(getattr(domain, "object_name", "")),
            "geometry_type": str(getattr(domain, "geometry_type", "")),
        }
    )
    surface = getattr(domain, "surface", None)
    if surface is not None:
        payload["surface"] = str(surface)
    return payload


def _assignment_pool(plan: Any, *, conditional: bool) -> dict[str, Any] | None:
    meta = _meta(plan)
    kind = _semantic_kind(meta)
    if kind not in {POOL_VARIANT_CHOICE, POOL_RELATION_SURFACE_RANGE_CHOICE}:
        return None
    cases = tuple(getattr(plan, "cases", ()) or ())
    if kind == POOL_VARIANT_CHOICE:
        weights = _weights_from_items(cases)
        pool_id = str(meta.get("variant_id") or getattr(plan, "plan_id", ""))
    else:
        weights = _weights_from_mapping(meta.get("range_choice_weights"))
        pool_id = str(getattr(plan, "plan_id", ""))
    payload = _base_payload(kind=kind, pool_id=pool_id, meta=meta, weights=weights)
    key = "conditional_assignment" if conditional else "assignment"
    payload[key] = str(getattr(plan, "plan_id", ""))
    if conditional:
        payload["parent"] = str(getattr(plan, "parent", ""))
        payload["objects"] = [str(value) for value in getattr(plan, "children", ()) or ()]
    else:
        payload["objects"] = [str(value) for value in getattr(plan, "objects", ()) or ()]
    for field in ("surface", "region", "scatter_group"):
        value = meta.get(field)
        if value:
            payload[field] = str(value)
    return payload


def _semantic_kind(meta: Mapping[str, Any]) -> str:
    semantic_kind = str(meta.get(META_SEMANTIC_POOL_KIND) or "").strip()
    if semantic_kind:
        return semantic_kind
    plan_kind = str(meta.get(META_PLAN_KIND) or "").strip()
    if plan_kind in {POOL_SURFACE_RANGE_CHOICE, POOL_POSE_CHOICE, POOL_YAW_CHOICE, PLAN_VARIANT_CHOICE_POOL}:
        return POOL_VARIANT_CHOICE if plan_kind == PLAN_VARIANT_CHOICE_POOL else plan_kind
    if meta.get("surface_range_choice_pool"):
        return POOL_RELATION_SURFACE_RANGE_CHOICE
    return ""


def _base_payload(
    *,
    kind: str,
    pool_id: str,
    meta: Mapping[str, Any],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": pool_id,
        "kind": kind,
        "weights": dict(weights),
        "weight_total": _weight_total(meta, weights),
    }
    plan_kind = str(meta.get(META_PLAN_KIND) or "").strip()
    if plan_kind:
        payload["plan_kind"] = plan_kind
    for key in (META_SOURCE_SCENE_FIELD, META_WEIGHT_SCOPE, META_ZERO_WEIGHT_EXCLUDED):
        if key in meta:
            payload[key] = meta[key]
    return payload


def _weight_total(meta: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    raw = meta.get(META_WEIGHT_TOTAL)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(sum(weight for weight in weights.values() if weight > 0.0))


def _weights_from_items(items: Iterable[Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        candidate_id = str(item.get("id") or f"candidate_{index:03d}")
        try:
            out[candidate_id] = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            continue
    return out


def _weights_from_mapping(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _meta(item: Any) -> dict[str, Any]:
    raw = getattr(item, "meta", {}) or {}
    return dict(raw) if isinstance(raw, Mapping) else {}
