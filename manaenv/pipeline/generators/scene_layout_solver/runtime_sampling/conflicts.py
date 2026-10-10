from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from ..common.diagnostics import LayoutDiagnostic
from ..common.geometry import Bounds2D, Range1D
from ..common.partitioning import ObjectProfile
from .footprint_domains import bounds_from_payload, bounds_overlap, float_pair, swept_bounds
from .meta import (
    META_FOOTPRINT,
    META_FOOTPRINT_BOUNDS,
    META_PLAN_KIND,
    META_SEMANTIC_POOL_KIND,
    PLAN_VARIANT_CHOICE_POOL,
)


@dataclass(frozen=True)
class _ConflictScope:
    scope_id: str
    group_key: str
    owner_id: str
    object_name: str
    domain_id: str
    frame: str
    frame_key: tuple[str, str]
    center_bounds: Bounds2D
    swept_bounds: Bounds2D
    plan_kind: str = ""
    semantic_pool_kind: str = ""
    source_scene_field: str = ""


def runtime_conflict_diagnostics(plan: Any) -> list[LayoutDiagnostic]:
    domains_by_id = {
        str(getattr(domain, "domain_id", "")): domain
        for domain in getattr(plan, "domains", []) or []
        if getattr(domain, "domain_id", "")
    }
    assignments = getattr(plan, "assignments", []) or []
    diagnostics: list[LayoutDiagnostic] = []
    diagnostics.extend(_variant_case_diagnostics(assignments, domains_by_id))
    diagnostics.extend(
        _independent_overlap_diagnostics(
            _assignment_scopes(assignments, domains_by_id),
            _conditional_scopes(
                getattr(plan, "conditional_assignments", []) or [],
                domains_by_id,
                skip_parent_objects=_assignment_owned_objects(assignments),
            ),
        )
    )
    return diagnostics


def _variant_case_diagnostics(assignments: Iterable[Any], domains_by_id: Mapping[str, Any]) -> list[LayoutDiagnostic]:
    diagnostics: list[LayoutDiagnostic] = []
    seen: set[tuple[str, str, str, str]] = set()
    for assignment in assignments:
        meta = _meta(assignment)
        if str(meta.get(META_PLAN_KIND) or "") != PLAN_VARIANT_CHOICE_POOL:
            continue
        for case in getattr(assignment, "cases", ()) or ():
            if not isinstance(case, Mapping):
                continue
            raw_bind = case.get("bind")
            if not isinstance(raw_bind, Mapping):
                continue
            scopes = [
                _scope_from_domain(
                    domains_by_id.get(str(domain_id)),
                    object_name=str(object_name),
                    group_key=f"variant:{getattr(assignment, 'plan_id', '')}:{case.get('id', '')}",
                    owner_id=f"assignment:{getattr(assignment, 'plan_id', '')}",
                    source_scene_field=str(meta.get("source_scene_field") or ""),
                )
                for object_name, domain_id in raw_bind.items()
            ]
            for left, right in combinations([scope for scope in scopes if scope is not None], 2):
                if left.frame_key != right.frame_key or not bounds_overlap(left.swept_bounds, right.swept_bounds):
                    continue
                pair = tuple(sorted((left.object_name, right.object_name)))
                key = (str(getattr(assignment, "plan_id", "")), str(case.get("id") or ""), pair[0], pair[1])
                if key in seen:
                    continue
                seen.add(key)
                diagnostics.append(
                    LayoutDiagnostic(
                        code="runtime_variant_case_overlap",
                        message="variant choice case can place objects in overlapping swept domains",
                        target=str(getattr(assignment, "plan_id", "")),
                        severity="error",
                        meta={
                            "case_id": str(case.get("id") or ""),
                            "objects": list(pair),
                            "domains": [left.domain_id, right.domain_id],
                            "frame": left.frame,
                            "surface": left.frame_key[1] if left.frame_key[0] == "surface" else "",
                            "left_swept_bounds": bounds2d_json(left.swept_bounds),
                            "right_swept_bounds": bounds2d_json(right.swept_bounds),
                            "recommendation": "author non-overlapping variant slots/ranges or split the case",
                        },
                    )
                )
    return diagnostics


def _independent_overlap_diagnostics(
    assignment_scopes: Iterable[_ConflictScope],
    conditional_scopes: Iterable[_ConflictScope],
) -> list[LayoutDiagnostic]:
    scopes = list(assignment_scopes) + list(conditional_scopes)
    diagnostics: list[LayoutDiagnostic] = []
    seen: set[tuple[str, str, str, str]] = set()
    for left, right in combinations(scopes, 2):
        if left.object_name == right.object_name:
            continue
        if left.group_key == right.group_key:
            continue
        if left.frame_key != right.frame_key:
            continue
        if not bounds_overlap(left.swept_bounds, right.swept_bounds):
            continue
        objects = tuple(sorted((left.object_name, right.object_name)))
        owners = tuple(sorted((left.group_key, right.group_key)))
        key = (owners[0], owners[1], objects[0], objects[1])
        if key in seen:
            continue
        seen.add(key)
        diagnostics.append(
            LayoutDiagnostic(
                code="runtime_surface_domain_overlap_unsupported",
                message="independent runtime domains on the same placement frame can overlap",
                target=",".join(objects),
                severity="error",
                meta={
                    "objects": list(objects),
                    "owners": [left.owner_id, right.owner_id],
                    "domains": [left.domain_id, right.domain_id],
                    "frame": left.frame,
                    "surface": left.frame_key[1] if left.frame_key[0] == "surface" else "",
                    "left_plan_kind": left.plan_kind,
                    "right_plan_kind": right.plan_kind,
                    "left_semantic_pool_kind": left.semantic_pool_kind,
                    "right_semantic_pool_kind": right.semantic_pool_kind,
                    "left_swept_bounds": bounds2d_json(left.swept_bounds),
                    "right_swept_bounds": bounds2d_json(right.swept_bounds),
                    "recommendation": "use a region/scatter_group case pool or explicit variants for shared areas",
                },
            )
        )
    return diagnostics


def _assignment_scopes(assignments: Iterable[Any], domains_by_id: Mapping[str, Any]) -> list[_ConflictScope]:
    scopes: list[_ConflictScope] = []
    for assignment in assignments:
        plan_id = str(getattr(assignment, "plan_id", ""))
        mode = str(getattr(assignment, "mode", ""))
        owner_id = f"assignment:{plan_id}"
        if mode == "fixed":
            for object_name, domain_id in dict(getattr(assignment, "bind", {}) or {}).items():
                scope = _scope_from_domain(
                    domains_by_id.get(str(domain_id)),
                    object_name=str(object_name),
                    group_key=f"fixed:{plan_id}:{object_name}",
                    owner_id=owner_id,
                )
                if scope is not None:
                    scopes.append(scope)
            continue
        if mode == "case_pool":
            for domain_id in _assignment_domain_ids(assignment):
                domain = domains_by_id.get(domain_id)
                scope = _scope_from_domain(
                    domain,
                    object_name=str(getattr(domain, "object_name", "")) if domain is not None else "",
                    group_key=f"case_pool:{plan_id}",
                    owner_id=owner_id,
                )
                if scope is not None:
                    scopes.append(scope)
    return scopes


def _conditional_scopes(
    conditionals: Iterable[Any],
    domains_by_id: Mapping[str, Any],
    *,
    skip_parent_objects: set[str],
) -> list[_ConflictScope]:
    scopes: list[_ConflictScope] = []
    for conditional in conditionals:
        plan_id = str(getattr(conditional, "plan_id", ""))
        owner_id = f"conditional:{plan_id}"
        parent = str(getattr(conditional, "parent", ""))
        if parent not in skip_parent_objects:
            parent_domain = domains_by_id.get(str(getattr(conditional, "parent_domain", "")))
            parent_scope = _scope_from_domain(
                parent_domain,
                object_name=parent,
                group_key=f"conditional_parent:{plan_id}:{parent}",
                owner_id=owner_id,
            )
            if parent_scope is not None:
                scopes.append(parent_scope)
        child_group_key = f"conditional_case_pool:{plan_id}" if getattr(conditional, "cases", ()) else ""
        for child, domain_id in dict(getattr(conditional, "child_domains", {}) or {}).items():
            scope = _scope_from_domain(
                domains_by_id.get(str(domain_id)),
                object_name=str(child),
                group_key=child_group_key or f"conditional_child:{plan_id}:{child}",
                owner_id=owner_id,
            )
            if scope is not None:
                scopes.append(scope)
        if getattr(conditional, "cases", ()):
            for domain_id in _conditional_case_domain_ids(conditional):
                domain = domains_by_id.get(domain_id)
                scope = _scope_from_domain(
                    domain,
                    object_name=str(getattr(domain, "object_name", "")) if domain is not None else "",
                    group_key=child_group_key,
                    owner_id=owner_id,
                )
                if scope is not None:
                    scopes.append(scope)
    return scopes


def _scope_from_domain(
    domain: Any,
    *,
    object_name: str,
    group_key: str,
    owner_id: str,
    source_scene_field: str = "",
) -> _ConflictScope | None:
    if domain is None:
        return None
    frame_key = _frame_key(domain)
    if frame_key is None:
        return None
    center_bounds = _domain_center_bounds(domain)
    profile = _profile_from_domain(domain)
    if center_bounds is None or profile is None:
        return None
    meta = _meta(domain)
    swept = swept_bounds(center_bounds, profile.footprint_bounds)
    return _ConflictScope(
        scope_id=f"{group_key}:{object_name}:{getattr(domain, 'domain_id', '')}",
        group_key=group_key,
        owner_id=owner_id,
        object_name=str(object_name or getattr(domain, "object_name", "")),
        domain_id=str(getattr(domain, "domain_id", "")),
        frame=str(getattr(domain, "frame", "")),
        frame_key=frame_key,
        center_bounds=center_bounds,
        swept_bounds=swept,
        plan_kind=str(meta.get(META_PLAN_KIND) or ""),
        semantic_pool_kind=str(meta.get(META_SEMANTIC_POOL_KIND) or ""),
        source_scene_field=source_scene_field,
    )


def _frame_key(domain: Any) -> tuple[str, str] | None:
    frame = str(getattr(domain, "frame", "") or "")
    if frame.startswith("surface:"):
        surface = str(getattr(domain, "surface", "") or frame.split(":", 1)[1]).strip()
        return ("surface", surface) if surface else None
    if frame.startswith("object:"):
        return ("object", frame)
    return None


def _domain_center_bounds(domain: Any) -> Bounds2D | None:
    geometry_type = str(getattr(domain, "geometry_type", "") or "rectangle")
    if geometry_type in {"rectangle", "box", "cylinder"}:
        return _xy_bounds(getattr(domain, "x", None), getattr(domain, "y", None))
    if geometry_type == "cells":
        return _cells_bounds(getattr(domain, "cells", ()) or ())
    if geometry_type == "surface_range_choice":
        return _range_choices_bounds(getattr(domain, "range_choices", ()) or ())
    return None


def _range_choices_bounds(choices: Iterable[Any]) -> Bounds2D | None:
    current: Bounds2D | None = None
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        bounds = (
            _cells_bounds(choice.get("cells", []) or ()) if str(choice.get("geometry_type") or "") == "cells" else None
        )
        if bounds is None:
            bounds = bounds_from_payload(choice)
        current = _bounds_union(current, bounds)
    return current


def _cells_bounds(cells: Iterable[Any]) -> Bounds2D | None:
    current: Bounds2D | None = None
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        raw = cell.get("center_domain") if isinstance(cell.get("center_domain"), Mapping) else cell
        current = _bounds_union(current, bounds_from_payload(raw))
    return current


def _xy_bounds(raw_x: Any, raw_y: Any) -> Bounds2D | None:
    x = float_pair(raw_x)
    y = float_pair(raw_y)
    if x is None or y is None:
        return None
    return Bounds2D(Range1D(float(x[0]), float(x[1])), Range1D(float(y[0]), float(y[1])))


def _bounds_union(left: Bounds2D | None, right: Bounds2D | None) -> Bounds2D | None:
    if right is None:
        return left
    if left is None:
        return right
    return Bounds2D(
        Range1D(min(left.x.min, right.x.min), max(left.x.max, right.x.max)),
        Range1D(min(left.y.min, right.y.min), max(left.y.max, right.y.max)),
    )


def bounds2d_json(bounds: Bounds2D) -> dict[str, list[float]]:
    return {"x": [float(bounds.x.min), float(bounds.x.max)], "y": [float(bounds.y.min), float(bounds.y.max)]}


def _profile_from_domain(domain: Any) -> ObjectProfile | None:
    meta = _meta(domain)
    bounds = bounds_from_payload(meta.get(META_FOOTPRINT_BOUNDS))
    footprint = float_pair(meta.get(META_FOOTPRINT))
    if footprint is None and bounds is not None:
        footprint = (bounds.x.length, bounds.y.length)
    if bounds is None and footprint is not None:
        bounds = Bounds2D(
            Range1D(-footprint[0] * 0.5, footprint[0] * 0.5),
            Range1D(-footprint[1] * 0.5, footprint[1] * 0.5),
        )
    z = getattr(domain, "z", None)
    cells = getattr(domain, "cells", ()) or ()
    if z is None and cells:
        first = cells[0] if isinstance(cells, tuple) else list(cells)[0]
        z = float_pair(first.get("z")) if isinstance(first, Mapping) else None
    if footprint is None or bounds is None or z is None:
        return None
    return ObjectProfile(
        str(getattr(domain, "object_name", "")),
        (float(footprint[0]), float(footprint[1])),
        bounds,
        (float(z[0]), float(z[1])),
    )


def _assignment_domain_ids(assignment: Any) -> list[str]:
    ids: list[str] = []
    for domain_id in getattr(assignment, "domains", ()) or ():
        _append_unique(ids, str(domain_id))
    for case in getattr(assignment, "cases", ()) or ():
        if not isinstance(case, Mapping):
            continue
        for domain_id in case.get("domains", []) or []:
            _append_unique(ids, str(domain_id))
        raw_bind = case.get("bind")
        if isinstance(raw_bind, Mapping):
            for domain_id in raw_bind.values():
                _append_unique(ids, str(domain_id))
    return ids


def _conditional_case_domain_ids(conditional: Any) -> list[str]:
    ids: list[str] = []
    for case in getattr(conditional, "cases", ()) or ():
        if not isinstance(case, Mapping):
            continue
        for domain_id in case.get("domains", []) or []:
            _append_unique(ids, str(domain_id))
        raw_bind = case.get("bind")
        if isinstance(raw_bind, Mapping):
            for domain_id in raw_bind.values():
                _append_unique(ids, str(domain_id))
    return ids


def _assignment_owned_objects(assignments: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for assignment in assignments:
        out.update(str(value) for value in getattr(assignment, "objects", ()) or ())
        out.update(str(value) for value in dict(getattr(assignment, "bind", {}) or {}))
        for case in getattr(assignment, "cases", ()) or ():
            raw_bind = case.get("bind") if isinstance(case, Mapping) else None
            if isinstance(raw_bind, Mapping):
                out.update(str(value) for value in raw_bind)
    return out


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _meta(item: Any) -> dict[str, Any]:
    raw = getattr(item, "meta", {}) or {}
    return dict(raw) if isinstance(raw, Mapping) else {}
