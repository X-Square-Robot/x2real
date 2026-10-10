from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ...common.geometry import Bounds2D, Range1D
from ..footprint_domains import bounds_from_payload, bounds_overlap, float_pair, swept_bounds
from ..meta import (
    META_BOUNDS,
    META_FOOTPRINT,
    META_FOOTPRINT_BOUNDS,
    META_PLAN_KIND,
    META_POLYGON_FILTER,
    PLAN_PARTITION_CASE_POOL,
)
from ..partitioning import ObjectProfile
from ..schema import AssignmentPlan, PlacementDomain, RuntimeSamplingPlan
from .models import CouplingComponent, FixedScope, GroupScope, PlacementScope


def group_scopes(plan: RuntimeSamplingPlan) -> list[GroupScope]:
    return [scope for assignment in plan.assignments if (scope := _group_scope(plan, assignment)) is not None]


def fixed_scopes(plan: RuntimeSamplingPlan) -> list[FixedScope]:
    domains_by_id = {domain.domain_id: domain for domain in plan.domains}
    out: list[FixedScope] = []
    for assignment in plan.assignments:
        if assignment.mode != "fixed":
            continue
        for object_name, domain_id in assignment.bind.items():
            domain = domains_by_id.get(str(domain_id))
            if domain is None or not domain.frame.startswith("surface:") or not domain.surface:
                continue
            domain_bounds = domain_center_bounds(domain)
            profile = profile_from_domain(domain)
            if domain_bounds is None or profile is None:
                continue
            out.append(
                FixedScope(
                    assignment_id=assignment.plan_id,
                    object_name=str(object_name),
                    domain=domain,
                    profile=profile,
                    center_bounds=domain_bounds,
                    swept_bounds=swept_bounds(domain_bounds, profile.footprint_bounds),
                    source_event=assignment.source_event,
                    origin="assignment",
                )
            )
    for conditional in plan.conditional_assignments:
        domain = domains_by_id.get(str(conditional.parent_domain))
        if domain is None or not domain.frame.startswith("surface:") or not domain.surface:
            continue
        domain_bounds = domain_center_bounds(domain)
        profile = profile_from_domain(domain)
        if domain_bounds is None or profile is None:
            continue
        out.append(
            FixedScope(
                assignment_id=conditional.plan_id,
                object_name=str(conditional.parent),
                domain=domain,
                profile=profile,
                center_bounds=domain_bounds,
                swept_bounds=swept_bounds(domain_bounds, profile.footprint_bounds),
                source_event=conditional.source_event,
                origin="conditional_parent",
            )
        )
    return out


def overlapping_fixed_scopes(group: GroupScope, scopes: list[FixedScope]) -> list[FixedScope]:
    return [
        fixed
        for fixed in scopes
        if fixed.domain.surface == group.surface
        and fixed.object_name not in group.assignment.objects
        and bounds_overlap(fixed.swept_bounds, group.bounds)
    ]


def placement_scopes(plan: RuntimeSamplingPlan) -> list[PlacementScope]:
    out: list[PlacementScope] = []
    for fixed in fixed_scopes(plan):
        out.append(
            PlacementScope(
                scope_id=f"fixed:{fixed.assignment_id}:{fixed.object_name}",
                kind="fixed",
                assignment_id=fixed.assignment_id,
                objects=(fixed.object_name,),
                surface=str(fixed.domain.surface or ""),
                frame=fixed.domain.frame,
                center_bounds=fixed.center_bounds,
                trigger_bounds=fixed.swept_bounds,
                fixed=fixed,
            )
        )
    for group in group_scopes(plan):
        out.append(
            PlacementScope(
                scope_id=f"group:{group.assignment.plan_id}",
                kind="case_pool",
                assignment_id=group.assignment.plan_id,
                objects=tuple(group.assignment.objects),
                surface=group.surface,
                frame=group.frame,
                center_bounds=None,
                trigger_bounds=group.bounds,
                group=group,
            )
        )
    return out


def overlap_components(scopes: list[PlacementScope]) -> list[CouplingComponent]:
    by_key: dict[tuple[str, str], list[int]] = {}
    for index, scope in enumerate(scopes):
        by_key.setdefault((scope.surface, scope.frame), []).append(index)

    edges: dict[int, set[int]] = {index: set() for index in range(len(scopes))}
    for indexes in by_key.values():
        for offset, left_index in enumerate(indexes):
            left = scopes[left_index]
            for right_index in indexes[offset + 1 :]:
                right = scopes[right_index]
                if set(left.objects) & set(right.objects):
                    continue
                if bounds_overlap(left.trigger_bounds, right.trigger_bounds):
                    edges[left_index].add(right_index)
                    edges[right_index].add(left_index)

    seen: set[int] = set()
    components: list[CouplingComponent] = []
    for index in range(len(scopes)):
        if index in seen or not edges[index]:
            continue
        stack = [index]
        group: list[int] = []
        seen.add(index)
        while stack:
            current = stack.pop()
            group.append(current)
            for next_index in sorted(edges[current]):
                if next_index in seen:
                    continue
                seen.add(next_index)
                stack.append(next_index)
        if len(group) > 1:
            components.append(CouplingComponent(tuple(scopes[item] for item in sorted(group))))
    return components


def polygon_overlap_warnings(plan: RuntimeSamplingPlan, scopes: list[FixedScope]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for assignment in plan.assignments:
        meta = assignment.meta or {}
        if (
            assignment.mode != "case_pool"
            or meta.get(META_PLAN_KIND) != PLAN_PARTITION_CASE_POOL
            or not bool(meta.get(META_POLYGON_FILTER))
        ):
            continue
        surface = str(meta.get("surface") or "").strip()
        bounds = bounds_from_payload(meta.get(META_BOUNDS))
        if not surface or bounds is None:
            continue
        overlapping = [
            fixed.object_name
            for fixed in scopes
            if fixed.domain.surface == surface
            and fixed.object_name not in assignment.objects
            and bounds_overlap(fixed.swept_bounds, bounds)
        ]
        if overlapping:
            warnings.append(
                {
                    "target": assignment.plan_id,
                    "surface": surface,
                    "group_objects": list(assignment.objects),
                    "fixed_objects": overlapping,
                }
            )
    return warnings


def profile_from_domain(domain: PlacementDomain) -> ObjectProfile | None:
    bounds = bounds_from_payload((domain.meta or {}).get(META_FOOTPRINT_BOUNDS))
    footprint = float_pair((domain.meta or {}).get(META_FOOTPRINT))
    if footprint is None and bounds is not None:
        footprint = (bounds.x.length, bounds.y.length)
    if bounds is None and footprint is not None:
        bounds = Bounds2D(
            Range1D(-footprint[0] * 0.5, footprint[0] * 0.5), Range1D(-footprint[1] * 0.5, footprint[1] * 0.5)
        )
    z = domain.z
    if z is None and domain.cells:
        z = float_pair(domain.cells[0].get("z"))
    if footprint is None or bounds is None or z is None:
        return None
    return ObjectProfile(
        domain.object_name, (float(footprint[0]), float(footprint[1])), bounds, (float(z[0]), float(z[1]))
    )


def domain_center_bounds(domain: PlacementDomain) -> Bounds2D | None:
    if domain.geometry_type == "rectangle" and domain.x is not None and domain.y is not None:
        return Bounds2D(
            Range1D(float(domain.x[0]), float(domain.x[1])), Range1D(float(domain.y[0]), float(domain.y[1]))
        )
    if domain.geometry_type == "surface_range_choice":
        return _range_choices_bounds(domain.range_choices or ())
    if domain.geometry_type != "cells" or not domain.cells:
        return None
    current: Bounds2D | None = None
    for cell in domain.cells:
        raw = cell.get("center_domain") if isinstance(cell.get("center_domain"), Mapping) else cell
        bounds = bounds_from_payload(raw)
        if bounds is None:
            return None
        current = (
            bounds
            if current is None
            else Bounds2D(
                Range1D(min(current.x.min, bounds.x.min), max(current.x.max, bounds.x.max)),
                Range1D(min(current.y.min, bounds.y.min), max(current.y.max, bounds.y.max)),
            )
        )
    return current


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
        raw = (
            cell.get("center_domain")
            if isinstance(cell, Mapping) and isinstance(cell.get("center_domain"), Mapping)
            else cell
        )
        bounds = bounds_from_payload(raw)
        if bounds is None:
            return None
        current = _bounds_union(current, bounds)
    return current


def _bounds_union(left: Bounds2D | None, right: Bounds2D | None) -> Bounds2D | None:
    if right is None:
        return left
    if left is None:
        return right
    return Bounds2D(
        Range1D(min(left.x.min, right.x.min), max(left.x.max, right.x.max)),
        Range1D(min(left.y.min, right.y.min), max(left.y.max, right.y.max)),
    )


def _group_scope(plan: RuntimeSamplingPlan, assignment: AssignmentPlan) -> GroupScope | None:
    meta = assignment.meta or {}
    if assignment.mode != "case_pool" or meta.get(META_PLAN_KIND) != PLAN_PARTITION_CASE_POOL:
        return None
    surface = str(meta.get("surface") or "").strip()
    if not surface or bool(meta.get(META_POLYGON_FILTER)):
        return None
    bounds = bounds_from_payload(meta.get(META_BOUNDS))
    if bounds is None:
        return None
    domains_by_id = {domain.domain_id: domain for domain in plan.domains}
    group_domains: dict[str, PlacementDomain] = {}
    profiles: dict[str, ObjectProfile] = {}
    frame = ""
    for object_name in assignment.objects:
        domain = _domain_for_object(assignment, domains_by_id, object_name)
        if domain is None or domain.surface != surface:
            return None
        profile = profile_from_domain(domain)
        if profile is None:
            return None
        group_domains[object_name] = domain
        profiles[object_name] = profile
        frame = frame or domain.frame
    if not frame.startswith("surface:"):
        return None
    return GroupScope(
        assignment=assignment, surface=surface, bounds=bounds, frame=frame, domains=group_domains, profiles=profiles
    )


def _domain_for_object(
    assignment: AssignmentPlan,
    domains_by_id: Mapping[str, PlacementDomain],
    object_name: str,
) -> PlacementDomain | None:
    for case in assignment.cases:
        bind = case.get("bind") if isinstance(case, Mapping) else None
        if isinstance(bind, Mapping) and object_name in bind:
            domain = domains_by_id.get(str(bind[object_name]))
            if domain is not None:
                return domain
    for domain_id in assignment.domains:
        domain = domains_by_id.get(str(domain_id))
        if domain is not None and domain.object_name == object_name:
            return domain
    return None
