from __future__ import annotations

from typing import Any

from ...common.constants import (
    DEFAULT_RUNTIME_PARTITION_CANDIDATES,
    DEFAULT_RUNTIME_PARTITION_CASES,
    GEOM_EPS,
    MIN_RUNTIME_PARTITION_CANDIDATES,
    RUNTIME_PARTITION_SEED_STRIDE,
)
from ...common.geometry import Bounds2D, Range1D
from ...common.partitioning.geometry import obstacle_pair_safe
from ..cell_domains import placement_domain_from_cell, safe_id
from ..compile_utils import bounds2d_json
from ..meta import CELL_SHAPE_RECTANGLE, META_CELL_SHAPE, META_PLAN_KIND, PLAN_JOINT_PARTITION_CASE_POOL
from ..partitioning import (
    PartitionCase,
    PartitionCell,
    PartitionObstacle,
    PartitionProblem,
    choose_strategy_policy,
    export_partition_cases,
    generate_partition_cases,
    select_cases,
    tier_counts,
    without_point_fallback,
)
from ..schema import AssignmentPlan, PlacementDomain
from ..yaw_choices import yaw_choice_semantic_meta
from .models import FixedCase, FixedScope, GroupScope, JointCase


def build_joint_assignment(
    group: GroupScope,
    fixed_scopes: list[FixedScope],
    *,
    seed: int,
) -> tuple[list[PlacementDomain], AssignmentPlan, dict[str, Any]] | None:
    target_cases = max(1, len(group.assignment.cases) or int((group.assignment.meta or {}).get("selected_count") or 1))
    candidate_budget = max(
        MIN_RUNTIME_PARTITION_CANDIDATES,
        int((group.assignment.meta or {}).get("candidate_count") or DEFAULT_RUNTIME_PARTITION_CANDIDATES),
    )
    fixed_candidates = _fixed_case_candidates(fixed_scopes, target_cases=target_cases)
    if not fixed_candidates:
        return None

    joint_cases: list[JointCase] = []
    signatures: set[tuple[tuple[str, float, float, float, float], ...]] = set()
    for index, fixed_case in enumerate(fixed_candidates):
        obstacles = tuple(_obstacle_from_fixed(item) for item in fixed_case)
        min_spacing = float((group.assignment.meta or {}).get("min_spacing") or 0.0)
        if not _obstacles_safe(obstacles, min_spacing=min_spacing):
            continue
        problem = PartitionProblem(
            names=tuple(group.assignment.objects),
            bounds=group.bounds,
            profiles=dict(group.profiles),
            min_spacing=min_spacing,
            jitter_ratio=0.0,
            sample_method=str((group.assignment.meta or {}).get("sample_method") or "random"),
            align_axis=(group.assignment.meta or {}).get("align_axis"),
            order_axis=(group.assignment.meta or {}).get("order_axis"),
            accept_cell=None,
            obstacles=obstacles,
        )
        policy = without_point_fallback(choose_strategy_policy(problem, relation_kind="region_sample"))
        candidates = generate_partition_cases(
            problem,
            policy,
            seed=seed + index * RUNTIME_PARTITION_SEED_STRIDE,
            candidate_count=max(MIN_RUNTIME_PARTITION_CANDIDATES, candidate_budget // max(1, target_cases)),
        )
        selected = select_cases(problem, candidates, max_count=1)
        if not selected:
            continue
        signature = _joint_signature(fixed_case, selected[0])
        if signature in signatures:
            continue
        signatures.add(signature)
        joint_cases.append(JointCase(fixed=tuple(fixed_case), group=selected[0]))
        if len(joint_cases) >= target_cases:
            break
    if not joint_cases:
        return None

    group_problem = PartitionProblem(
        names=tuple(group.assignment.objects),
        bounds=group.bounds,
        profiles=dict(group.profiles),
        min_spacing=float((group.assignment.meta or {}).get("min_spacing") or 0.0),
        sample_method=str((group.assignment.meta or {}).get("sample_method") or "random"),
    )
    group_domains, group_payload_cases, group_meta = export_partition_cases(
        [case.group for case in joint_cases],
        problem=group_problem,
        domain_id=f"{group.assignment.plan_id}:joint",
        frame=group.frame,
        source_event=group.assignment.source_event,
        authoring_frame="scene-surface",
        parent=None,
        surface=group.surface,
        yaw_by_object={name: group.domains[name].yaw for name in group.assignment.objects if name in group.domains},
        yaw_choices_by_object={
            name: group.domains[name].yaw_choices for name in group.assignment.objects if name in group.domains
        },
        base_quaternion_by_object={
            name: group.domains[name].base_quaternion for name in group.assignment.objects if name in group.domains
        },
        default_yaw=None,
    )
    fixed_domains, cases = _fixed_domains_and_cases(group, joint_cases, group_payload_cases)
    all_domains = [*fixed_domains, *group_domains]
    assignment = AssignmentPlan(
        plan_id=f"{group.assignment.plan_id}:joint",
        mode="case_pool",
        objects=tuple([fixed.object_name for fixed in fixed_scopes] + list(group.assignment.objects)),
        domains=tuple(domain.domain_id for domain in all_domains),
        bind=dict(cases[0].get("bind") or {}) if cases else {},
        cases=tuple(cases),
        source_event=group.assignment.source_event,
        meta={
            **dict(group.assignment.meta or {}),
            META_PLAN_KIND: PLAN_JOINT_PARTITION_CASE_POOL,
            "coupled_group_assignment": group.assignment.plan_id,
            "coupled_fixed_objects": [fixed.object_name for fixed in fixed_scopes],
            "joint_case_count": len(cases),
            "group_selected_tiers": tier_counts([case.group for case in joint_cases]),
            "fixed_subdomain_count": len(fixed_domains),
            "group_meta": group_meta,
        },
    )
    return (
        all_domains,
        assignment,
        {
            "group_assignment": group.assignment.plan_id,
            "joint_assignment": assignment.plan_id,
            "surface": group.surface,
            "fixed_objects": [fixed.object_name for fixed in fixed_scopes],
            "group_objects": list(group.assignment.objects),
            "case_count": len(cases),
        },
    )


def build_fixed_joint_assignment(
    fixed_scopes: list[FixedScope],
    *,
    seed: int,
) -> tuple[list[PlacementDomain], AssignmentPlan, dict[str, Any]] | None:
    if len(fixed_scopes) < 2:
        return None
    surface = str(fixed_scopes[0].domain.surface or "")
    frame = fixed_scopes[0].domain.frame
    if not surface or any(scope.domain.surface != surface or scope.domain.frame != frame for scope in fixed_scopes):
        return None

    profiles = {scope.object_name: scope.profile for scope in fixed_scopes}
    center_bounds = {scope.object_name: scope.center_bounds for scope in fixed_scopes}
    problem = PartitionProblem(
        names=tuple(scope.object_name for scope in fixed_scopes),
        bounds=_allocated_union_bounds(fixed_scopes),
        profiles=profiles,
        min_spacing=0.0,
        sample_method="stratified",
        clip_cell=lambda name, x, y: _clip_cell_to_fixed_domain(center_bounds[str(name)], x, y),
    )
    policy = without_point_fallback(choose_strategy_policy(problem, relation_kind="region_sample"))
    candidates = generate_partition_cases(
        problem,
        policy,
        seed=seed,
        candidate_count=max(MIN_RUNTIME_PARTITION_CANDIDATES, DEFAULT_RUNTIME_PARTITION_CANDIDATES),
    )
    selected = select_cases(problem, candidates, max_count=DEFAULT_RUNTIME_PARTITION_CASES)
    if not selected:
        return None

    base_plan_id = _joint_base_plan_id(fixed_scopes)
    source_event = next((scope.source_event for scope in fixed_scopes if scope.source_event), "")
    yaw_by_object = {scope.object_name: scope.domain.yaw for scope in fixed_scopes}
    yaw_choices_by_object = {scope.object_name: scope.domain.yaw_choices for scope in fixed_scopes}
    base_quaternion_by_object = {scope.object_name: scope.domain.base_quaternion for scope in fixed_scopes}
    domains, cases, meta = export_partition_cases(
        selected,
        problem=problem,
        domain_id=f"{base_plan_id}:joint",
        frame=frame,
        source_event=source_event,
        authoring_frame="scene-surface",
        parent=None,
        surface=surface,
        yaw_by_object=yaw_by_object,
        yaw_choices_by_object=yaw_choices_by_object,
        base_quaternion_by_object=base_quaternion_by_object,
        default_yaw=None,
    )
    meta["polygon_filter"] = False
    meta["domain_filter"] = True
    assignment = AssignmentPlan(
        plan_id=f"{base_plan_id}:joint",
        mode="case_pool",
        objects=tuple(scope.object_name for scope in fixed_scopes),
        domains=tuple(domain.domain_id for domain in domains),
        bind=dict(cases[0].get("bind") or {}) if cases else {},
        cases=tuple(cases),
        source_event=source_event,
        meta={
            **meta,
            META_PLAN_KIND: PLAN_JOINT_PARTITION_CASE_POOL,
            "coupled_fixed_objects": [scope.object_name for scope in fixed_scopes],
            "joint_case_count": len(cases),
            "fixed_source_domains": {scope.object_name: scope.domain.domain_id for scope in fixed_scopes},
            "fixed_source_center_bounds": {
                scope.object_name: bounds2d_json(scope.center_bounds) for scope in fixed_scopes
            },
        },
    )
    return (
        list(domains),
        assignment,
        {
            "joint_assignment": assignment.plan_id,
            "surface": surface,
            "fixed_objects": [scope.object_name for scope in fixed_scopes],
            "case_count": len(cases),
            "mode": "fixed_component",
        },
    )


def _fixed_case_candidates(fixed_scopes: list[FixedScope], *, target_cases: int) -> list[tuple[FixedCase, ...]]:
    windows = [_windows_for_fixed(scope, count=max(1, target_cases)) for scope in fixed_scopes]
    if not windows or any(not item for item in windows):
        return []
    out: list[tuple[FixedCase, ...]] = []
    seen: set[tuple[tuple[str, float, float, float, float], ...]] = set()
    for index in range(max(len(item) for item in windows)):
        bundle = tuple(item[(index + offset) % len(item)] for offset, item in enumerate(windows))
        signature = _fixed_signature(bundle)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(bundle)
    return out


def _windows_for_fixed(scope: FixedScope, *, count: int) -> list[FixedCase]:
    bounds = scope.center_bounds
    cols, rows = _fixed_window_grid_dims(bounds, target_count=max(1, count))
    x_intervals = _sliding_ranges(bounds.x, window=_fixed_window_width(bounds.x, slots=cols), count=cols)
    y_intervals = _sliding_ranges(bounds.y, window=_fixed_window_width(bounds.y, slots=rows), count=rows)
    out: list[FixedCase] = []
    for iy, y_range in enumerate(y_intervals):
        row = list(x_intervals)
        if iy % 2:
            row.reverse()
        for x_range in row:
            out.append(FixedCase(scope=scope, cell=PartitionCell(scope.object_name, x_range, y_range, scope.profile.z)))
    return out


def _fixed_window_grid_dims(bounds: Bounds2D, *, target_count: int) -> tuple[int, int]:
    if target_count <= 1:
        return (1, 1)
    aspect = max(bounds.x.length, GEOM_EPS) / max(bounds.y.length, GEOM_EPS)
    cols = max(1, int(round((target_count * aspect) ** 0.5)))
    rows = max(1, int((target_count + cols - 1) // cols))
    while cols * rows < target_count:
        if bounds.x.length >= bounds.y.length:
            cols += 1
        else:
            rows += 1
    return (cols, rows)


def _fixed_window_width(axis: Range1D, *, slots: int) -> float:
    if axis.length <= GEOM_EPS or slots <= 1:
        return axis.length
    return min(axis.length, max(axis.length / float(slots), min(0.04, axis.length * 0.5)))


def _fixed_domains_and_cases(
    group: GroupScope,
    joint_cases: list[JointCase],
    group_payload_cases: tuple[dict[str, Any], ...],
) -> tuple[list[PlacementDomain], list[dict[str, Any]]]:
    domains: list[PlacementDomain] = []
    cases: list[dict[str, Any]] = []
    for index, (joint_case, group_case) in enumerate(zip(joint_cases, group_payload_cases)):
        bind = dict(group_case.get("bind") or {})
        case_domains = list(group_case.get("domains") or [])
        for fixed in joint_case.fixed:
            domain_id = f"{group.assignment.plan_id}:joint:case_{index:03d}:{safe_id(fixed.scope.object_name)}"
            bind[fixed.scope.object_name] = domain_id
            case_domains.append(domain_id)
            domains.append(
                _domain_from_fixed_case(
                    domain_id, fixed, group.surface, group.assignment.source_event, case_index=index
                )
            )
        cases.append(
            {
                **dict(group_case),
                "id": f"joint_partition_case_{index:03d}",
                "bind": bind,
                "domains": case_domains,
                "fixed_objects": [item.scope.object_name for item in joint_case.fixed],
            }
        )
    return domains, cases


def _domain_from_fixed_case(
    domain_id: str,
    fixed: FixedCase,
    surface: str,
    source_event: str,
    *,
    case_index: int,
) -> PlacementDomain:
    return placement_domain_from_cell(
        domain_id=domain_id,
        object_name=fixed.scope.object_name,
        frame=fixed.scope.domain.frame,
        surface=surface,
        cell=fixed.cell,
        profile=fixed.scope.profile,
        yaw=fixed.scope.domain.yaw,
        yaw_choices=fixed.scope.domain.yaw_choices,
        base_quaternion=fixed.scope.domain.base_quaternion,
        meta={
            **dict(fixed.scope.domain.meta or {}),
            "source_event": source_event,
            "authoring_frame": "scene-surface",
            **yaw_choice_semantic_meta(fixed.scope.object_name, fixed.scope.domain.yaw_choices),
            META_PLAN_KIND: PLAN_JOINT_PARTITION_CASE_POOL,
            "partition_case": case_index,
            META_CELL_SHAPE: CELL_SHAPE_RECTANGLE,
            "surface": surface,
            "source_domain": fixed.scope.domain.domain_id,
        },
    )


def _obstacle_from_fixed(fixed: FixedCase) -> PartitionObstacle:
    return PartitionObstacle(
        obstacle_id=fixed.scope.object_name,
        x=fixed.scope.profile.swept_x(fixed.cell),
        y=fixed.scope.profile.swept_y(fixed.cell),
        source_object=fixed.scope.object_name,
        source_domain=fixed.scope.domain.domain_id,
    )


def _obstacles_safe(obstacles: tuple[PartitionObstacle, ...], *, min_spacing: float) -> bool:
    problem = PartitionProblem(
        names=(),
        bounds=Bounds2D(Range1D(0.0, 0.0), Range1D(0.0, 0.0)),
        profiles={},
        min_spacing=min_spacing,
    )
    for i, left in enumerate(obstacles):
        for right in obstacles[i + 1 :]:
            if not obstacle_pair_safe(problem, left, right):
                return False
    return True


def _allocated_union_bounds(fixed_scopes: list[FixedScope]) -> Bounds2D:
    current: Bounds2D | None = None
    for scope in fixed_scopes:
        bounds = Bounds2D(
            Range1D(
                scope.center_bounds.x.min + scope.profile.footprint_bounds.x.min,
                scope.center_bounds.x.max + scope.profile.footprint_bounds.x.max,
            ),
            Range1D(
                scope.center_bounds.y.min + scope.profile.footprint_bounds.y.min,
                scope.center_bounds.y.max + scope.profile.footprint_bounds.y.max,
            ),
        )
        current = (
            bounds
            if current is None
            else Bounds2D(
                Range1D(min(current.x.min, bounds.x.min), max(current.x.max, bounds.x.max)),
                Range1D(min(current.y.min, bounds.y.min), max(current.y.max, bounds.y.max)),
            )
        )
    if current is None:
        return Bounds2D(Range1D(0.0, 0.0), Range1D(0.0, 0.0))
    return current


def _clip_cell_to_fixed_domain(bounds: Bounds2D, x: Range1D, y: Range1D) -> tuple[Range1D, Range1D] | None:
    clipped_x = x.intersect(bounds.x)
    clipped_y = y.intersect(bounds.y)
    if clipped_x is None or clipped_y is None:
        return None
    return clipped_x, clipped_y


def _joint_base_plan_id(fixed_scopes: list[FixedScope]) -> str:
    ids = []
    for scope in fixed_scopes:
        if scope.assignment_id not in ids:
            ids.append(scope.assignment_id)
    return ids[0] if len(ids) == 1 else "runtime"


def _sliding_ranges(axis: Range1D, *, window: float, count: int) -> list[Range1D]:
    width = min(axis.length, max(0.0, float(window)))
    if axis.length <= GEOM_EPS or width >= axis.length - GEOM_EPS:
        return [axis]
    steps = max(1, int(count) - 1)
    return [
        Range1D(axis.min + (axis.length - width) * i / steps, axis.min + (axis.length - width) * i / steps + width)
        for i in range(steps + 1)
    ]


def _fixed_signature(cases: tuple[FixedCase, ...]) -> tuple[tuple[str, float, float, float, float], ...]:
    return tuple(
        (
            case.scope.object_name,
            round(case.cell.x.min, 4),
            round(case.cell.x.max, 4),
            round(case.cell.y.min, 4),
            round(case.cell.y.max, 4),
        )
        for case in cases
    )


def _joint_signature(
    fixed: tuple[FixedCase, ...], group: PartitionCase
) -> tuple[tuple[str, float, float, float, float], ...]:
    fixed_items = list(_fixed_signature(fixed))
    group_items = [
        (name, round(cell.x.min, 4), round(cell.x.max, 4), round(cell.y.min, 4), round(cell.y.max, 4))
        for name, cell in sorted(group.cells.items())
    ]
    return tuple(sorted([*fixed_items, *group_items]))
