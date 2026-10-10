from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..assets.bridge import LayoutAssetRecord
from ..common.constants import DEFAULT_RUNTIME_PARTITION_CANDIDATES, DEFAULT_RUNTIME_PARTITION_CASES, GEOM_EPS
from ..common.diagnostics import LayoutDiagnostic
from ..common.geometry import Bounds2D, Range1D
from ..common.partitioning.geometry import center_domain_min_distance
from ..common.relations import relation_axis_and_order
from ..scene_spec.schema import RegionSampleSpec, RelativeDirectionSpec, ScatterGroupSpec
from .compile_utils import bounds2d_json
from .partitioning import (
    ObjectProfile,
    PartitionObstacle,
    PartitionProblem,
    choose_strategy_policy,
    export_partition_cases,
    generate_partition_cases,
    select_cases,
    tier_counts,
)
from .schema import PlacementDomain


@dataclass(frozen=True)
class PartitionCasePoolResult:
    domains: tuple[PlacementDomain, ...] = ()
    cases: tuple[dict[str, Any], ...] = ()
    diagnostic: LayoutDiagnostic | None = None
    meta: dict[str, Any] | None = None


def build_partition_case_pool(
    *,
    object_names: list[str],
    domain_id: str,
    frame: str,
    bounds: Bounds2D,
    records: Mapping[str, LayoutAssetRecord],
    z_by_object: Mapping[str, tuple[float, float]],
    footprint_by_object: Mapping[str, tuple[float, float]] | None = None,
    footprint_bounds_by_object: Mapping[str, Bounds2D] | None = None,
    yaw_by_object: Mapping[str, tuple[float, float] | None] | None = None,
    yaw_choices_by_object: Mapping[str, tuple[dict[str, Any], ...]] | None = None,
    base_quaternion_by_object: Mapping[str, tuple[float, float, float, float]] | None = None,
    source_event: str,
    relation: RegionSampleSpec | ScatterGroupSpec | None,
    relation_kind: str = "region_sample",
    authoring_frame: str,
    parent: str | None = None,
    surface: str | None = None,
    accept_cell_fn: Callable[[str, Range1D, Range1D], bool] | None = None,
    clip_cell_fn: Callable[[str, Range1D, Range1D], tuple[Range1D, Range1D] | None] | None = None,
    obstacles: tuple[PartitionObstacle, ...] = (),
    yaw: tuple[float, float] | None = None,
    relative_directions: tuple[RelativeDirectionSpec, ...] = (),
    min_spacing: float | None = None,
    sample_method: str | None = None,
    grid_step: float | None = None,
    grid_snap: Mapping[str, Any] | None = None,
    align_axis: str | None = None,
    order_axis: str | None = None,
    jitter_ratio: float | None = None,
    seed: int = 0,
    case_count: int = DEFAULT_RUNTIME_PARTITION_CASES,
    candidate_count: int = DEFAULT_RUNTIME_PARTITION_CANDIDATES,
) -> PartitionCasePoolResult:
    names = tuple(str(name) for name in object_names if str(name) in records)
    if not names:
        return _error("partition_case_pool_objects_empty", "partition case pool requires objects", target=domain_id)
    footprints = dict(footprint_by_object or {})
    footprint_bounds = dict(footprint_bounds_by_object or {})
    missing_z = [name for name in names if name not in z_by_object]
    if missing_z:
        return _error(
            "partition_case_pool_z_missing",
            "partition case pool requires z ranges for every object",
            target=domain_id,
            objects=missing_z,
        )

    profiles = {name: _object_profile(name, records, footprints, footprint_bounds, z_by_object) for name in names}
    relation_min_spacing = max(0.0, float(relation.min_spacing if relation is not None else 0.0))
    relation_spacing_mode = _relation_spacing_mode(relation)
    relation_jitter = max(0.0, float(relation.jitter if relation is not None else 0.0))
    relation_sample_method = _relation_sample_method(relation)
    relation_grid_step = relation.grid_step if isinstance(relation, RegionSampleSpec) else None
    relation_grid_snap = (
        dict(relation.grid_snap) if isinstance(relation, RegionSampleSpec) and relation.grid_snap is not None else None
    )
    relation_align_axis = relation.align_axis if isinstance(relation, RegionSampleSpec) else None
    relation_order_axis = relation.order_axis if isinstance(relation, RegionSampleSpec) else None
    problem = PartitionProblem(
        names=names,
        bounds=bounds,
        profiles=profiles,
        min_spacing=relation_min_spacing if min_spacing is None else max(0.0, float(min_spacing)),
        spacing_mode=relation_spacing_mode,
        jitter_ratio=relation_jitter if jitter_ratio is None else max(0.0, float(jitter_ratio)),
        sample_method=relation_sample_method if sample_method is None else str(sample_method),
        grid_step=relation_grid_step if grid_step is None else float(grid_step),
        grid_snap=relation_grid_snap if grid_snap is None else dict(grid_snap),
        align_axis=relation_align_axis if align_axis is None else str(align_axis),
        order_axis=relation_order_axis if order_axis is None else str(order_axis),
        accept_cell=accept_cell_fn,
        clip_cell=clip_cell_fn,
        obstacles=tuple(obstacles),
    )
    policy = choose_strategy_policy(problem, relation_kind=relation_kind)
    candidates = generate_partition_cases(
        problem,
        policy,
        seed=seed,
        candidate_count=max(int(candidate_count), int(case_count) * 8),
    )
    if problem.order_axis in {"x", "y"}:
        candidates = _filter_order_axis_cases(problem, candidates)
    if relative_directions:
        candidates = _filter_relative_direction_cases(problem, candidates, relative_directions)
    if problem.spacing_mode == "center_distance" and problem.min_spacing > 0.0:
        candidates = _filter_center_distance_cases(problem, candidates)
    if not candidates:
        return _error(
            "partition_case_pool_no_candidate",
            "could not generate any runtime-safe partition case",
            target=domain_id,
            object_count=len(names),
            min_spacing=problem.min_spacing,
            spacing_mode=problem.spacing_mode,
            bounds=bounds2d_json(bounds),
            polygon_filter=accept_cell_fn is not None,
            domain_filter=clip_cell_fn is not None,
            obstacle_count=len(problem.obstacles),
            strategy_policy=list(policy.strategies),
        )

    selected = select_cases(problem, candidates, max_count=max(1, int(case_count)))
    domains, cases, meta = export_partition_cases(
        selected,
        problem=problem,
        domain_id=domain_id,
        frame=frame,
        source_event=source_event,
        authoring_frame=authoring_frame,
        parent=parent,
        surface=surface,
        yaw_by_object=dict(yaw_by_object or {}),
        yaw_choices_by_object=dict(yaw_choices_by_object or {}),
        default_yaw=yaw,
        base_quaternion_by_object=dict(base_quaternion_by_object or {}),
    )
    meta.update(
        {
            "candidate_count": len(candidates),
            "candidate_tiers": tier_counts(candidates),
            "selected_count": len(selected),
            "selected_tiers": tier_counts(selected),
            "strategy_policy": list(policy.strategies),
            "sample_method": problem.sample_method,
            "spacing_mode": problem.spacing_mode,
            "grid_step": problem.grid_step,
            "grid_snap": dict(problem.grid_snap or {}),
            "align_axis": problem.align_axis,
            "order_axis": problem.order_axis,
            "jitter": problem.jitter_ratio,
            "relative_direction_count": len(relative_directions),
            "obstacle_count": len(problem.obstacles),
        }
    )
    return PartitionCasePoolResult(domains=domains, cases=cases, meta=meta)


def _error(code: str, message: str, *, target: str = "", **meta: Any) -> PartitionCasePoolResult:
    return PartitionCasePoolResult(
        diagnostic=LayoutDiagnostic(code=code, message=message, target=target, meta=dict(meta)),
        meta=dict(meta),
    )


def _relation_sample_method(relation: RegionSampleSpec | ScatterGroupSpec | None) -> str:
    if isinstance(relation, RegionSampleSpec):
        return str(relation.sample_method or "random")
    if isinstance(relation, ScatterGroupSpec):
        return str(relation.mode or "jittered_grid")
    return "random"


def _relation_spacing_mode(relation: RegionSampleSpec | ScatterGroupSpec | None) -> str:
    raw = (
        str(getattr(relation, "spacing_mode", "footprint_gap") if relation is not None else "footprint_gap")
        .strip()
        .lower()
    )
    return "center_distance" if raw in {"center", "center_distance", "center-distance"} else "footprint_gap"


def _filter_relative_direction_cases(
    problem: PartitionProblem,
    cases: list[Any],
    relative_directions: tuple[RelativeDirectionSpec, ...],
) -> list[Any]:
    constraints = []
    names = set(problem.names)
    for rel in relative_directions:
        parsed = relation_axis_and_order(rel)
        if parsed is None:
            continue
        axis, pre, post = parsed
        if pre in names and post in names:
            constraints.append((axis, pre, post))
    if not constraints:
        return cases
    return [case for case in cases if _case_satisfies_relative_directions(case, constraints)]


def _filter_order_axis_cases(problem: PartitionProblem, cases: list[Any]) -> list[Any]:
    """Keep authored members ordered and footprint-separated on ``order_axis``.

    Runtime strategies can emit permutations and point fallbacks. Apply the
    same footprint-aware ordering constraint to every generated case before
    selecting the runtime pool.
    """
    axis = str(problem.order_axis or "").strip().lower()
    if axis not in {"x", "y"} or len(problem.names) < 2:
        return cases
    out: list[Any] = []
    for case in cases:
        valid = True
        previous = None
        for name in problem.names:
            cell = case.cells.get(name)
            if cell is None:
                valid = False
                break
            profile = problem.profiles[name]
            current = profile.swept_x(cell) if axis == "x" else profile.swept_y(cell)
            if previous is not None and previous.max + problem.min_spacing > current.min + GEOM_EPS:
                valid = False
                break
            previous = current
        if valid:
            out.append(case)
    return out


def _filter_center_distance_cases(problem: PartitionProblem, cases: list[Any]) -> list[Any]:
    names = tuple(problem.names)
    if len(names) < 2:
        return cases
    out = []
    for case in cases:
        ok = True
        for i, left in enumerate(names):
            left_cell = case.cells.get(left)
            if left_cell is None:
                ok = False
                break
            for right in names[i + 1 :]:
                right_cell = case.cells.get(right)
                if right_cell is None or not _center_domains_axis_separated(
                    left_cell, right_cell, min_gap=problem.min_spacing
                ):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append(case)
    return out


def _center_domains_axis_separated(left: Any, right: Any, *, min_gap: float) -> bool:
    return center_domain_min_distance(left.x, left.y, right.x, right.y) >= float(min_gap) - 1e-9


def _case_satisfies_relative_directions(case: Any, constraints: list[tuple[str, str, str]]) -> bool:
    for axis, pre, post in constraints:
        pre_cell = case.cells.get(pre)
        post_cell = case.cells.get(post)
        if pre_cell is None or post_cell is None:
            return False
        pre_range = pre_cell.x if axis == "x" else pre_cell.y
        post_range = post_cell.x if axis == "x" else post_cell.y
        if pre_range.max >= post_range.min:
            return False
    return True


def _object_profile(
    name: str,
    records: Mapping[str, LayoutAssetRecord],
    footprints: Mapping[str, tuple[float, float]],
    footprint_bounds: Mapping[str, Bounds2D],
    z_by_object: Mapping[str, tuple[float, float]],
) -> ObjectProfile:
    footprint = tuple(
        float(v)
        for v in footprints.get(name, (records[name].local_bounds.x.length, records[name].local_bounds.y.length))
    )
    bounds = footprint_bounds.get(name) or Bounds2D(
        Range1D(-footprint[0] * 0.5, footprint[0] * 0.5),
        Range1D(-footprint[1] * 0.5, footprint[1] * 0.5),
    )
    return ObjectProfile(
        name=name,
        footprint=footprint,  # type: ignore[arg-type]
        footprint_bounds=bounds,
        z=tuple(float(v) for v in z_by_object[name]),  # type: ignore[arg-type]
    )
