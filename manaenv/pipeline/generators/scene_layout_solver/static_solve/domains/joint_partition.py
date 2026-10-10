from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from ...common.constants import DEFAULT_SUPPORT_CONTACT_OFFSET_M, GEOM_EPS
from ...common.geometry import (
    Bounds2D,
    Pose,
    Range1D,
    projected_footprint_bounds,
    quat_mul,
    support_contact_offset,
    yaw_quat,
)
from ...common.partitioning import (
    ObjectProfile,
    PartitionCase,
    PartitionObstacle,
    PartitionProblem,
    choose_strategy_policy,
    generate_partition_cases,
    score_cases,
    tier_counts,
    without_point_fallback,
)
from ...common.partitioning.geometry import case_signature
from ...common.relations import relation_axis_and_order
from ...common.support_surfaces import support_cell_accept_fn, surface_child_z_range
from ...scene_spec.schema import RegionSampleSpec, ScatterGroupSpec, SurfaceRangeSpec
from ..constraints.relation_geometry import footprint_ranges_on_surface, footprint_ranges_separated
from ..problem import LayoutProblem
from .providers import SupportOnProvider, object_base_quaternion, object_quaternion, object_scene_yaw_range

DEFAULT_STATIC_PARTITION_CASES = 8
DEFAULT_STATIC_PARTITION_CANDIDATES = 128
MAX_STATIC_PARTITION_CASES = 16
MAX_STATIC_PARTITION_CANDIDATES = 256


@dataclass(frozen=True)
class StaticJointCases:
    cases: list[dict[str, Pose]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def build_static_joint_cases(
    problem: LayoutProblem,
    *,
    relation: RegionSampleSpec | ScatterGroupSpec,
    relation_kind: str,
    members: list[str],
    surface_id: str,
    relation_overrides: dict[str, Any],
    assignment: dict[str, Pose] | None,
    rng: random.Random,
    budget: int,
    group_surface_range: SurfaceRangeSpec | None = None,
) -> StaticJointCases:
    names = [str(name) for name in members if str(name) in problem.objects]
    if len(names) <= 1:
        return StaticJointCases(meta={"joint_cases_skipped": "single_or_empty_group"})
    assignment = dict(assignment or {})
    surface = problem.resolve_surface(surface_id, assignment)
    if surface is None:
        reason = "local_surface_parent_unplaced" if problem.local_surface_parent(surface_id) else "surface_not_found"
        return StaticJointCases(meta={"joint_cases_reason": reason, "surface": surface_id})
    base_bounds = surface.effective_bounds()
    if base_bounds is None:
        return StaticJointCases(meta={"joint_cases_reason": "surface_effective_bounds_empty", "surface": surface_id})

    provider = SupportOnProvider(problem, relation_overrides=relation_overrides, assignment=assignment, rng=rng)
    raw_bounds = provider.apply_surface_range(base_bounds, group_surface_range)
    if raw_bounds is None:
        return StaticJointCases(meta={"joint_cases_reason": "surface_range_empty", "surface": surface_id})

    yaw_range_by_object = {name: _effective_yaw_range(problem, name, relation) for name in names}
    center_bounds_by_object: dict[str, Bounds2D] = {}
    for name in names:
        center_bounds, meta = provider.center_bounds(
            name,
            surface_override=surface_id,
            group_surface_range=group_surface_range,
            yaw_range=yaw_range_by_object[name],
        )
        if center_bounds is None:
            return StaticJointCases(
                meta={
                    "joint_cases_reason": str(meta.get("reason") or "center_domain_empty"),
                    "empty_object": name,
                    "surface": surface_id,
                }
            )
        center_bounds_by_object[name] = center_bounds

    static_yaw_by_object = {name: _sample_static_yaw(rng, yaw_range_by_object[name]) for name in names}
    base_quaternion_by_object = {name: object_base_quaternion(problem, name) for name in names}
    quaternions = {
        name: (
            quat_mul(yaw_quat(float(static_yaw_by_object[name])), base_quaternion_by_object[name])
            if static_yaw_by_object[name] is not None
            else object_quaternion(problem, name)
        )
        for name in names
    }
    footprint_bounds_by_object = {
        name: projected_footprint_bounds(
            problem.objects[name].asset.local_bounds,
            quaternions[name],
            surface.x_axis,
            surface.y_axis,
        )
        for name in names
    }
    footprint_by_object = {
        name: (
            footprint_bounds_by_object[name].x.length,
            footprint_bounds_by_object[name].y.length,
        )
        for name in names
    }
    z_by_object = {
        name: surface_child_z_range(surface, problem.objects[name].asset, quaternion=quaternions[name])
        for name in names
    }
    relative_directions = tuple(
        rel
        for rel in problem.spec.relative_directions
        if rel.node_1 in center_bounds_by_object and rel.node_2 in center_bounds_by_object
    )
    line_cases = _static_line_pose_cases(
        problem,
        names=names,
        surface=surface,
        center_bounds_by_object=center_bounds_by_object,
        footprint_bounds_by_object=footprint_bounds_by_object,
        static_yaw_by_object=static_yaw_by_object,
        base_quaternion_by_object=base_quaternion_by_object,
        relation=relation,
        relative_directions=relative_directions,
    )
    if line_cases:
        return StaticJointCases(
            cases=line_cases,
            meta={
                "joint_case_count": len(line_cases),
                "joint_candidate_count": len(line_cases),
                "joint_selected_tiers": {"static_line": len(line_cases)},
                "strategy_policy": ["static_line"],
            },
        )

    pack_min_spacing = _relation_min_spacing(relation)
    partition_problem = PartitionProblem(
        names=tuple(names),
        bounds=raw_bounds,
        profiles={
            name: ObjectProfile(
                name=name,
                footprint=footprint_by_object[name],
                footprint_bounds=footprint_bounds_by_object[name],
                z=z_by_object[name],
            )
            for name in names
        },
        min_spacing=pack_min_spacing,
        spacing_mode=_spacing_mode(relation),
        jitter_ratio=max(0.0, float(getattr(relation, "jitter", 0.0) or 0.0)),
        sample_method=_relation_sample_method(relation),
        grid_step=getattr(relation, "grid_step", None) if isinstance(relation, RegionSampleSpec) else None,
        grid_snap=(
            dict(getattr(relation, "grid_snap", None) or {})
            if isinstance(relation, RegionSampleSpec) and getattr(relation, "grid_snap", None) is not None
            else None
        ),
        align_axis=getattr(relation, "align_axis", None) if isinstance(relation, RegionSampleSpec) else None,
        order_axis=getattr(relation, "order_axis", None) if isinstance(relation, RegionSampleSpec) else None,
        boundary_spacing=0.0,
        accept_cell=(
            support_cell_accept_fn(
                surface,
                {name: problem.objects[name].asset for name in names},
                quaternions,
                footprint_by_object,
                footprint_bounds_by_object,
            )
            if _surface_needs_cell_filter(surface)
            else None
        ),
        clip_cell=_clip_to_center_bounds(center_bounds_by_object),
        obstacles=_surface_hole_obstacles(surface),
    )
    seed = rng.getrandbits(63)
    generated_problem, generated, generation_meta = _generate_static_partition_cases(
        partition_problem,
        relation_kind=relation_kind,
        seed=seed,
        candidate_count=max(DEFAULT_STATIC_PARTITION_CANDIDATES, min(MAX_STATIC_PARTITION_CANDIDATES, int(budget) * 2)),
    )
    if relative_directions:
        generated = [
            case
            for case in generated
            if _static_case_satisfies_relative_directions(generated_problem, case, relative_directions)
        ]
    selected = _select_static_cases(
        generated_problem,
        generated,
        max_count=max(DEFAULT_STATIC_PARTITION_CASES, min(MAX_STATIC_PARTITION_CASES, int(budget))),
    )
    if not selected:
        return StaticJointCases(
            meta={
                "joint_cases_reason": "static_partition_no_candidate",
                "candidate_count": len(generated),
                **generation_meta,
                "relative_direction_count": len(relative_directions),
                "obstacle_count": len(generated_problem.obstacles),
            }
        )
    return StaticJointCases(
        cases=[
            _partition_case_to_poses(
                problem,
                case,
                surface,
                static_yaw_by_object=static_yaw_by_object,
                base_quaternion_by_object=base_quaternion_by_object,
            )
            for case in selected
        ],
        meta={
            "joint_case_count": len(selected),
            "joint_candidate_count": len(generated),
            "joint_candidate_tiers": tier_counts(generated),
            "joint_selected_tiers": tier_counts(selected),
            **generation_meta,
        },
    )


def build_static_joint_cases_from_candidates(
    problem: LayoutProblem,
    *,
    members: list[str],
    surface_id: str,
    relation: RegionSampleSpec | ScatterGroupSpec,
    member_candidates: dict[str, list[Pose]],
    assignment: dict[str, Pose] | None,
    rng: random.Random,
    budget: int,
) -> StaticJointCases:
    names = [name for name in members if name in member_candidates and member_candidates.get(name)]
    if len(names) <= 1:
        return StaticJointCases(meta={"candidate_joint_skipped": "single_or_empty_group"})
    surface = problem.resolve_surface(surface_id, assignment or {})
    if surface is None:
        return StaticJointCases(meta={"candidate_joint_reason": "surface_unresolved", "surface": surface_id})
    min_spacing = _relation_min_spacing(relation)
    spacing_mode = _spacing_mode(relation)
    target_cases = max(1, min(32, int(budget)))
    candidate_limit = max(16, min(128, int(budget) * 2))
    ordered = sorted(
        names,
        key=lambda name: (
            len(member_candidates.get(name, ())),
            -_footprint_area(problem, name),
            name,
        ),
    )
    candidate_items: dict[str, list[tuple[Pose, Any]]] = {}
    for name in ordered:
        poses = list(member_candidates.get(name, []))[:candidate_limit]
        rng.shuffle(poses)
        candidate_items[name] = [
            (
                pose,
                footprint_ranges_on_surface(record=problem.objects[name].asset, pose=pose, surface=surface),
            )
            for pose in poses
        ]
        if not candidate_items[name]:
            return StaticJointCases(meta={"candidate_joint_reason": "member_candidates_empty", "object": name})

    out: list[dict[str, Pose]] = []
    selected: dict[str, tuple[Pose, Any]] = {}
    steps = 0
    max_steps = max(1024, int(budget) * len(ordered) * 64)

    def search(index: int) -> bool:
        nonlocal steps
        if len(out) >= target_cases:
            return True
        if steps >= max_steps:
            return False
        if index >= len(ordered):
            out.append({name: item[0] for name, item in selected.items()})
            return len(out) >= target_cases
        name = ordered[index]
        for pose, ranges in candidate_items[name]:
            steps += 1
            if all(
                _candidate_pair_safe(
                    surface, pose, ranges, other_pose, other_ranges, min_spacing=min_spacing, spacing_mode=spacing_mode
                )
                for other_pose, other_ranges in selected.values()
            ):
                selected[name] = (pose, ranges)
                search(index + 1)
                selected.pop(name, None)
                if len(out) >= target_cases or steps >= max_steps:
                    break
        return len(out) >= target_cases

    search(0)
    return StaticJointCases(
        cases=out,
        meta={
            "candidate_joint_case_count": len(out),
            "candidate_joint_steps": steps,
            "candidate_joint_candidate_limit": candidate_limit,
            "candidate_joint_order": list(ordered),
            "spacing_mode": spacing_mode,
        },
    )


def _candidate_pair_safe(
    surface: Any,
    left_pose: Pose,
    left_ranges: Any,
    right_pose: Pose,
    right_ranges: Any,
    *,
    min_spacing: float,
    spacing_mode: str,
) -> bool:
    if spacing_mode == "center_distance":
        lx, ly = surface.world_to_local_xy(left_pose.position)
        rx, ry = surface.world_to_local_xy(right_pose.position)
        return math.hypot(float(rx) - float(lx), float(ry) - float(ly)) >= min_spacing - GEOM_EPS
    return footprint_ranges_separated(left_ranges, right_ranges, min_gap=min_spacing)


def _effective_yaw_range(
    problem: LayoutProblem, name: str, relation: RegionSampleSpec | ScatterGroupSpec
) -> tuple[float, float] | None:
    return object_scene_yaw_range(problem, name) or getattr(relation, "yaw", None)


def _sample_static_yaw(rng: random.Random, yaw_range: tuple[float, float] | None) -> float | None:
    if yaw_range is None:
        return None
    lo, hi = float(yaw_range[0]), float(yaw_range[1])
    return lo if abs(hi - lo) <= 1e-12 else rng.uniform(lo, hi)


def _relation_sample_method(relation: RegionSampleSpec | ScatterGroupSpec) -> str:
    if isinstance(relation, RegionSampleSpec):
        return str(relation.sample_method or "random")
    return str(relation.mode or "jittered_grid")


def _relation_min_spacing(relation: RegionSampleSpec | ScatterGroupSpec) -> float:
    return max(0.0, float(getattr(relation, "min_spacing", 0.0) or 0.0))


def _spacing_mode(relation: RegionSampleSpec | ScatterGroupSpec) -> str:
    raw = str(getattr(relation, "spacing_mode", "footprint_gap") or "footprint_gap").strip().lower()
    return "center_distance" if raw in {"center", "center_distance", "center-distance"} else "footprint_gap"


def _select_static_cases(
    problem: PartitionProblem, cases: list[PartitionCase], *, max_count: int
) -> list[PartitionCase]:
    scored = score_cases(problem, cases)
    scored.sort(key=lambda case: (-case.quality, case.tier, case.topology, case_signature(case.cells)))
    return scored[: max(1, int(max_count))]


def _generate_static_partition_cases(
    problem: PartitionProblem,
    *,
    relation_kind: str,
    seed: int,
    candidate_count: int,
) -> tuple[PartitionProblem, list[PartitionCase], dict[str, Any]]:
    strict_policy = without_point_fallback(choose_strategy_policy(problem, relation_kind=relation_kind))
    strict = generate_partition_cases(problem, strict_policy, seed=seed, candidate_count=candidate_count)
    return problem, strict, {"strategy_policy": list(strict_policy.strategies)}


def _static_case_satisfies_relative_directions(
    problem: PartitionProblem,
    case: PartitionCase,
    relative_directions: tuple[Any, ...],
) -> bool:
    for rel in relative_directions:
        parsed = relation_axis_and_order(rel)
        if parsed is None:
            continue
        axis, pre, post = parsed
        pre_cell = case.cells.get(pre)
        post_cell = case.cells.get(post)
        if pre_cell is None or post_cell is None:
            return False
        pre_profile = problem.profiles[pre]
        post_profile = problem.profiles[post]
        pre_range = pre_profile.swept_x(pre_cell) if axis == "x" else pre_profile.swept_y(pre_cell)
        post_range = post_profile.swept_x(post_cell) if axis == "x" else post_profile.swept_y(post_cell)
        margin = max(0.0, float(getattr(rel, "margin", 0.0) or 0.0))
        if pre_range.max + margin > post_range.min + GEOM_EPS:
            return False
    return True


def _partition_case_to_poses(
    problem: LayoutProblem,
    case: PartitionCase,
    surface: Any,
    *,
    static_yaw_by_object: dict[str, float | None],
    base_quaternion_by_object: dict[str, tuple[float, float, float, float]],
) -> dict[str, Pose]:
    out: dict[str, Pose] = {}
    for name, cell in case.cells.items():
        yaw = static_yaw_by_object.get(name)
        out[name] = _domain_center_pose(
            problem,
            name,
            surface,
            cell.x,
            cell.y,
            yaw,
            base_quaternion_by_object[name],
        )
    return out


def _static_line_pose_cases(
    problem: LayoutProblem,
    *,
    names: list[str],
    surface: Any,
    center_bounds_by_object: dict[str, Bounds2D],
    footprint_bounds_by_object: dict[str, Bounds2D],
    static_yaw_by_object: dict[str, float | None],
    base_quaternion_by_object: dict[str, tuple[float, float, float, float]],
    relation: RegionSampleSpec | ScatterGroupSpec,
    relative_directions: tuple[Any, ...],
) -> list[dict[str, Pose]]:
    axes = _static_line_axes(names, center_bounds_by_object, footprint_bounds_by_object, relation)
    if not axes:
        return []
    min_spacing = _relation_min_spacing(relation)
    spacing_mode = _spacing_mode(relation)
    cases: list[dict[str, Pose]] = []
    for axis in axes:
        for ordered in _line_orders(names, axis=axis, relation=relation, relative_directions=relative_directions):
            centers = _pack_line_centers(
                ordered,
                axis=axis,
                center_bounds_by_object=center_bounds_by_object,
                footprint_bounds_by_object=footprint_bounds_by_object,
                min_spacing=min_spacing,
                spacing_mode=spacing_mode,
            )
            if centers is None:
                continue
            poses: dict[str, Pose] = {}
            cross_axis = "y" if axis == "x" else "x"
            common_cross = _common_axis_range(center_bounds_by_object, ordered, cross_axis)
            for name in ordered:
                x = (
                    centers[name]
                    if axis == "x"
                    else _cross_center(center_bounds_by_object[name], cross_axis, common_cross)
                )
                y = (
                    centers[name]
                    if axis == "y"
                    else _cross_center(center_bounds_by_object[name], cross_axis, common_cross)
                )
                yaw = static_yaw_by_object.get(name)
                poses[name] = _domain_center_pose(
                    problem,
                    name,
                    surface,
                    (x, x),
                    (y, y),
                    yaw,
                    base_quaternion_by_object[name],
                )
            if not _poses_pairwise_safe(problem, poses, surface, min_spacing=min_spacing, spacing_mode=spacing_mode):
                continue
            if relative_directions and not _poses_satisfy_relative(problem, poses, surface, relative_directions):
                continue
            cases.append(poses)
            if len(cases) >= MAX_STATIC_PARTITION_CASES:
                return cases
    return cases


def _static_line_axes(
    names: list[str],
    center_bounds_by_object: dict[str, Bounds2D],
    footprint_bounds_by_object: dict[str, Bounds2D],
    relation: RegionSampleSpec | ScatterGroupSpec,
) -> list[str]:
    align_axis = getattr(relation, "align_axis", None) if isinstance(relation, RegionSampleSpec) else None
    if align_axis == "x":
        return ["y"]
    if align_axis == "y":
        return ["x"]
    if len(names) < 2:
        return []
    if len(names) <= 4:
        return ["x", "y"]
    axes: list[str] = []
    for axis in ("x", "y"):
        cross = "y" if axis == "x" else "x"
        common_cross = _common_axis_range(center_bounds_by_object, names, cross)
        if common_cross is None:
            continue
        cross_span = max(_axis_length(footprint_bounds_by_object[name], cross) for name in names)
        if common_cross.length <= max(1e-9, cross_span):
            axes.append(axis)
    return axes


def _line_orders(
    names: list[str],
    *,
    axis: str,
    relation: RegionSampleSpec | ScatterGroupSpec,
    relative_directions: tuple[Any, ...],
) -> list[list[str]]:
    order = list(names)
    if isinstance(relation, RegionSampleSpec) and relation.order_axis == axis:
        return [order]
    for rel in relative_directions:
        parsed = relation_axis_and_order(rel)
        if parsed is None:
            continue
        rel_axis, pre, post = parsed
        if rel_axis == axis and pre in order and post in order:
            order = sorted(order, key=lambda name: (name == post, name))
            break
    reversed_order = list(reversed(order))
    return [order] if reversed_order == order else [order, reversed_order]


def _pack_line_centers(
    names: list[str],
    *,
    axis: str,
    center_bounds_by_object: dict[str, Bounds2D],
    footprint_bounds_by_object: dict[str, Bounds2D],
    min_spacing: float,
    spacing_mode: str,
) -> dict[str, float] | None:
    centers: dict[str, float] = {}
    previous_max: float | None = None
    for name in names:
        center_axis = _axis_range(center_bounds_by_object[name], axis)
        footprint_axis = _axis_range(footprint_bounds_by_object[name], axis)
        if previous_max is None:
            value = center_axis.min
        elif spacing_mode == "center_distance":
            value = max(center_axis.min, previous_max + min_spacing)
        else:
            value = max(center_axis.min, previous_max + min_spacing - footprint_axis.min)
        if value > center_axis.max + GEOM_EPS:
            return None
        centers[name] = value
        previous_max = value if spacing_mode == "center_distance" else value + footprint_axis.max
    slack = min(_axis_range(center_bounds_by_object[name], axis).max - centers[name] for name in names)
    if slack > GEOM_EPS:
        shift = slack * 0.5
        centers = {name: value + shift for name, value in centers.items()}
    return centers


def _poses_pairwise_safe(
    problem: LayoutProblem,
    poses: dict[str, Pose],
    surface: Any,
    *,
    min_spacing: float,
    spacing_mode: str = "footprint_gap",
) -> bool:
    ranges = {
        name: footprint_ranges_on_surface(record=problem.objects[name].asset, pose=pose, surface=surface)
        for name, pose in poses.items()
    }
    names = sorted(poses)
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            if spacing_mode == "center_distance":
                lx, ly = surface.world_to_local_xy(poses[left].position)
                rx, ry = surface.world_to_local_xy(poses[right].position)
                if math.hypot(float(rx) - float(lx), float(ry) - float(ly)) < min_spacing - GEOM_EPS:
                    return False
                continue
            if not footprint_ranges_separated(ranges[left], ranges[right], min_gap=min_spacing):
                return False
    return True


def _poses_satisfy_relative(
    problem: LayoutProblem,
    poses: dict[str, Pose],
    surface: Any,
    relative_directions: tuple[Any, ...],
) -> bool:
    pseudo = PartitionCase(
        cells={name: _pose_point_cell(name, pose, surface) for name, pose in poses.items()},
        tier="static_pose",
        topology="static_pose",
    )
    profiles = {
        name: ObjectProfile(
            name=name,
            footprint=(0.0, 0.0),
            footprint_bounds=projected_footprint_bounds(
                problem.objects[name].asset.local_bounds, pose.quaternion, surface.x_axis, surface.y_axis
            ),
            z=(pose.position[2], pose.position[2]),
        )
        for name, pose in poses.items()
    }
    bounds = surface.effective_bounds() or Bounds2D(Range1D(0.0, 0.0), Range1D(0.0, 0.0))
    partition_problem = PartitionProblem(names=tuple(poses), bounds=bounds, profiles=profiles)
    return _static_case_satisfies_relative_directions(partition_problem, pseudo, relative_directions)


def _pose_point_cell(name: str, pose: Pose, surface: Any):
    from ...common.partitioning import PartitionCell

    x, y = surface.world_to_local_xy(pose.position)
    return PartitionCell(
        name, Range1D(float(x), float(x)), Range1D(float(y), float(y)), (pose.position[2], pose.position[2])
    )


def _common_axis_range(bounds_by_object: dict[str, Bounds2D], names: list[str], axis: str) -> Range1D | None:
    out: Range1D | None = None
    for name in names:
        item = _axis_range(bounds_by_object[name], axis)
        out = item if out is None else out.intersect(item)
        if out is None:
            return None
    return out


def _axis_range(bounds: Bounds2D, axis: str) -> Range1D:
    return bounds.x if axis == "x" else bounds.y


def _axis_length(bounds: Bounds2D, axis: str) -> float:
    return _axis_range(bounds, axis).length


def _cross_center(bounds: Bounds2D, axis: str, common: Range1D | None) -> float:
    return float(common.center if common is not None else _axis_range(bounds, axis).center)


def _domain_center_pose(
    problem: LayoutProblem,
    name: str,
    surface: Any,
    x_range: tuple[float, float] | Range1D,
    y_range: tuple[float, float] | Range1D,
    yaw: float | None,
    base_quaternion: tuple[float, float, float, float],
) -> Pose:
    quat = quat_mul(yaw_quat(float(yaw)), base_quaternion) if yaw is not None else object_quaternion(problem, name)
    z_offset = support_contact_offset(
        problem.objects[name].asset.local_bounds,
        quat,
        surface.normal,
        contact_offset=DEFAULT_SUPPORT_CONTACT_OFFSET_M,
    )
    return Pose(
        position=surface.local_to_world(_range_center(x_range), _range_center(y_range), z_offset),
        quaternion=quat,
    )


def _range_center(axis: tuple[float, float] | Range1D) -> float:
    if isinstance(axis, Range1D):
        return float(axis.center)
    return (float(axis[0]) + float(axis[1])) * 0.5


def _clip_to_center_bounds(center_bounds_by_object: dict[str, Bounds2D]):
    def clip(name: str, x: Range1D, y: Range1D) -> tuple[Range1D, Range1D] | None:
        bounds = center_bounds_by_object.get(name)
        if bounds is None:
            return None
        clipped_x = x.intersect(bounds.x)
        clipped_y = y.intersect(bounds.y)
        if clipped_x is None or clipped_y is None:
            return None
        return clipped_x, clipped_y

    return clip


def _surface_needs_cell_filter(surface: Any) -> bool:
    return getattr(surface, "polygon", None) is not None or str(
        getattr(surface, "shape", "rectangle") or "rectangle"
    ).lower() not in {"", "rectangle"}


def _footprint_area(problem: LayoutProblem, name: str) -> float:
    size = problem.objects[name].asset.local_bounds.size
    return max(0.0, float(size[0])) * max(0.0, float(size[1]))


def _surface_hole_obstacles(surface: Any) -> tuple[PartitionObstacle, ...]:
    polygon = getattr(surface, "polygon", None)
    holes = getattr(polygon, "holes", ()) if polygon is not None else ()
    out: list[PartitionObstacle] = []
    for index, hole in enumerate(holes or ()):
        xs = [float(point[0]) for point in hole]
        ys = [float(point[1]) for point in hole]
        out.append(
            PartitionObstacle(
                obstacle_id=f"{getattr(surface, 'surface_id', 'surface')}:hole_{index}",
                x=Range1D(min(xs), max(xs)),
                y=Range1D(min(ys), max(ys)),
                source_object="support_surface_hole",
                source_domain=str(getattr(surface, "surface_id", "")),
            )
        )
    return tuple(out)
