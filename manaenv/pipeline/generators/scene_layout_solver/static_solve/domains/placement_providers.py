from __future__ import annotations

import random
from typing import Any

from ...common.constants import DEFAULT_SUPPORT_CONTACT_OFFSET_M
from ...common.diagnostics import LayoutDiagnostic
from ...common.geometry import Bounds2D, Pose, projected_footprint_bounds, quat_mul, support_contact_offset, yaw_quat
from ...common.reproducibility import stable_rng
from ...common.volumes import volume_parent
from ...common.weighted import weighted_choice
from ...scene_spec.schema import (
    InVolumeSpec,
    PoseSampleSpec,
    RegionSampleSpec,
    ScatterGroupSpec,
    SurfaceRangeSpec,
    VolumeGroupSpec,
)
from ..problem import LayoutProblem
from .base import Domain, DomainContext, DomainProvider, StaticGroupDomain, StaticNodeDomain
from .group_sampling import (
    GroupDomainBuilder,
    grid_candidates,
    poisson_disk_candidates,
    random_candidates,
    scatter_grid_step,
    stratified_candidates,
)
from .joint_partition import build_static_joint_cases, build_static_joint_cases_from_candidates
from .providers import (
    InVolumeProvider,
    PoseSampleProvider,
    SupportOnProvider,
    object_base_quaternion,
    object_scene_yaw_range,
)


class SupportOnDomainProvider(DomainProvider):
    def __init__(self, problem: LayoutProblem, object_name: str) -> None:
        self.problem = problem
        self.object_name = str(object_name)

    def build_domain(self, ctx: DomainContext) -> Domain:
        candidates, meta = support_candidates_for_object(
            self.problem,
            self.object_name,
            group_surface=None,
            group_surface_range=None,
            relation_overrides=ctx.relation_overrides,
            assignment=ctx.assignment,
            rng=ctx.rng,
            max_candidates=ctx.budget,
        )
        return StaticNodeDomain(candidates=candidates, diagnostics=meta)

    def cache_assignment_nodes(self) -> tuple[str, ...]:
        obj = self.problem.objects.get(self.object_name)
        parent = self.problem.local_surface_parent(obj.support) if obj is not None and obj.support else None
        return (parent,) if parent else ()


class PoseSampleDomainProvider(DomainProvider):
    def __init__(self, problem: LayoutProblem, sample: PoseSampleSpec) -> None:
        self.problem = problem
        self.sample = sample

    def build_domain(self, ctx: DomainContext) -> Domain:
        provider = PoseSampleProvider(self.problem)
        poses = provider.candidate_poses(self.sample, rng=ctx.rng)
        if isinstance(poses, LayoutDiagnostic):
            return StaticNodeDomain([], diagnostics={"provider": "pose_sample", "reason": poses.code})
        return StaticNodeDomain(list(poses), diagnostics={"provider": "pose_sample", "candidate_count": len(poses)})


class InVolumeDomainProvider(DomainProvider):
    def __init__(self, problem: LayoutProblem, relation: InVolumeSpec) -> None:
        self.problem = problem
        self.relation = relation

    def build_domain(self, ctx: DomainContext) -> Domain:
        provider = InVolumeProvider(self.problem)
        poses = provider.candidate_poses(
            self.relation, assignment=ctx.assignment, max_candidates=ctx.budget, rng=ctx.rng
        )
        if isinstance(poses, LayoutDiagnostic):
            return StaticNodeDomain([], diagnostics={"provider": "in_volume", "reason": poses.code})
        if len(poses) > ctx.budget:
            poses = list(poses)[: ctx.budget]
        return StaticNodeDomain(list(poses), diagnostics={"provider": "in_volume", "candidate_count": len(poses)})

    def cache_assignment_nodes(self) -> tuple[str, ...]:
        volume = self.problem.spec.volumes_by_id.get(self.relation.volume)
        parent = volume_parent(volume) if volume is not None else None
        return (parent,) if parent else ()


class VolumeGroupDomainProvider(DomainProvider):
    def __init__(
        self,
        problem: LayoutProblem,
        *,
        group_id: str,
        members: list[str],
        relation: VolumeGroupSpec,
    ) -> None:
        self.problem = problem
        self.group_id = str(group_id)
        self.members = list(members)
        self.relation = relation

    def build_domain(self, ctx: DomainContext) -> Domain:
        provider = InVolumeProvider(self.problem)
        member_candidates: dict[str, list[Pose]] = {}
        diagnostics: dict[str, Any] = {
            "provider": "volume_group",
            "volume": self.relation.volume,
            "mode": self.relation.mode,
        }
        for name in self.members:
            poses = provider.candidate_poses(
                InVolumeSpec(object_name=name, volume=self.relation.volume, mode="fit"),
                assignment=ctx.assignment,
                max_candidates=ctx.budget,
                rng=ctx.rng,
                yaw_range=self.relation.yaw,
            )
            if isinstance(poses, LayoutDiagnostic):
                diagnostics.setdefault("member_failures", {})[name] = poses.to_json()
                member_candidates[name] = []
                continue
            member_candidates[name] = list(poses)[: ctx.budget]
        diagnostics["candidate_counts"] = {name: len(cands) for name, cands in member_candidates.items()}
        return StaticGroupDomain(member_candidates=member_candidates, order=list(self.members), diagnostics=diagnostics)

    def cache_assignment_nodes(self) -> tuple[str, ...]:
        volume = self.problem.spec.volumes_by_id.get(self.relation.volume)
        parent = volume_parent(volume) if volume is not None else None
        return (parent,) if parent else ()


class TabletopGroupDomainProvider(DomainProvider):
    def __init__(
        self,
        problem: LayoutProblem,
        *,
        group_id: str,
        members: list[str],
        surface: str,
        relation: RegionSampleSpec | ScatterGroupSpec,
        relation_kind: str,
    ) -> None:
        self.problem = problem
        self.group_id = str(group_id)
        self.members = list(members)
        self.surface = str(surface)
        self.relation = relation
        self.relation_kind = str(relation_kind)

    def build_domain(self, ctx: DomainContext) -> Domain:
        group_surface_range = _relation_surface_range(self.relation, rng=ctx.rng)
        joint = build_static_joint_cases(
            self.problem,
            relation=self.relation,
            relation_kind=self.relation_kind,
            members=self.members,
            surface_id=self.surface,
            relation_overrides=ctx.relation_overrides,
            assignment=ctx.assignment,
            rng=ctx.rng,
            budget=ctx.budget,
            group_surface_range=group_surface_range,
        )
        member_candidates, meta = group_candidates_for_relation(
            self.problem,
            relation=self.relation,
            relation_kind=self.relation_kind,
            members=self.members,
            surface_id=self.surface,
            relation_overrides=ctx.relation_overrides,
            assignment=ctx.assignment,
            rng=ctx.rng,
            budget=ctx.budget,
            group_surface_range=group_surface_range,
        )
        if joint.meta:
            meta = {**meta, **joint.meta}
        if not joint.cases and all(member_candidates.get(name) for name in self.members):
            candidate_joint = build_static_joint_cases_from_candidates(
                self.problem,
                members=self.members,
                surface_id=self.surface,
                relation=self.relation,
                member_candidates=member_candidates,
                assignment=ctx.assignment,
                rng=ctx.rng,
                budget=ctx.budget,
            )
            if candidate_joint.cases:
                joint = candidate_joint
            if candidate_joint.meta:
                meta = {**meta, **candidate_joint.meta}
        order = list(self.members)
        order_axis = getattr(self.relation, "order_axis", None)
        if order_axis in {"x", "y"}:
            axis_idx = 0 if order_axis == "x" else 1
            order = sorted(
                self.members,
                key=lambda name: (
                    _candidate_axis_mean(member_candidates.get(name, []), axis_idx),
                    name,
                ),
            )
        return StaticGroupDomain(
            member_candidates=member_candidates, order=order, diagnostics=meta, joint_cases=joint.cases
        )

    def cache_assignment_nodes(self) -> tuple[str, ...]:
        parent = self.problem.local_surface_parent(self.surface)
        return (parent,) if parent else ()


def _candidate_axis_mean(candidates: list[Pose], axis_idx: int) -> float:
    if not candidates:
        return 0.0
    return sum(float(pose.position[axis_idx]) for pose in candidates) / float(len(candidates))


def support_candidates_for_object(
    problem: LayoutProblem,
    object_name: str,
    *,
    group_surface: str | None,
    group_surface_range: SurfaceRangeSpec | None,
    relation_overrides: dict[str, Any],
    assignment: dict[str, Pose] | None,
    rng: random.Random,
    max_candidates: int,
) -> tuple[list[Pose], dict[str, Any]]:
    provider = SupportOnProvider(problem, relation_overrides=relation_overrides, assignment=assignment, rng=rng)
    domain = provider.domain(
        object_name,
        surface_override=group_surface,
        group_surface_range=group_surface_range,
        target_min_candidates=min(32, max(8, int(max_candidates))),
        rng=rng,
    )
    if domain.issues:
        return [], dict(domain.meta)
    candidates = [placement.pose for placement in domain.candidates]
    rng.shuffle(candidates)
    return candidates[: max(1, int(max_candidates))], {**dict(domain.meta), "candidate_count": len(candidates)}


def group_candidates_for_relation(
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
) -> tuple[dict[str, list[Pose]], dict[str, Any]]:
    surface = problem.resolve_surface(surface_id, assignment or {})
    if surface is None:
        reason = "local_surface_parent_unplaced" if problem.local_surface_parent(surface_id) else "surface_not_found"
        return {name: [] for name in members}, {"reason": reason, "surface": surface_id}
    if group_surface_range is None:
        group_surface_range = _relation_surface_range(relation, rng=rng)
    provider = SupportOnProvider(problem, relation_overrides=relation_overrides, assignment=assignment, rng=rng)
    yaw_range = _relation_yaw_ranges(problem, relation, members)

    def range_for_object(name: str, region_range: SurfaceRangeSpec | None) -> Bounds2D | None:
        bounds, _ = provider.center_bounds(
            name,
            surface_override=surface_id,
            group_surface_range=region_range,
            yaw_range=_yaw_range_for_object(yaw_range, name),
        )
        return bounds

    def fit_bounds_for_object(bounds: Bounds2D, name: str) -> Bounds2D | None:
        return bounds

    def footprint_size(name: str) -> tuple[float, float]:
        return provider.footprint_size(name, surface, yaw_range=_yaw_range_for_object(yaw_range, name))

    domains = GroupDomainBuilder(
        range_for_object=range_for_object,
        fit_bounds_for_object=fit_bounds_for_object,
        footprint_size=footprint_size,
    ).build(
        names=members,
        group_surface_range=group_surface_range,
        target=getattr(relation, "region_id", getattr(relation, "group_id", relation_kind)),
        relation=relation_kind,
    )
    if isinstance(domains, LayoutDiagnostic):
        return {name: [] for name in members}, {"reason": domains.code, **dict(domains.meta)}

    def candidate_quaternion(name: str, yaw: float | None):
        base = object_base_quaternion(problem, name)
        if yaw is None:
            return base
        return quat_mul(yaw_quat(float(yaw)), base)

    def valid_placement(item) -> bool:
        quat = candidate_quaternion(item.name, item.yaw)
        footprint = projected_footprint_bounds(
            problem.objects[item.name].asset.local_bounds, quat, surface.x_axis, surface.y_axis
        )
        return surface.supports_local_footprint_bounds(item.x, item.y, footprint)

    candidate_min_spacing = (
        0.0
        if str(getattr(relation, "spacing_mode", "footprint_gap") or "footprint_gap") == "center_distance"
        else float(getattr(relation, "min_spacing", 0.0) or 0.0)
    )

    if isinstance(relation, RegionSampleSpec):
        method = str(relation.sample_method or "random").lower()
        if method in {"grid", "jittered_grid"} or relation.align_axis:
            candidate_set = grid_candidates(
                domains=domains,
                rng=rng,
                valid_placement=valid_placement,
                scope_id=f"region:{relation.region_id}",
                packer="region_sample",
                grid_step=relation.grid_step,
                grid_snap=relation.grid_snap,
                align_axis=relation.align_axis,
                jitter_ratio=relation.jitter if method in {"jittered_grid", "stratified"} else 0.0,
                max_candidates_per_object=max(32, min(1024, int(relation.max_attempts))),
                target_min_candidates=min(32, max(8, int(budget))),
                yaw_range=yaw_range,
            )
        elif method == "stratified":
            candidate_set = stratified_candidates(
                domains=domains,
                rng=rng,
                valid_placement=valid_placement,
                scope_id=f"region:{relation.region_id}",
                packer="region_stratified",
                jitter_ratio=relation.jitter if relation.jitter > 0.0 else 0.5,
                max_candidates_per_object=max(32, min(1024, int(relation.max_attempts))),
                target_min_candidates=min(64, max(16, int(budget))),
                yaw_range=yaw_range,
            )
        elif method in {"poisson", "poisson_disk"}:
            candidate_set = poisson_disk_candidates(
                domains=domains,
                rng=rng,
                attempts=max(int(relation.max_attempts), int(budget) * 8),
                valid_placement=valid_placement,
                scope_id=f"region:{relation.region_id}",
                packer="region_poisson_disk",
                min_spacing=candidate_min_spacing,
                max_candidates_per_object=max(32, min(1024, int(relation.max_attempts))),
                yaw_range=yaw_range,
            )
        else:
            candidate_set = random_candidates(
                domains=domains,
                rng=rng,
                attempts=max(int(relation.max_attempts), int(budget)),
                valid_placement=valid_placement,
                scope_id=f"region:{relation.region_id}",
                packer=f"region_{method}",
                yaw_range=yaw_range,
            )
    else:
        mode = str(relation.mode or "jittered_grid").lower()
        local_rng = (
            stable_rng(int(relation.seed), "scatter_group", relation.group_id) if relation.seed is not None else rng
        )
        if mode in {"grid", "jittered_grid"}:
            candidate_set = grid_candidates(
                domains=domains,
                rng=local_rng,
                valid_placement=valid_placement,
                scope_id=f"scatter:{relation.group_id}",
                packer=f"scatter_{mode}",
                grid_step=scatter_grid_step(domains=domains, count=len(members), min_spacing=candidate_min_spacing),
                jitter_ratio=relation.jitter if mode == "jittered_grid" else 0.0,
                max_candidates_per_object=max(32, min(1024, int(relation.max_attempts))),
                target_min_candidates=min(32, max(8, int(budget))),
                yaw_range=yaw_range,
            )
        else:
            candidate_set = poisson_disk_candidates(
                domains=domains,
                rng=local_rng,
                attempts=max(int(relation.max_attempts), int(budget) * 8),
                valid_placement=valid_placement,
                scope_id=f"scatter:{relation.group_id}",
                packer="scatter_poisson_disk",
                min_spacing=candidate_min_spacing,
                max_candidates_per_object=max(32, min(1024, int(relation.max_attempts))),
                yaw_range=yaw_range,
            )
    if isinstance(candidate_set, LayoutDiagnostic):
        return {name: [] for name in members}, {"reason": candidate_set.code, **dict(candidate_set.meta)}

    out: dict[str, list[Pose]] = {}
    for name, local_candidates in candidate_set.candidates_by_name.items():
        out[name] = [
            Pose(
                position=surface.local_to_world(
                    item.x,
                    item.y,
                    support_contact_offset(
                        problem.objects[name].asset.local_bounds,
                        candidate_quaternion(name, item.yaw),
                        surface.normal,
                        contact_offset=DEFAULT_SUPPORT_CONTACT_OFFSET_M,
                    ),
                ),
                quaternion=candidate_quaternion(name, item.yaw),
            )
            for item in local_candidates[: max(1, int(budget))]
        ]
    return out, dict(candidate_set.meta)


def _relation_surface_range(
    relation: RegionSampleSpec | ScatterGroupSpec,
    *,
    rng: random.Random,
) -> SurfaceRangeSpec | None:
    choices = tuple(getattr(relation, "surface_ranges", ()) or ())
    if choices:
        selected = weighted_choice(choices, rng=rng, weight_fn=lambda item: item.weight)
        return selected.surface_range if selected is not None else None
    return getattr(relation, "surface_range", None)


def _relation_yaw_ranges(
    problem: LayoutProblem,
    relation: RegionSampleSpec | ScatterGroupSpec,
    members: list[str],
) -> tuple[float, float] | dict[str, tuple[float, float]] | None:
    relation_yaw = getattr(relation, "yaw", None)
    by_name = {
        name: (float(yaw[0]), float(yaw[1]))
        for name in members
        if (yaw := object_scene_yaw_range(problem, name)) is not None
    }
    if relation_yaw is not None:
        fallback = (float(relation_yaw[0]), float(relation_yaw[1]))
        return {name: by_name.get(name, fallback) for name in members}
    return by_name or None


def _yaw_range_for_object(
    yaw_range: tuple[float, float] | dict[str, tuple[float, float]] | None,
    name: str,
) -> tuple[float, float] | None:
    if yaw_range is None:
        return None
    if isinstance(yaw_range, dict):
        return yaw_range.get(name)
    return yaw_range
