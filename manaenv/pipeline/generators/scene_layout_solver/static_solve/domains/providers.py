from __future__ import annotations

import random
from typing import Any, Protocol

from ...common.constants import DEFAULT_SUPPORT_CONTACT_OFFSET_M
from ...common.diagnostics import LayoutDiagnostic
from ...common.geometry import (
    Bounds2D,
    Bounds3D,
    ManualSupportSurface,
    Pose,
    Range1D,
    projected_footprint_bounds,
    quat_mul,
    support_contact_offset,
    yaw_quat,
)
from ...common.object_pose import object_base_quaternion_from_record, object_scene_yaw_range_from_record
from ...common.placement_domains import (
    apply_surface_range,
    fixed_yaw_range,
    support_center_domain,
    support_footprint_bounds,
    support_local_fits,
    union_bounds,
)
from ...common.reproducibility import stable_rng
from ...common.volumes import transform_volume_pose, volume_candidate_offsets, volume_fit_bounds, volume_parent
from ...common.weighted import weighted_choice, weighted_order
from ...scene_spec.schema import (
    InVolumeSpec,
    PoseSampleSpec,
    RelativeDirectionSpec,
    SurfaceRangeChoiceSpec,
    SurfaceRangeSpec,
)
from ..problem import LayoutProblem, ProblemObject
from .sampling import CandidateDomain, CandidatePlacement, sample_support_points


class ObjectDomainProvider(Protocol):
    name: str

    def precheck(self, object_name: str) -> list[LayoutDiagnostic]:
        raise NotImplementedError


class GroupPackerProvider(Protocol):
    name: str


class ConstraintFactor(Protocol):
    name: str


def object_quaternion(problem: LayoutProblem, object_name: str):
    return object_base_quaternion(problem, object_name)


def object_base_quaternion(problem: LayoutProblem, object_name: str):
    return object_base_quaternion_from_record(problem.objects[object_name])


def object_scene_yaw_range(problem: LayoutProblem, object_name: str) -> tuple[float, float] | None:
    yaw = object_scene_yaw_range_from_record(problem.objects[object_name])
    if yaw is not None:
        return yaw
    choices = object_scene_yaw_choices(problem, object_name)
    if not choices:
        return None
    yaws = [float(choice.yaw) for choice in choices if float(choice.weight) > 0.0]
    return (min(yaws), max(yaws)) if yaws else None


def object_scene_yaw_choices(problem: LayoutProblem, object_name: str) -> tuple[Any, ...]:
    return tuple(getattr(problem.objects[object_name].spec, "yaw_samples", ()) or ())


class SupportOnProvider:
    name = "support_on"

    def __init__(
        self,
        problem: LayoutProblem,
        *,
        relation_overrides: dict | None = None,
        assignment: dict[str, Pose] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.problem = problem
        self.relation_overrides = dict(relation_overrides or {})
        self.assignment = dict(assignment or {})
        self.rng = rng
        self._object_surface_range_choices: dict[str, SurfaceRangeChoiceSpec] = {}

    def effective_surface_range(self, surface_range: SurfaceRangeSpec | None) -> SurfaceRangeSpec | None:
        if surface_range is None:
            return None
        cfg = self.relation_overrides.get("support_on", {}) if isinstance(self.relation_overrides, dict) else {}
        if not isinstance(cfg, dict):
            return surface_range
        tier = str(cfg.get("surface_range_tier") or "strict")
        expand_m = float(cfg.get("surface_range_expand_m", 0.0) or 0.0)
        return surface_range.relaxed(tier=tier, expand_m=expand_m)

    def apply_surface_range(self, bounds: Bounds2D, surface_range: SurfaceRangeSpec | None) -> Bounds2D | None:
        return apply_surface_range(bounds, self.effective_surface_range(surface_range))

    def footprint_size(
        self,
        object_name: str,
        surface: ManualSupportSurface | None = None,
        *,
        yaw_range: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        footprint = self.footprint_bounds(object_name, surface, yaw_range=yaw_range)
        return (footprint.x.length, footprint.y.length)

    def footprint_bounds(
        self,
        object_name: str,
        surface: ManualSupportSurface | None = None,
        *,
        yaw_range: tuple[float, float] | None = None,
    ) -> Bounds2D:
        record = self.problem.objects[object_name].asset
        if surface is None:
            return projected_footprint_bounds(
                record.local_bounds,
                object_quaternion(self.problem, object_name),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        base_quat = (
            object_base_quaternion(self.problem, object_name)
            if yaw_range is not None
            else object_quaternion(self.problem, object_name)
        )
        return support_footprint_bounds(record, base_quat, surface, yaw_range=yaw_range)

    def object_surface_range(self, obj: ProblemObject) -> SurfaceRangeSpec | None:
        if obj.override_surface_range is not None:
            return obj.override_surface_range
        if obj.surface_ranges:
            choice = self.object_surface_range_choice(obj)
            return choice.surface_range if choice is not None else None
        return obj.surface_range

    def object_surface_range_choice(self, obj: ProblemObject) -> SurfaceRangeChoiceSpec | None:
        if not obj.surface_ranges:
            return None
        cached = self._object_surface_range_choices.get(obj.name)
        if cached is not None:
            return cached
        local_rng = self.rng or stable_rng(self.problem.spec.seed, "object_surface_range", obj.name)
        choice = weighted_choice(obj.surface_ranges, rng=local_rng, weight_fn=lambda item: item.weight)
        if choice is not None:
            self._object_surface_range_choices[obj.name] = choice
        return choice

    def center_bounds(
        self,
        object_name: str,
        *,
        surface_override: str | None = None,
        group_surface_range: SurfaceRangeSpec | None = None,
        yaw_range: tuple[float, float] | None = None,
    ) -> tuple[Bounds2D | None, dict]:
        obj = self.problem.objects[object_name]
        surface_id = surface_override or obj.support
        if not surface_id:
            return None, {"reason": "object_support_missing"}
        surface = self.surface(surface_id)
        if surface is None:
            reason = (
                "local_surface_parent_unplaced"
                if self.problem.local_surface_parent(surface_id)
                else "surface_not_found"
            )
            return None, {"reason": reason, "surface": surface_id}
        effective_yaw = yaw_range if yaw_range is not None else object_scene_yaw_range(self.problem, object_name)
        ranges = (
            self.effective_surface_range(group_surface_range),
            self.effective_surface_range(self.object_surface_range(obj)),
        )
        domain, meta = support_center_domain(
            object_name=object_name,
            record=obj.asset,
            surface=surface,
            base_quaternion=object_base_quaternion(self.problem, object_name),
            surface_ranges=ranges,
            yaw_range=effective_yaw,
        )
        if domain is None:
            return None, meta
        return domain.bounds, meta

    def supports_local(
        self,
        object_name: str,
        x: float,
        y: float,
        *,
        surface_override: str | None = None,
        yaw_range: tuple[float, float] | None = None,
    ) -> bool:
        obj = self.problem.objects[object_name]
        surface_id = surface_override or obj.support
        if not surface_id:
            return False
        surface = self.surface(surface_id)
        if surface is None:
            return False
        effective_yaw = yaw_range if yaw_range is not None else object_scene_yaw_range(self.problem, object_name)
        return support_local_fits(
            record=obj.asset,
            surface=surface,
            base_quaternion=object_base_quaternion(self.problem, object_name),
            x=x,
            y=y,
            yaw_range=effective_yaw,
        )

    def surface(self, surface_id: str | None) -> ManualSupportSurface | None:
        if not surface_id:
            return None
        return self.problem.resolve_surface(str(surface_id), self.assignment)

    def domain(
        self,
        object_name: str,
        *,
        surface_override: str | None = None,
        group_surface_range: SurfaceRangeSpec | None = None,
        target_min_candidates: int = 32,
        rng: random.Random | None = None,
    ) -> CandidateDomain:
        obj = self.problem.objects[object_name]
        yaw_range = object_scene_yaw_range(self.problem, object_name)
        surface_id = surface_override or obj.support
        if not surface_id:
            return CandidateDomain(
                object_name=object_name,
                surface=surface_id,
                issues=("object_support_missing",),
                meta={"provider": self.name, "reason": "object_support_missing"},
            )
        surface = self.surface(str(surface_id))
        if surface is None:
            reason = (
                "local_surface_parent_unplaced"
                if self.problem.local_surface_parent(str(surface_id))
                else "surface_not_found"
            )
            return CandidateDomain(
                object_name=object_name,
                surface=surface_id,
                issues=(reason,),
                meta={"provider": self.name, "reason": reason, "surface": surface_id},
            )
        base_quat = object_base_quaternion(self.problem, object_name)
        local_rng = rng or random.Random(0)
        yaw_values = _yaw_sample_values(
            yaw_range,
            local_rng,
            yaw_choices=object_scene_yaw_choices(self.problem, object_name),
        )
        per_yaw_target = max(1, int(target_min_candidates) // max(1, len(yaw_values)))
        candidates: list[CandidatePlacement] = []
        bounds_by_yaw: list[Bounds2D] = []
        last_meta: dict = {"provider": self.name}
        last_step: float | None = None
        for static_yaw in yaw_values:
            static_yaw_range = fixed_yaw_range(static_yaw)
            bounds, meta = self.center_bounds(
                object_name,
                surface_override=surface_override,
                group_surface_range=group_surface_range,
                yaw_range=static_yaw_range,
            )
            last_meta = {**last_meta, **meta}
            if bounds is None:
                continue
            bounds_by_yaw.append(bounds)
            footprint = self.footprint_size(object_name, surface, yaw_range=static_yaw_range)
            sample = sample_support_points(
                bounds,
                footprint=footprint,
                accept_fn=lambda x, y, yr=static_yaw_range: self.supports_local(
                    object_name, x, y, surface_override=surface_override, yaw_range=yr
                ),
                target_min_candidates=per_yaw_target,
            )
            last_step = sample.step
            last_meta = {**last_meta, **sample.meta}
            for x, y in sample.points:
                quat = (
                    quat_mul(yaw_quat(static_yaw), base_quat)
                    if static_yaw is not None
                    else object_quaternion(self.problem, object_name)
                )
                actual_footprint = projected_footprint_bounds(
                    obj.asset.local_bounds, quat, surface.x_axis, surface.y_axis
                )
                if not surface.supports_local_footprint_bounds(float(x), float(y), actual_footprint):
                    continue
                pose_z_offset = support_contact_offset(
                    obj.asset.local_bounds,
                    quat,
                    surface.normal,
                    contact_offset=DEFAULT_SUPPORT_CONTACT_OFFSET_M,
                )
                candidates.append(
                    CandidatePlacement(
                        object_name=object_name,
                        surface=str(surface_id),
                        x=float(x),
                        y=float(y),
                        pose=Pose(position=surface.local_to_world(x, y, pose_z_offset), quaternion=quat),
                        provenance={"source": "support_on_provider", "step": sample.step, "yaw": static_yaw},
                    )
                )
        bounds = union_bounds(bounds_by_yaw) if bounds_by_yaw else None
        if bounds is None:
            return CandidateDomain(
                object_name=object_name,
                surface=surface_id,
                issues=(str(last_meta.get("reason") or "domain_empty"),),
                meta=last_meta,
            )
        if not candidates:
            return CandidateDomain(
                object_name=object_name,
                surface=surface_id,
                bounds=bounds,
                step=last_step,
                issues=("support_surface_boundary_empty",),
                meta={
                    **last_meta,
                    "provider": self.name,
                    "relation_overrides": dict(self.relation_overrides),
                },
            )
        return CandidateDomain(
            object_name=object_name,
            surface=surface_id,
            candidates=tuple(candidates),
            bounds=bounds,
            step=last_step,
            meta={
                **last_meta,
                "provider": self.name,
                "candidate_count": len(candidates),
                "yaw_candidate_count": len(yaw_values),
                "relation_overrides": dict(self.relation_overrides),
            },
        )

    def precheck(self, object_name: str) -> list[LayoutDiagnostic]:
        domain = self.domain(object_name)
        if not domain.issues:
            return []
        code = str(domain.issues[0])
        return [
            LayoutDiagnostic(
                code,
                code,
                target=f"object:{object_name}",
                meta={"scope": self.name, **dict(domain.meta)},
            )
        ]


def _sample_yaw(rng: random.Random, yaw_range: tuple[float, float] | None) -> float | None:
    if yaw_range is None:
        return None
    lo, hi = float(yaw_range[0]), float(yaw_range[1])
    return lo if abs(hi - lo) <= 1e-12 else rng.uniform(lo, hi)


def _yaw_fit_values(yaw_range: tuple[float, float] | None) -> tuple[float | None, ...]:
    if yaw_range is None:
        return (None,)
    lo, hi = float(yaw_range[0]), float(yaw_range[1])
    if lo > hi:
        lo, hi = hi, lo
    if abs(hi - lo) <= 1e-12:
        return (lo,)
    values = (lo, hi, (lo + hi) * 0.5, lo + (hi - lo) * 0.25, lo + (hi - lo) * 0.75)
    return _unique_yaws(values)


def _yaw_sample_values(
    yaw_range: tuple[float, float] | None,
    rng: random.Random,
    *,
    yaw_choices: tuple[Any, ...] = (),
) -> tuple[float | None, ...]:
    if yaw_choices:
        return _unique_yaws(choice.yaw for choice in yaw_choices if float(choice.weight) > 0.0)
    values = list(_yaw_fit_values(yaw_range))
    if yaw_range is not None:
        lo, hi = float(yaw_range[0]), float(yaw_range[1])
        if lo > hi:
            lo, hi = hi, lo
        if hi - lo > 1e-12:
            values.extend(rng.uniform(lo, hi) for _ in range(2))
    return _unique_yaws(values)


def _unique_yaws(values) -> tuple[float | None, ...]:
    out: list[float | None] = []
    seen: set[float | None] = set()
    for value in values:
        key = None if value is None else round(float(value), 12)
        if key in seen:
            continue
        seen.add(key)
        out.append(None if value is None else float(value))
    return tuple(out)


class PoseSampleProvider:
    name = "pose_sample"

    def __init__(self, problem: LayoutProblem) -> None:
        self.problem = problem

    def candidate_poses(
        self,
        sample: PoseSampleSpec,
        *,
        rng: random.Random | None = None,
    ) -> list[Pose] | LayoutDiagnostic:
        has_explicit_weight = any(abs(float(pose.weight) - 1.0) > 1e-12 for pose in sample.poses)
        if has_explicit_weight:
            ordered = weighted_order(sample.poses, rng=rng or random.Random(0), weight_fn=lambda pose: pose.weight)
        else:
            ordered = list(sample.poses)
        poses = [Pose(position=pose.position, quaternion=pose.quaternion) for pose in ordered if pose.weight > 0.0]
        if sample.shuffle and rng is not None and not has_explicit_weight:
            rng.shuffle(poses)
        if not poses:
            return LayoutDiagnostic(
                "pose_sample_domain_empty", "pose_sample has no candidate poses", target=sample.object_name
            )
        return poses

    def precheck(self, object_name: str) -> list[LayoutDiagnostic]:
        sample = self.problem.relation_index.pose_samples_by_object.get(object_name)
        if sample is None or sample.poses:
            return []
        return [LayoutDiagnostic("pose_sample_domain_empty", "pose_sample has no candidate poses", target=object_name)]


class InVolumeProvider:
    name = "in_volume"

    def __init__(self, problem: LayoutProblem) -> None:
        self.problem = problem

    @staticmethod
    def fit_bounds(volume: Bounds3D, child: Bounds3D, margin: float, *, mode: str) -> Bounds3D | None:
        m = max(0.0, float(margin))
        if mode == "center":
            fit = Bounds3D(
                Range1D(volume.x.min + m, volume.x.max - m),
                Range1D(volume.y.min + m, volume.y.max - m),
                Range1D(volume.z.min + m, volume.z.max - m),
            )
        else:
            fit = Bounds3D(
                Range1D(volume.x.min + m - child.x.min, volume.x.max - m - child.x.max),
                Range1D(volume.y.min + m - child.y.min, volume.y.max - m - child.y.max),
                Range1D(volume.z.min + m - child.z.min, volume.z.max - m - child.z.max),
            )
        if fit.x.min > fit.x.max or fit.y.min > fit.y.max or fit.z.min > fit.z.max:
            return None
        return fit

    def candidate_poses(
        self,
        rel: InVolumeSpec,
        *,
        assignment: dict[str, Pose] | None = None,
        max_candidates: int = 64,
        rng: random.Random | None = None,
        yaw_range: tuple[float, float] | None = None,
    ) -> list[Pose] | LayoutDiagnostic:
        obj = self.problem.objects.get(rel.object_name)
        volume = self.problem.spec.volumes_by_id.get(rel.volume)
        if obj is None:
            return LayoutDiagnostic(
                "in_volume_object_unknown", "in_volume references unknown object", target=rel.object_name
            )
        if volume is None:
            return LayoutDiagnostic(
                "in_volume_anchor_unknown", "in_volume references unknown volume", target=rel.object_name
            )
        parent = volume_parent(volume)
        parent_pose = None
        if parent:
            parent_pose = (assignment or {}).get(parent)
            if parent_pose is None:
                return LayoutDiagnostic(
                    "in_volume_parent_unplaced",
                    "object-local volume requires parent pose before child placement",
                    target=rel.object_name,
                    meta={"volume": rel.volume, "parent": parent},
                )
        fit = volume_fit_bounds(volume, obj.asset.local_bounds, mode=rel.mode)
        if fit is None:
            return LayoutDiagnostic(
                "in_volume_capacity_insufficient",
                "object does not fit in volume",
                target=rel.object_name,
                meta={"volume": rel.volume},
            )
        base_quat = object_base_quaternion(self.problem, rel.object_name)
        effective_yaw = yaw_range if yaw_range is not None else object_scene_yaw_range(self.problem, rel.object_name)
        local_rng = rng or random.Random(0)
        offsets = volume_candidate_offsets(volume, obj.asset.local_bounds, mode=rel.mode, max_candidates=max_candidates)
        poses = []
        for offset in offsets:
            static_yaw = _sample_yaw(local_rng, effective_yaw)
            quat = quat_mul(yaw_quat(static_yaw), base_quat) if static_yaw is not None else base_quat
            poses.append(transform_volume_pose(volume, offset, quat, parent_pose))
        poses.sort(
            key=lambda pose: (
                abs(pose.position[0] - fit.x.center)
                + abs(pose.position[1] - fit.y.center)
                + abs(pose.position[2] - fit.z.center)
            )
        )
        if not poses:
            return LayoutDiagnostic(
                "in_volume_domain_empty", "in_volume has no candidate poses", target=rel.object_name
            )
        return poses

    def precheck(self, object_name: str) -> list[LayoutDiagnostic]:
        rel = self.problem.relation_index.in_volumes_by_object.get(object_name)
        if rel is None:
            return []
        obj = self.problem.objects.get(rel.object_name)
        volume = self.problem.spec.volumes_by_id.get(rel.volume)
        if obj is None or volume is None:
            return []
        fit = volume_fit_bounds(volume, obj.asset.local_bounds, mode=rel.mode)
        if fit is None:
            return [
                LayoutDiagnostic(
                    "in_volume_capacity_insufficient",
                    "object does not fit in volume",
                    target=rel.object_name,
                    meta={"volume": rel.volume},
                )
            ]
        return []


class RelativeDirectionConstraint:
    name = "relative_direction"

    @staticmethod
    def axis_order(direction: str) -> tuple[str, str]:
        if direction == "left_of":
            return ("x", "forward")
        if direction == "right_of":
            return ("x", "reverse")
        if direction == "back_of":
            return ("y", "forward")
        if direction == "front_of":
            return ("y", "reverse")
        return ("unknown", "forward")

    @staticmethod
    def _has_cycle(edges: dict[str, set[str]]) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for nxt in sorted(edges.get(node, set())):
                if dfs(nxt):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(dfs(node) for node in sorted(edges))

    def cycle_diagnostics(self, problem: LayoutProblem) -> list[LayoutDiagnostic]:
        out: list[LayoutDiagnostic] = []
        for surface_id, relations in sorted(problem.relation_index.relative_by_surface.items()):
            edges_by_axis = {"x": {}, "y": {}}
            for rel in relations:
                axis, order = self.axis_order(rel.direction)
                if axis not in edges_by_axis:
                    continue
                if order == "forward":
                    src, dst = rel.node_1, rel.node_2
                else:
                    src, dst = rel.node_2, rel.node_1
                edges_by_axis[axis].setdefault(src, set()).add(dst)
                edges_by_axis[axis].setdefault(dst, set())
            for axis, edges in edges_by_axis.items():
                if self._has_cycle(edges):
                    out.append(
                        LayoutDiagnostic(
                            "relative_direction_cycle",
                            "relative_direction_cycle",
                            target=f"surface:{surface_id}",
                            meta={"axis": axis, "relation_count": len(relations), "factor": self.name},
                        )
                    )
        return out

    def relations_for_surface(
        self, problem: LayoutProblem, surface_id: str, names: list[str]
    ) -> list[RelativeDirectionSpec]:
        name_set = set(names)
        return [
            rel
            for rel in problem.relation_index.relative_by_surface.get(surface_id, [])
            if rel.node_1 in name_set and rel.node_2 in name_set
        ]
