from __future__ import annotations

from dataclasses import dataclass, field

from ..common.constants import GEOM_EPS
from ..common.diagnostics import LayoutDiagnostic
from ..common.geometry import Bounds2D, ManualSupportSurface
from ..scene_spec.schema import RegionSampleSpec
from .domains.providers import InVolumeProvider, PoseSampleProvider, RelativeDirectionConstraint, SupportOnProvider
from .problem import LayoutProblem


@dataclass(frozen=True)
class AlignedRegionCapacity:
    bounds: Bounds2D
    fixed_axis: str
    vary_axis: str
    sizes: dict[str, tuple[float, float]]
    required: float
    available: float


def aligned_region_capacity(
    *,
    surface: ManualSupportSurface,
    region: RegionSampleSpec,
    names: list[str],
    footprint_size,
) -> AlignedRegionCapacity | LayoutDiagnostic:
    bounds = surface.effective_bounds()
    if bounds is None:
        return LayoutDiagnostic(
            "surface_range_empty", "surface margin leaves no placeable area", target=surface.surface_id
        )
    surface_range = region.surface_range
    if surface_range is None and region.surface_ranges:
        surface_range = next((choice.surface_range for choice in region.surface_ranges if choice.weight > 0.0), None)
    if surface_range is not None:
        bounds = bounds.intersect(surface_range.to_bounds(bounds))
        if bounds is None:
            return LayoutDiagnostic("surface_range_empty", "region surface_range is empty", target=region.region_id)

    fixed_axis = str(region.align_axis or "")
    vary_axis = "y" if fixed_axis == "x" else "x"
    sizes = {name: footprint_size(name) for name in names}
    required = float(region.min_spacing) * max(0, len(names) - 1)
    spacing_mode = str(getattr(region, "spacing_mode", "footprint_gap") or "footprint_gap").strip().lower()
    spacing_mode = (
        "center_distance" if spacing_mode in {"center", "center_distance", "center-distance"} else "footprint_gap"
    )
    if spacing_mode != "center_distance":
        for name in names:
            sx, sy = sizes[name]
            required += sy if vary_axis == "y" else sx
    available = bounds.y.length if vary_axis == "y" else bounds.x.length
    if required > available + GEOM_EPS:
        return LayoutDiagnostic(
            "region_capacity_insufficient",
            "aligned region does not have enough 1D capacity",
            target=region.region_id,
            meta={
                "required": required,
                "available": available,
                "vary_axis": vary_axis,
                "packer": "region_aligned",
                "spacing_mode": spacing_mode,
            },
        )
    return AlignedRegionCapacity(
        bounds=bounds,
        fixed_axis=fixed_axis,
        vary_axis=vary_axis,
        sizes=sizes,
        required=required,
        available=available,
    )


@dataclass(frozen=True)
class FeasibilityIssue:
    code: str
    target: str
    message: str
    meta: dict = field(default_factory=dict)

    def to_diagnostic(self) -> LayoutDiagnostic:
        return LayoutDiagnostic(code=self.code, message=self.message, target=self.target, meta=dict(self.meta))

    def to_json(self) -> dict:
        return self.to_diagnostic().to_json()


@dataclass(frozen=True)
class FeasibilityReport:
    success: bool
    issues: tuple[FeasibilityIssue, ...] = ()
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "success": bool(self.success),
            "issues": [issue.to_json() for issue in self.issues],
            "meta": dict(self.meta),
        }


class FeasibilityAnalyzer:
    def __init__(self, problem: LayoutProblem, *, relation_overrides: dict | None = None) -> None:
        self.problem = problem
        self.relation_overrides = dict(relation_overrides or {})
        self.support_provider = SupportOnProvider(problem, relation_overrides=self.relation_overrides)
        self.pose_provider = PoseSampleProvider(problem)
        self.in_volume_provider = InVolumeProvider(problem)
        self.relative_factor = RelativeDirectionConstraint()

    def analyze(self) -> FeasibilityReport:
        issues: list[FeasibilityIssue] = []
        candidate_counts: dict[str, int] = {}

        for name, obj in sorted(self.problem.objects.items()):
            if not obj.support:
                continue
            if obj.placement_owner != "support_scope":
                continue
            if self.problem.local_surface_parent(obj.support):
                candidate_counts[name] = 0
                continue
            domain = self.support_provider.domain(name)
            candidate_counts[name] = len(domain.candidates)
            if domain.issues:
                code = str(domain.issues[0])
                issues.append(
                    FeasibilityIssue(
                        code=code,
                        target=f"object:{name}",
                        message=code,
                        meta={"scope": "support_on", **dict(domain.meta)},
                    )
                )

        issues.extend(self._relative_direction_issues())
        issues.extend(self._pose_sample_issues())
        issues.extend(self._in_volume_issues())
        issues.extend(self._region_capacity_issues())
        issues.extend(self._surface_group_spacing_issues())
        return FeasibilityReport(
            success=not issues,
            issues=tuple(issues),
            meta={"candidate_counts": candidate_counts},
        )

    def _relative_direction_issues(self) -> list[FeasibilityIssue]:
        return [self._issue_from_diagnostic(diag) for diag in self.relative_factor.cycle_diagnostics(self.problem)]

    def _pose_sample_issues(self) -> list[FeasibilityIssue]:
        out: list[FeasibilityIssue] = []
        for object_name, sample in sorted(self.problem.relation_index.pose_samples_by_object.items()):
            if object_name not in self.problem.objects:
                continue
            out.extend(self._issue_from_diagnostic(diag) for diag in self.pose_provider.precheck(object_name))
        return out

    def _in_volume_issues(self) -> list[FeasibilityIssue]:
        out: list[FeasibilityIssue] = []
        volumes = self.problem.spec.volumes_by_id
        for object_name, rel in sorted(self.problem.relation_index.in_volumes_by_object.items()):
            obj = self.problem.objects.get(object_name)
            volume = volumes.get(rel.volume)
            if obj is None or volume is None:
                continue
            for diag in self.in_volume_provider.precheck(object_name):
                meta = dict(diag.meta)
                meta.setdefault("object_size", list(obj.asset.local_bounds.size))
                meta.setdefault("volume_size", [volume.bounds.x.length, volume.bounds.y.length, volume.bounds.z.length])
                out.append(
                    FeasibilityIssue(code=diag.code, target=f"object:{object_name}", message=diag.message, meta=meta)
                )
        return out

    @staticmethod
    def _issue_from_diagnostic(diag: LayoutDiagnostic) -> FeasibilityIssue:
        return FeasibilityIssue(code=diag.code, target=diag.target, message=diag.message, meta=dict(diag.meta))

    def _region_capacity_issues(self) -> list[FeasibilityIssue]:
        out: list[FeasibilityIssue] = []
        for region in self.problem.spec.regions:
            if not region.align_axis:
                continue
            names = [name for name in region.objects if name in self.problem.objects]
            if not names:
                continue
            if self.problem.local_surface_parent(region.surface):
                continue
            surface = self.problem.resolve_surface(region.surface, {})
            if surface is None:
                continue
            capacity = aligned_region_capacity(
                surface=surface,
                region=region,
                names=names,
                footprint_size=lambda name: (
                    self.support_provider.footprint_size(
                        name,
                        surface,
                        yaw_range=_relation_yaw_for_object(self.problem, region, name),
                    )
                ),
            )
            if isinstance(capacity, LayoutDiagnostic):
                out.append(
                    FeasibilityIssue(
                        code=capacity.code,
                        target=f"region:{region.region_id}",
                        message=capacity.message,
                        meta=dict(capacity.meta),
                    )
                )
        return out

    def _surface_group_spacing_issues(self) -> list[FeasibilityIssue]:
        out: list[FeasibilityIssue] = []
        for relation, kind, target in self._surface_group_relations():
            min_spacing = max(0.0, float(getattr(relation, "min_spacing", 0.0) or 0.0))
            if min_spacing <= 0.0:
                continue
            spacing_mode = str(getattr(relation, "spacing_mode", "footprint_gap") or "footprint_gap").strip().lower()
            spacing_mode = (
                "center_distance"
                if spacing_mode in {"center", "center_distance", "center-distance"}
                else "footprint_gap"
            )
            surface_id = str(getattr(relation, "surface", "") or "")
            surface = self.problem.resolve_surface(surface_id, {})
            if surface is None:
                continue
            names = [name for name in getattr(relation, "objects", []) if name in self.problem.objects]
            if len(names) < 2:
                continue
            domains: dict[str, tuple[Bounds2D, Bounds2D]] = {}
            for name in names:
                center_bounds, _ = self.support_provider.center_bounds(
                    name,
                    surface_override=surface_id,
                    group_surface_range=getattr(relation, "surface_range", None),
                    yaw_range=_relation_yaw_for_object(self.problem, relation, name),
                )
                if center_bounds is None:
                    continue
                footprint = self.support_provider.footprint_bounds(
                    name,
                    surface,
                    yaw_range=_relation_yaw_for_object(self.problem, relation, name),
                )
                domains[name] = (center_bounds, footprint)
            for index, left in enumerate(names):
                for right in names[index + 1 :]:
                    if left not in domains or right not in domains:
                        continue
                    if spacing_mode == "center_distance":
                        max_value = _max_possible_center_distance(domains[left][0], domains[right][0])
                        reason = "pair_center_distance_impossible"
                        message = "group pair cannot satisfy center-distance min_spacing within authored domains"
                        meta_key = "max_possible_center_distance"
                    else:
                        max_value = _max_possible_footprint_gap(domains[left], domains[right])
                        reason = "pair_min_spacing_impossible"
                        message = "group pair cannot satisfy footprint min_spacing within authored domains"
                        meta_key = "max_possible_gap"
                    if max_value + GEOM_EPS < min_spacing:
                        out.append(
                            FeasibilityIssue(
                                code=(
                                    "region_capacity_insufficient"
                                    if kind == "region_sample"
                                    else "scatter_group_capacity_insufficient"
                                ),
                                target=target,
                                message=message,
                                meta={
                                    "reason": reason,
                                    "pair": [left, right],
                                    "min_spacing": min_spacing,
                                    meta_key: max_value,
                                    "spacing_mode": spacing_mode,
                                    "surface": surface_id,
                                    "scope": kind,
                                    "packer": (
                                        "region_sample"
                                        if kind == "region_sample"
                                        else f"scatter_{str(getattr(relation, 'mode', 'group') or 'group')}"
                                    ),
                                    "unknown_rejection_count": True,
                                },
                            )
                        )
                        return out
        return out

    def _surface_group_relations(self):
        for region in self.problem.spec.regions:
            yield region, "region_sample", f"region:{region.region_id}"
        for group in self.problem.spec.scatter_groups:
            yield group, "scatter_group", f"scatter:{group.group_id}"


def _relation_yaw_for_object(problem: LayoutProblem, relation, name: str):
    obj = problem.objects.get(name)
    if obj is not None and obj.spec.yaw is not None:
        return obj.spec.yaw
    return getattr(relation, "yaw", None)


def _max_possible_center_distance(left: Bounds2D, right: Bounds2D) -> float:
    return max(
        ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
        for lx in (left.x.min, left.x.max)
        for ly in (left.y.min, left.y.max)
        for rx in (right.x.min, right.x.max)
        for ry in (right.y.min, right.y.max)
    )


def _max_possible_footprint_gap(
    left: tuple[Bounds2D, Bounds2D],
    right: tuple[Bounds2D, Bounds2D],
) -> float:
    left_center, left_fp = left
    right_center, right_fp = right
    return max(
        _max_axis_gap(left_center.x, left_fp.x, right_center.x, right_fp.x),
        _max_axis_gap(left_center.y, left_fp.y, right_center.y, right_fp.y),
    )


def _max_axis_gap(left_center, left_fp, right_center, right_fp) -> float:
    return max(
        right_center.max + right_fp.min - (left_center.min + left_fp.max),
        left_center.max + left_fp.min - (right_center.min + right_fp.max),
    )
