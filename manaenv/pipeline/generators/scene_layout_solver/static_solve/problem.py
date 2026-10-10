from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace

from ..assets.bridge import LayoutAssetRecord
from ..common.geometry import Bounds2D, ManualSupportSurface, Pose, Range1D, quat_rotate, vec_add
from ..common.surface_polygon import Polygon2D
from ..scene_spec.schema import (
    InVolumeSpec,
    LayoutObjectSpec,
    LocalSupportSurfaceSpec,
    PoseSampleSpec,
    RegionSampleSpec,
    RelativeDirectionSpec,
    ScatterGroupSpec,
    SceneLayoutSpec,
    SurfaceRangeChoiceSpec,
    SurfaceRangeSpec,
    VolumeGroupSpec,
)
from ..variants.planner import LayoutCase


@dataclass(frozen=True)
class ProblemObject:
    name: str
    spec: LayoutObjectSpec
    asset: LayoutAssetRecord
    support: str | None
    surface_range: SurfaceRangeSpec | None = None
    surface_ranges: tuple[SurfaceRangeChoiceSpec, ...] = ()
    override_surface_range: SurfaceRangeSpec | None = None
    override_yaw: tuple[float, float] | None = None
    slot_id: str | None = None
    placement_owner: str = "support_scope"


@dataclass(frozen=True)
class ProblemRelationIndex:
    regions_by_surface: dict[str, list[RegionSampleSpec]] = field(default_factory=dict)
    scatter_by_surface: dict[str, list[ScatterGroupSpec]] = field(default_factory=dict)
    relative_by_surface: dict[str, list[RelativeDirectionSpec]] = field(default_factory=dict)
    pose_samples_by_object: dict[str, PoseSampleSpec] = field(default_factory=dict)
    in_volumes_by_object: dict[str, InVolumeSpec] = field(default_factory=dict)
    volume_groups: list[VolumeGroupSpec] = field(default_factory=list)
    grouped_objects: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class LayoutProblem:
    spec: SceneLayoutSpec
    case: LayoutCase
    surfaces: Mapping[str, ManualSupportSurface]
    local_surfaces: Mapping[str, LocalSupportSurfaceSpec]
    objects: Mapping[str, ProblemObject]
    relation_index: ProblemRelationIndex

    @property
    def object_names(self) -> list[str]:
        return sorted(self.objects)

    @property
    def objects_by_surface(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for obj in self.objects.values():
            if obj.support:
                out.setdefault(obj.support, []).append(obj.name)
        return {surface: sorted(names) for surface, names in out.items()}

    def local_surface_parent(self, surface_id: str) -> str | None:
        surface = self.local_surfaces.get(str(surface_id))
        return surface.parent if surface is not None else None

    def resolve_surface(
        self, surface_id: str, assignment: Mapping[str, Pose] | None = None
    ) -> ManualSupportSurface | None:
        sid = str(surface_id)
        surface = self.surfaces.get(sid)
        if surface is not None:
            return surface
        local = self.local_surfaces.get(sid)
        if local is None:
            return None
        parent_pose = (assignment or {}).get(local.parent)
        if parent_pose is None:
            return None
        parent = self.objects.get(local.parent)
        parent_scale = parent.asset.scale if parent is not None else (1.0, 1.0, 1.0)
        return _local_surface_to_world(local, parent_pose, parent_scale=parent_scale)


def build_layout_problem(
    spec: SceneLayoutSpec,
    case: LayoutCase,
    surfaces: Mapping[str, ManualSupportSurface],
    records: Mapping[str, LayoutAssetRecord],
) -> LayoutProblem:
    objects_by_name = spec.objects_by_name
    problem_objects: dict[str, ProblemObject] = {}
    placement_owner_by_object = _placement_owners(spec)
    for name, obj in objects_by_name.items():
        override = case.object_overrides.get(name)
        support = (override.surface if override else None) or obj.support
        problem_objects[name] = ProblemObject(
            name=name,
            spec=obj,
            asset=records[name],
            support=support,
            surface_range=obj.surface_range,
            surface_ranges=tuple(obj.surface_ranges),
            override_surface_range=(override.surface_range if override else None),
            override_yaw=(override.yaw if override else None),
            slot_id=(override.slot_id if override else None),
            placement_owner=placement_owner_by_object.get(name, "support_scope" if support else "global_remaining"),
        )

    grouped: set[str] = set()
    regions_by_surface: dict[str, list[RegionSampleSpec]] = {}
    for region in spec.regions:
        regions_by_surface.setdefault(region.surface, []).append(region)
        grouped.update(region.objects)

    scatter_by_surface: dict[str, list[ScatterGroupSpec]] = {}
    for group in spec.scatter_groups:
        scatter_by_surface.setdefault(group.surface, []).append(group)
        grouped.update(group.objects)

    for group in spec.volume_groups:
        grouped.update(group.objects)

    relative_by_surface: dict[str, list[RelativeDirectionSpec]] = {}
    for rel in spec.relative_directions:
        left = problem_objects.get(rel.node_1)
        right = problem_objects.get(rel.node_2)
        if left is None or right is None:
            continue
        if left.support and left.support == right.support:
            relative_by_surface.setdefault(left.support, []).append(rel)

    return LayoutProblem(
        spec=spec,
        case=case,
        surfaces=surfaces,
        local_surfaces=spec.local_surfaces_by_id,
        objects=problem_objects,
        relation_index=ProblemRelationIndex(
            regions_by_surface=regions_by_surface,
            scatter_by_surface=scatter_by_surface,
            relative_by_surface=relative_by_surface,
            pose_samples_by_object=spec.pose_samples_by_object,
            in_volumes_by_object=spec.in_volumes_by_object,
            volume_groups=list(spec.volume_groups),
            grouped_objects=grouped,
        ),
    )


def _local_surface_to_world(
    surface: LocalSupportSurfaceSpec,
    parent_pose: Pose,
    *,
    parent_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ManualSupportSurface:
    center_local = _scale_vec3(surface.center, parent_scale)
    x_local = _scale_vec3(surface.x_axis, parent_scale)
    y_local = _scale_vec3(surface.y_axis, parent_scale)
    x_scale = _vec_norm(x_local)
    y_scale = _vec_norm(y_local)
    center = vec_add(parent_pose.position, quat_rotate(parent_pose.quaternion, center_local))
    x_axis = quat_rotate(parent_pose.quaternion, x_local)
    y_axis = quat_rotate(parent_pose.quaternion, y_local)
    normal = quat_rotate(parent_pose.quaternion, surface.normal) if surface.normal is not None else None
    return ManualSupportSurface.from_spec(
        SimpleNamespace(
            surface_id=surface.surface_id,
            center=center,
            x_axis=x_axis,
            y_axis=y_axis,
            normal=normal,
            bounds=_scale_bounds2d(surface.bounds, x_scale=x_scale, y_scale=y_scale),
            shape=surface.shape,
            polygon=_scale_polygon2d(surface.polygon, x_scale=x_scale, y_scale=y_scale),
            margin=surface.margin * min(x_scale, y_scale),
        )
    )


def _scale_vec3(value, scale: tuple[float, float, float]):
    return (float(value[0]) * scale[0], float(value[1]) * scale[1], float(value[2]) * scale[2])


def _vec_norm(value) -> float:
    return max(1e-12, (float(value[0]) ** 2 + float(value[1]) ** 2 + float(value[2]) ** 2) ** 0.5)


def _scale_bounds2d(bounds: Bounds2D, *, x_scale: float, y_scale: float) -> Bounds2D:
    return Bounds2D(
        Range1D(bounds.x.min * x_scale, bounds.x.max * x_scale),
        Range1D(bounds.y.min * y_scale, bounds.y.max * y_scale),
    )


def _scale_polygon2d(polygon: Polygon2D | None, *, x_scale: float, y_scale: float) -> Polygon2D | None:
    if polygon is None:
        return None
    return Polygon2D(
        outer=tuple((x * x_scale, y * y_scale) for x, y in polygon.outer),
        holes=tuple(tuple((x * x_scale, y * y_scale) for x, y in ring) for ring in polygon.holes),
    )


def _placement_owners(spec: SceneLayoutSpec) -> dict[str, str]:
    owners: dict[str, str] = {}
    for region in spec.regions:
        for name in region.objects:
            owners[name] = f"region_sample:{region.region_id}"
    for group in spec.scatter_groups:
        for name in group.objects:
            owners[name] = f"scatter_group:{group.group_id}"
    for sample in spec.pose_samples:
        owners[sample.object_name] = "pose_sample"
    for rel in spec.in_volumes:
        owners[rel.object_name] = f"in_volume:{rel.volume}"
    for group in spec.volume_groups:
        for name in group.objects:
            owners[name] = f"volume_group:{group.group_id}"
    return owners
