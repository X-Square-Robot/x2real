from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from ..common.constants import (
    DEFAULT_RUNTIME_PARTITION_CANDIDATES,
    DEFAULT_RUNTIME_PARTITION_CASES,
    GEOM_EPS,
    MIN_RUNTIME_PARTITION_CANDIDATES,
    RUNTIME_PARTITION_SEED_STRIDE,
)
from ..common.geometry import (
    Bounds2D,
    ManualSupportSurface,
    Range1D,
    projected_footprint_bounds_with_yaw_range,
    projected_footprint_size_with_yaw_range,
)
from ..common.volumes import runtime_volume_domain_geometry, volume_parent
from ..static_solve.domains.providers import (
    SupportOnProvider,
    object_base_quaternion,
    object_quaternion,
    object_scene_yaw_range,
)
from ..variants.planner import LayoutCase, ObjectPlacementOverride
from .cell_domains import safe_id
from .compile_utils import bounds2d_json, local_jitter_bounds, range_from_pair, relative_sample_bounds, sample_xy_bounds
from .events import CompiledSamplingDomain, NormalizedRuntimeEvent
from .meta import (
    META_PLAN_KIND,
    META_SEMANTIC_POOL,
    META_SEMANTIC_POOL_KIND,
    META_SOURCE_SCENE_FIELD,
    META_WEIGHT_SCOPE,
    META_WEIGHT_TOTAL,
    META_ZERO_WEIGHT_EXCLUDED,
    PLAN_PARTITION_CASE_POOL,
    POOL_POSE_CHOICE,
    POOL_RELATION_SURFACE_RANGE_CHOICE,
    POOL_SURFACE_RANGE_CHOICE,
)
from .partition_cases import PartitionCasePoolResult, build_partition_case_pool
from .schema import AssignmentPlan, PlacementDomain
from .support_domains import (
    local_surface_child_z_range,
    support_cell_accept_fn,
    surface_bounds_to_parent_xy_bounds,
    surface_child_z_range,
)
from .yaw_choices import object_yaw_choice_payloads, object_yaw_choices_by_object, yaw_choice_semantic_meta


@dataclass(frozen=True)
class _RuntimeSurfaceRangeChoice:
    choice_id: str
    bounds: Bounds2D
    weight: float = 1.0
    source: str = "surface_range"


def _clip_runtime_surface_range_choices(
    compiler: Any,
    choices: list[_RuntimeSurfaceRangeChoice],
    *,
    effective_bounds: Bounds2D | None,
    target: str,
    surface_id: str,
) -> list[_RuntimeSurfaceRangeChoice] | None:
    """Intersect authored relation ranges with the physical support surface."""

    if effective_bounds is None:
        compiler.plan.add_diagnostic(
            "layout_runtime_sampling_surface_effective_bounds_empty",
            "runtime sampling surface has no usable effective bounds",
            target=target,
            surface=surface_id,
        )
        return None
    clipped = [
        replace(choice, bounds=bounds)
        for choice in choices
        if (bounds := choice.bounds.intersect(effective_bounds)) is not None
    ]
    if clipped:
        return clipped
    compiler.plan.add_diagnostic(
        "layout_runtime_sampling_surface_range_outside_support",
        "runtime sampling range does not intersect the effective support surface",
        target=target,
        surface=surface_id,
        effective_bounds=bounds2d_json(effective_bounds),
    )
    return None


def _deterministic_yaw(yaw: tuple[float, float] | None) -> float | None:
    if yaw is None:
        return None
    lo, hi = (float(yaw[0]), float(yaw[1]))
    if abs(lo - hi) > GEOM_EPS:
        return None
    return lo


def _offset_yaw_range(yaw: tuple[float, float] | None, offset: float) -> tuple[float, float] | None:
    if yaw is None:
        return None
    return (float(yaw[0]) - float(offset), float(yaw[1]) - float(offset))


def _offset_yaw_choices(
    yaw_choices: tuple[dict[str, Any], ...],
    offset: float,
) -> tuple[dict[str, Any], ...]:
    return tuple({**dict(choice), "yaw": float(choice.get("yaw", 0.0)) - float(offset)} for choice in yaw_choices)


class RuntimeAuthoringCompilerMixin:
    """Compile vNext runtime placement domains into RuntimeSamplingPlan backend objects."""

    def _variant_choice_domain(
        self,
        object_name: str,
        override: ObjectPlacementOverride,
        *,
        domain_id: str,
        target: str,
        event: NormalizedRuntimeEvent,
        case: LayoutCase,
    ) -> PlacementDomain | None:
        assert self.problem is not None
        if object_name not in self.problem.objects:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_variant_object_unknown",
                "runtime variant choice references object outside layout spec",
                target=target,
                object=object_name,
                case_id=case.case_id,
            )
            return None
        obj = self.problem.objects[object_name]
        surface_id = str(override.surface or obj.support or "").strip()
        if not surface_id:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_variant_surface_missing",
                "runtime variant choice object has no support surface",
                target=target,
                object=object_name,
                case_id=case.case_id,
            )
            return None
        local_parent = self.problem.local_surface_parent(surface_id)
        raw_local_surface = self.problem.local_surfaces.get(surface_id) if local_parent else None
        local_surface = (
            _runtime_scaled_local_surface(raw_local_surface, parent=local_parent, records=self.records)
            if raw_local_surface is not None and local_parent
            else None
        )
        surface = local_surface if local_surface is not None else self.problem.resolve_surface(surface_id, {})
        if surface is None:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_variant_surface_unknown",
                "runtime variant choice references unknown support surface",
                target=target,
                object=object_name,
                surface=surface_id,
                case_id=case.case_id,
            )
            return None
        surface_bounds = (
            surface.bounds.shrink(surface.margin) if local_surface is not None else surface.effective_bounds()
        )
        if surface_bounds is None:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_variant_surface_empty",
                "runtime variant choice surface has empty effective bounds",
                target=target,
                object=object_name,
                surface=surface_id,
                case_id=case.case_id,
            )
            return None
        yaw = object_scene_yaw_range(self.problem, object_name)
        yaw_choices = object_yaw_choice_payloads(self.problem, object_name)
        override_yaw_is_surface_local = override.yaw is not None
        if override.yaw is not None:
            yaw = tuple(float(value) for value in override.yaw)
            yaw_choices = ()
        if local_surface is not None and local_parent:
            parent_yaw = object_scene_yaw_range(self.problem, local_parent)
            parent_yaw_value = _deterministic_yaw(parent_yaw)
            if parent_yaw is not None and parent_yaw_value is None and (yaw is not None or yaw_choices):
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_variant_local_surface_parent_yaw_dynamic",
                    "runtime variant choice on object-local slots requires deterministic parent yaw when child yaw is authored",
                    target=target,
                    object=object_name,
                    parent=local_parent,
                    surface=surface_id,
                    case_id=case.case_id,
                )
                return None
            if parent_yaw_value is not None and not override_yaw_is_surface_local:
                yaw = _offset_yaw_range(yaw, parent_yaw_value)
                yaw_choices = _offset_yaw_choices(yaw_choices, parent_yaw_value)
        base_quaternion = object_base_quaternion(self.problem, object_name)
        footprint_quaternion = base_quaternion if yaw is not None else object_quaternion(self.problem, object_name)
        footprint_bounds = projected_footprint_bounds_with_yaw_range(
            self.records[object_name].local_bounds,
            footprint_quaternion,
            yaw,
            surface.x_axis,
            surface.y_axis,
        )
        fit_bounds = Bounds2D(
            Range1D(surface_bounds.x.min - footprint_bounds.x.min, surface_bounds.x.max - footprint_bounds.x.max),
            Range1D(surface_bounds.y.min - footprint_bounds.y.min, surface_bounds.y.max - footprint_bounds.y.max),
        )
        if fit_bounds.x.min > fit_bounds.x.max or fit_bounds.y.min > fit_bounds.y.max:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_variant_object_too_large",
                "runtime variant choice object does not fit support surface",
                target=target,
                object=object_name,
                surface=surface_id,
                case_id=case.case_id,
            )
            return None
        bounds = fit_bounds
        if override.surface_range is not None:
            bounds = fit_bounds.intersect(override.surface_range.to_bounds(fit_bounds))
            if bounds is None:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_variant_domain_empty",
                    "runtime variant choice surface_range excludes object fit bounds",
                    target=target,
                    object=object_name,
                    surface=surface_id,
                    case_id=case.case_id,
                    slot_id=override.slot_id or "",
                )
                return None
        if local_surface is not None:
            parent_bounds = surface_bounds_to_parent_xy_bounds(local_surface, bounds)
            if parent_bounds is None:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_variant_local_surface_frame_unsupported",
                    "runtime variant choice requires object-local variant slots to be on a parent-local horizontal surface",
                    target=target,
                    object=object_name,
                    surface=surface_id,
                    case_id=case.case_id,
                )
                return None
            z = local_surface_child_z_range(local_surface, self.records[object_name], quaternion=footprint_quaternion)
            frame = f"object:{local_parent}"
            domain_bounds = parent_bounds
        else:
            z = surface_child_z_range(surface, self.records[object_name], quaternion=footprint_quaternion)
            frame = f"surface:{surface_id}"
            domain_bounds = bounds
        return PlacementDomain(
            domain_id=domain_id,
            object_name=object_name,
            frame=frame,
            geometry_type="rectangle",
            surface=surface_id,
            x=(domain_bounds.x.min, domain_bounds.x.max),
            y=(domain_bounds.y.min, domain_bounds.y.max),
            z=z,
            yaw=yaw,
            yaw_choices=yaw_choices,
            base_quaternion=base_quaternion,
            meta={
                "source_event": event.name,
                "authoring_frame": "variant-choice",
                **yaw_choice_semantic_meta(object_name, yaw_choices),
                "sample_type": "variant_choice",
                "case_id": case.case_id,
                "source_case_id": str(case.params.get("source_case_id") or case.case_id),
                "case_weight": _layout_case_weight(case),
                "slot_id": override.slot_id or "",
                "surface": surface_id,
                "parent": local_parent or "",
                "footprint": [footprint_bounds.x.length, footprint_bounds.y.length],
                "footprint_bounds": bounds2d_json(footprint_bounds),
                "sample_domain": bounds2d_json(domain_bounds),
            },
        )

    def _compile_sampling_domain(
        self,
        raw: Mapping[str, Any],
        *,
        event: NormalizedRuntimeEvent,
        index: int,
    ) -> CompiledSamplingDomain | None:
        domain_id = str(raw.get("id") or "").strip()
        if not domain_id:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_domain_id_missing",
                "sampling domain requires id",
                target=event.name,
                index=index,
            )
            return None
        frame = raw.get("frame")
        sample = raw.get("sample")
        if not isinstance(frame, Mapping):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_frame_invalid", "sampling domain requires frame mapping", target=domain_id
            )
            return None
        if not isinstance(sample, Mapping):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_sample_invalid", "sampling domain requires sample mapping", target=domain_id
            )
            return None
        if "type" not in sample:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_sample_type_missing",
                "sampling domain sample requires explicit type",
                target=domain_id,
            )
            return None
        frame_type = str(frame.get("type") or "").strip().lower()
        if frame_type == "scene-surface":
            domain_nodes = self._sampling_domain_nodes(raw, target=domain_id)
            if not domain_nodes:
                return None
            if len(domain_nodes) > 1 or raw.get("region") is not None:
                return self._scene_surface_sampling_domains(
                    domain_nodes, domain_id=domain_id, event=event, sample=sample, raw=raw
                )
            domain = self._scene_surface_sampling_domain(
                domain_nodes[0], domain_id=domain_id, event=event, sample=sample
            )
            return CompiledSamplingDomain((domain,), None) if domain is not None else None
        if frame_type == "object":
            node = str(raw.get("node") or "").strip()
            parent = str(frame.get("parent") or "").strip()
            if not node or not parent:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_object_frame_invalid",
                    "object frame sampling requires node and frame.parent",
                    target=domain_id,
                )
                return None
            domain = self._object_relative_sampling_domain(
                node, domain_id=domain_id, parent=parent, event=event, sample=sample
            )
            return CompiledSamplingDomain((domain,), parent) if domain is not None else None
        if frame_type == "pose-sample":
            node = str(raw.get("node") or "").strip()
            if not node:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_pose_sample_node_missing",
                    "pose-sample frame sampling requires node",
                    target=domain_id,
                )
                return None
            domain = self._pose_sample_sampling_domain(node, domain_id=domain_id, event=event, sample=sample)
            return CompiledSamplingDomain((domain,), None) if domain is not None else None
        if frame_type == "object-volume":
            volume_group = self._volume_group_by_id(str(raw.get("volume_group") or ""))
            parent = str(frame.get("node") or frame.get("parent") or "").strip()
            volume_name = str(frame.get("volume") or frame.get("id") or "").strip()
            if volume_group is not None and not volume_name:
                volume_name = str(volume_group.volume)
            if volume_group is not None and not parent:
                volume_id = str(volume_group.volume)
                volume = self.problem.spec.volumes_by_id.get(volume_id) if self.problem is not None else None
                parent = volume_parent(volume) if volume is not None else ""
            if not parent or not volume_name:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_object_volume_frame_invalid",
                    "object-volume sampling requires frame.node and frame.volume",
                    target=domain_id,
                )
                return None
            domain_nodes = self._sampling_domain_nodes(raw, target=domain_id)
            if not domain_nodes:
                return None
            return self._object_volume_sampling_domains(
                domain_nodes,
                domain_id=domain_id,
                parent=parent,
                volume_name=volume_name,
                event=event,
                sample=sample,
                volume_group_id=str(raw.get("volume_group") or ""),
            )
        if frame_type == "node-surface":
            parent = str(frame.get("node") or "").strip()
            surface_name = str(frame.get("surface") or "").strip()
            if not parent or not surface_name:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_node_surface_frame_invalid",
                    "node-surface sampling requires frame.node and frame.surface",
                    target=domain_id,
                )
                return None
            domain_nodes = self._sampling_domain_nodes(raw, target=domain_id)
            if not domain_nodes:
                return None
            return self._node_surface_sampling_domains(
                domain_nodes,
                domain_id=domain_id,
                parent=parent,
                surface_name=surface_name,
                event=event,
                sample=sample,
                raw=raw,
            )
        self.plan.add_diagnostic(
            "layout_runtime_sampling_frame_unsupported",
            "unsupported runtime sampling frame type",
            target=domain_id,
            frame_type=frame_type,
        )
        return None

    def _pose_sample_sampling_domain(
        self,
        object_name: str,
        *,
        domain_id: str,
        event: NormalizedRuntimeEvent,
        sample: Mapping[str, Any],
    ) -> PlacementDomain | None:
        assert self.problem is not None
        assert self.spec is not None
        if object_name not in self.problem.objects:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_object_unknown",
                "pose-sample frame references object outside layout spec",
                target=domain_id,
                object=object_name,
            )
            return None
        sample_type = str(sample.get("type") or "").strip().lower()
        if sample_type != "inherit":
            self.plan.add_diagnostic(
                "layout_runtime_sampling_pose_sample_sample_unsupported",
                "pose-sample frame currently requires inherit sample",
                target=domain_id,
                sample_type=sample_type,
            )
            return None
        if sample_xy_bounds(sample) is not None or sample.get("z") is not None or _sample_has_runtime_yaw(sample):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_pose_sample_override_unsupported",
                "pose-sample inherit cannot override x/y/z/yaw; edit scene pose_samples or use explicit sampling",
                target=domain_id,
            )
            return None
        pose_sample = self.spec.pose_samples_by_object.get(object_name)
        if pose_sample is None:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_pose_sample_unknown",
                "pose-sample frame references object without scene pose_samples",
                target=domain_id,
                object=object_name,
            )
            return None
        poses = tuple(
            {
                "id": pose.pose_id,
                "position": [float(pose.position[0]), float(pose.position[1]), float(pose.position[2])],
                "quaternion": [float(value) for value in pose.quaternion],
                "weight": float(pose.weight),
            }
            for pose in pose_sample.poses
            if pose.weight > 0.0
        )
        if not poses:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_pose_sample_empty",
                "scene pose_sample has no candidate poses",
                target=domain_id,
                object=object_name,
            )
            return None
        return PlacementDomain(
            domain_id=domain_id,
            object_name=object_name,
            frame="world",
            geometry_type="pose_list",
            poses=poses,
            meta={
                "source_event": event.name,
                "authoring_frame": "pose-sample",
                "pose_sample": object_name,
                "candidate_count": len(poses),
                "shuffle": bool(pose_sample.shuffle),
                META_PLAN_KIND: POOL_POSE_CHOICE,
                META_SEMANTIC_POOL: True,
                META_SEMANTIC_POOL_KIND: POOL_POSE_CHOICE,
                META_SOURCE_SCENE_FIELD: "pose_samples[].poses",
                META_WEIGHT_SCOPE: "pose",
                META_WEIGHT_TOTAL: sum(float(pose["weight"]) for pose in poses),
                META_ZERO_WEIGHT_EXCLUDED: True,
            },
        )

    def _sampling_domain_nodes(self, raw: Mapping[str, Any], *, target: str) -> list[str]:
        if raw.get("node") is not None:
            return [str(raw.get("node"))]
        if raw.get("nodes") is not None:
            nodes = [str(value) for value in raw.get("nodes") or [] if str(value).strip()]
            if nodes:
                return nodes
            self.plan.add_diagnostic(
                "layout_runtime_sampling_nodes_empty", "nodes must be a non-empty list", target=target
            )
            return []
        region_id = str(raw.get("region") or "").strip()
        if region_id:
            assert self.spec is not None
            for region in self.spec.regions:
                if region.region_id == region_id:
                    return list(region.objects)
            self.plan.add_diagnostic(
                "layout_runtime_sampling_region_unknown",
                "sampling domain references unknown scene region",
                target=target,
                region=region_id,
            )
            return []
        scatter_id = str(raw.get("scatter_group") or "").strip()
        if scatter_id:
            scatter = self._scatter_group_by_id(scatter_id)
            if scatter is not None:
                return list(scatter.objects)
            self.plan.add_diagnostic(
                "layout_runtime_sampling_scatter_group_unknown",
                "sampling domain references unknown scene scatter_group",
                target=target,
                scatter_group=scatter_id,
            )
            return []
        volume_group_id = str(raw.get("volume_group") or "").strip()
        if volume_group_id:
            volume_group = self._volume_group_by_id(volume_group_id)
            if volume_group is not None:
                return list(volume_group.objects)
            self.plan.add_diagnostic(
                "layout_runtime_sampling_volume_group_unknown",
                "sampling domain references unknown scene volume_group",
                target=target,
                volume_group=volume_group_id,
            )
            return []
        self.plan.add_diagnostic(
            "layout_runtime_sampling_target_missing",
            "sampling domain requires node, nodes, region, scatter_group, or volume_group",
            target=target,
        )
        return []

    def _scene_surface_sampling_domain(
        self,
        object_name: str,
        *,
        domain_id: str,
        event: NormalizedRuntimeEvent,
        sample: Mapping[str, Any],
    ) -> PlacementDomain | None:
        assert self.problem is not None
        if object_name not in self.problem.objects:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_object_unknown",
                "sampling domain references object outside layout spec",
                target=domain_id,
                object=object_name,
            )
            return None
        sample_type = str(sample.get("type") or "").strip().lower()
        if sample_type not in {"explicit", "local_jitter", "inherit"}:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_scene_surface_sample_unsupported",
                "scene-surface frame currently requires explicit, local_jitter, or inherit sample",
                target=domain_id,
                sample_type=sample_type,
            )
            return None
        explicit_geometry = str(sample.get("geometry") or "").strip().lower() if sample_type == "explicit" else ""
        if sample_type == "explicit" and explicit_geometry != "rectangle":
            self.plan.add_diagnostic(
                "layout_runtime_sampling_explicit_geometry_unsupported",
                "scene-surface explicit sampling requires geometry=rectangle",
                target=domain_id,
                geometry=explicit_geometry,
            )
            return None
        assert self.domain_exporter is not None
        provider = SupportOnProvider(self.problem)
        relative_to = str(sample.get("relative_to") or "").strip().lower()
        if _sample_has_runtime_yaw(sample):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_runtime_yaw_unsupported",
                "runtime yaw must be authored in the scene domain; task sampling cannot override yaw",
                target=domain_id,
            )
            return None
        scene_yaw = object_scene_yaw_range(self.problem, object_name)
        if sample_type == "inherit":
            if sample_xy_bounds(sample) is not None or sample.get("z") is not None:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_inherit_range_unsupported",
                    "inherit sampling cannot override x/y/z; edit the scene domain or use an explicit runtime domain",
                    target=domain_id,
                )
                return None
            result = self.domain_exporter.export_scene_surface_domain(
                object_name,
                domain_id=domain_id,
                source_event=event.name,
                yaw=scene_yaw,
                sample_type=sample_type,
                use_scene_object_range=True,
            )
            if result.diagnostic is not None:
                self._add_export_diagnostic(result.diagnostic)
                return None
            return result.domain
        elif sample_type == "local_jitter" or relative_to in {"current", "node", "slot", "solved_pose"}:
            if sample_type == "local_jitter":
                shape = str(sample.get("shape") or "rectangle").strip().lower()
                if shape not in {"rectangle", "square", "box", "disk", "circle"}:
                    self.plan.add_diagnostic(
                        "layout_runtime_sampling_local_jitter_shape_unsupported",
                        "local_jitter shape must be rectangle or disk",
                        target=domain_id,
                        shape=shape,
                    )
                    return None
            current_bounds, current_meta = provider.center_bounds(object_name)
            if current_bounds is None:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_current_domain_empty",
                    "current variant/node support domain is empty",
                    target=domain_id,
                    object=object_name,
                    **current_meta,
                )
                return None
            sample_bounds = (
                local_jitter_bounds(current_bounds, sample)
                if sample_type == "local_jitter"
                else relative_sample_bounds(current_bounds, sample)
            )
            meta = {
                "anchor": "domain_center",
                "sample_type": sample_type,
                "shape": str(sample.get("shape") or "rectangle"),
                "current_domain": bounds2d_json(current_bounds),
            }
            if "yaw_delta" in sample:
                meta["yaw_delta"] = list(range_from_pair(sample.get("yaw_delta"), default=(0.0, 0.0)) or (0.0, 0.0))
        else:
            sample_bounds = sample_xy_bounds(sample)
            meta = {"sample_type": sample_type, "geometry": explicit_geometry}
            if "yaw_delta" in sample:
                meta["yaw_delta"] = list(range_from_pair(sample.get("yaw_delta"), default=(0.0, 0.0)) or (0.0, 0.0))
        result = self.domain_exporter.export_scene_surface_domain(
            object_name,
            domain_id=domain_id,
            source_event=event.name,
            yaw=scene_yaw,
            sample_bounds=sample_bounds,
            sample_type=sample_type,
            authoring_frame="scene-surface",
            use_scene_object_range=False,
            meta=meta,
        )
        if result.diagnostic is not None:
            self._add_export_diagnostic(result.diagnostic)
            return None
        return result.domain

    def _scene_surface_sampling_domains(
        self,
        object_names: list[str],
        *,
        domain_id: str,
        event: NormalizedRuntimeEvent,
        sample: Mapping[str, Any],
        raw: Mapping[str, Any],
    ) -> CompiledSamplingDomain | None:
        assert self.problem is not None
        for name in object_names:
            if name not in self.problem.objects:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_object_unknown",
                    "sampling domain references object outside layout spec",
                    target=domain_id,
                    object=name,
                )
                return None
        sample_type = str(sample.get("type") or "").strip().lower()
        if sample_type not in {"inherit", "explicit"}:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_scene_surface_group_sample_unsupported",
                "multi-object scene-surface sampling requires inherit or explicit sample",
                target=domain_id,
                sample_type=sample_type,
            )
            return None
        explicit_geometry = str(sample.get("geometry") or "").strip().lower() if sample_type == "explicit" else ""
        if sample_type == "explicit" and explicit_geometry != "rectangle":
            self.plan.add_diagnostic(
                "layout_runtime_sampling_explicit_geometry_unsupported",
                "multi-object scene-surface explicit sampling requires geometry=rectangle",
                target=domain_id,
                geometry=explicit_geometry,
            )
            return None
        if _sample_has_runtime_yaw(sample):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_runtime_yaw_unsupported",
                "runtime yaw must be authored in the scene domain; task sampling cannot override yaw",
                target=domain_id,
            )
            return None
        if sample_type == "inherit" and (sample_xy_bounds(sample) is not None or sample.get("z") is not None):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_inherit_range_unsupported",
                "inherit sampling cannot override x/y/z; edit the scene domain or use an explicit runtime domain",
                target=domain_id,
            )
            return None
        region = self._region_by_id(str(raw.get("region") or ""))
        scatter = self._scatter_group_by_id(str(raw.get("scatter_group") or ""))
        relation = region or scatter
        surface_id = str(region.surface if region is not None else raw.get("surface") or "").strip()
        if scatter is not None:
            surface_id = str(scatter.surface)
        if not surface_id:
            supports = {str(self.problem.objects[name].support or "") for name in object_names}
            supports.discard("")
            if len(supports) == 1:
                surface_id = next(iter(supports))
        if not surface_id:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_surface_missing",
                "multi-object scene-surface sampling requires a region surface, explicit surface, or common object support",
                target=domain_id,
            )
            return None
        if self.problem.local_surface_parent(surface_id):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_scene_surface_local_surface_unsupported",
                "scene-surface group sampling only supports global support surfaces; use node-surface for object-local surfaces",
                target=domain_id,
                surface=surface_id,
            )
            return None
        range_choices = self._scene_surface_sample_bound_choices(
            raw, sample=sample, surface_id=surface_id, target=domain_id
        )
        if range_choices is None:
            return None
        surface = self.problem.resolve_surface(surface_id, {})
        if surface is None:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_support_missing",
                "sampling surface cannot be resolved",
                target=domain_id,
                surface=surface_id,
            )
            return None
        range_choices = _clip_runtime_surface_range_choices(
            self,
            range_choices,
            effective_bounds=surface.effective_bounds(),
            target=domain_id,
            surface_id=surface_id,
        )
        if range_choices is None:
            return None
        yaw_range = getattr(relation, "yaw", None)
        yaw_by_object = _object_yaw_ranges_from_scene(self.problem, object_names, default=yaw_range)
        yaw_choices_by_object = object_yaw_choices_by_object(self.problem, object_names)
        base_quaternion_by_object = {name: object_base_quaternion(self.problem, name) for name in object_names}
        quaternions = {
            name: (
                base_quaternion_by_object[name]
                if yaw_by_object[name] is not None
                else object_quaternion(self.problem, name)
            )
            for name in object_names
        }
        z_by_object = {
            name: surface_child_z_range(surface, self.records[name], quaternion=quaternions[name])
            for name in object_names
        }
        footprint_by_object = {
            name: projected_footprint_size_with_yaw_range(
                self.records[name].local_bounds, quaternions[name], yaw_by_object[name], surface.x_axis, surface.y_axis
            )
            for name in object_names
        }
        footprint_bounds_by_object = {
            name: projected_footprint_bounds_with_yaw_range(
                self.records[name].local_bounds, quaternions[name], yaw_by_object[name], surface.x_axis, surface.y_axis
            )
            for name in object_names
        }
        relative_directions = self._relative_directions_for_scope(object_names, target=domain_id)
        if relative_directions is None:
            return None
        choice_results: list[tuple[_RuntimeSurfaceRangeChoice, PartitionCasePoolResult]] = []
        for choice_index, choice in enumerate(range_choices):
            result = build_partition_case_pool(
                object_names=object_names,
                domain_id=_range_choice_domain_id(domain_id, choice, choice_index, choice_count=len(range_choices)),
                frame=f"surface:{surface_id}",
                bounds=choice.bounds,
                records=self.records,
                z_by_object=z_by_object,
                footprint_by_object=footprint_by_object,
                footprint_bounds_by_object=footprint_bounds_by_object,
                yaw_by_object=yaw_by_object,
                yaw_choices_by_object=yaw_choices_by_object,
                base_quaternion_by_object=base_quaternion_by_object,
                source_event=event.name,
                relation=relation,
                authoring_frame="scene-surface",
                surface=surface_id,
                accept_cell_fn=(
                    support_cell_accept_fn(
                        surface, self.records, quaternions, footprint_by_object, footprint_bounds_by_object
                    )
                    if _surface_needs_cell_filter(surface)
                    else None
                ),
                yaw=yaw_range,
                relative_directions=relative_directions,
                relation_kind="scatter_group" if scatter is not None else "region_sample",
                seed=self.seed + len(domain_id) * RUNTIME_PARTITION_SEED_STRIDE + choice_index * 4099,
                case_count=max(
                    1,
                    int(
                        sample.get("case_count", sample.get("partition_cases", DEFAULT_RUNTIME_PARTITION_CASES))
                        or DEFAULT_RUNTIME_PARTITION_CASES
                    ),
                ),
                candidate_count=max(
                    MIN_RUNTIME_PARTITION_CANDIDATES,
                    int(
                        sample.get(
                            "candidate_count", sample.get("partition_candidates", DEFAULT_RUNTIME_PARTITION_CANDIDATES)
                        )
                        or DEFAULT_RUNTIME_PARTITION_CANDIDATES
                    ),
                ),
            )
            if result.diagnostic is not None:
                self._add_range_choice_diagnostic(result.diagnostic, choice=choice)
                return None
            choice_results.append((choice, result))
        domains, cases, case_pool_meta = _merge_range_choice_case_pools(choice_results)
        self.plan.assignments.append(
            AssignmentPlan(
                plan_id=str(event.params.get("plan_id") or "runtime").strip() or "runtime",
                mode="case_pool",
                objects=tuple(object_names),
                domains=tuple(domain.domain_id for domain in domains),
                bind=dict(cases[0].get("bind") or {}) if cases else {},
                cases=tuple(cases),
                source_event=event.name,
                meta={
                    "backend_source": "runtime_domain_compiler",
                    "surface": surface_id,
                    "region": region.region_id if region is not None else "",
                    "scatter_group": scatter.group_id if scatter is not None else "",
                    **case_pool_meta,
                },
            )
        )
        parent_domains_by_object = {}
        if cases:
            parent_domains_by_object = {
                str(name): str(domain_id) for name, domain_id in dict(cases[0].get("bind") or {}).items()
            }
        return CompiledSamplingDomain(
            tuple(domains),
            None,
            meta={"assignment_emitted": True, "parent_domains_by_object": parent_domains_by_object},
        )

    def _object_relative_sampling_domain(
        self,
        object_name: str,
        *,
        domain_id: str,
        parent: str,
        event: NormalizedRuntimeEvent,
        sample: Mapping[str, Any],
    ) -> PlacementDomain | None:
        assert self.problem is not None
        if object_name not in self.problem.objects:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_object_unknown",
                "sampling domain references object outside layout spec",
                target=domain_id,
                object=object_name,
            )
            return None
        if parent not in self.problem.objects:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_parent_unknown",
                "object frame parent is outside layout spec",
                target=domain_id,
                parent=parent,
            )
            return None
        sample_type = str(sample.get("type") or "").strip().lower()
        explicit_geometry = str(sample.get("geometry") or "").strip().lower() if sample_type == "explicit" else ""
        if sample_type != "explicit" or explicit_geometry not in {"box", "rectangle"}:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_object_sample_unsupported",
                "object frame requires explicit geometry=box or rectangle sample",
                target=domain_id,
                sample_type=sample_type,
                geometry=explicit_geometry,
            )
            return None
        if _sample_has_runtime_yaw(sample):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_runtime_yaw_unsupported",
                "runtime yaw must be authored in the scene domain; task sampling cannot override yaw",
                target=domain_id,
            )
            return None
        x_range = range_from_pair(sample.get("x"), default=None)
        y_range = range_from_pair(sample.get("y"), default=None)
        if x_range is None or y_range is None:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_xy_missing", "object frame sample requires x and y ranges", target=domain_id
            )
            return None
        yaw = object_scene_yaw_range(self.problem, object_name)
        yaw_choices = object_yaw_choice_payloads(self.problem, object_name)
        quaternion = (
            object_base_quaternion(self.problem, object_name)
            if yaw is not None
            else object_quaternion(self.problem, object_name)
        )
        footprint_bounds = projected_footprint_bounds_with_yaw_range(
            self.records[object_name].local_bounds,
            quaternion,
            yaw,
        )
        footprint = (footprint_bounds.x.length, footprint_bounds.y.length)
        return PlacementDomain(
            domain_id=domain_id,
            object_name=object_name,
            frame=f"object:{parent}",
            geometry_type="rectangle",
            x=x_range,
            y=y_range,
            z=range_from_pair(sample.get("z"), default=(0.0, 0.0)),
            yaw=yaw,
            yaw_choices=yaw_choices,
            base_quaternion=object_base_quaternion(self.problem, object_name),
            meta={
                "source_event": event.name,
                "authoring_frame": "object",
                **yaw_choice_semantic_meta(object_name, yaw_choices),
                "parent": parent,
                "footprint": [footprint[0], footprint[1]],
                "footprint_bounds": bounds2d_json(footprint_bounds),
                "yaw_aware_footprint": yaw is not None,
            },
        )

    def _node_surface_sampling_domains(
        self,
        object_names: list[str],
        *,
        domain_id: str,
        parent: str,
        surface_name: str,
        event: NormalizedRuntimeEvent,
        sample: Mapping[str, Any],
        raw: Mapping[str, Any],
    ) -> CompiledSamplingDomain | None:
        assert self.problem is not None
        for name in object_names:
            if name not in self.problem.objects:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_object_unknown",
                    "sampling domain references object outside layout spec",
                    target=domain_id,
                    object=name,
                )
                return None
        if parent not in self.problem.objects:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_parent_unknown",
                "node-surface parent is outside layout spec",
                target=domain_id,
                parent=parent,
            )
            return None
        surface_id = surface_name if "." in surface_name else f"{parent}.{surface_name}"
        raw_local_surface = self.problem.local_surfaces.get(surface_id)
        local_surface = _runtime_scaled_local_surface(raw_local_surface, parent=parent, records=self.records)
        if local_surface is None:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_local_surface_unknown",
                "node-surface frame references unknown local support surface",
                target=domain_id,
                surface=surface_id,
            )
            return None
        sample_type = str(sample.get("type") or "").strip().lower()
        explicit_geometry = str(sample.get("geometry") or "").strip().lower() if sample_type == "explicit" else ""
        if sample_type not in {"inherit", "explicit"}:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_node_surface_sample_unsupported",
                "node-surface frame requires inherit or explicit sample",
                target=domain_id,
                sample_type=sample_type,
            )
            return None
        if sample_type == "explicit" and explicit_geometry != "rectangle":
            self.plan.add_diagnostic(
                "layout_runtime_sampling_explicit_geometry_unsupported",
                "node-surface explicit sampling requires geometry=rectangle",
                target=domain_id,
                geometry=explicit_geometry,
            )
            return None
        if _sample_has_runtime_yaw(sample):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_runtime_yaw_unsupported",
                "runtime yaw must be authored in the scene domain; task sampling cannot override yaw",
                target=domain_id,
            )
            return None
        if sample_type == "inherit" and (sample_xy_bounds(sample) is not None or sample.get("z") is not None):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_inherit_range_unsupported",
                "inherit sampling cannot override x/y/z; edit the scene domain or use an explicit runtime domain",
                target=domain_id,
            )
            return None
        range_choices = self._node_surface_sample_bound_choices(
            raw,
            sample=sample,
            surface_id=surface_id,
            target=domain_id,
            object_names=object_names,
            fallback_bounds=local_surface.effective_bounds(),
        )
        if range_choices is None:
            return None
        range_choices = _clip_runtime_surface_range_choices(
            self,
            range_choices,
            effective_bounds=local_surface.effective_bounds(),
            target=domain_id,
            surface_id=surface_id,
        )
        if range_choices is None:
            return None
        offset_choices: list[_RuntimeSurfaceRangeChoice] = []
        for choice in range_choices:
            local_bounds = choice.bounds
            if len(object_names) == 1:
                name = object_names[0]
                local_bounds = _local_surface_center_fit_bounds(
                    local_surface,
                    local_bounds,
                    record=self.records[name],
                    quaternion=object_base_quaternion(self.problem, name),
                    yaw=object_scene_yaw_range(self.problem, name),
                )
                if local_bounds is None:
                    self.plan.add_diagnostic(
                        "layout_runtime_sampling_node_surface_fit_empty",
                        "node-surface range cannot fit the child footprint",
                        target=domain_id,
                        object=name,
                        surface=surface_id,
                        surface_range_choice=choice.choice_id,
                    )
                    return None
            offset_bounds = surface_bounds_to_parent_xy_bounds(local_surface, local_bounds)
            if offset_bounds is None:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_node_surface_frame_unsupported",
                    "node-surface runtime sampling currently requires a parent-local horizontal surface",
                    target=domain_id,
                    surface=surface_id,
                    surface_range_choice=choice.choice_id,
                )
                return None
            offset_choices.append(replace(choice, bounds=offset_bounds))
        region = self._region_by_id(str(raw.get("region") or ""))
        scatter = self._scatter_group_by_id(str(raw.get("scatter_group") or ""))
        relation = region or scatter
        if relation is not None and sample_type == "inherit" and len(object_names) > 1:
            yaw_range = getattr(relation, "yaw", None)
            yaw_by_object = _object_yaw_ranges_from_scene(self.problem, object_names, default=yaw_range)
            yaw_choices_by_object = object_yaw_choices_by_object(self.problem, object_names)
            quaternions = {
                name: (
                    object_base_quaternion(self.problem, name)
                    if yaw_by_object[name] is not None
                    else object_quaternion(self.problem, name)
                )
                for name in object_names
            }
            base_quaternion_by_object = {name: object_base_quaternion(self.problem, name) for name in object_names}
            z_by_object = {
                name: local_surface_child_z_range(local_surface, self.records[name], quaternion=quaternions[name])
                for name in object_names
            }
            footprint_by_object = {
                name: projected_footprint_size_with_yaw_range(
                    self.records[name].local_bounds,
                    quaternions[name],
                    yaw_by_object[name],
                    local_surface.x_axis,
                    local_surface.y_axis,
                )
                for name in object_names
            }
            footprint_bounds_by_object = {
                name: projected_footprint_bounds_with_yaw_range(
                    self.records[name].local_bounds,
                    quaternions[name],
                    yaw_by_object[name],
                    local_surface.x_axis,
                    local_surface.y_axis,
                )
                for name in object_names
            }
            relative_directions = self._relative_directions_for_scope(object_names, target=domain_id)
            if relative_directions is None:
                return None
            choice_results: list[tuple[_RuntimeSurfaceRangeChoice, PartitionCasePoolResult]] = []
            for choice_index, choice in enumerate(offset_choices):
                result = build_partition_case_pool(
                    object_names=object_names,
                    domain_id=_range_choice_domain_id(
                        domain_id, choice, choice_index, choice_count=len(offset_choices)
                    ),
                    frame=f"object:{parent}",
                    parent=parent,
                    bounds=choice.bounds,
                    records=self.records,
                    z_by_object=z_by_object,
                    footprint_by_object=footprint_by_object,
                    footprint_bounds_by_object=footprint_bounds_by_object,
                    yaw_by_object=yaw_by_object,
                    yaw_choices_by_object=yaw_choices_by_object,
                    base_quaternion_by_object=base_quaternion_by_object,
                    source_event=event.name,
                    relation=relation,
                    authoring_frame="node-surface",
                    accept_cell_fn=(
                        support_cell_accept_fn(
                            local_surface, self.records, quaternions, footprint_by_object, footprint_bounds_by_object
                        )
                        if _surface_needs_cell_filter(local_surface)
                        else None
                    ),
                    yaw=yaw_range,
                    relative_directions=relative_directions,
                    relation_kind="scatter_group" if scatter is not None else "region_sample",
                    seed=self.seed + len(domain_id) * RUNTIME_PARTITION_SEED_STRIDE + choice_index * 4099,
                    case_count=max(
                        1,
                        int(
                            sample.get("case_count", sample.get("partition_cases", DEFAULT_RUNTIME_PARTITION_CASES))
                            or DEFAULT_RUNTIME_PARTITION_CASES
                        ),
                    ),
                    candidate_count=max(
                        MIN_RUNTIME_PARTITION_CANDIDATES,
                        int(
                            sample.get(
                                "candidate_count",
                                sample.get("partition_candidates", DEFAULT_RUNTIME_PARTITION_CANDIDATES),
                            )
                            or DEFAULT_RUNTIME_PARTITION_CANDIDATES
                        ),
                    ),
                )
                if result.diagnostic is not None:
                    self._add_range_choice_diagnostic(result.diagnostic, choice=choice)
                    return None
                choice_results.append((choice, result))
            domains, cases, case_pool_meta = _merge_range_choice_case_pools(choice_results)
            return CompiledSamplingDomain(
                domains=tuple(domains),
                parent=parent,
                cases=tuple(cases),
                meta={
                    META_PLAN_KIND: PLAN_PARTITION_CASE_POOL,
                    "region": region.region_id if region is not None else "",
                    "scatter_group": scatter.group_id if scatter is not None else "",
                    **case_pool_meta,
                },
            )
        if len(object_names) == 1 and any(choice.source == "surface_ranges" for choice in offset_choices):
            name = object_names[0]
            yaw_choices = object_yaw_choice_payloads(self.problem, name)
            z = local_surface_child_z_range(
                local_surface, self.records[name], quaternion=object_quaternion(self.problem, name)
            )
            choices = tuple(
                {
                    "id": choice.choice_id,
                    "weight": float(choice.weight),
                    "x": [float(choice.bounds.x.min), float(choice.bounds.x.max)],
                    "y": [float(choice.bounds.y.min), float(choice.bounds.y.max)],
                    "z": [float(z[0]), float(z[1])],
                    "geometry_type": "rectangle",
                    "source": choice.source,
                }
                for choice in offset_choices
            )
            return CompiledSamplingDomain(
                (
                    PlacementDomain(
                        domain_id=domain_id,
                        object_name=name,
                        frame=f"object:{parent}",
                        geometry_type="surface_range_choice",
                        surface=surface_id,
                        z=z,
                        yaw=object_scene_yaw_range(self.problem, name),
                        yaw_choices=yaw_choices,
                        base_quaternion=object_base_quaternion(self.problem, name),
                        range_choices=choices,
                        meta={
                            "source_event": event.name,
                            "authoring_frame": "node-surface",
                            **yaw_choice_semantic_meta(name, yaw_choices),
                            "parent": parent,
                            "surface": surface_id,
                            "region": raw.get("region", ""),
                            "domain_index": 0,
                            "domain_encoding": "surface_range_choice",
                            META_PLAN_KIND: POOL_SURFACE_RANGE_CHOICE,
                            META_SEMANTIC_POOL: True,
                            META_SEMANTIC_POOL_KIND: POOL_SURFACE_RANGE_CHOICE,
                            META_SOURCE_SCENE_FIELD: "objects[].surface_ranges",
                            META_WEIGHT_SCOPE: "choice",
                            META_WEIGHT_TOTAL: sum(float(choice["weight"]) for choice in choices),
                            META_ZERO_WEIGHT_EXCLUDED: True,
                            "choice_count": len(choices),
                        },
                    ),
                ),
                parent,
            )
        if len(offset_choices) != 1:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_node_surface_weighted_range_requires_group",
                "node-surface weighted surface_ranges require a relation group case_pool",
                target=domain_id,
                choice_count=len(offset_choices),
            )
            return None
        offset_bounds = offset_choices[0].bounds
        out: list[PlacementDomain] = []
        for index, name in enumerate(object_names):
            yaw_choices = object_yaw_choice_payloads(self.problem, name)
            z = local_surface_child_z_range(
                local_surface, self.records[name], quaternion=object_quaternion(self.problem, name)
            )
            out.append(
                PlacementDomain(
                    domain_id=f"{domain_id}:{name}" if len(object_names) > 1 else domain_id,
                    object_name=name,
                    frame=f"object:{parent}",
                    geometry_type="rectangle",
                    x=(offset_bounds.x.min, offset_bounds.x.max),
                    y=(offset_bounds.y.min, offset_bounds.y.max),
                    z=z,
                    yaw=object_scene_yaw_range(self.problem, name),
                    yaw_choices=yaw_choices,
                    base_quaternion=object_base_quaternion(self.problem, name),
                    meta={
                        "source_event": event.name,
                        "authoring_frame": "node-surface",
                        **yaw_choice_semantic_meta(name, yaw_choices),
                        "parent": parent,
                        "surface": surface_id,
                        "region": raw.get("region", ""),
                        "domain_index": index,
                    },
                )
            )
        return CompiledSamplingDomain(tuple(out), parent)

    def _object_volume_sampling_domains(
        self,
        object_names: list[str],
        *,
        domain_id: str,
        parent: str,
        volume_name: str,
        event: NormalizedRuntimeEvent,
        sample: Mapping[str, Any],
        volume_group_id: str = "",
    ) -> CompiledSamplingDomain | None:
        assert self.problem is not None
        for name in object_names:
            if name not in self.problem.objects:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_object_unknown",
                    "sampling domain references object outside layout spec",
                    target=domain_id,
                    object=name,
                )
                return None
        if parent not in self.problem.objects:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_parent_unknown",
                "object-volume parent is outside layout spec",
                target=domain_id,
                parent=parent,
            )
            return None
        volume_id = volume_name if "." in volume_name else f"{parent}.{volume_name}"
        volume = self.problem.spec.volumes_by_id.get(volume_id)
        if volume is None:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_volume_unknown",
                "object-volume frame references unknown volume",
                target=domain_id,
                volume=volume_id,
            )
            return None
        if volume_parent(volume) != parent:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_volume_parent_mismatch",
                "object-volume frame parent must match the volume parent",
                target=domain_id,
                volume=volume_id,
                parent=parent,
                volume_parent=volume_parent(volume),
            )
            return None
        sample_type = str(sample.get("type") or "").strip().lower()
        explicit_geometry = str(sample.get("geometry") or "").strip().lower() if sample_type == "explicit" else ""
        if sample_type not in {"inherit", "explicit"}:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_object_volume_sample_unsupported",
                "object-volume frame requires inherit or explicit sample",
                target=domain_id,
                sample_type=sample_type,
            )
            return None
        if sample_type == "explicit" and explicit_geometry not in {"box", "cylinder", "volume"}:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_explicit_geometry_unsupported",
                "object-volume explicit sampling requires geometry=box, cylinder, or volume",
                target=domain_id,
                geometry=explicit_geometry,
            )
            return None
        if sample_type == "inherit" and any(axis in sample for axis in ("x", "y", "z")):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_inherit_range_unsupported",
                "inherit sampling cannot override x/y/z; edit the scene volume or use an explicit runtime domain",
                target=domain_id,
            )
            return None
        if _sample_has_runtime_yaw(sample):
            self.plan.add_diagnostic(
                "layout_runtime_sampling_runtime_yaw_unsupported",
                "runtime yaw must be authored in the scene domain; task sampling cannot override yaw",
                target=domain_id,
            )
            return None
        volume_group = self._volume_group_by_id(volume_group_id)
        raw_group_id = volume_group.group_id if volume_group is not None else ""
        if volume_group is None and self.spec is not None:
            object_set = set(object_names)
            for group in self.spec.volume_groups:
                if str(group.volume) == volume_id and set(group.objects) == object_set:
                    volume_group = group
                    raw_group_id = group.group_id
                    break
        default_yaw = getattr(volume_group, "yaw", None)
        yaw_by_object = _object_yaw_ranges_from_scene(self.problem, object_names, default=default_yaw)
        yaw_choices_by_object = object_yaw_choices_by_object(self.problem, object_names)
        if _volume_group_requires_case_pool(volume_group, object_names):
            return self._object_volume_partition_domains(
                object_names,
                domain_id=domain_id,
                parent=parent,
                volume=volume,
                volume_id=volume_id,
                volume_group=volume_group,
                event=event,
                sample=sample,
                yaw_by_object=yaw_by_object,
                yaw_choices_by_object=yaw_choices_by_object,
                raw_group_id=raw_group_id,
            )
        out: list[PlacementDomain] = []
        for index, name in enumerate(object_names):
            yaw_choices = yaw_choices_by_object.get(name, ())
            domain_geometry = runtime_volume_domain_geometry(
                volume, self.records[name].local_bounds, mode=str(sample.get("mode") or "fit")
            )
            if domain_geometry is None:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_volume_capacity_insufficient",
                    "object does not fit in runtime object volume",
                    target=domain_id,
                    object=name,
                    volume=volume_id,
                )
                return None
            geometry_type, bounds, meta = domain_geometry
            out.append(
                PlacementDomain(
                    domain_id=f"{domain_id}:{name}" if len(object_names) > 1 else domain_id,
                    object_name=name,
                    frame=f"object:{parent}",
                    geometry_type=geometry_type,
                    x=(bounds.x.min, bounds.x.max),
                    y=(bounds.y.min, bounds.y.max),
                    z=(bounds.z.min, bounds.z.max),
                    yaw=yaw_by_object[name],
                    yaw_choices=yaw_choices,
                    base_quaternion=object_base_quaternion(self.problem, name),
                    meta={
                        "source_event": event.name,
                        "authoring_frame": "object-volume",
                        **yaw_choice_semantic_meta(name, yaw_choices),
                        "parent": parent,
                        "volume_group": raw_group_id,
                        **meta,
                        "domain_index": index,
                    },
                )
            )
        return CompiledSamplingDomain(tuple(out), parent, meta={"plan_kind": "volume_group", "volume": volume_id})

    def _object_volume_partition_domains(
        self,
        object_names: list[str],
        *,
        domain_id: str,
        parent: str,
        volume: Any,
        volume_id: str,
        volume_group: Any,
        event: NormalizedRuntimeEvent,
        sample: Mapping[str, Any],
        yaw_by_object: Mapping[str, tuple[float, float] | None],
        yaw_choices_by_object: Mapping[str, tuple[dict[str, Any], ...]],
        raw_group_id: str,
    ) -> CompiledSamplingDomain | None:
        fit_bounds_by_object = {}
        geometry_type_by_object = {}
        for name in object_names:
            domain_geometry = runtime_volume_domain_geometry(
                volume, self.records[name].local_bounds, mode=str(sample.get("mode") or "fit")
            )
            if domain_geometry is None:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_volume_capacity_insufficient",
                    "object does not fit in runtime object volume",
                    target=domain_id,
                    object=name,
                    volume=volume_id,
                )
                return None
            geometry_type, bounds, _meta = domain_geometry
            geometry_type_by_object[name] = geometry_type
            fit_bounds_by_object[name] = bounds

        common_bounds = _intersect_volume_xy_bounds(fit_bounds_by_object.values())
        if common_bounds is None:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_volume_group_common_domain_empty",
                "volume group members do not share a common runtime xy domain",
                target=domain_id,
                volume_group=raw_group_id,
                volume=volume_id,
            )
            return None

        base_quaternion_by_object = {name: object_base_quaternion(self.problem, name) for name in object_names}
        quaternions = {
            name: (
                base_quaternion_by_object[name]
                if yaw_by_object[name] is not None
                else object_quaternion(self.problem, name)
            )
            for name in object_names
        }
        footprint_by_object = {
            name: projected_footprint_size_with_yaw_range(
                self.records[name].local_bounds, quaternions[name], yaw_by_object[name]
            )
            for name in object_names
        }
        footprint_bounds_by_object = {
            name: projected_footprint_bounds_with_yaw_range(
                self.records[name].local_bounds, quaternions[name], yaw_by_object[name]
            )
            for name in object_names
        }
        z_by_object = {
            name: (fit_bounds_by_object[name].z.min, fit_bounds_by_object[name].z.max) for name in object_names
        }
        result = build_partition_case_pool(
            object_names=object_names,
            domain_id=domain_id,
            frame=f"object:{parent}",
            parent=parent,
            bounds=common_bounds,
            records=self.records,
            z_by_object=z_by_object,
            footprint_by_object=footprint_by_object,
            footprint_bounds_by_object=footprint_bounds_by_object,
            yaw_by_object=yaw_by_object,
            yaw_choices_by_object=yaw_choices_by_object,
            base_quaternion_by_object=base_quaternion_by_object,
            source_event=event.name,
            relation=None,
            authoring_frame="object-volume",
            accept_cell_fn=_volume_accept_cell_fn(volume, fit_bounds_by_object),
            yaw=getattr(volume_group, "yaw", None),
            relation_kind="volume_group",
            min_spacing=max(0.0, float(getattr(volume_group, "min_spacing", 0.0) or 0.0)),
            sample_method=str(getattr(volume_group, "mode", "random") or "random"),
            seed=self.seed + len(domain_id) * RUNTIME_PARTITION_SEED_STRIDE,
            case_count=max(
                1,
                int(
                    sample.get("case_count", sample.get("partition_cases", DEFAULT_RUNTIME_PARTITION_CASES))
                    or DEFAULT_RUNTIME_PARTITION_CASES
                ),
            ),
            candidate_count=max(
                MIN_RUNTIME_PARTITION_CANDIDATES,
                int(
                    sample.get(
                        "candidate_count", sample.get("partition_candidates", DEFAULT_RUNTIME_PARTITION_CANDIDATES)
                    )
                    or DEFAULT_RUNTIME_PARTITION_CANDIDATES
                ),
            ),
        )
        if result.diagnostic is not None:
            self._add_export_diagnostic(result.diagnostic)
            return None
        return CompiledSamplingDomain(
            tuple(result.domains),
            parent,
            cases=tuple(result.cases),
            meta={
                META_PLAN_KIND: PLAN_PARTITION_CASE_POOL,
                "volume": volume_id,
                "volume_group": raw_group_id,
                "volume_geometries": geometry_type_by_object,
                **dict(result.meta or {}),
            },
        )

    def _region_by_id(self, region_id: str) -> Any | None:
        if not region_id:
            return None
        assert self.spec is not None
        for region in self.spec.regions:
            if region.region_id == region_id:
                return region
        return None

    def _scatter_group_by_id(self, group_id: str) -> Any | None:
        if not group_id:
            return None
        assert self.spec is not None
        for group in self.spec.scatter_groups:
            if group.group_id == group_id:
                return group
        return None

    def _volume_group_by_id(self, group_id: str) -> Any | None:
        if not group_id:
            return None
        assert self.spec is not None
        for group in self.spec.volume_groups:
            if group.group_id == group_id:
                return group
        return None

    def _relative_directions_for_scope(self, object_names: list[str], *, target: str) -> tuple[Any, ...] | None:
        if self.spec is None:
            return ()
        object_set = set(object_names)
        out = []
        for rel in self.spec.relative_directions:
            left_in = rel.node_1 in object_set
            right_in = rel.node_2 in object_set
            if not left_in and not right_in:
                continue
            if left_in and right_in:
                out.append(rel)
                continue
            self.plan.add_diagnostic(
                "layout_runtime_sampling_relative_direction_scope_unsupported",
                "runtime compile only supports relative_direction within one inherited group/scope",
                target=target,
                node_1=rel.node_1,
                node_2=rel.node_2,
                direction=rel.direction,
            )
            return None
        return tuple(out)

    def _add_range_choice_diagnostic(self, diagnostic: Any, *, choice: _RuntimeSurfaceRangeChoice) -> None:
        code = str(getattr(diagnostic, "code", ""))
        message = str(getattr(diagnostic, "message", ""))
        if code == "partition_case_pool_no_candidate":
            code = "runtime_relation_capacity_insufficient"
            message = "relation surface range cannot generate any runtime-safe partition case"
        self.plan.add_diagnostic(
            code,
            message,
            target=diagnostic.target,
            severity=diagnostic.severity,
            **{
                **dict(diagnostic.meta),
                "source_diagnostic_code": getattr(diagnostic, "code", ""),
                "surface_range_choice": choice.choice_id,
                "surface_range_choice_weight": choice.weight,
                "surface_range_choice_bounds": bounds2d_json(choice.bounds),
            },
        )

    def _node_surface_sample_bound_choices(
        self,
        raw: Mapping[str, Any],
        *,
        sample: Mapping[str, Any],
        surface_id: str,
        target: str,
        object_names: list[str],
        fallback_bounds: Bounds2D | None,
    ) -> list[_RuntimeSurfaceRangeChoice] | None:
        explicit = sample_xy_bounds(sample)
        if explicit is not None:
            return [_RuntimeSurfaceRangeChoice(choice_id="explicit", bounds=explicit, source="explicit")]
        region_id = str(raw.get("region") or "").strip()
        if region_id:
            assert self.spec is not None
            for region in self.spec.regions:
                if region.region_id == region_id:
                    if region.surface != surface_id:
                        self.plan.add_diagnostic(
                            "layout_runtime_sampling_region_surface_mismatch",
                            "sampling region is not authored on the requested node-surface frame",
                            target=target,
                            region=region_id,
                            region_surface=region.surface,
                            frame_surface=surface_id,
                        )
                        return None
                    return self._relation_surface_range_choices(
                        region,
                        target=target,
                        relation_kind="region",
                        relation_id=region_id,
                    )
        scatter_id = str(raw.get("scatter_group") or "").strip()
        if scatter_id:
            scatter = self._scatter_group_by_id(scatter_id)
            if scatter is None:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_scatter_group_unknown",
                    "sampling domain references unknown scene scatter_group",
                    target=target,
                    scatter_group=scatter_id,
                )
                return None
            if scatter.surface != surface_id:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_scatter_surface_mismatch",
                    "sampling scatter_group is not authored on the requested node-surface frame",
                    target=target,
                    scatter_group=scatter_id,
                    scatter_surface=scatter.surface,
                    frame_surface=surface_id,
                )
                return None
            return self._relation_surface_range_choices(
                scatter,
                target=target,
                relation_kind="scatter_group",
                relation_id=scatter_id,
            )
        if len(object_names) == 1 and fallback_bounds is not None:
            return self._object_surface_range_choices(
                object_names[0],
                fallback_bounds=fallback_bounds,
                target=target,
            )
        self.plan.add_diagnostic(
            "layout_runtime_sampling_xy_missing",
            "sampling domain requires x/y ranges, an object surface_range, or a region/scatter_group with surface_range",
            target=target,
        )
        return None

    def _object_surface_range_choices(
        self,
        object_name: str,
        *,
        fallback_bounds: Bounds2D,
        target: str,
    ) -> list[_RuntimeSurfaceRangeChoice] | None:
        assert self.problem is not None
        obj = self.problem.objects[object_name]
        if obj.override_surface_range is not None:
            bounds = fallback_bounds.intersect(obj.override_surface_range.to_bounds(fallback_bounds))
            if bounds is None:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_object_range_empty",
                    "object surface_range override does not intersect the local support surface",
                    target=target,
                    object=object_name,
                )
                return None
            return [_RuntimeSurfaceRangeChoice(choice_id="override", bounds=bounds, source="override")]
        if obj.surface_ranges:
            choices = []
            for choice in obj.surface_ranges:
                if float(choice.weight) <= 0.0:
                    continue
                bounds = fallback_bounds.intersect(choice.surface_range.to_bounds(fallback_bounds))
                if bounds is None:
                    continue
                choices.append(
                    _RuntimeSurfaceRangeChoice(
                        choice_id=str(choice.range_id),
                        bounds=bounds,
                        weight=float(choice.weight),
                        source="surface_ranges",
                    )
                )
            if choices:
                return choices
            self.plan.add_diagnostic(
                "layout_runtime_sampling_weighted_surface_ranges_empty",
                "weighted object surface_ranges produce no valid local support choices",
                target=target,
                object=object_name,
            )
            return None
        bounds = fallback_bounds
        if obj.surface_range is not None:
            bounds = fallback_bounds.intersect(obj.surface_range.to_bounds(fallback_bounds))
        if bounds is None:
            self.plan.add_diagnostic(
                "layout_runtime_sampling_object_range_empty",
                "object surface_range does not intersect the local support surface",
                target=target,
                object=object_name,
            )
            return None
        return [
            _RuntimeSurfaceRangeChoice(
                choice_id="surface_range" if obj.surface_range is not None else "surface",
                bounds=bounds,
                source="surface_range" if obj.surface_range is not None else "surface",
            )
        ]

    def _scene_surface_sample_bound_choices(
        self,
        raw: Mapping[str, Any],
        *,
        sample: Mapping[str, Any],
        surface_id: str,
        target: str,
    ) -> list[_RuntimeSurfaceRangeChoice] | None:
        explicit = sample_xy_bounds(sample)
        if explicit is not None:
            return [_RuntimeSurfaceRangeChoice(choice_id="explicit", bounds=explicit, source="explicit")]
        region_id = str(raw.get("region") or "").strip()
        if region_id:
            assert self.spec is not None
            for region in self.spec.regions:
                if region.region_id == region_id:
                    if region.surface != surface_id:
                        self.plan.add_diagnostic(
                            "layout_runtime_sampling_region_surface_mismatch",
                            "sampling region is not authored on the requested scene-surface frame",
                            target=target,
                            region=region_id,
                            region_surface=region.surface,
                            frame_surface=surface_id,
                        )
                        return None
                    return self._relation_surface_range_choices(
                        region,
                        target=target,
                        relation_kind="region",
                        relation_id=region_id,
                    )
        scatter_id = str(raw.get("scatter_group") or "").strip()
        if scatter_id:
            scatter = self._scatter_group_by_id(scatter_id)
            if scatter is None:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_scatter_group_unknown",
                    "sampling domain references unknown scene scatter_group",
                    target=target,
                    scatter_group=scatter_id,
                )
                return None
            if scatter.surface != surface_id:
                self.plan.add_diagnostic(
                    "layout_runtime_sampling_scatter_surface_mismatch",
                    "sampling scatter_group is not authored on the requested scene-surface frame",
                    target=target,
                    scatter_group=scatter_id,
                    scatter_surface=scatter.surface,
                    frame_surface=surface_id,
                )
                return None
            return self._relation_surface_range_choices(
                scatter,
                target=target,
                relation_kind="scatter_group",
                relation_id=scatter_id,
            )
        self.plan.add_diagnostic(
            "layout_runtime_sampling_xy_missing",
            "sampling domain requires x/y ranges or a region/scatter_group with surface_range",
            target=target,
        )
        return None

    def _relation_surface_range_choices(
        self,
        relation: Any,
        *,
        target: str,
        relation_kind: str,
        relation_id: str,
    ) -> list[_RuntimeSurfaceRangeChoice] | None:
        code_kind = "scatter" if relation_kind == "scatter_group" else relation_kind
        choices = tuple(getattr(relation, "surface_ranges", ()) or ())
        if choices:
            out = [
                _RuntimeSurfaceRangeChoice(
                    choice_id=str(choice.range_id or f"{relation_id}_range_{index}"),
                    bounds=Bounds2D(choice.surface_range.x, choice.surface_range.y),
                    weight=float(choice.weight),
                    source="surface_ranges",
                )
                for index, choice in enumerate(choices)
                if float(choice.weight) > 0.0
            ]
            if out:
                return out
            self.plan.add_diagnostic(
                f"layout_runtime_sampling_{code_kind}_ranges_empty",
                "scene relation surface_ranges has no positive-weight choice to inherit",
                target=target,
                **{relation_kind: relation_id},
            )
            return None
        surface_range = getattr(relation, "surface_range", None)
        if surface_range is None:
            self.plan.add_diagnostic(
                f"layout_runtime_sampling_{code_kind}_range_missing",
                "scene relation has no surface_range to inherit",
                target=target,
                **{relation_kind: relation_id},
            )
            return None
        return [
            _RuntimeSurfaceRangeChoice(
                choice_id="surface_range",
                bounds=Bounds2D(surface_range.x, surface_range.y),
                source="surface_range",
            )
        ]


def _range_choice_domain_id(
    domain_id: str,
    choice: _RuntimeSurfaceRangeChoice,
    index: int,
    *,
    choice_count: int,
) -> str:
    if choice_count <= 1:
        return domain_id
    return f"{domain_id}:range_{index:03d}_{safe_id(choice.choice_id)}"


def _layout_case_weight(case: LayoutCase) -> float:
    try:
        return max(0.0, float(case.params.get("weight", 1.0)))
    except (TypeError, ValueError):
        return 0.0


def _merge_range_choice_case_pools(
    results: list[tuple[_RuntimeSurfaceRangeChoice, PartitionCasePoolResult]],
) -> tuple[tuple[PlacementDomain, ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    if not results:
        return (), (), {}
    semantic_choices = any(choice.source == "surface_ranges" for choice, _ in results)
    if len(results) == 1 and not semantic_choices:
        _, result = results[0]
        return tuple(result.domains), tuple(result.cases), dict(result.meta or {})

    domains: list[PlacementDomain] = []
    cases: list[dict[str, Any]] = []
    candidate_tiers: Counter[str] = Counter()
    selected_tiers: Counter[str] = Counter()
    strategy_policy: set[str] = set()
    candidate_count = 0
    range_case_counts: dict[str, int] = {}
    range_weights: dict[str, float] = {}
    range_bounds: dict[str, dict[str, list[float]]] = {}
    base_meta = dict(results[0][1].meta or {})

    for choice, result in results:
        choice_case_count = len(result.cases)
        range_case_counts[choice.choice_id] = choice_case_count
        range_weights[choice.choice_id] = float(choice.weight)
        range_bounds[choice.choice_id] = bounds2d_json(choice.bounds)
        result_meta = dict(result.meta or {})
        candidate_count += int(result_meta.get("candidate_count") or 0)
        _update_counter(candidate_tiers, result_meta.get("candidate_tiers"))
        strategy_policy.update(str(item) for item in result_meta.get("strategy_policy", []) or [])
        case_weight = (float(choice.weight) / float(choice_case_count)) if choice_case_count else 0.0
        for domain in result.domains:
            domains.append(
                replace(
                    domain,
                    meta={
                        **dict(domain.meta or {}),
                        "surface_range_choice": choice.choice_id,
                        "surface_range_choice_weight": float(choice.weight),
                    },
                )
            )
        for case in result.cases:
            payload = dict(case)
            selected_tiers[str(payload.get("tier") or "")] += 1
            payload["id"] = f"{safe_id(choice.choice_id)}:{payload.get('id', 'partition_case')}"
            payload["surface_range_choice"] = choice.choice_id
            payload["surface_range_choice_weight"] = float(choice.weight)
            payload["surface_range_choice_bounds"] = bounds2d_json(choice.bounds)
            payload["weight"] = case_weight
            cases.append(payload)

    selected_tiers.pop("", None)
    base_meta.update(
        {
            "surface_range_choice_pool": True,
            META_SEMANTIC_POOL: True,
            META_SEMANTIC_POOL_KIND: POOL_RELATION_SURFACE_RANGE_CHOICE,
            META_SOURCE_SCENE_FIELD: "regions[].surface_ranges/scatter_groups[].surface_ranges",
            META_WEIGHT_SCOPE: "range_group",
            META_WEIGHT_TOTAL: sum(weight for weight in range_weights.values() if weight > 0.0),
            META_ZERO_WEIGHT_EXCLUDED: True,
            "range_choice_count": len(results),
            "range_choice_weights": range_weights,
            "range_choice_case_counts": range_case_counts,
            "range_choice_bounds": range_bounds,
            "candidate_count": candidate_count,
            "candidate_tiers": _counter_payload(candidate_tiers),
            "selected_count": len(cases),
            "selected_tiers": _counter_payload(selected_tiers),
            "strategy_policy": sorted(strategy_policy),
        }
    )
    return tuple(domains), tuple(cases), base_meta


def _update_counter(counter: Counter[str], raw: Any) -> None:
    if not isinstance(raw, Mapping):
        return
    for key, value in raw.items():
        try:
            counter[str(key)] += int(value)
        except (TypeError, ValueError):
            continue


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _sample_has_runtime_yaw(sample: Mapping[str, Any]) -> bool:
    if "yaw" in sample:
        return True
    overrides = sample.get("object_overrides")
    if not isinstance(overrides, Mapping):
        return False
    return any(isinstance(raw, Mapping) and "yaw" in raw for raw in overrides.values())


def _object_yaw_ranges_from_scene(
    problem: Any,
    object_names: list[str],
    *,
    default: tuple[float, float] | None,
) -> dict[str, tuple[float, float] | None]:
    out: dict[str, tuple[float, float] | None] = {}
    for name in object_names:
        out[name] = object_scene_yaw_range(problem, name) or default
    return out


def _surface_needs_cell_filter(surface: Any) -> bool:
    return getattr(surface, "polygon", None) is not None or str(
        getattr(surface, "shape", "rectangle") or "rectangle"
    ).lower() not in {"", "rectangle"}


def _local_surface_center_fit_bounds(
    surface: ManualSupportSurface,
    authored_bounds: Bounds2D,
    *,
    record: Any,
    quaternion: tuple[float, float, float, float],
    yaw: tuple[float, float] | None,
) -> Bounds2D | None:
    base_bounds = surface.effective_bounds()
    if base_bounds is None:
        return None
    footprint = projected_footprint_bounds_with_yaw_range(
        record.local_bounds,
        quaternion,
        yaw,
        surface.x_axis,
        surface.y_axis,
    )
    fit_bounds = Bounds2D(
        Range1D(base_bounds.x.min - footprint.x.min, base_bounds.x.max - footprint.x.max),
        Range1D(base_bounds.y.min - footprint.y.min, base_bounds.y.max - footprint.y.max),
    )
    return authored_bounds.intersect(fit_bounds)


def _runtime_scaled_local_surface(
    surface: Any | None,
    *,
    parent: str,
    records: Mapping[str, Any],
) -> ManualSupportSurface | None:
    if surface is None:
        return None
    parent_record = records.get(parent)
    scale = tuple(float(v) for v in getattr(parent_record, "scale", (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0))
    center = _scale_vec3(getattr(surface, "center"), scale)
    x_axis = _scale_vec3(getattr(surface, "x_axis"), scale)
    y_axis = _scale_vec3(getattr(surface, "y_axis"), scale)
    x_scale = _vec_norm(x_axis)
    y_scale = _vec_norm(y_axis)
    return ManualSupportSurface.from_spec(
        SimpleNamespace(
            surface_id=getattr(surface, "surface_id"),
            center=center,
            x_axis=x_axis,
            y_axis=y_axis,
            normal=getattr(surface, "normal", None),
            bounds=_scale_bounds2d(getattr(surface, "bounds"), x_scale=x_scale, y_scale=y_scale),
            shape=getattr(surface, "shape", "rectangle"),
            polygon=_scale_polygon2d(getattr(surface, "polygon", None), x_scale=x_scale, y_scale=y_scale),
            margin=float(getattr(surface, "margin", 0.0) or 0.0) * min(x_scale, y_scale),
        )
    )


def _scale_vec3(value: Any, scale: tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(value[0]) * scale[0], float(value[1]) * scale[1], float(value[2]) * scale[2])


def _vec_norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


def _scale_bounds2d(bounds: Bounds2D, *, x_scale: float, y_scale: float) -> Bounds2D:
    return Bounds2D(
        Range1D(bounds.x.min * x_scale, bounds.x.max * x_scale),
        Range1D(bounds.y.min * y_scale, bounds.y.max * y_scale),
    )


def _scale_polygon2d(polygon: Any | None, *, x_scale: float, y_scale: float) -> Any | None:
    if polygon is None:
        return None
    return type(polygon)(
        outer=tuple((x * x_scale, y * y_scale) for x, y in polygon.outer),
        holes=tuple(tuple((x * x_scale, y * y_scale) for x, y in ring) for ring in polygon.holes),
    )


def _volume_group_requires_case_pool(volume_group: Any | None, object_names: list[str]) -> bool:
    if volume_group is None or len(object_names) <= 1:
        return False
    return not bool(getattr(volume_group, "ignore_intra_group_collision", True))


def _intersect_volume_xy_bounds(bounds_items: Iterable[Any]) -> Bounds2D | None:
    x_min = y_min = float("-inf")
    x_max = y_max = float("inf")
    seen = False
    for bounds in bounds_items:
        seen = True
        x_min = max(x_min, float(bounds.x.min))
        x_max = min(x_max, float(bounds.x.max))
        y_min = max(y_min, float(bounds.y.min))
        y_max = min(y_max, float(bounds.y.max))
    if not seen or x_min > x_max + GEOM_EPS or y_min > y_max + GEOM_EPS:
        return None
    return Bounds2D(Range1D(x_min, x_max), Range1D(y_min, y_max))


def _volume_accept_cell_fn(
    volume: Any, fit_bounds_by_object: Mapping[str, Any]
) -> Callable[[str, Range1D, Range1D], bool] | None:
    if str(getattr(volume, "shape", "box") or "box").lower() != "cylinder":
        return None
    center = getattr(volume, "center", None)
    if center is None:
        return None
    cx, cy = float(center[0]), float(center[1])

    def accept(name: str, x: Range1D, y: Range1D) -> bool:
        fit = fit_bounds_by_object.get(str(name))
        if fit is None:
            return False
        radius = min(float(fit.x.length), float(fit.y.length)) * 0.5
        for xx in (float(x.min), float(x.max)):
            for yy in (float(y.min), float(y.max)):
                if (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) > (radius + GEOM_EPS) * (radius + GEOM_EPS):
                    return False
        return True

    return accept
