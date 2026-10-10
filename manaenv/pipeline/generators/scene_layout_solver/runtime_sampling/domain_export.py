from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..assets.bridge import LayoutAssetRecord
from ..common.candidate_grid import adaptive_cell_step, axis_cells
from ..common.constants import DEFAULT_RUNTIME_DOMAIN_MAX_CELLS, DEFAULT_SUPPORT_CONTACT_OFFSET_M
from ..common.diagnostics import LayoutDiagnostic
from ..common.geometry import (
    Bounds2D,
    ManualSupportSurface,
    QuatXYZW,
    Range1D,
    projected_footprint_bounds_with_yaw_range,
    support_contact_offset,
)
from ..static_solve.domains.providers import SupportOnProvider, object_base_quaternion, object_quaternion
from ..static_solve.ir import CompiledLayoutIR
from ..static_solve.problem import LayoutProblem
from .compile_utils import bounds2d_json
from .footprint_domains import swept_bounds
from .meta import (
    META_PLAN_KIND,
    META_SEMANTIC_POOL,
    META_SEMANTIC_POOL_KIND,
    META_SOURCE_SCENE_FIELD,
    META_WEIGHT_SCOPE,
    META_WEIGHT_TOTAL,
    META_ZERO_WEIGHT_EXCLUDED,
    POOL_SURFACE_RANGE_CHOICE,
)
from .schema import PlacementDomain
from .support_domains import aabb2_intersects, ring_bounds
from .yaw_choices import object_yaw_choice_payloads, yaw_choice_semantic_meta


@dataclass(frozen=True)
class RuntimeDomainExportResult:
    domain: PlacementDomain | None = None
    diagnostic: LayoutDiagnostic | None = None
    meta: dict[str, Any] | None = None


class RuntimeDomainExporter:
    """Export solve-side domains into cheap runtime sampling domains."""

    def __init__(
        self,
        problem: LayoutProblem,
        records: Mapping[str, LayoutAssetRecord],
        ir: CompiledLayoutIR | None = None,
    ) -> None:
        self.problem = problem
        self.records = dict(records)
        self.ir = ir

    def export_scene_surface_domain(
        self,
        object_name: str,
        *,
        domain_id: str,
        source_event: str,
        yaw: tuple[float, float] | None = None,
        sample_bounds: Bounds2D | None = None,
        sample_type: str = "inherit",
        authoring_frame: str = "scene-surface",
        use_scene_object_range: bool = True,
        meta: Mapping[str, Any] | None = None,
    ) -> RuntimeDomainExportResult:
        if object_name not in self.problem.objects:
            return _error(
                "layout_runtime_sampling_object_unknown",
                "sampling domain references object outside layout spec",
                target=domain_id,
                object=object_name,
            )
        obj = self.problem.objects[object_name]
        yaw_choices = object_yaw_choice_payloads(self.problem, object_name)
        provider = SupportOnProvider(self.problem)
        weighted_scene_ranges = bool(
            use_scene_object_range and obj.override_surface_range is None and obj.surface_ranges
        )
        fit_bounds, provider_meta = self.support_center_bounds(
            object_name,
            provider=provider,
            yaw=yaw,
            use_scene_object_range=use_scene_object_range and not weighted_scene_ranges,
        )
        if fit_bounds is None:
            code = (
                "layout_runtime_sampling_inherit_domain_empty"
                if use_scene_object_range
                else "layout_runtime_sampling_support_domain_empty"
            )
            return _error(
                code,
                "node has no solver-certified support domain",
                target=domain_id,
                object=object_name,
                **provider_meta,
            )
        bounds = fit_bounds if sample_bounds is None else fit_bounds.intersect(sample_bounds)
        if bounds is None:
            return _error(
                "layout_runtime_sampling_domain_excludes_support",
                "sample domain has empty intersection with support-domain fit bounds",
                target=domain_id,
                object=object_name,
                support_bounds=bounds2d_json(fit_bounds),
                sample_bounds=bounds2d_json(sample_bounds) if sample_bounds is not None else None,
            )
        surface = self.problem.resolve_surface(str(obj.support), {})
        if surface is None:
            return _error(
                "layout_runtime_sampling_support_missing",
                "node has no resolved scene support surface",
                target=domain_id,
                object=object_name,
                surface=obj.support,
            )
        z_quat = (
            object_base_quaternion(self.problem, object_name)
            if yaw is not None
            else object_quaternion(self.problem, object_name)
        )
        z_range = _surface_domain_z_range(surface, self.records[object_name], quaternion=z_quat)
        out_meta = {
            **provider_meta,
            **dict(meta or {}),
            "source_event": source_event,
            "authoring_frame": authoring_frame,
            "sample_type": sample_type,
            "sample_domain": bounds2d_json(bounds),
        }
        if sample_type == "inherit":
            out_meta.setdefault("inherited_domain", bounds2d_json(bounds))
        if weighted_scene_ranges:
            return self._surface_range_choice_domain(
                object_name,
                domain_id=domain_id,
                source_event=source_event,
                surface=surface,
                provider=provider,
                base_bounds=bounds,
                provider_meta=provider_meta,
                out_meta=out_meta,
                z_range=z_range,
                yaw=yaw,
            )
        if _should_export_cells(surface):
            cells = _cells_for_support_domain(
                bounds,
                object_name=object_name,
                provider=provider,
                surface_id=str(obj.support),
                z_range=z_range,
                footprint_bounds=_meta_footprint_bounds(provider_meta),
            )
            if not cells:
                return _error(
                    "layout_runtime_sampling_cells_empty",
                    "support domain could not be converted to valid runtime cells",
                    target=domain_id,
                    object=object_name,
                    support_bounds=bounds2d_json(bounds),
                )
            return RuntimeDomainExportResult(
                domain=PlacementDomain(
                    domain_id=domain_id,
                    object_name=object_name,
                    frame=f"surface:{surface.surface_id}",
                    geometry_type="cells",
                    surface=surface.surface_id,
                    z=z_range,
                    yaw=yaw,
                    yaw_choices=yaw_choices,
                    base_quaternion=object_base_quaternion(self.problem, object_name),
                    cells=tuple(cells),
                    meta={
                        **out_meta,
                        **yaw_choice_semantic_meta(object_name, yaw_choices),
                        "cell_count": len(cells),
                        "domain_encoding": "cells",
                    },
                ),
                meta=out_meta,
            )
        return RuntimeDomainExportResult(
            domain=PlacementDomain(
                domain_id=domain_id,
                object_name=object_name,
                frame=f"surface:{surface.surface_id}",
                geometry_type="rectangle",
                surface=surface.surface_id,
                x=(bounds.x.min, bounds.x.max),
                y=(bounds.y.min, bounds.y.max),
                z=z_range,
                yaw=yaw,
                yaw_choices=yaw_choices,
                base_quaternion=object_base_quaternion(self.problem, object_name),
                meta={
                    **out_meta,
                    **yaw_choice_semantic_meta(object_name, yaw_choices),
                    "domain_encoding": "rectangle",
                },
            ),
            meta=out_meta,
        )

    def _surface_range_choice_domain(
        self,
        object_name: str,
        *,
        domain_id: str,
        source_event: str,
        surface: ManualSupportSurface,
        provider: SupportOnProvider,
        base_bounds: Bounds2D,
        provider_meta: Mapping[str, Any],
        out_meta: Mapping[str, Any],
        z_range: tuple[float, float],
        yaw: tuple[float, float] | None,
    ) -> RuntimeDomainExportResult:
        obj = self.problem.objects[object_name]
        yaw_choices = object_yaw_choice_payloads(self.problem, object_name)
        choices: list[dict[str, Any]] = []
        for choice in obj.surface_ranges:
            if choice.weight <= 0.0:
                continue
            bounds = provider.apply_surface_range(base_bounds, choice.surface_range)
            if bounds is None:
                continue
            payload: dict[str, Any] = {
                "id": choice.range_id,
                "weight": float(choice.weight),
                "x": [float(bounds.x.min), float(bounds.x.max)],
                "y": [float(bounds.y.min), float(bounds.y.max)],
                "z": [float(z_range[0]), float(z_range[1])],
                "geometry_type": "rectangle",
            }
            if _should_export_cells(surface):
                cells = _cells_for_support_domain(
                    bounds,
                    object_name=object_name,
                    provider=provider,
                    surface_id=str(obj.support),
                    z_range=z_range,
                    footprint_bounds=_meta_footprint_bounds(provider_meta),
                )
                if not cells:
                    continue
                payload.update({"geometry_type": "cells", "cells": cells})
            choices.append(payload)
        if not choices:
            return _error(
                "layout_runtime_sampling_weighted_surface_ranges_empty",
                "weighted surface_ranges produce no valid runtime choices",
                target=domain_id,
                object=object_name,
                support_bounds=bounds2d_json(base_bounds),
            )
        return RuntimeDomainExportResult(
            domain=PlacementDomain(
                domain_id=domain_id,
                object_name=object_name,
                frame=f"surface:{surface.surface_id}",
                geometry_type="surface_range_choice",
                surface=surface.surface_id,
                z=z_range,
                yaw=yaw,
                yaw_choices=yaw_choices,
                base_quaternion=object_base_quaternion(self.problem, object_name),
                range_choices=tuple(choices),
                meta={
                    **dict(out_meta),
                    "source_event": source_event,
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
            meta=dict(out_meta),
        )

    def support_center_bounds(
        self,
        object_name: str,
        *,
        provider: SupportOnProvider,
        yaw: tuple[float, float] | None = None,
        use_scene_object_range: bool,
    ) -> tuple[Bounds2D | None, dict[str, Any]]:
        obj = self.problem.objects[object_name]
        if not obj.support:
            return None, {"reason": "object_support_missing"}
        surface = self.problem.resolve_surface(obj.support, {})
        if surface is None:
            reason = (
                "local_surface_parent_unplaced"
                if self.problem.local_surface_parent(obj.support)
                else "surface_not_found"
            )
            return None, {"reason": reason, "surface": obj.support}
        base = surface.effective_bounds()
        if base is None:
            return None, {"reason": "surface_effective_bounds_empty", "surface": obj.support}
        base_quat = (
            object_base_quaternion(self.problem, object_name)
            if yaw is not None
            else object_quaternion(self.problem, object_name)
        )
        footprint = runtime_footprint_bounds(self.records[object_name], base_quat, surface, yaw)
        sx, sy = footprint.x.length, footprint.y.length
        fit_bounds = Bounds2D(
            Range1D(base.x.min - footprint.x.min, base.x.max - footprint.x.max),
            Range1D(base.y.min - footprint.y.min, base.y.max - footprint.y.max),
        )
        if fit_bounds.x.min > fit_bounds.x.max or fit_bounds.y.min > fit_bounds.y.max:
            return None, {
                "reason": "object_larger_than_surface",
                "surface": obj.support,
                "footprint": [sx, sy],
                "surface_size": [base.x.length, base.y.length],
            }
        bounds = fit_bounds
        if use_scene_object_range:
            bounds = provider.apply_surface_range(bounds, provider.object_surface_range(obj))
            if bounds is None:
                return None, {
                    "reason": "surface_range_excludes_fit_bounds",
                    "surface": obj.support,
                    "footprint": [sx, sy],
                }
        return bounds, {
            "provider": "support_on",
            "surface": obj.support,
            "footprint": [sx, sy],
            "footprint_bounds": {"x": [footprint.x.min, footprint.x.max], "y": [footprint.y.min, footprint.y.max]},
            "surface_size": [base.x.length, base.y.length],
            "ignored_scene_object_surface_range": not use_scene_object_range,
            "yaw_aware_footprint": yaw is not None,
        }


def _should_export_cells(surface: ManualSupportSurface) -> bool:
    return surface.polygon is not None or str(surface.shape).lower() not in {"", "rectangle"}


def _cells_for_support_domain(
    bounds: Bounds2D,
    *,
    object_name: str,
    provider: SupportOnProvider,
    surface_id: str,
    z_range: tuple[float, float],
    footprint_bounds: Bounds2D,
    max_cells: int = DEFAULT_RUNTIME_DOMAIN_MAX_CELLS,
) -> list[dict[str, Any]]:
    step = adaptive_cell_step(bounds, (footprint_bounds.x.length, footprint_bounds.y.length))
    xs = axis_cells(bounds.x, step)
    ys = axis_cells(bounds.y, step)
    cells: list[dict[str, Any]] = []
    for ix, xr in enumerate(xs):
        for iy, yr in enumerate(ys):
            if not _cell_supported(provider, surface_id, xr, yr, footprint_bounds):
                continue
            cells.append(
                {
                    "id": f"cell_{ix:03d}_{iy:03d}",
                    "x": [xr.min, xr.max],
                    "y": [yr.min, yr.max],
                    "z": [z_range[0], z_range[1]],
                }
            )
            if len(cells) >= max_cells:
                return cells
    return cells


def _cell_supported(
    provider: SupportOnProvider,
    surface_id: str,
    xr: Range1D,
    yr: Range1D,
    footprint_bounds: Bounds2D,
) -> bool:
    surface = provider.surface(surface_id)
    if surface is None:
        return False
    points = (
        (xr.center, yr.center),
        (xr.min, yr.min),
        (xr.min, yr.max),
        (xr.max, yr.min),
        (xr.max, yr.max),
    )
    if not all(surface.supports_local_footprint_bounds(x, y, footprint_bounds) for x, y in points):
        return False
    polygon = getattr(surface, "polygon", None)
    if polygon is None:
        return True
    swept = swept_bounds(Bounds2D(xr, yr), footprint_bounds)
    swept_aabb = (swept.x.min, swept.x.max, swept.y.min, swept.y.max)
    return not any(aabb2_intersects(swept_aabb, ring_bounds(hole)) for hole in getattr(polygon, "holes", ()) or ())


def runtime_footprint_bounds(
    record: LayoutAssetRecord,
    quaternion: QuatXYZW,
    surface: ManualSupportSurface,
    yaw: tuple[float, float] | None,
) -> Bounds2D:
    return projected_footprint_bounds_with_yaw_range(
        record.local_bounds, quaternion, yaw, surface.x_axis, surface.y_axis
    )


def _meta_footprint_bounds(meta: Mapping[str, Any]) -> Bounds2D:
    raw = meta.get("footprint_bounds") if isinstance(meta, Mapping) else None
    if not isinstance(raw, Mapping):
        raise ValueError("runtime_domain_missing_footprint_bounds")
    return Bounds2D(Range1D.from_pair(raw["x"]), Range1D.from_pair(raw["y"]))


def _surface_domain_z_range(
    surface: ManualSupportSurface,
    record: LayoutAssetRecord,
    *,
    quaternion: QuatXYZW = (0.0, 0.0, 0.0, 1.0),
) -> tuple[float, float]:
    pose_z_offset = support_contact_offset(
        record.local_bounds, quaternion, surface.normal, contact_offset=DEFAULT_SUPPORT_CONTACT_OFFSET_M
    )
    center = surface.local_to_world(0.0, 0.0, pose_z_offset)
    return (center[2], center[2])


def _error(code: str, message: str, *, target: str = "", **meta: Any) -> RuntimeDomainExportResult:
    return RuntimeDomainExportResult(
        diagnostic=LayoutDiagnostic(code=code, message=message, target=target, meta=dict(meta)),
        meta=dict(meta),
    )
