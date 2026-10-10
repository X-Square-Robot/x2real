from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..assets.bridge import LayoutAssetRecord
from .candidate_grid import adaptive_cell_step, axis_cells
from .constants import DEFAULT_RUNTIME_DOMAIN_MAX_CELLS, DEFAULT_SUPPORT_CONTACT_OFFSET_M
from .geometry import (
    Bounds2D,
    ManualSupportSurface,
    QuatXYZW,
    Range1D,
    projected_footprint_bounds,
    projected_footprint_bounds_with_yaw_range,
    support_contact_offset,
)
from .support_surfaces import aabb2_intersects, ring_bounds


@dataclass(frozen=True)
class SupportFitOption:
    bounds: Bounds2D
    footprint_bounds: Bounds2D
    yaw: float | None

    @property
    def footprint_size(self) -> tuple[float, float]:
        return (self.footprint_bounds.x.length, self.footprint_bounds.y.length)


@dataclass(frozen=True)
class SupportCenterDomain:
    bounds: Bounds2D
    surface: ManualSupportSurface
    base_bounds: Bounds2D
    fit_options: tuple[SupportFitOption, ...]
    meta: dict[str, Any]

    @property
    def footprint_bounds(self) -> Bounds2D:
        return union_bounds([item.footprint_bounds for item in self.fit_options])


def apply_surface_range(bounds: Bounds2D, surface_range: Any | None) -> Bounds2D | None:
    if surface_range is None:
        return bounds
    return bounds.intersect(surface_range.to_bounds(bounds))


def apply_surface_ranges(bounds: Bounds2D, surface_ranges: Iterable[Any | None]) -> Bounds2D | None:
    out: Bounds2D | None = bounds
    for surface_range in surface_ranges:
        if out is None:
            return None
        out = apply_surface_range(out, surface_range)
    return out


def support_footprint_bounds(
    record: LayoutAssetRecord,
    base_quaternion: QuatXYZW,
    surface: ManualSupportSurface,
    *,
    yaw_range: tuple[float, float] | None = None,
) -> Bounds2D:
    if yaw_range is not None:
        return projected_footprint_bounds_with_yaw_range(
            record.local_bounds,
            base_quaternion,
            yaw_range,
            surface.x_axis,
            surface.y_axis,
        )
    return projected_footprint_bounds(record.local_bounds, base_quaternion, surface.x_axis, surface.y_axis)


def support_center_domain(
    *,
    object_name: str,
    record: LayoutAssetRecord,
    surface: ManualSupportSurface,
    base_quaternion: QuatXYZW,
    surface_ranges: Iterable[Any | None] = (),
    yaw_range: tuple[float, float] | None = None,
    yaw_fit_mode: str = "sampled",
) -> tuple[SupportCenterDomain | None, dict[str, Any]]:
    base = surface.effective_bounds()
    if base is None:
        return None, {"reason": "surface_effective_bounds_empty", "surface": surface.surface_id}

    authored_center_bounds = apply_surface_ranges(base, surface_ranges)
    if authored_center_bounds is None:
        return None, {"reason": "surface_range_empty", "surface": surface.surface_id}

    fit_options = support_fit_options(
        record=record,
        surface=surface,
        base_quaternion=base_quaternion,
        base_bounds=base,
        center_bounds=authored_center_bounds,
        yaw_range=yaw_range,
        yaw_fit_mode=yaw_fit_mode,
    )
    if not fit_options:
        footprint = support_footprint_bounds(record, base_quaternion, surface, yaw_range=yaw_range)
        return None, {
            "reason": "object_larger_than_surface",
            "surface": surface.surface_id,
            "footprint": [footprint.x.length, footprint.y.length],
            "footprint_bounds": bounds2d_payload(footprint),
            "surface_size": [base.x.length, base.y.length],
        }

    bounds = union_bounds([item.bounds for item in fit_options])
    footprints = [item.footprint_size for item in fit_options]
    footprint_bounds = union_bounds([item.footprint_bounds for item in fit_options])
    meta = {
        "provider": "support_on",
        "surface": surface.surface_id,
        "footprint": [max(item[0] for item in footprints), max(item[1] for item in footprints)],
        "footprint_bounds": bounds2d_payload(footprint_bounds),
        "surface_size": [base.x.length, base.y.length],
    }
    return (
        SupportCenterDomain(
            bounds=bounds,
            surface=surface,
            base_bounds=base,
            fit_options=tuple(fit_options),
            meta=meta,
        ),
        meta,
    )


def support_fit_options(
    *,
    record: LayoutAssetRecord,
    surface: ManualSupportSurface,
    base_quaternion: QuatXYZW,
    base_bounds: Bounds2D,
    center_bounds: Bounds2D,
    yaw_range: tuple[float, float] | None,
    yaw_fit_mode: str = "sampled",
) -> list[SupportFitOption]:
    out: list[SupportFitOption] = []
    yaws: tuple[float | None, ...]
    if str(yaw_fit_mode).strip().lower() == "swept":
        yaws = (None,)
    else:
        yaws = yaw_fit_values(yaw_range)
    for yaw in yaws:
        footprint = support_footprint_bounds(
            record,
            base_quaternion,
            surface,
            yaw_range=yaw_range if str(yaw_fit_mode).strip().lower() == "swept" else fixed_yaw_range(yaw),
        )
        fit_bounds = Bounds2D(
            Range1D(base_bounds.x.min - footprint.x.min, base_bounds.x.max - footprint.x.max),
            Range1D(base_bounds.y.min - footprint.y.min, base_bounds.y.max - footprint.y.max),
        )
        if fit_bounds.x.min > fit_bounds.x.max or fit_bounds.y.min > fit_bounds.y.max:
            continue
        bounds = fit_bounds.intersect(center_bounds)
        if bounds is None:
            continue
        out.append(SupportFitOption(bounds=bounds, footprint_bounds=footprint, yaw=yaw))
    return out


def support_local_fits(
    *,
    record: LayoutAssetRecord,
    surface: ManualSupportSurface,
    base_quaternion: QuatXYZW,
    x: float,
    y: float,
    yaw_range: tuple[float, float] | None = None,
) -> bool:
    for yaw in yaw_fit_values(yaw_range):
        footprint = support_footprint_bounds(
            record,
            base_quaternion,
            surface,
            yaw_range=fixed_yaw_range(yaw),
        )
        if surface.supports_local_footprint_bounds(float(x), float(y), footprint):
            return True
    return False


def support_child_z_range(
    surface: ManualSupportSurface,
    record: LayoutAssetRecord,
    *,
    quaternion: QuatXYZW = (0.0, 0.0, 0.0, 1.0),
) -> tuple[float, float]:
    pose_z_offset = support_contact_offset(
        record.local_bounds,
        quaternion,
        surface.normal,
        contact_offset=DEFAULT_SUPPORT_CONTACT_OFFSET_M,
    )
    center = surface.local_to_world(0.0, 0.0, pose_z_offset)
    return (center[2], center[2])


def should_export_support_cells(surface: ManualSupportSurface) -> bool:
    return surface.polygon is not None or str(surface.shape).lower() not in {"", "rectangle"}


def support_domain_cells(
    bounds: Bounds2D,
    *,
    surface: ManualSupportSurface,
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
            if not support_cell_supported(surface=surface, x=xr, y=yr, footprint_bounds=footprint_bounds):
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


def support_cell_supported(
    *,
    surface: ManualSupportSurface,
    x: Range1D,
    y: Range1D,
    footprint_bounds: Bounds2D,
) -> bool:
    points = (
        (x.center, y.center),
        (x.min, y.min),
        (x.min, y.max),
        (x.max, y.min),
        (x.max, y.max),
    )
    if not all(surface.supports_local_footprint_bounds(px, py, footprint_bounds) for px, py in points):
        return False
    polygon = getattr(surface, "polygon", None)
    if polygon is None:
        return True
    swept = swept_bounds(Bounds2D(x, y), footprint_bounds)
    swept_aabb = (swept.x.min, swept.x.max, swept.y.min, swept.y.max)
    return not any(aabb2_intersects(swept_aabb, ring_bounds(hole)) for hole in getattr(polygon, "holes", ()) or ())


def swept_bounds(center_bounds: Bounds2D, footprint_bounds: Bounds2D) -> Bounds2D:
    return Bounds2D(
        Range1D(center_bounds.x.min + footprint_bounds.x.min, center_bounds.x.max + footprint_bounds.x.max),
        Range1D(center_bounds.y.min + footprint_bounds.y.min, center_bounds.y.max + footprint_bounds.y.max),
    )


def yaw_fit_values(yaw_range: tuple[float, float] | None) -> tuple[float | None, ...]:
    if yaw_range is None:
        return (None,)
    lo, hi = float(yaw_range[0]), float(yaw_range[1])
    if lo > hi:
        lo, hi = hi, lo
    if abs(hi - lo) <= 1e-12:
        return (lo,)
    values = (lo, hi, (lo + hi) * 0.5, lo + (hi - lo) * 0.25, lo + (hi - lo) * 0.75)
    return unique_yaws(values)


def yaw_sample_values(yaw_range: tuple[float, float] | None, rng: Any) -> tuple[float | None, ...]:
    values = list(yaw_fit_values(yaw_range))
    if yaw_range is not None:
        lo, hi = float(yaw_range[0]), float(yaw_range[1])
        if lo > hi:
            lo, hi = hi, lo
        if hi - lo > 1e-12:
            values.extend(rng.uniform(lo, hi) for _ in range(2))
    return unique_yaws(values)


def unique_yaws(values: Iterable[float | None]) -> tuple[float | None, ...]:
    out: list[float | None] = []
    seen: set[float | None] = set()
    for value in values:
        key = None if value is None else round(float(value), 12)
        if key in seen:
            continue
        seen.add(key)
        out.append(None if value is None else float(value))
    return tuple(out)


def fixed_yaw_range(yaw: float | None) -> tuple[float, float] | None:
    if yaw is None:
        return None
    value = float(yaw)
    return (value, value)


def union_bounds(bounds: Sequence[Bounds2D]) -> Bounds2D:
    return Bounds2D(
        Range1D(min(item.x.min for item in bounds), max(item.x.max for item in bounds)),
        Range1D(min(item.y.min for item in bounds), max(item.y.max for item in bounds)),
    )


def bounds2d_payload(bounds: Bounds2D) -> dict[str, list[float]]:
    return {"x": [bounds.x.min, bounds.x.max], "y": [bounds.y.min, bounds.y.max]}
