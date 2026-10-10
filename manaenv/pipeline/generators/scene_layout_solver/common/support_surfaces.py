from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..assets.bridge import LayoutAssetRecord
from .constants import AXIS_ALIGN_EPS, DEFAULT_SUPPORT_CONTACT_OFFSET_M, GEOM_EPS
from .geometry import (
    Bounds2D,
    ManualSupportSurface,
    QuatXYZW,
    Range1D,
    projected_footprint_bounds,
    support_contact_offset,
)


def surface_bounds_to_parent_xy_bounds(
    surface: Any, bounds: Bounds2D, *, eps: float = AXIS_ALIGN_EPS
) -> Bounds2D | None:
    x_axis = tuple(float(v) for v in surface.x_axis)
    y_axis = tuple(float(v) for v in surface.y_axis)
    normal = surface_normal(surface)
    if abs(x_axis[2]) > eps or abs(y_axis[2]) > eps:
        return None
    if normal[2] < 1.0 - eps:
        return None
    center = tuple(float(v) for v in surface.center)
    points = [
        (
            center[0] + x_axis[0] * x + y_axis[0] * y,
            center[1] + x_axis[1] * x + y_axis[1] * y,
        )
        for x in (bounds.x.min, bounds.x.max)
        for y in (bounds.y.min, bounds.y.max)
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return Bounds2D(Range1D(min(xs), max(xs)), Range1D(min(ys), max(ys)))


def local_surface_child_z_range(
    surface: Any,
    record: LayoutAssetRecord,
    *,
    quaternion: QuatXYZW = (0.0, 0.0, 0.0, 1.0),
) -> tuple[float, float]:
    normal = surface_normal(surface)
    pose_z_offset = support_contact_offset(
        record.local_bounds, quaternion, normal, contact_offset=DEFAULT_SUPPORT_CONTACT_OFFSET_M
    )
    z = float(surface.center[2]) + normal[2] * pose_z_offset
    return (z, z)


def surface_child_z_range(
    surface: Any,
    record: LayoutAssetRecord,
    *,
    quaternion: QuatXYZW = (0.0, 0.0, 0.0, 1.0),
) -> tuple[float, float]:
    normal = surface_normal(surface)
    pose_z_offset = support_contact_offset(
        record.local_bounds, quaternion, normal, contact_offset=DEFAULT_SUPPORT_CONTACT_OFFSET_M
    )
    if hasattr(surface, "local_to_world"):
        center = surface.local_to_world(0.0, 0.0, pose_z_offset)
        return (float(center[2]), float(center[2]))
    z = float(surface.center[2]) + normal[2] * pose_z_offset
    return (z, z)


def support_cell_accept_fn(
    surface: Any,
    records: Mapping[str, LayoutAssetRecord],
    quaternions: Mapping[str, QuatXYZW] | None = None,
    footprint_by_object: Mapping[str, tuple[float, float]] | None = None,
    footprint_bounds_by_object: Mapping[str, Bounds2D] | None = None,
):
    support_surface = (
        surface if hasattr(surface, "supports_local_footprint") else ManualSupportSurface.from_spec(surface)
    )
    object_quats = dict(quaternions or {})
    footprints = dict(footprint_by_object or {})
    footprint_bounds = dict(footprint_bounds_by_object or {})

    def accept(object_name: str, x: Range1D, y: Range1D) -> bool:
        record = records[object_name]
        q = object_quats.get(object_name, (0.0, 0.0, 0.0, 1.0))
        bounds = footprint_bounds.get(object_name)
        if bounds is None:
            bounds = projected_footprint_bounds(record.local_bounds, q, support_surface.x_axis, support_surface.y_axis)
        sx, sy = footprints.get(object_name) or (bounds.x.length, bounds.y.length)
        points = (
            (x.center, y.center),
            (x.min, y.min),
            (x.min, y.center),
            (x.min, y.max),
            (x.center, y.min),
            (x.center, y.max),
            (x.max, y.min),
            (x.max, y.center),
            (x.max, y.max),
        )
        if not all(support_surface.supports_local_footprint_bounds(px, py, bounds) for px, py in points):
            return False
        return not swept_footprint_intersects_hole(support_surface, x, y, half_x=sx * 0.5, half_y=sy * 0.5)

    return accept


def swept_footprint_intersects_hole(
    surface: ManualSupportSurface, x: Range1D, y: Range1D, *, half_x: float, half_y: float
) -> bool:
    polygon = getattr(surface, "polygon", None)
    if polygon is None:
        return False
    swept = (x.min - half_x, x.max + half_x, y.min - half_y, y.max + half_y)
    for hole in getattr(polygon, "holes", ()) or ():
        if aabb2_intersects(swept, ring_bounds(hole)):
            return True
    return False


def ring_bounds(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    pts = list(points)
    return (
        min(point[0] for point in pts),
        max(point[0] for point in pts),
        min(point[1] for point in pts),
        max(point[1] for point in pts),
    )


def aabb2_intersects(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float], *, eps: float = GEOM_EPS
) -> bool:
    return a[0] < b[1] - eps and b[0] < a[1] - eps and a[2] < b[3] - eps and b[2] < a[3] - eps


def surface_normal(surface: Any) -> tuple[float, float, float]:
    if getattr(surface, "normal", None) is not None:
        normal = tuple(float(v) for v in surface.normal)
        if len(normal) != 3:
            raise ValueError(f"support surface normal must contain three values, got {normal!r}")
        return normal[0], normal[1], normal[2]
    x_axis = tuple(float(v) for v in surface.x_axis)
    y_axis = tuple(float(v) for v in surface.y_axis)
    return (
        x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
        x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
        x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
    )


def surface_xy_axis_aligned(surface: ManualSupportSurface, *, eps: float = AXIS_ALIGN_EPS) -> bool:
    vectors = (surface.x_axis, surface.y_axis, surface.normal)
    if abs(vectors[0][2]) > eps or abs(vectors[1][2]) > eps:
        return False
    if abs(abs(vectors[2][2]) - 1.0) > eps:
        return False
    xy_axes = [(1.0, 0.0), (0.0, 1.0)]
    for vec in vectors[:2]:
        if max(abs(vec[0] * axis[0] + vec[1] * axis[1]) for axis in xy_axes) < 1.0 - eps:
            return False
    return True
