from __future__ import annotations

from typing import Any

from ..common.geometry import ManualSupportSurface
from ..common.support_surfaces import (
    aabb2_intersects,
    local_surface_child_z_range,
    ring_bounds,
    support_cell_accept_fn,
    surface_bounds_to_parent_xy_bounds,
    surface_child_z_range,
    surface_normal,
    surface_xy_axis_aligned,
    swept_footprint_intersects_hole,
)
from .compile_utils import bounds2d_json


def runtime_surface_json(surface: ManualSupportSurface) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": surface.surface_id,
        "center": [float(v) for v in surface.center],
        "x_axis": [float(v) for v in surface.x_axis],
        "y_axis": [float(v) for v in surface.y_axis],
        "normal": [float(v) for v in surface.normal],
        "bounds": bounds2d_json(surface.bounds),
        "shape": surface.shape,
        "margin": float(surface.margin),
    }
    if surface.polygon is not None:
        payload["outer"] = [[float(x), float(y)] for x, y in surface.polygon.outer]
        payload["holes"] = [[[float(x), float(y)] for x, y in ring] for ring in surface.polygon.holes]
    return payload


__all__ = [
    "aabb2_intersects",
    "local_surface_child_z_range",
    "ring_bounds",
    "runtime_surface_json",
    "support_cell_accept_fn",
    "surface_bounds_to_parent_xy_bounds",
    "surface_child_z_range",
    "surface_normal",
    "surface_xy_axis_aligned",
    "swept_footprint_intersects_hole",
]
