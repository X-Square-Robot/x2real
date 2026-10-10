from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .compile_utils import bounds2d_json
from .meta import CELL_SHAPE_RECTANGLE, META_ALLOCATED_BOUNDS, META_FOOTPRINT, META_FOOTPRINT_BOUNDS
from .schema import PlacementDomain


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "object"


def allocated_bounds_for_cell(cell: Any, profile: Any, *, boundary_gap: float = 0.0) -> dict[str, list[float]]:
    gap = max(0.0, float(boundary_gap))
    return {
        "x": [
            cell.x.min + profile.footprint_bounds.x.min - gap,
            cell.x.max + profile.footprint_bounds.x.max + gap,
        ],
        "y": [
            cell.y.min + profile.footprint_bounds.y.min - gap,
            cell.y.max + profile.footprint_bounds.y.max + gap,
        ],
    }


def cell_payload(
    *,
    cell_id: str,
    cell: Any,
    profile: Any,
    boundary_gap: float = 0.0,
) -> dict[str, Any]:
    return {
        "id": cell_id,
        "shape": CELL_SHAPE_RECTANGLE,
        "x": [cell.x.min, cell.x.max],
        "y": [cell.y.min, cell.y.max],
        "z": [cell.z[0], cell.z[1]],
        "center_domain": {"x": [cell.x.min, cell.x.max], "y": [cell.y.min, cell.y.max]},
        META_ALLOCATED_BOUNDS: allocated_bounds_for_cell(cell, profile, boundary_gap=boundary_gap),
        META_FOOTPRINT: [profile.footprint[0], profile.footprint[1]],
        META_FOOTPRINT_BOUNDS: bounds2d_json(profile.footprint_bounds),
    }


def placement_domain_from_cell(
    *,
    domain_id: str,
    object_name: str,
    frame: str,
    surface: str | None,
    cell: Any,
    profile: Any,
    yaw: tuple[float, float] | None,
    yaw_choices: tuple[dict[str, Any], ...] = (),
    base_quaternion: tuple[float, float, float, float] | None = None,
    meta: Mapping[str, Any] | None = None,
    boundary_gap: float = 0.0,
) -> PlacementDomain:
    payload = cell_payload(cell_id=f"{domain_id}:cell_000", cell=cell, profile=profile, boundary_gap=boundary_gap)
    return PlacementDomain(
        domain_id=domain_id,
        object_name=object_name,
        frame=frame,
        geometry_type="cells",
        surface=surface,
        x=(cell.x.min, cell.x.max),
        y=(cell.y.min, cell.y.max),
        z=cell.z,
        yaw=yaw,
        yaw_choices=tuple(dict(choice) for choice in yaw_choices),
        base_quaternion=base_quaternion,
        cells=(payload,),
        meta={
            **dict(meta or {}),
            META_ALLOCATED_BOUNDS: dict(payload[META_ALLOCATED_BOUNDS]),
        },
    )
