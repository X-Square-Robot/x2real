from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..common.constants import GEOM_EPS
from ..common.geometry import Bounds2D, Range1D


def swept_bounds(center_bounds: Bounds2D, footprint_bounds: Bounds2D, *, margin: float = 0.0) -> Bounds2D:
    """Return the occupied 2D range induced by sampling object centers in a domain."""
    pad = max(0.0, float(margin))
    return Bounds2D(
        Range1D(center_bounds.x.min + footprint_bounds.x.min - pad, center_bounds.x.max + footprint_bounds.x.max + pad),
        Range1D(center_bounds.y.min + footprint_bounds.y.min - pad, center_bounds.y.max + footprint_bounds.y.max + pad),
    )


def bounds_overlap(left: Bounds2D, right: Bounds2D) -> bool:
    return _ranges_overlap(left.x, right.x) and _ranges_overlap(left.y, right.y)


def bounds_from_payload(payload: Any) -> Bounds2D | None:
    if not isinstance(payload, Mapping):
        return None
    x = float_pair(payload.get("x"))
    y = float_pair(payload.get("y"))
    if x is None or y is None:
        return None
    return Bounds2D(Range1D(float(x[0]), float(x[1])), Range1D(float(y[0]), float(y[1])))


def float_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        return (v, v)
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return (float(value[0]), float(value[1]))


def _ranges_overlap(left: Range1D, right: Range1D) -> bool:
    return left.min <= right.max + GEOM_EPS and right.min <= left.max + GEOM_EPS
