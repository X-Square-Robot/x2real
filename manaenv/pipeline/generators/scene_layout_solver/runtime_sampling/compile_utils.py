from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..common.geometry import Bounds2D, Range1D
from .schema import PlacementDomain


def sample_xy_bounds(sample: Mapping[str, Any]) -> Bounds2D | None:
    xr = range_from_pair(sample.get("x"), default=None)
    yr = range_from_pair(sample.get("y"), default=None)
    if xr is None or yr is None:
        return None
    return Bounds2D(Range1D(xr[0], xr[1]), Range1D(yr[0], yr[1]))


def relative_sample_bounds(current: Bounds2D, sample: Mapping[str, Any]) -> Bounds2D:
    dx = range_from_pair(sample.get("dx", sample.get("x")), default=(0.0, 0.0)) or (0.0, 0.0)
    dy = range_from_pair(sample.get("dy", sample.get("y")), default=(0.0, 0.0)) or (0.0, 0.0)
    cx = current.x.center
    cy = current.y.center
    return Bounds2D(
        Range1D(cx + dx[0], cx + dx[1]),
        Range1D(cy + dy[0], cy + dy[1]),
    )


def local_jitter_bounds(current: Bounds2D, sample: Mapping[str, Any]) -> Bounds2D:
    radius = max(0.0, float(sample.get("radius", 0.0) or 0.0))
    x_radius = max(0.0, float(sample.get("x_radius", radius) or 0.0))
    y_radius = max(0.0, float(sample.get("y_radius", radius) or 0.0))
    cx = current.x.center
    cy = current.y.center
    return Bounds2D(
        Range1D(cx - x_radius, cx + x_radius),
        Range1D(cy - y_radius, cy + y_radius),
    )


def range_from_pair(value: Any, *, default: tuple[float, float] | None) -> tuple[float, float] | None:
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        return (v, v)
    if not isinstance(value, (list, tuple)):
        return default
    if len(value) == 1:
        lo = hi = float(value[0])
    elif len(value) == 2:
        lo = float(value[0])
        hi = float(value[1])
    else:
        return default
    return (min(lo, hi), max(lo, hi))


def event_constraints(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("min_separation", "max_sample_tries"):
        if key in params:
            out[key] = params[key]
    return out


def bounds2d_json(bounds: Bounds2D) -> dict[str, list[float]]:
    return {"x": [bounds.x.min, bounds.x.max], "y": [bounds.y.min, bounds.y.max]}


def surface_range_json(surface_range: Any) -> dict[str, list[float] | None] | None:
    if surface_range is None:
        return None
    return {
        "x": [surface_range.x.min, surface_range.x.max] if surface_range.x is not None else None,
        "y": [surface_range.y.min, surface_range.y.max] if surface_range.y is not None else None,
    }


def common_bounds(domains: list[PlacementDomain]) -> Bounds2D | None:
    current: Bounds2D | None = None
    for domain in domains:
        if domain.x is None or domain.y is None:
            return None
        bounds = Bounds2D(Range1D(domain.x[0], domain.x[1]), Range1D(domain.y[0], domain.y[1]))
        current = bounds if current is None else current.intersect(bounds)
        if current is None:
            return None
    return current


def dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
