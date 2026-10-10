from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .geometry import Bounds3D, Pose, QuatXYZW, Range1D, Vec3, quat_mul, quat_rotate, vec_add


def volume_parent(volume: Any) -> str | None:
    parent = getattr(volume, "parent", None)
    return str(parent) if parent else None


def is_object_local_volume(volume: Any) -> bool:
    return bool(volume_parent(volume))


def box_fit_bounds(bounds: Bounds3D, child: Bounds3D, margin: float, *, mode: str) -> Bounds3D | None:
    m = max(0.0, float(margin))
    if mode == "center":
        fit = Bounds3D(
            Range1D(bounds.x.min + m, bounds.x.max - m),
            Range1D(bounds.y.min + m, bounds.y.max - m),
            Range1D(bounds.z.min + m, bounds.z.max - m),
        )
    else:
        fit = Bounds3D(
            Range1D(bounds.x.min + m - child.x.min, bounds.x.max - m - child.x.max),
            Range1D(bounds.y.min + m - child.y.min, bounds.y.max - m - child.y.max),
            Range1D(bounds.z.min + m - child.z.min, bounds.z.max - m - child.z.max),
        )
    return fit if _valid_bounds(fit) else None


def volume_fit_bounds(volume: Any, child: Bounds3D, *, mode: str = "fit") -> Bounds3D | None:
    shape = str(getattr(volume, "shape", "box") or "box").lower()
    bounds = getattr(volume, "bounds")
    margin = float(getattr(volume, "margin", 0.0) or 0.0)
    if shape == "box":
        return box_fit_bounds(bounds, child, margin, mode=mode)
    if shape == "cylinder":
        return cylinder_fit_bounds(volume, child, mode=mode)
    return None


def cylinder_fit_bounds(volume: Any, child: Bounds3D, *, mode: str) -> Bounds3D | None:
    bounds = getattr(volume, "bounds")
    center = getattr(volume, "center", (bounds.x.center, bounds.y.center, bounds.z.center))
    radius = float(getattr(volume, "radius", 0.0) or 0.0)
    margin = max(0.0, float(getattr(volume, "margin", 0.0) or 0.0))
    if mode == "center":
        fit_radius = radius - margin
        z = Range1D(bounds.z.min + margin, bounds.z.max - margin)
    else:
        child_radius = max(abs(child.x.min), abs(child.x.max), abs(child.y.min), abs(child.y.max))
        fit_radius = radius - margin - child_radius
        z = Range1D(bounds.z.min + margin - child.z.min, bounds.z.max - margin - child.z.max)
    if fit_radius < 0.0 or z.min > z.max:
        return None
    return Bounds3D(
        Range1D(float(center[0]) - fit_radius, float(center[0]) + fit_radius),
        Range1D(float(center[1]) - fit_radius, float(center[1]) + fit_radius),
        z,
    )


def volume_candidate_offsets(
    volume: Any,
    child: Bounds3D,
    *,
    mode: str = "fit",
    max_candidates: int = 64,
) -> list[Vec3]:
    fit = volume_fit_bounds(volume, child, mode=mode)
    if fit is None:
        return []
    shape = str(getattr(volume, "shape", "box") or "box").lower()
    if shape == "cylinder":
        return _cylinder_offsets(fit, max_candidates=max_candidates)
    return _box_offsets(fit, max_candidates=max_candidates)


def transform_volume_pose(volume: Any, offset: Vec3, quaternion: QuatXYZW, parent_pose: Pose | None) -> Pose:
    if volume_parent(volume):
        if parent_pose is None:
            raise ValueError("object_local_volume_requires_parent_pose")
        return Pose(
            position=vec_add(parent_pose.position, quat_rotate(parent_pose.quaternion, offset)),
            quaternion=quat_mul(parent_pose.quaternion, quaternion),
        )
    return Pose(position=offset, quaternion=quaternion)


def runtime_volume_domain_geometry(
    volume: Any, child: Bounds3D, *, mode: str = "fit"
) -> tuple[str, Bounds3D, dict[str, Any]] | None:
    fit = volume_fit_bounds(volume, child, mode=mode)
    if fit is None:
        return None
    shape = str(getattr(volume, "shape", "box") or "box").lower()
    meta: dict[str, Any] = {
        "volume": str(getattr(volume, "volume_id", "")),
        "volume_shape": shape,
    }
    if shape == "cylinder":
        radius = min(fit.x.length, fit.y.length) * 0.5
        meta["radius"] = radius
        return ("cylinder", fit, meta)
    return ("box", fit, meta)


def _box_offsets(bounds: Bounds3D, *, max_candidates: int) -> list[Vec3]:
    budget = max(1, int(max_candidates))
    per_axis = max(1, int(round(budget ** (1.0 / 3.0))))
    xs = _axis_values(bounds.x, per_axis)
    ys = _axis_values(bounds.y, per_axis)
    zs = _axis_values(bounds.z, per_axis)
    out = [(x, y, z) for z in zs for y in ys for x in xs]
    out.sort(key=lambda p: abs(p[0] - bounds.x.center) + abs(p[1] - bounds.y.center) + abs(p[2] - bounds.z.center))
    return out[:budget]


def _cylinder_offsets(bounds: Bounds3D, *, max_candidates: int) -> list[Vec3]:
    budget = max(1, int(max_candidates))
    radius = min(bounds.x.length, bounds.y.length) * 0.5
    cx = bounds.x.center
    cy = bounds.y.center
    if radius <= 1e-12:
        xy = [(cx, cy)]
    else:
        rings = 3 if budget >= 48 else 2
        spokes = 12 if budget >= 48 else 8
        xy = [(cx, cy)]
        for ring in range(1, rings + 1):
            r = radius * float(ring) / float(rings)
            for idx in range(spokes):
                angle = 2.0 * math.pi * float(idx) / float(spokes)
                xy.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    z_count = max(1, min(6, budget // max(1, len(xy)) + 1))
    zs = _axis_values(bounds.z, z_count)
    out = [(x, y, z) for z in zs for x, y in xy]
    out.sort(key=lambda p: abs(p[0] - cx) + abs(p[1] - cy) + abs(p[2] - bounds.z.center))
    return out[:budget]


def _axis_values(axis: Range1D, count: int) -> list[float]:
    n = max(1, int(count))
    if n == 1 or axis.length <= 1e-12:
        return [axis.center]
    step = axis.length / float(n - 1)
    return [axis.min + step * idx for idx in range(n)]


def _valid_bounds(bounds: Bounds3D) -> bool:
    return bounds.x.min <= bounds.x.max and bounds.y.min <= bounds.y.max and bounds.z.min <= bounds.z.max


def bounds_from_center_size(center: Sequence[float], size: Sequence[float]) -> Bounds3D:
    if len(center) != 3 or len(size) != 3:
        raise ValueError("center_size_must_be_vec3")
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    sx, sy, sz = (float(size[0]), float(size[1]), float(size[2]))
    return Bounds3D(
        Range1D(cx - sx * 0.5, cx + sx * 0.5),
        Range1D(cy - sy * 0.5, cy + sy * 0.5),
        Range1D(cz - sz * 0.5, cz + sz * 0.5),
    )
