"""Lightweight geometry helpers for the deployable scene layout solver.

This module intentionally keeps pure-Python tuple-based vector and XYZW
quaternion operations instead of depending on ManaEnv's numpy-oriented
``manaenv.utils.math_utils`` helpers. The solver, preflight, and runtime
compile paths run in lightweight contexts where stable tuple input/output and
minimal dependencies are more important than array batching.

Quaternions are XYZW-ordered ``(x, y, z, w)`` with identity ``(0, 0, 0, 1)``,
matching the ManaEnv/IsaacLab 3.0 full-stack convention end to end.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..common.constants import GEOM_EPS, QUAT_EPS
from ..common.surface_polygon import Polygon2D

Vec3 = tuple[float, float, float]
QuatXYZW = tuple[float, float, float, float]


def _v3(values: Sequence[float], *, name: str) -> Vec3:
    if len(values) != 3:
        raise ValueError(f"{name}_must_be_vec3")
    return (float(values[0]), float(values[1]), float(values[2]))


def vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_mul(a: Vec3, scale: float) -> Vec3:
    s = float(scale)
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vec3, *, name: str = "vector") -> Vec3:
    n = norm(a)
    if n <= QUAT_EPS:
        raise ValueError(f"{name}_zero_length")
    return (a[0] / n, a[1] / n, a[2] / n)


def quat_normalize(q: Sequence[float]) -> QuatXYZW:
    if len(q) != 4:
        raise ValueError("quaternion_must_be_xyzw")
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= QUAT_EPS:
        raise ValueError("quaternion_zero_length")
    return (x / n, y / n, z / n, w / n)


def quat_mul(a: QuatXYZW, b: QuatXYZW) -> QuatXYZW:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conj(q: QuatXYZW) -> QuatXYZW:
    return (-q[0], -q[1], -q[2], q[3])


def quat_rotate(q: Sequence[float], v: Vec3) -> Vec3:
    qn = quat_normalize(q)
    out = quat_mul(quat_mul(qn, (v[0], v[1], v[2], 0.0)), quat_conj(qn))
    return (out[0], out[1], out[2])


def yaw_quat(yaw_rad: float) -> QuatXYZW:
    half = float(yaw_rad) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


@dataclass(frozen=True)
class Range1D:
    min: float
    max: float

    @classmethod
    def from_pair(cls, values: Sequence[float] | float) -> Range1D:
        if isinstance(values, (int, float)) and not isinstance(values, bool):
            v = float(values)
            return cls(v, v)
        if isinstance(values, (str, bytes)):
            raise ValueError("range_must_have_two_values")
        if not isinstance(values, Sequence):
            raise ValueError("range_must_have_two_values")
        if len(values) == 1:
            a = b = float(values[0])
        elif len(values) == 2:
            a = float(values[0])
            b = float(values[1])
        else:
            raise ValueError("range_must_have_two_values")
        if a > b:
            raise ValueError("range_min_gt_max")
        return cls(a, b)

    @property
    def center(self) -> float:
        return (self.min + self.max) * 0.5

    @property
    def length(self) -> float:
        return max(0.0, self.max - self.min)

    def expanded(self, amount: float) -> Range1D:
        pad = max(0.0, float(amount))
        return Range1D(self.min - pad, self.max + pad)

    def intersect(self, other: Range1D) -> Range1D | None:
        lo = max(self.min, other.min)
        hi = min(self.max, other.max)
        if lo > hi:
            return None
        return Range1D(lo, hi)

    def contains(self, value: float, *, eps: float = GEOM_EPS) -> bool:
        return self.min - eps <= float(value) <= self.max + eps


@dataclass(frozen=True)
class Bounds2D:
    x: Range1D
    y: Range1D

    @classmethod
    def from_size(cls, size: Sequence[float]) -> Bounds2D:
        if len(size) != 2:
            raise ValueError("surface_size_must_be_2d")
        sx = float(size[0])
        sy = float(size[1])
        if sx <= 0.0 or sy <= 0.0:
            raise ValueError("surface_size_must_be_positive")
        return cls(Range1D(-sx * 0.5, sx * 0.5), Range1D(-sy * 0.5, sy * 0.5))

    @classmethod
    def from_dict(cls, data: dict) -> Bounds2D:
        return cls(Range1D.from_pair(data["x"]), Range1D.from_pair(data["y"]))

    def shrink(self, margin: float) -> Bounds2D | None:
        m = max(0.0, float(margin))
        x = Range1D(self.x.min + m, self.x.max - m)
        y = Range1D(self.y.min + m, self.y.max - m)
        if x.min > x.max or y.min > y.max:
            return None
        return Bounds2D(x, y)

    def intersect(self, other: Bounds2D) -> Bounds2D | None:
        x = self.x.intersect(other.x)
        y = self.y.intersect(other.y)
        if x is None or y is None:
            return None
        return Bounds2D(x, y)

    def contains(self, x: float, y: float, *, eps: float = GEOM_EPS) -> bool:
        return self.x.contains(x, eps=eps) and self.y.contains(y, eps=eps)


@dataclass(frozen=True)
class Bounds3D:
    x: Range1D
    y: Range1D
    z: Range1D

    @classmethod
    def from_size(cls, size: Sequence[float]) -> Bounds3D:
        if len(size) != 3:
            raise ValueError("size_must_be_3d")
        sx, sy, sz = (float(size[0]), float(size[1]), float(size[2]))
        if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
            raise ValueError("size_must_be_positive")
        return cls(
            Range1D(-sx * 0.5, sx * 0.5),
            Range1D(-sy * 0.5, sy * 0.5),
            Range1D(-sz * 0.5, sz * 0.5),
        )

    @classmethod
    def from_dict(cls, data: dict) -> Bounds3D:
        return cls(
            Range1D.from_pair(data["x"]),
            Range1D.from_pair(data["y"]),
            Range1D.from_pair(data["z"]),
        )

    @classmethod
    def from_list(cls, values: Sequence[float]) -> Bounds3D:
        if len(values) != 6:
            raise ValueError("bounds_list_must_have_six_values")
        return cls(
            Range1D(float(values[0]), float(values[1])),
            Range1D(float(values[2]), float(values[3])),
            Range1D(float(values[4]), float(values[5])),
        )

    @property
    def size(self) -> Vec3:
        return (self.x.length, self.y.length, self.z.length)

    def corners(self) -> list[Vec3]:
        return [
            (x, y, z)
            for x in (self.x.min, self.x.max)
            for y in (self.y.min, self.y.max)
            for z in (self.z.min, self.z.max)
        ]

    def scaled(self, scale: Sequence[float] | float) -> Bounds3D:
        if isinstance(scale, (int, float)):
            sx = sy = sz = float(scale)
        else:
            if len(scale) != 3:
                raise ValueError("scale_must_be_scalar_or_vec3")
            sx, sy, sz = (float(scale[0]), float(scale[1]), float(scale[2]))
        return Bounds3D(
            Range1D(self.x.min * sx, self.x.max * sx),
            Range1D(self.y.min * sy, self.y.max * sy),
            Range1D(self.z.min * sz, self.z.max * sz),
        )


@dataclass(frozen=True)
class Pose:
    position: Vec3
    quaternion: QuatXYZW = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ManualSupportSurface:
    surface_id: str
    center: Vec3
    x_axis: Vec3
    y_axis: Vec3
    normal: Vec3
    bounds: Bounds2D
    shape: str = "rectangle"
    polygon: Polygon2D | None = None
    margin: float = 0.0

    @classmethod
    def from_spec(cls, spec: object) -> ManualSupportSurface:
        center = _v3(getattr(spec, "center"), name="surface_center")
        x_axis = normalize(_v3(getattr(spec, "x_axis"), name="surface_x_axis"), name="surface_x_axis")
        raw_y_axis = normalize(_v3(getattr(spec, "y_axis"), name="surface_y_axis"), name="surface_y_axis")
        # Orthogonalize y against x to make local coordinates stable.
        y_proj = vec_sub(raw_y_axis, vec_mul(x_axis, dot(raw_y_axis, x_axis)))
        y_axis = normalize(y_proj, name="surface_y_axis_orthogonalized")
        raw_normal = getattr(spec, "normal", None)
        if raw_normal is None:
            normal = normalize(cross(x_axis, y_axis), name="surface_normal")
        else:
            normal = normalize(_v3(raw_normal, name="surface_normal"), name="surface_normal")
        return cls(
            surface_id=str(getattr(spec, "surface_id")),
            center=center,
            x_axis=x_axis,
            y_axis=y_axis,
            normal=normal,
            bounds=getattr(spec, "bounds"),
            shape=str(getattr(spec, "shape", "rectangle") or "rectangle"),
            polygon=getattr(spec, "polygon", None),
            margin=float(getattr(spec, "margin", 0.0) or 0.0),
        )

    def local_to_world(self, x: float, y: float, z_offset: float = 0.0) -> Vec3:
        return vec_add(
            vec_add(
                vec_add(self.center, vec_mul(self.x_axis, float(x))),
                vec_mul(self.y_axis, float(y)),
            ),
            vec_mul(self.normal, float(z_offset)),
        )

    def world_to_local_xy(self, point: Vec3) -> tuple[float, float]:
        delta = vec_sub(point, self.center)
        return (dot(delta, self.x_axis), dot(delta, self.y_axis))

    def effective_bounds(self) -> Bounds2D | None:
        return self.bounds.shrink(self.margin)

    def contains_local_center(self, x: float, y: float, *, eps: float = GEOM_EPS) -> bool:
        bounds = self.effective_bounds()
        if bounds is None or not bounds.contains(x, y, eps=eps):
            return False
        if self.polygon is None:
            return True
        return self.polygon.contains_point(x, y, eps=eps)

    def supports_local_footprint(
        self,
        x: float,
        y: float,
        half_x: float,
        half_y: float,
        *,
        eps: float = GEOM_EPS,
    ) -> bool:
        hx = max(0.0, float(half_x))
        hy = max(0.0, float(half_y))
        bounds = self.effective_bounds()
        if bounds is None:
            return False
        if not bounds.contains(float(x) - hx, float(y) - hy, eps=eps):
            return False
        if not bounds.contains(float(x) + hx, float(y) + hy, eps=eps):
            return False
        if self.polygon is None:
            return True
        return self.polygon.contains_box(x, y, hx, hy, eps=eps)

    def supports_local_footprint_bounds(
        self,
        x: float,
        y: float,
        footprint: Bounds2D,
        *,
        eps: float = GEOM_EPS,
    ) -> bool:
        """Return whether an origin-relative footprint fits on this surface.

        Unlike ``supports_local_footprint`` this does not assume the object
        origin is centered in its support footprint. That matters for authored
        assets whose pivot is at an edge or contact point.
        """
        bounds = self.effective_bounds()
        if bounds is None:
            return False
        min_x = float(x) + footprint.x.min
        max_x = float(x) + footprint.x.max
        min_y = float(y) + footprint.y.min
        max_y = float(y) + footprint.y.max
        if not bounds.contains(min_x, min_y, eps=eps):
            return False
        if not bounds.contains(max_x, max_y, eps=eps):
            return False
        if self.polygon is None:
            return True
        return self.polygon.contains_rect(min_x, max_x, min_y, max_y, eps=eps)


def world_aabb_for_pose(local_bounds: Bounds3D, pose: Pose) -> Bounds3D:
    corners = []
    for corner in local_bounds.corners():
        rotated = quat_rotate(pose.quaternion, corner)
        corners.append(vec_add(pose.position, rotated))
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    zs = [p[2] for p in corners]
    return Bounds3D(Range1D(min(xs), max(xs)), Range1D(min(ys), max(ys)), Range1D(min(zs), max(zs)))


def support_contact_offset(
    local_bounds: Bounds3D,
    quaternion: Sequence[float],
    normal: Vec3,
    *,
    contact_offset: float = 0.0,
) -> float:
    """Return origin offset along support normal needed for contact.

    This is orientation-aware: an upside-down object should contact the support
    with its rotated top, not with the unrotated local z-min.
    """
    n = normalize(normal, name="support_normal")
    min_projection = min(dot(quat_rotate(quaternion, corner), n) for corner in local_bounds.corners())
    return -float(min_projection) + float(contact_offset)


def projected_footprint_size(
    local_bounds: Bounds3D,
    quaternion: Sequence[float],
    x_axis: Vec3 = (1.0, 0.0, 0.0),
    y_axis: Vec3 = (0.0, 1.0, 0.0),
) -> tuple[float, float]:
    """Return the rotated local AABB footprint projected into a support frame."""
    bounds = projected_footprint_bounds(local_bounds, quaternion, x_axis, y_axis)
    return (bounds.x.length, bounds.y.length)


def projected_footprint_bounds(
    local_bounds: Bounds3D,
    quaternion: Sequence[float],
    x_axis: Vec3 = (1.0, 0.0, 0.0),
    y_axis: Vec3 = (0.0, 1.0, 0.0),
) -> Bounds2D:
    """Return origin-relative rotated AABB footprint bounds in a support frame."""
    sx = normalize(x_axis, name="footprint_x_axis")
    sy = normalize(y_axis, name="footprint_y_axis")
    projected = [quat_rotate(quaternion, corner) for corner in local_bounds.corners()]
    xs = [dot(point, sx) for point in projected]
    ys = [dot(point, sy) for point in projected]
    return Bounds2D(Range1D(min(xs), max(xs)), Range1D(min(ys), max(ys)))


def projected_footprint_size_with_yaw_range(
    local_bounds: Bounds3D,
    quaternion: Sequence[float],
    yaw_range: tuple[float, float] | None,
    x_axis: Vec3 = (1.0, 0.0, 0.0),
    y_axis: Vec3 = (0.0, 1.0, 0.0),
) -> tuple[float, float]:
    """Return a conservative exact footprint bound for world-z yaw variation."""
    bounds = projected_footprint_bounds_with_yaw_range(local_bounds, quaternion, yaw_range, x_axis, y_axis)
    return (bounds.x.length, bounds.y.length)


def projected_footprint_bounds_with_yaw_range(
    local_bounds: Bounds3D,
    quaternion: Sequence[float],
    yaw_range: tuple[float, float] | None,
    x_axis: Vec3 = (1.0, 0.0, 0.0),
    y_axis: Vec3 = (0.0, 1.0, 0.0),
) -> Bounds2D:
    """Return conservative origin-relative footprint bounds for yaw variation."""
    if yaw_range is None or abs(float(yaw_range[1]) - float(yaw_range[0])) <= GEOM_EPS:
        yaw = 0.0 if yaw_range is None else float(yaw_range[0])
        return projected_footprint_bounds(
            local_bounds,
            quat_mul(yaw_quat(yaw), quat_normalize(quaternion)),
            x_axis,
            y_axis,
        )
    sx = normalize(x_axis, name="footprint_x_axis")
    sy = normalize(y_axis, name="footprint_y_axis")
    base = [quat_rotate(quaternion, corner) for corner in local_bounds.corners()]
    coords = [(dot(point, sx), dot(point, sy)) for point in base]
    lo, hi = float(yaw_range[0]), float(yaw_range[1])
    if lo > hi:
        lo, hi = hi, lo
    x_min, x_max = _yaw_projection_bounds(coords, lo, hi, axis="x")
    y_min, y_max = _yaw_projection_bounds(coords, lo, hi, axis="y")
    return Bounds2D(Range1D(x_min, x_max), Range1D(y_min, y_max))


def _yaw_projection_bounds(
    coords: list[tuple[float, float]], lo: float, hi: float, *, axis: str
) -> tuple[float, float]:
    if hi - lo >= math.tau:
        radius = max(math.sqrt(x * x + y * y) for x, y in coords)
        return (-radius, radius)
    values: list[float] = []
    for coord in coords:
        candidates = [lo, hi]
        candidates.extend(_single_projection_peak_angles(coord, axis=axis, lo=lo, hi=hi))
        for yaw in candidates:
            values.append(_single_projection(coord, yaw, axis=axis))
    return (min(values), max(values))


def _single_projection_peak_angles(
    point: tuple[float, float],
    *,
    axis: str,
    lo: float,
    hi: float,
) -> list[float]:
    x, y = point
    if axis == "x":
        a, b = x, -y
    else:
        a, b = y, x
    if abs(a) <= GEOM_EPS and abs(b) <= GEOM_EPS:
        return []
    peak = math.atan2(b, a)
    start = math.floor((lo - peak) / math.tau) - 1
    end = math.ceil((hi - peak) / math.tau) + 1
    out: list[float] = []
    for k in range(start, end + 1):
        angle = peak + math.tau * k
        if lo <= angle <= hi:
            out.append(angle)
        opposite = angle + math.pi
        if lo <= opposite <= hi:
            out.append(opposite)
    return out


def _single_projection(point: tuple[float, float], yaw: float, *, axis: str) -> float:
    x, y = point
    if axis == "x":
        return x * math.cos(yaw) - y * math.sin(yaw)
    return x * math.sin(yaw) + y * math.cos(yaw)


def _max_yaw_extent(coords: list[tuple[float, float]], lo: float, hi: float, *, axis: str) -> float:
    if hi - lo >= math.tau:
        return max(_pair_projection_amplitude(left, right, axis=axis) for left in coords for right in coords)
    best = 0.0
    for left in coords:
        for right in coords:
            candidates = [lo, hi]
            candidates.extend(_projection_peak_angles(left, right, axis=axis, lo=lo, hi=hi))
            for yaw in candidates:
                best = max(best, _pair_projection(left, right, yaw, axis=axis))
    return max(0.0, best)


def _projection_peak_angles(
    left: tuple[float, float],
    right: tuple[float, float],
    *,
    axis: str,
    lo: float,
    hi: float,
) -> list[float]:
    lx, ly = left
    rx, ry = right
    dx = lx - rx
    dy = ly - ry
    if axis == "x":
        a, b = dx, -dy
    else:
        a, b = dy, dx
    if abs(a) <= GEOM_EPS and abs(b) <= GEOM_EPS:
        return []
    peak = math.atan2(b, a)
    start = math.floor((lo - peak) / math.tau) - 1
    end = math.ceil((hi - peak) / math.tau) + 1
    return [peak + math.tau * k for k in range(start, end + 1) if lo <= peak + math.tau * k <= hi]


def _pair_projection(left: tuple[float, float], right: tuple[float, float], yaw: float, *, axis: str) -> float:
    lx, ly = left
    rx, ry = right
    if axis == "x":
        return (lx - rx) * math.cos(yaw) - (ly - ry) * math.sin(yaw)
    return (lx - rx) * math.sin(yaw) + (ly - ry) * math.cos(yaw)


def _pair_projection_amplitude(left: tuple[float, float], right: tuple[float, float], *, axis: str) -> float:
    lx, ly = left
    rx, ry = right
    dx = lx - rx
    dy = ly - ry
    return math.sqrt(dx * dx + dy * dy)


def bounds_overlap(a: Bounds3D, b: Bounds3D, *, eps: float = GEOM_EPS) -> bool:
    return (
        a.x.min < b.x.max - eps
        and b.x.min < a.x.max - eps
        and a.y.min < b.y.max - eps
        and b.y.min < a.y.max - eps
        and a.z.min < b.z.max - eps
        and b.z.min < a.z.max - eps
    )


def union_aabb(bounds: Iterable[Bounds3D]) -> Bounds3D:
    items = list(bounds)
    if not items:
        raise ValueError("bounds_empty")
    return Bounds3D(
        Range1D(min(b.x.min for b in items), max(b.x.max for b in items)),
        Range1D(min(b.y.min for b in items), max(b.y.max for b in items)),
        Range1D(min(b.z.min for b in items), max(b.z.max for b in items)),
    )
