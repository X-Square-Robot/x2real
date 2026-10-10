from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..common.constants import GEOM_EPS

Point2D = tuple[float, float]


def _point2(raw: Sequence[float], *, target: str) -> Point2D:
    if len(raw) != 2:
        raise ValueError(f"{target}_must_be_point2")
    return (float(raw[0]), float(raw[1]))


def points2(raw: object, *, target: str) -> tuple[Point2D, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{target}_must_be_point_list")
    points_list: list[Point2D] = []
    for item in raw:
        if not isinstance(item, (list, tuple)):
            raise ValueError(f"{target}_must_be_point_list")
        points_list.append(_point2(item, target=target))
    points = tuple(points_list)
    if len(points) < 3:
        raise ValueError(f"{target}_must_have_at_least_three_points")
    return points


def _point_on_segment(point: Point2D, a: Point2D, b: Point2D, *, eps: float = GEOM_EPS) -> bool:
    px, py = point
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= eps


def _point_in_ring(point: Point2D, ring: tuple[Point2D, ...], *, eps: float = GEOM_EPS) -> bool:
    x, y = point
    inside = False
    n = len(ring)
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        if _point_on_segment(point, a, b, eps=eps):
            return True
        ax, ay = a
        bx, by = b
        crosses = (ay > y) != (by > y)
        if not crosses:
            continue
        x_at_y = (bx - ax) * (y - ay) / (by - ay) + ax
        if x < x_at_y + eps:
            inside = not inside
    return inside


def _ring_edges(ring: tuple[Point2D, ...]) -> tuple[tuple[Point2D, Point2D], ...]:
    return tuple((ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring)))


def _rect_points(min_x: float, max_x: float, min_y: float, max_y: float) -> tuple[Point2D, Point2D, Point2D, Point2D]:
    return (
        (float(min_x), float(min_y)),
        (float(max_x), float(min_y)),
        (float(max_x), float(max_y)),
        (float(min_x), float(max_y)),
    )


def _rect_edges(min_x: float, max_x: float, min_y: float, max_y: float) -> tuple[tuple[Point2D, Point2D], ...]:
    points = _rect_points(min_x, max_x, min_y, max_y)
    return tuple((points[i], points[(i + 1) % len(points)]) for i in range(len(points)))


def _point_in_rect(
    point: Point2D, min_x: float, max_x: float, min_y: float, max_y: float, *, eps: float = GEOM_EPS
) -> bool:
    x, y = point
    return min_x - eps <= x <= max_x + eps and min_y - eps <= y <= max_y + eps


def _orientation(a: Point2D, b: Point2D, c: Point2D) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Point2D, b: Point2D, c: Point2D, d: Point2D, *, eps: float = GEOM_EPS) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    if abs(ab_c) <= eps and _point_on_segment(c, a, b, eps=eps):
        return True
    if abs(ab_d) <= eps and _point_on_segment(d, a, b, eps=eps):
        return True
    if abs(cd_a) <= eps and _point_on_segment(a, c, d, eps=eps):
        return True
    if abs(cd_b) <= eps and _point_on_segment(b, c, d, eps=eps):
        return True
    return (ab_c > eps) != (ab_d > eps) and (cd_a > eps) != (cd_b > eps)


def _ring_intersects_rect(
    ring: tuple[Point2D, ...],
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    *,
    eps: float = GEOM_EPS,
) -> bool:
    rect = _rect_points(min_x, max_x, min_y, max_y)
    if any(_point_in_ring(point, ring, eps=eps) for point in rect):
        return True
    if any(_point_in_rect(point, min_x, max_x, min_y, max_y, eps=eps) for point in ring):
        return True
    rect_edges = _rect_edges(min_x, max_x, min_y, max_y)
    for edge_a, edge_b in _ring_edges(ring):
        for rect_a, rect_b in rect_edges:
            if _segments_intersect(edge_a, edge_b, rect_a, rect_b, eps=eps):
                return True
    return False


@dataclass(frozen=True)
class Polygon2D:
    outer: tuple[Point2D, ...]
    holes: tuple[tuple[Point2D, ...], ...] = ()

    @classmethod
    def from_payload(cls, outer: object, holes: object | None = None) -> Polygon2D:
        parsed_outer = points2(outer, target="surface_polygon_outer")
        parsed_holes: list[tuple[Point2D, ...]] = []
        if holes is not None:
            if not isinstance(holes, (list, tuple)):
                raise ValueError("surface_polygon_holes_must_be_list")
            for idx, hole in enumerate(holes):
                parsed_holes.append(points2(hole, target=f"surface_polygon_hole_{idx}"))
        return cls(outer=parsed_outer, holes=tuple(parsed_holes))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [point[0] for point in self.outer]
        ys = [point[1] for point in self.outer]
        return (min(xs), max(xs), min(ys), max(ys))

    def contains_point(self, x: float, y: float, *, eps: float = GEOM_EPS) -> bool:
        point = (float(x), float(y))
        if not _point_in_ring(point, self.outer, eps=eps):
            return False
        for hole in self.holes:
            if _point_in_ring(point, hole, eps=eps):
                return False
        return True

    def contains_box(self, x: float, y: float, half_x: float, half_y: float, *, eps: float = GEOM_EPS) -> bool:
        hx = max(0.0, float(half_x))
        hy = max(0.0, float(half_y))
        return self.contains_rect(float(x) - hx, float(x) + hx, float(y) - hy, float(y) + hy, eps=eps)

    def contains_rect(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        *,
        eps: float = GEOM_EPS,
    ) -> bool:
        rect = _rect_points(min_x, max_x, min_y, max_y)
        if not all(_point_in_ring(point, self.outer, eps=eps) for point in rect):
            return False
        for edge_a, edge_b in _ring_edges(self.outer):
            for rect_a, rect_b in _rect_edges(min_x, max_x, min_y, max_y):
                if _segments_intersect(edge_a, edge_b, rect_a, rect_b, eps=eps):
                    return False
        for hole in self.holes:
            if _ring_intersects_rect(hole, min_x, max_x, min_y, max_y, eps=eps):
                return False
        return True
