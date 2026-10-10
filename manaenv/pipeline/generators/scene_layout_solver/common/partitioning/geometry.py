from __future__ import annotations

import math

from ..constants import GEOM_EPS
from ..geometry import Bounds2D, Range1D
from .models import PartitionCell, PartitionObstacle, PartitionProblem


def center_domain_for_object(
    problem: PartitionProblem,
    object_name: str,
    raw: Bounds2D,
    *,
    boundary_spacing: float | None = None,
) -> PartitionCell | None:
    profile = problem.profiles[object_name]
    raw_boundary_spacing = problem.boundary_spacing if boundary_spacing is None else boundary_spacing
    spacing = problem.min_spacing * 0.5 if raw_boundary_spacing is None else max(0.0, float(raw_boundary_spacing))
    cell = PartitionCell(
        object_name=object_name,
        x=Range1D(
            raw.x.min + spacing - profile.footprint_bounds.x.min, raw.x.max - spacing - profile.footprint_bounds.x.max
        ),
        y=Range1D(
            raw.y.min + spacing - profile.footprint_bounds.y.min, raw.y.max - spacing - profile.footprint_bounds.y.max
        ),
        z=profile.z,
    )
    return accepted_cell(problem, cell)


def accepted_cell(problem: PartitionProblem, cell: PartitionCell) -> PartitionCell | None:
    if cell.x.min > cell.x.max + GEOM_EPS or cell.y.min > cell.y.max + GEOM_EPS:
        return None
    cell = _normalized_cell(cell)
    if problem.clip_cell is not None:
        clipped = problem.clip_cell(cell.object_name, cell.x, cell.y)
        if clipped is None:
            return None
        x, y = clipped
        if x.min > x.max + GEOM_EPS or y.min > y.max + GEOM_EPS:
            return None
        cell = PartitionCell(cell.object_name, x, y, cell.z)
    if problem.accept_cell is None:
        return cell if cell_obstacle_safe(problem, cell) else None
    cx, cy = cell.center
    for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
        x = _scale_range(cell.x, cx, scale)
        y = _scale_range(cell.y, cy, scale)
        scaled = PartitionCell(cell.object_name, x, y, cell.z)
        if problem.accept_cell(cell.object_name, x, y) and cell_obstacle_safe(problem, scaled):
            return scaled
    return None


def case_is_safe(problem: PartitionProblem, cells: dict[str, PartitionCell]) -> bool:
    if set(cells) != set(problem.names):
        return False
    names = sorted(cells)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if not cells_pair_safe(problem, cells[left], cells[right]):
                return False
    return True


def cells_pair_safe(problem: PartitionProblem, left: PartitionCell, right: PartitionCell) -> bool:
    if _spacing_mode(problem) == "center_distance":
        return center_domain_min_distance(left.x, left.y, right.x, right.y) >= problem.min_spacing - GEOM_EPS
    left_profile = problem.profiles[left.object_name]
    right_profile = problem.profiles[right.object_name]
    left_x = left_profile.swept_x(left)
    right_x = right_profile.swept_x(right)
    left_y = left_profile.swept_y(left)
    right_y = right_profile.swept_y(right)
    return (
        range_gap(left_x, right_x) >= problem.min_spacing - GEOM_EPS
        or range_gap(left_y, right_y) >= problem.min_spacing - GEOM_EPS
    )


def _spacing_mode(problem: PartitionProblem) -> str:
    raw = str(getattr(problem, "spacing_mode", "footprint_gap") or "footprint_gap").strip().lower()
    return "center_distance" if raw in {"center", "center_distance", "center-distance"} else "footprint_gap"


def cell_obstacle_safe(problem: PartitionProblem, cell: PartitionCell) -> bool:
    if not problem.obstacles:
        return True
    profile = problem.profiles[cell.object_name]
    cell_x = profile.swept_x(cell)
    cell_y = profile.swept_y(cell)
    return all(
        ranges_separated(cell_x, cell_y, obstacle.x, obstacle.y, min_gap=problem.min_spacing)
        for obstacle in problem.obstacles
    )


def obstacle_pair_safe(problem: PartitionProblem, left: PartitionObstacle, right: PartitionObstacle) -> bool:
    return ranges_separated(left.x, left.y, right.x, right.y, min_gap=problem.min_spacing)


def ranges_separated(
    left_x: Range1D,
    left_y: Range1D,
    right_x: Range1D,
    right_y: Range1D,
    *,
    min_gap: float,
) -> bool:
    return (
        range_gap(left_x, right_x) >= float(min_gap) - GEOM_EPS
        or range_gap(left_y, right_y) >= float(min_gap) - GEOM_EPS
    )


def range_gap(a: Range1D, b: Range1D) -> float:
    return max(b.min - a.max, a.min - b.max)


def center_domain_min_distance(left_x: Range1D, left_y: Range1D, right_x: Range1D, right_y: Range1D) -> float:
    dx = max(0.0, range_gap(left_x, right_x))
    dy = max(0.0, range_gap(left_y, right_y))
    return math.hypot(dx, dy)


def case_signature(cells: dict[str, PartitionCell]) -> tuple[tuple[str, float, float, float, float], ...]:
    return tuple(
        (
            name,
            round(cells[name].x.min, 4),
            round(cells[name].x.max, 4),
            round(cells[name].y.min, 4),
            round(cells[name].y.max, 4),
        )
        for name in sorted(cells)
    )


def object_size_similarity(problem: PartitionProblem) -> float:
    areas = [max(profile.area, GEOM_EPS) for profile in problem.profiles.values()]
    if not areas:
        return 1.0
    return max(areas) / max(min(areas), GEOM_EPS)


def _scale_range(axis: Range1D, center: float, scale: float) -> Range1D:
    half = max(0.0, axis.length) * 0.5 * max(0.0, min(1.0, float(scale)))
    return Range1D(center - half, center + half)


def _normalized_cell(cell: PartitionCell) -> PartitionCell:
    return PartitionCell(
        cell.object_name,
        Range1D(min(cell.x.min, cell.x.max), max(cell.x.min, cell.x.max)),
        Range1D(min(cell.y.min, cell.y.max), max(cell.y.min, cell.y.max)),
        cell.z,
    )
