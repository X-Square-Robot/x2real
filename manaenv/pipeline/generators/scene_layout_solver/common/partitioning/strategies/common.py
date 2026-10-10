from __future__ import annotations

import random
from collections.abc import Iterable

from ...candidate_grid import grid_dims
from ...constants import GEOM_EPS
from ...geometry import Bounds2D
from ..geometry import case_is_safe, center_domain_for_object
from ..models import PartitionCell, PartitionProblem


def cells_from_raw_assignment(
    problem: PartitionProblem, names: list[str], raw_cells: list[Bounds2D]
) -> dict[str, PartitionCell] | None:
    raw = {name: cell for name, cell in zip(names, raw_cells)}
    return cells_from_named_raw(problem, raw)


def cells_from_named_raw(problem: PartitionProblem, raw: dict[str, Bounds2D]) -> dict[str, PartitionCell] | None:
    cells: dict[str, PartitionCell] = {}
    for name, bounds in raw.items():
        cell = center_domain_for_object(problem, name, bounds)
        if cell is None:
            return None
        cells[name] = cell
    if not case_is_safe(problem, cells):
        return None
    return cells


def strategy_dims(count: int, *, strip_only: bool) -> list[tuple[int, int]]:
    if strip_only:
        return [(count, 1), (1, count)]
    dims = grid_dims(count)
    extras = [(cols, rows) for cols in range(1, count + 2) for rows in range(1, count + 2) if cols * rows >= count]
    out: list[tuple[int, int]] = []
    for item in [*dims, *extras]:
        if item not in out:
            out.append(item)
    return out


def select_cells(cells: list[Bounds2D], count: int, *, rng: random.Random) -> list[Bounds2D]:
    if len(cells) == count:
        out = list(cells)
    else:
        out = [cells[index] for index in sorted(rng.sample(range(len(cells)), count))]
    rng.shuffle(out)
    return out


def select_coords(
    coords: list[tuple[int, int]], count: int, *, rng: random.Random, require_full_axes: bool
) -> list[tuple[int, int]]:
    if len(coords) == count:
        return list(coords)
    for _ in range(32):
        selected = rng.sample(coords, count)
        if not require_full_axes:
            return selected
        xs = {item[0] for item in selected}
        ys = {item[1] for item in selected}
        if len(xs) > 1 and len(ys) > 1:
            return selected
    return rng.sample(coords, count)


def allocate_lengths(total: float, weights: list[float], *, rng: random.Random) -> list[float]:
    weights = [max(0.001, float(value)) * rng.uniform(0.9, 1.1) for value in weights]
    required = sum(weights)
    if required <= GEOM_EPS:
        return [total / max(1, len(weights)) for _ in weights]
    if required >= total:
        return [total * value / required for value in weights]
    extra = total - required
    extra_weights = [value * rng.uniform(0.5, 2.0) for value in weights]
    extra_total = sum(extra_weights) or float(len(weights))
    return [base + extra * extra_weights[index] / extra_total for index, base in enumerate(weights)]


def boundaries(start: float, sizes: Iterable[float]) -> list[float]:
    out = [float(start)]
    for size in sizes:
        out.append(out[-1] + float(size))
    return out


def split_axis(bounds: Bounds2D, *, depth: int, rng: random.Random) -> str:
    if depth % 3 == 0:
        return "x" if bounds.x.length >= bounds.y.length else "y"
    return "x" if rng.random() < 0.5 else "y"


def group_weight(problem: PartitionProblem, names: list[str], *, axis: str) -> float:
    idx = 0 if axis == "x" else 1
    return sum(problem.profiles[name].footprint[idx] + problem.min_spacing for name in names)


def profile_axis(problem: PartitionProblem, name: str, axis: str) -> float:
    return problem.profiles[name].footprint[0 if axis == "x" else 1]
