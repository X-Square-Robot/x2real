from __future__ import annotations

import random

from ...geometry import Bounds2D, Range1D
from ..models import PartitionCase, PartitionProblem
from ..policy import SHELF
from .common import allocate_lengths, boundaries, cells_from_named_raw, group_weight, profile_axis


def shelf_cases(problem: PartitionProblem, *, rng: random.Random, budget: int) -> list[PartitionCase]:
    """Generate size-aware row/column shelf layouts for heterogeneous objects.

    The cells remain continuous domains, not fixed points: shelves define a
    topology and then leave as much per-object area as possible inside it.
    """
    out: list[PartitionCase] = []
    attempts = 0
    max_attempts = max(24, budget * 4)
    while len(out) < budget and attempts < max_attempts:
        attempts += 1
        axis = "x" if attempts % 2 == 0 else "y"
        names = sorted(
            problem.names, key=lambda name: (-profile_axis(problem, name, axis), -problem.profiles[name].area, name)
        )
        if attempts % 3 == 0:
            rng.shuffle(names)
        groups = _shelf_groups(problem, names, axis=axis, rng=rng, attempt=attempts)
        raw = _shelf_raw(problem, groups, axis=axis, rng=rng)
        if raw is None:
            continue
        cells = cells_from_named_raw(problem, raw)
        if cells is None:
            continue
        out.append(PartitionCase(cells=cells, tier=SHELF, topology=f"shelf_{axis}_{len(groups)}"))
    return out


def _shelf_groups(
    problem: PartitionProblem,
    names: list[str],
    *,
    axis: str,
    rng: random.Random,
    attempt: int,
) -> list[list[str]]:
    if not names:
        return []
    count = len(names)
    shelf_count = max(1, min(count, int(round(count**0.5))))
    if attempt % 5 == 0:
        shelf_count = min(count, shelf_count + 1)
    groups: list[list[str]] = [[] for _ in range(shelf_count)]
    lengths = [0.0 for _ in groups]
    cross_axis = "y" if axis == "x" else "x"
    for name in names:
        index = min(range(len(groups)), key=lambda idx: lengths[idx])
        groups[index].append(name)
        lengths[index] += profile_axis(problem, name, cross_axis) + problem.min_spacing
    if attempt % 4 == 0:
        rng.shuffle(groups)
    return [group for group in groups if group]


def _shelf_raw(
    problem: PartitionProblem,
    groups: list[list[str]],
    *,
    axis: str,
    rng: random.Random,
) -> dict[str, Bounds2D] | None:
    if not groups:
        return {}
    if axis == "x":
        shelf_weights = [group_weight(problem, group, axis="y") for group in groups]
        shelf_sizes = allocate_lengths(problem.bounds.y.length, shelf_weights, rng=rng)
        shelf_bounds = boundaries(problem.bounds.y.min, shelf_sizes)
        raw: dict[str, Bounds2D] = {}
        for shelf_index, group in enumerate(groups):
            item_weights = [profile_axis(problem, name, "x") + problem.min_spacing for name in group]
            item_sizes = allocate_lengths(problem.bounds.x.length, item_weights, rng=rng)
            item_bounds = boundaries(problem.bounds.x.min, item_sizes)
            y_range = Range1D(shelf_bounds[shelf_index], shelf_bounds[shelf_index + 1])
            for item_index, name in enumerate(group):
                raw[name] = Bounds2D(Range1D(item_bounds[item_index], item_bounds[item_index + 1]), y_range)
        return raw
    shelf_weights = [group_weight(problem, group, axis="x") for group in groups]
    shelf_sizes = allocate_lengths(problem.bounds.x.length, shelf_weights, rng=rng)
    shelf_bounds = boundaries(problem.bounds.x.min, shelf_sizes)
    raw = {}
    for shelf_index, group in enumerate(groups):
        item_weights = [profile_axis(problem, name, "y") + problem.min_spacing for name in group]
        item_sizes = allocate_lengths(problem.bounds.y.length, item_weights, rng=rng)
        item_bounds = boundaries(problem.bounds.y.min, item_sizes)
        x_range = Range1D(shelf_bounds[shelf_index], shelf_bounds[shelf_index + 1])
        for item_index, name in enumerate(group):
            raw[name] = Bounds2D(x_range, Range1D(item_bounds[item_index], item_bounds[item_index + 1]))
    return raw
