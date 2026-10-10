from __future__ import annotations

import random

from ...geometry import Bounds2D, Range1D
from ..models import PartitionCase, PartitionProblem
from ..policy import GUILLOTINE
from .common import cells_from_named_raw, group_weight, split_axis


def guillotine_cases(problem: PartitionProblem, *, rng: random.Random, budget: int) -> list[PartitionCase]:
    out: list[PartitionCase] = []
    attempts = 0
    max_attempts = max(96, budget * 20)
    while len(out) < budget and attempts < max_attempts:
        attempts += 1
        names = sorted(problem.names, key=lambda name: (-problem.profiles[name].area, name))
        if attempts % 4:
            rng.shuffle(names)
        raw = guillotine_raw(problem, names, problem.bounds, rng=rng, depth=0)
        if raw is None:
            continue
        cells = cells_from_named_raw(problem, raw)
        if cells is None:
            continue
        out.append(PartitionCase(cells=cells, tier=GUILLOTINE, topology="guillotine"))
    return out


def guillotine_raw(
    problem: PartitionProblem,
    names: list[str],
    bounds: Bounds2D,
    *,
    rng: random.Random,
    depth: int,
) -> dict[str, Bounds2D] | None:
    if not names:
        return {}
    if len(names) == 1:
        return {names[0]: bounds}
    axis = split_axis(bounds, depth=depth, rng=rng)
    ordered = list(names)
    if depth % 2:
        ordered = sorted(
            names, key=lambda name: problem.profiles[name].footprint[0 if axis == "x" else 1], reverse=True
        )
    split = rng.randint(1, len(ordered) - 1)
    left_names = ordered[:split]
    right_names = ordered[split:]
    left_weight = group_weight(problem, left_names, axis=axis)
    right_weight = group_weight(problem, right_names, axis=axis)
    total = max(1e-9, left_weight + right_weight)
    ratio = min(0.85, max(0.15, left_weight / total + rng.uniform(-0.12, 0.12)))
    if axis == "x":
        cut = bounds.x.min + bounds.x.length * ratio
        left_bounds = Bounds2D(Range1D(bounds.x.min, cut), bounds.y)
        right_bounds = Bounds2D(Range1D(cut, bounds.x.max), bounds.y)
    else:
        cut = bounds.y.min + bounds.y.length * ratio
        left_bounds = Bounds2D(bounds.x, Range1D(bounds.y.min, cut))
        right_bounds = Bounds2D(bounds.x, Range1D(cut, bounds.y.max))
    left_raw = guillotine_raw(problem, left_names, left_bounds, rng=rng, depth=depth + 1)
    right_raw = guillotine_raw(problem, right_names, right_bounds, rng=rng, depth=depth + 1)
    if left_raw is None or right_raw is None:
        return None
    return {**left_raw, **right_raw}
