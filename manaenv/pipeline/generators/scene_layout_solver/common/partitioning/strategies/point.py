from __future__ import annotations

import random

from ...geometry import Bounds2D, Range1D
from ..geometry import accepted_cell, case_is_safe
from ..models import PartitionCase, PartitionCell, PartitionProblem
from ..policy import POINT_FALLBACK


def point_fallback_cases(problem: PartitionProblem, *, rng: random.Random, budget: int) -> list[PartitionCase]:
    out: list[PartitionCase] = []
    attempts = 0
    max_attempts = max(200, budget * 40)
    remaining_trials = [min(20000, max(1000, budget * max(1, problem.count) * 16))]
    names = sorted(problem.names, key=lambda name: (-problem.profiles[name].area, name))
    while len(out) < budget and attempts < max_attempts and remaining_trials[0] > 0:
        attempts += 1
        shuffled = list(names)
        if attempts % 3:
            rng.shuffle(shuffled)
        cells: dict[str, PartitionCell] = {}
        if not _place_points(problem, shuffled, rng=rng, cells=cells, remaining_trials=remaining_trials):
            continue
        out.append(PartitionCase(cells=cells, tier=POINT_FALLBACK, topology="point"))
    return out


def _place_points(
    problem: PartitionProblem,
    names: list[str],
    *,
    rng: random.Random,
    cells: dict[str, PartitionCell],
    remaining_trials: list[int],
) -> bool:
    if not names:
        return True
    if remaining_trials[0] <= 0:
        return False
    name = names[0]
    profile = problem.profiles[name]
    fit = Bounds2D(
        Range1D(problem.bounds.x.min + profile.footprint[0] * 0.5, problem.bounds.x.max - profile.footprint[0] * 0.5),
        Range1D(problem.bounds.y.min + profile.footprint[1] * 0.5, problem.bounds.y.max - profile.footprint[1] * 0.5),
    )
    if fit.x.min > fit.x.max or fit.y.min > fit.y.max:
        return False
    for _ in range(96):
        remaining_trials[0] -= 1
        if remaining_trials[0] <= 0:
            return False
        x = rng.uniform(fit.x.min, fit.x.max)
        y = rng.uniform(fit.y.min, fit.y.max)
        cell = accepted_cell(problem, PartitionCell(name, Range1D(x, x), Range1D(y, y), profile.z))
        if cell is None:
            continue
        trial = {**cells, name: cell}
        if case_is_safe(
            PartitionProblem(
                names=tuple(trial),
                bounds=problem.bounds,
                profiles=problem.profiles,
                min_spacing=problem.min_spacing,
                spacing_mode=problem.spacing_mode,
                jitter_ratio=problem.jitter_ratio,
                sample_method=problem.sample_method,
                grid_step=problem.grid_step,
                grid_snap=problem.grid_snap,
                align_axis=problem.align_axis,
                order_axis=problem.order_axis,
                boundary_spacing=problem.boundary_spacing,
                accept_cell=problem.accept_cell,
                clip_cell=problem.clip_cell,
                obstacles=problem.obstacles,
            ),
            trial,
        ):
            cells[name] = cell
            if _place_points(problem, names[1:], rng=rng, cells=cells, remaining_trials=remaining_trials):
                return True
            cells.pop(name, None)
    return False
