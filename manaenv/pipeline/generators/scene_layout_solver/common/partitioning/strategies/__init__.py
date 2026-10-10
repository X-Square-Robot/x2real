from __future__ import annotations

import random

from ..models import PartitionCase, PartitionProblem
from ..policy import (
    FREE_RECT,
    GUILLOTINE,
    PERMUTED_UNIFORM_GRID,
    POINT_FALLBACK,
    SHELF,
    STRIP_FALLBACK,
    UNIFORM_GRID,
    WEIGHTED_GRID,
)
from .free_rect import free_rect_cases
from .grid import permuted_uniform_grid_cases, uniform_grid_cases, weighted_grid_cases
from .guillotine import guillotine_cases
from .point import point_fallback_cases
from .shelf import shelf_cases


def generate_for_strategy(
    problem: PartitionProblem, strategy: str, *, rng: random.Random, budget: int
) -> list[PartitionCase]:
    if strategy == UNIFORM_GRID:
        return uniform_grid_cases(problem, rng=rng, budget=budget, tier=UNIFORM_GRID, permute=False, strip_only=False)
    if strategy == PERMUTED_UNIFORM_GRID:
        return permuted_uniform_grid_cases(problem, rng=rng, budget=budget)
    if strategy == WEIGHTED_GRID:
        return weighted_grid_cases(problem, rng=rng, budget=budget)
    if strategy == FREE_RECT:
        return free_rect_cases(problem, rng=rng, budget=budget)
    if strategy == SHELF:
        return shelf_cases(problem, rng=rng, budget=budget)
    if strategy == GUILLOTINE:
        return guillotine_cases(problem, rng=rng, budget=budget)
    if strategy == STRIP_FALLBACK:
        return uniform_grid_cases(problem, rng=rng, budget=budget, tier=STRIP_FALLBACK, permute=True, strip_only=True)
    if strategy == POINT_FALLBACK:
        return point_fallback_cases(problem, rng=rng, budget=budget)
    return []
