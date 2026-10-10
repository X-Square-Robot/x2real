from __future__ import annotations

from .geometry import object_size_similarity
from .models import PartitionProblem, StrategyPolicy

UNIFORM_GRID = "uniform_grid"
PERMUTED_UNIFORM_GRID = "permuted_uniform_grid"
WEIGHTED_GRID = "weighted_grid"
FREE_RECT = "free_rect"
SHELF = "shelf"
GUILLOTINE = "guillotine"
STRIP_FALLBACK = "strip_fallback"
POINT_FALLBACK = "point_fallback"


def choose_strategy_policy(problem: PartitionProblem, *, relation_kind: str = "region_sample") -> StrategyPolicy:
    if problem.count <= 1:
        return StrategyPolicy((UNIFORM_GRID,), relation_kind=relation_kind)
    if problem.obstacles:
        return StrategyPolicy(
            (FREE_RECT, GUILLOTINE, WEIGHTED_GRID, UNIFORM_GRID, STRIP_FALLBACK, POINT_FALLBACK),
            relation_kind=relation_kind,
        )
    if problem.align_axis or problem.order_axis:
        return StrategyPolicy((STRIP_FALLBACK, POINT_FALLBACK), relation_kind=relation_kind)

    method = (problem.sample_method or "random").strip().lower()
    similar = object_size_similarity(problem) <= 2.25

    if relation_kind == "volume_group":
        return StrategyPolicy((UNIFORM_GRID, PERMUTED_UNIFORM_GRID), relation_kind=relation_kind)

    if relation_kind == "scatter_group":
        if method in {"poisson", "poisson_disk"}:
            return StrategyPolicy((UNIFORM_GRID, PERMUTED_UNIFORM_GRID, POINT_FALLBACK), relation_kind=relation_kind)
        return StrategyPolicy(
            (UNIFORM_GRID, PERMUTED_UNIFORM_GRID, STRIP_FALLBACK, POINT_FALLBACK), relation_kind=relation_kind
        )

    if method in {"grid", "jittered_grid"} and similar:
        return StrategyPolicy(
            (UNIFORM_GRID, PERMUTED_UNIFORM_GRID, STRIP_FALLBACK, POINT_FALLBACK),
            relation_kind=relation_kind,
        )

    if method == "stratified" and similar:
        return StrategyPolicy(
            (UNIFORM_GRID, PERMUTED_UNIFORM_GRID, GUILLOTINE, STRIP_FALLBACK, POINT_FALLBACK),
            relation_kind=relation_kind,
        )

    if similar:
        return StrategyPolicy(
            (UNIFORM_GRID, PERMUTED_UNIFORM_GRID, GUILLOTINE, STRIP_FALLBACK, POINT_FALLBACK),
            relation_kind=relation_kind,
        )

    return StrategyPolicy(
        (SHELF, WEIGHTED_GRID, GUILLOTINE, UNIFORM_GRID, STRIP_FALLBACK, POINT_FALLBACK), relation_kind=relation_kind
    )


def without_point_fallback(policy: StrategyPolicy) -> StrategyPolicy:
    strategies = tuple(strategy for strategy in policy.strategies if strategy != POINT_FALLBACK)
    return StrategyPolicy(strategies or policy.strategies, relation_kind=policy.relation_kind)
