from __future__ import annotations

from ..reproducibility import stable_rng
from .geometry import case_signature
from .models import PartitionCase, PartitionProblem, StrategyPolicy
from .policy import POINT_FALLBACK, STRIP_FALLBACK
from .strategies import generate_for_strategy


def generate_partition_cases(
    problem: PartitionProblem,
    policy: StrategyPolicy,
    *,
    seed: int,
    candidate_count: int,
) -> list[PartitionCase]:
    out: list[PartitionCase] = []
    signatures: set[tuple[tuple[str, float, float, float, float], ...]] = set()
    target = max(1, int(candidate_count))
    strategy_count = max(1, len(policy.strategies))
    for index, strategy in enumerate(policy.strategies):
        if strategy == POINT_FALLBACK and len(out) >= _fallback_threshold(problem, target):
            continue
        base_budget = max(problem.count * 2, target // strategy_count)
        budget = (
            base_budget if strategy not in {STRIP_FALLBACK, POINT_FALLBACK} else max(problem.count, base_budget // 2)
        )
        generated = generate_for_strategy(
            problem,
            strategy,
            rng=stable_rng(seed, "partition_strategy", index, strategy, payload=_problem_seed_payload(problem)),
            budget=budget,
        )
        _append_unique(out, signatures, generated)
    return out


def _append_unique(
    out: list[PartitionCase],
    signatures: set[tuple[tuple[str, float, float, float, float], ...]],
    incoming: list[PartitionCase],
) -> None:
    for case in incoming:
        signature = case_signature(case.cells)
        if signature in signatures:
            continue
        signatures.add(signature)
        out.append(case)


def _fallback_threshold(problem: PartitionProblem, target: int) -> int:
    return max(1, int(target))


def _problem_seed_payload(problem: PartitionProblem) -> dict:
    return {
        "names": list(problem.names),
        "bounds": [
            round(problem.bounds.x.min, 8),
            round(problem.bounds.x.max, 8),
            round(problem.bounds.y.min, 8),
            round(problem.bounds.y.max, 8),
        ],
        "profiles": {
            name: {
                "footprint": [round(value, 8) for value in problem.profiles[name].footprint],
                "footprint_bounds": [
                    round(problem.profiles[name].footprint_bounds.x.min, 8),
                    round(problem.profiles[name].footprint_bounds.x.max, 8),
                    round(problem.profiles[name].footprint_bounds.y.min, 8),
                    round(problem.profiles[name].footprint_bounds.y.max, 8),
                ],
            }
            for name in sorted(problem.profiles)
        },
        "min_spacing": round(problem.min_spacing, 8),
        "spacing_mode": problem.spacing_mode,
        "sample_method": problem.sample_method,
        "grid_step": None if problem.grid_step is None else round(float(problem.grid_step), 8),
        "grid_snap": dict(problem.grid_snap or {}),
        "align_axis": problem.align_axis,
        "order_axis": problem.order_axis,
        "boundary_spacing": None if problem.boundary_spacing is None else round(float(problem.boundary_spacing), 8),
        "obstacle_count": len(problem.obstacles),
    }
