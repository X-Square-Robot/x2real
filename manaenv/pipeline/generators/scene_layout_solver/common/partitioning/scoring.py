from __future__ import annotations

import math

from .geometry import case_signature, range_gap
from .models import PartitionCase, PartitionProblem
from .policy import POINT_FALLBACK, STRIP_FALLBACK


def score_cases(problem: PartitionProblem, cases: list[PartitionCase]) -> list[PartitionCase]:
    return [case.with_score(*_score_case(problem, case)) for case in cases]


def select_cases(problem: PartitionProblem, cases: list[PartitionCase], *, max_count: int) -> list[PartitionCase]:
    usable_cases = _prefer_primary_cases(cases, max_count=max_count)
    scored = sorted(score_cases(problem, usable_cases), key=lambda item: (-item.quality, _case_sort_key(item)))
    if len(scored) <= max_count:
        return scored
    selected = [scored[0]]
    remaining = scored[1:]
    best_quality = max(scored[0].quality, 1e-9)
    quality_floor = best_quality * 0.35 if best_quality > 0.0 else -float("inf")
    preferred = [case for case in remaining if case.quality >= quality_floor]
    fallback = [case for case in remaining if case.quality < quality_floor]
    while preferred and len(selected) < max_count:
        best = min(
            preferred,
            key=lambda item: (
                -(item.quality + 0.01 * _normalized_distance(problem, item, selected)),
                _case_sort_key(item),
            ),
        )
        selected.append(best)
        preferred.remove(best)
    while fallback and len(selected) < max_count:
        best = min(
            fallback,
            key=lambda item: (
                -(item.quality + 0.005 * _normalized_distance(problem, item, selected)),
                _case_sort_key(item),
            ),
        )
        selected.append(best)
        fallback.remove(best)
    return selected


def _prefer_primary_cases(cases: list[PartitionCase], *, max_count: int) -> list[PartitionCase]:
    primary = [case for case in cases if case.tier not in {STRIP_FALLBACK, POINT_FALLBACK}]
    if len(primary) >= max(1, int(max_count)):
        return primary
    return cases


def tier_counts(cases: list[PartitionCase]) -> dict[str, int]:
    out: dict[str, int] = {}
    for case in cases:
        out[case.tier] = out.get(case.tier, 0) + 1
    return out


def _score_case(problem: PartitionProblem, case: PartitionCase) -> tuple[float, dict[str, float]]:
    cells = list(case.cells.values())
    total_area = sum(cell.area for cell in cells)
    min_area = min((cell.area for cell in cells), default=0.0)
    min_short_edge = min((cell.short_edge for cell in cells), default=0.0)
    mean_aspect = sum(_aspect_score(cell.x.length, cell.y.length) for cell in cells) / max(1, len(cells))
    min_gap = _min_pair_gap(case)
    obstacle_gap = _min_obstacle_gap(problem, case)
    fallback_penalty = 0.0
    if case.tier == STRIP_FALLBACK:
        fallback_penalty = -0.02
    elif case.tier == POINT_FALLBACK:
        fallback_penalty = -0.08
    quality = (
        total_area
        + min_area * 4.0
        + min_short_edge * 0.35
        + mean_aspect * 0.015
        + min_gap * 0.005
        + obstacle_gap * 0.005
        + fallback_penalty
    )
    return quality, {
        "total_cell_area": float(total_area),
        "min_cell_area": float(min_area),
        "min_short_edge": float(min_short_edge),
        "mean_aspect_score": float(mean_aspect),
        "min_pair_gap": float(min_gap),
        "min_obstacle_gap": float(obstacle_gap),
    }


def _aspect_score(x_len: float, y_len: float) -> float:
    x = max(0.0, float(x_len))
    y = max(0.0, float(y_len))
    if x <= 1e-12 or y <= 1e-12:
        return 0.0
    return min(x / y, y / x)


def _min_pair_gap(case: PartitionCase) -> float:
    names = sorted(case.cells)
    gaps: list[float] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            gaps.append(
                max(
                    range_gap(case.cells[left].x, case.cells[right].x),
                    range_gap(case.cells[left].y, case.cells[right].y),
                )
            )
    return min(gaps) if gaps else 0.0


def _min_obstacle_gap(problem: PartitionProblem, case: PartitionCase) -> float:
    if not problem.obstacles:
        return 0.0
    gaps: list[float] = []
    for cell in case.cells.values():
        profile = problem.profiles[cell.object_name]
        swept_x = profile.swept_x(cell)
        swept_y = profile.swept_y(cell)
        for obstacle in problem.obstacles:
            gaps.append(max(range_gap(swept_x, obstacle.x), range_gap(swept_y, obstacle.y)))
    return min(gaps) if gaps else 0.0


def _normalized_distance(problem: PartitionProblem, candidate: PartitionCase, selected: list[PartitionCase]) -> float:
    diag = math.sqrt(
        problem.bounds.x.length * problem.bounds.x.length + problem.bounds.y.length * problem.bounds.y.length
    )
    if diag <= 1e-9:
        return 0.0
    return min(_case_distance(candidate, item) / diag for item in selected)


def _case_distance(a: PartitionCase, b: PartitionCase) -> float:
    names = sorted(set(a.cells) & set(b.cells))
    if not names:
        return 0.0
    total = 0.0
    for name in names:
        ax, ay = a.cells[name].center
        bx, by = b.cells[name].center
        total += math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    topology_bonus = 0.05 if a.topology != b.topology else 0.0
    return total / float(len(names)) + topology_bonus


def _case_sort_key(case: PartitionCase) -> tuple:
    return (case.tier, case.topology, case_signature(case.cells))
