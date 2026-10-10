from __future__ import annotations

import random

from ...constants import GEOM_EPS
from ...geometry import Bounds2D, Range1D
from ..models import PartitionCase, PartitionProblem
from ..policy import FREE_RECT
from .common import cells_from_named_raw
from .guillotine import guillotine_raw


def free_rect_cases(problem: PartitionProblem, *, rng: random.Random, budget: int) -> list[PartitionCase]:
    if not problem.obstacles:
        return []
    rects = free_rectangles(problem)
    if not rects:
        return []
    out: list[PartitionCase] = []
    attempts = 0
    max_attempts = max(64, budget * 16)
    while len(out) < budget and attempts < max_attempts:
        attempts += 1
        names = sorted(problem.names, key=lambda name: (-problem.profiles[name].area, name))
        if attempts % 3:
            rng.shuffle(names)
        groups = _assign_to_free_rects(problem, names, rects, rng=rng)
        if groups is None:
            continue
        raw: dict[str, Bounds2D] = {}
        for rect_index, group in groups.items():
            rect = rects[rect_index]
            if len(group) == 1:
                raw[group[0]] = rect
                continue
            child_raw = guillotine_raw(problem, list(group), rect, rng=rng, depth=0)
            if child_raw is None:
                raw = {}
                break
            raw.update(child_raw)
        if not raw:
            continue
        cells = cells_from_named_raw(problem, raw)
        if cells is None:
            continue
        out.append(PartitionCase(cells=cells, tier=FREE_RECT, topology=f"free_rect_{len(rects)}"))
    return out


def free_rectangles(problem: PartitionProblem) -> list[Bounds2D]:
    rects = [problem.bounds]
    for obstacle in problem.obstacles:
        blocked = Bounds2D(obstacle.x.expanded(problem.min_spacing), obstacle.y.expanded(problem.min_spacing))
        next_rects: list[Bounds2D] = []
        for rect in rects:
            next_rects.extend(_subtract_rect(rect, blocked))
        rects = next_rects
        if not rects:
            return []
    return sorted(
        (rect for rect in rects if rect.x.length > GEOM_EPS and rect.y.length > GEOM_EPS),
        key=lambda item: item.x.length * item.y.length,
        reverse=True,
    )


def _subtract_rect(rect: Bounds2D, blocked: Bounds2D) -> list[Bounds2D]:
    overlap = rect.intersect(blocked)
    if overlap is None:
        return [rect]
    out: list[Bounds2D] = []
    if rect.x.min < overlap.x.min - GEOM_EPS:
        out.append(Bounds2D(Range1D(rect.x.min, overlap.x.min), rect.y))
    if overlap.x.max < rect.x.max - GEOM_EPS:
        out.append(Bounds2D(Range1D(overlap.x.max, rect.x.max), rect.y))
    middle_x = Range1D(overlap.x.min, overlap.x.max)
    if rect.y.min < overlap.y.min - GEOM_EPS:
        out.append(Bounds2D(middle_x, Range1D(rect.y.min, overlap.y.min)))
    if overlap.y.max < rect.y.max - GEOM_EPS:
        out.append(Bounds2D(middle_x, Range1D(overlap.y.max, rect.y.max)))
    return out


def _assign_to_free_rects(
    problem: PartitionProblem,
    names: list[str],
    rects: list[Bounds2D],
    *,
    rng: random.Random,
) -> dict[int, list[str]] | None:
    groups: dict[int, list[str]] = {}
    loads = [0.0 for _ in rects]
    for name in names:
        feasible = [idx for idx, rect in enumerate(rects) if _object_can_use_rect(problem, name, rect)]
        if not feasible:
            return None
        rng.shuffle(feasible)
        chosen = min(
            feasible,
            key=lambda idx: (
                loads[idx] / max(rects[idx].x.length * rects[idx].y.length, GEOM_EPS),
                len(groups.get(idx, [])),
            ),
        )
        groups.setdefault(chosen, []).append(name)
        loads[chosen] += problem.profiles[name].area + problem.min_spacing * max(problem.profiles[name].footprint)
    return groups


def _object_can_use_rect(problem: PartitionProblem, name: str, rect: Bounds2D) -> bool:
    profile = problem.profiles[name]
    return (
        rect.x.length + GEOM_EPS >= profile.footprint[0] + problem.min_spacing
        and rect.y.length + GEOM_EPS >= profile.footprint[1] + problem.min_spacing
    )
