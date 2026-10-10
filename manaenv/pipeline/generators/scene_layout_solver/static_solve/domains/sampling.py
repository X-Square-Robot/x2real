from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from ...common.geometry import Bounds2D, Pose, Range1D


@dataclass(frozen=True)
class CandidatePlacement:
    object_name: str
    surface: str
    x: float
    y: float
    pose: Pose
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateDomain:
    object_name: str
    surface: str | None
    candidates: tuple[CandidatePlacement, ...] = ()
    bounds: Bounds2D | None = None
    step: float | None = None
    issues: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.candidates


@dataclass(frozen=True)
class SupportPointSample:
    points: tuple[tuple[float, float], ...]
    step: float
    meta: dict


def adaptive_support_step(bounds: Bounds2D, *, footprint: tuple[float, float]) -> float:
    default_step = 0.05
    spans = [value for value in (bounds.x.length, bounds.y.length) if value > 1e-9]
    if not spans:
        return 0.01
    min_span = min(spans)
    needs_refinement = min_span < 0.4
    if not needs_refinement:
        return default_step
    candidates = [default_step, min_span / 8.0]
    child_extents = [value for value in footprint if value > 1e-9]
    if child_extents:
        candidates.append(min(child_extents) / 3.0)
    return max(0.01, min(value for value in candidates if value > 1e-9))


def axis_values(axis: Range1D, step: float, *, max_values: int = 80) -> list[float]:
    if axis.length <= 1e-9:
        return [axis.center]
    count = min(max_values - 1, max(1, int(math.ceil(axis.length / max(step, 1e-9)))))
    values = [axis.min + axis.length * i / count for i in range(count + 1)]
    center = axis.center
    if all(abs(value - center) > 1e-9 for value in values):
        values.append(center)
    return sorted(values)


def sample_support_points(
    bounds: Bounds2D,
    *,
    footprint: tuple[float, float],
    accept_fn: Callable[[float, float], bool],
    max_axis_values: int = 80,
    target_min_candidates: int = 32,
    low_acceptance_ratio: float = 0.15,
    refill_budget: int = 1024,
) -> SupportPointSample:
    step = adaptive_support_step(bounds, footprint=footprint)
    points: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()

    def visit_grid(
        grid_step: float,
        *,
        grid_max_axis_values: int,
        budget: int | None = None,
        target_count: int | None = None,
    ) -> tuple[int, int]:
        raw = 0
        accepted = 0
        xs = axis_values(bounds.x, grid_step, max_values=grid_max_axis_values)
        ys = axis_values(bounds.y, grid_step, max_values=grid_max_axis_values)
        for x in xs:
            for y in ys:
                key = (round(float(x), 9), round(float(y), 9))
                if key in seen:
                    continue
                if budget is not None and raw >= budget:
                    return raw, accepted
                seen.add(key)
                raw += 1
                if not accept_fn(float(x), float(y)):
                    continue
                points.append((float(x), float(y)))
                accepted += 1
                if target_count is not None and len(points) >= target_count:
                    return raw, accepted
        return raw, accepted

    base_raw, base_accepted = visit_grid(step, grid_max_axis_values=max_axis_values)
    base_ratio = (base_accepted / base_raw) if base_raw else 0.0
    refill_triggered = base_accepted < target_min_candidates or (base_raw > 0 and base_ratio < low_acceptance_ratio)
    refill_raw = 0
    refill_accepted = 0
    refill_step = None

    if refill_triggered and refill_budget > 0:
        refill_step = max(0.005, step * 0.5)
        refill_target = max(
            target_min_candidates,
            len(points) + target_min_candidates if base_ratio < low_acceptance_ratio else target_min_candidates,
        )
        refill_raw, refill_accepted = visit_grid(
            refill_step,
            grid_max_axis_values=max(max_axis_values, min(max_axis_values * 2, 128)),
            budget=refill_budget,
            target_count=refill_target,
        )

    raw_total = base_raw + refill_raw
    accepted_total = len(points)
    rejected_total = max(0, raw_total - accepted_total)
    return SupportPointSample(
        points=tuple(points),
        step=step,
        meta={
            "raw_candidate_count": raw_total,
            "accepted_candidate_count": accepted_total,
            "rejected_by_surface_boundary": rejected_total,
            "acceptance_ratio": (accepted_total / raw_total) if raw_total else 0.0,
            "base_raw_candidate_count": base_raw,
            "base_accepted_candidate_count": base_accepted,
            "base_acceptance_ratio": base_ratio,
            "refill_triggered": refill_triggered,
            "refill_attempt_count": refill_raw,
            "refill_accepted_count": refill_accepted,
            "refill_step": refill_step,
            "target_min_candidates": target_min_candidates,
        },
    )
