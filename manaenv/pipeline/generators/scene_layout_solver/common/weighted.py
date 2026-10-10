from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from math import isfinite


class WeightError(ValueError):
    pass


def parse_nonnegative_weight(raw: object, *, default: float = 1.0) -> float:
    if raw is None:
        return float(default)
    try:
        weight = float(raw)
    except (TypeError, ValueError) as exc:
        raise WeightError("weight must be a finite non-negative number") from exc
    if not isfinite(weight) or weight < 0.0:
        raise WeightError("weight must be a finite non-negative number")
    return weight


def positive_weight_items[T](items: Iterable[T], *, weight_fn: Callable[[T], float]) -> list[T]:
    return [item for item in items if float(weight_fn(item)) > 0.0]


def weighted_choice[T](items: Iterable[T], *, rng: random.Random, weight_fn: Callable[[T], float]) -> T | None:
    candidates = positive_weight_items(items, weight_fn=weight_fn)
    total = sum(float(weight_fn(item)) for item in candidates)
    if total <= 0.0:
        return None
    threshold = rng.random() * total
    cumulative = 0.0
    for item in candidates:
        cumulative += float(weight_fn(item))
        if cumulative >= threshold:
            return item
    return candidates[-1]


def weighted_order[T](items: Iterable[T], *, rng: random.Random, weight_fn: Callable[[T], float]) -> list[T]:
    remaining = positive_weight_items(items, weight_fn=weight_fn)
    ordered: list[T] = []
    while remaining:
        selected = weighted_choice(remaining, rng=rng, weight_fn=weight_fn)
        if selected is None:
            break
        ordered.append(selected)
        remaining.remove(selected)
    return ordered
