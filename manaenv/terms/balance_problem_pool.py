"""Pure helpers for balance_scale problem enumeration and pool slicing."""

from __future__ import annotations

import itertools
from typing import Any


def enumerate_balance_problems(
    denoms: list[int],
    count_per_denom: int,
    add_count: int,
    final_total_cap: int,
    same_target_pan: bool = False,
    empty_target_denoms: bool = False,
) -> list[dict[str, Any]]:
    """Enumerate distinct solvable balance problems for a fixed add-count.

    A *final* balanced state assigns each denomination's copies to left/right/pile
    with equal left/right mass (> 0) and at most ``final_total_cap`` weights placed.
    A *problem* removes ``add_count`` placed weights (the robot must add them back)
    such that the remaining *initial* configuration is unbalanced. When
    ``same_target_pan`` is true, all removed weights must belong to one pan.
    ``empty_target_denoms`` excludes problems that preplace a target denomination
    on that pan, keeping denomination-count success checks unambiguous. Distinct
    problems are keyed by ``(left_counts, right_counts, sorted(add_targets))`` so
    a given add multiset is not double-counted across removal orders.

    Each problem dict holds:
      ``left``/``right``: per-denomination preplaced counts (tuples aligned to *denoms*).
      ``add``: sorted tuple of ``(denom_index, pan_code)`` (pan_code 0=left, 1=right).
    """
    n = int(count_per_denom)
    ndim = len(denoms)
    per_denom_opts = [[(nl, nr) for nl in range(n + 1) for nr in range(n + 1) if nl + nr <= n] for _ in denoms]

    problems: dict[tuple, dict[str, Any]] = {}
    for combo in itertools.product(*per_denom_opts):
        lsum = sum(d * lr[0] for d, lr in zip(denoms, combo))
        rsum = sum(d * lr[1] for d, lr in zip(denoms, combo))
        if lsum != rsum or lsum == 0:
            continue
        if sum(lr[0] + lr[1] for lr in combo) > final_total_cap:
            continue
        # Placed items as (denom_index, pan_code) with multiplicity.
        items: list[tuple[int, int]] = []
        for i, lr in enumerate(combo):
            items.extend([(i, 0)] * lr[0])
            items.extend([(i, 1)] * lr[1])
        if len(items) < add_count:
            continue
        for removed_pos in itertools.combinations(range(len(items)), add_count):
            removed = [items[p] for p in removed_pos]
            if same_target_pan and len({pan for _, pan in removed}) > 1:
                continue
            keep = [items[p] for p in range(len(items)) if p not in set(removed_pos)]
            left_counts = [0] * ndim
            right_counts = [0] * ndim
            for di, pan in keep:
                if pan == 0:
                    left_counts[di] += 1
                else:
                    right_counts[di] += 1
            pre_lsum = sum(denoms[i] * left_counts[i] for i in range(ndim))
            pre_rsum = sum(denoms[i] * right_counts[i] for i in range(ndim))
            if pre_lsum == pre_rsum:
                continue  # initial must be unbalanced
            if empty_target_denoms:
                if any((left_counts if pan == 0 else right_counts)[di] > 0 for di, pan in removed):
                    continue
            add_sorted = tuple(sorted(removed))
            key = (tuple(left_counts), tuple(right_counts), add_sorted)
            if key not in problems:
                problems[key] = {
                    "left": tuple(left_counts),
                    "right": tuple(right_counts),
                    "add": add_sorted,
                }
    return list(problems.values())


def balance_problem_slice(
    pool: list[dict[str, Any]],
    problem_limit: int | list[Any] | tuple[Any, ...] | None,
) -> list[dict[str, Any]]:
    """Restrict a deterministic problem pool with ``problem_limit``.

    Accepted forms:
      ``None`` — full pool
      ``int N`` — ``pool[:N]`` (compat)
      ``[start, stop]`` — half-open ``pool[start:stop]`` (``stop`` may be ``None``)
      ``[start]`` — ``pool[start:]``
    """
    if problem_limit is None:
        return pool
    if isinstance(problem_limit, bool):
        raise TypeError("sample_balance_problem problem_limit must be int, [start], or [start, stop]")
    if isinstance(problem_limit, int):
        if problem_limit <= 0:
            raise ValueError("sample_balance_problem problem_limit int must be positive")
        return pool[:problem_limit]
    if isinstance(problem_limit, (list, tuple)):
        if len(problem_limit) == 1:
            start, stop = int(problem_limit[0]), None
        elif len(problem_limit) == 2:
            start = int(problem_limit[0])
            stop = None if problem_limit[1] is None else int(problem_limit[1])
        else:
            raise ValueError("sample_balance_problem problem_limit slice must be [start] or [start, stop]")
        if start < 0 or (stop is not None and stop < start):
            raise ValueError(f"sample_balance_problem invalid problem_limit slice: {problem_limit!r}")
        return pool[start:stop]
    raise TypeError(
        "sample_balance_problem problem_limit must be int, [start], [start, stop], or None; "
        f"got {type(problem_limit).__name__}"
    )
