"""Per-episode grasp-target selection — TASK/SCENE authority, sim-free and pure.

Ownership (spacepod refactor): choosing WHICH drink to grasp each episode belongs
to the task/scene, NOT the data-collection planner. This module is the pure
picker; :func:`manaenv.terms.events.select_episode_grasp_target` is the reset
event that applies it and pins the choice into an env store both the success
checker and the (obeying) planner read.

Pure Python (no torch / IsaacSim) → directly unit-testable.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

# Env attribute holding the per-env pinned grasp target name (one per episode).
EPISODE_GRASP_TARGET_ATTR = "_mana_episode_grasp_target"


def normalize_pool(pool: Any) -> list[str]:
    """Coerce a role value (str | list | None) into a clean list of asset names."""
    if isinstance(pool, str):
        name = pool.strip()
        return [name] if name else []
    if isinstance(pool, (list, tuple)):
        names = [str(item).strip() for item in pool if str(item).strip()]
        # de-dup, order-preserving
        seen: set[str] = set()
        out: list[str] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out
    return []


def select_episode_target(
    pool: Sequence[str] | str,
    *,
    rng: random.Random | None = None,
    exclude: str | None = None,
) -> str | None:
    """Pick one target name from *pool*, avoiding *exclude* when alternatives exist.

    Deterministic when *rng* is supplied. Returns ``None`` for an empty pool.
    Mirrors the planner's former ``select_random_grasp_target_name`` semantics so
    moving selection to the task layer is behavior-preserving.
    """
    names = normalize_pool(pool)
    if exclude and len(names) > 1:
        names = [name for name in names if name != exclude]
    if not names:
        return None
    return (rng or random).choice(names)


def read_pinned_target(env: Any, env_idx: int) -> str | None:
    """Return the task-pinned grasp target for *env_idx*, or ``None`` if unpinned.

    The pin (``env.<EPISODE_GRASP_TARGET_ATTR>`` = ``{env_id: name}``) is written
    by the task/scene selection event. This is the read side the (obeying)
    collection planner uses so it executes the bottle the TASK chose instead of
    computing its own — the same pin the reach success checker narrows to.
    """
    store = getattr(env, EPISODE_GRASP_TARGET_ATTR, None)
    if not isinstance(store, dict):
        return None
    value = store.get(int(env_idx))
    if isinstance(value, str) and value.strip():
        return value
    return None


def resolve_episode_target(
    env: Any,
    env_idx: int,
    pool: Sequence[str] | str,
    *,
    rng: random.Random | None = None,
    exclude: str | None = None,
) -> str | None:
    """Resolve this episode's grasp target — OBEY the task pin, else fall back.

    Authority order: if the task pinned a target for *env_idx* (and it is in the
    candidate *pool*), use it verbatim — the collection planner must not override
    the task's choice. Otherwise fall back to :func:`select_episode_target`
    (legacy random pick) so unpinned runs behave exactly as before. This is the
    single seam that lets the planner stop computing the target and start
    obeying it, while staying backward-compatible.
    """
    pinned = read_pinned_target(env, env_idx)
    if pinned is not None:
        names = normalize_pool(pool)
        # Honor the pin when it is a real candidate; if the pool is empty we
        # still trust the pin (the task chose it), else require membership so a
        # stale pin can't point outside the live candidate set.
        if not names or pinned in names:
            return pinned
    return select_episode_target(pool, rng=rng, exclude=exclude)
