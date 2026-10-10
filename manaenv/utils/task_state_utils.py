"""Helpers for clearing task-local runtime state on environments."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

_TASK_STATE_BASELINE_ATTR = "_mana_task_state_baseline_attrs"


def _unwrap_env(env: Any) -> Any:
    """Return the deepest available unwrapped environment object."""
    current = env
    while True:
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is None or unwrapped is current:
            return current
        current = unwrapped


def _get_or_init_task_state_baseline(env: Any) -> tuple[Any, frozenset[str]]:
    """Capture environment baseline attrs once and reuse it on later clears.

    The first call defines the contract: invoke it only after all persistent
    task/env attributes are initialized. Attributes injected after that point
    are treated as task-local runtime state and may be cleared by
    :func:`clear_task_state`.
    """
    unwrapped = _unwrap_env(env)
    baseline = getattr(unwrapped, _TASK_STATE_BASELINE_ATTR, None)
    if baseline is None:
        baseline = frozenset(vars(unwrapped).keys())
        setattr(unwrapped, _TASK_STATE_BASELINE_ATTR, baseline)
    elif not isinstance(baseline, frozenset):
        baseline = frozenset(baseline)
        setattr(unwrapped, _TASK_STATE_BASELINE_ATTR, baseline)
    return unwrapped, baseline


def iter_task_state_items(env: Any) -> Iterator[tuple[str, Any]]:
    """Yield runtime attrs that appeared after the environment baseline."""
    unwrapped, baseline = _get_or_init_task_state_baseline(env)
    for name, value in vars(unwrapped).items():
        if name == _TASK_STATE_BASELINE_ATTR:
            continue
        if name in baseline:
            continue
        if not name.startswith("_"):
            continue
        yield name, value


def _clear_task_state_value(value: Any) -> bool:
    """Reset one task-state value in place when its type supports it."""
    zero_fn = getattr(value, "zero_", None)
    if callable(zero_fn):
        zero_fn()
        return True

    fill_fn = getattr(value, "fill_", None)
    if callable(fill_fn):
        fill_fn(0)
        return True

    if isinstance(value, (dict, list, set)):
        value.clear()
        return True

    return False


def clear_task_state(env: Any) -> tuple[str, ...]:
    """Clear task-local runtime state and return cleared attribute names."""
    cleared_attrs: list[str] = []
    for name, value in iter_task_state_items(env):
        if _clear_task_state_value(value):
            cleared_attrs.append(name)
    return tuple(cleared_attrs)
