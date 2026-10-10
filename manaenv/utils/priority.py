"""Priority-based sorting for pipeline objects (CFGs, generators, etc.)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def sort_by_priority(items: list[Any], key: Callable[[Any], int] | None = None) -> list[Any]:
    """Sort any objects that carry a ``priority`` attribute (lower first).

    Works on ComponentCFG, GeneratorCFG, or anything else with a
    numeric ``priority``.  Items without the attribute default to 2.

    Args:
        items: Objects to sort.
        key: Optional callable to extract the priority value from each
            item.  Defaults to ``lambda x: getattr(x, "priority", 2)``.
    """
    if key is None:
        key = lambda x: getattr(x, "priority", 2)  # noqa: E731
    return sorted(items, key=key)
