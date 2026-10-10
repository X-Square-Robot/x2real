from __future__ import annotations

from typing import Any


def relation_axis_and_order(rel: Any) -> tuple[str, str, str] | None:
    """Return the support-local axis and ordered node pair for a direction relation."""

    direction = str(getattr(rel, "direction", "") or "")
    node_1 = str(getattr(rel, "node_1", "") or "")
    node_2 = str(getattr(rel, "node_2", "") or "")
    if direction == "left_of":
        return ("x", node_1, node_2)
    if direction == "right_of":
        return ("x", node_2, node_1)
    if direction == "back_of":
        return ("y", node_1, node_2)
    if direction == "front_of":
        return ("y", node_2, node_1)
    return None
