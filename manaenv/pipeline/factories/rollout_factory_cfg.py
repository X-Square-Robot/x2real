"""Typed public evaluation and manual-collection rollout orders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.planners.planner_base import PlannerCFG

__all__ = ["MANUAL_PLANNERS", "RolloutOrderCFG", "RolloutOrdersCFG", "select_rollout_orders"]

MANUAL_PLANNERS = frozenset({"KeyboardTeleopPlanner", "VRTeleopPlanner", "MasterSlaveTeleopPlanner"})


@dataclass
class RolloutOrderCFG:
    """Run one policy or manual-input planner against an assembled environment."""

    env_ref: str | list[str] = ""
    planner: PlannerCFG | None = None
    num_episodes: int | None = None
    num_demos: int | None = None
    max_steps: int | None = None
    episode_length_s: float | None = None
    num_parallel_envs: int = 1
    show_evaluation_overlay: bool = False
    stage_progress_log_interval: int = 0
    enable_hotkeys: bool = False
    resume: bool = False
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutOrdersCFG:
    """Typed envelope for the top-level ``rollout`` section."""

    orders: list[RolloutOrderCFG] = field(default_factory=list)


def select_rollout_orders(orders: list[RolloutOrderCFG], *, mode: str | None) -> list[RolloutOrderCFG]:
    """Select manual gendata or policy benchmark orders; None preserves mixed suites."""
    if mode is None:
        return list(orders)
    if mode not in {"gendata", "benchmark"}:
        raise ValueError(f"Unsupported public rollout mode: {mode!r}")
    allowed = MANUAL_PLANNERS if mode == "gendata" else {"PolicyPlanner"}
    return [order for order in orders if type(order.planner).__name__.removesuffix("CFG") in allowed]
