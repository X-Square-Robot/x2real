"""Typed configuration for evaluation-only rollout orders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.planners.planner_base import PlannerCFG

__all__ = ["RolloutOrderCFG", "RolloutOrdersCFG"]


@dataclass
class RolloutOrderCFG:
    """Evaluate one policy planner against one assembled environment."""

    env_ref: str | list[str] = ""
    planner: PlannerCFG | None = None
    num_episodes: int | None = None
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
