"""Minimal planner contract for batched policy evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from manaenv.pipeline.registry.global_provider import global_registry


@dataclass
class PlannerCFG:
    """Typed configuration shared by evaluation planners."""

    max_steps: int = 400
    step_hz: int = 30
    num_success_steps: int = 10
    class_type: type["PlannerBase"] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PlannerBase(ABC):
    """Batched actor interface consumed by the evaluation loop."""

    CFG: type[PlannerCFG] = PlannerCFG

    def __init__(self) -> None:
        self.cfg: PlannerCFG | None = None

    def load(self, cfg: PlannerCFG) -> None:
        self.cfg = cfg

    @abstractmethod
    def setup(self, env: Any, task: Any) -> bool:
        """Bind the assembled environment and task."""

    @abstractmethod
    def get_actions(self, env: Any, obs: dict[str, Any]) -> torch.Tensor:
        """Return one action row for every active environment."""

    def on_reset(
        self,
        env: Any,
        obs: dict[str, Any],
        env_ids: torch.Tensor,
        env_id_to_success: dict[int, bool] | None = None,
    ) -> None:
        """Reset planner state for the listed environments."""

    def initial_plan(self, env: Any, settle_steps: int = 30) -> None:
        """Perform optional first-episode initialization."""

    def cleanup(self) -> None:
        """Release planner-owned resources."""

    def requested_action_mode(self) -> str | None:
        """Return an optional embodiment action-mode override."""
        return None


def create_planner(cfg: PlannerCFG) -> PlannerBase:
    """Instantiate and load the planner selected by ``cfg.class_type``."""
    if cfg.class_type is None:
        raise ValueError("cfg.class_type required")
    planner = cfg.class_type()
    planner.load(cfg)
    return planner


def get_planner_class(name: str) -> type[PlannerBase] | None:
    """Resolve a planner implementation through the lazy registry."""
    return global_registry.get_impl(name)


__all__ = ["PlannerBase", "PlannerCFG", "create_planner", "get_planner_class"]
