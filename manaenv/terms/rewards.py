"""Reward builder functions for declarative RewardsComponent.

Each function returns a ``RewardTermCfg`` with the appropriate ``func``
and ``params`` / ``weight``.  These are collected by the component and
wrapped into a dynamic configclass for ``combine_configclass_instances``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.managers import RewardTermCfg


def reach_target(func: Any = None, weight: float = 1.0, **kwargs) -> RewardTermCfg:
    """Reward for reaching towards a target object."""
    from isaaclab.managers import RewardTermCfg

    return RewardTermCfg(func=func, weight=weight, params=kwargs if kwargs else {})


def grasp_success(func: Any = None, weight: float = 1.0, **kwargs) -> RewardTermCfg:
    """Reward for successfully grasping an object."""
    from isaaclab.managers import RewardTermCfg

    return RewardTermCfg(func=func, weight=weight, params=kwargs if kwargs else {})


def task_completion(func: Any = None, weight: float = 10.0, **kwargs) -> RewardTermCfg:
    """Sparse reward for completing the full task."""
    from isaaclab.managers import RewardTermCfg

    return RewardTermCfg(func=func, weight=weight, params=kwargs if kwargs else {})
