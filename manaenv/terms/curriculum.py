"""Curriculum builder functions for declarative CurriculumComponent.

Each function returns a ``CurriculumTermCfg`` that the component
collects into a dynamic configclass for ``combine_configclass_instances``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.managers import CurriculumTermCfg


def modify_reward_weight(func: Any = None, **kwargs) -> CurriculumTermCfg:
    """Curriculum term that modifies a reward weight over training."""
    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import CurriculumTermCfg

    if func is None:
        func = mdp.modify_reward_weight
    return CurriculumTermCfg(func=func, params=kwargs if kwargs else {})
