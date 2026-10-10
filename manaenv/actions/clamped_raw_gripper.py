"""Clamped raw gripper joint targets for raw_joint policy mode (no state machine)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

# lazy_loader shadows isaaclab.utils.configclass with the submodule once isaaclab.envs loads; take the function directly.
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.managers.action_manager import ActionTerm


class ClampedRawGripperAction(joint_actions.JointPositionAction):
    """Absolute joint targets with optional clamp bounds."""

    cfg: ClampedRawGripperActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        lo = self.cfg.clamp_min
        hi = self.cfg.clamp_max
        if lo is None and hi is None:
            return
        min_val = float(lo) if lo is not None else None
        max_val = float(hi) if hi is not None else None
        self._processed_actions = torch.clamp(self._processed_actions, min=min_val, max=max_val)


@configclass
class ClampedRawGripperActionCfg(JointPositionActionCfg):
    """Config for :class:`ClampedRawGripperAction`.

    Clamp bounds are optional and should be specified by each robot embodiment.
    """

    class_type: type[ActionTerm] = ClampedRawGripperAction

    clamp_min: float | None = None
    clamp_max: float | None = None
