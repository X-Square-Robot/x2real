"""Single-scalar gripper commands broadcast to multiple finger joints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.managers.action_manager import ActionTerm


class BroadcastGripperAction(joint_actions.JointPositionAction):
    """Map one raw gripper scalar to all resolved finger joints.

    Franka exposes two actuated finger joints but the skill planner uses the
    same 1-D gripper slot as EX001.  This action term keeps the external
    action dimension at one and expands the processed target before IsaacLab
    writes joint position targets.
    """

    cfg: BroadcastGripperActionCfg

    @property
    def action_dim(self) -> int:
        return int(self.cfg.input_action_dim)

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"{type(self).__name__} expected {self.action_dim} gripper action(s), "
                f"got shape {tuple(actions.shape)}."
            )

        self._raw_actions[:] = actions
        command = self._raw_actions * self._scale + self._offset

        lo = self.cfg.clamp_min
        hi = self.cfg.clamp_max
        if lo is not None or hi is not None:
            command = torch.clamp(
                command,
                min=float(lo) if lo is not None else None,
                max=float(hi) if hi is not None else None,
            )

        multipliers = self._joint_multipliers.to(command.device, dtype=command.dtype)
        offsets = self._joint_offsets.to(command.device, dtype=command.dtype)
        self._processed_actions = command * multipliers.unsqueeze(0) + offsets.unsqueeze(0)

    def __init__(self, cfg: BroadcastGripperActionCfg, env) -> None:
        super().__init__(cfg, env)
        if self.action_dim != 1:
            raise ValueError(f"{type(self).__name__} currently supports exactly one input scalar.")

        multipliers = tuple(float(v) for v in (cfg.joint_multipliers or (1.0,) * self._num_joints))
        offsets = tuple(float(v) for v in (cfg.joint_offsets or (0.0,) * self._num_joints))
        if len(multipliers) != self._num_joints or len(offsets) != self._num_joints:
            raise ValueError(
                f"{type(self).__name__} resolved {self._num_joints} joints but got "
                f"{len(multipliers)} multipliers and {len(offsets)} offsets."
            )
        self._joint_multipliers = torch.tensor(multipliers, device=self.device, dtype=torch.float32)
        self._joint_offsets = torch.tensor(offsets, device=self.device, dtype=torch.float32)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)


@configclass
class BroadcastGripperActionCfg(JointPositionActionCfg):
    """Config for :class:`BroadcastGripperAction`."""

    class_type: type[ActionTerm] = BroadcastGripperAction

    input_action_dim: int = 1
    joint_multipliers: tuple[float, ...] | None = None
    joint_offsets: tuple[float, ...] | None = None
    clamp_min: float | None = None
    clamp_max: float | None = None


__all__ = ["BroadcastGripperAction", "BroadcastGripperActionCfg"]
