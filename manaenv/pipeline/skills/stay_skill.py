"""Joint-space stay skill for moving an arm to a configured standby posture."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from manaenv.utils.tensor_utils import as_torch

from .base import SkillTarget, register_skill
from .home_skill import HomeSkill, HomeSkillCfg


@dataclass
class StaySkillCfg(HomeSkillCfg):
    """Configuration for moving one arm to a named stay joint posture."""

    skill_type: str = "stay"
    stay_joints: list[float] | None = None
    left_stay_joints: list[float] | None = None
    right_stay_joints: list[float] | None = None


class StaySkill(HomeSkill):
    """Emit joint-space targets that interpolate the active arm to stay."""

    Cfg = StaySkillCfg

    def plan_targets(self, observation: dict, evaluator: object | None) -> list[SkillTarget]:
        arm = self._resolve_arm(observation)
        if arm == "both":
            return self._plan_both_arm_targets(observation)
        current_q = self._current_arm_joints(observation, arm)
        stay_joints = self._resolve_stay_joints(arm)
        stay_q = torch.as_tensor(stay_joints, dtype=torch.float32, device=current_q.device).reshape(-1)
        if stay_q.numel() != current_q.numel():
            raise ValueError(f"stay_joints for {arm} must have {current_q.numel()} values, got {stay_q.numel()}.")

        steps = max(int(self.cfg.steps), 1)
        pos = observation["current_ee_pos"]
        quat = observation["current_ee_quat"]
        targets: list[SkillTarget] = []
        for idx in range(steps):
            ratio = float(idx + 1) / float(steps)
            joint_pos = stay_q * ratio + current_q * (1.0 - ratio)
            targets.append(
                SkillTarget(
                    pos=pos,
                    quat=quat,
                    gripper=float(self.cfg.gripper),
                    phase="stay",
                    arm=arm,
                    joint_pos=joint_pos,
                )
            )
        return targets

    def _plan_both_arm_targets(self, observation: dict) -> list[SkillTarget]:
        left_current = self._current_arm_joints(observation, "left")
        right_current = self._current_arm_joints(observation, "right")
        left_stay = torch.as_tensor(
            self._resolve_stay_joints("left"), dtype=torch.float32, device=left_current.device
        ).reshape(-1)
        right_stay = torch.as_tensor(
            self._resolve_stay_joints("right"), dtype=torch.float32, device=right_current.device
        ).reshape(-1)
        if left_stay.numel() != left_current.numel():
            raise ValueError(f"left_stay_joints must have {left_current.numel()} values, got {left_stay.numel()}.")
        if right_stay.numel() != right_current.numel():
            raise ValueError(f"right_stay_joints must have {right_current.numel()} values, got {right_stay.numel()}.")

        steps = max(int(self.cfg.steps), 1)
        pos = observation["current_ee_pos"]
        quat = observation["current_ee_quat"]
        targets: list[SkillTarget] = []
        for idx in range(steps):
            ratio = float(idx + 1) / float(steps)
            left_joint = left_stay * ratio + left_current * (1.0 - ratio)
            right_joint = right_stay * ratio + right_current * (1.0 - ratio)
            targets.append(
                SkillTarget(
                    pos=pos,
                    quat=quat,
                    gripper=float(self.cfg.gripper),
                    phase="stay",
                    arm="both",
                    joint_pos=torch.cat((left_joint, right_joint), dim=0),
                )
            )
        return targets

    def _current_arm_joints(self, observation: dict, arm: str) -> torch.Tensor:
        robot = observation["robot"]
        env_idx = int(observation["env_idx"])
        data = getattr(robot, "data", None)
        joint_names = list(getattr(data, "joint_names", []) or [])
        joint_pos = as_torch(getattr(data, "joint_pos", None))
        indices = self._arm_joint_indices(joint_names, arm)
        if not indices or not isinstance(joint_pos, torch.Tensor):
            raise RuntimeError(f"StaySkill could not resolve {arm} arm joints from robot.data.")
        return joint_pos[env_idx, indices].detach().clone().to(dtype=torch.float32)

    def _resolve_stay_joints(self, arm: str) -> list[float]:
        if self.cfg.stay_joints is not None:
            return self.cfg.stay_joints
        if arm == "left" and self.cfg.left_stay_joints is not None:
            return self.cfg.left_stay_joints
        if arm == "right" and self.cfg.right_stay_joints is not None:
            return self.cfg.right_stay_joints
        raise ValueError(f"StaySkill requires stay_joints or {arm}_stay_joints for arm={arm}.")


register_skill("stay", StaySkill)

__all__ = ["StaySkill", "StaySkillCfg"]
