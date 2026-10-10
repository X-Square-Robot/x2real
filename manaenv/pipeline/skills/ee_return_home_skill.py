"""Cartesian home skill that returns one arm to its episode-start EE pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .base import Skill, SkillCfg, SkillTarget, register_skill


@dataclass
class EeReturnHomeSkillCfg(SkillCfg):
    """Return the selected arm to the pose captured at episode start."""

    skill_type: str = "ee_return_home"
    arm: str = "current"
    gripper: float = 0.0
    # Optional cuRobo slowdown for the home segment (smaller => slower joint motion).
    time_dilation_factor: float | None = None


class EeReturnHomeSkill(Skill):
    """Emit one EE target that restores the arm to its episode-start pose."""

    Cfg = EeReturnHomeSkillCfg

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        current_ee_pos = obs.get("current_ee_pos")
        current_ee_quat = obs.get("current_ee_quat")
        if current_ee_pos is None or current_ee_quat is None:
            raise ValueError("EeReturnHomeSkill.observe requires current_ee_pos/current_ee_quat in planner context.")
        return {
            "env_idx": env_idx,
            "current_ee_pos": torch.as_tensor(current_ee_pos, dtype=torch.float32).reshape(3),
            "current_ee_quat": torch.as_tensor(current_ee_quat, dtype=torch.float32).reshape(4),
            "active_arm": obs.get("active_arm"),
            "episode_start_ee": obs.get("episode_start_ee"),
            "gym_env": env,
        }

    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        arm = self._resolve_arm(observation)
        episode_start_ee = observation.get("episode_start_ee")
        if not isinstance(episode_start_ee, dict):
            raise ValueError("EeReturnHomeSkill requires episode_start_ee in the planner observation.")

        start_pose = episode_start_ee.get(arm)
        if start_pose is None:
            raise ValueError(f"EeReturnHomeSkill missing episode-start pose for arm={arm!r}.")

        device = observation["current_ee_pos"].device
        return [
            self.build_return_home_target(
                start_pose,
                arm=arm,
                gripper=float(self.cfg.gripper),
                time_dilation_factor=getattr(self.cfg, "time_dilation_factor", None),
                device=device,
            )
        ]


register_skill("ee_return_home", EeReturnHomeSkill)

__all__ = ["EeReturnHomeSkill", "EeReturnHomeSkillCfg"]
