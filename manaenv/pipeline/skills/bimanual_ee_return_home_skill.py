"""Bimanual home skill: both arms return to episode-start EE poses together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .base import BimanualSkill, Skill, SkillCfg, SkillTarget, register_skill


@dataclass
class BimanualEeReturnHomeSkillCfg(SkillCfg):
    """Return left and right arms to poses captured at episode start."""

    skill_type: str = "bimanual_ee_return_home"
    object_name: str = ""
    gripper: float = 0.0
    time_dilation_factor: float | None = None


class BimanualEeReturnHomeSkill(Skill, BimanualSkill):
    """Emit synchronized home targets for both arms."""

    Cfg = BimanualEeReturnHomeSkillCfg

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        return {}

    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        return []

    def plan_targets_per_arm(
        self,
        observation: dict[str, Any],
        evaluator: Any | None,
    ) -> dict[str, list[SkillTarget]]:
        episode_start_ee = observation.get("episode_start_ee")
        if not isinstance(episode_start_ee, dict):
            raise ValueError("BimanualEeReturnHomeSkill requires episode_start_ee in the planner observation.")

        device = torch.device("cpu")
        for arm in ("left", "right"):
            pose = episode_start_ee.get(arm)
            if pose is not None:
                device = torch.as_tensor(pose[0], dtype=torch.float32).device
                break

        tdf = getattr(self.cfg, "time_dilation_factor", None)
        result: dict[str, list[SkillTarget]] = {}
        for arm in ("left", "right"):
            start_pose = episode_start_ee.get(arm)
            if start_pose is None:
                raise ValueError(f"BimanualEeReturnHomeSkill missing episode-start pose for arm={arm!r}.")
            result[arm] = [
                self.build_return_home_target(
                    start_pose,
                    arm=arm,
                    gripper=float(self.cfg.gripper),
                    time_dilation_factor=tdf,
                    device=device,
                    dual_ee_abs=True,
                )
            ]
        return result


register_skill("bimanual_ee_return_home", BimanualEeReturnHomeSkill)

__all__ = ["BimanualEeReturnHomeSkill", "BimanualEeReturnHomeSkillCfg"]
