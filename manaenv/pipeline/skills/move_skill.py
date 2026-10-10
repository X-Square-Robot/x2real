"""Relative object-alignment move skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from manaenv.utils.scene_object_utils import get_env_origin

from .base import Skill, SkillCfg, SkillTarget, register_skill


@dataclass
class MoveSkillCfg(SkillCfg):
    """Configuration for a relative EE translation skill."""

    skill_type: str = "move"
    object_name: str = ""
    target_object_name: str = ""
    arm: str = "current"
    delta_trans: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0]])
    gripper: float = 0.0
    project_xy: bool = True


class MoveSkill(Skill):
    """Move the active EE by the relative displacement between two objects."""

    Cfg = MoveSkillCfg

    @classmethod
    def default_role_keys(cls) -> tuple[str, ...]:
        return ("pickable", "object")

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        env_origin = get_env_origin(env, env_idx)
        obj_pos, obj_quat, _ = self.read_object_pose_and_scale(env, self.cfg.object_name, env_idx, env_origin)
        tgt_pos, tgt_quat, _ = self.read_object_pose_and_scale(env, self.cfg.target_object_name, env_idx, env_origin)
        current_ee_pos = obs.get("current_ee_pos")
        current_ee_quat = obs.get("current_ee_quat")
        if current_ee_pos is None or current_ee_quat is None:
            raise ValueError("MoveSkill.observe requires current_ee_pos/current_ee_quat in planner context.")
        return {
            "object_pose_env": (obj_pos, obj_quat),
            "target_pose_env": (tgt_pos, tgt_quat),
            "current_ee_pos": torch.as_tensor(current_ee_pos, dtype=torch.float32).reshape(3),
            "current_ee_quat": torch.as_tensor(current_ee_quat, dtype=torch.float32).reshape(4),
            "active_arm": obs.get("active_arm"),
            "held_object_name": obs.get("held_object_name"),
            "raw_observation": obs.get("raw_observation"),
        }

    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        obj_pos, _ = observation["object_pose_env"]
        tgt_pos, _ = observation["target_pose_env"]
        current_pos = observation["current_ee_pos"]
        current_quat = observation["current_ee_quat"]
        arm = self._resolve_arm(observation)

        delta = torch.as_tensor(tgt_pos - obj_pos, dtype=torch.float32, device=current_pos.device)
        if self.cfg.project_xy:
            delta[2] = 0.0

        targets: list[SkillTarget] = []
        for raw_offset in self.cfg.delta_trans:
            offset = torch.as_tensor(raw_offset, dtype=torch.float32, device=current_pos.device).reshape(3)
            targets.append(
                SkillTarget(
                    pos=current_pos + delta + offset,
                    quat=current_quat,
                    gripper=float(self.cfg.gripper),
                    phase="move",
                    arm=arm,
                )
            )
        return targets


register_skill("move", MoveSkill)

__all__ = ["MoveSkill", "MoveSkillCfg"]
