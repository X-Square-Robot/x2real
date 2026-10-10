"""Joint-space home skill for the active arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from manaenv.utils.tensor_utils import as_torch

from .base import Skill, SkillCfg, SkillTarget, register_skill

HOME_JOINT_POSITION_THRESHOLD = 0.03


@dataclass
class HomeSkillCfg(SkillCfg):
    """Configuration for returning one arm to a home joint posture."""

    skill_type: str = "home"
    arm: str = "current"
    # Numeric joint target or robot-agnostic gripper symbol ("open"/"closed"),
    # resolved to the embodiment's native joint value at setup. StaySkillCfg
    # inherits this field. Standby postures (home/stay) default to a CLOSED
    # gripper; only manipulation skills (pick/place/touch/...) open it, so the
    # hand is not left gaping open while merely transiting between operations.
    gripper: float | str = "closed"
    steps: int = 50
    home_joints: list[float] | None = None
    left_home_joints: list[float] | None = None
    right_home_joints: list[float] | None = None
    # Joint-name prefix for single-arm robots whose arm joints do NOT follow the
    # shared ``left_/right_arm_joint`` multi-arm convention (e.g. Franka exposes
    # ``panda_joint1..7``). Robot-owned, so it is declared in the embodiment
    # skill_planner YAML (skill_defaults.home) rather than hardcoded here.
    arm_joint_prefix: str | None = None


class HomeSkill(Skill):
    """Emit joint-space targets that interpolate the active arm to home."""

    Cfg = HomeSkillCfg

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        current_ee_pos = obs.get("current_ee_pos")
        current_ee_quat = obs.get("current_ee_quat")
        robot = obs.get("robot")
        if current_ee_pos is None or current_ee_quat is None:
            raise ValueError("HomeSkill.observe requires current_ee_pos/current_ee_quat in planner context.")
        if robot is None:
            raise ValueError("HomeSkill.observe requires robot in planner context.")
        return {
            "current_ee_pos": torch.as_tensor(current_ee_pos, dtype=torch.float32).reshape(3),
            "current_ee_quat": torch.as_tensor(current_ee_quat, dtype=torch.float32).reshape(4),
            "robot": robot,
            "env_idx": int(env_idx),
            "active_arm": obs.get("active_arm"),
            "raw_observation": obs.get("raw_observation"),
        }

    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        arm = self._resolve_arm(observation)
        if arm == "both":
            return self._plan_both_arm_targets(observation)
        current_q, home_q = self._resolve_joint_pair(observation, arm)
        steps = max(int(self.cfg.steps), 1)
        pos = observation["current_ee_pos"]
        quat = observation["current_ee_quat"]
        targets: list[SkillTarget] = []
        for idx in range(steps):
            ratio = float(idx + 1) / float(steps)
            joint_pos = home_q * ratio + current_q * (1.0 - ratio)
            targets.append(
                SkillTarget(
                    pos=pos,
                    quat=quat,
                    gripper=float(self.cfg.gripper),
                    phase="return_home",
                    arm=arm,
                    joint_pos=joint_pos,
                    extras={"joint_position_threshold": HOME_JOINT_POSITION_THRESHOLD},
                )
            )
        return targets

    def expand_to_waypoints(
        self,
        targets: list[SkillTarget],
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        settings: Any,
    ) -> list[dict[str, Any]]:
        """Return joint targets directly; cuRobo Cartesian densification is unnecessary."""

        return [self._target_to_waypoint(target, phase=target.phase) for target in targets]

    def _plan_both_arm_targets(self, observation: dict[str, Any]) -> list[SkillTarget]:
        left_current, left_home = self._resolve_joint_pair(observation, "left")
        right_current, right_home = self._resolve_joint_pair(observation, "right")
        steps = max(int(self.cfg.steps), 1)
        pos = observation["current_ee_pos"]
        quat = observation["current_ee_quat"]
        targets: list[SkillTarget] = []
        for idx in range(steps):
            ratio = float(idx + 1) / float(steps)
            left_joint = left_home * ratio + left_current * (1.0 - ratio)
            right_joint = right_home * ratio + right_current * (1.0 - ratio)
            targets.append(
                SkillTarget(
                    pos=pos,
                    quat=quat,
                    gripper=float(self.cfg.gripper),
                    phase="return_home",
                    arm="both",
                    joint_pos=torch.cat((left_joint, right_joint), dim=0),
                    extras={"joint_position_threshold": HOME_JOINT_POSITION_THRESHOLD},
                )
            )
        return targets

    def _resolve_joint_pair(self, observation: dict[str, Any], arm: str) -> tuple[torch.Tensor, torch.Tensor]:
        robot = observation["robot"]
        env_idx = int(observation["env_idx"])
        data = getattr(robot, "data", None)
        joint_names = list(getattr(data, "joint_names", []) or [])
        joint_pos = as_torch(getattr(data, "joint_pos", None))
        default_joint_pos = as_torch(getattr(data, "default_joint_pos", None))
        indices = self._arm_joint_indices(joint_names, arm)
        if not indices or not isinstance(joint_pos, torch.Tensor):
            raise RuntimeError(f"HomeSkill could not resolve {arm} arm joints from robot.data.")

        current_q = joint_pos[env_idx, indices].detach().clone().to(dtype=torch.float32)
        configured_joints = self._resolve_configured_home_joints(arm)
        if configured_joints is not None:
            home_q = torch.as_tensor(configured_joints, dtype=torch.float32, device=current_q.device).reshape(-1)
            if home_q.numel() != current_q.numel():
                raise ValueError(f"home_joints for {arm} must have {current_q.numel()} values, got {home_q.numel()}.")
            return current_q, home_q

        if isinstance(default_joint_pos, torch.Tensor):
            if default_joint_pos.ndim == 2:
                return current_q, default_joint_pos[env_idx, indices].detach().clone().to(dtype=torch.float32)
            if default_joint_pos.ndim == 1:
                return current_q, default_joint_pos[indices].detach().clone().to(dtype=torch.float32)
        return current_q, torch.zeros_like(current_q)

    def _resolve_configured_home_joints(self, arm: str) -> list[float] | None:
        if self.cfg.home_joints is not None:
            return self.cfg.home_joints
        if arm == "left" and self.cfg.left_home_joints is not None:
            return self.cfg.left_home_joints
        if arm == "right" and self.cfg.right_home_joints is not None:
            return self.cfg.right_home_joints
        return None

    def _arm_joint_indices(self, joint_names: list[str], arm: str) -> list[int]:
        prefix = "right_arm_joint" if arm == "right" else "left_arm_joint"
        candidates = self._indices_for_prefix(joint_names, prefix)
        if candidates:
            return candidates

        # Single-arm robots whose arm joints don't follow the shared
        # ``left_/right_arm_joint`` convention declare their prefix via the
        # embodiment skill_planner YAML (skill_defaults.home.arm_joint_prefix).
        # Such robots have only one arm, so the active/left side owns the joints
        # and the right side is intentionally empty.
        if arm == "right":
            return []
        fallback_prefix = self.cfg.arm_joint_prefix
        if not fallback_prefix:
            return []
        return self._indices_for_prefix(joint_names, fallback_prefix)

    @staticmethod
    def _indices_for_prefix(joint_names: list[str], prefix: str) -> list[int]:
        matches: list[tuple[int, int]] = []
        for idx, name in enumerate(joint_names):
            if not isinstance(name, str) or not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                matches.append((int(suffix), idx))
        matches.sort(key=lambda item: item[0])
        return [idx for _, idx in matches]


register_skill("home", HomeSkill)

__all__ = ["HomeSkill", "HomeSkillCfg"]
