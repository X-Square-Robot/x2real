"""Bimanual approach-pose skill: both arms move to episode-start EE + offset together."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .base import BimanualSkill, Skill, SkillCfg, SkillTarget, register_skill
from .ee_offset_from_start_skill import compute_offset_target_pose, resolve_offset_steps


@dataclass
class BimanualEeOffsetFromStartSkillCfg(SkillCfg):
    """Move left and right arms to the same relative offset from episode start."""

    skill_type: str = "bimanual_ee_offset_from_start"
    position_offset: list[float] = field(default_factory=lambda: [0.0, -0.05, 0.3])
    # Optional multi-step offsets (each relative to episode-start EE). When set,
    # overrides ``position_offset``. Intermediate steps use ``pre_approach_pose``.
    position_offsets: list[list[float]] | None = None
    offset_frame: str = "ee"
    gripper: float = 0.0
    time_dilation_factor: float | None = None
    # Phase name for the last offset step. Use ``return_approach_pose`` after
    # grasp so every FK sample waits for convergence (not streaming ``approach_pose``).
    final_phase: str = "approach_pose"
    # Optional per-episode env-frame Y jitter applied to every offset step for each arm.
    left_y_offset: float = 0.0
    right_y_offset: float = 0.0


class BimanualEeOffsetFromStartSkill(Skill, BimanualSkill):
    """Emit synchronized approach targets for both arms."""

    Cfg = BimanualEeOffsetFromStartSkillCfg

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        # Bimanual skills are observed via SkillTrajectoryPlanner._run_bimanual_skill_observe.
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
            raise ValueError("BimanualEeOffsetFromStartSkill requires episode_start_ee in the planner observation.")

        device = torch.device("cpu")
        for arm in ("left", "right"):
            pose = episode_start_ee.get(arm)
            if pose is not None:
                device = torch.as_tensor(pose[0], dtype=torch.float32).device
                break

        frame = str(getattr(self.cfg, "offset_frame", "ee") or "ee")
        offset_steps = self._resolve_offset_steps()
        tdf = getattr(self.cfg, "time_dilation_factor", None)

        result: dict[str, list[SkillTarget]] = {}
        for arm in ("left", "right"):
            start_pose = episode_start_ee.get(arm)
            if start_pose is None:
                raise ValueError(f"BimanualEeOffsetFromStartSkill missing episode-start pose for arm={arm!r}.")
            arm_y_extra = float(getattr(self.cfg, f"{arm}_y_offset", 0.0) or 0.0)
            targets: list[SkillTarget] = []
            for offset_raw, phase in offset_steps:
                offset_list = [float(v) for v in offset_raw]
                if len(offset_list) == 3 and arm_y_extra != 0.0:
                    offset_list[1] += arm_y_extra
                offset = torch.as_tensor(offset_list, dtype=torch.float32, device=device).reshape(3)
                target_pos, target_quat = compute_offset_target_pose(
                    start_pose[0],
                    start_pose[1],
                    offset=offset,
                    offset_frame=frame,
                    device=device,
                )
                extras: dict[str, Any] = {"dual_ee_abs": True}
                if tdf is not None:
                    extras["time_dilation_factor"] = float(tdf)
                targets.append(
                    SkillTarget(
                        pos=target_pos,
                        quat=target_quat,
                        gripper=float(self.cfg.gripper),
                        phase=phase,
                        arm=arm,
                        extras=extras,
                    )
                )
            result[arm] = targets
        return result

    def _resolve_offset_steps(self) -> list[tuple[list[float], str]]:
        return resolve_offset_steps(
            getattr(self.cfg, "position_offsets", None),
            self.cfg.position_offset,
            str(getattr(self.cfg, "final_phase", "approach_pose") or "approach_pose"),
            skill_name="bimanual_ee_offset_from_start",
        )


register_skill("bimanual_ee_offset_from_start", BimanualEeOffsetFromStartSkill)

__all__ = ["BimanualEeOffsetFromStartSkill", "BimanualEeOffsetFromStartSkillCfg"]
