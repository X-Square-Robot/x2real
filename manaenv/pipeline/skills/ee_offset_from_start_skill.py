"""Cartesian skill that moves one arm to an offset from its episode-start EE pose."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import torch

from .base import Skill, SkillCfg, SkillTarget, register_skill


def _isaaclab_math_utils() -> Any:
    return importlib.import_module("isaaclab.utils.math")


def compute_offset_target_pose(
    start_pos: Any,
    start_quat: Any,
    *,
    offset: torch.Tensor,
    offset_frame: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return env-local (pos, quat_xyzw) for episode-start EE plus ``offset``."""

    pos = torch.as_tensor(start_pos, dtype=torch.float32, device=device).reshape(3)
    quat = torch.as_tensor(start_quat, dtype=torch.float32, device=device).reshape(4)
    frame = str(offset_frame or "ee").strip().lower()
    if frame == "env":
        offset_env = offset.reshape(3)
    elif frame == "ee":
        math_utils = _isaaclab_math_utils()
        offset_env = math_utils.quat_apply(quat.reshape(1, 4), offset.reshape(1, 3)).reshape(3)
    else:
        raise ValueError(f"Unsupported offset_frame: {frame!r}")
    return pos + offset_env, quat


def resolve_offset_steps(
    position_offsets: list[list[float]] | None,
    position_offset: list[float],
    final_phase: str,
    *,
    skill_name: str,
) -> list[tuple[list[float], str]]:
    """Expand configured offsets into ``(offset, phase)`` steps.

    Intermediate steps use ``pre_approach_pose``; the last uses ``final_phase``.
    """
    if position_offsets:
        offsets = [list(item) for item in position_offsets]
    else:
        offsets = [list(position_offset)]
    if not offsets:
        raise ValueError(f"{skill_name} requires at least one position offset.")

    last_phase = str(final_phase or "approach_pose").strip()
    steps: list[tuple[list[float], str]] = []
    for idx, offset in enumerate(offsets):
        phase = "pre_approach_pose" if idx < len(offsets) - 1 else last_phase
        steps.append((offset, phase))
    return steps


@dataclass
class EeOffsetFromStartSkillCfg(SkillCfg):
    """Move the selected arm to episode-start EE plus a configurable offset."""

    skill_type: str = "ee_offset_from_start"
    arm: str = "current"
    # When set, exclude this scene role from cuRobo collision (grasped object).
    object_name: str = ""
    # Offset in env-local coordinates when ``offset_frame`` is ``env``, or in the
    # episode-start EE frame when ``offset_frame`` is ``ee``.
    position_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Optional multi-step offsets (each relative to episode-start EE). When set,
    # overrides ``position_offset`` and mirrors the bimanual pre_approach -> approach
    # geometry used before grasp.
    position_offsets: list[list[float]] | None = None
    offset_frame: str = "ee"
    gripper: float = 1.89
    # Waypoint phase label (e.g. ``return_approach_pose`` after grasp).
    target_phase: str = "approach_pose"
    # Optional cuRobo slowdown for this segment (smaller => slower joint motion).
    time_dilation_factor: float | None = None


class EeOffsetFromStartSkill(Skill):
    """Emit one EE target offset from the arm's episode-start pose."""

    Cfg = EeOffsetFromStartSkillCfg

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        current_ee_pos = obs.get("current_ee_pos")
        current_ee_quat = obs.get("current_ee_quat")
        if current_ee_pos is None or current_ee_quat is None:
            raise ValueError(
                "EeOffsetFromStartSkill.observe requires current_ee_pos/current_ee_quat in planner context."
            )
        return {
            "env_idx": env_idx,
            "current_ee_pos": torch.as_tensor(current_ee_pos, dtype=torch.float32).reshape(3),
            "current_ee_quat": torch.as_tensor(current_ee_quat, dtype=torch.float32).reshape(4),
            "active_arm": obs.get("active_arm"),
            "episode_start_ee": obs.get("episode_start_ee"),
            "approach_ee_env": obs.get("approach_ee_env"),
            "held_object_name": obs.get("held_object_name"),
            "gym_env": env,
        }

    def _resolve_offset_steps(self) -> list[tuple[list[float], str]]:
        return resolve_offset_steps(
            getattr(self.cfg, "position_offsets", None),
            self.cfg.position_offset,
            str(getattr(self.cfg, "target_phase", "approach_pose") or "approach_pose"),
            skill_name="ee_offset_from_start",
        )

    def _plan_post_grasp_targets(
        self,
        *,
        arm: str,
        observation: dict[str, Any],
        approach_pose: tuple[Any, Any],
        offset_steps: list[tuple[list[float], str]],
        frame: str,
        extras: dict[str, Any],
    ) -> list[SkillTarget]:
        """Return from shelf retract to the pinned pre-grasp approach pose.

        Intermediate offsets are applied from the *current* EE so the arm stays
        at shelf height; the final waypoint is ``approach_ee_env`` from the
        bimanual pre-grasp step (not episode-start home height).
        """
        device = observation["current_ee_pos"].device
        current_pos = observation["current_ee_pos"].reshape(3)
        current_quat = observation["current_ee_quat"].reshape(4)
        final_pos, final_quat = approach_pose
        final_pos = torch.as_tensor(final_pos, dtype=torch.float32, device=device).reshape(3)
        final_quat = torch.as_tensor(final_quat, dtype=torch.float32, device=device).reshape(4)

        targets: list[SkillTarget] = []
        for offset_raw, phase in offset_steps[:-1]:
            offset = torch.as_tensor(offset_raw, dtype=torch.float32, device=device).reshape(3)
            target_pos, target_quat = compute_offset_target_pose(
                current_pos,
                current_quat,
                offset=offset,
                offset_frame=frame,
                device=device,
            )
            targets.append(
                SkillTarget(
                    pos=target_pos,
                    quat=target_quat,
                    gripper=float(self.cfg.gripper),
                    phase=phase,
                    arm=arm,
                    extras=dict(extras),
                )
            )

        final_phase = offset_steps[-1][1]
        targets.append(
            SkillTarget(
                pos=final_pos,
                quat=final_quat,
                gripper=float(self.cfg.gripper),
                phase=final_phase,
                arm=arm,
                extras=dict(extras),
            )
        )
        return targets

    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        arm = self._resolve_arm(observation)
        episode_start_ee = observation.get("episode_start_ee")
        if not isinstance(episode_start_ee, dict):
            raise ValueError("EeOffsetFromStartSkill requires episode_start_ee in the planner observation.")

        start_pose = episode_start_ee.get(arm)
        if start_pose is None:
            raise ValueError(f"EeOffsetFromStartSkill missing episode-start pose for arm={arm!r}.")

        device = observation["current_ee_pos"].device
        start_pos, start_quat = start_pose
        frame = str(getattr(self.cfg, "offset_frame", "ee") or "ee")
        offset_steps = self._resolve_offset_steps()

        extras: dict[str, Any] = {}
        tdf = getattr(self.cfg, "time_dilation_factor", None)
        if tdf is not None:
            extras["time_dilation_factor"] = float(tdf)

        held = str(observation.get("held_object_name") or "").strip()
        approach_ee_env = observation.get("approach_ee_env")
        approach_pose = approach_ee_env.get(arm) if isinstance(approach_ee_env, dict) else None
        if held and approach_pose is not None:
            return self._plan_post_grasp_targets(
                arm=arm,
                observation=observation,
                approach_pose=approach_pose,
                offset_steps=offset_steps,
                frame=frame,
                extras=extras,
            )

        targets: list[SkillTarget] = []
        for offset_raw, phase in offset_steps:
            offset = torch.as_tensor(offset_raw, dtype=torch.float32, device=device).reshape(3)
            target_pos, target_quat = compute_offset_target_pose(
                start_pos,
                start_quat,
                offset=offset,
                offset_frame=frame,
                device=device,
            )
            targets.append(
                SkillTarget(
                    pos=target_pos,
                    quat=target_quat,
                    gripper=float(self.cfg.gripper),
                    phase=phase,
                    arm=arm,
                    extras=dict(extras),
                )
            )
        return targets


register_skill("ee_offset_from_start", EeOffsetFromStartSkill)

__all__ = ["EeOffsetFromStartSkill", "EeOffsetFromStartSkillCfg"]
