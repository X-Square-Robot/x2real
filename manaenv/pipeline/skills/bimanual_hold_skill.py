"""Minimal cooperative bimanual skill placeholder.

Emits a single ``hold`` target per arm at the current end-effector pose so the
planner has a concrete :class:`BimanualSkill` it can instantiate while the real
two-arm cooperative skills (carry, handover, dual-grasp) are still being
designed. The targets carry no motion intent; they only certify that the
``BimanualSkill`` contract round-trips through the registry, the YAML loader
and ``SkillTrajectoryPlanner``'s bimanual code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .base import BimanualSkill, SkillCfg, SkillTarget, register_skill


@dataclass
class BimanualHoldSkillCfg(SkillCfg):
    """Config for :class:`BimanualHoldSkill`.

    The ``arm`` field is inherited from :class:`SkillCfg` but is unused here; a
    bimanual skill always produces targets for both arms. ``hold_gripper``
    chooses the per-arm gripper command of the emitted hold target (default
    ``0.0`` = open / no-op for the three-state gripper).
    """

    skill_type: str = "bimanual_hold"
    hold_gripper: float = 0.0


def _arm_pose_from_observation(observation: dict[str, Any], arm: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull current EE ``(pos, quat_xyzw)`` for ``arm`` from a planner observation.

    The observation dict shape mirrors what ``SkillTrajectoryPlanner`` builds
    via :meth:`Skill.observe`; bimanual skills receive a per-arm sub-dict keyed
    by ``"left"`` / ``"right"``. Falls back to a zero-pose so the skill stays
    instantiable in unit tests that have not populated EE state yet.
    """

    arm_obs = observation.get(arm) if isinstance(observation, dict) else None
    if isinstance(arm_obs, dict):
        pos = arm_obs.get("ee_pos")
        quat = arm_obs.get("ee_quat")
        if pos is not None and quat is not None:
            pos_t = torch.as_tensor(pos, dtype=torch.float32).reshape(3)
            quat_t = torch.as_tensor(quat, dtype=torch.float32).reshape(4)
            return pos_t, quat_t

    pos_t = torch.zeros(3, dtype=torch.float32)
    quat_t = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    return pos_t, quat_t


class BimanualHoldSkill(BimanualSkill):
    """Identity hold skill: each arm receives one ``hold`` target at its current EE pose.

    Use this only as a smoke-test target for the bimanual plumbing. Real
    cooperative behaviours (e.g. ``BimanualCarrySkill``) will subclass
    :class:`BimanualSkill` and emit phase-aligned per-arm sequences.
    """

    Cfg: type[BimanualHoldSkillCfg] = BimanualHoldSkillCfg

    def __init__(self, cfg: BimanualHoldSkillCfg):
        self.cfg = cfg

    def plan_targets_per_arm(
        self,
        observation: dict[str, Any],
        evaluator: Any | None,
    ) -> dict[str, list[SkillTarget]]:
        gripper = float(getattr(self.cfg, "hold_gripper", 0.0))
        arms = ("left", "right")
        result: dict[str, list[SkillTarget]] = {}
        for arm in arms:
            pos, quat = _arm_pose_from_observation(observation, arm)
            result[arm] = [
                SkillTarget(
                    pos=pos,
                    quat=quat,
                    gripper=gripper,
                    phase="hold",
                    arm=arm,
                )
            ]
        return result


register_skill("bimanual_hold", BimanualHoldSkill)


__all__ = ["BimanualHoldSkill", "BimanualHoldSkillCfg"]
