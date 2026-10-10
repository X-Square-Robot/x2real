"""``open`` skill: grip a door handle and swing it open about the hinge.

A revolute door cannot be reliably pushed open without a grasp, so this is a
grip-based arc sweep. Drawer OPEN is a separate skill
(:class:`~manaenv.pipeline.skills.pull_skill.PullSkill`, ``pull``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .articulation_base import ArticulationGripSkill, ArticulationGripSkillCfg
from .base import register_skill


@dataclass
class OpenDoorSkillCfg(ArticulationGripSkillCfg):
    skill_type: str = "open"
    joint_type: str = "revolute"
    direction: str = "open"


class OpenDoorSkill(ArticulationGripSkill):
    """Grip a door handle and swing it open about the hinge."""

    Cfg = OpenDoorSkillCfg

    @classmethod
    def default_role_keys(cls) -> tuple[str, ...]:
        return ("openable", "object")


register_skill("open", OpenDoorSkill)

__all__ = ["OpenDoorSkill", "OpenDoorSkillCfg"]
