"""``close`` skill: grip a door handle and swing it shut about the hinge.

A revolute door cannot be reliably pushed shut without a grasp, so this is a
grip-based arc sweep. Drawer CLOSE needs no grasp and is handled instead by
:class:`~manaenv.pipeline.skills.push_skill.PushSkill` (``push``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .articulation_base import ArticulationGripSkill, ArticulationGripSkillCfg
from .base import register_skill


@dataclass
class CloseDoorSkillCfg(ArticulationGripSkillCfg):
    skill_type: str = "close"
    joint_type: str = "revolute"
    direction: str = "close"


class CloseDoorSkill(ArticulationGripSkill):
    """Grip a door handle and swing it shut about the hinge."""

    Cfg = CloseDoorSkillCfg

    @classmethod
    def default_role_keys(cls) -> tuple[str, ...]:
        return ("openable", "object")


register_skill("close", CloseDoorSkill)

__all__ = ["CloseDoorSkill", "CloseDoorSkillCfg"]
