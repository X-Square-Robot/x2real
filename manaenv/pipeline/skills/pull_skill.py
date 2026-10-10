"""``pull`` skill: grip a drawer handle and pull it open (prismatic).

Drawer OPEN only. Closing a drawer needs no grasp and is handled by
:class:`~manaenv.pipeline.skills.push_skill.PushSkill` (``push``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .articulation_base import ArticulationGripSkill, ArticulationGripSkillCfg
from .base import register_skill


@dataclass
class PullSkillCfg(ArticulationGripSkillCfg):
    skill_type: str = "pull"
    joint_type: str = "prismatic"


class PullSkill(ArticulationGripSkill):
    """Grip a drawer handle and slide it out along the prismatic axis."""

    Cfg = PullSkillCfg

    @classmethod
    def default_role_keys(cls) -> tuple[str, ...]:
        return ("openable", "containable", "object")


register_skill("pull", PullSkill)

__all__ = ["PullSkill", "PullSkillCfg"]
