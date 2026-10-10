"""Top-level skill planner CFGs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillConfigCFG:
    """Top-level skill planner defaults."""

    arm_mode: str = ""
    action_type: str = ""
    position_threshold: float | None = None
    orientation_threshold: float | None = None
    joint_position_threshold: float | None = None
    group_retry_enabled: bool | None = None
    max_group_retries: int | None = None
    verify_settle_steps: int | None = None
    verify_hard_cutoff_steps: int | None = None
    left_curobo: dict[str, Any] = field(default_factory=dict)
    right_curobo: dict[str, Any] = field(default_factory=dict)
