"""Last-joint limit math for limit-aware knob twisting (no sim / torch).

Limit values come from embodiment ``skill_defaults.twist`` via
:func:`resolve_joint_limit_spec`. The monitored joint is the **last DOF** of the
active arm (resolved at runtime in :class:`TwistSkill`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JointLimitSpec:
    """Soft travel envelope for the arm's last joint (degrees)."""

    min_deg: float
    max_deg: float
    margin_deg: float = 10.0
    flip_deg: float = 180.0

    def enabled(self) -> bool:
        return self.min_deg < self.max_deg


def resolve_joint_limit_spec(cfg: Any) -> JointLimitSpec | None:
    """Build a limit spec from skill/embodiment cfg; ``None`` when not configured."""
    if cfg is None:
        return None
    lo = getattr(cfg, "last_joint_limit_min_deg", None)
    hi = getattr(cfg, "last_joint_limit_max_deg", None)
    if lo is None or hi is None:
        return None
    margin = float(getattr(cfg, "limit_margin_deg", 10.0) or 10.0)
    flip = float(getattr(cfg, "wrist_flip_deg", 180.0) or 180.0)
    spec = JointLimitSpec(min_deg=float(lo), max_deg=float(hi), margin_deg=margin, flip_deg=flip)
    return spec if spec.enabled() else None


def near_min_limit(joint_deg: float, spec: JointLimitSpec) -> bool:
    return float(joint_deg) <= float(spec.min_deg) + float(spec.margin_deg)


def near_max_limit(joint_deg: float, spec: JointLimitSpec) -> bool:
    return float(joint_deg) >= float(spec.max_deg) - float(spec.margin_deg)


def near_limit_for_direction(joint_deg: float, direction: int, spec: JointLimitSpec) -> bool:
    if int(direction) > 0:
        return near_max_limit(joint_deg, spec)
    if int(direction) < 0:
        return near_min_limit(joint_deg, spec)
    return False


def max_travel_deg(joint_deg: float, direction: int, spec: JointLimitSpec) -> float:
    if int(direction) > 0:
        return max(0.0, (float(spec.max_deg) - float(spec.margin_deg)) - float(joint_deg))
    if int(direction) < 0:
        return max(0.0, float(joint_deg) - (float(spec.min_deg) + float(spec.margin_deg)))
    return 0.0


def flip_joint_deg(joint_deg: float, spec: JointLimitSpec) -> float:
    """Flip the last joint by ``flip_deg``; if +flip exceeds limits, use -flip."""
    joint = float(joint_deg)
    flip = float(spec.flip_deg)
    add = joint + flip
    if add > float(spec.max_deg) or add < float(spec.min_deg):
        return clamp_joint_deg(joint - flip, spec)
    return clamp_joint_deg(add, spec)


def clamp_joint_deg(joint_deg: float, spec: JointLimitSpec) -> float:
    return max(float(spec.min_deg), min(float(spec.max_deg), float(joint_deg)))


def joint_rad_to_deg(joint_rad: float) -> float:
    return math.degrees(float(joint_rad))


def joint_deg_to_rad(joint_deg: float) -> float:
    return math.radians(float(joint_deg))
