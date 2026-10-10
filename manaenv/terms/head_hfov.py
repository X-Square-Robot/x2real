"""Task-neutral EX001 head-camera horizontal FOV helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

# atan((1280 / 2) / 720.3193) ~= 41.63 degrees for the EX001 head camera.
EX001_HEAD_HFOV_HALF_DEG = 41.6
EX001_HEAD_HFOV_MARGIN_DEG = 3.0


def wrap_pi(angle_rad: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def camera_heading_rad(base_yaw_rad: float, head_yaw_rad: float) -> float:
    """Return the world heading of the head camera optical axis."""
    return wrap_pi(float(base_yaw_rad) + float(head_yaw_rad))


def target_bearing_rad(robot_xy: Sequence[float], target_xy: Sequence[float]) -> float:
    """Return the target bearing in the environment-local XY frame."""
    return math.atan2(float(target_xy[1]) - float(robot_xy[1]), float(target_xy[0]) - float(robot_xy[0]))


def heading_error_rad(bearing_rad: float, heading_rad: float) -> float:
    """Return the signed shortest bearing-to-camera-heading error."""
    return wrap_pi(float(bearing_rad) - float(heading_rad))


def target_in_horizontal_fov(
    robot_xy: Sequence[float],
    base_yaw_rad: float,
    head_yaw_rad: float,
    target_xy: Sequence[float],
    *,
    half_fov_deg: float = EX001_HEAD_HFOV_HALF_DEG,
    margin_deg: float = EX001_HEAD_HFOV_MARGIN_DEG,
) -> bool:
    """Return whether the target lies inside the horizontal FOV minus margin."""
    half_angle = math.radians(max(0.0, float(half_fov_deg) - float(margin_deg)))
    heading = camera_heading_rad(base_yaw_rad, head_yaw_rad)
    bearing = target_bearing_rad(robot_xy, target_xy)
    return abs(heading_error_rad(bearing, heading)) <= half_angle + 1e-12


__all__ = [
    "EX001_HEAD_HFOV_HALF_DEG",
    "EX001_HEAD_HFOV_MARGIN_DEG",
    "camera_heading_rad",
    "heading_error_rad",
    "target_bearing_rad",
    "target_in_horizontal_fov",
    "wrap_pi",
]
