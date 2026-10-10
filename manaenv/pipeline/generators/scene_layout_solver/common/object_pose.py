from __future__ import annotations

from typing import Any

from .geometry import QuatXYZW


def object_base_quaternion_from_record(obj: Any) -> QuatXYZW:
    """Return the authored/default XYZW orientation for a layout problem object.

    This is the base that runtime/static yaw is composed onto
    (``yaw_quat(yaw) * base``). It must stay free of any previously sampled
    scene yaw baked into instance ``asset.initial_pose`` by ``apply_solved_poses``.
    """

    spec = obj.spec
    if spec.orientation is not None:
        return spec.orientation
    if obj.asset.default_pose is not None:
        return obj.asset.default_pose.quaternion
    return (0.0, 0.0, 0.0, 1.0)


def object_scene_yaw_range_from_record(obj: Any) -> tuple[float, float] | None:
    """Return the object-level scene yaw range, if authored."""

    override = getattr(obj, "override_yaw", None)
    if override is not None:
        return override
    return obj.spec.yaw
