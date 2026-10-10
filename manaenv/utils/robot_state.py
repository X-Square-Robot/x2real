"""Minimal live robot-state readers needed by evaluation skills."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch

from manaenv.utils.tensor_utils import as_torch


def resolve_robot_from_gym_env(gym_env: Any) -> Any | None:
    scene = getattr(gym_env, "scene", None)
    if scene is not None:
        with suppress(KeyError, TypeError, AttributeError):
            robot = scene["robot"]
            if robot is not None:
                return robot
    cfg = getattr(gym_env, "cfg", None)
    arena = getattr(cfg, "isaaclab_arena_env", None) if cfg is not None else None
    embodiment = getattr(arena, "embodiment", None) if arena is not None else None
    scene_cfg = getattr(embodiment, "scene_config", None) if embodiment is not None else None
    return getattr(scene_cfg, "robot", None) if scene_cfg is not None else None


_EE_BODY_CANDIDATES = {
    "left": ("left_arm_gripper_base_link", "left_gripper_base_link", "panda_hand"),
    "right": ("right_arm_gripper_base_link", "right_gripper_base_link"),
}


def _body_pose(robot: Any, body_name: str, env_idx: int, cache: dict[str, int]) -> tuple[Any, Any]:
    data = getattr(robot, "data", None)
    names = getattr(data, "body_names", None) if data is not None else None
    if data is None or not names:
        return None, None
    key = f"ee::{body_name}"
    index = cache.get(key)
    if index is None:
        if body_name not in names:
            cache[key] = -1
            return None, None
        index = int(list(names).index(body_name))
        cache[key] = index
    if index < 0:
        return None, None
    try:
        pos = as_torch(data.body_pos_w)[env_idx, index].detach().to(dtype=torch.float32)
        quat = as_torch(data.body_quat_w)[env_idx, index].detach().to(dtype=torch.float32)
    except (AttributeError, IndexError, RuntimeError, TypeError):
        return None, None
    return pos, quat


def get_arm_ee_local_pose(
    *, gym_env: Any, arm: str, env_idx: int, index_cache: dict[str, int]
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Read an arm end-effector pose in environment-local coordinates."""

    robot = resolve_robot_from_gym_env(gym_env)
    candidates = _EE_BODY_CANDIDATES.get(str(arm or "").strip().lower(), ())
    if robot is None or not candidates:
        return None, None
    pos = quat = None
    for body_name in candidates:
        pos, quat = _body_pose(robot, body_name, int(env_idx), index_cache)
        if pos is not None and quat is not None:
            break
    if pos is None or quat is None:
        return None, None
    origins = getattr(getattr(gym_env, "scene", None), "env_origins", None)
    if origins is not None:
        try:
            pos = pos - as_torch(origins)[int(env_idx)].to(device=pos.device, dtype=pos.dtype)
        except (IndexError, RuntimeError, TypeError):
            pass
    return pos, quat


__all__ = ["get_arm_ee_local_pose", "resolve_robot_from_gym_env"]
