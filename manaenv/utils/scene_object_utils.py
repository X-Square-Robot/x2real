"""Helpers for reading rigid-object state from an assembled IsaacLab scene."""

from __future__ import annotations

from typing import Any

import torch

from manaenv.utils.tensor_utils import as_torch


def get_env_origin(gym_env: Any, env_idx: int) -> torch.Tensor | None:
    """Return one env origin tensor if available."""

    scene = getattr(gym_env, "scene", None)
    env_origins = getattr(scene, "env_origins", None) if scene is not None else None
    if env_origins is None:
        return None
    try:
        return env_origins[env_idx].clone().to(dtype=torch.float32)
    except Exception:
        return None


def _resolve_scene_asset(gym_env: Any, object_name: str) -> Any | None:
    """Resolve a scene asset by name from rigid objects or articulations.

    Articulation assets (e.g. a drawer with a prismatic joint) live in
    ``scene.articulations`` rather than ``scene.rigid_objects``; both expose
    ``data.root_pos_w`` / ``root_quat_w`` so pose reads are uniform.
    """

    scene = getattr(gym_env, "scene", None)
    if scene is None:
        return None
    for attr in ("rigid_objects", "articulations"):
        registry = getattr(scene, attr, None)
        if registry is not None and object_name in registry:
            return registry[object_name]
    return None


def get_rigid_object_world_pose(
    gym_env: Any, object_name: str, env_idx: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(pos_w, quat_xyzw)`` for one rigid object or articulation root."""

    obj = _resolve_scene_asset(gym_env, object_name)
    if obj is None:
        return None
    try:
        pos_w = obj.data.root_pos_w.torch[env_idx].clone().to(dtype=torch.float32)
        quat_w = obj.data.root_quat_w.torch[env_idx].clone().to(dtype=torch.float32)
    except Exception:
        return None
    return pos_w, quat_w


def get_rigid_object_env_pose(
    gym_env: Any,
    object_name: str,
    env_idx: int,
    env_origin: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(pos_env_local, quat_xyzw)`` for one rigid object."""

    world_pose = get_rigid_object_world_pose(gym_env, object_name, env_idx)
    if world_pose is None:
        return None
    pos_w, quat_w = world_pose
    origin = env_origin if isinstance(env_origin, torch.Tensor) else get_env_origin(gym_env, env_idx)
    if isinstance(origin, torch.Tensor):
        pos_env = pos_w - origin[:3].to(device=pos_w.device)
    else:
        pos_env = pos_w
    return pos_env, quat_w


def get_articulation_body_env_pose(
    gym_env: Any,
    articulation_name: str,
    body_name: str,
    env_idx: int,
    env_origin: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(pos_env_local, quat_xyzw)`` for one articulation child link.

    Unlike :func:`get_rigid_object_env_pose` (which reads the articulation ROOT
    link ``root_pos_w``), this reads a specific body/link via ``data.body_pos_w``
    / ``data.body_quat_w`` indexed through ``body_names``. Articulation grasp
    skills anchor their decode to the *operated* link (e.g. a drawer body) so
    labels authored in that link's local frame stay exact and track the joint
    at any open state. Returns ``None`` when the articulation, its body data, or
    the named body cannot be resolved (caller should fall back to the root).
    """

    scene = getattr(gym_env, "scene", None)
    articulations = getattr(scene, "articulations", None) if scene is not None else None
    if not articulations or articulation_name not in articulations:
        return None
    art = articulations[articulation_name]
    data = getattr(art, "data", None)
    if data is None:
        return None
    names = list(getattr(art, "body_names", None) or getattr(data, "body_names", None) or [])
    if body_name not in names:
        return None
    body_index = names.index(body_name)
    try:
        pos_w = as_torch(data.body_pos_w)[env_idx, body_index].clone().to(dtype=torch.float32)
        quat_w = as_torch(data.body_quat_w)[env_idx, body_index].clone().to(dtype=torch.float32)
    except Exception:
        return None
    origin = env_origin if isinstance(env_origin, torch.Tensor) else get_env_origin(gym_env, env_idx)
    if isinstance(origin, torch.Tensor):
        pos_env = pos_w - origin[:3].to(device=pos_w.device)
    else:
        pos_env = pos_w
    return pos_env, quat_w


def get_articulation_joint_value(
    gym_env: Any,
    articulation_name: str,
    env_idx: int,
    joint_name: str | None = None,
) -> float | None:
    """Return the live position of one articulation joint, or ``None``.

    Used by drawer-style skills to read how far a prismatic drawer joint is
    currently open so a grasp label (defined in the closed-state root frame)
    can be shifted onto the live handle. When ``joint_name`` is ``None`` the
    first joint is used. Returns ``None`` when the articulation, its joint
    data, or the named joint cannot be resolved (caller should treat that as a
    zero offset).
    """

    scene = getattr(gym_env, "scene", None)
    articulations = getattr(scene, "articulations", None) if scene is not None else None
    if not articulations or articulation_name not in articulations:
        return None
    art = articulations[articulation_name]
    data = getattr(art, "data", None)
    joint_pos = as_torch(getattr(data, "joint_pos", None))
    if joint_pos is None:
        return None
    joint_index = 0
    if joint_name:
        names = list(getattr(data, "joint_names", None) or getattr(art, "joint_names", None) or [])
        if joint_name not in names:
            return None
        joint_index = names.index(joint_name)
    try:
        return float(joint_pos[env_idx, joint_index].item())
    except Exception:
        return None


def get_rigid_object_scale(gym_env: Any, object_name: str) -> tuple[float, float, float]:
    """Return the spawn-time XForm scale of a rigid object.

    Reads ``scene.rigid_objects[name].cfg.spawn.scale`` (set via the YAML
    ``scale: [sx, sy, sz]`` field on scene objects). Per-episode dynamic
    rescaling of the live USD prim is not honored; if randomization is later
    added at the prim level, this helper must instead read
    ``UsdGeom.Xformable.GetOrderedXformOps()`` directly.

    Returns ``(1.0, 1.0, 1.0)`` when the object is missing, has no spawn
    config, or ``spawn.scale`` is ``None``.
    """

    obj = _resolve_scene_asset(gym_env, object_name)
    if obj is None:
        return (1.0, 1.0, 1.0)
    spawn = getattr(getattr(obj, "cfg", None), "spawn", None)
    raw = getattr(spawn, "scale", None)
    if raw is None:
        return (1.0, 1.0, 1.0)
    try:
        sx, sy, sz = (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError, IndexError):
        return (1.0, 1.0, 1.0)
    return (sx, sy, sz)


def compute_per_env_max_rigid_speed(gym_env: Any) -> torch.Tensor | None:
    """Return ``(num_envs,)`` max linear speed across all scene rigid objects.

    Used by the evaluation settle-detector to decide which envs are
    still bouncing after a (re)spawn. Returns ``None`` when the scene has
    no rigid objects (in that case the settle window degenerates to a
    one-step no-op).
    """

    scene = getattr(gym_env, "scene", None)
    rigid_objects = getattr(scene, "rigid_objects", None) if scene is not None else None
    if not rigid_objects:
        return None
    speeds: list[torch.Tensor] = []
    for obj in rigid_objects.values():
        vel = obj.data.root_lin_vel_w.torch
        speeds.append(torch.linalg.norm(vel, dim=-1))
    stacked = torch.stack(speeds, dim=-1)
    return stacked.max(dim=-1).values


__all__ = [
    "compute_per_env_max_rigid_speed",
    "get_articulation_body_env_pose",
    "get_articulation_joint_value",
    "get_env_origin",
    "get_rigid_object_env_pose",
    "get_rigid_object_scale",
    "get_rigid_object_world_pose",
]
