"""Place into the region exposed by a prismatic joint opening.

General (not asset-specific):
  1) sliding body = the link the joint moves
  2) closed tray XY from that body's footprint, rewound by joint travel
  3) opened placeable XY = protruding slab along the open axis
  4) place Z = that body's AABB top in env + place_z_offset (above the lid;
     horizontal open does not change Z)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class OpenedJointRegion:
    """Env-frame opened placeable box + place center."""

    closed_min_xy: torch.Tensor  # (2,)
    closed_max_xy: torch.Tensor
    open_min_xy: torch.Tensor
    open_max_xy: torch.Tensor
    place_xy: torch.Tensor  # (2,)
    open_distance: float
    axis_world: torch.Tensor  # (3,)
    tray_z: float  # env-frame floor of the sliding body
    tray_top_z: float  # env-frame top of the sliding-body AABB (bbox lid)
    body_pos: torch.Tensor  # (3,) live sliding-body origin in env


def opened_placeable_xy_box(
    closed_min_xy: torch.Tensor,
    closed_max_xy: torch.Tensor,
    open_vec_xy: torch.Tensor,
    *,
    protruding_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shift closed XY by ``open_vec_xy``; optionally keep only the protruding slab."""
    cmin = closed_min_xy.reshape(2).to(dtype=torch.float32)
    cmax = closed_max_xy.reshape(2).to(dtype=torch.float32)
    dxy = open_vec_xy.reshape(2).to(dtype=torch.float32)
    tray_min = cmin + dxy
    tray_max = cmax + dxy
    if not protruding_only:
        center = 0.5 * (tray_min + tray_max)
        return tray_min, tray_max, center

    open_min = tray_min.clone()
    open_max = tray_max.clone()
    # Dominant open axis only — ignore tiny off-axis noise.
    dom = int(torch.argmax(torch.abs(dxy)).item())
    di = float(dxy[dom].item())
    if abs(di) > 1.0e-6:
        if di < 0.0:
            open_min[dom] = tray_min[dom]
            open_max[dom] = cmin[dom]
        else:
            open_min[dom] = cmax[dom]
            open_max[dom] = tray_max[dom]
    if float((open_max - open_min).min().item()) <= 1.0e-6:
        open_min, open_max = tray_min, tray_max
    center = 0.5 * (open_min + open_max)
    return open_min, open_max, center


def _axis_world_from_root(
    env: Any,
    object_name: str,
    env_idx: int,
    joint_axis_local: tuple[float, float, float] | list[float],
    env_origin: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    from manaenv.utils.scene_object_utils import get_rigid_object_env_pose

    pose = get_rigid_object_env_pose(env, object_name, env_idx, env_origin=env_origin)
    axis_l = torch.as_tensor(joint_axis_local, dtype=torch.float32, device=device).reshape(3)
    n = float(torch.linalg.norm(axis_l).item())
    axis_l = axis_l / n if n >= 1.0e-8 else torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    if pose is None:
        return axis_l
    _pos, quat = pose
    try:
        from isaaclab.utils.math import matrix_from_quat

        R = matrix_from_quat(quat.reshape(1, 4).to(device=device)).reshape(3, 3)
    except Exception:
        R = torch.eye(3, dtype=torch.float32, device=device)
    axis_w = R @ axis_l
    n_w = float(torch.linalg.norm(axis_w).item())
    return axis_w / n_w if n_w >= 1.0e-8 else axis_l


def _sliding_body_local_aabb(
    env: Any,
    *,
    object_name: str,
    env_idx: int,
    sliding_body_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local AABB of the sliding articulation body (the joint's moving link)."""
    fallback_min = torch.tensor([-0.08, -0.06, -0.01], dtype=torch.float32, device=device)
    fallback_max = torch.tensor([0.08, 0.06, 0.05], dtype=torch.float32, device=device)
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception:
        return fallback_min, fallback_max

    scene = getattr(env, "scene", None)
    arts = getattr(scene, "articulations", None) if scene is not None else None
    if not arts or object_name not in arts:
        return fallback_min, fallback_max
    root = str(getattr(getattr(arts[object_name], "cfg", None), "prim_path", "") or "").strip()
    if not root:
        return fallback_min, fallback_max
    root = root.replace("{ENV_REGEX_NS}", f"/World/envs/env_{int(env_idx)}")
    root = root.replace("env_.*", f"env_{int(env_idx)}")
    body = str(sliding_body_name or "").strip().lstrip("/")
    stage = omni.usd.get_context().get_stage()
    if stage is None or not body:
        return fallback_min, fallback_max
    prim = stage.GetPrimAtPath(f"{root.rstrip('/')}/{body}")
    if prim is None or not prim.IsValid():
        return fallback_min, fallback_max
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeUntransformedBound(prim).GetBox()
    bmin, bmax = box.GetMin(), box.GetMax()
    local_min = torch.tensor([float(bmin[0]), float(bmin[1]), float(bmin[2])], dtype=torch.float32, device=device)
    local_max = torch.tensor([float(bmax[0]), float(bmax[1]), float(bmax[2])], dtype=torch.float32, device=device)
    if float((local_max - local_min).max().item()) <= 1.0e-6:
        return fallback_min, fallback_max
    return local_min, local_max


def _body_z_extents_env(
    *,
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    local_min: torch.Tensor,
    local_max: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Env-frame (floor_z, top_z) of the sliding-body AABB after live pose."""
    try:
        from isaaclab.utils.math import matrix_from_quat

        R = matrix_from_quat(body_quat.reshape(1, 4).to(device=device)).reshape(3, 3)
    except Exception:
        R = torch.eye(3, dtype=torch.float32, device=device)
    corners = []
    for ix in (local_min[0], local_max[0]):
        for iy in (local_min[1], local_max[1]):
            for iz in (local_min[2], local_max[2]):
                corners.append(torch.stack([ix, iy, iz]))
    local = torch.stack(corners, dim=0)
    world = (R @ local.T).T + body_pos.reshape(1, 3)
    return float(world[:, 2].min().item()), float(world[:, 2].max().item())


def resolve_opened_joint_region(
    *,
    env: Any,
    object_name: str,
    env_idx: int,
    env_origin: torch.Tensor | None,
    sliding_body_name: str,
    joint_name: str,
    joint_axis_local: tuple[float, float, float] | list[float],
    device: torch.device,
    protruding_only: bool = True,
    xy_inset_ratio: float = 0.0,
) -> OpenedJointRegion | None:
    """Live joint open → closed tray XY → opened placeable XY; Z from body floor."""
    from manaenv.utils.scene_object_utils import get_articulation_body_env_pose, get_articulation_joint_value

    body_pose = get_articulation_body_env_pose(
        env,
        object_name,
        sliding_body_name,
        env_idx,
        env_origin=env_origin,
    )
    if body_pose is None:
        return None
    body_pos = body_pose[0].to(device=device, dtype=torch.float32).reshape(3)
    body_quat = body_pose[1].to(device=device, dtype=torch.float32).reshape(4)

    open_dist = get_articulation_joint_value(env, object_name, env_idx, joint_name=joint_name or None)
    open_dist = float(open_dist) if open_dist is not None else 0.0

    axis_w = _axis_world_from_root(env, object_name, env_idx, joint_axis_local, env_origin, device)
    open_vec = axis_w * open_dist

    local_min, local_max = _sliding_body_local_aabb(
        env,
        object_name=object_name,
        env_idx=env_idx,
        sliding_body_name=sliding_body_name,
        device=device,
    )
    inset = max(0.0, min(0.49, float(xy_inset_ratio)))
    half_xy = 0.5 * (local_max[:2] - local_min[:2]) * (1.0 - inset)
    # Body origin is the open tray frame; rewind XY by joint travel for closed box.
    closed_center_xy = body_pos[:2] - open_vec[:2]
    closed_min = closed_center_xy - half_xy
    closed_max = closed_center_xy + half_xy
    open_min, open_max, place_xy = opened_placeable_xy_box(
        closed_min, closed_max, open_vec[:2], protruding_only=protruding_only
    )
    # Horizontal prismatic: Z extents from live sliding-body AABB.
    tray_z, tray_top_z = _body_z_extents_env(
        body_pos=body_pos,
        body_quat=body_quat,
        local_min=local_min,
        local_max=local_max,
        device=device,
    )
    return OpenedJointRegion(
        closed_min_xy=closed_min,
        closed_max_xy=closed_max,
        open_min_xy=open_min,
        open_max_xy=open_max,
        place_xy=place_xy,
        open_distance=open_dist,
        axis_world=axis_w,
        tray_z=tray_z,
        tray_top_z=tray_top_z,
        body_pos=body_pos.detach().clone(),
    )


__all__ = [
    "OpenedJointRegion",
    "opened_placeable_xy_box",
    "resolve_opened_joint_region",
]
