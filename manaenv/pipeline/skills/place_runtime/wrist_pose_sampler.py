"""Natural wrist-pose sampling for place candidates."""

from __future__ import annotations

import importlib
from typing import Any

import torch


def _math_utils() -> Any:
    return importlib.import_module("isaaclab.utils.math")


def _unit_approach_axis_local(cfg: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    axis_local = torch.as_tensor(cfg.approach_axis_local, dtype=dtype, device=device).reshape(3)
    axis_norm = torch.linalg.norm(axis_local)
    if float(axis_norm) <= 0.0:
        raise ValueError(
            "PlaceSkillCfg.approach_axis_local must be a non-zero 3-vector; " f"got {tuple(cfg.approach_axis_local)!r}."
        )
    return axis_local / axis_norm


def _unit_wrist_yaw_axis_local(cfg: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    yaw_local = torch.as_tensor(cfg.wrist_yaw_axis_local, dtype=dtype, device=device).reshape(3)
    yaw_norm = torch.linalg.norm(yaw_local)
    if float(yaw_norm) <= 0.0:
        raise ValueError(
            "PlaceSkillCfg.wrist_yaw_axis_local must be a non-zero 3-vector; "
            f"got {tuple(cfg.wrist_yaw_axis_local)!r}."
        )
    return yaw_local / yaw_norm


def _pitch_angles_rad(cfg: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    n = max(1, int(cfg.pitch_samples))
    lo = float(cfg.pitch_min_deg)
    hi = float(cfg.pitch_max_deg)
    if n == 1:
        return torch.tensor([0.5 * (lo + hi)], dtype=dtype, device=device).deg2rad()
    return torch.linspace(lo, hi, n, dtype=dtype, device=device).deg2rad()


def build_natural_rotation_batch(
    cfg: Any,
    approach_world: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    fallback_up_world: torch.Tensor | None = None,
) -> torch.Tensor:
    m = _math_utils()
    approach = approach_world.reshape(-1, 3).to(dtype=dtype, device=device)
    x_local = _unit_approach_axis_local(cfg, dtype=dtype, device=device)
    z_local = _unit_wrist_yaw_axis_local(cfg, dtype=dtype, device=device)
    y_local = torch.linalg.cross(z_local, x_local)
    y_norm = torch.linalg.norm(y_local)
    if float(y_norm) < 1e-6:
        raise ValueError("PlaceSkillCfg.approach_axis_local and wrist_yaw_axis_local must be linearly independent.")
    y_local = y_local / y_norm
    z_local = torch.linalg.cross(x_local, y_local)

    x_axis = approach / approach.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    up = torch.zeros_like(x_axis)
    up[:, 2] = 1.0
    proj = (up * x_axis).sum(dim=-1, keepdim=True) * x_axis
    z_raw = up - proj
    z_norm = torch.linalg.norm(z_raw, dim=-1, keepdim=True)

    if fallback_up_world is None:
        fb_in = torch.zeros_like(x_axis)
        fb_in[:, 1] = 1.0
    else:
        fb_in = fallback_up_world.reshape(-1, 3).to(dtype=dtype, device=device).expand_as(x_axis)
    fb_proj = (fb_in * x_axis).sum(dim=-1, keepdim=True) * x_axis
    fb = fb_in - fb_proj
    fb_norm = fb.norm(dim=-1, keepdim=True)

    plus_y = torch.zeros_like(x_axis)
    plus_y[:, 1] = 1.0
    py_proj = (plus_y * x_axis).sum(dim=-1, keepdim=True) * x_axis
    py = plus_y - py_proj
    py = py / py.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    fb = torch.where(fb_norm > 1e-6, fb / fb_norm.clamp(min=1e-9), py)

    z_axis = torch.where(z_norm > 1e-6, z_raw / z_norm.clamp(min=1e-9), fb)
    y_axis = torch.linalg.cross(z_axis, x_axis)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    z_axis = torch.linalg.cross(x_axis, y_axis)

    world_basis = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    local_basis = torch.stack([x_local, y_local, z_local], dim=-1)
    return m.quat_from_matrix(world_basis @ local_basis.transpose(-1, -2))


def make_natural_pose_candidates(
    cfg: Any,
    target_pos_world: torch.Tensor,
    base_pos_world: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = _math_utils()
    targets = target_pos_world.reshape(-1, 3).to(dtype=dtype, device=device)
    base = base_pos_world.reshape(3).to(dtype=dtype, device=device)
    n_t = targets.shape[0]

    delta = targets - base.reshape(1, 3)
    horiz = delta.clone()
    horiz[:, 2] = 0.0
    horiz_norm = torch.linalg.norm(horiz, dim=-1, keepdim=True)
    fallback = torch.zeros_like(horiz)
    fallback[:, 0] = 1.0
    horiz = torch.where(horiz_norm > 1e-6, horiz / horiz_norm.clamp(min=1e-9), fallback)

    pitches = _pitch_angles_rad(cfg, dtype=dtype, device=device)
    n_p = int(pitches.shape[0])
    sin_p = pitches.sin().reshape(1, n_p, 1)
    cos_p = pitches.cos().reshape(1, n_p, 1)
    horiz_e = horiz.reshape(n_t, 1, 3)
    neg_z = torch.zeros((1, 1, 3), dtype=dtype, device=device)
    neg_z[..., 2] = -1.0

    approach = sin_p * horiz_e + cos_p * neg_z
    approach_flat = approach.reshape(-1, 3)
    approach_flat = approach_flat / approach_flat.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    horiz_fallback = horiz.reshape(n_t, 1, 3).expand(-1, n_p, -1).reshape(-1, 3)
    place_quat = build_natural_rotation_batch(
        cfg,
        approach_flat,
        dtype=dtype,
        device=device,
        fallback_up_world=horiz_fallback,
    )

    axis_local = _unit_approach_axis_local(cfg, dtype=dtype, device=device).reshape(1, 3)
    approach_world = m.quat_apply(place_quat, axis_local.expand(place_quat.shape[0], -1))

    target_flat = targets.reshape(n_t, 1, 3).expand(-1, n_p, -1).reshape(-1, 3)
    place_link6 = target_flat - float(cfg.sampler_to_link6_backoff) * approach_world
    pre_link6 = place_link6 - float(cfg.pre_place_z_offset) * approach_world
    return pre_link6, place_link6, place_quat
