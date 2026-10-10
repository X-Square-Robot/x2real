"""Pose filtering helpers for place candidates."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from manaenv.utils.grasp_runtime import _axis_direction_mask

from ..pick_runtime import poses_env_to_base


def _math_utils() -> Any:
    return importlib.import_module("isaaclab.utils.math")


def filter_place_ee_rotations(
    *,
    cfg: Any,
    ee_pos_env: torch.Tensor,
    ee_quat_env: torch.Tensor,
    observation: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Filter final place EE rotations using PickSkill's local-axis DSL."""

    specs = {
        "x": getattr(cfg, "filter_x_dir", None),
        "y": getattr(cfg, "filter_y_dir", None),
        "z": getattr(cfg, "filter_z_dir", None),
    }
    n_total = int(ee_quat_env.reshape(-1, 4).shape[0])
    if all(spec is None for spec in specs.values()):
        return torch.ones(n_total, dtype=torch.bool, device=device)

    _, ee_quat_base = poses_env_to_base(
        pos_env=ee_pos_env.reshape(-1, 3).to(dtype=dtype, device=device),
        quat_world=ee_quat_env.reshape(-1, 4).to(dtype=dtype, device=device),
        observation=observation,
    )
    rot_base = _math_utils().matrix_from_quat(ee_quat_base).to(dtype=dtype, device=device)
    keep = torch.ones(int(rot_base.shape[0]), dtype=torch.bool, device=device)
    for axis_name, spec in specs.items():
        if spec is not None:
            keep &= _axis_direction_mask(rot_base, axis_name=axis_name, filter_spec=spec)
    return keep
