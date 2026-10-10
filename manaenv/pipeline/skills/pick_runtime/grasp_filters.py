"""Geometric filtering helpers for pick grasp candidates."""

from __future__ import annotations

from typing import Any

import torch

from manaenv.utils.grasp_runtime import (
    GraspFilterProfile,
    _axis_direction_mask,
    horizontal_close_filter_y_dir,
    merge_resolved_profile,
    resolve_filter_profile,
    should_force_horizontal_close,
)


def resolve_profile_chain(
    *,
    cfg: Any,
    metadata: Any,
    object_quat: torch.Tensor,
) -> tuple[GraspFilterProfile, ...]:
    return resolve_filter_profile(cfg, metadata, object_quat)


def apply_filter_spec(
    *,
    rot_base: torch.Tensor,
    grasp_pos_base: torch.Tensor,
    obj_pos_base: torch.Tensor | None,
    spec: dict[str, Any],
    grasp_pos_obj: torch.Tensor | None = None,
) -> torch.Tensor:
    device = rot_base.device
    n_total = rot_base.shape[0]
    keep = torch.ones(n_total, dtype=torch.bool, device=device)

    if spec.get("filter_x_dir") is not None:
        keep &= _axis_direction_mask(rot_base, axis_name="x", filter_spec=spec["filter_x_dir"])

    if spec.get("filter_y_dir") is not None:
        keep &= _axis_direction_mask(rot_base, axis_name="y", filter_spec=spec["filter_y_dir"])

    if spec.get("filter_z_dir") is not None:
        keep &= _axis_direction_mask(rot_base, axis_name="z", filter_spec=spec["filter_z_dir"])

    if spec.get("grasp_region") is not None:
        keep &= _grasp_region_mask(grasp_pos_obj=grasp_pos_obj, grasp_region=spec["grasp_region"])

    direction_to_obj = str(spec.get("direction_to_obj") or "").strip().lower()
    if direction_to_obj and obj_pos_base is not None:
        obj_y = float(obj_pos_base[0, 1].item())
        if direction_to_obj == "right":
            keep &= grasp_pos_base[:, 1] <= obj_y
        elif direction_to_obj == "left":
            keep &= grasp_pos_base[:, 1] > obj_y
        else:
            raise ValueError("direction_to_obj must be either 'left' or 'right'.")

    return keep


def _grasp_region_mask(*, grasp_pos_obj: torch.Tensor | None, grasp_region: Any) -> torch.Tensor:
    if grasp_pos_obj is None:
        raise ValueError("grasp_region requires object-local grasp positions.")
    if not isinstance(grasp_region, dict):
        raise ValueError("grasp_region must be a mapping.")

    frame = str(grasp_region.get("frame", "object_local")).strip().lower()
    if frame != "object_local":
        raise ValueError("grasp_region.frame currently supports only 'object_local'.")

    axis_to_col = {"x": 0, "y": 1, "z": 2}
    axis = str(grasp_region.get("axis", "")).strip().lower()
    if axis not in axis_to_col:
        raise ValueError("grasp_region.axis must be one of: x, y, z.")

    raw_range = grasp_region.get("range")
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        raise ValueError("grasp_region.range must be [min, max].")
    lo = float(raw_range[0])
    hi = float(raw_range[1])
    if lo > hi:
        raise ValueError(f"grasp_region.range min ({lo}) must not exceed max ({hi}).")

    coord = grasp_pos_obj[:, axis_to_col[axis]]
    if lo < -1.0 or hi > 1.0:
        raise ValueError("grasp_region.range values must be within [-1, 1] (normalized over candidate extent).")
    coord_min = coord.min()
    coord_max = coord.max()
    coord_half_span = (coord_max - coord_min) * 0.5
    coord_mid = (coord_max + coord_min) * 0.5
    lo_t = coord_mid + coord_half_span * lo
    hi_t = coord_mid + coord_half_span * hi
    return (coord >= lo_t) & (coord <= hi_t)


def filter_candidates(
    *,
    cfg: Any,
    rot_base: torch.Tensor,
    grasp_pos_base: torch.Tensor,
    obj_pos_base: torch.Tensor | None,
    metadata: Any,
    object_quat: torch.Tensor,
    log_fn: Any,
    grasp_pos_obj: torch.Tensor | None = None,
) -> torch.Tensor:
    n_total = grasp_pos_base.shape[0]
    chain = resolve_profile_chain(cfg=cfg, metadata=metadata, object_quat=object_quat)

    if not chain:
        spec = {
            "filter_x_dir": cfg.filter_x_dir,
            "filter_y_dir": cfg.filter_y_dir,
            "filter_z_dir": cfg.filter_z_dir,
            "grasp_region": getattr(cfg, "grasp_region", None),
            "direction_to_obj": cfg.direction_to_obj,
        }
        keep = apply_filter_spec(
            rot_base=rot_base,
            grasp_pos_base=grasp_pos_base,
            obj_pos_base=obj_pos_base,
            spec=spec,
            grasp_pos_obj=grasp_pos_obj,
        )
        return torch.nonzero(keep, as_tuple=False).squeeze(-1)

    force_horizontal = should_force_horizontal_close(metadata, object_quat)

    last_keep = torch.ones(n_total, dtype=torch.bool, device=grasp_pos_base.device)
    for idx, profile in enumerate(chain):
        spec = merge_resolved_profile(cfg, profile)
        spec["grasp_region"] = getattr(cfg, "grasp_region", None)
        if force_horizontal and spec.get("filter_y_dir") is None:
            spec["filter_y_dir"] = horizontal_close_filter_y_dir(idx)
        last_keep = apply_filter_spec(
            rot_base=rot_base,
            grasp_pos_base=grasp_pos_base,
            obj_pos_base=obj_pos_base,
            spec=spec,
            grasp_pos_obj=grasp_pos_obj,
        )
        if int(last_keep.sum().item()) > 0:
            return torch.nonzero(last_keep, as_tuple=False).squeeze(-1)

    return torch.nonzero(last_keep, as_tuple=False).squeeze(-1)
