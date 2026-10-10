"""Object-pose constrained place candidate generation."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from manaenv.utils.math_utils import quat_conjugate_xyzw_tensor, quat_mul_xyzw_tensor

from .pose_filters import filter_place_ee_rotations


def _math_utils() -> Any:
    return importlib.import_module("isaaclab.utils.math")


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-9)


def _unit_vector(value: Any, *, dtype: torch.dtype, device: torch.device, field_name: str) -> torch.Tensor:
    vec = torch.as_tensor(value, dtype=dtype, device=device).reshape(3)
    norm = torch.linalg.norm(vec)
    if float(norm) <= 1e-9:
        raise ValueError(f"{field_name} must be a non-zero 3-vector; got {value!r}.")
    return vec / norm


def _basis_from_axis(axis: torch.Tensor) -> torch.Tensor:
    z_axis = axis.reshape(3)
    ref = torch.tensor([0.0, 0.0, 1.0], dtype=z_axis.dtype, device=z_axis.device)
    if float(torch.abs(torch.dot(ref, z_axis))) > 0.95:
        ref = torch.tensor([1.0, 0.0, 0.0], dtype=z_axis.dtype, device=z_axis.device)
    x_axis = torch.linalg.cross(ref, z_axis)
    x_axis = x_axis / torch.linalg.norm(x_axis).clamp(min=1e-9)
    y_axis = torch.linalg.cross(z_axis, x_axis)
    y_axis = y_axis / torch.linalg.norm(y_axis).clamp(min=1e-9)
    return torch.stack([x_axis, y_axis, z_axis], dim=-1)


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor | float) -> torch.Tensor:
    axis = axis / torch.linalg.norm(axis).clamp(min=1e-9)
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(float(angle), dtype=axis.dtype, device=axis.device)
    x, y, z = axis
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    skew = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ],
        dim=0,
    )
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return eye + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


def _project_onto_axis_plane(vec: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    return vec - torch.dot(vec, axis) * axis


def _signed_angle_around_axis(source: torch.Tensor, target: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    cross = torch.linalg.cross(source, target)
    return torch.atan2(torch.dot(axis, cross), torch.dot(source, target))


def _yaw_center_side_vector(
    cfg: Any,
    *,
    root_pos: torch.Tensor,
    observation: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mode = str(getattr(cfg, "yaw_center_mode", "arm_base_side") or "").strip().lower()
    if mode == "arm_base_side":
        base_pos = observation.get("robot_base_pos_world")
        if base_pos is None:
            raise ValueError("object_pose_constraint.yaw_center_mode='arm_base_side' requires robot_base_pos_world.")
        side = torch.as_tensor(base_pos, dtype=dtype, device=device).reshape(3)
        env_origin = observation.get("env_origin")
        if env_origin is not None:
            side = side - torch.as_tensor(env_origin, dtype=dtype, device=device).reshape(3)
    elif mode == "current_ee_side":
        side = torch.as_tensor(observation["current_ee_pos"], dtype=dtype, device=device).reshape(3)
    else:
        raise ValueError(
            "Unsupported object_pose_constraint.yaw_center_mode="
            f"{mode!r}; expected 'arm_base_side' or 'current_ee_side'."
        )
    return side - root_pos.reshape(3)


def _object_to_ee_transform(
    observation: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_ee_quat = observation.get("current_ee_quat")
    if current_ee_quat is None:
        raise ValueError("PlaceSkill object_pose_constraint requires current_ee_quat in planner context.")
    obj_pos = torch.as_tensor(observation["object_root_pos"], dtype=dtype, device=device).reshape(3)
    obj_quat = _normalize_quat(
        torch.as_tensor(observation["object_root_quat"], dtype=dtype, device=device).reshape(1, 4)
    )
    ee_pos = torch.as_tensor(observation["current_ee_pos"], dtype=dtype, device=device).reshape(3)
    ee_quat = _normalize_quat(torch.as_tensor(current_ee_quat, dtype=dtype, device=device).reshape(1, 4))

    m = _math_utils()
    obj_quat_inv = quat_conjugate_xyzw_tensor(obj_quat)
    rel_pos = m.quat_apply(obj_quat_inv, (ee_pos - obj_pos).reshape(1, 3)).reshape(3)
    rel_quat = _normalize_quat(quat_mul_xyzw_tensor(obj_quat_inv, ee_quat)).reshape(4)
    return rel_pos, rel_quat


def _compose_object_to_ee(
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    object_to_ee_pos: torch.Tensor,
    object_to_ee_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = _math_utils()
    n = int(object_quat.reshape(-1, 4).shape[0])
    parent_pos = object_pos.reshape(n, 3)
    parent_quat = _normalize_quat(object_quat.reshape(n, 4))
    rel_pos = object_to_ee_pos.reshape(1, 3).expand(n, -1)
    rel_quat = object_to_ee_quat.reshape(1, 4).expand(n, -1)
    ee_pos = parent_pos + m.quat_apply(parent_quat, rel_pos)
    ee_quat = _normalize_quat(quat_mul_xyzw_tensor(parent_quat, rel_quat))
    return ee_pos, ee_quat


def _object_orientation_candidates(
    *,
    cfg: Any,
    root_positions_env: torch.Tensor,
    observation: dict[str, Any],
    object_to_ee_pos: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mode = str(getattr(cfg, "rotation_mode", "") or "").strip().lower()
    if mode != "axis_parallel_free_yaw":
        raise ValueError(f"Unsupported object_pose_constraint.rotation_mode={mode!r}.")

    object_axis = _unit_vector(
        getattr(cfg, "object_axis_local", (0.0, 0.0, 1.0)),
        dtype=dtype,
        device=device,
        field_name="PlaceObjectPoseConstraintCfg.object_axis_local",
    )
    target_axis = _unit_vector(
        getattr(cfg, "target_axis_world", (0.0, 0.0, 1.0)),
        dtype=dtype,
        device=device,
        field_name="PlaceObjectPoseConstraintCfg.target_axis_world",
    )
    directions = [target_axis]
    if bool(getattr(cfg, "allow_axis_flip", False)):
        directions.append(-target_axis)

    yaw_count = max(1, int(getattr(cfg, "yaw_samples", 1)))
    yaw_span = torch.tensor(float(getattr(cfg, "yaw_span_deg", 20.0)), dtype=dtype, device=device).deg2rad()
    yaw_offsets = (
        torch.zeros(1, dtype=dtype, device=device)
        if yaw_count == 1
        else torch.linspace(-0.5 * yaw_span, 0.5 * yaw_span, yaw_count, dtype=dtype, device=device)
    )
    tilt_count = max(1, int(getattr(cfg, "tilt_samples", 1)))
    max_tilt = torch.tensor(float(getattr(cfg, "max_tilt_deg", 0.0)), dtype=dtype, device=device).deg2rad()
    tilt_angles = (
        torch.zeros(1, dtype=dtype, device=device)
        if tilt_count <= 1 or float(max_tilt) <= 0.0
        else torch.linspace(-max_tilt, max_tilt, tilt_count, dtype=dtype, device=device)
    )

    local_basis = _basis_from_axis(object_axis)
    per_target_rotations: list[torch.Tensor] = []
    rel_pos = object_to_ee_pos.reshape(3).to(dtype=dtype, device=device)
    for root_pos in root_positions_env.reshape(-1, 3):
        rotations: list[torch.Tensor] = []
        for axis_world in directions:
            world_basis = _basis_from_axis(axis_world)
            align = world_basis @ local_basis.transpose(-1, -2)
            base_offset = _project_onto_axis_plane(align @ rel_pos, axis_world)
            base_norm = torch.linalg.norm(base_offset)
            if float(base_norm) <= 1e-6:
                raise ValueError(
                    "object_pose_constraint yaw centering requires object_to_ee_pos to have a component "
                    "perpendicular to target_axis_world."
                )
            desired_side = _project_onto_axis_plane(
                _yaw_center_side_vector(cfg, root_pos=root_pos, observation=observation, dtype=dtype, device=device),
                axis_world,
            )
            desired_norm = torch.linalg.norm(desired_side)
            if float(desired_norm) <= 1e-6:
                raise ValueError(
                    "object_pose_constraint yaw centering side vector is degenerate; "
                    "check target position and yaw_center_mode."
                )
            center_yaw = _signed_angle_around_axis(base_offset / base_norm, desired_side / desired_norm, axis_world)
            tilt_x_axis = world_basis[:, 0]
            tilt_y_axis = world_basis[:, 1]
            for yaw_offset in yaw_offsets:
                yaw_rot = _axis_angle_matrix(axis_world, center_yaw + yaw_offset)
                for tilt_x in tilt_angles:
                    for tilt_y in tilt_angles:
                        tilt_rot = _axis_angle_matrix(tilt_y_axis, tilt_y) @ _axis_angle_matrix(tilt_x_axis, tilt_x)
                        rotations.append(tilt_rot @ yaw_rot @ align)
        per_target_rotations.append(torch.stack(rotations, dim=0))

    rot = torch.stack(per_target_rotations, dim=0)
    quats = _math_utils().quat_from_matrix(rot.reshape(-1, 3, 3)).to(dtype=dtype, device=device)
    return _normalize_quat(quats).reshape(rot.shape[0], rot.shape[1], 4)


def make_object_pose_constrained_candidates(
    *,
    cfg: Any,
    target_pos_world: torch.Tensor,
    observation: dict[str, Any],
    pre_place_z_offset: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = str(getattr(cfg, "root_position_source", "target_place_pos") or "").strip().lower()
    if source != "target_place_pos":
        raise ValueError(f"Unsupported object_pose_constraint.root_position_source={source!r}.")

    targets = target_pos_world.reshape(-1, 3).to(dtype=dtype, device=device)
    offset = torch.as_tensor(getattr(cfg, "root_position_offset", (0.0, 0.0, 0.0)), dtype=dtype, device=device)
    desired_root = targets + offset.reshape(1, 3)
    pre_root = desired_root + torch.tensor([0.0, 0.0, pre_place_z_offset], dtype=dtype, device=device).reshape(1, 3)

    object_to_ee_pos, object_to_ee_quat = _object_to_ee_transform(observation, dtype=dtype, device=device)
    root_quats = _object_orientation_candidates(
        cfg=cfg,
        root_positions_env=desired_root,
        observation=observation,
        object_to_ee_pos=object_to_ee_pos,
        dtype=dtype,
        device=device,
    )
    n_t = int(desired_root.shape[0])
    n_q = int(root_quats.shape[1])
    root_pos_flat = desired_root.reshape(n_t, 1, 3).expand(-1, n_q, -1).reshape(-1, 3)
    pre_root_flat = pre_root.reshape(n_t, 1, 3).expand(-1, n_q, -1).reshape(-1, 3)
    root_quat_flat = root_quats.reshape(-1, 4)

    place_ee_pos, place_ee_quat = _compose_object_to_ee(
        root_pos_flat,
        root_quat_flat,
        object_to_ee_pos,
        object_to_ee_quat,
    )
    pre_ee_pos, _ = _compose_object_to_ee(
        pre_root_flat,
        root_quat_flat,
        object_to_ee_pos,
        object_to_ee_quat,
    )
    keep = filter_place_ee_rotations(
        cfg=cfg,
        ee_pos_env=place_ee_pos,
        ee_quat_env=place_ee_quat,
        observation=observation,
        dtype=dtype,
        device=device,
    )
    if int(keep.sum().item()) == 0:
        return pre_ee_pos[:0], place_ee_pos[:0], place_ee_quat[:0]
    return pre_ee_pos[keep], place_ee_pos[keep], place_ee_quat[keep]
