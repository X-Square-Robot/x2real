"""Runtime contract for the EX001 whole-body policy wire.

The gateway publishes a model-neutral 20-D action chunk. Only this module
converts it at the simulator boundary into the EX001 21-D environment layout.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

WIRE_SCHEMA = "manaenv_ex001_wholebody_policy_wire_20d_v1"
WIRE_ACTION_DIM = 20
BASE_REPRESENTATION = "vx_vy_yaw_rate"
EE_ROTATION_REPRESENTATION = "euler_xyz"
EE_REFERENCE = "init_ee_pose"

ENV_ACTION_LAYOUT = "manaenv_ex001_wholebody_env_21d_v1"
ENV_ACTION_DIM = 21
ENV_BASE_REPRESENTATION = "wheel_radps_lr"
DIFF_DRIVE_WHEEL_RADIUS_M = 0.078
DIFF_DRIVE_TRACK_WIDTH_M = 0.48
BASE_VY_ABS_MAX = 1e-3

WIRE_ACTION_FIELDS = (
    "left_ee_delta_x_m",
    "left_ee_delta_y_m",
    "left_ee_delta_z_m",
    "left_ee_delta_roll_rad",
    "left_ee_delta_pitch_rad",
    "left_ee_delta_yaw_rad",
    "left_gripper_normalized",
    "right_ee_delta_x_m",
    "right_ee_delta_y_m",
    "right_ee_delta_z_m",
    "right_ee_delta_roll_rad",
    "right_ee_delta_pitch_rad",
    "right_ee_delta_yaw_rad",
    "right_gripper_normalized",
    "base_vx_mps",
    "base_vy_mps",
    "base_yaw_rate_radps",
    "lift_position_m",
    "head_yaw_position_rad",
    "head_pitch_position_rad",
)
ENV_ACTION_FIELDS = (
    "left_ee_x_m",
    "left_ee_y_m",
    "left_ee_z_m",
    "left_ee_qx",
    "left_ee_qy",
    "left_ee_qz",
    "left_ee_qw",
    "left_gripper_position",
    "right_ee_x_m",
    "right_ee_y_m",
    "right_ee_z_m",
    "right_ee_qx",
    "right_ee_qy",
    "right_ee_qz",
    "right_ee_qw",
    "right_gripper_position",
    "left_wheel_radps",
    "right_wheel_radps",
    "lift_position_m",
    "head_yaw_position_rad",
    "head_pitch_position_rad",
)


class WholeBodyPolicyWireError(ValueError):
    """Invalid EX001 whole-body policy-wire data."""


def euler_xyz_to_quat_xyzw(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert Euler angles while keeping the server-only import surface Torch-free."""
    from manaenv.utils.math_utils import euler_xyz_to_quat_xyzw as convert

    return convert(euler_xyz)


def quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternions through the simulator-only math implementation."""
    from manaenv.utils.math_utils import quat_multiply_xyzw as multiply

    return multiply(q1, q2)


def validate_wire_envelope(payload: Mapping[str, Any]) -> np.ndarray:
    """Fail closed unless ``payload`` is the complete supported gateway wire."""
    if not isinstance(payload, Mapping):
        raise WholeBodyPolicyWireError("policy-wire response must be a mapping")
    required_metadata = {
        "wire_schema": WIRE_SCHEMA,
        "action_dim": WIRE_ACTION_DIM,
        "base_representation": BASE_REPRESENTATION,
        "ee_rotation_representation": EE_ROTATION_REPRESENTATION,
        "ee_reference": EE_REFERENCE,
    }
    for key, expected in required_metadata.items():
        actual = payload.get(key)
        if actual != expected:
            raise WholeBodyPolicyWireError(f"policy-wire {key} must be {expected!r}, got {actual!r}")
    if "action_chunk" not in payload:
        raise WholeBodyPolicyWireError("policy-wire response requires action_chunk")
    forbidden = {"follow1_pos", "follow2_pos", "velocity_decomposed_odom", "wheel_command"}
    present = sorted(forbidden.intersection(payload))
    if present:
        raise WholeBodyPolicyWireError(f"policy-wire response must not expose native or wheel fields: {present}")
    return validate_action_chunk(payload["action_chunk"])


def validate_action_chunk(action_chunk: Any) -> np.ndarray:
    """Validate the public action chunk shape and numerical domain."""
    array = np.asarray(action_chunk, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != WIRE_ACTION_DIM:
        raise WholeBodyPolicyWireError(
            f"policy-wire action_chunk must have shape [T,{WIRE_ACTION_DIM}] with T >= 1, got {array.shape!r}"
        )
    if not np.isfinite(array).all():
        raise WholeBodyPolicyWireError("policy-wire action_chunk contains NaN/Inf")
    return np.ascontiguousarray(array)


def policy_wire_to_env21(
    row: Sequence[float] | np.ndarray,
    *,
    left_init_ee_pos: Sequence[float] | np.ndarray,
    left_init_ee_quat_xyzw: Sequence[float] | np.ndarray,
    right_init_ee_pos: Sequence[float] | np.ndarray,
    right_init_ee_quat_xyzw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convert one policy-wire row to the simulator-only EX001 env layout."""
    wire = np.asarray(row, dtype=np.float32).reshape(-1)
    if wire.shape != (WIRE_ACTION_DIM,) or not np.isfinite(wire).all():
        raise WholeBodyPolicyWireError(f"policy-wire row must be finite shape ({WIRE_ACTION_DIM},), got {wire.shape}")
    left_pos = _reference(left_init_ee_pos, "left_init_ee_pos", 3)
    left_quat = _reference(left_init_ee_quat_xyzw, "left_init_ee_quat_xyzw", 4)
    right_pos = _reference(right_init_ee_pos, "right_init_ee_pos", 3)
    right_quat = _reference(right_init_ee_quat_xyzw, "right_init_ee_quat_xyzw", 4)

    env = np.zeros((ENV_ACTION_DIM,), dtype=np.float32)
    env[0:3] = left_pos + wire[0:3]
    env[3:7] = quat_multiply_xyzw(euler_xyz_to_quat_xyzw(wire[3:6]), left_quat)
    env[7] = wire[6]
    env[8:11] = right_pos + wire[7:10]
    env[11:15] = quat_multiply_xyzw(euler_xyz_to_quat_xyzw(wire[10:13]), right_quat)
    env[15] = wire[13]
    env[16:18] = odom_to_wheel_vel(wire[14:17])
    env[18] = wire[17]
    env[19:21] = wire[18:20]
    return env


def odom_to_wheel_vel(odom_vx_vy_yaw: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert ``[vx, vy, yaw_rate]`` to left/right wheel radians per second."""
    base = np.asarray(odom_vx_vy_yaw, dtype=np.float32).reshape(-1)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise WholeBodyPolicyWireError(f"base command must be finite shape (3,), got {base.shape}")
    if abs(float(base[1])) > BASE_VY_ABS_MAX:
        raise WholeBodyPolicyWireError(f"base.vy={float(base[1])!r} exceeds |vy|<={BASE_VY_ABS_MAX}")
    vx, _, yaw = base
    return np.asarray(
        [
            (vx - 0.5 * yaw * DIFF_DRIVE_TRACK_WIDTH_M) / DIFF_DRIVE_WHEEL_RADIUS_M,
            (vx + 0.5 * yaw * DIFF_DRIVE_TRACK_WIDTH_M) / DIFF_DRIVE_WHEEL_RADIUS_M,
        ],
        dtype=np.float32,
    )


def _reference(value: Sequence[float] | np.ndarray, name: str, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise WholeBodyPolicyWireError(f"{name} must be finite shape ({width},), got {array.shape}")
    return array
