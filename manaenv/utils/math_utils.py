"""Centralised math helpers — quaternions, rotations, coordinate transforms.

Convention Reference (IsaacLab / Isaac Sim)
===========================================
https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_conventions.html

Coordinate system
    X-Forward, Y-Left, Z-Up  (same as ROS REP-103 for body frames)

Quaternion order
    **XYZW** is the ManaEnv default for new code, matching IsaacLab 3.0 tensor
    math and ROS quaternion messages. Legacy WXYZ code must use
    ``quat_wxyz_to_xyzw`` / ``quat_xyzw_to_wxyz`` at the boundary.

Rotation order
    ZYX intrinsic Tait-Bryan  (= XYZ extrinsic)
    ``euler_xyz_to_quat_wxyz`` and ``quat_xyzw_to_euler_xyz`` follow this.

New functions in this module should default to **XYZW**. Legacy helpers that
keep WXYZ behavior are named explicitly with ``wxyz``. Boundary swizzles preserve
tensor inputs and return numpy float32 for array-like inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import numpy as np
import torch

# ---------------------------------------------------------------------------
#  Quaternion order swizzle (boundary conversion)
# ---------------------------------------------------------------------------


@overload
def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray: ...  # noqa: E704 - Black overload stub style


@overload
def quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor: ...  # noqa: E704 - Black overload stub style


@overload
def quat_wxyz_to_xyzw(q: Sequence[float]) -> np.ndarray: ...  # noqa: E704 - Black overload stub style


def quat_wxyz_to_xyzw(q: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray | torch.Tensor:
    """``(W, X, Y, Z)`` → ``(X, Y, Z, W)`` for external boundaries."""
    if isinstance(q, torch.Tensor):
        return q[..., [1, 2, 3, 0]]
    q = np.asarray(q, dtype=np.float32)
    if q.ndim == 1:
        return q[[1, 2, 3, 0]]
    return q[..., [1, 2, 3, 0]]


@overload
def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray: ...  # noqa: E704 - Black overload stub style


@overload
def quat_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor: ...  # noqa: E704 - Black overload stub style


@overload
def quat_xyzw_to_wxyz(q: Sequence[float]) -> np.ndarray: ...  # noqa: E704 - Black overload stub style


def quat_xyzw_to_wxyz(q: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray | torch.Tensor:
    """``(X, Y, Z, W)`` → ``(W, X, Y, Z)`` from external boundaries."""
    if isinstance(q, torch.Tensor):
        return q[..., [3, 0, 1, 2]]
    q = np.asarray(q, dtype=np.float32)
    if q.ndim == 1:
        return q[[3, 0, 1, 2]]
    return q[..., [3, 0, 1, 2]]


def quat_wxyz_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize WXYZ quaternion(s) and return float32."""
    quat = np.asarray(q, dtype=np.float64)
    squeeze = quat.ndim == 1
    if squeeze:
        quat = quat[None, :]
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (*, 4), got {quat.shape}")

    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("Cannot normalize a zero-length quaternion")

    normalized = (quat / norms).astype(np.float32)
    return normalized[0] if squeeze else normalized


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert WXYZ quaternion(s) to rotation matrix / matrices."""
    quat = quat_wxyz_normalize(q).astype(np.float64)
    squeeze = quat.ndim == 1
    if squeeze:
        quat = quat[None, :]

    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    mats = np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    mats = mats.astype(np.float32)
    return mats[0] if squeeze else mats


# ---------------------------------------------------------------------------
#  Euler ↔ Quaternion
# ---------------------------------------------------------------------------


def euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles (radians) to WXYZ quaternion.

    Args:
        euler_xyz: ``(3,)`` or ``(N, 3)`` array.

    Returns:
        WXYZ quaternion with the same leading dims.
    """
    euler = np.asarray(euler_xyz, dtype=np.float32)
    squeeze = euler.ndim == 1
    if squeeze:
        euler = euler[None, :]
    if euler.shape[-1] != 3:
        raise ValueError(f"Expected euler shape (*, 3), got {euler.shape}")

    half = euler.astype(np.float64) * 0.5
    cx, cy, cz = np.cos(half[:, 0]), np.cos(half[:, 1]), np.cos(half[:, 2])
    sx, sy, sz = np.sin(half[:, 0]), np.sin(half[:, 1]), np.sin(half[:, 2])

    quat = np.stack(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        axis=1,
    ).astype(np.float32)
    return quat[0] if squeeze else quat


def euler_xyz_to_quat_xyzw(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles (radians) to ManaEnv-default XYZW quaternion."""
    return quat_wxyz_to_xyzw(euler_xyz_to_quat_wxyz(euler_xyz))


def quat_xyzw_to_euler_xyz(quat_xyzw: np.ndarray, unwrap: bool = True) -> np.ndarray:
    """Convert XYZW quaternions to XYZ Euler angles (radians).

    Args:
        quat_xyzw: ``(4,)`` or ``(N, 4)`` — **XYZW** order.
        unwrap: If *True*, apply ``np.unwrap`` along the time axis.

    Returns:
        ``(3,)`` or ``(N, 3)`` Euler angles.
    """
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    squeeze = quat.ndim == 1
    if squeeze:
        quat = quat[None, :]
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (*, 4), got {quat.shape}")

    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp.astype(np.float64), cosr_cosp.astype(np.float64)).astype(np.float32)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(
        np.abs(sinp) >= 1.0,
        np.sign(sinp) * (np.pi / 2.0),
        np.arcsin(np.clip(sinp, -1.0, 1.0)),
    ).astype(np.float32)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp.astype(np.float64), cosy_cosp.astype(np.float64)).astype(np.float32)

    euler = np.stack([roll, pitch, yaw], axis=1).astype(np.float32)
    if unwrap:
        euler = np.unwrap(euler, axis=0).astype(np.float32)
    out = euler[0] if squeeze else euler
    return out


# ---------------------------------------------------------------------------
#  Quaternion algebra (single-vector, numpy, WXYZ)
# ---------------------------------------------------------------------------


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate / inverse (unit quaternion). WXYZ convention."""
    q = np.asarray(q, dtype=np.float32)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two WXYZ quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_conjugate_tensor(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for WXYZ torch tensors."""
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_conjugate_xyzw_tensor(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for ManaEnv-default XYZW torch tensors."""
    out = q.clone()
    out[..., :3] = -out[..., :3]
    return out


def quat_mul_tensor(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for WXYZ torch tensors."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_mul_xyzw_tensor(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for ManaEnv-default XYZW torch tensors."""
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


def euler_xyz_from_quat_tensor(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert WXYZ torch quaternions to XYZ Euler angles."""
    w, x, y, z = q.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1.0,
        torch.sign(sinp) * (torch.pi / 2.0),
        torch.asin(torch.clamp(sinp, -1.0, 1.0)),
    )
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def euler_xyz_from_quat_xyzw_tensor(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ManaEnv-default XYZW torch quaternions to XYZ Euler angles."""
    x, y, z, w = q.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1.0,
        torch.sign(sinp) * (torch.pi / 2.0),
        torch.asin(torch.clamp(sinp, -1.0, 1.0)),
    )
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def axis_angle_from_quat_tensor(q: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert WXYZ torch quaternions to axis-angle vectors.

    Inputs are normalized internally; ``eps`` clamps near-zero quaternion and axis norms.
    """
    quat = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=eps)
    xyz = quat[..., 1:4]
    xyz_norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, quat[..., 0:1])
    axis = xyz / torch.clamp(xyz_norm, min=eps)
    small_angle = xyz_norm < eps
    return torch.where(small_angle, 2.0 * xyz, axis * angle)


def axis_angle_from_quat_xyzw_tensor(q: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert ManaEnv-default XYZW torch quaternions to axis-angle vectors.

    Inputs are normalized internally; ``eps`` clamps near-zero quaternion and axis norms.
    """
    quat = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=eps)
    xyz = quat[..., :3]
    xyz_norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, quat[..., 3:4])
    axis = xyz / torch.clamp(xyz_norm, min=eps)
    small_angle = xyz_norm < eps
    return torch.where(small_angle, 2.0 * xyz, axis * angle)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a WXYZ quaternion."""
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    q_v = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    result = quat_multiply(quat_multiply(q, q_v), quat_inverse(q))
    return result[1:4]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by the inverse of a WXYZ quaternion."""
    q_inv = quat_inverse(q)
    q_v = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    result = quat_multiply(quat_multiply(q_inv, q_v), q)
    return result[1:4]


def quat_inverse_xyzw(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate / inverse for a unit XYZW quaternion."""
    return np.asarray(quat_wxyz_to_xyzw(quat_inverse(quat_xyzw_to_wxyz(q))), dtype=np.float32)


def quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two XYZW quaternions."""
    product_wxyz = quat_multiply(quat_xyzw_to_wxyz(q1), quat_xyzw_to_wxyz(q2))
    return np.asarray(quat_wxyz_to_xyzw(product_wxyz), dtype=np.float32)


def quat_rotate_inverse_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by the inverse of an XYZW quaternion."""
    return quat_rotate_inverse(np.asarray(quat_xyzw_to_wxyz(q)), v)


# ---------------------------------------------------------------------------
#  Quaternion ↔ yaw (scalar, numpy)
# ---------------------------------------------------------------------------


def quat_wxyz_to_yaw(q: np.ndarray) -> float:
    """Extract yaw (rotation around Z) from a single WXYZ quaternion.

    ``yaw = atan2(2*(w*z + x*y), 1 - 2*(y² + z²))``
    """
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quat_xyzw_to_yaw(q: np.ndarray) -> float:
    """Extract yaw (rotation around Z) from a single XYZW quaternion."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    """Build a WXYZ quaternion from a single yaw angle (rotation around Z)."""
    half = float(yaw) * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


# ---------------------------------------------------------------------------
#  2-D rotation
# ---------------------------------------------------------------------------


def rotate_2d(vec: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 3-vector around the Z-axis by *angle* (radians).

    Only the XY components are rotated; Z passes through unchanged.
    """
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return np.array(
        [vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c, vec[2]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
#  Pose helpers
# ---------------------------------------------------------------------------


def compute_relative_pose(
    pos: np.ndarray,
    quat: np.ndarray,
    origin_pos: np.ndarray,
    origin_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pose relative to *origin* (numpy, WXYZ quaternions)."""
    pos_diff = np.asarray(pos, dtype=np.float32) - np.asarray(origin_pos, dtype=np.float32)
    rel_pos = quat_rotate_inverse(origin_quat, pos_diff)
    rel_quat = quat_multiply(quat_inverse(origin_quat), np.asarray(quat, dtype=np.float32))
    return rel_pos, rel_quat


def compute_relative_pose_xyzw(
    pos: np.ndarray,
    quat_xyzw: np.ndarray,
    origin_pos: np.ndarray,
    origin_quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a pose relative to *origin* using runtime-default XYZW quaternions."""
    pos_diff = np.asarray(pos, dtype=np.float32) - np.asarray(origin_pos, dtype=np.float32)
    rel_pos = quat_rotate_inverse_xyzw(origin_quat_xyzw, pos_diff)
    rel_quat = quat_multiply_xyzw(
        quat_inverse_xyzw(origin_quat_xyzw),
        np.asarray(quat_xyzw, dtype=np.float32),
    )
    return rel_pos, rel_quat


# ---------------------------------------------------------------------------
#  Scalar helpers
# ---------------------------------------------------------------------------


def normalize_angle_radians(angle_radians: float) -> float:
    """Normalize an angle to the range [-pi, pi) radians."""

    return float((float(angle_radians) + np.pi) % (2.0 * np.pi) - np.pi)


def meters_to_mm(meters: float) -> float:
    """Convert metres to millimetres."""
    return meters * 1000.0


def map_gripper_state(
    val: float,
    open_threshold: float = 3.3,
    close_threshold: float = 1.0,
) -> float:
    """Map a continuous gripper value to a three-state command.

    Returns ``1.0`` (open) if *val* >= *open_threshold*,
    ``-1.0`` (close) if *val* <= *close_threshold*, else ``0.0``.
    """
    if val >= open_threshold:
        return 1.0
    if val <= close_threshold:
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Torch tensor rotation utilities (used by X2Robot batched FK / IK)
# ---------------------------------------------------------------------------


def quat_wxyz_to_rotmat_tensor(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) batch to rotation matrix. (..., 4) -> (..., 3, 3)."""

    w, x, y, z = q.unbind(-1)
    n = (w * w + x * x + y * y + z * z).clamp(min=1e-10).sqrt()
    w, x, y, z = w / n, x / n, y / n, z / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return R.reshape(*q.shape[:-1], 3, 3)


def skew_batch(v: torch.Tensor) -> torch.Tensor:
    """Batch skew-symmetric matrix. (N, 3) -> (N, 3, 3)."""

    N = v.shape[0]
    zero = torch.zeros(N, device=v.device, dtype=v.dtype)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return torch.stack(
        [
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ],
        dim=-1,
    ).reshape(N, 3, 3)


def rotmat_from_axis_angle_tensor(angle: torch.Tensor, axis: int) -> torch.Tensor:
    """Rotation matrix about axis 0=X/1=Y/2=Z. angle: (N,) -> (N, 3, 3)."""

    c, s = torch.cos(angle), torch.sin(angle)
    o, z = torch.ones_like(angle), torch.zeros_like(angle)
    if axis == 0:
        rows = [o, z, z, z, c, -s, z, s, c]
    elif axis == 1:
        rows = [c, z, s, z, o, z, -s, z, c]
    else:
        rows = [c, -s, z, s, c, z, z, z, o]
    return torch.stack(rows, dim=-1).reshape(-1, 3, 3)


def so3_log_tensor(R: torch.Tensor) -> torch.Tensor:
    """SO(3) logarithm: (N, 3, 3) rotation matrix -> (N, 3) axis-angle.

    Handles near-zero and near-π edge cases.
    """

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    skew = (R - R.transpose(-2, -1)) * 0.5
    omega = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1)

    sin_theta = torch.sin(theta)
    near_zero = theta.abs() < 1e-6
    near_pi = (theta - torch.pi).abs() < 1e-6

    safe_sin = sin_theta.clamp(min=1e-10)
    scale = torch.where(near_zero, torch.ones_like(theta), theta / safe_sin)
    result = omega * scale.unsqueeze(-1)

    if near_pi.any():
        diag = torch.stack([R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], dim=-1)
        axis = torch.sqrt(((diag + 1.0) * 0.5).clamp(min=0.0))
        sign_xy = torch.sign(R[:, 0, 1] + R[:, 1, 0])
        sign_xz = torch.sign(R[:, 0, 2] + R[:, 2, 0])
        # Fallback to +1 when off-diagonal is zero (axis aligned with coordinate axis)
        sign_xy = torch.where(sign_xy == 0, torch.ones_like(sign_xy), sign_xy)
        sign_xz = torch.where(sign_xz == 0, torch.ones_like(sign_xz), sign_xz)
        axis[:, 1] *= sign_xy
        axis[:, 2] *= sign_xz
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        result = torch.where(near_pi.unsqueeze(-1), axis * theta.unsqueeze(-1), result)

    return result
