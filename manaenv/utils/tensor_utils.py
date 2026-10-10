"""Tensor / array conversion and image serialization helpers.

Centralises the many ``to_numpy`` variants and shape-wrangling utilities
that were previously scattered across ros_utils, motion_utils, formatter
utils, and individual policy files.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  warp ProxyArray → torch (IsaacLab 3.0 boundary unwrap)
# ---------------------------------------------------------------------------


def as_torch(value: Any) -> Any:
    """Return the zero-copy Torch view of ProxyArray and raw Warp arrays.

    IsaacLab 3.0 exposes ``robot.data.*`` buffers as warp ``ProxyArray``, whose
    ``.torch`` property is a cached zero-copy tensor view. PhysX tensor views can
    also return raw ``wp.array`` objects; Torch consumes those through Warp's
    array interface without a host copy. Plain values pass through unchanged.
    """
    unwrapped = getattr(value, "torch", value)
    if unwrapped is not value:
        return unwrapped
    if type(value).__module__.startswith("warp.") and hasattr(value, "__dlpack__"):
        return torch.as_tensor(value)
    return value


# ---------------------------------------------------------------------------
#  tensor → numpy
# ---------------------------------------------------------------------------


def select_env_item(val: Any, env_idx: int | None = None) -> Any:
    """Select one vectorized-env row while preserving unbatched values."""
    if val is None or env_idx is None:
        return val
    val = as_torch(val)
    idx = int(env_idx)
    if isinstance(val, torch.Tensor):
        if val.dim() > 1 and int(val.shape[0]) > idx:
            return val[idx]
        return val
    arr = np.asarray(val)
    if arr.ndim > 1 and int(arr.shape[0]) > idx:
        return arr[idx]
    return val


def normalize_env_ids(env_ids: Any = None) -> set[int] | None:
    """Normalize scalar, tensor, numpy, or iterable env ids into a set of ints."""
    if env_ids is None:
        return None
    if isinstance(env_ids, int):
        return {int(env_ids)}
    if hasattr(env_ids, "dim") and env_ids.dim() == 0:
        return {int(env_ids.item())}
    ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
    if isinstance(ids, int):
        return {int(ids)}
    return {int(i) for i in ids}


def tensor_to_numpy(val: Any) -> np.ndarray | None:
    """Convert a tensor / array to a 1-D float32 numpy array.

    Squeezes the leading batch dimension when present so that
    ``(1, D)`` becomes ``(D,)``.  Returns *None* if *val* is *None*.
    """
    if val is None:
        return None
    val = to_numpy_keep_shape(val)
    if val is None:
        return None
    val = np.asarray(val, dtype=np.float32)
    if val.ndim > 1:
        val = val[0]
    return val


def to_numpy_flat(val: Any) -> np.ndarray:
    """Convert tensor / array-like to a **flattened** numpy array."""
    converted = to_numpy_keep_shape(val)
    return np.asarray(converted).flatten()


def to_numpy_keep_shape(val: Any) -> np.ndarray | None:
    """Convert tensor / array-like to numpy **without** shape changes.

    Returns *None* when *val* is *None*.
    """
    if val is None:
        return None
    val = as_torch(val)
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu().numpy()
    elif hasattr(val, "numpy") and callable(val.numpy):
        val = val.numpy()
    return np.asarray(val)


# ---------------------------------------------------------------------------
#  Tensor shape helpers
# ---------------------------------------------------------------------------


def squeeze_if_single(t: torch.Tensor) -> torch.Tensor:
    """``(1, …)`` → ``squeeze(0)``; otherwise return as-is."""
    return t.squeeze(0) if t.shape[0] == 1 else t


def as_quat_tensor(q: Any, device: Any = None) -> torch.Tensor:
    """Coerce an already-ordered quaternion to float32 tensor ``(…, 4)``.

    ManaEnv runtime callers use XYZW. This helper deliberately preserves the
    supplied order; explicit WXYZ boundaries must swizzle before calling it.

    A 1-D input is unsqueezed to ``(1, 4)``.
    """
    if isinstance(q, torch.Tensor):
        t = q
    else:
        t = torch.as_tensor(np.asarray(q, dtype=np.float32))
    if device is not None:
        t = t.to(device=device)
    t = t.to(dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


def as_vec3_tensor(v: Any, device: Any = None) -> torch.Tensor:
    """Coerce 3-vector (np / tensor) to float32 tensor ``(…, 3)``.

    A 1-D input is unsqueezed to ``(1, 3)``.
    """
    if isinstance(v, torch.Tensor):
        t = v
    else:
        t = torch.as_tensor(np.asarray(v, dtype=np.float32))
    if device is not None:
        t = t.to(device=device)
    t = t.to(dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


def resample_temporal_linear(chunk: torch.Tensor, out_len: int) -> torch.Tensor:
    """Linearly upsample rows along time (dim 0).

    Used for policy action chunks: element-wise linear interpolation between
    consecutive rows. Not rotation-aware (no SLERP); suitable for deltas or
    joint targets where per-dim linear blending is acceptable.

    Args:
        chunk: ``(T, D)`` tensor.
        out_len: Target row count; must be ``>= T``. If ``out_len == T``, returns ``chunk``.

    Returns:
        Tensor of shape ``(out_len, D)``.
    """
    if chunk.dim() != 2:
        raise ValueError(f"resample_temporal_linear expected (T, D), got {tuple(chunk.shape)}")
    t, _d = chunk.shape
    if out_len < t:
        raise ValueError(f"out_len ({out_len}) must be >= T ({t})")
    if out_len == t:
        return chunk
    if t == 1:
        return chunk.expand(out_len, -1).clone()
    # (T, D) -> (1, D, T): single fused 1D linear resize; faster than double gather on GPU.
    x = chunk.contiguous().transpose(0, 1).unsqueeze(0)
    y = F.interpolate(x, size=out_len, mode="linear", align_corners=True)
    return y.squeeze(0).transpose(0, 1).contiguous()


def _align_corners_query_positions(t_rows: int, out_len: int) -> np.ndarray:
    """Continuous time indices matching :func:`resample_temporal_linear` (``align_corners=True``)."""
    if t_rows <= 1:
        return np.zeros(out_len, dtype=np.float64)
    return np.linspace(0.0, float(t_rows - 1), out_len, dtype=np.float64)


def upsample_xyz_euler_gripper_trajectory_numpy(actions_np: np.ndarray, out_len: int) -> np.ndarray:
    """Upsample ``(T, 7)`` or ``(T, 14)`` rows: ``[x,y,z, r,p,y, gripper]`` per arm.

    Args:
        actions_np: ``(T, 7)`` or ``(T, 14)`` trajectory.
        out_len: Target row count (must be ``>= T``).

    Position (0-2) and gripper (last column of each 7-block) use :func:`numpy.interp`.
    Orientation (3-5) uses quaternion NLERP: Euler ``xyz`` -> scipy quat (xyzw), per-component
    interp, normalize, back to Euler. Matches common ``end_pose`` wire layouts (dual arm = two blocks).
    """
    from scipy.spatial.transform import Rotation as ScipyRotation

    a = np.asarray(actions_np, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    t_rows, d = int(a.shape[0]), int(a.shape[1])
    if t_rows <= 0 or out_len <= t_rows:
        return a
    if d == 7:
        return _upsample_single_xyz_euler_gripper_block(a, out_len, ScipyRotation)
    if d == 14:
        left = _upsample_single_xyz_euler_gripper_block(a[:, :7], out_len, ScipyRotation)
        right = _upsample_single_xyz_euler_gripper_block(a[:, 7:14], out_len, ScipyRotation)
        return np.concatenate([left, right], axis=1)
    raise ValueError(f"upsample_xyz_euler_gripper_trajectory_numpy expected D=7 or 14, got {d}")


def _upsample_single_xyz_euler_gripper_block(
    block: np.ndarray,
    out_len: int,
    rot_cls: Any,
) -> np.ndarray:
    """Upsample one ``(T, 7)`` arm trajectory."""
    t_rows = int(block.shape[0])
    if t_rows == 1:
        return np.repeat(block, out_len, axis=0)
    xp = _align_corners_query_positions(t_rows, out_len)
    x = np.arange(t_rows, dtype=np.float64)
    out = np.zeros((out_len, 7), dtype=np.float32)
    for i in range(3):
        out[:, i] = np.interp(xp, x, block[:, i].astype(np.float64)).astype(np.float32)
    out[:, 6] = np.interp(xp, x, block[:, -1].astype(np.float64)).astype(np.float32)
    quats_xyzw = rot_cls.from_euler("xyz", block[:, 3:6], degrees=False).as_quat().astype(np.float64)
    for i in range(1, t_rows):
        if float(np.dot(quats_xyzw[i], quats_xyzw[i - 1])) < 0.0:
            quats_xyzw[i] *= -1.0
    iq = np.zeros((out_len, 4), dtype=np.float64)
    for i in range(4):
        iq[:, i] = np.interp(xp, x, quats_xyzw[:, i])
    norms = np.linalg.norm(iq, axis=1, keepdims=True)
    iq /= np.maximum(norms, 1e-8)
    out[:, 3:6] = rot_cls.from_quat(iq).as_euler("xyz", degrees=False).astype(np.float32)
    return out


def upsample_xyz_quat_xyzw_gripper_trajectory_numpy(actions_np: np.ndarray, out_len: int) -> np.ndarray:
    """Upsample ``(T, 8)`` or ``(T, 16)`` rows: ``[x,y,z, qx,qy,qz,qw, gripper]`` per arm.

    This is the decoded IsaacLab dual-arm EE action after
    :class:`~manaenv.policy.action_transforms.EEAbsoluteIKTransform`. Position
    and gripper use :func:`numpy.interp`. Orientation uses quaternion NLERP:
    hemisphere-align consecutive XYZW keyframes, interpolate components, then
    renormalize. Component lerp of ``q`` and ``-q`` otherwise crosses the origin
    and produces illegal ``|q|`` such as 0.12.
    """
    a = np.asarray(actions_np, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    t_rows, d = int(a.shape[0]), int(a.shape[1])
    if t_rows <= 0 or out_len <= t_rows:
        return a
    if d == 8:
        return _upsample_single_xyz_quat_xyzw_gripper_block(a, out_len)
    if d == 16:
        left = _upsample_single_xyz_quat_xyzw_gripper_block(a[:, :8], out_len)
        right = _upsample_single_xyz_quat_xyzw_gripper_block(a[:, 8:16], out_len)
        return np.concatenate([left, right], axis=1)
    raise ValueError(f"upsample_xyz_quat_xyzw_gripper_trajectory_numpy expected D=8 or 16, got {d}")


def _upsample_single_xyz_quat_xyzw_gripper_block(block: np.ndarray, out_len: int) -> np.ndarray:
    """Upsample one ``(T, 8)`` arm trajectory with XYZW NLERP."""
    t_rows = int(block.shape[0])
    if t_rows == 1:
        return np.repeat(block, out_len, axis=0)
    xp = _align_corners_query_positions(t_rows, out_len)
    x = np.arange(t_rows, dtype=np.float64)
    out = np.zeros((out_len, 8), dtype=np.float32)
    for i in range(3):
        out[:, i] = np.interp(xp, x, block[:, i].astype(np.float64)).astype(np.float32)
    out[:, 7] = np.interp(xp, x, block[:, 7].astype(np.float64)).astype(np.float32)
    quats = np.asarray(block[:, 3:7], dtype=np.float64).copy()
    norms0 = np.linalg.norm(quats, axis=1, keepdims=True)
    quats /= np.maximum(norms0, 1e-8)
    for i in range(1, t_rows):
        if float(np.dot(quats[i], quats[i - 1])) < 0.0:
            quats[i] *= -1.0
    iq = np.zeros((out_len, 4), dtype=np.float64)
    for i in range(4):
        iq[:, i] = np.interp(xp, x, quats[:, i])
    norms = np.linalg.norm(iq, axis=1, keepdims=True)
    iq /= np.maximum(norms, 1e-8)
    out[:, 3:7] = iq.astype(np.float32)
    return out


def upsample_temporal_rows_numpy(arr: np.ndarray, mult: float) -> np.ndarray:
    """Upsample rows along time (axis 0).

    ``mult`` may be a non-integer (e.g. 1.5); the target row count is
    ``round(rows * mult)``.

    For last dimension **7** or **14** (``[x,y,z, r,p,y, gripper]`` per arm), uses
    :func:`upsample_xyz_euler_gripper_trajectory_numpy`. For **8** or **16**
    (``[x,y,z, qx,qy,qz,qw, gripper]`` per arm, IsaacLab XYZW), uses
    :func:`upsample_xyz_quat_xyzw_gripper_trajectory_numpy`. Otherwise matches
    :func:`resample_temporal_linear` element-wise.
    """
    a = np.asarray(arr, dtype=np.float32)
    if mult <= 1.0:
        return a
    if a.ndim == 1:
        a = a.reshape(1, -1)
    rows, d = int(a.shape[0]), int(a.shape[1])
    if rows <= 0:
        return a
    out_len = max(rows, round(rows * mult))
    if d in (7, 14):
        return upsample_xyz_euler_gripper_trajectory_numpy(a, out_len)
    if d in (8, 16):
        return upsample_xyz_quat_xyzw_gripper_trajectory_numpy(a, out_len)
    t = torch.as_tensor(a, dtype=torch.float32)
    out = resample_temporal_linear(t, out_len)
    return out.detach().cpu().numpy()


# ---------------------------------------------------------------------------
#  Image encoding
# ---------------------------------------------------------------------------


def tensor_to_jpeg_b64(rgb: Any, quality: int = 95) -> str | None:
    """Encode an image tensor / array to a base64 JPEG string.

    Handles batch dim removal, float-to-uint8 scaling, and RGB-to-BGR
    conversion required by ``cv2.imencode``.

    Args:
        rgb: Image data (tensor, ndarray, or None).
        quality: JPEG compression quality 0-100.
    """
    import cv2

    if rgb is None:
        return None
    rgb = to_numpy_keep_shape(rgb)
    if rgb is None:
        return None
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0).clip(0, 255)
        rgb = rgb.astype(np.uint8)
    if len(rgb.shape) == 3 and rgb.shape[2] == 3:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return None
    return base64.b64encode(encoded).decode("utf-8")
