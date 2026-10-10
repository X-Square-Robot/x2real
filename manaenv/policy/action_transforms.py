"""YAML-driven action adapter runtime.

Each transform converts a policy response dict into an action-chunk tensor
of shape ``(action_horizon, action_dim)``.  Selection is driven by the
``command_space`` field in the action-space YAML.

All response key names, slot layouts, and EE frame names are read from
the ``action_config`` dict at construction time — zero hardcoded literals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from manaenv.utils.debug_print import manaprint
from manaenv.utils.math_utils import (
    axis_angle_from_quat_tensor,
    euler_xyz_to_quat_wxyz,
    quat_conjugate_tensor,
    quat_mul_tensor,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)
from manaenv.utils.tensor_utils import as_torch, normalize_env_ids, upsample_temporal_rows_numpy


class ActionTransform(ABC):
    """Base class for action-space transforms."""

    def __init__(self, action_horizon: int, action_dim: int, **kwargs: Any):
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    @abstractmethod
    def __call__(
        self,
        response: dict[str, Any],
        env: Any,
        device: torch.device | str,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        """Convert *response* dict into ``(action_horizon, action_dim)`` tensor."""
        ...

    def reset(self, env_ids: Any = None) -> None:  # noqa: B027
        pass


def _pad_chunk(chunk: torch.Tensor, horizon: int, total: int) -> torch.Tensor:
    """Repeat the last valid row to fill remaining rows of the chunk."""
    if horizon < total:
        chunk[horizon:] = chunk[horizon - 1]
    return chunk


def _parse_response_keys(action_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract response_keys list from action_config, with fallback to action_keys."""
    rkeys = action_config.get("response_keys")
    if rkeys:
        return rkeys
    akeys = action_config.get("action_keys", [])
    if not akeys:
        return []
    dim = action_config.get("action_dim", 14)
    per_key = dim // max(len(akeys), 1)
    result = []
    for i, k in enumerate(akeys):
        result.append({"name": k, "slots": [i * per_key, (i + 1) * per_key]})
    return result


def _env_index(env_idx: int | None) -> int:
    return 0 if env_idx is None else int(env_idx)


def _get_env_origin(env: Any, env_idx: int, device: torch.device | str | None = None) -> torch.Tensor:
    scene = getattr(env, "scene", None)
    origins = getattr(scene, "env_origins", None)
    if origins is None:
        return torch.zeros(3, dtype=torch.float32, device=device)
    return origins[env_idx].detach().to(device=device, dtype=torch.float32)


# ------------------------------------------------------------------
# FollowPosConcatTransform
# ------------------------------------------------------------------


def _should_skip_first_frame(action_config: dict[str, Any]) -> bool:
    """Return True when the YAML declares first_frame != prediction_only.

    wall_x desktop mode prepends the current state as frame 0 of the
    action trajectory.  When the profile says ``prediction_only`` the
    consumer must skip that echo frame.
    """
    ts = action_config.get("trajectory_semantics") or {}
    return ts.get("first_frame") == "prediction_only"


class FollowPosConcatTransform(ActionTransform):
    """Concatenate multiple response arrays into a flat action chunk.

    Response key names and slot ranges are read from ``action_config``.
    No env access required.
    """

    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        action_config: dict[str, Any] | None = None,
        interpolation_multiplier: int = 1,
        **kwargs: Any,
    ):
        super().__init__(action_horizon, action_dim, **kwargs)
        cfg = action_config or {}
        self._response_keys: list[dict[str, Any]] = _parse_response_keys(cfg)
        self._skip_first: bool = _should_skip_first_frame(cfg)
        self._interpolation_multiplier: int = max(1, int(interpolation_multiplier))

    def __call__(
        self,
        response: dict[str, Any],
        env: Any,
        device: torch.device | str,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        arrays = []
        for rk in self._response_keys:
            name = rk["name"]
            val = response.get(name)
            if val is None:
                manaprint(f"WARNING: [FollowPosConcat] Missing key '{name}': {list(response.keys())}")
                return None
            arr = np.asarray(val, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if self._skip_first and arr.shape[0] > 1:
                arr = arr[1:]
            arrays.append(arr)

        min_horizon = min(a.shape[0] for a in arrays)
        horizon = min(min_horizon, self.action_horizon)
        if horizon <= 0:
            return None

        cropped = [a[:horizon] for a in arrays]
        if self._interpolation_multiplier > 1:
            cropped = [upsample_temporal_rows_numpy(a, self._interpolation_multiplier) for a in cropped]
            horizon = int(cropped[0].shape[0])

        combined = np.concatenate(cropped, axis=-1)
        out_rows = max(horizon, self.action_horizon)
        chunk = torch.zeros((out_rows, self.action_dim), dtype=torch.float32, device=device)
        actual_dim = min(combined.shape[-1], self.action_dim)
        chunk[:horizon, :actual_dim] = torch.as_tensor(combined[:, :actual_dim], dtype=torch.float32, device=device)
        if horizon < self.action_horizon:
            return _pad_chunk(chunk, horizon, self.action_horizon)
        return chunk[:horizon]


# ------------------------------------------------------------------
# Dual-arm EE transforms (shared base + delta / absolute-IK variants)
# ------------------------------------------------------------------


def _parse_slot_layout(action_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract per-key slot layout from action_config.

    Returns a dict keyed by response key name, each containing
    ``pos``, ``rot``, ``gripper`` index ranges.
    """
    return action_config.get("slot_layout", {})


class _DualArmEETransformBase(ActionTransform):
    """Shared parsing for dual-arm EE wire protocol responses.

    Handles response key validation, numpy array reshaping, first-frame
    skipping, horizon cropping, temporal interpolation, and slot layout
    resolution.  Subclasses implement ``__call__`` to convert the parsed
    arrays into sim-ready action chunks.

    Wire protocol input is 7-D per arm: ``[x, y, z, roll, pitch, yaw, gripper]``.
    """

    _WIRE_POS = slice(0, 3)
    _WIRE_EULER = slice(3, 6)
    _WIRE_GRIPPER_IDX = 6

    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        action_config: dict[str, Any] | None = None,
        interpolation_multiplier: int = 1,
        **kwargs: Any,
    ):
        super().__init__(action_horizon, action_dim, **kwargs)
        self._interpolation_multiplier: int = max(1, int(interpolation_multiplier))

        cfg = action_config or {}
        self._response_keys: list[dict[str, Any]] = _parse_response_keys(cfg)
        ee_frames = cfg.get("ee_frames") or {}
        self._left_ee_frame_key: str = ee_frames.get("left", "")
        self._right_ee_frame_key: str = ee_frames.get("right", "")
        self._slot_layout: dict[str, dict[str, Any]] = _parse_slot_layout(cfg)
        self._skip_first: bool = _should_skip_first_frame(cfg)

    def _parse_dual_arm_arrays(
        self, response: dict[str, Any], label: str = "DualArmEE"
    ) -> tuple[np.ndarray, np.ndarray, int, str, str] | None:
        """Validate and parse dual-arm wire response into numpy arrays.

        Returns ``(follow1, follow2, horizon, left_key, right_key)``
        or ``None`` when the response is invalid or empty.
        """
        if len(self._response_keys) < 2:
            manaprint(f"WARNING: [{label}] Need at least 2 response_keys for dual-arm")
            return None

        left_key = self._response_keys[0]["name"]
        right_key = self._response_keys[1]["name"]
        follow1 = response.get(left_key)
        follow2 = response.get(right_key)
        if follow1 is None or follow2 is None:
            manaprint(f"WARNING: [{label}] Missing keys: {list(response.keys())}")
            return None

        follow1 = np.asarray(follow1, dtype=np.float32)
        follow2 = np.asarray(follow2, dtype=np.float32)
        if follow1.ndim == 1:
            follow1 = follow1.reshape(1, -1)
        if follow2.ndim == 1:
            follow2 = follow2.reshape(1, -1)

        if self._skip_first:
            if follow1.shape[0] > 1:
                follow1 = follow1[1:]
            if follow2.shape[0] > 1:
                follow2 = follow2[1:]

        horizon = min(follow1.shape[0], follow2.shape[0], self.action_horizon)
        if horizon <= 0:
            return None

        follow1 = follow1[:horizon]
        follow2 = follow2[:horizon]
        if self._interpolation_multiplier > 1:
            follow1 = upsample_temporal_rows_numpy(follow1, self._interpolation_multiplier)
            follow2 = upsample_temporal_rows_numpy(follow2, self._interpolation_multiplier)
            horizon = int(follow1.shape[0])
            if follow2.shape[0] != horizon:
                return None

        return follow1, follow2, horizon, left_key, right_key

    def _get_slot_spec(self, key_name: str) -> dict[str, Any]:
        """Get slot spec for a response key from ``slot_layout``.

        Raises :class:`ValueError` if the key is not present — every
        dual-arm EE action YAML must declare an explicit ``slot_layout``.
        """
        if key_name in self._slot_layout:
            return self._slot_layout[key_name]
        raise ValueError(
            f"slot_layout missing for response key '{key_name}'. " "Define slot_layout in the action-space YAML."
        )


class EEAbsoluteToDeltaTransform(_DualArmEETransformBase):
    """Convert absolute EE poses from wire protocol to delta actions for the sim.

    The remote server (wall_x) returns ``follow*_pos`` as absolute 7-D
    EE poses ``[x, y, z, roll, pitch, yaw, gripper]`` in env-local world
    coordinates. This transform computes per-step deltas against the current
    sim pose in the same frame so the relative IK controller can consume them.

    Gripper values are passed through unchanged as raw joint targets.

    Response key names, EE frame scene keys, and slot layout are all
    read from ``action_config``.
    """

    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        action_config: dict[str, Any] | None = None,
        pos_gain: float = 1.0,
        rot_gain: float = 1.0,
        interpolation_multiplier: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            action_horizon,
            action_dim,
            action_config=action_config,
            interpolation_multiplier=interpolation_multiplier,
            **kwargs,
        )
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain
        self._default_ee_poses: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._stale_after_reset: set[int] = set()

    def _get_current_ee_poses(
        self,
        env: Any,
        env_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = _env_index(env_idx)
        left = env.scene[self._left_ee_frame_key]
        right = env.scene[self._right_ee_frame_key]
        left_pos_w = as_torch(left.data.target_pos_w)
        left_quat_w = as_torch(left.data.target_quat_w)
        right_pos_w = as_torch(right.data.target_pos_w)
        right_quat_w = as_torch(right.data.target_quat_w)
        origin = _get_env_origin(env, idx, device=left_pos_w.device)
        left_pos = left_pos_w[idx, 0, :].detach().clone() - origin
        # IsaacLab runtime pose buffers are XYZW; this transform's wire and
        # action contract is explicitly WXYZ.  Convert at this boundary before
        # composing delta rotations with the WXYZ wire target.
        left_quat = quat_xyzw_to_wxyz(left_quat_w[idx, 0, :].detach().clone())
        right_pos = right_pos_w[idx, 0, :].detach().clone() - origin
        right_quat = quat_xyzw_to_wxyz(right_quat_w[idx, 0, :].detach().clone())
        return left_pos, left_quat, right_pos, right_quat

    def __call__(
        self,
        response: dict[str, Any],
        env: Any,
        device: torch.device | str,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        parsed = self._parse_dual_arm_arrays(response, label="EEAbsToDelta")
        if parsed is None:
            return None
        follow1, follow2, horizon, left_key, right_key = parsed

        idx = _env_index(env_idx)
        default_poses = self._default_ee_poses.get(idx)
        if idx in self._stale_after_reset and default_poses is not None:
            cur_lp, cur_lq, cur_rp, cur_rq = (t.to(device) for t in default_poses)
            self._stale_after_reset.discard(idx)
        else:
            cur_lp, cur_lq, cur_rp, cur_rq = self._get_current_ee_poses(env, env_idx=idx)
            if idx not in self._default_ee_poses:
                self._default_ee_poses[idx] = (cur_lp.clone(), cur_lq.clone(), cur_rp.clone(), cur_rq.clone())
        prev_lp, prev_rp = cur_lp.clone(), cur_rp.clone()
        prev_lq, prev_rq = cur_lq.clone(), cur_rq.clone()

        left_slot = self._get_slot_spec(left_key)
        right_slot = self._get_slot_spec(right_key)
        l_pos_s = left_slot["pos"]
        l_rot_s = left_slot["rot"]
        l_grip_s = left_slot["gripper"]
        r_pos_s = right_slot["pos"]
        r_rot_s = right_slot["rot"]
        r_grip_s = right_slot["gripper"]

        out_rows = max(horizon, self.action_horizon)
        chunk = torch.zeros((out_rows, self.action_dim), dtype=torch.float32, device=device)

        g = self._WIRE_GRIPPER_IDX
        for i in range(horizon):
            tgt_lp = torch.as_tensor(follow1[i, self._WIRE_POS], dtype=torch.float32, device=device)
            tgt_rp = torch.as_tensor(follow2[i, self._WIRE_POS], dtype=torch.float32, device=device)

            tgt_lq = torch.as_tensor(
                euler_xyz_to_quat_wxyz(np.asarray([follow1[i, self._WIRE_EULER]], dtype=np.float32))[0],
                dtype=torch.float32,
                device=device,
            )
            tgt_rq = torch.as_tensor(
                euler_xyz_to_quat_wxyz(np.asarray([follow2[i, self._WIRE_EULER]], dtype=np.float32))[0],
                dtype=torch.float32,
                device=device,
            )

            ref_lp = cur_lp if i == 0 else prev_lp
            ref_rp = cur_rp if i == 0 else prev_rp
            ref_lq = cur_lq if i == 0 else prev_lq
            ref_rq = cur_rq if i == 0 else prev_rq

            l_dpos = (tgt_lp - ref_lp) * self.pos_gain
            r_dpos = (tgt_rp - ref_rp) * self.pos_gain
            l_drot = axis_angle_from_quat_tensor(
                quat_mul_tensor(tgt_lq.unsqueeze(0), quat_conjugate_tensor(ref_lq.unsqueeze(0)))
            )[0]
            l_drot = l_drot * self.rot_gain
            r_drot = axis_angle_from_quat_tensor(
                quat_mul_tensor(tgt_rq.unsqueeze(0), quat_conjugate_tensor(ref_rq.unsqueeze(0)))
            )[0]
            r_drot = r_drot * self.rot_gain

            chunk[i, l_pos_s[0] : l_pos_s[1]] = l_dpos
            chunk[i, l_rot_s[0] : l_rot_s[1]] = l_drot
            chunk[i, l_grip_s[0]] = float(follow1[i, g])
            chunk[i, r_pos_s[0] : r_pos_s[1]] = r_dpos
            chunk[i, r_rot_s[0] : r_rot_s[1]] = r_drot
            chunk[i, r_grip_s[0]] = float(follow2[i, g])

            prev_lp, prev_rp = tgt_lp, tgt_rp
            prev_lq, prev_rq = tgt_lq, tgt_rq

        if horizon < self.action_horizon:
            return _pad_chunk(chunk, horizon, self.action_horizon)
        return chunk[:horizon]

    def reset(self, env_ids: Any = None) -> None:
        ids = normalize_env_ids(env_ids)
        if ids is None:
            self._default_ee_poses.clear()
            self._stale_after_reset.clear()
            return
        for idx in ids:
            if idx in self._default_ee_poses:
                self._stale_after_reset.add(idx)


# ------------------------------------------------------------------
# EEAbsoluteIKTransform
# ------------------------------------------------------------------


class EEAbsoluteIKTransform(_DualArmEETransformBase):
    """Wire-protocol EE pose chunks -> absolute root-frame IK targets.

    Wire input per arm is 7-D ``[x, y, z, roll, pitch, yaw, gripper]`` in the
    robot ``base_link`` (root) frame. Output writes
    ``[pos(3) + quat_xyzw(4) + grip(1)] * 2`` into the configured arm/gripper
    slots in the same root frame, ready for IsaacLab 3.0
    ``DifferentialInverseKinematicsAction`` with ``use_relative_mode=False``.
    Extra mobile-base action slots remain zero
    unless another transform fills them.

    Two wire pose references (``action_config['wire_pose_reference']``; see
    :class:`~manaenv.policy.policy_embodiment.PolicyEmbodimentCFG`):

    * ``"base_link"`` — each wire frame is already an absolute end-effector
      pose in the robot ``base_link`` (root) frame; the chunk is fed to IK
      as-is for OpenPI-style training.
    * ``"init_ee_pose"`` — each wire frame is relative to the home EE pose
      (``default_joint_pos`` configuration; historical key name kept for
      compatibility), while still expressed in the root frame. The transform
      reads the per-env home EE pose cached at reset
      (``env._left_ee_ref`` / ``env._right_ee_ref`` by default, overridable via
      ``action_config['ref_keys']``) and composes absolute IK targets:
      ``target_rotation_root = wire_relative * home_rotation_root``. This is
      the inverse of the observation/recording convention
      ``wire_relative = current_root * inverse(home_root)``. Translation is a
      root-frame offset and is therefore reconstructed by addition. Home refs
      are written at reset (not mid-follow); this transform only reads them.

    All poses are **root-local**; there is no env-world or sim-world frame
    on this path. Omitting ``wire_pose_reference`` defaults to ``"init_ee_pose"``.
    """

    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        action_config: dict[str, Any] | None = None,
        interpolation_multiplier: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            action_horizon,
            action_dim,
            action_config=action_config,
            interpolation_multiplier=interpolation_multiplier,
            **kwargs,
        )
        cfg = action_config or {}
        self._robot_entity: str = cfg.get("robot_entity", "robot")
        wire_pose_reference = self._resolve_wire_pose_reference(cfg)
        if wire_pose_reference not in ("base_link", "init_ee_pose"):
            raise ValueError(
                "EEAbsoluteIKTransform: unsupported "
                f"wire_pose_reference={wire_pose_reference!r}; expected 'base_link' or 'init_ee_pose'"
            )
        self._wire_pose_reference: str = wire_pose_reference
        # Per-arm env attr names for the cached home EE pose in root frame.
        # Reset hooks write these while joints are at ``default_joint_pos``;
        # this transform only reads them for ``init_ee_pose`` chunks.
        ref_keys = cfg.get("ref_keys") or {}
        self._left_ref_key: str = str(ref_keys.get("left", "_left_ee_ref"))
        self._right_ref_key: str = str(ref_keys.get("right", "_right_ee_ref"))

    @classmethod
    def _resolve_wire_pose_reference(cls, cfg: dict[str, Any]) -> str:
        raw = cfg.get("wire_pose_reference")
        if raw is None:
            return "init_ee_pose"
        reference = str(raw).strip().lower()
        if reference not in ("base_link", "init_ee_pose"):
            raise ValueError(
                "EEAbsoluteIKTransform: unsupported "
                f"wire_pose_reference={raw!r}; expected 'init_ee_pose' or 'base_link'."
            )
        return reference

    def __call__(
        self,
        response: dict[str, Any],
        env: Any,
        device: torch.device | str,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        parsed = self._parse_dual_arm_arrays(response, label="EEAbsIK")
        if parsed is None:
            return None
        follow1, follow2, horizon, left_key, right_key = parsed

        idx = _env_index(env_idx)

        left_slot = self._get_slot_spec(left_key)
        right_slot = self._get_slot_spec(right_key)
        l_pos_s = left_slot["pos"]
        l_quat_s = left_slot["quat"]
        l_grip_s = left_slot["gripper"]
        r_pos_s = right_slot["pos"]
        r_quat_s = right_slot["quat"]
        r_grip_s = right_slot["gripper"]

        out_rows = max(horizon, self.action_horizon)
        chunk = torch.zeros((out_rows, self.action_dim), dtype=torch.float32, device=device)

        # Home EE reference: read the per-env home EE pose captured at reset
        # (root frame). Both arms get a single (pos, quat) tuple sliced by env
        # idx; reset hooks refresh these while joints are at default, so reading
        # once outside the per-frame loop is safe.
        if self._wire_pose_reference == "init_ee_pose":
            l_init_pos, l_init_quat = self._read_init_ee_root(env, self._left_ref_key, idx, device)
            r_init_pos, r_init_quat = self._read_init_ee_root(env, self._right_ref_key, idx, device)
        else:
            l_init_pos = l_init_quat = r_init_pos = r_init_quat = None

        if self._wire_pose_reference == "init_ee_pose":
            assert l_init_pos is not None and l_init_quat is not None
            assert r_init_pos is not None and r_init_quat is not None

        g = self._WIRE_GRIPPER_IDX
        for i in range(horizon):
            tgt_lp = torch.as_tensor(follow1[i, self._WIRE_POS], dtype=torch.float32, device=device)
            tgt_rp = torch.as_tensor(follow2[i, self._WIRE_POS], dtype=torch.float32, device=device)
            tgt_lq = torch.as_tensor(
                euler_xyz_to_quat_wxyz(np.asarray([follow1[i, self._WIRE_EULER]], dtype=np.float32))[0],
                dtype=torch.float32,
                device=device,
            )
            tgt_rq = torch.as_tensor(
                euler_xyz_to_quat_wxyz(np.asarray([follow2[i, self._WIRE_EULER]], dtype=np.float32))[0],
                dtype=torch.float32,
                device=device,
            )

            if self._wire_pose_reference == "init_ee_pose":
                assert l_init_pos is not None and l_init_quat is not None
                assert r_init_pos is not None and r_init_quat is not None
                # The wire rotation is home-relative in the root frame:
                # ``q_rel = q_target * inverse(q_home)``. Reconstruct with
                # ``q_target = q_rel * q_home``. This ordering is important
                # for non-commuting, non-identity home orientations and
                # matches embodiment_obs/action_recording exactly.
                l_pos_b = l_init_pos + tgt_lp
                r_pos_b = r_init_pos + tgt_rp
                l_quat_b_wxyz = quat_mul_tensor(tgt_lq, l_init_quat)
                r_quat_b_wxyz = quat_mul_tensor(tgt_rq, r_init_quat)
            else:
                # Base-link reference: chunk is already a root-frame IK target.
                l_pos_b = tgt_lp
                l_quat_b_wxyz = tgt_lq
                r_pos_b = tgt_rp
                r_quat_b_wxyz = tgt_rq

            chunk[i, l_pos_s[0] : l_pos_s[1]] = l_pos_b
            # Wall-X Euler decoding and init-reference composition above use
            # the legacy WXYZ helpers. IsaacLab 3.0 absolute DiffIK commands
            # consume XYZW, so swizzle exactly once at the action boundary.
            chunk[i, l_quat_s[0] : l_quat_s[1]] = quat_wxyz_to_xyzw(l_quat_b_wxyz)
            chunk[i, l_grip_s[0]] = float(follow1[i, g])

            chunk[i, r_pos_s[0] : r_pos_s[1]] = r_pos_b
            chunk[i, r_quat_s[0] : r_quat_s[1]] = quat_wxyz_to_xyzw(r_quat_b_wxyz)
            chunk[i, r_grip_s[0]] = float(follow2[i, g])

        if horizon < self.action_horizon:
            return _pad_chunk(chunk, horizon, self.action_horizon)
        return chunk[:horizon]

    def _read_init_ee_root(
        self, env: Any, ref_key: str, env_idx: int, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one env's init EE pose, expressed in the robot root frame."""
        ref = getattr(env, ref_key, None)
        if not isinstance(ref, tuple) or len(ref) != 2:
            raise RuntimeError(
                f"EEAbsoluteIKTransform: env attr {ref_key!r} missing or malformed. "
                "Ensure the embodiment's PolicyCfg includes the matching "
                "init-EE-pose-relative WallX ObsTerm so the init EE root pose is captured."
            )
        init_pos = ref[0][env_idx].to(device=device, dtype=torch.float32)
        # ObsTerm caches IsaacLab's XYZW root-frame pose. Convert to WXYZ only
        # for the legacy Wall-X composition helpers; __call__ converts the
        # resulting target back to XYZW at the IsaacLab action boundary.
        init_quat_xyzw = ref[1][env_idx].to(device=device, dtype=torch.float32)
        init_quat = quat_xyzw_to_wxyz(init_quat_xyzw)
        return init_pos, init_quat

    def reset(self, env_ids: Any = None) -> None:
        pass


# ------------------------------------------------------------------
# RawActionTransform
# ------------------------------------------------------------------


class RawActionTransform(ActionTransform):
    """Passthrough for servers that return a flat action array.

    The response key name is read from ``action_config``.
    """

    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        action_config: dict[str, Any] | None = None,
        interpolation_multiplier: int = 1,
        **kwargs: Any,
    ):
        super().__init__(action_horizon, action_dim, **kwargs)
        cfg = action_config or {}
        rkeys = _parse_response_keys(cfg)
        self._response_key: str = rkeys[0]["name"] if rkeys else "actions"
        self._interpolation_multiplier: int = max(1, int(interpolation_multiplier))

    def __call__(
        self,
        response: dict[str, Any],
        env: Any,
        device: torch.device | str,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        raw = response.get(self._response_key)
        if raw is None:
            manaprint(f"WARNING: [RawAction] '{self._response_key}' key missing: {list(response.keys())}")
            return None

        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        horizon = min(arr.shape[0], self.action_horizon)
        segment = arr[:horizon]
        if self._interpolation_multiplier > 1:
            segment = upsample_temporal_rows_numpy(segment, self._interpolation_multiplier)
            horizon = int(segment.shape[0])
        out_rows = max(horizon, self.action_horizon)
        chunk = torch.zeros((out_rows, self.action_dim), dtype=torch.float32, device=device)
        actual_dim = min(segment.shape[-1], self.action_dim)
        chunk[:horizon, :actual_dim] = torch.as_tensor(segment[:, :actual_dim], dtype=torch.float32, device=device)
        if horizon < self.action_horizon:
            return _pad_chunk(chunk, horizon, self.action_horizon)
        return chunk[:horizon]


# ------------------------------------------------------------------
# WholeBodyEEAbsoluteIKTransform — EX001 whole-body inference
# ------------------------------------------------------------------


class WholeBodyEEAbsoluteIKTransform(EEAbsoluteIKTransform):
    """Dual-arm absolute EE IK + passthrough base velocity / lift / head.

    For EX001 whole-body wallx inference.  Arms are handled identically to
    :class:`EEAbsoluteIKTransform` (euler → quat, init-EE-pose or base_link
    reference).  Extra response keys beyond the first two (follow1/follow2)
    are read from ``slot_layout`` and written directly into the action chunk:

    * ``velocity_decomposed_odom`` — converted from ``[vx, vy, vyaw]`` to
      differential-drive wheel velocities ``[left, right]`` before writing.
    * ``lift`` / ``head_pos`` — passed through unchanged.

    Diff-drive parameters (``wheel_radius``, ``wheel_track_width``) are read
    from ``action_config["metadata"]["diff_drive"]``.
    """

    _DIFF_DRIVE_TRANSFORM = "diff_drive"

    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        action_config: dict[str, Any] | None = None,
        interpolation_multiplier: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            action_horizon,
            action_dim,
            action_config=action_config,
            interpolation_multiplier=interpolation_multiplier,
            **kwargs,
        )
        cfg = action_config or {}

        self._extra_keys: list[dict[str, Any]] = []
        for rk in self._response_keys[2:]:
            name = rk["name"]
            spec = self._slot_layout.get(name)
            if spec is None:
                continue
            slots = spec.get("slots")
            if slots is None or len(slots) != 2:
                continue
            self._extra_keys.append(
                {
                    "name": name,
                    "slots": slots,
                    "transform": spec.get("transform"),
                }
            )

        metadata = cfg.get("metadata") or {}
        diff_drive = metadata.get("diff_drive") or {}
        self._wheel_radius: float = float(diff_drive.get("wheel_radius", 0.078))
        self._wheel_track_width: float = float(diff_drive.get("wheel_track_width", 0.48))

    def __call__(
        self,
        response: dict[str, Any],
        env: Any,
        device: torch.device | str,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        chunk = super().__call__(response, env, device, env_idx)
        if chunk is None or not self._extra_keys:
            return chunk

        T = chunk.shape[0]
        for spec in self._extra_keys:
            name = spec["name"]
            val = response.get(name)
            if val is None:
                continue
            arr = np.asarray(val, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if self._skip_first and arr.shape[0] > 1:
                arr = arr[1:]
            arr = arr[: self.action_horizon]
            if self._interpolation_multiplier > 1:
                arr = upsample_temporal_rows_numpy(arr, self._interpolation_multiplier)

            if spec.get("transform") == self._DIFF_DRIVE_TRANSFORM:
                arr = self._odom_to_wheel_vel(arr)

            slots = spec["slots"]
            out_dim = slots[1] - slots[0]
            horizon = min(arr.shape[0], T)
            src_dim = min(arr.shape[-1], out_dim)
            chunk[:horizon, slots[0] : slots[0] + src_dim] = torch.as_tensor(
                arr[:horizon, :src_dim],
                dtype=torch.float32,
                device=device,
            )
            if horizon < T:
                chunk[horizon:, slots[0] : slots[1]] = chunk[horizon - 1 : horizon, slots[0] : slots[1]]

        return chunk

    def _odom_to_wheel_vel(self, odom: np.ndarray) -> np.ndarray:
        """``[vx, vy, vyaw]`` → ``[left_wheel_vel, right_wheel_vel]``."""
        vx = odom[:, 0]
        vyaw = odom[:, 2] if odom.shape[-1] > 2 else np.zeros_like(vx)
        left = (vx - 0.5 * vyaw * self._wheel_track_width) / self._wheel_radius
        right = (vx + 0.5 * vyaw * self._wheel_track_width) / self._wheel_radius
        return np.stack([left, right], axis=-1)


# ------------------------------------------------------------------
# Registry — keyed by command_space values from action YAML
# ------------------------------------------------------------------

_COMMAND_SPACE_TO_SYMBOL: dict[str, str] = {
    "eef_delta": "EEAbsoluteToDeltaTransform",
    "ee_absolute": "EEAbsoluteToDeltaTransform",
    "eef_concat": "FollowPosConcatTransform",
    "follow_pos_concat": "FollowPosConcatTransform",
    "joint_raw": "RawActionTransform",
    "raw_flat": "RawActionTransform",
    "raw": "RawActionTransform",
    "eef_abs": "EEAbsoluteIKTransform",
    "ee_abs_ik": "EEAbsoluteIKTransform",
    "eef_abs_wholebody": "WholeBodyEEAbsoluteIKTransform",
}


def create_action_transform(
    action_config: dict[str, Any],
    interpolation_multiplier: int = 1,
    **kwargs: Any,
) -> ActionTransform:
    """Instantiate a transform from the action YAML config dict.

    The ``command_space`` field selects the transform class (resolved through
    ``global_registry``).  The entire
    ``action_config`` is forwarded so each transform can read its own
    response keys, slot layout, and EE frame names.

    ``transform_params`` from the action YAML are merged with ``kwargs``,
    with YAML values taking precedence over caller-supplied defaults.

    ``interpolation_multiplier`` is forwarded so transforms can upsample
    wire trajectories before delta conversion (``ManaRemotePolicy``).
    """
    from manaenv.pipeline.registry.bootstrap import bootstrap_discovery
    from manaenv.pipeline.registry.global_provider import global_registry

    command_space = action_config.get("command_space", "ee_absolute")
    symbol = _COMMAND_SPACE_TO_SYMBOL.get(command_space)
    if symbol is None:
        raise ValueError(f"Unknown command_space '{command_space}'. Available: {list(_COMMAND_SPACE_TO_SYMBOL)}")
    bootstrap_discovery()
    cls = global_registry.get_impl(symbol)
    if not callable(cls):
        raise RuntimeError(f"Action transform registry symbol {symbol!r} is not callable")
    yaml_params = action_config.get("transform_params") or {}
    merged_kwargs = dict(kwargs)
    merged_kwargs.update(yaml_params)
    merged_kwargs.pop("interpolation_multiplier", None)
    imult = max(1, int(interpolation_multiplier))

    return cls(
        action_horizon=action_config.get("action_horizon", merged_kwargs.pop("action_horizon", 16)),
        action_dim=action_config.get("action_dim", merged_kwargs.pop("action_dim", 14)),
        action_config=action_config,
        interpolation_multiplier=imult,
        **merged_kwargs,
    )
