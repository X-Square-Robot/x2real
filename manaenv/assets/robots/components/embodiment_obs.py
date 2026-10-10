"""Shared observation-term helpers for FrameTransformer-based embodiments.

Used by dual-arm embodiments (EX001-6R and peers) whose policy
observation groups need:

- per-arm EE pose from a ``FrameTransformer``, expressed in the articulation
  root (``base_link``) via ``subtract_frame_transforms``
- home EE pose reference (``default_joint_pos`` configuration) for relative EE
  pose observations and matching wallx ``follow*_pos`` wire rows (7-D:
  relative pos + relative Euler XYZ + gripper)
- absolute wallx ``follow*_pos`` wire rows (7-D: absolute root-local pos +
  root-local Euler XYZ + gripper) for models trained on base-link poses

All EE quantities here are **root-local**. Policy-side
:class:`~manaenv.policy.action_transforms.EEAbsoluteIKTransform` reads the
same ``env`` ref attributes when ``wire_pose_reference`` is ``init_ee_pose``
(home / default-joint EE, despite the historical key name); no extra
world->root step.

Keeping these in a single module avoids the fragile copy-paste pattern that
previously drifted between embodiment files (and silently violated the
"math/tensor utilities centralised" project rule).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from manaenv.utils.asset_utils import scene_has_asset
from manaenv.utils.math_utils import euler_xyz_from_quat_xyzw_tensor, quat_conjugate_xyzw_tensor, quat_mul_xyzw_tensor
from manaenv.utils.tensor_utils import as_torch

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg

_DEFAULT_LEFT_EE_FRAME = "left_ee_frame"
_DEFAULT_RIGHT_EE_FRAME = "right_ee_frame"
_DEFAULT_LEFT_REF_KEY = "_left_ee_ref"
_DEFAULT_RIGHT_REF_KEY = "_right_ee_ref"
_JOINTS_AT_DEFAULT_ATOL = 1e-3


def get_ee_root_local(env, ee_frame_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pos, quat_xyzw)`` for an EE frame in the robot root frame.

    Isaac Lab 3.0 returns XYZW quaternions; ManaEnv keeps that convention on
    this runtime path and converts only at explicit legacy WXYZ boundaries.
    """
    from isaaclab.utils.math import subtract_frame_transforms

    ee = env.scene[ee_frame_cfg.name]
    pos_w = as_torch(ee.data.target_pos_w)[:, 0, :]
    quat_w = as_torch(ee.data.target_quat_w)[:, 0, :]
    robot = env.scene["robot"]
    pos_b, quat_b = subtract_frame_transforms(
        as_torch(robot.data.root_pos_w),
        as_torch(robot.data.root_quat_w),
        pos_w,
        quat_w,
    )
    return pos_b, quat_b


def get_ee_body_local(env, ee_frame_cfg: SceneEntityCfg, reference_body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return EE pose in one named articulation body frame, fail-closed."""
    from isaaclab.utils.math import subtract_frame_transforms

    robot = env.scene["robot"]
    body_ids, body_names = robot.find_bodies(reference_body_name)
    if len(body_ids) != 1:
        raise RuntimeError(
            f"Expected exactly one EX001 EE reference body {reference_body_name!r}; "
            f"found {len(body_ids)}: {body_names}."
        )
    ee = env.scene[ee_frame_cfg.name]
    pos_w = as_torch(ee.data.target_pos_w)[:, 0, :]
    quat_w = as_torch(ee.data.target_quat_w)[:, 0, :]
    index = int(body_ids[0])
    return subtract_frame_transforms(
        as_torch(robot.data.body_pos_w)[:, index, :],
        as_torch(robot.data.body_quat_w)[:, index, :],
        pos_w,
        quat_w,
    )


def _home_reference_body_name(env) -> str | None:
    """Use the EX001 lift carriage when present; tabletop embodiments use root."""
    names = tuple(str(name) for name in getattr(env.scene["robot"], "body_names", ()))
    return "lift_link" if "lift_link" in names else None


def _normalize_env_ids(env, env_ids: Any) -> torch.Tensor:
    """Return int64 env-id tensor on the robot device."""
    robot = env.scene["robot"]
    joint_pos = as_torch(robot.data.joint_pos)
    device = joint_pos.device
    num_envs = int(joint_pos.shape[0])
    if env_ids is None:
        return torch.arange(num_envs, dtype=torch.int64, device=device)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.int64).reshape(-1)
    return torch.as_tensor(list(env_ids), dtype=torch.int64, device=device)


def joints_at_default(env, env_ids: Any = None, atol: float = _JOINTS_AT_DEFAULT_ATOL) -> torch.Tensor:
    """Return a boolean mask over ``env_ids`` for joints near ``default_joint_pos``."""
    robot = env.scene["robot"]
    ids = _normalize_env_ids(env, env_ids)
    joint_pos = as_torch(robot.data.joint_pos)[ids]
    default = as_torch(robot.data.default_joint_pos)[ids]
    return torch.all(torch.abs(joint_pos - default) <= atol, dim=-1)


def _ensure_ee_ref_buffers(
    env,
    key: str,
    pos: torch.Tensor,
    quat: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Allocate per-env EE ref buffers if missing; return ``(ref, needs_refresh)``."""
    refresh_key = key + "_needs_refresh"
    if not hasattr(env, key):
        setattr(env, key, (pos.clone(), quat.clone()))
        setattr(env, refresh_key, torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device))
    return getattr(env, key), getattr(env, refresh_key)


def store_home_ee_ref(
    env,
    key: str,
    ee_frame_cfg: SceneEntityCfg,
    env_ids: Any = None,
    *,
    require_at_default: bool = True,
    atol: float = _JOINTS_AT_DEFAULT_ATOL,
) -> torch.Tensor:
    """Write reset-home EE pose in the embodiment's command frame into ``env.<key>``.

    EX001 whole-body uses ``lift_link`` when it is available, so both the
    recorded home pose and its later absolute IK target composition stay
    invariant to common lift motion. Tabletop embodiments retain the legacy
    robot-root frame fallback.

    Callers must invoke this only after reset when FrameTransformer data is
    fresh. When ``require_at_default`` is True, only envs whose joints are near
    ``default_joint_pos`` are written (home configuration).

    Returns:
        Boolean mask over the requested ``env_ids`` indicating which slots were written.
    """
    ids = _normalize_env_ids(env, env_ids)
    if ids.numel() == 0:
        device = as_torch(env.scene["robot"].data.joint_pos).device
        return torch.zeros(0, dtype=torch.bool, device=device)

    reference_body = _home_reference_body_name(env)
    pos, quat = (
        get_ee_body_local(env, ee_frame_cfg, reference_body)
        if reference_body is not None
        else get_ee_root_local(env, ee_frame_cfg)
    )
    ref, needs_refresh = _ensure_ee_ref_buffers(env, key, pos, quat)

    write_mask = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
    if require_at_default:
        write_mask = joints_at_default(env, ids, atol=atol)
    if not write_mask.any():
        return write_mask

    written_ids = ids[write_mask]
    ref[0][written_ids] = pos[written_ids]
    ref[1][written_ids] = quat[written_ids]
    needs_refresh[written_ids] = False
    return write_mask


def store_home_ee_refs(
    env,
    env_ids: Any = None,
    *,
    left_frame: str = _DEFAULT_LEFT_EE_FRAME,
    right_frame: str = _DEFAULT_RIGHT_EE_FRAME,
    left_key: str = _DEFAULT_LEFT_REF_KEY,
    right_key: str = _DEFAULT_RIGHT_REF_KEY,
    require_at_default: bool = True,
    atol: float = _JOINTS_AT_DEFAULT_ATOL,
) -> bool:
    """Store left/right home EE refs for ``env_ids`` when the frames exist.

    Returns True if at least one arm ref was written for any requested env.
    """
    from isaaclab.managers import SceneEntityCfg

    wrote = False
    if scene_has_asset(env, left_frame):
        mask = store_home_ee_ref(
            env,
            left_key,
            SceneEntityCfg(left_frame),
            env_ids,
            require_at_default=require_at_default,
            atol=atol,
        )
        wrote = wrote or bool(mask.any().item())
    if scene_has_asset(env, right_frame):
        mask = store_home_ee_ref(
            env,
            right_key,
            SceneEntityCfg(right_frame),
            env_ids,
            require_at_default=require_at_default,
            atol=atol,
        )
        wrote = wrote or bool(mask.any().item())
    return wrote


def refresh_scene_and_store_home_ee_refs(env, env_ids: Any = None) -> bool:
    """Refresh FrameTransformer data, then store home EE refs for ``env_ids``.

    Safe to call from reset hooks. No-ops when EE frames are absent or joints
    are not at the default (home) configuration.
    """
    scene = getattr(env, "scene", None)
    if scene is None:
        return False
    if not scene_has_asset(env, _DEFAULT_LEFT_EE_FRAME) and not scene_has_asset(env, _DEFAULT_RIGHT_EE_FRAME):
        return False
    physics_dt = getattr(env, "physics_dt", None)
    if physics_dt is not None and hasattr(scene, "update"):
        scene.update(dt=physics_dt)
    return store_home_ee_refs(env, env_ids=env_ids)


def get_or_init_ee_ref(
    env,
    key: str,
    pos: torch.Tensor,
    quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached home EE pose (root-local); do not overwrite from motion.

    The reference is stored as an attribute on ``env`` under ``key`` so that
    the same buffer is reused across:

    * matching home-relative position / rotation observation terms within one tick;
    * the wallx ``follow*_pos`` wire payload computed at observation time;
    * the policy-side :class:`EEAbsoluteIKTransform` which reads the same
      attribute to recover absolute root-frame IK targets from an
      home-relative wire chunk (``wire_pose_reference: init_ee_pose``).

    Writes happen via :func:`store_home_ee_refs` at reset (after FrameTransformer
    refresh, while joints are at ``default_joint_pos``). This function only reads
    the cache. On first access before any store, it lazily seeds from the current
    pose once; subsequent episode resets must not replace that home ref with a
    mid-follow pose (the previous ``episode_length_buf == 1`` overwrite path).
    """
    refresh_key = key + "_needs_refresh"
    if not hasattr(env, key):
        # First-ever call fallback: prefer an explicit store_home_ee_refs at reset.
        setattr(env, key, (pos.clone(), quat.clone()))
        setattr(env, refresh_key, torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device))
    ref = getattr(env, key)
    if hasattr(env, "episode_length_buf"):
        # Mark only; explicit reset-boundary storage owns home-reference
        # updates. Keeping this branch-free avoids a per-observation CUDA sync.
        needs_refresh: torch.Tensor = getattr(env, refresh_key)
        needs_refresh |= env.episode_length_buf == 0
    return ref


_GRIPPER_JOINT_IDX_CACHE: dict[tuple[int, str], int] = {}
_JOINT_GROUP_IDX_CACHE: dict[tuple[int, tuple[str, ...]], list[int]] = {}


def gripper_joint_idx(env, joint_name: str) -> int:
    """Resolve and memoise the joint index for ``joint_name`` on ``env.scene["robot"]``.

    The cache key is namespaced by ``id(env)`` so multiple concurrent envs with
    differently-shaped articulations cannot collide.
    """
    cache_key = (id(env), joint_name)
    if cache_key not in _GRIPPER_JOINT_IDX_CACHE:
        ids, _ = env.scene["robot"].find_joints(joint_name)
        _GRIPPER_JOINT_IDX_CACHE[cache_key] = ids[0]
    return _GRIPPER_JOINT_IDX_CACHE[cache_key]


def joint_pos_by_names(env, joint_names: tuple[str, ...] | list[str]) -> torch.Tensor:
    """Return absolute joint positions for the given joint names/patterns."""
    names = tuple(joint_names)
    cache_key = (id(env), names)
    if cache_key not in _JOINT_GROUP_IDX_CACHE:
        ids: list[int] = []
        for joint_name in names:
            found, _ = env.scene["robot"].find_joints(joint_name)
            ids.extend(found)
        _JOINT_GROUP_IDX_CACHE[cache_key] = ids
    return as_torch(env.scene["robot"].data.joint_pos)[:, _JOINT_GROUP_IDX_CACHE[cache_key]]


def gripper_joint_pos(env, gripper_joint_name: str) -> torch.Tensor:
    """Return one raw gripper joint position as ``(num_envs, 1)``."""
    idx = gripper_joint_idx(env, gripper_joint_name)
    return as_torch(env.scene["robot"].data.joint_pos)[:, idx : idx + 1]


def base_velocity_decomposed_odom(env) -> torch.Tensor:
    """3-D base planar twist ``[vx, vy, yaw_rate]`` in the robot base frame.

    The wallx ``velocity_decomposed_odom`` proprio channel for mobile bases
    The differential-drive base command is the base linear (x, y) velocity plus yaw rate.
    Near-zero while the robot is stationary at the shelf, matching the x2robot
    training distribution.
    """
    robot = env.scene["robot"]
    lin_xy = as_torch(robot.data.root_lin_vel_b)[:, :2]
    yaw_rate = as_torch(robot.data.root_ang_vel_b)[:, 2:3]
    return torch.cat([lin_xy, yaw_rate], dim=-1)


def zero_height(env) -> torch.Tensor:
    """Format-matching 1-D zero ``height`` proprio ``(num_envs, 1)``.

    A folding-waist robot with no prismatic lift has no real
    ``height`` channel. The wallx whole-body wire schema still carries a 1-D
    ``height``/``lift`` slot; emit a zero so the server's ``state_height`` stays
    correctly shaped (avoids a 0-dim concatenate) without inventing a value.
    """
    ref = as_torch(env.scene["robot"].data.joint_pos)
    return ref.new_zeros((ref.shape[0], 1))


def ee_pos_root_local(env, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return EE position expressed in the robot root/base_link frame."""
    pos, _ = get_ee_root_local(env, ee_frame_cfg)
    return pos


def ee_euler_xyz_root_local(env, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return EE rotation as Euler XYZ in the robot root/base_link frame."""
    _, quat = get_ee_root_local(env, ee_frame_cfg)
    roll, pitch, yaw = euler_xyz_from_quat_xyzw_tensor(quat)
    return torch.stack([roll, pitch, yaw], dim=-1)


def wallx_follow_absolute_root_local(
    env,
    ee_frame_cfg: SceneEntityCfg,
    gripper_joint_name: str,
) -> torch.Tensor:
    """7-D wallx ``follow*_pos`` absolute pose in robot root/base_link.

    Layout is ``[pos(3), euler_xyz(3), gripper(1)]``. This is the observation
    counterpart of ``wire_pose_reference: base_link``: the policy sees and
    returns poses in the same articulation-root frame, so a non-zero ``/World``
    or spawned robot pose does not leak into inference.
    """
    return torch.cat(
        [
            ee_pos_root_local(env, ee_frame_cfg),
            ee_euler_xyz_root_local(env, ee_frame_cfg),
            gripper_joint_pos(env, gripper_joint_name),
        ],
        dim=-1,
    )


def ee_pos_from_init_ee_root_local(env, ee_frame_cfg: SceneEntityCfg, ref_key: str) -> torch.Tensor:
    """Return root-local EE position offset from the home EE pose."""
    pos, quat = get_ee_root_local(env, ee_frame_cfg)
    ref_pos, _ = get_or_init_ee_ref(env, ref_key, pos, quat)
    return pos - ref_pos


def ee_quat_xyzw_from_init_ee_root_local(env, ee_frame_cfg: SceneEntityCfg, ref_key: str) -> torch.Tensor:
    """Return root-local EE rotation from the home EE orientation, quat XYZW.

    Pairs with :func:`ee_pos_from_init_ee_root_local` (shared ``ref_key``) so a
    single env attribute anchors the home reference for both pos and quat
    outputs.
    """
    pos, quat = get_ee_root_local(env, ee_frame_cfg)
    _, ref_quat = get_or_init_ee_ref(env, ref_key, pos, quat)
    return quat_mul_xyzw_tensor(quat, quat_conjugate_xyzw_tensor(ref_quat))


def wallx_follow_from_init_ee_root_local(
    env,
    ee_frame_cfg: SceneEntityCfg,
    gripper_joint_name: str,
    ref_key: str,
) -> torch.Tensor:
    """7-D wallx ``follow*_pos`` offset from the home EE pose.

    The position and Euler XYZ components are expressed in the robot root
    (``base_link``) frame, with the home (``default_joint_pos``) EE pose as
    reference. Shares ``ref_key`` with :func:`ee_pos_from_init_ee_root_local` /
    :func:`ee_quat_xyzw_from_init_ee_root_local` and with
    :class:`~manaenv.policy.action_transforms.EEAbsoluteIKTransform` init refs
    when ``wire_pose_reference`` is ``init_ee_pose``.
    """
    pos, quat = get_ee_root_local(env, ee_frame_cfg)
    ref_pos, ref_quat = get_or_init_ee_ref(env, ref_key, pos, quat)
    relative_pos = pos - ref_pos
    relative_quat = quat_mul_xyzw_tensor(quat, quat_conjugate_xyzw_tensor(ref_quat))
    roll, pitch, yaw = euler_xyz_from_quat_xyzw_tensor(relative_quat)
    relative_euler = torch.stack([roll, pitch, yaw], dim=-1)
    return torch.cat([relative_pos, relative_euler, gripper_joint_pos(env, gripper_joint_name)], dim=-1)


def _get_or_init_lift_ref(env, lift_joint_name: str) -> torch.Tensor:
    """Cache the lift joint position at episode reset, same lifecycle as EE refs."""
    key = "_lift_ref_" + lift_joint_name
    refresh_key = key + "_needs_refresh"
    lift_pos = joint_pos_by_names(env, [lift_joint_name])  # (N, 1)
    if not hasattr(env, key):
        setattr(env, key, lift_pos.clone())
        setattr(env, refresh_key, torch.zeros(lift_pos.shape[0], dtype=torch.bool, device=lift_pos.device))
    ref = getattr(env, key)
    if hasattr(env, "episode_length_buf"):
        # branch-free, sync-free; see get_or_init_ee_ref
        needs_refresh: torch.Tensor = getattr(env, refresh_key)
        needs_refresh |= env.episode_length_buf == 0
        deferred = needs_refresh & (env.episode_length_buf == 1)
        ref.copy_(torch.where(deferred.unsqueeze(-1), lift_pos, ref))
        needs_refresh &= ~deferred
    return ref


def wallx_follow_from_init_ee_root_local_lift_compensated(
    env,
    ee_frame_cfg: SceneEntityCfg,
    gripper_joint_name: str,
    ref_key: str,
    lift_joint_name: str,
) -> torch.Tensor:
    """7-D wallx ``follow*_pos`` offset from home EE, with lift displacement removed.

    Same as :func:`wallx_follow_from_init_ee_root_local` but subtracts the
    lift-joint delta from the Z component so the obs reflects only arm motion
    relative to the lift carriage, not the chassis.
    """
    pos, quat = get_ee_root_local(env, ee_frame_cfg)
    ref_pos, ref_quat = get_or_init_ee_ref(env, ref_key, pos, quat)
    relative_pos = pos - ref_pos
    lift_now = joint_pos_by_names(env, [lift_joint_name])  # (N, 1)
    lift_init = _get_or_init_lift_ref(env, lift_joint_name)
    relative_pos = relative_pos.clone()
    relative_pos[:, 2:3] -= lift_now - lift_init
    relative_quat = quat_mul_xyzw_tensor(quat, quat_conjugate_xyzw_tensor(ref_quat))
    roll, pitch, yaw = euler_xyz_from_quat_xyzw_tensor(relative_quat)
    relative_euler = torch.stack([roll, pitch, yaw], dim=-1)
    return torch.cat([relative_pos, relative_euler, gripper_joint_pos(env, gripper_joint_name)], dim=-1)


def wallx_follow_from_init_ee_lift_local(
    env, ee_frame_cfg: SceneEntityCfg, gripper_joint_name: str, ref_key: str
) -> torch.Tensor:
    """EX001 whole-body follow pose in ``lift_link``, relative to reset home."""
    pos, quat = get_ee_body_local(env, ee_frame_cfg, "lift_link")
    ref_pos, ref_quat = get_or_init_ee_ref(env, ref_key, pos, quat)
    relative_quat = quat_mul_xyzw_tensor(quat, quat_conjugate_xyzw_tensor(ref_quat))
    roll, pitch, yaw = euler_xyz_from_quat_xyzw_tensor(relative_quat)
    return torch.cat(
        [pos - ref_pos, torch.stack([roll, pitch, yaw], dim=-1), gripper_joint_pos(env, gripper_joint_name)], dim=-1
    )
