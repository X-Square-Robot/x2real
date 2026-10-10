"""X2Robot IK action term for IsaacLab environments.

Wraps X2RobotIKController as an ActionTerm, following the same pattern as
DifferentialInverseKinematicsAction but using Pinocchio-based IK with
Armijo line search.

Quaternion / command conventions
--------------------------------
* Isaac Lab articulation buffers + ``isaaclab.utils.math``: **XYZW** (the
  supported Isaac Sim 6.0 / IsaacLab environment is XYZW throughout — buffers,
  ``quat_from_euler_xyz``, ``subtract_frame_transforms``, etc.)
* Pinocchio / X2RobotIKController: **WXYZ**

Because the two sides disagree, every quaternion crossing the action→controller
boundary is converted XYZW→WXYZ here:
* Current EE pose from ``_compute_frame_pose`` (isaaclab XYZW → WXYZ)
* ``quat_xyzw`` command format via ``_command_pose_xyzw_to_wxyz``
* Master-slave 26D ``pos3 + euler_xyz3``: ``quat_from_euler_xyz`` (XYZW) → WXYZ
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm

from manaenv.controllers.x2robot_ik_cfg import X2RobotIKControllerCfg
from manaenv.controllers.x2robot_ik_controller import X2RobotIKController
from manaenv.utils.math_utils import quat_xyzw_to_wxyz
from manaenv.utils.tensor_utils import as_torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

_VALID_POSE_FORMATS = frozenset({"quat_xyzw", "euler_xyz"})


def _command_pose_xyzw_to_wxyz(command: torch.Tensor) -> torch.Tensor:
    """Convert absolute pose command ``[pos3, quat_xyzw4]`` to controller WXYZ."""
    if command.shape[-1] < 7:
        return command
    out = command.clone()
    out[..., 3:7] = quat_xyzw_to_wxyz(command[..., 3:7])
    return out


def _euler_pose_to_controller_wxyz(command: torch.Tensor) -> torch.Tensor:
    """Convert ``[pos3, euler_xyz3]`` → controller ``[pos3, quat_wxyz4]``.

    ``isaaclab.utils.math.quat_from_euler_xyz`` returns XYZW in this tree.
    """
    from isaaclab.utils import math as math_utils

    pos = command[..., :3]
    quat_xyzw = math_utils.quat_from_euler_xyz(command[..., 3], command[..., 4], command[..., 5])
    quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
    # Hemisphere: keep w >= 0 so consecutive IK cmds do not flip double-cover.
    flip = quat_wxyz[..., 0:1] < 0.0
    quat_wxyz = torch.where(flip, -quat_wxyz, quat_wxyz)
    return torch.cat([pos, quat_wxyz], dim=-1)


class X2RobotIKAction(ActionTerm):
    """Action term that uses X2RobotIKController to convert EEF targets to joint positions."""

    cfg: X2RobotIKActionCfg
    _asset: Articulation

    def __init__(self, cfg: X2RobotIKActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Keep joint order identical to Pinocchio reduced-model chain.
        joint_query = self.cfg.ik_joint_names or self.cfg.joint_names
        try:
            self._joint_ids, self._joint_names = self._asset.find_joints(joint_query, preserve_order=True)
        except TypeError:
            self._joint_ids, self._joint_names = self._asset.find_joints(joint_query)
        self._num_joints = len(self._joint_ids)
        if self.cfg.ik_joint_names is not None and list(self._joint_names) != list(self.cfg.ik_joint_names):
            raise ValueError(
                "X2RobotIKAction joint order mismatch: "
                f"find_joints -> {list(self._joint_names)}, "
                f"ik_joint_names -> {list(self.cfg.ik_joint_names)}"
            )

        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        self._body_idx = body_ids[0]

        # Pinocchio uses URDF ``base_frame``; Isaac USD may expose a different body
        # name (``isaac_base_body``). Soft-resolve without find_bodies on missing names.
        self._base_body_idx = self._resolve_isaac_base_body_idx()

        controller_cfg = X2RobotIKControllerCfg(
            urdf_path=self.cfg.urdf_path,
            base_frame=self.cfg.base_frame,
            ee_frame=self.cfg.ee_frame,
            joint_names=list(self.cfg.ik_joint_names or self._joint_names),
            command_type=self.cfg.command_type,
            use_relative_mode=self.cfg.use_relative_mode,
            jacobian_damping=self.cfg.jacobian_damping,
            armijo_alpha_init=self.cfg.armijo_alpha_init,
            armijo_c=self.cfg.armijo_c,
            armijo_beta=self.cfg.armijo_beta,
            armijo_alpha_min=self.cfg.armijo_alpha_min,
            dt=self.cfg.ik_dt,
        )
        self._ik_controller = X2RobotIKController(cfg=controller_cfg, num_envs=self.num_envs, device=self.device)

        fmt = str(getattr(self.cfg, "pose_command_format", "quat_xyzw") or "quat_xyzw").strip().lower()
        if fmt not in _VALID_POSE_FORMATS:
            raise ValueError(
                f"X2RobotIKActionCfg.pose_command_format={fmt!r} invalid; "
                f"expected one of {sorted(_VALID_POSE_FORMATS)}"
            )
        self._pose_command_format = fmt

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.as_tensor(self.cfg.scale, device=self.device)

        if self.cfg.body_offset is not None:
            self._offset_pos = torch.as_tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            # body_offset.rot feeds isaaclab combine_frame_transforms → XYZW (identity = 0,0,0,1).
            self._offset_rot = torch.as_tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

    @property
    def action_dim(self) -> int:
        if getattr(self, "_pose_command_format", "quat_xyzw") == "euler_xyz":
            return 6
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale
        command = self._processed_actions
        if self.cfg.command_type == "pose" and not self.cfg.use_relative_mode:
            if self._pose_command_format == "euler_xyz":
                command = _euler_pose_to_controller_wxyz(command)
            else:
                command = _command_pose_xyzw_to_wxyz(command)
            self._ik_controller.set_command(command)
            return
        ee_pos_curr, ee_quat_curr_wxyz = self._compute_frame_pose()
        self._ik_controller.set_command(command, ee_pos_curr, ee_quat_curr_wxyz)

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr_wxyz = self._compute_frame_pose()
        joint_pos = as_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr_wxyz, joint_pos)
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        if env_ids is None:
            reset_ids = None
        elif isinstance(env_ids, torch.Tensor):
            reset_ids = env_ids.detach().to(device=self.device, dtype=torch.long)
        else:
            reset_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._ik_controller.reset(reset_ids)

    def _resolve_isaac_base_body_idx(self) -> int | None:
        """Map the IK command frame onto an Isaac articulation body index.

        ``isaac_base_body`` in ``{"root", "root_link", "articulation_root"}``
        (or unset with no matching named body) uses ``root_pos_w`` /
        ``root_quat_w`` — the same frame interactive hold/reference capture uses.
        """
        available = [str(name) for name in getattr(self._asset, "body_names", ())]
        if not available:
            return None
        candidates: list[str] = []
        isaac_base = getattr(self.cfg, "isaac_base_body", None)
        if isaac_base:
            key = str(isaac_base).strip().lower()
            if key in ("root", "root_link", "articulation_root"):
                return None
            candidates.append(str(isaac_base).strip())
            if candidates[0] not in available:
                raise RuntimeError(
                    f"X2RobotIKAction required isaac_base_body={candidates[0]!r} is absent; "
                    f"available bodies are {available}."
                )
            return int(available.index(candidates[0]))
        candidates.extend(["base_link", "base"])
        if self.cfg.base_frame:
            candidates.append(str(self.cfg.base_frame))
        seen: set[str] = set()
        for name in candidates:
            if not name or name in seen:
                continue
            seen.add(name)
            if name in available:
                return int(available.index(name))
        return None

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return EE pose in the IK base frame, quaternion converted to controller **WXYZ**.

        Isaac Lab buffers and ``isaaclab.utils.math`` are XYZW in this tree, so
        all frame math runs in XYZW and the result is swizzled to WXYZ at the
        boundary for the Pinocchio controller. ``body_offset.rot`` is XYZW too.
        """
        from isaaclab.utils import math as math_utils

        body_pos_w = as_torch(self._asset.data.body_pos_w)
        body_quat_w = as_torch(self._asset.data.body_quat_w)
        ee_pos_w = body_pos_w[:, self._body_idx]
        ee_quat_w = body_quat_w[:, self._body_idx]
        if self._base_body_idx is not None:
            base_pos_w = body_pos_w[:, self._base_body_idx]
            base_quat_w = body_quat_w[:, self._base_body_idx]
        else:
            base_pos_w = as_torch(self._asset.data.root_pos_w)
            base_quat_w = as_torch(self._asset.data.root_quat_w)
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )
        return ee_pose_b, quat_xyzw_to_wxyz(ee_quat_b)


@dataclass
class BodyOffsetCfg:
    pos: tuple = (0.0, 0.0, 0.0)
    # Fed to isaaclab combine_frame_transforms → XYZW in this tree. Identity = (0, 0, 0, 1).
    rot: tuple = (0.0, 0.0, 0.0, 1.0)


@dataclass
class X2RobotIKActionCfg(ActionTermCfg):
    class_type: type = X2RobotIKAction

    joint_names: list[str] = field(default_factory=list)
    body_name: str = ""
    body_offset: BodyOffsetCfg | None = None
    scale: float | tuple = 1.0

    urdf_path: str = ""
    # Pinocchio / URDF frame (EX001 URDF uses base_platform_link).
    base_frame: str = "base_link"
    # Optional Isaac USD body for command-frame EE pose.
    isaac_base_body: str | None = None
    ee_frame: str = "link6"
    ik_joint_names: list[str] | None = None

    command_type: str = "pose"
    use_relative_mode: bool = False
    # ``quat_xyzw`` → 7D pose command; ``euler_xyz`` → 6D master-slave command.
    pose_command_format: str = "quat_xyzw"

    jacobian_damping: float = 0.0005
    armijo_alpha_init: float = 0.2
    armijo_c: float = 0.1
    armijo_beta: float = 0.5
    armijo_alpha_min: float = 1e-4
    ik_dt: float = 0.005
