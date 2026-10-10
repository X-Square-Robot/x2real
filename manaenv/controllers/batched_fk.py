"""Batched forward kinematics and geometric Jacobian in PyTorch.

Extracts the serial kinematic chain from a Pinocchio reduced model at init
time (CPU, once), then performs FK / Jacobian / SE3-error evaluation entirely
on GPU using batched 4×4 homogeneous transforms.

Supports revolute joints only (the standard for 6-DOF robot arms).
"""

from __future__ import annotations

import torch
from torch import Tensor

from manaenv.utils.math_utils import quat_wxyz_to_rotmat_tensor as _quat_to_rotmat
from manaenv.utils.math_utils import rotmat_from_axis_angle_tensor as _rotmat_axis
from manaenv.utils.math_utils import skew_batch as _skew_batch
from manaenv.utils.math_utils import so3_log_tensor as so3_log_batch

# ---------------------------------------------------------------------------
# SE3 batch helpers
# ---------------------------------------------------------------------------


def pose_to_se3_batch(pos: Tensor, quat_wxyz: Tensor) -> Tensor:
    """Convert (N,3) position + (N,4) quaternion **WXYZ** to (N,4,4) SE3.

    X2Robot / Pinocchio controller state is WXYZ; convert ManaEnv XYZW at the
    action boundary before calling this helper.
    """
    N = pos.shape[0]
    R = _quat_to_rotmat(quat_wxyz)
    T = torch.zeros(N, 4, 4, device=pos.device, dtype=pos.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = pos
    T[:, 3, 3] = 1.0
    return T


def se3_inv_batch(T: Tensor) -> Tensor:
    """Invert (N,4,4) SE3 transforms: [R|t] -> [R^T | -R^T t]."""
    R = T[:, :3, :3]
    t = T[:, :3, 3:]
    Rt = R.transpose(-2, -1)
    out = torch.zeros_like(T)
    out[:, :3, :3] = Rt
    out[:, :3, 3:] = -torch.bmm(Rt, t)
    out[:, 3, 3] = 1.0
    return out


def se3_log_batch(T: Tensor) -> Tensor:
    """SE(3) logarithm: (N,4,4) -> (N,6) twist [linear, angular].

    Returns the 6D tangent vector in the LOCAL (body) frame, matching
    Pinocchio's ``log6(SE3).vector`` convention: [linear_3, angular_3].
    """
    R = T[:, :3, :3]
    p = T[:, :3, 3]

    omega = so3_log_batch(R)  # (N, 3) — from math_utils
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-10)

    # V^{-1} p: left Jacobian inverse of SO(3) applied to translation.
    # Near zero: V ≈ I → v ≈ p. General: Chirikjian's closed form.
    theta_sq = theta * theta
    theta_s = theta.squeeze(-1)

    omega_x = _skew_batch(omega)
    omega_x_sq = torch.bmm(omega_x, omega_x)

    cos_t = torch.cos(theta_s)
    sin_t = torch.sin(theta_s)
    denom = 2.0 * (1.0 - cos_t).clamp(min=1e-10)
    c2_closed = (1.0 / theta_sq.squeeze(-1).clamp(min=1e-10)) * (1.0 - theta_s * sin_t / denom)
    # The closed form suffers catastrophic cancellation for small float32
    # angles. The series is accurate through O(theta^6) in this range.
    c2_series = 1.0 / 12.0 + theta_sq.squeeze(-1) / 720.0
    c2 = torch.where(theta_s.abs() < 1e-2, c2_series, c2_closed)

    eye = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0)
    V_inv = eye - 0.5 * omega_x + c2.unsqueeze(-1).unsqueeze(-1) * omega_x_sq

    v = torch.bmm(V_inv, p.unsqueeze(-1)).squeeze(-1)
    return torch.cat([v, omega], dim=-1)  # [linear, angular] — matches Pinocchio Motion.vector


# ---------------------------------------------------------------------------
# BatchedFK — serial-chain FK and geometric Jacobian
# ---------------------------------------------------------------------------


class BatchedFK:
    """GPU-batched forward kinematics for a serial revolute chain.

    At construction, extracts the kinematic chain parameters from a Pinocchio
    reduced model (one-time CPU work). At runtime, ``forward()``, ``jacobian()``,
    and ``cartesian_error()`` operate entirely on GPU tensors.
    """

    def __init__(
        self,
        urdf_path: str,
        joint_names: list[str],
        base_frame: str,
        ee_frame: str,
        device: str,
    ):
        import numpy as np
        import pinocchio as pin

        full_model = pin.buildModelFromUrdf(urdf_path)
        joint_ids_to_lock = []
        for jid in range(1, len(full_model.joints)):
            if full_model.names[jid] not in joint_names:
                joint_ids_to_lock.append(jid)

        if joint_ids_to_lock:
            model = pin.buildReducedModel(full_model, joint_ids_to_lock, pin.neutral(full_model))
        else:
            model = full_model
        model_joint_names = [model.names[i] for i in range(1, len(model.joints))]
        if model_joint_names != list(joint_names):
            raise RuntimeError(
                f"Pinocchio reduced model joint order {model_joint_names} does not match "
                f"cfg.joint_names {list(joint_names)}. "
                "Reorder cfg.joint_names to match the URDF kinematic chain."
            )
        data = model.createData()

        self._num_joints = model.nq
        self._device = device

        base_frame_id = model.getFrameId(base_frame)
        ee_frame_id = model.getFrameId(ee_frame)

        # Extract kinematic chain: for each joint i, store the fixed
        # transform from the parent joint (or base) to this joint's frame,
        # and the joint axis (revolute joints rotate about local Z).
        #
        # Pinocchio convention:
        #   model.jointPlacements[i] = transform from parent joint to joint i
        #   Joint axis is always Z in the joint's local frame for revolute joints.

        # Compute FK at zero config to get all frame transforms
        q_zero = pin.neutral(model)
        pin.framesForwardKinematics(model, data, q_zero)
        base_se3 = data.oMf[base_frame_id]

        # Collect joint placements and axis types
        link_offsets = []
        joint_axes = []  # 0=X, 1=Y, 2=Z
        for jid in range(1, len(model.joints)):
            placement = model.jointPlacements[jid]
            T = np.eye(4)
            T[:3, :3] = placement.rotation
            T[:3, 3] = placement.translation
            link_offsets.append(T)

            shortname = model.joints[jid].shortname()
            if "RX" in shortname:
                joint_axes.append(0)
            elif "RY" in shortname:
                joint_axes.append(1)
            elif "RZ" in shortname:
                joint_axes.append(2)
            else:
                raise ValueError(f"Unsupported joint type: {shortname} for joint {model.names[jid]}")

        # EE frame offset: transform from last joint frame to EE frame
        # Compute at q=0: T_ee_in_last_joint = T_last_joint^{-1} @ T_ee
        last_joint_id = len(model.joints) - 1
        pin.forwardKinematics(model, data, q_zero)
        T_last_joint = data.oMi[last_joint_id]

        T_ee = data.oMf[ee_frame_id]

        T_ee_in_last = T_last_joint.actInv(T_ee)
        ee_offset = np.eye(4)
        ee_offset[:3, :3] = T_ee_in_last.rotation
        ee_offset[:3, 3] = T_ee_in_last.translation

        # Base frame offset: transform from world origin to base frame
        # For IK we compute everything relative to base, so we need
        # T_joint1_in_base = T_base^{-1} @ T_joint1_world_at_q0
        # But since reduced model joints start at the first active joint,
        # the first joint placement is already relative to the model root.
        # We need: T_base^{-1} @ T_model_root @ joint_placement[1]
        # For fixed-base robots, T_model_root = identity.
        # We pre-compose the base offset into the first link offset.
        T_base_inv = np.eye(4)
        T_base_inv[:3, :3] = base_se3.rotation.T
        T_base_inv[:3, 3] = -base_se3.rotation.T @ base_se3.translation
        link_offsets[0] = T_base_inv @ link_offsets[0]

        # Convert to torch tensors
        self._link_offsets = torch.tensor(
            np.stack(link_offsets), device=device, dtype=torch.float32
        )  # (num_joints, 4, 4)
        self._ee_offset = torch.tensor(ee_offset, device=device, dtype=torch.float32)  # (4, 4)
        self._joint_axes = joint_axes  # list of int: 0=X, 1=Y, 2=Z

    @property
    def num_joints(self) -> int:
        return self._num_joints

    def forward(self, q: Tensor) -> Tensor:
        """Batched FK: q (N, nj) -> T_ee_in_base (N, 4, 4)."""
        N = q.shape[0]
        nj = self._num_joints
        T = torch.eye(4, device=q.device, dtype=q.dtype).unsqueeze(0).expand(N, -1, -1).clone()

        # Reuse a single (N,4,4) homogeneous rotation buffer each iteration
        # instead of allocating a fresh clone; only the 3x3 block changes.
        Rh = torch.zeros(N, 4, 4, device=q.device, dtype=q.dtype)
        Rh[:, 3, 3] = 1.0

        for i in range(nj):
            T = T @ self._link_offsets[i].unsqueeze(0)
            Rh[:, :3, :3] = _rotmat_axis(q[:, i], self._joint_axes[i])
            T = T @ Rh

        return T @ self._ee_offset.unsqueeze(0)

    def forward_all(self, q: Tensor) -> Tensor:
        """Batched FK returning all intermediate frames.

        q: (N, nj) -> frames (N, nj+1, 4, 4)
        frames[:, i] = T_joint_i_in_base (after applying joint i rotation)
        frames[:, nj] = T_ee_in_base
        """
        N = q.shape[0]
        nj = self._num_joints
        frames = []
        T = torch.eye(4, device=q.device, dtype=q.dtype).unsqueeze(0).expand(N, -1, -1).clone()

        Rh = torch.zeros(N, 4, 4, device=q.device, dtype=q.dtype)
        Rh[:, 3, 3] = 1.0

        for i in range(nj):
            T = T @ self._link_offsets[i].unsqueeze(0)
            Rh[:, :3, :3] = _rotmat_axis(q[:, i], self._joint_axes[i])
            T = T @ Rh
            frames.append(T.clone())

        frames.append(T @ self._ee_offset.unsqueeze(0))
        return torch.stack(frames, dim=1)

    def forward_with_jacobian(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """Compute FK and LOCAL Jacobian in a single pass.

        q: (N, nj)
        Returns: (T_ee (N,4,4), J (N,6,nj))
        """
        N = q.shape[0]
        nj = self._num_joints

        joint_frames = []
        T = torch.eye(4, device=q.device, dtype=q.dtype).unsqueeze(0).expand(N, -1, -1).clone()

        Rh = torch.zeros(N, 4, 4, device=q.device, dtype=q.dtype)
        Rh[:, 3, 3] = 1.0

        for i in range(nj):
            T = T @ self._link_offsets[i].unsqueeze(0)
            Rh[:, :3, :3] = _rotmat_axis(q[:, i], self._joint_axes[i])
            T = T @ Rh
            joint_frames.append(T)

        T_ee = T @ self._ee_offset.unsqueeze(0)
        T_ee_inv = se3_inv_batch(T_ee)

        J = torch.zeros(N, 6, nj, device=q.device, dtype=q.dtype)
        for i in range(nj):
            T_rel = T_ee_inv @ joint_frames[i]
            R_rel = T_rel[:, :3, :3]
            axis_col = self._joint_axes[i]
            z_in_ee = R_rel[:, :3, axis_col]
            p_rel = T_rel[:, :3, 3]
            J[:, 3:, i] = z_in_ee
            J[:, :3, i] = torch.cross(z_in_ee, -p_rel, dim=-1)

        return T_ee, J

    def compute_error_twist_and_jacobian(self, q: Tensor, target_se3: Tensor) -> tuple[Tensor, Tensor]:
        """Compute SE3 error twist and Jacobian with only ONE FK pass.

        Returns: (delta_x (N,6), J (N,6,nj))
        """
        T_ee, J = self.forward_with_jacobian(q)
        T_err = se3_inv_batch(T_ee) @ target_se3
        delta_x = se3_log_batch(T_err)
        return delta_x, J

    def jacobian(self, q: Tensor) -> Tensor:
        """Batched geometric Jacobian in LOCAL (EE body) frame.

        q: (N, nj) -> J (N, 6, nj)
        """
        _, J = self.forward_with_jacobian(q)
        return J

    def cartesian_error(self, q: Tensor, target_se3: Tensor) -> Tensor:
        """Cartesian error norm for Armijo line search.

        q: (N, nj), target_se3: (N, 4, 4)
        Returns: (N,) — ||log(T_ee^{-1} @ T_target)||
        """
        T_ee = self.forward(q)
        T_err = se3_inv_batch(T_ee) @ target_se3
        twist = se3_log_batch(T_err)
        return twist.norm(dim=-1)

    def compute_error_twist(self, q: Tensor, target_se3: Tensor) -> Tensor:
        """SE3 error twist in LOCAL frame.

        q: (N, nj), target_se3: (N, 4, 4)
        Returns: (N, 6) — log(T_ee^{-1} @ T_target)
        """
        T_ee = self.forward(q)
        T_err = se3_inv_batch(T_ee) @ target_se3
        return se3_log_batch(T_err)
