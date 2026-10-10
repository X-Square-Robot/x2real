"""X2Robot IK Controller — GPU-batched DLS with Armijo line search.

Port of the C++ CartesianPositionController from x2robot_controllers-v26.06.02.00.
Key features over IsaacLab's DifferentialIKController:
- Armijo backtracking line search for adaptive step size
- SE3 error via Lie-algebra log (batched)
- Measured-pose feedback that decouples Isaac and URDF base frames

IK is GPU-batched only (``BatchedFK`` + PyTorch). There is no CPU / Ruckig path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from manaenv.controllers.batched_fk import BatchedFK, pose_to_se3_batch, se3_inv_batch, se3_log_batch

if TYPE_CHECKING:
    from manaenv.controllers.x2robot_ik_cfg import X2RobotIKControllerCfg


class X2RobotIKController:
    """Cartesian IK controller using GPU-batched FK/Jacobian + DLS + Armijo."""

    def __init__(self, cfg: X2RobotIKControllerCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        self._desired_ee_pos = torch.zeros(num_envs, 3, device=device)
        # Pinocchio / X2Robot controller convention: WXYZ (identity = 1,0,0,0).
        self._desired_ee_quat = torch.zeros(num_envs, 4, device=device)
        self._desired_ee_quat[:, 0] = 1.0

        self._batched_fk = BatchedFK(
            urdf_path=self.cfg.urdf_path,
            joint_names=list(self.cfg.joint_names),
            base_frame=self.cfg.base_frame,
            ee_frame=self.cfg.ee_frame,
            device=self.device,
        )
        self._num_joints = self._batched_fk.num_joints
        self._identity = torch.eye(self._num_joints, device=self.device)

        self._limit_low = (
            torch.tensor(self.cfg.position_limit_low, device=self.device)
            if self.cfg.position_limit_low is not None
            else None
        )
        self._limit_high = (
            torch.tensor(self.cfg.position_limit_high, device=self.device)
            if self.cfg.position_limit_high is not None
            else None
        )

    @property
    def action_dim(self) -> int:
        if self.cfg.command_type == "position":
            return 3
        return 7 if not self.cfg.use_relative_mode else 6

    @property
    def num_joints(self) -> int:
        return self._num_joints

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._desired_ee_pos[env_ids] = 0.0
        self._desired_ee_quat[env_ids] = 0.0
        self._desired_ee_quat[env_ids, 0] = 1.0

    def set_command(
        self,
        command: torch.Tensor,
        ee_pos: torch.Tensor | None = None,
        ee_quat: torch.Tensor | None = None,
    ) -> None:
        if self.cfg.command_type == "position":
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError("relative position commands require the current end-effector pose")
                self._desired_ee_pos = ee_pos + command[:, :3]
                self._desired_ee_quat = ee_quat.clone()
            else:
                if ee_quat is None:
                    raise ValueError("position commands require the current end-effector orientation")
                self._desired_ee_pos = command[:, :3].clone()
                self._desired_ee_quat = ee_quat.clone()
        else:
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError("relative pose commands require the current end-effector pose")
                self._desired_ee_pos = ee_pos + command[:, :3]
                dquat = _axis_angle_to_quat(command[:, 3:6])
                self._desired_ee_quat = _quat_mul(dquat, ee_quat)
            else:
                self._desired_ee_pos = command[:, :3].clone()
                self._desired_ee_quat = command[:, 3:7].clone()
        qnorm = torch.norm(self._desired_ee_quat, dim=-1, keepdim=True).clamp(min=1e-8)
        self._desired_ee_quat = self._desired_ee_quat / qnorm

    def compute(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one feedback IK step from measured joint and EE state.

        The measured and commanded EE poses share an Isaac command frame, while
        FK/Jacobian values use the URDF base frame. Their relative transform is
        base-frame invariant, so applying it to current URDF FK avoids any
        static USD-to-URDF base calibration.
        """
        measured_se3 = pose_to_se3_batch(ee_pos, ee_quat)
        command_se3 = pose_to_se3_batch(self._desired_ee_pos, self._desired_ee_quat)
        measured_to_command = se3_inv_batch(measured_se3) @ command_se3
        delta_x = se3_log_batch(measured_to_command)

        current_fk, J = self._batched_fk.forward_with_jacobian(joint_pos)
        target_se3 = current_fk @ measured_to_command

        error_norm = delta_x.norm(dim=-1)
        vel_norm = error_norm / self.cfg.dt
        converged = (vel_norm < self.cfg.convergence_vel_eps) | (error_norm < self.cfg.gpu_convergence_error_eps)

        JtJ = torch.bmm(J.transpose(-2, -1), J)
        Jt_dx = torch.bmm(J.transpose(-2, -1), delta_x.unsqueeze(-1))
        delta_q = torch.linalg.solve(JtJ + self.cfg.jacobian_damping * self._identity, Jt_dx).squeeze(-1)

        projected_error_reduction = torch.bmm(J, delta_q.unsqueeze(-1)).squeeze(-1)
        directional_derivative = -(delta_x * projected_error_reduction).sum(dim=-1)
        alpha = self._batched_armijo(
            joint_pos,
            delta_q,
            target_se3,
            initial_error=error_norm,
            directional_derivative=directional_derivative,
        )
        new_q = joint_pos + alpha.unsqueeze(-1) * delta_q
        result = torch.where(converged.unsqueeze(-1), joint_pos, new_q)
        if self._limit_low is not None:
            result = torch.max(result, self._limit_low)
        if self._limit_high is not None:
            result = torch.min(result, self._limit_high)
        return result

    def _batched_armijo(
        self,
        q: torch.Tensor,
        direction: torch.Tensor,
        target_se3: torch.Tensor,
        *,
        initial_error: torch.Tensor,
        directional_derivative: torch.Tensor,
        max_iter: int = 10,
    ) -> torch.Tensor:
        """Return accepted Armijo step sizes, or zero when none decreases error."""
        N = q.shape[0]
        alpha = torch.full((N,), self.cfg.armijo_alpha_init, device=q.device, dtype=q.dtype)
        f0 = 0.5 * initial_error.square()
        accepted = torch.zeros(N, dtype=torch.bool, device=q.device)
        active = torch.ones(N, dtype=torch.bool, device=q.device)

        for _ in range(max_iter):
            q_new = q + alpha.unsqueeze(-1) * direction
            candidate_error = self._batched_fk.cartesian_error(q_new, target_se3)
            f_new = 0.5 * candidate_error.square()
            sufficient_decrease = f_new <= (f0 + self.cfg.armijo_c * alpha * directional_derivative)
            satisfied = active & (directional_derivative < 0.0) & sufficient_decrease
            accepted = accepted | satisfied
            active = active & ~satisfied & (alpha > self.cfg.armijo_alpha_min)
            if not active.any():
                break
            alpha = torch.where(active, alpha * self.cfg.armijo_beta, alpha)

        return torch.where(accepted, alpha, torch.zeros_like(alpha))


def _axis_angle_to_quat(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (N,3) to **WXYZ** quaternion (N,4)."""
    angle = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis_angle / angle
    half = angle * 0.5
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w, xyz], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for **WXYZ** quaternions (N,4)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
