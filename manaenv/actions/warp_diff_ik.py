"""Warp-native differential inverse-kinematics action terms."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from isaaclab.managers.action_manager import ActionTerm

from manaenv.actions.warp_diff_ik_kernels import (
    IK_METHOD_DLS,
    IK_METHOD_PINV,
    IK_METHOD_TRANSPOSE,
    compute_diff_ik_target_kernel,
    solve_diff_ik_kernel,
    write_diff_ik_target_kernel,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor


logger = logging.getLogger(__name__)

_IK_METHOD_CODES = {
    "pinv": IK_METHOD_PINV,
    "dls": IK_METHOD_DLS,
    "trans": IK_METHOD_TRANSPOSE,
}


class WarpDifferentialInverseKinematicsAction(ActionTerm):
    """Differential IK whose target mapping, solve, and target write run in Warp.

    Torch remains only at the policy action boundary. The internal action buffers
    are viewed zero-copy by Warp, while pose transforms, Jacobian transforms,
    pseudo-inverse/DLS math, and articulation target writes stay on the Warp
    device. When the ManaEnv runtime (or, as a fallback, the resolved physics
    config) enables CUDA Graph, each fixed-buffer action phase is captured once
    and replayed until its input buffer addresses change.
    """

    cfg: DifferentialInverseKinematicsActionCfg

    def __init__(self, cfg: DifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv) -> None:
        import isaaclab.utils.string as string_utils

        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._joint_ids = list(self._joint_ids)
        self._num_joints = len(self._joint_ids)
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for Warp DiffIK body_name={self.cfg.body_name!r}; "
                f"found {len(body_ids)}: {body_names}."
            )
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1 if self._asset.is_fixed_base else self._body_idx
        self._jacobi_joint_ids = [joint_id + self._asset.num_base_dofs for joint_id in self._joint_ids]

        controller_cfg = self.cfg.controller
        if controller_cfg.ik_method not in _IK_METHOD_CODES:
            supported = ", ".join(sorted(_IK_METHOD_CODES))
            raise ValueError(
                f"Warp DiffIK does not support ik_method={controller_cfg.ik_method!r}; "
                f"expected one of: {supported}."
            )
        if controller_cfg.ik_params is None:
            raise ValueError(f"Warp DiffIK requires ik_params for method {controller_cfg.ik_method!r}.")
        if getattr(controller_cfg, "joint_limit_avoidance_gain", 0.0) > 0.0:
            raise ValueError("Warp DiffIK does not yet support joint-limit null-space avoidance.")

        self._ik_method = _IK_METHOD_CODES[controller_cfg.ik_method]
        if controller_cfg.ik_method == "dls":
            self._method_parameter = float(controller_cfg.ik_params["lambda_val"])
        else:
            self._method_parameter = float(controller_cfg.ik_params["k_val"])

        self._command_code = 0 if controller_cfg.command_type == "position" else 1
        self._pose_task = int(controller_cfg.command_type == "pose")
        self._use_relative = int(controller_cfg.use_relative_mode)
        self._action_dim = self._resolve_action_dim(controller_cfg.command_type, controller_cfg.use_relative_mode)

        requested_device = getattr(self.cfg, "_mana_device", None)
        self._control_device = str(requested_device or self.device)
        self._wp_device = wp.get_device(self._control_device)
        self._simulation_wp_device = wp.get_device(self.device)
        self._split_device = self._wp_device != self._simulation_wp_device

        # Keep the public ActionTerm buffers on the environment device. With a
        # split controller device, fixed internal mirrors feed the captured
        # Warp graphs while explicit copies form the CPU-PhysX/CUDA boundary.
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scale = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        self._clip = torch.zeros((self.num_envs, self._action_dim, 2), device=self.device)
        self._has_clip = int(self.cfg.clip is not None)
        if self.cfg.clip is not None:
            if not isinstance(self.cfg.clip, dict):
                raise ValueError(f"Unsupported Warp DiffIK clip type: {type(self.cfg.clip)}. Expected dict.")
            self._clip[:, :, 0] = -float("inf")
            self._clip[:, :, 1] = float("inf")
            indices, _, values = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
            self._clip[:, indices] = torch.tensor(values, device=self.device)

        offset_pos = (0.0, 0.0, 0.0)
        offset_rot = (0.0, 0.0, 0.0, 1.0)
        if self.cfg.body_offset is not None:
            offset_pos = self.cfg.body_offset.pos
            offset_rot = self.cfg.body_offset.rot
        self._offset = wp.transformf(wp.vec3f(*offset_pos), wp.quatf(*offset_rot))

        orientation_weight = getattr(controller_cfg, "orientation_weight", None)
        if orientation_weight is None:
            orientation_weight = (1.0, 1.0, 1.0)
        elif isinstance(orientation_weight, (int, float)):
            orientation_weight = (float(orientation_weight),) * 3
        else:
            orientation_weight = tuple(float(value) for value in orientation_weight)
        if len(orientation_weight) != 3:
            raise ValueError("Warp DiffIK orientation_weight must be a scalar or a 3-vector.")
        self._orientation_weight = wp.vec3f(*orientation_weight)

        self._raw_actions_host_wp = wp.from_torch(self._raw_actions, dtype=wp.float32)
        self._processed_actions_host_wp = wp.from_torch(self._processed_actions, dtype=wp.float32)
        if self._split_device:
            self._raw_actions_control = torch.zeros(self.num_envs, self._action_dim, device=self._control_device)
            self._processed_actions_control = torch.zeros_like(self._raw_actions_control)
            self._scale_control = self._scale.to(device=self._control_device).contiguous()
            self._clip_control = self._clip.to(device=self._control_device).contiguous()
            self._raw_actions_wp = wp.from_torch(self._raw_actions_control, dtype=wp.float32)
            self._processed_actions_wp = wp.from_torch(self._processed_actions_control, dtype=wp.float32)
            self._scale_wp = wp.from_torch(self._scale_control, dtype=wp.float32)
            self._clip_wp = wp.from_torch(self._clip_control, dtype=wp.float32)
        else:
            self._raw_actions_wp = self._raw_actions_host_wp
            self._processed_actions_wp = self._processed_actions_host_wp
            self._scale_wp = wp.from_torch(self._scale, dtype=wp.float32)
            self._clip_wp = wp.from_torch(self._clip, dtype=wp.float32)
        self._joint_ids_wp = wp.array(self._joint_ids, dtype=wp.int32, device=self._control_device)
        self._joint_ids_sim_wp = (
            wp.array(self._joint_ids, dtype=wp.int32, device=self.device) if self._split_device else self._joint_ids_wp
        )
        self._jacobi_joint_ids_wp = wp.array(self._jacobi_joint_ids, dtype=wp.int32, device=self._control_device)
        self._target_pos = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self._control_device)
        self._target_quat = wp.zeros(self.num_envs, dtype=wp.quatf, device=self._control_device)
        self._joint_pos_des = wp.zeros((self.num_envs, self._num_joints), dtype=wp.float32, device=self._control_device)
        self._joint_pos_des_host = (
            wp.zeros((self.num_envs, self._num_joints), dtype=wp.float32, device=self.device)
            if self._split_device
            else self._joint_pos_des
        )
        self._staging: dict[str, wp.array] = {}

        physics_cfg = getattr(getattr(getattr(env, "sim", None), "cfg", None), "physics", None)
        runtime_cuda_graph = getattr(self.cfg, "_mana_use_cuda_graph", None)
        requested_cuda_graph = (
            bool(runtime_cuda_graph)
            if runtime_cuda_graph is not None
            else bool(getattr(physics_cfg, "use_cuda_graph", False))
        )
        self._use_cuda_graph = requested_cuda_graph and self._wp_device.is_cuda
        self._graphs: dict[str, tuple[tuple[Any, ...], Any]] = {}
        self._capture_disabled: set[str] = set()
        if requested_cuda_graph and not self._wp_device.is_cuda:
            logger.warning(
                "Warp DiffIK CUDA Graph was requested on non-CUDA device %s; using eager launches.",
                self._control_device,
            )

        logger.info(
            "Resolved Warp DiffIK action body %s [%s], joints %s [%s], method=%s, "
            "sim_device=%s, control_device=%s, cuda_graph=%s.",
            self._body_name,
            self._body_idx,
            self._joint_names,
            self._joint_ids,
            controller_cfg.ik_method,
            self.device,
            self._control_device,
            self._use_cuda_graph,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self._action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "WarpDifferentialInverseKinematicsAction"
        self._IO_descriptor.body_name = self._body_name
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.scale = self._scale
        self._IO_descriptor.clip = self.cfg.clip
        self._IO_descriptor.extras["controller_cfg"] = self.cfg.controller.__dict__
        self._IO_descriptor.extras["body_offset"] = (
            None if self.cfg.body_offset is None else self.cfg.body_offset.__dict__
        )
        self._IO_descriptor.extras["cuda_graph"] = self._use_cuda_graph
        self._IO_descriptor.extras["control_device"] = self._control_device
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        if self._split_device:
            wp.copy(self._raw_actions_wp, self._raw_actions_host_wp)
        body_pos_w = self._stage_input("body_pos_w", self._asset.data.body_pos_w.warp)
        body_quat_w = self._stage_input("body_quat_w", self._asset.data.body_quat_w.warp)
        root_pos_w = self._stage_input("root_pos_w", self._asset.data.root_pos_w.warp)
        root_quat_w = self._stage_input("root_quat_w", self._asset.data.root_quat_w.warp)

        def launch() -> None:
            wp.launch(
                compute_diff_ik_target_kernel,
                dim=self.num_envs,
                inputs=[
                    self._raw_actions_wp,
                    self._scale_wp,
                    self._clip_wp,
                    self._has_clip,
                    self._action_dim,
                    body_pos_w,
                    body_quat_w,
                    root_pos_w,
                    root_quat_w,
                    self._body_idx,
                    self._offset,
                    self._command_code,
                    self._use_relative,
                    self._processed_actions_wp,
                    self._target_pos,
                    self._target_quat,
                ],
                device=self._wp_device,
            )

        self._launch_or_replay_graph(
            "target",
            self._signature(
                self._raw_actions_wp,
                body_pos_w,
                body_quat_w,
                root_pos_w,
                root_quat_w,
                self._target_pos,
                self._target_quat,
            ),
            launch,
        )
        if self._split_device:
            wp.copy(self._processed_actions_host_wp, self._processed_actions_wp)
            wp.synchronize_device(self._wp_device)

    def apply_actions(self) -> None:
        jacobian_w = self._stage_input("jacobian_w", self._asset.data.body_link_jacobian_w.warp)
        joint_pos = self._stage_input("joint_pos", self._asset.data.joint_pos.warp)
        body_pos_w = self._stage_input("body_pos_w", self._asset.data.body_pos_w.warp)
        body_quat_w = self._stage_input("body_quat_w", self._asset.data.body_quat_w.warp)
        root_pos_w = self._stage_input("root_pos_w", self._asset.data.root_pos_w.warp)
        root_quat_w = self._stage_input("root_quat_w", self._asset.data.root_quat_w.warp)
        joint_pos_target = (
            self._staging_output("joint_pos_target", self._asset.data.joint_pos_target.warp)
            if self._split_device
            else self._asset.data.joint_pos_target.warp
        )

        def launch() -> None:
            wp.launch(
                solve_diff_ik_kernel,
                dim=self.num_envs,
                inputs=[
                    jacobian_w,
                    joint_pos,
                    body_pos_w,
                    body_quat_w,
                    root_pos_w,
                    root_quat_w,
                    self._target_pos,
                    self._target_quat,
                    self._joint_ids_wp,
                    self._jacobi_joint_ids_wp,
                    self._body_idx,
                    self._jacobi_body_idx,
                    self._offset,
                    self._orientation_weight,
                    self._pose_task,
                    self._ik_method,
                    self._method_parameter,
                    self._joint_pos_des,
                    joint_pos_target,
                ],
                device=self._wp_device,
            )

        self._launch_or_replay_graph(
            "solve",
            self._signature(
                jacobian_w,
                joint_pos,
                body_pos_w,
                body_quat_w,
                root_pos_w,
                root_quat_w,
                joint_pos_target,
            ),
            launch,
        )
        if self._split_device:
            self._write_control_target_to_sim()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0

    def _rewrite_cached_targets(self) -> None:
        if self._split_device:
            self._write_control_target_to_sim()
            return
        joint_pos_target = self._asset.data.joint_pos_target.warp

        def launch() -> None:
            wp.launch(
                write_diff_ik_target_kernel,
                dim=(self.num_envs, self._num_joints),
                inputs=[self._joint_pos_des, self._joint_ids_wp, joint_pos_target],
                device=self._wp_device,
            )

        self._launch_or_replay_graph(
            "cached_target",
            self._signature(self._joint_pos_des, joint_pos_target),
            launch,
        )

    def _stage_input(self, name: str, source: wp.array) -> wp.array:
        if not self._split_device:
            return source
        staged = self._staging.get(name)
        if staged is None:
            staged = wp.empty(source.shape, dtype=source.dtype, device=self._control_device)
            self._staging[name] = staged
        elif staged.shape != source.shape or staged.dtype != source.dtype:
            raise RuntimeError(
                f"Warp DiffIK staging buffer {name!r} changed layout from "
                f"{staged.shape}/{staged.dtype} to {source.shape}/{source.dtype}."
            )
        wp.copy(staged, source)
        return staged

    def _staging_output(self, name: str, target: wp.array) -> wp.array:
        staged = self._staging.get(name)
        if staged is None:
            staged = wp.empty(target.shape, dtype=target.dtype, device=self._control_device)
            self._staging[name] = staged
        return staged

    def _write_control_target_to_sim(self) -> None:
        wp.copy(self._joint_pos_des_host, self._joint_pos_des)
        wp.synchronize_device(self._wp_device)
        wp.launch(
            write_diff_ik_target_kernel,
            dim=(self.num_envs, self._num_joints),
            inputs=[
                self._joint_pos_des_host,
                self._joint_ids_sim_wp,
                self._asset.data.joint_pos_target.warp,
            ],
            device=self._simulation_wp_device,
        )

    def _launch_or_replay_graph(
        self,
        name: str,
        signature: tuple[Any, ...],
        launch: Callable[[], None],
    ) -> None:
        if not self._use_cuda_graph or name in self._capture_disabled:
            launch()
            return

        cached = self._graphs.get(name)
        if cached is not None and cached[0] == signature:
            wp.capture_launch(cached[1])
            return

        # Execute once before capture so the first command is valid and Warp's
        # JIT/module loading cannot occur inside CUDA stream capture.
        launch()
        wp.synchronize_device(self._wp_device)
        try:
            with wp.ScopedCapture(device=self._wp_device) as capture:
                launch()
        except Exception as exc:
            self._capture_disabled.add(name)
            logger.warning("Warp DiffIK CUDA Graph capture failed for %s; using eager launches: %s", name, exc)
            return

        self._graphs[name] = (signature, capture.graph)
        logger.info("Captured Warp DiffIK CUDA Graph phase %s.", name)

    @staticmethod
    def _signature(*arrays: wp.array) -> tuple[Any, ...]:
        return tuple((int(array.ptr), tuple(array.shape), str(array.dtype)) for array in arrays)

    @staticmethod
    def _resolve_action_dim(command_type: str, use_relative_mode: bool) -> int:
        if command_type == "position":
            return 3
        if command_type == "pose" and use_relative_mode:
            return 6
        if command_type == "pose":
            return 7
        raise ValueError(f"Unsupported Warp DiffIK command_type={command_type!r}.")


class WarpCommandRateDifferentialInverseKinematicsAction(WarpDifferentialInverseKinematicsAction):
    """Warp DiffIK evaluated once per command with graphable cached writes."""

    def __init__(self, cfg: DifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._ik_update_pending = True

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._ik_update_pending = True

    def apply_actions(self) -> None:
        if self._ik_update_pending:
            super().apply_actions()
            self._ik_update_pending = False
            return
        self._rewrite_cached_targets()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._ik_update_pending = True


__all__ = [
    "WarpCommandRateDifferentialInverseKinematicsAction",
    "WarpDifferentialInverseKinematicsAction",
]
