"""VR teleoperation planner."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
import torch
from manaenv.runtime_owners import mana_embodiment
from manaenv.utils.camera_utils import deactivate_robot_camera_prims, disable_embodiment_camera_config
from manaenv.utils.motion_utils import world_to_base_frame
from .action_packing import bimanual_pose_parts
from .teleop_planner import TeleopPlanner
from .vr_teleop_planner_cfg import VRTeleopPlannerCFG, _apply_joystick_deadband

if TYPE_CHECKING:
    from isaaclab.markers import VisualizationMarkers


class VRTeleopPlanner(TeleopPlanner):
    """Bimanual Vuer/WebXR teleoperation for EE control.

    Action layout (X2Robot / DiffIK, typically 18D):

    ``[L_pos(3), L_quat(4), R_pos(3), R_quat(4), L_grip(1), R_grip(1),
       L_wheel(1), R_wheel(1)]``  (poses-first for EX001-6R X2Robot;
       CX002 X2Robot uses DiffIK interleaved grip packing)

    Raw Vuer device layout (retargeter ordering from
    ``build_motion_controller_retargeters``):

    ``[L_pos(3), L_quat(4), L_grip(1), R_pos(3), R_quat(4), R_grip(1),
       buttons(4)?, joystick(2..4)?]``
    """

    CFG = VRTeleopPlannerCFG
    device_name = "vr"
    _per_arm_pose_dim = 7
    _raw_pose_dim = 16
    _pink_arm_action_dim = 16
    _controller_button_dim = 4
    _controller_joystick_axis_dim = 2
    _base_action_dim = 2
    _lift_action_dim = 1
    _squeeze_dim = 2
    _LEFT_POSE_SLOT = slice(0, 7)
    _RIGHT_POSE_SLOT = slice(7, 14)
    _LEFT_POS_SLOT = slice(0, 3)
    _LEFT_QUAT_SLOT = slice(3, 7)
    _RIGHT_POS_SLOT = slice(7, 10)
    _RIGHT_QUAT_SLOT = slice(10, 14)
    _LEFT_GRIP_SLOT = slice(14, 15)
    _RIGHT_GRIP_SLOT = slice(15, 16)
    _BASE_SLOT = slice(16, 18)
    _LEFT_GRIP_INDEX = 14
    _RIGHT_GRIP_INDEX = 15
    _DIFFIK_ARM_ACTION_DIM = 16
    _DIFFIK_LEFT_POS_SLOT = slice(0, 3)
    _DIFFIK_LEFT_QUAT_SLOT = slice(3, 7)
    _DIFFIK_LEFT_GRIP_SLOT = slice(7, 8)
    _DIFFIK_RIGHT_POS_SLOT = slice(8, 11)
    _DIFFIK_RIGHT_QUAT_SLOT = slice(11, 15)
    _DIFFIK_RIGHT_GRIP_SLOT = slice(15, 16)
    _DIFFIK_BASE_SLOT = slice(16, 18)
    _DIFFIK_LEFT_GRIP_INDEX = 7
    _DIFFIK_RIGHT_GRIP_INDEX = 15

    def __init__(self):
        """Initialize planner-local device state."""
        super().__init__()
        self._vr_controller = None
        self._expected_action_dim: int | None = None
        self._vr_gripper_targets: dict[int, tuple[float, float]] = {}
        self._vr_gripper_joint_names: tuple[str, str] | None = None
        self._vr_ee_frame_names: tuple[str, str] = ("left_ee_frame", "right_ee_frame")
        self._session_active = False
        self._teleop_active = False
        self._usd_cameras_deactivated = False
        self._gym_env = None
        self._base_control_enabled = False
        self._base_control_cfg = None
        self._base_joystick_sides: list[str] = []
        self._lift_enabled = False
        self._lift_joint_name = "lift_joint"
        self._lift_limits: tuple[float, float] = (0.0, 0.78)
        self._lift_target: float | None = None
        self._lift_ref: float | None = None
        self._hold_action: torch.Tensor | None = None
        self._target_markers: VisualizationMarkers | None = None
        self._command_target_markers: VisualizationMarkers | None = None
        self._left_target_pos_base: torch.Tensor | None = None
        self._left_target_quat_base: torch.Tensor | None = None
        self._right_target_pos_base: torch.Tensor | None = None
        self._right_target_quat_base: torch.Tensor | None = None
        self._absolute_target_quat_offset: torch.Tensor | None = None
        self._motion_mode = "diffik"
        self._configured_motion_mode = "diffik"
        self._previous_controller_buttons: torch.Tensor | None = None
        self._vuer_delta_reference: dict[str, dict[str, torch.Tensor]] = {}
        self._vuer_axis_mapping_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _init_device(self, env: Any) -> None:
        from manaenv.teleop_devices.keyboard import _get_expected_action_dim

        embodiment = mana_embodiment(env)
        if embodiment is None:
            raise RuntimeError(
                "VR teleop needs a robot embodiment (set env.embodiment or build via Arena cfg.isaaclab_arena_env)."
            )
        input_backend = str(getattr(self.cfg, "input_backend", "vuer")).strip().lower()
        if input_backend in {"openxr", "cloudxr", "isaac_xr"}:
            raise RuntimeError(
                f"VRTeleopPlanner input_backend={input_backend!r} is removed; CloudXR/OpenXR teleop is deprecated. Use input_backend='vuer'."
            )
        if input_backend != "vuer":
            raise RuntimeError(f"Unsupported VRTeleopPlanner input_backend={input_backend!r}; expected 'vuer'.")
        if self.cfg.disable_all_cameras_in_vr:
            disable_embodiment_camera_config(embodiment)
        self._configured_motion_mode = self._resolve_configured_motion_mode()
        self._maybe_align_default_xr_anchor(env, embodiment)
        self._absolute_target_quat_offset = self._resolve_absolute_target_quat_offset(env, embodiment)
        sim_device = str(env.device)
        self._base_control_cfg = getattr(embodiment, "diff_drive_keyboard_controller_cfg", None)
        self._base_joystick_sides = self._resolve_base_joystick_sides()
        self._base_control_enabled = (
            bool(getattr(self.cfg, "use_motion_controller_base", False)) and self._base_control_cfg is not None
        )
        lift_getter = getattr(embodiment, "get_vr_lift_joint_name", None)
        if callable(lift_getter):
            self._lift_enabled = True
            self._lift_joint_name = lift_getter()
            limits_getter = getattr(embodiment, "get_vr_lift_limits", None)
            if callable(limits_getter):
                lo, hi = limits_getter()
                self._lift_limits = (float(lo), float(hi))
        else:
            self._lift_enabled = False
        callbacks = {"START": self._start_teleop, "STOP": self._stop_teleop, "RESET": self._reset_episode}
        self._vr_controller = self._create_vuer_controller(
            embodiment=embodiment, callbacks=callbacks, sim_device=sim_device
        )
        self._expected_action_dim = _get_expected_action_dim(env)
        self._gym_env = env
        self._vr_ee_frame_names = self._resolve_vr_ee_frame_names(embodiment)
        self._vr_gripper_targets = self._resolve_vr_gripper_targets(embodiment)
        self._vr_gripper_joint_names = self._resolve_vr_gripper_joint_names(embodiment)
        expected_action_dim = self._expected_env_action_dim()
        if self._expected_action_dim != expected_action_dim:
            raise RuntimeError(
                f"VR {self._configured_motion_mode} mode expects a {expected_action_dim}D env action vector, got {self._expected_action_dim}."
            )
        self._motion_mode = self._configured_motion_mode

    def _create_vuer_controller(self, *, embodiment: Any, callbacks: dict[str, Any], sim_device: str) -> Any:
        from manaenv.teleop_devices.vuer_controller import VuerControllerDevice, VuerControllerDeviceCfg

        cfg = VuerControllerDeviceCfg(
            sim_device=sim_device,
            host=self.cfg.vuer_host,
            port=self.cfg.vuer_port,
            display_fps=self.cfg.vuer_display_fps,
            apply_xr_anchor=self.cfg.vuer_apply_xr_anchor,
            anchor_pose_provider=self._vuer_xr_anchor_pose,
            gripper_input=self.cfg.motion_controller_gripper_input,
            gripper_open_threshold=self.cfg.motion_controller_gripper_open_threshold,
            gripper_close_threshold=self.cfg.motion_controller_gripper_close_threshold,
            include_buttons=self.cfg.use_motion_controller_buttons,
            include_joysticks=self._base_control_enabled,
            include_squeeze=self._lift_enabled,
        )
        device = VuerControllerDevice(cfg)
        for key, callback in callbacks.items():
            device.add_callback(key, callback)
        return device

    def requested_action_mode(self) -> str | None:
        mode = self._resolve_configured_motion_mode()
        if mode == "pink":
            return "pink_ik"
        return "ik_eef_abs"

    def _get_device_action(self, obs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        raw = self._vr_controller.advance().clone()
        raw_pose, raw_buttons, raw_joystick, raw_squeeze = self._split_raw_pose_and_buttons(raw)
        button_edges = self._get_controller_button_edges(raw_buttons)
        self._handle_active_toggle(button_edges, raw_pose)
        if not self._teleop_active:
            self._visualize_input_target_poses(raw_pose)
            action = self._arm_hold_action_with_live_controls(raw_pose, raw_joystick, raw_squeeze)
        else:
            action = self._build_pose_action(raw_pose, raw_joystick, raw_squeeze)
            if self._vr_marker_visualization_enabled() and self.cfg.visualize_target_positions:
                self._visualize_action_markers(self._visualized_target_pose_action(raw_pose), self._target_markers)
        if self._configured_motion_mode == "pink":
            self._visualize_action_markers(
                action, self._command_target_markers, guard_attr="visualize_commanded_target_positions"
            )
            if self._pink_action_needs_world_frame_conversion():
                action = self._action_pose_base_to_world(action)
        return (action, {"device": self.device_name})

    def prepare_episode(self, gym_env: Any, obs: dict[str, Any]) -> None:
        super().prepare_episode(gym_env, obs)
        self._gym_env = gym_env
        self._session_active = True
        self._teleop_active = False
        self._lift_target = self._read_lift_position()
        self._capture_hold_action()
        if self.cfg.disable_all_cameras_in_vr and (not self._usd_cameras_deactivated):
            deactivate_robot_camera_prims()
            self._usd_cameras_deactivated = True

    def _reset_device(self) -> None:
        self._session_active = False
        self._teleop_active = False
        self._hold_action = None
        self._previous_controller_buttons = None
        self._vuer_delta_reference = {}
        self._lift_target = None
        self._lift_ref = None
        self._left_target_pos_base = None
        self._left_target_quat_base = None
        self._right_target_pos_base = None
        self._right_target_quat_base = None
        self._motion_mode = self._configured_motion_mode
        if self._vr_controller is not None:
            self._vr_controller.reset()

    def cleanup(self) -> None:
        if self._vr_controller is not None and hasattr(self._vr_controller, "close"):
            self._vr_controller.close()
        self._gym_env = None
        super().cleanup()

    def _start_teleop(self) -> None:
        self._session_active = True
        self._teleop_active = False
        self._motion_mode = self._configured_motion_mode
        if self._gym_env is not None:
            self._lift_target = self._read_lift_position()
            self._capture_hold_action()

    def _stop_teleop(self) -> None:
        self._session_active = False
        self._teleop_active = False
        if self._gym_env is not None:
            self._capture_hold_action()

    def _toggle_teleop_active(self) -> None:
        if not self._session_active:
            return
        if self._teleop_active:
            self._teleop_active = False
            if self._gym_env is not None:
                self._capture_hold_action()
            return
        self._teleop_active = True
        self._motion_mode = self._configured_motion_mode
        if self._gym_env is not None:
            self._capture_hold_action(reset_target_state=False)

    def _reset_episode(self) -> None:
        self._session_active = False
        self._teleop_active = False
        self._vuer_delta_reference = {}
        self.emit_reset_event()

    def is_session_active(self) -> bool:
        return bool(self._session_active)

    def is_control_active(self) -> bool:
        return bool(self._teleop_active)

    def _resolve_configured_motion_mode(self) -> str:
        mode = int(self.cfg.gen_method if self.cfg.gen_method is not None else 0)
        if mode not in (0, 2):
            raise ValueError("Manual VR collection supports gen_method=0 (DiffIK) or 2 (X2Robot IK).")
        return {0: "diffik", 2: "pink"}[mode]

    def _expected_env_action_dim(self) -> int:
        dim = self._pink_arm_action_dim if self._configured_motion_mode == "pink" else self._DIFFIK_ARM_ACTION_DIM
        if self._base_control_enabled:
            dim += self._base_action_dim
        if self._lift_enabled:
            dim += self._lift_action_dim
        return dim

    def _handle_active_toggle(self, button_edges: dict[str, bool], raw_pose: torch.Tensor) -> None:
        if self._handle_left_x_start(button_edges, raw_pose):
            return
        if self._handle_right_a_reset(button_edges):
            return
        if self._handle_left_y_success(button_edges):
            return
        if button_edges["left_toggle"] or button_edges["right_toggle"]:
            self._toggle_teleop_active()

    def _handle_left_x_start(self, button_edges: dict[str, bool], raw_pose: torch.Tensor) -> bool:
        if not bool(getattr(self.cfg, "motion_controller_left_x_starts_teleop", False)):
            return False
        if not button_edges["left_commit"]:
            return False
        button_edges["left_commit"] = False
        if self._teleop_active:
            self._teleop_active = False
            if self._gym_env is not None:
                self._capture_hold_action()
        else:
            self._teleop_active = True
            self._motion_mode = self._configured_motion_mode
            self._capture_vuer_delta_reference(raw_pose)
            if self._gym_env is not None:
                self._capture_hold_action(reset_target_state=False)
        return True

    def _handle_right_a_reset(self, button_edges: dict[str, bool]) -> bool:
        if not bool(getattr(self.cfg, "motion_controller_right_a_resets_episode", False)):
            return False
        if not button_edges["right_commit"]:
            return False
        button_edges["right_commit"] = False
        self._reset_episode()
        return True

    def _handle_left_y_success(self, button_edges: dict[str, bool]) -> bool:
        if not bool(getattr(self.cfg, "motion_controller_left_y_marks_success", False)):
            return False
        if not button_edges["left_toggle"]:
            return False
        button_edges["left_toggle"] = False
        self.emit_success_event()
        return True

    def _is_vuer_backend(self) -> bool:
        return str(getattr(self.cfg, "input_backend", "vuer")).strip().lower() == "vuer"

    def _vuer_delta_enabled(self) -> bool:
        return self._is_vuer_backend() and bool(getattr(self.cfg, "vuer_use_delta_targets", False))

    def _vuer_delta_rotation_mode(self) -> str:
        return str(getattr(self.cfg, "vuer_delta_rotation_mode", "hold")).strip().lower()

    def _raw_side_pose_base(
        self, raw: torch.Tensor, *, side: str, root_pos: torch.Tensor, root_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if side == "left":
            pos_w = raw[0:3]
            quat_w = raw[3:7]
        else:
            pos_w = raw[8:11]
            quat_w = raw[11:15]
        pos_base, quat_base = world_to_base_frame(pos_w, quat_w, root_pos, root_quat)
        quat_base = self._apply_absolute_target_quat_offset(quat_base)
        return (pos_base, self._normalize_quat(quat_base))

    def _raw_side_position(self, raw: torch.Tensor, *, side: str) -> torch.Tensor:
        if side == "left":
            return raw[0:3]
        return raw[8:11]

    def _raw_side_quat(self, raw: torch.Tensor, *, side: str) -> torch.Tensor:
        if side == "left":
            return self._normalize_quat(raw[3:7])
        return self._normalize_quat(raw[11:15])

    def _map_vuer_position_delta(self, delta: torch.Tensor) -> torch.Tensor:
        axis_values = {"x": delta[0], "y": delta[1], "z": delta[2]}
        mapped = []
        for axis in getattr(self.cfg, "vuer_delta_position_axes", ("-z", "-x", "y")):
            key = str(axis).strip().lower()
            sign = -1.0 if key.startswith("-") else 1.0
            key = key[1:] if key.startswith(("-", "+")) else key
            if key not in axis_values:
                raise ValueError(f"Invalid Vuer delta axis mapping: {axis!r}")
            mapped.append(axis_values[key] * sign)
        return torch.stack(mapped)

    def _vuer_axis_mapping_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (device, dtype)
        cached = self._vuer_axis_mapping_cache.get(cache_key)
        if cached is not None:
            return cached
        rows = []
        for axis in getattr(self.cfg, "vuer_delta_position_axes", ("-z", "-x", "y")):
            key = str(axis).strip().lower()
            sign = -1.0 if key.startswith("-") else 1.0
            key = key[1:] if key.startswith(("-", "+")) else key
            if key not in ("x", "y", "z"):
                raise ValueError(f"Invalid Vuer delta axis mapping: {axis!r}")
            row = torch.zeros(3, dtype=dtype, device=device)
            row[{"x": 0, "y": 1, "z": 2}[key]] = sign
            rows.append(row)
        mapping = torch.stack(rows, dim=0)
        self._vuer_axis_mapping_cache[cache_key] = mapping
        return mapping

    def _map_vuer_rotation_delta(self, quat_delta: torch.Tensor) -> torch.Tensor:
        import isaaclab.utils.math as math_utils

        mapping = self._vuer_axis_mapping_matrix(quat_delta.device, quat_delta.dtype)
        rot_vuer = math_utils.matrix_from_quat(self._normalize_quat(quat_delta).unsqueeze(0))[0]
        rot_robot = mapping @ rot_vuer @ mapping.transpose(0, 1)
        return self._normalize_quat(math_utils.quat_from_matrix(rot_robot.unsqueeze(0))[0])

    def _capture_vuer_delta_reference(self, raw: torch.Tensor) -> None:
        if not self._vuer_delta_enabled() or self._gym_env is None:
            return
        self._lift_ref = self._read_lift_position()
        robot = self._gym_env.scene["robot"]
        root_pos = robot.data.root_pos_w.torch[0].clone()
        root_quat = robot.data.root_quat_w.torch[0].clone()
        left_pos_w, left_quat_w, right_pos_w, right_quat_w = self._get_ee_world_poses()
        left_target_pos, left_target_quat = world_to_base_frame(left_pos_w, left_quat_w, root_pos, root_quat)
        right_target_pos, right_target_quat = world_to_base_frame(right_pos_w, right_quat_w, root_pos, root_quat)
        left_ctrl_quat = self._raw_side_quat(raw, side="left")
        right_ctrl_quat = self._raw_side_quat(raw, side="right")
        self._vuer_delta_reference = {
            "left": {
                "controller_pos": self._raw_side_position(raw, side="left").detach().clone(),
                "controller_quat": left_ctrl_quat.detach().clone(),
                "target_pos": left_target_pos.detach().clone(),
                "target_quat": self._normalize_quat(left_target_quat.detach().clone()),
            },
            "right": {
                "controller_pos": self._raw_side_position(raw, side="right").detach().clone(),
                "controller_quat": right_ctrl_quat.detach().clone(),
                "target_pos": right_target_pos.detach().clone(),
                "target_quat": self._normalize_quat(right_target_quat.detach().clone()),
            },
        }

    def _vuer_delta_pose_targets_base(
        self, raw: torch.Tensor, *, root_pos: torch.Tensor, root_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        import isaaclab.utils.math as math_utils

        if not self._vuer_delta_enabled():
            return None
        if not self._vuer_delta_reference:
            self._capture_vuer_delta_reference(raw)
        if "left" not in self._vuer_delta_reference or "right" not in self._vuer_delta_reference:
            return None
        scale = float(getattr(self.cfg, "vuer_delta_position_scale", 1.0))
        lift_offset = 0.0
        if self._lift_enabled and self._lift_ref is not None:
            lift_offset = self._read_lift_position() - self._lift_ref
        outputs: list[torch.Tensor] = []
        for side in ("left", "right"):
            ref = self._vuer_delta_reference[side]
            ctrl_quat = self._raw_side_quat(raw, side=side)
            raw_pos = self._raw_side_position(raw, side=side)
            raw_delta = raw_pos - ref["controller_pos"].to(raw_pos.device)
            pos_delta = self._map_vuer_position_delta(raw_delta) * scale
            target_pos = ref["target_pos"].to(raw_pos.device) + pos_delta
            if lift_offset != 0.0:
                target_pos = target_pos.clone()
                target_pos[2] = target_pos[2] + lift_offset
            ref_target_quat = ref["target_quat"].to(ctrl_quat.device)
            if self._vuer_delta_rotation_mode() == "controller":
                ref_ctrl_quat = ref["controller_quat"].to(ctrl_quat.device)
                quat_delta = math_utils.quat_mul(
                    ctrl_quat.unsqueeze(0), math_utils.quat_inv(ref_ctrl_quat.unsqueeze(0))
                )[0]
                quat_delta = self._map_vuer_rotation_delta(quat_delta)
                target_quat = math_utils.quat_mul(quat_delta.unsqueeze(0), ref_target_quat.unsqueeze(0))[0]
            else:
                target_quat = ref_target_quat
            outputs.extend([target_pos, self._normalize_quat(target_quat)])
        return (outputs[0], outputs[1], outputs[2], outputs[3])

    def _resolve_bimanual_targets_base(
        self, raw: torch.Tensor, *, smooth: bool = True, apply_offset: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        robot = self._gym_env.scene["robot"]
        root_pos = robot.data.root_pos_w.torch[0].clone()
        root_quat = robot.data.root_quat_w.torch[0].clone()
        delta_targets = self._vuer_delta_pose_targets_base(raw, root_pos=root_pos, root_quat=root_quat)
        if delta_targets is not None:
            lp, lq, rp, rq = delta_targets
        else:
            lp, lq = self._raw_side_pose_base(raw, side="left", root_pos=root_pos, root_quat=root_quat)
            rp, rq = self._raw_side_pose_base(raw, side="right", root_pos=root_pos, root_quat=root_quat)
        if apply_offset:
            offset = self._xr_target_base_pos_offset(root_pos)
            lp, rp = (lp + offset, rp + offset)
        if smooth:
            lp, lq = self._smooth_absolute_target_pose(lp, lq, side="left")
            rp, rq = self._smooth_absolute_target_pose(rp, rq, side="right")
        return (lp, lq, rp, rq)

    def _uses_interleaved_grip_packing(self) -> bool:
        """Return True when env action layout is DiffIK-style interleaved grips.

        CX002 X2Robot IK matches Abs DiffIK (``L7, Lg, R7, Rg, ...``).
        EX001-6R X2Robot IK keeps the former Pink poses-first packing
        (``L7, R7, Lg, Rg``). Detect via action-cfg field order when available.
        """
        if self._configured_motion_mode != "pink":
            return True
        assembled = None
        if self._gym_env is not None:
            assembled = getattr(self._gym_env, "_mana_assembled_environment", None)
        embodiment = getattr(assembled, "embodiment", None) if assembled is not None else None
        action_cfg = getattr(embodiment, "action_config", None)
        if action_cfg is None:
            name = str(getattr(embodiment, "name", "") or "").upper()
            type_name = type(embodiment).__name__
            if "6R" in name or "EX0016R" in type_name:
                return False
            return name in ("CX002", "EX001") or "CX002" in type_name or type_name == "EX001Embodiment"
        try:
            from dataclasses import fields as dc_fields

            names = [f.name for f in dc_fields(type(action_cfg))]
        except Exception:
            names = [n for n in dir(action_cfg) if not n.startswith("_")]
        if "gripper_action" in names and "right_arm_action" in names:
            return names.index("gripper_action") < names.index("right_arm_action")
        return False

    def _build_pose_action(
        self, raw: torch.Tensor, raw_joystick: torch.Tensor | None = None, raw_squeeze: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._gym_env is None:
            raise RuntimeError("VR teleop planner has no gym environment bound.")
        lp, lq, rp, rq = self._resolve_bimanual_targets_base(raw)
        parts = list(
            bimanual_pose_parts(
                lp,
                self._normalize_quat(lq),
                raw[7:8],
                rp,
                self._normalize_quat(rq),
                raw[15:16],
                interleaved_grippers=self._uses_interleaved_grip_packing(),
            )
        )
        if self._base_control_enabled:
            parts.append(self._compute_base_wheel_action(raw_joystick))
        if self._lift_enabled:
            parts.append(self._compute_lift_action(raw_squeeze))
        return self._apply_vr_gripper_targets(torch.cat(parts))

    def _pink_pose_action(
        self, raw: torch.Tensor, raw_joystick: torch.Tensor | None = None, raw_squeeze: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._build_pose_action(raw, raw_joystick, raw_squeeze)

    def _diffik_pose_action(
        self, raw: torch.Tensor, raw_joystick: torch.Tensor | None = None, raw_squeeze: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._build_pose_action(raw, raw_joystick, raw_squeeze)

    def _visualized_target_pose_action(self, raw: torch.Tensor) -> torch.Tensor:
        if self._gym_env is None:
            raise RuntimeError("VR teleop planner has no gym environment bound for visualized target conversion.")
        lp, lq, rp, rq = self._resolve_bimanual_targets_base(raw, smooth=False, apply_offset=False)
        return torch.cat([lp, self._normalize_quat(lq), rp, self._normalize_quat(rq), raw[7:8], raw[15:16]])

    def _resolve_vr_gripper_targets(self, embodiment: Any) -> dict[int, tuple[float, float]]:
        """Map VR binary gripper commands to raw joint targets when needed.

        Resolution order:
          1. Legacy / replay configs that still expose ``gripper_action`` and
             ``right_gripper_action`` ``ClampedRawGripperActionCfg`` terms with
             ``clamp_min`` / ``clamp_max``: read the bounds straight from those
             action terms.
          2. Configs where grippers are folded into a combined IK term (legacy
             Pink ``hand_joint_names``) and no separate clamp action term exists:
             ask the embodiment for per-joint bounds via
             ``get_vr_gripper_clamp``, then map the embodiment-provided left /
             right gripper names to the planner's left / right grip slots.
        """
        action_cfg = getattr(embodiment, "action_config", None)
        if action_cfg is None:
            return {}
        targets: dict[int, tuple[float, float]] = {}
        left_slot, right_slot = self._gripper_slot_indices()
        legacy_index_to_term = {
            left_slot: getattr(action_cfg, "master_left_gripper_action", None)
            or getattr(action_cfg, "gripper_action", None)
        }
        right_term = getattr(action_cfg, "master_right_gripper_action", None) or getattr(
            action_cfg, "right_gripper_action", None
        )
        if right_term is not None:
            legacy_index_to_term[right_slot] = right_term
        for index, term in legacy_index_to_term.items():
            clamp_min = getattr(term, "clamp_min", None)
            clamp_max = getattr(term, "clamp_max", None)
            if clamp_min is None or clamp_max is None:
                continue
            targets[index] = (float(clamp_min), float(clamp_max))
        if targets:
            return targets
        get_clamp = getattr(embodiment, "get_vr_gripper_clamp", None)
        if not callable(get_clamp):
            return targets
        try:
            clamp_map = get_clamp() or {}
        except Exception:
            return targets
        gripper_names = self._resolve_vr_gripper_joint_names(embodiment)
        if gripper_names is None:
            return targets
        slot_for_joint = {gripper_names[0]: left_slot, gripper_names[1]: right_slot}
        for joint_name, slot in slot_for_joint.items():
            bounds = clamp_map.get(joint_name)
            if bounds is None:
                continue
            close_val, open_val = bounds
            targets[slot] = (float(close_val), float(open_val))
        return targets

    def _resolve_vr_gripper_joint_names(self, embodiment: Any) -> tuple[str, str] | None:
        """Return left/right gripper joint names through embodiment/action metadata."""
        get_names = getattr(embodiment, "get_vr_gripper_joint_names", None)
        if callable(get_names):
            names = list(get_names() or [])
            if len(names) >= 2:
                return (str(names[0]), str(names[1]))
        action_cfg = getattr(embodiment, "action_config", None)
        if action_cfg is None:
            return None
        left_term = getattr(action_cfg, "master_left_gripper_action", None) or getattr(
            action_cfg, "gripper_action", None
        )
        right_term = getattr(action_cfg, "master_right_gripper_action", None) or getattr(
            action_cfg, "right_gripper_action", None
        )
        left_names = list(getattr(left_term, "joint_names", []) or [])
        right_names = list(getattr(right_term, "joint_names", []) or [])
        if left_names and right_names:
            return (str(left_names[0]), str(right_names[0]))
        arm_action = getattr(action_cfg, "arm_action", None)
        hand_joint_names = list(getattr(arm_action, "hand_joint_names", []) or [])
        if len(hand_joint_names) >= 2:
            return (str(hand_joint_names[0]), str(hand_joint_names[1]))
        return None

    def _resolve_vr_ee_frame_names(self, embodiment: Any) -> tuple[str, str]:
        """Return left/right logical EE frame keys from the embodiment."""
        get_names = getattr(embodiment, "get_vr_ee_frame_names", None)
        if callable(get_names):
            names = list(get_names() or [])
            if len(names) >= 2:
                return (str(names[0]), str(names[1]))
        return ("left_ee_frame", "right_ee_frame")

    def _apply_vr_gripper_targets(self, action: torch.Tensor) -> torch.Tensor:
        """Convert VR gripper commands into raw gripper joint positions.

        The incoming gripper slot is an analog command: legacy binary devices
        send -1.0 (close) / +1.0 (open), while analog devices (Vuer trigger)
        send [0, 1]. Both are clamped to [0, 1] and linearly mapped onto
        [close_val, open_val].
        """
        if not self._vr_gripper_targets:
            return action
        remapped = action.clone()
        for index, (close_val, open_val) in self._vr_gripper_targets.items():
            if remapped.shape[-1] <= index:
                continue
            frac = remapped[index].clamp(0.0, 1.0)
            remapped[index] = close_val + frac * (open_val - close_val)
        return remapped

    def _gripper_slot_indices(self) -> tuple[int, int]:
        if self._uses_interleaved_grip_packing():
            return (self._DIFFIK_LEFT_GRIP_INDEX, self._DIFFIK_RIGHT_GRIP_INDEX)
        return (self._LEFT_GRIP_INDEX, self._RIGHT_GRIP_INDEX)

    def _base_slot(self) -> slice:
        if self._uses_interleaved_grip_packing():
            return self._DIFFIK_BASE_SLOT
        return self._BASE_SLOT

    def _arm_hold_action_with_live_controls(
        self, raw: torch.Tensor, raw_joystick: torch.Tensor | None = None, raw_squeeze: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._hold_action is None:
            raise RuntimeError("VR teleop hold action has not been captured.")
        action = self._hold_action.clone()
        left_gripper_slot, right_gripper_slot = self._gripper_slot_indices()
        action[left_gripper_slot : left_gripper_slot + 1] = raw[7:8]
        action[right_gripper_slot : right_gripper_slot + 1] = raw[15:16]
        action = self._apply_vr_gripper_targets(action)
        if self._base_control_enabled and action.numel() >= self._base_slot().stop:
            action[self._base_slot()] = self._compute_base_wheel_action(raw_joystick)
        if self._lift_enabled:
            lift_start = action.numel() - self._lift_action_dim
            action[lift_start:] = self._compute_lift_action(raw_squeeze)
        return action

    def _base_to_world_pose(
        self, pos_base: torch.Tensor, quat_base: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import isaaclab.utils.math as math_utils

        pos_world = root_pos + math_utils.quat_apply(root_quat.unsqueeze(0), pos_base.unsqueeze(0))[0]
        quat_world = math_utils.quat_mul(root_quat.unsqueeze(0), quat_base.unsqueeze(0))[0]
        return (pos_world, quat_world)

    def _pink_action_needs_world_frame_conversion(self) -> bool:
        """Return True when the active gen_method=2 action term expects world-frame poses.

        Pink IK historically consumed world/env poses. X2Robot IK on EX001-6R
        consumes robot-root / base-frame poses (same frame the planner already
        builds), so missing ``input_frame`` means no conversion.
        """
        if self._gym_env is None:
            return False
        assembled = getattr(self._gym_env, "_mana_assembled_environment", None)
        embodiment = getattr(assembled, "embodiment", None)
        action_cfg = getattr(embodiment, "action_config", None)
        arm_action = getattr(action_cfg, "arm_action", None)
        if arm_action is None:
            return False
        class_type = getattr(arm_action, "class_type", None)
        type_name = getattr(class_type, "__name__", "") or type(arm_action).__name__
        if "X2RobotIK" in type_name:
            return False
        input_frame = str(getattr(arm_action, "input_frame", "") or "").strip().lower()
        return input_frame in ("world", "env", "env_origin")

    def _action_pose_base_to_world(self, action: torch.Tensor) -> torch.Tensor:
        """Convert pose slots in ``action`` from base-frame to env/world-frame.

        IsaacLab's ``PinkInverseKinematicsAction`` interprets the pose slots
        in the action vector as env-origin / world-frame poses, then applies
        a world->base transform internally before passing the targets to
        Pink. The planner keeps targets in base-frame everywhere (smoothing,
        XR offsets, CuRobo waypoints, hold capture) so they stay invariant
        under base motion. This helper applies the base->world conversion at
        the action vector boundary so Pink's world->base transform recovers
        the constant base-frame target — making the EE follow the base when
        the wheels move instead of drifting in world.

        Only the left/right arm pose slots are touched; gripper slots and
        base slots remain unchanged.
        """
        if self._gym_env is None:
            return action
        robot = self._gym_env.scene["robot"]
        root_pos_world = robot.data.root_pos_w.torch[0]
        env_origin = self._gym_env.scene.env_origins[0]
        root_pos_env = (root_pos_world - env_origin).clone()
        root_quat = robot.data.root_quat_w.torch[0].clone()
        out = action.clone()
        left_pos_env, left_quat_env = self._base_to_world_pose(
            out[self._LEFT_POS_SLOT].clone(),
            self._normalize_quat(out[self._LEFT_QUAT_SLOT].clone()),
            root_pos_env,
            root_quat,
        )
        right_pos_env, right_quat_env = self._base_to_world_pose(
            out[self._RIGHT_POS_SLOT].clone(),
            self._normalize_quat(out[self._RIGHT_QUAT_SLOT].clone()),
            root_pos_env,
            root_quat,
        )
        out[self._LEFT_POS_SLOT] = left_pos_env
        out[self._LEFT_QUAT_SLOT] = self._normalize_quat(left_quat_env)
        out[self._RIGHT_POS_SLOT] = right_pos_env
        out[self._RIGHT_QUAT_SLOT] = self._normalize_quat(right_quat_env)
        return out

    def _resolve_base_joystick_sides(self) -> list[str]:
        sides = [
            str(getattr(self.cfg, "motion_controller_base_linear_side", "left")),
            str(getattr(self.cfg, "motion_controller_base_angular_side", "right")),
        ]
        unique_sides: list[str] = []
        for side in sides:
            if side not in unique_sides:
                unique_sides.append(side)
        return unique_sides

    def _get_base_joystick_axes(self, raw_joystick: torch.Tensor) -> dict[str, torch.Tensor]:
        axes: dict[str, torch.Tensor] = {}
        for idx, side in enumerate(self._base_joystick_sides):
            start = idx * self._controller_joystick_axis_dim
            end = start + self._controller_joystick_axis_dim
            if raw_joystick.numel() >= end:
                axes[side] = raw_joystick[start:end]
        return axes

    def _split_raw_pose_and_buttons(
        self, raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pose_raw = raw[: self._pink_arm_action_dim]
        cursor = self._pink_arm_action_dim
        if (
            self.cfg.use_motion_controller_buttons
            and raw.numel() >= self._pink_arm_action_dim + self._controller_button_dim
        ):
            buttons = raw[cursor : cursor + self._controller_button_dim]
            cursor += self._controller_button_dim
        else:
            buttons = raw.new_zeros(self._controller_button_dim)
        joystick_dim = len(self._base_joystick_sides) * self._controller_joystick_axis_dim
        if self._base_control_enabled and raw.numel() >= cursor + joystick_dim:
            joystick = raw[cursor : cursor + joystick_dim]
            cursor += joystick_dim
        else:
            joystick = raw.new_zeros(joystick_dim)
        if self._lift_enabled and raw.numel() >= cursor + self._squeeze_dim:
            squeeze = raw[cursor : cursor + self._squeeze_dim]
        else:
            squeeze = raw.new_zeros(self._squeeze_dim)
            pass
        return (pose_raw, buttons, joystick, squeeze)

    def _get_controller_button_edges(self, buttons: torch.Tensor) -> dict[str, bool]:
        threshold = float(self.cfg.motion_controller_button_press_threshold)
        pressed = (buttons.detach().to("cpu") > threshold).to(torch.bool)
        prev_pressed = self._previous_controller_buttons
        if prev_pressed is None or prev_pressed.shape != pressed.shape:
            prev_pressed = torch.zeros_like(pressed, dtype=torch.bool)
        edges = pressed & torch.logical_not(prev_pressed)
        self._previous_controller_buttons = pressed.clone()
        return {
            "left_commit": bool(edges[int(self.cfg.motion_controller_left_commit_button_index)].item()),
            "left_toggle": bool(edges[int(self.cfg.motion_controller_left_toggle_button_index)].item()),
            "right_commit": bool(edges[int(self.cfg.motion_controller_right_commit_button_index)].item()),
            "right_toggle": bool(edges[int(self.cfg.motion_controller_right_toggle_button_index)].item()),
        }

    def _xr_target_base_pos_offset(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(self.cfg.xr_target_base_pos_offset)

    def _get_ee_world_poses(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left = self._gym_env.scene[self._vr_ee_frame_names[0]]
        right = self._gym_env.scene[self._vr_ee_frame_names[1]]
        return (
            left.data.target_pos_w.torch[0, 0, :].clone(),
            left.data.target_quat_w.torch[0, 0, :].clone(),
            right.data.target_pos_w.torch[0, 0, :].clone(),
            right.data.target_quat_w.torch[0, 0, :].clone(),
        )

    def _read_joint_pos(self, robot: Any, joint_name: str) -> float:
        ids, _ = robot.find_joints(joint_name)
        return float(robot.data.joint_pos.torch[0, ids[0]].item())

    def _read_current_gripper_positions(self, robot: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._vr_gripper_joint_names is None:
            raise RuntimeError("VR teleop could not resolve left/right gripper joint names from the embodiment.")
        left_name, right_name = self._vr_gripper_joint_names
        left_gripper = torch.tensor([self._read_joint_pos(robot, left_name)], device=device)
        right_gripper = torch.tensor([self._read_joint_pos(robot, right_name)], device=device)
        return (left_gripper, right_gripper)

    def _compute_base_wheel_action(self, raw_joystick: torch.Tensor | None) -> torch.Tensor:
        if not self._base_control_enabled or raw_joystick is None:
            device = self._hold_action.device if self._hold_action is not None else "cpu"
            return torch.zeros(self._base_action_dim, dtype=torch.float32, device=device)
        cfg = self._base_control_cfg
        if cfg is None:
            return raw_joystick.new_zeros(self._base_action_dim)
        joystick_axes = self._get_base_joystick_axes(raw_joystick)
        linear_side = str(getattr(self.cfg, "motion_controller_base_linear_side", "left"))
        angular_side = str(getattr(self.cfg, "motion_controller_base_angular_side", "right"))
        linear_axes = joystick_axes.get(linear_side, raw_joystick.new_zeros(self._controller_joystick_axis_dim))
        angular_axes = joystick_axes.get(angular_side, raw_joystick.new_zeros(self._controller_joystick_axis_dim))
        thumb_x = angular_axes[0]
        thumb_y = linear_axes[1]
        deadband = max(float(getattr(self.cfg, "motion_controller_base_deadband", 0.0)), 0.0)
        thumb_x = _apply_joystick_deadband(thumb_x, deadband)
        thumb_y = _apply_joystick_deadband(thumb_y, deadband)
        linear_velocity = (
            -thumb_y
            * float(getattr(cfg, "linear_velocity", 0.0))
            * float(getattr(self.cfg, "motion_controller_base_linear_scale", 1.0))
        )
        angular_velocity = (
            -thumb_x
            * float(getattr(cfg, "angular_velocity", 0.0))
            * float(getattr(self.cfg, "motion_controller_base_angular_scale", 1.0))
        )
        left_wheel_velocity = (
            linear_velocity - 0.5 * angular_velocity * float(getattr(cfg, "wheel_track_width", 0.0))
        ) / float(getattr(cfg, "wheel_radius", 1.0))
        right_wheel_velocity = (
            linear_velocity + 0.5 * angular_velocity * float(getattr(cfg, "wheel_track_width", 0.0))
        ) / float(getattr(cfg, "wheel_radius", 1.0))
        return torch.stack([left_wheel_velocity, right_wheel_velocity])

    def _read_lift_position(self) -> float:
        if not self._lift_enabled or self._gym_env is None:
            return 0.0
        robot = self._gym_env.scene["robot"]
        ids, _ = robot.find_joints(self._lift_joint_name)
        return float(robot.data.joint_pos.torch[0, ids[0]].item())

    def _compute_lift_action(self, raw_squeeze: torch.Tensor | None) -> torch.Tensor:
        if self._lift_target is None:
            self._lift_target = self._read_lift_position()
        device = "cpu"
        if raw_squeeze is not None and raw_squeeze.numel() >= 2:
            device = raw_squeeze.device
            left_sq = float(raw_squeeze[0])
            right_sq = float(raw_squeeze[1])
            net = left_sq - right_sq
            if abs(net) > 0:
                step = float(self.cfg.motion_controller_lift_step)
                self._lift_target += net * step
        lo, hi = self._lift_limits
        self._lift_target = max(lo, min(hi, self._lift_target))
        return torch.tensor([self._lift_target], dtype=torch.float32, device=device)

    def _read_ee_poses_base(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._gym_env is None:
            raise RuntimeError("VR teleop planner has no gym environment bound.")
        robot = self._gym_env.scene["robot"]
        root_pos = robot.data.root_pos_w.torch[0].clone()
        root_quat = robot.data.root_quat_w.torch[0].clone()
        left_pos_w, left_quat_w, right_pos_w, right_quat_w = self._get_ee_world_poses()
        left_pos_base, left_quat_base = world_to_base_frame(left_pos_w, left_quat_w, root_pos, root_quat)
        right_pos_base, right_quat_base = world_to_base_frame(right_pos_w, right_quat_w, root_pos, root_quat)
        return (
            left_pos_base,
            self._normalize_quat(left_quat_base),
            right_pos_base,
            self._normalize_quat(right_quat_base),
        )

    def _capture_hold_action(self, reset_target_state: bool = True) -> None:
        if self._gym_env is None:
            raise RuntimeError("VR teleop planner has no gym environment bound for hold capture.")
        robot = self._gym_env.scene["robot"]
        left_pos_base, left_quat_base, right_pos_base, right_quat_base = self._read_ee_poses_base()
        left_gripper, right_gripper = self._read_current_gripper_positions(robot, left_pos_base.device)
        hold_parts = list(
            bimanual_pose_parts(
                left_pos_base,
                left_quat_base,
                left_gripper,
                right_pos_base,
                right_quat_base,
                right_gripper,
                interleaved_grippers=self._uses_interleaved_grip_packing(),
            )
        )
        if self._base_control_enabled:
            hold_parts.append(torch.zeros(self._base_action_dim, dtype=torch.float32, device=left_pos_base.device))
        if self._lift_enabled:
            hold_parts.append(
                torch.tensor([self._read_lift_position()], dtype=torch.float32, device=left_pos_base.device)
            )
        self._hold_action = torch.cat(hold_parts)
        if reset_target_state:
            self._left_target_pos_base = left_pos_base.clone()
            self._left_target_quat_base = self._normalize_quat(left_quat_base)
            self._right_target_pos_base = right_pos_base.clone()
            self._right_target_quat_base = self._normalize_quat(right_quat_base)

    def _smooth_absolute_target_pose(
        self, target_pos_base: torch.Tensor, target_quat_base: torch.Tensor, side: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_attr = f"_{side}_target_pos_base"
        quat_attr = f"_{side}_target_quat_base"
        target_quat_base = self._normalize_quat(target_quat_base)
        prev_pos = getattr(self, pos_attr)
        prev_quat = getattr(self, quat_attr)
        if prev_pos is None or prev_quat is None:
            setattr(self, pos_attr, target_pos_base.clone())
            setattr(self, quat_attr, target_quat_base.clone())
            return (target_pos_base, target_quat_base)
        pos_deadband = float(self.cfg.absolute_target_pos_deadband)
        pos_delta = target_pos_base - prev_pos
        pos_delta_norm = torch.linalg.norm(pos_delta)
        if pos_deadband > 0.0 and pos_delta_norm <= pos_deadband:
            target_pos_base = prev_pos.clone()
            pos_delta = target_pos_base - prev_pos
            pos_delta_norm = torch.linalg.norm(pos_delta)
        pos_motion = float(pos_delta_norm.item())
        rot_step = self._quat_angular_distance(prev_quat, target_quat_base)
        rot_deadband_rad = torch.deg2rad(target_pos_base.new_tensor(self.cfg.absolute_target_rot_deadband_deg))
        if rot_deadband_rad.item() > 0.0 and rot_step <= rot_deadband_rad:
            target_quat_base = prev_quat.clone()
            rot_step = target_pos_base.new_tensor(0.0)
        rot_motion = float(rot_step.item())
        max_pos_step = float(self.cfg.absolute_target_max_pos_step)
        if max_pos_step > 0.0 and pos_delta_norm > max_pos_step:
            pos_delta = pos_delta * (max_pos_step / pos_delta_norm.clamp_min(1e-08))
            target_pos_base = prev_pos + pos_delta
        pos_alpha = self._motion_adaptive_alpha(
            still_alpha=float(self.cfg.absolute_target_still_pos_alpha),
            moving_alpha=float(self.cfg.absolute_target_pos_alpha),
            motion=pos_motion,
            deadband=pos_deadband,
            motion_window=float(self.cfg.absolute_target_pos_alpha_motion_window),
        )
        smooth_pos = prev_pos + pos_alpha * (target_pos_base - prev_pos)
        max_rot_step_rad = torch.deg2rad(target_pos_base.new_tensor(self.cfg.absolute_target_max_rot_step_deg))
        if max_rot_step_rad.item() > 0.0 and rot_step > max_rot_step_rad:
            target_quat_base = self._quat_slerp(
                prev_quat, target_quat_base, float((max_rot_step_rad / rot_step).item())
            )
        rot_alpha = self._motion_adaptive_alpha(
            still_alpha=float(self.cfg.absolute_target_still_rot_alpha),
            moving_alpha=float(self.cfg.absolute_target_rot_alpha),
            motion=rot_motion,
            deadband=float(rot_deadband_rad.item()),
            motion_window=float(
                torch.deg2rad(target_pos_base.new_tensor(self.cfg.absolute_target_rot_alpha_motion_window_deg)).item()
            ),
        )
        smooth_quat = self._quat_slerp(prev_quat, target_quat_base, rot_alpha)
        setattr(self, pos_attr, smooth_pos.clone())
        setattr(self, quat_attr, smooth_quat.clone())
        return (smooth_pos, smooth_quat)

    def _motion_adaptive_alpha(
        self, still_alpha: float, moving_alpha: float, motion: float, deadband: float, motion_window: float
    ) -> float:
        still_alpha = min(max(still_alpha, 0.0), 1.0)
        moving_alpha = min(max(moving_alpha, 0.0), 1.0)
        if motion_window <= 1e-08:
            return moving_alpha
        motion_above_deadband = max(motion - max(deadband, 0.0), 0.0)
        blend = min(max(motion_above_deadband / motion_window, 0.0), 1.0)
        return still_alpha + (moving_alpha - still_alpha) * blend

    def _normalize_quat(self, quat: torch.Tensor) -> torch.Tensor:
        return quat / torch.linalg.norm(quat).clamp_min(1e-08)

    def _resolve_absolute_target_quat_offset(self, env: Any, embodiment: Any) -> torch.Tensor:
        device = str(env.device)
        if self.cfg.use_motion_controller_pose and hasattr(embodiment, "get_vr_controller_abs_target_quat_offset"):
            quat = embodiment.get_vr_controller_abs_target_quat_offset()
            return self._normalize_quat(torch.tensor(quat, dtype=torch.float32, device=device))
        if hasattr(embodiment, "get_vr_abs_target_quat_offset"):
            quat = embodiment.get_vr_abs_target_quat_offset()
            return self._normalize_quat(torch.tensor(quat, dtype=torch.float32, device=device))
        return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    def _apply_absolute_target_quat_offset(self, quat: torch.Tensor) -> torch.Tensor:
        import isaaclab.utils.math as math_utils

        offset = self._absolute_target_quat_offset
        if offset is None:
            return self._normalize_quat(quat)
        return self._normalize_quat(math_utils.quat_mul(quat.unsqueeze(0), offset.unsqueeze(0))[0])

    def _quat_angular_distance(self, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        q0 = self._normalize_quat(q0)
        q1 = self._normalize_quat(q1)
        dot = torch.abs(torch.dot(q0, q1)).clamp(max=1.0)
        return 2.0 * torch.acos(dot)

    def _quat_slerp(self, q0: torch.Tensor, q1: torch.Tensor, tau: float) -> torch.Tensor:
        import isaaclab.utils.math as math_utils

        q0 = self._normalize_quat(q0)
        q1 = self._normalize_quat(q1)
        return self._normalize_quat(math_utils.quat_slerp(q0, q1, float(tau)))

    def _ensure_target_markers(self) -> None:
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG

        if self._target_markers is not None:
            return
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/VRTargetEE"
        marker_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
        self._target_markers = VisualizationMarkers(marker_cfg)
        self._target_markers.set_visibility(True)

    def _ensure_command_target_markers(self) -> None:
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG

        if self._command_target_markers is not None:
            return
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/VRCommandedTargetEE"
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self._command_target_markers = VisualizationMarkers(marker_cfg)
        self._command_target_markers.set_visibility(True)

    def _vr_marker_visualization_enabled(self) -> bool:
        return bool(getattr(self.cfg, "enable_visualization", True)) and (not self._is_vuer_backend())

    def _visualize_input_target_poses(self, raw: torch.Tensor) -> None:
        if not self._vr_marker_visualization_enabled() or not self.cfg.visualize_target_positions:
            return
        self._ensure_target_markers()
        left_pos_w = raw[0:3]
        left_quat_w = self._apply_absolute_target_quat_offset(self._normalize_quat(raw[3:7]))
        right_pos_w = raw[8:11]
        right_quat_w = self._apply_absolute_target_quat_offset(self._normalize_quat(raw[11:15]))
        translations = torch.stack([left_pos_w, right_pos_w], dim=0).detach().cpu()
        orientations = torch.stack([left_quat_w, right_quat_w], dim=0).detach().cpu()
        self._target_markers.visualize(translations=translations, orientations=orientations, marker_indices=[0, 0])

    def _visualize_action_markers(
        self, action: torch.Tensor, markers: VisualizationMarkers | None, guard_attr: str = "visualize_target_positions"
    ) -> None:
        if not self._vr_marker_visualization_enabled() or not getattr(self.cfg, guard_attr, False):
            return
        if self._gym_env is None:
            return
        if markers is None:
            if guard_attr == "visualize_commanded_target_positions":
                self._ensure_command_target_markers()
                markers = self._command_target_markers
            else:
                self._ensure_target_markers()
                markers = self._target_markers
        robot = self._gym_env.scene["robot"]
        root_pos = robot.data.root_pos_w.torch[0].clone()
        root_quat = robot.data.root_quat_w.torch[0].clone()
        left_pos_w, left_quat_w = self._base_to_world_pose(
            action[self._LEFT_POS_SLOT], self._normalize_quat(action[self._LEFT_QUAT_SLOT]), root_pos, root_quat
        )
        right_pos_w, right_quat_w = self._base_to_world_pose(
            action[self._RIGHT_POS_SLOT], self._normalize_quat(action[self._RIGHT_QUAT_SLOT]), root_pos, root_quat
        )
        translations = torch.stack([left_pos_w, right_pos_w], dim=0).detach().cpu()
        orientations = torch.stack([left_quat_w, right_quat_w], dim=0).detach().cpu()
        markers.visualize(translations=translations, orientations=orientations, marker_indices=[0, 0])

    def _maybe_align_default_xr_anchor(self, env: Any, embodiment: Any) -> None:
        if not hasattr(embodiment, "get_xr_cfg"):
            return
        xr_cfg = embodiment.get_xr_cfg()
        if xr_cfg is None or getattr(xr_cfg, "anchor_prim_path", None) is not None:
            return
        robot = env.scene["robot"]
        xr_cfg.anchor_prim_path = self._resolve_dynamic_xr_anchor_prim_path(robot)
        try:
            from isaaclab_teleop.xr_cfg import XrAnchorRotationMode

            if getattr(xr_cfg, "anchor_rotation_mode", None) is not XrAnchorRotationMode.FOLLOW_PRIM:
                xr_cfg.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM
        except ModuleNotFoundError:
            pass

    def _vuer_xr_anchor_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the base-following XR anchor pose for optional Vuer rebasing."""
        import isaaclab.utils.math as math_utils

        if self._gym_env is None:
            return (np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        robot = self._gym_env.scene["robot"]
        root_pos = robot.data.root_pos_w.torch[0].detach().clone()
        root_quat = self._normalize_quat(robot.data.root_quat_w.torch[0].detach().clone())
        embodiment = mana_embodiment(self._gym_env)
        xr_cfg = embodiment.get_xr_cfg() if embodiment is not None and hasattr(embodiment, "get_xr_cfg") else None
        anchor_pos = torch.zeros(3, dtype=root_pos.dtype, device=root_pos.device)
        anchor_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=root_quat.dtype, device=root_quat.device)
        if xr_cfg is not None:
            anchor_pos = torch.tensor(
                getattr(xr_cfg, "anchor_pos", (0.0, 0.0, 0.0)), dtype=root_pos.dtype, device=root_pos.device
            )
            anchor_quat = self._normalize_quat(
                torch.tensor(
                    getattr(xr_cfg, "anchor_rot", (0.0, 0.0, 0.0, 1.0)), dtype=root_quat.dtype, device=root_quat.device
                )
            )
        pos_w = root_pos + math_utils.quat_apply(root_quat.unsqueeze(0), anchor_pos.unsqueeze(0))[0]
        quat_w = self._normalize_quat(math_utils.quat_mul(root_quat.unsqueeze(0), anchor_quat.unsqueeze(0))[0])
        return (pos_w.detach().cpu().numpy().astype(np.float32), quat_w.detach().cpu().numpy().astype(np.float32))

    def _resolve_dynamic_xr_anchor_prim_path(self, robot: Any) -> str:
        robot_prim_path = self._resolve_runtime_robot_prim_path(robot)
        body_names = [str(name) for name in list(getattr(robot.data, "body_names", []) or [])]
        if body_names:
            root_body_name = "base_link" if "base_link" in body_names else body_names[0]
            body_prim_path = self._find_body_prim_path(robot_prim_path, root_body_name)
            if body_prim_path is not None:
                return body_prim_path
        return robot_prim_path

    def _resolve_runtime_robot_prim_path(self, robot: Any) -> str:
        prim_path = str(getattr(robot.cfg, "prim_path", "") or "")
        if "{ENV_REGEX_NS}" in prim_path:
            prim_path = prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        if "env_.*" in prim_path:
            prim_path = prim_path.replace("env_.*", "env_0")
        if ".*" in prim_path:
            prim_path = prim_path.replace(".*", "0")
        if not prim_path:
            prim_path = "/World/envs/env_0/Robot"
        return prim_path

    def _find_body_prim_path(self, robot_prim_path: str, body_name: str) -> str | None:
        try:
            import omni.usd
            from pxr import Usd
        except ModuleNotFoundError:
            return None
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        if not robot_prim.IsValid():
            return None
        direct_child_path = f"{robot_prim_path}/{body_name}"
        direct_child = stage.GetPrimAtPath(direct_child_path)
        if direct_child.IsValid():
            return direct_child_path
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() == body_name:
                return str(prim.GetPath())
        return None


__all__ = ["VRTeleopPlanner", "VRTeleopPlannerCFG"]
