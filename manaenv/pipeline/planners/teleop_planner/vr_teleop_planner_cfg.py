"""Lightweight CFG sidecar for ``VRTeleopPlanner`` (importable without IsaacSim)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from manaenv.pipeline.planners.teleop_planner.teleop_planner import TeleopPlannerCFG

__all__ = ["VRTeleopPlannerCFG"]


def _apply_joystick_deadband(value: torch.Tensor, deadband: float) -> torch.Tensor:
    threshold = max(float(deadband), 0.0)
    if threshold <= 0.0:
        return value
    return torch.where(torch.abs(value) <= threshold, torch.zeros_like(value), value)


@dataclass
class VRTeleopPlannerCFG(TeleopPlannerCFG):
    """Settings for bimanual Vuer/WebXR teleoperation."""

    # ── Core ──
    # 0: absolute DiffIK, 2: X2Robot IK.
    gen_method: int | None = 0
    # Only ``vuer`` is supported; CloudXR/OpenXR backends were removed.
    input_backend: str = "vuer"
    disable_all_cameras_in_vr: bool = False

    # ── Controller input ──
    use_motion_controller_pose: bool = True
    use_motion_controller_gripper: bool = True
    use_motion_controller_buttons: bool = True
    use_motion_controller_base: bool = True
    motion_controller_base_linear_side: str = "left"
    motion_controller_base_angular_side: str = "right"
    motion_controller_base_deadband: float = 0.0
    motion_controller_base_linear_scale: float = 1.0
    motion_controller_base_angular_scale: float = 1.0
    motion_controller_gripper_input: str = "trigger"

    # ── Lift (squeeze trigger) ──
    motion_controller_lift_step: float = 0.01
    motion_controller_gripper_open_threshold: float = 0.3
    motion_controller_gripper_close_threshold: float = 0.6
    motion_controller_quat_offset_wxyz: tuple[float, float, float, float] = (0.5, -0.5, 0.5, 0.5)

    # ── Button mapping ──
    motion_controller_button_press_threshold: float = 0.5
    motion_controller_left_commit_button_index: int = 0
    motion_controller_left_toggle_button_index: int = 1
    motion_controller_right_commit_button_index: int = 2
    motion_controller_right_toggle_button_index: int = 3
    motion_controller_left_x_starts_teleop: bool = False
    motion_controller_right_a_resets_episode: bool = False
    motion_controller_left_y_marks_success: bool = False

    # ── Vuer / WebXR ──
    vuer_host: str = "127.0.0.1"
    vuer_port: int | None = None
    vuer_display_fps: float = 60.0
    vuer_apply_xr_anchor: bool = True
    vuer_use_delta_targets: bool = False
    vuer_delta_position_scale: float = 1.0
    vuer_delta_position_axes: tuple[str, str, str] = ("-z", "-x", "y")
    vuer_delta_rotation_mode: str = "controller"

    # ── Target offset ──
    xr_target_base_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    absolute_target_pos_alpha: float = 1.0
    absolute_target_rot_alpha: float = 1.0
    absolute_target_max_pos_step: float = 0.0
    absolute_target_max_rot_step_deg: float = 0.0
    absolute_target_pos_deadband: float = 0.0
    absolute_target_rot_deadband_deg: float = 0.0
    absolute_target_still_pos_alpha: float = 1.0
    absolute_target_still_rot_alpha: float = 1.0
    absolute_target_pos_alpha_motion_window: float = 0.03
    absolute_target_rot_alpha_motion_window_deg: float = 8.0

    # ── Visualization ──
    enable_visualization: bool = True
    visualize_target_positions: bool = True
    visualize_commanded_target_positions: bool = True

    # ── Runtime ──
    zero_out_xy_rotation: bool = False
    # Legacy hand-tracking knobs (unused by Vuer motion-controller path).
    use_wrist_rotation: bool = True
    use_wrist_position: bool = True
