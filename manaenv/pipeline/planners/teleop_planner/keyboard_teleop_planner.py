"""Keyboard teleoperation planner.

Keyboard emits per-arm delta pose commands. The planner integrates those deltas
into an absolute DiffIK hold target so that when no keys are pressed the last
commanded pose is re-sent every step (prevents arms from sagging under gravity).

Base/chassis wheel velocities are passed through live (zero when idle).
"""

from dataclasses import dataclass
from typing import Any

import torch

from manaenv.pipeline.datacollection.events import MARK_SUCCESS, RESET_EPISODE
from manaenv.runtime_owners import mana_embodiment
from manaenv.utils.motion_utils import world_to_base_frame

from .teleop_planner import TeleopPlanner, TeleopPlannerCFG

_BASE_DIM = 2


@dataclass
class KeyboardTeleopPlannerCFG(TeleopPlannerCFG):
    """Keyboard-specific settings."""

    position_scale: float = 0.005
    rotation_scale: float = 0.05


class KeyboardTeleopPlanner(TeleopPlanner):
    """Bimanual keyboard teleoperation (W/S/A/D + rot + gripper).

    Integrates keyboard deltas into an absolute DiffIK hold pose so the arms
    maintain their position when idle.
    """

    CFG = KeyboardTeleopPlannerCFG
    device_name = "keyboard"

    def __init__(self):
        super().__init__()
        self._keyboard_controller = None
        self._expected_action_dim: int | None = None
        self._keyboard_gripper_targets: dict[int, tuple[float, float]] = {}
        self._gym_env = None
        self._hold_action: torch.Tensor | None = None
        self._ee_frame_names: tuple[str, str] = ("left_ee_frame", "right_ee_frame")
        self._command_body_names: tuple[str, str] | None = None
        self._gripper_joint_names: tuple[str, str] = ("left_arm_gripper", "right_arm_gripper")

    def requested_action_mode(self) -> str | None:
        return "ik_eef_abs"

    def prepare_episode(self, gym_env: Any, obs: dict[str, Any]) -> None:
        super().prepare_episode(gym_env, obs)
        self._gym_env = gym_env
        self._capture_hold_action()

    def _init_device(self, env: Any) -> None:
        from manaenv.teleop_devices.keyboard import (
            BimanualSe3Keyboard,
            BimanualSe3KeyboardCfg,
            _get_expected_action_dim,
        )

        expected_action_dim = _get_expected_action_dim(env)
        embodiment = mana_embodiment(env)
        differential_cfg = getattr(
            embodiment,
            "diff_drive_keyboard_controller_cfg",
            None,
        )
        cfg = BimanualSe3KeyboardCfg(
            pos_sensitivity=self.cfg.position_scale,
            rot_sensitivity=self.cfg.rotation_scale,
            sim_device=str(env.device),
            differential_cfg=differential_cfg,
        )
        self._keyboard_controller = BimanualSe3Keyboard(cfg)
        self._expected_action_dim = expected_action_dim
        self._keyboard_gripper_targets = self._resolve_keyboard_gripper_targets(embodiment)
        self._ee_frame_names = self._resolve_ee_frame_names(embodiment)
        self._command_body_names = self._resolve_command_body_names(embodiment)
        self._gripper_joint_names = self._resolve_gripper_joint_names(embodiment)

        self.bind_controller_events(
            self._keyboard_controller,
            {"R": RESET_EPISODE, "ENTER": MARK_SUCCESS},
        )

    # ── EE frame / gripper resolution ───────────────────────────────────

    def _resolve_ee_frame_names(self, embodiment: Any) -> tuple[str, str]:
        getter = getattr(embodiment, "get_vr_ee_frame_names", None)
        if callable(getter):
            return getter()
        return ("left_ee_frame", "right_ee_frame")

    def _resolve_gripper_joint_names(self, embodiment: Any) -> tuple[str, str]:
        getter = getattr(embodiment, "get_vr_gripper_joint_names", None)
        if callable(getter):
            return getter()
        return ("left_arm_gripper", "right_arm_gripper")

    def _resolve_command_body_names(self, embodiment: Any) -> tuple[str, str] | None:
        getter = getattr(embodiment, "get_command_body_names", None)
        if callable(getter):
            names = tuple(getter())
            if len(names) == 2:
                return names

        action_cfg = getattr(embodiment, "action_config", None)
        left_name = getattr(getattr(action_cfg, "arm_action", None), "body_name", None)
        right_name = getattr(getattr(action_cfg, "right_arm_action", None), "body_name", None)
        if left_name and right_name:
            return str(left_name), str(right_name)
        return None

    def _resolve_keyboard_gripper_targets(self, embodiment: Any) -> dict[int, tuple[float, float]]:
        """Map keyboard gripper sign commands to raw joint targets when needed."""
        clamp = getattr(embodiment, "get_vr_gripper_clamp", None)
        if not callable(clamp):
            action_cfg = getattr(embodiment, "action_config", None)
            if action_cfg is None:
                return {}
            index_to_term = {
                6: getattr(action_cfg, "master_left_gripper_action", None)
                or getattr(action_cfg, "gripper_action", None),
                13: getattr(action_cfg, "master_right_gripper_action", None)
                or getattr(action_cfg, "right_gripper_action", None),
            }
            targets: dict[int, tuple[float, float]] = {}
            for index, term in index_to_term.items():
                clamp_min = getattr(term, "clamp_min", None)
                clamp_max = getattr(term, "clamp_max", None)
                if clamp_min is None or clamp_max is None:
                    continue
                targets[index] = (float(clamp_min), float(clamp_max))
            return targets

        bounds = clamp() or {}
        names = self._resolve_gripper_joint_names(embodiment)
        out: dict[int, tuple[float, float]] = {}
        if names[0] in bounds:
            out[6] = tuple(float(v) for v in bounds[names[0]])
        if names[1] in bounds:
            out[13] = tuple(float(v) for v in bounds[names[1]])
        return out

    # ── Hold-pose capture and delta integration ─────────────────────────

    def _normalize_quat(self, quat: torch.Tensor) -> torch.Tensor:
        return quat / torch.linalg.norm(quat).clamp_min(1e-8)

    def _rotvec_to_quat_xyzw(self, rotvec: torch.Tensor) -> torch.Tensor:
        import isaaclab.utils.math as math_utils

        angle = torch.linalg.norm(rotvec)
        if float(angle) == 0.0:
            device = rotvec.device
            return torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        axis = rotvec / angle
        return math_utils.quat_from_angle_axis(angle, axis).reshape(4)

    def _ee_world_poses(self):
        if self._command_body_names is not None:
            robot = self._gym_env.scene["robot"]
            poses = []
            for body_name in self._command_body_names:
                body_ids, body_names = robot.find_bodies(body_name)
                if len(body_ids) != 1:
                    raise RuntimeError(
                        f"Keyboard teleop expected one robot body matching {body_name!r}, got {body_names}."
                    )
                body_id = body_ids[0]
                poses.extend(
                    (
                        robot.data.body_pos_w.torch[0, body_id, :].clone(),
                        robot.data.body_quat_w.torch[0, body_id, :].clone(),
                    )
                )
            return tuple(poses)

        left = self._gym_env.scene[self._ee_frame_names[0]]
        right = self._gym_env.scene[self._ee_frame_names[1]]
        poses = (
            left.data.target_pos_w.torch[0, 0, :].clone(),
            left.data.target_quat_w.torch[0, 0, :].clone(),
            right.data.target_pos_w.torch[0, 0, :].clone(),
            right.data.target_quat_w.torch[0, 0, :].clone(),
        )
        if not all(bool(torch.isfinite(value).all().item()) for value in poses) or any(
            float(torch.linalg.norm(quat).item()) < 1.0e-8 for quat in (poses[1], poses[3])
        ):
            raise RuntimeError(
                "Keyboard teleop cannot initialize its hold pose because the end-effector frame data is not ready."
            )
        return poses

    def _read_gripper(self, robot, name: str) -> float:
        ids, _ = robot.find_joints(name)
        return float(robot.data.joint_pos.torch[0, ids[0]].item())

    def _capture_hold_action(self) -> None:
        """Snapshot current EE poses as the initial absolute hold target."""
        if self._gym_env is None:
            return
        robot = self._gym_env.scene["robot"]
        root_pos = robot.data.root_pos_w.torch[0].clone()
        root_quat = robot.data.root_quat_w.torch[0].clone()
        lpw, lqw, rpw, rqw = self._ee_world_poses()
        lpb, lqb = world_to_base_frame(lpw, lqw, root_pos, root_quat)
        rpb, rqb = world_to_base_frame(rpw, rqw, root_pos, root_quat)
        device = root_pos.device
        lg = torch.tensor([self._read_gripper(robot, self._gripper_joint_names[0])], device=device)
        rg = torch.tensor([self._read_gripper(robot, self._gripper_joint_names[1])], device=device)
        parts = [
            lpb.reshape(-1),
            self._normalize_quat(lqb.reshape(-1)),
            lg,
            rpb.reshape(-1),
            self._normalize_quat(rqb.reshape(-1)),
            rg,
        ]
        if self._expected_action_dim is not None and self._expected_action_dim > 16:
            parts.append(torch.zeros(_BASE_DIM, dtype=torch.float32, device=device))
        lift_name = getattr(mana_embodiment(self._gym_env), "get_vr_lift_joint_name", None)
        if callable(lift_name):
            parts.append(torch.tensor([self._read_gripper(robot, lift_name())], device=device))
        self._hold_action = torch.cat(parts)

    def _integrate_arm_delta(
        self,
        hold: torch.Tensor,
        pos_slice: slice,
        quat_slice: slice,
        dpos: torch.Tensor,
        drot: torch.Tensor,
    ) -> None:
        import isaaclab.utils.math as math_utils

        if float(dpos.abs().max()) > 0.0:
            hold[pos_slice] = hold[pos_slice] + dpos
        if float(drot.abs().max()) > 0.0:
            delta_q = self._rotvec_to_quat_xyzw(drot)
            current_q = hold[quat_slice]
            hold[quat_slice] = self._normalize_quat(
                math_utils.quat_mul(delta_q.unsqueeze(0), current_q.unsqueeze(0)).squeeze(0)
            )

    def _integrate_keyboard_into_hold(self, raw: torch.Tensor) -> torch.Tensor:
        if self._hold_action is None:
            self._capture_hold_action()
        if self._hold_action is None:
            return raw
        hold = self._hold_action.clone()
        self._integrate_arm_delta(hold, slice(0, 3), slice(3, 7), raw[0:3], raw[3:6])
        hold[7:8] = raw[6:7]
        self._integrate_arm_delta(hold, slice(8, 11), slice(11, 15), raw[7:10], raw[10:13])
        hold[15:16] = raw[13:14]
        if hold.shape[-1] > 16 and raw.shape[-1] >= 16:
            hold[16:18] = raw[14:16]
        elif hold.shape[-1] > 16:
            hold[16:18] = 0.0
        self._hold_action = hold
        return hold

    # ── Device action / reset ───────────────────────────────────────────

    def _get_device_action(
        self,
        obs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        from manaenv.teleop_devices.keyboard import _map_action_dim

        raw = self._keyboard_controller.advance()
        raw = self._apply_keyboard_gripper_targets(raw)
        action = self._integrate_keyboard_into_hold(raw)
        if self._expected_action_dim is not None:
            action = _map_action_dim(action, self._expected_action_dim)
        return action, {"device": self.device_name}

    def _reset_device(self) -> None:
        if self._keyboard_controller is not None:
            self._keyboard_controller.reset()
        self._hold_action = None

    def _apply_keyboard_gripper_targets(self, action: torch.Tensor) -> torch.Tensor:
        """Convert keyboard toggle states into raw gripper joint positions."""
        if not self._keyboard_gripper_targets:
            return action
        remapped = action.clone()
        for index, (close_val, open_val) in self._keyboard_gripper_targets.items():
            if remapped.shape[-1] <= index:
                continue
            remapped[index] = open_val if float(remapped[index]) > 0.0 else close_val
        return remapped


__all__ = ["KeyboardTeleopPlanner", "KeyboardTeleopPlannerCFG"]
