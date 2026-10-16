"""WebSocket master-arm teleop device for EX001 whole-body (base + lift)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from .ex001_6r_master import Ex0016RMasterCfg, Ex0016RMasterTeleop

logger = logging.getLogger(__name__)


@dataclass
class Ex001MasterCfg(Ex0016RMasterCfg):
    """Configuration for the EX001 whole-body master-arm remote device."""

    max_linear_velocity: float = 0.5
    max_angular_velocity: float = 2.0
    # Align with auto-collection / bench: r=0.078, track=0.48.
    # 0.084 / 0.458 are experimental sim-effective parameters.
    wheel_radius: float = 0.078
    wheel_track_width: float = 0.48
    lift_step: float = 0.005
    lift_lower: float = 0.0
    lift_upper: float = 0.78


class Ex001MasterTeleop(Ex0016RMasterTeleop):
    """Remote master-arm device for EX001 whole-body master-slave teleoperation.

    Extends the 6R device with diff-drive base control (from joystick) and
    prismatic lift control (from sw1/sw2 buttons).
    """

    def __init__(self, cfg: Ex001MasterCfg):
        super().__init__(cfg)
        self._wb_cfg = cfg
        self._lift_target = 0.0

    def reset(self) -> None:
        super().reset()
        self._lift_target = 0.0

    def _zero_action(self) -> torch.Tensor:
        return torch.zeros(19, dtype=torch.float32, device=self._sim_device)

    def advance_with_record(self) -> tuple[torch.Tensor, torch.Tensor]:
        arm_action, record = super().advance_with_record()
        base_action = self._compute_base_action()
        lift_action = self._compute_lift_action()
        head_action = torch.zeros(2, dtype=torch.float32, device=self._sim_device)
        action = torch.cat([arm_action, base_action, lift_action, head_action])
        return action, record

    def _compute_base_action(self) -> torch.Tensor:
        with self._lock:
            state = self._latest
        if state is None:
            return torch.zeros(2, dtype=torch.float32, device=self._sim_device)

        base = state.get("base") or {}
        jx = float(base.get("joystick_x", 0.0))
        jy = float(base.get("joystick_y", 0.0))

        linear = jy * self._wb_cfg.max_linear_velocity
        angular = -jx * self._wb_cfg.max_angular_velocity
        r = self._wb_cfg.wheel_radius
        w = self._wb_cfg.wheel_track_width
        left_wheel = (linear - 0.5 * angular * w) / r
        right_wheel = (linear + 0.5 * angular * w) / r
        return torch.tensor(
            [left_wheel, right_wheel],
            dtype=torch.float32,
            device=self._sim_device,
        )

    def _compute_lift_action(self) -> torch.Tensor:
        with self._lock:
            state = self._latest
        if state is None:
            return torch.tensor(
                [self._lift_target],
                dtype=torch.float32,
                device=self._sim_device,
            )

        lift = state.get("lift") or {}
        up = bool(lift.get("up", False))
        down = bool(lift.get("down", False))

        step = self._wb_cfg.lift_step
        if up:
            self._lift_target += step
        if down:
            self._lift_target -= step
        self._lift_target = max(
            self._wb_cfg.lift_lower,
            min(self._wb_cfg.lift_upper, self._lift_target),
        )
        return torch.tensor(
            [self._lift_target],
            dtype=torch.float32,
            device=self._sim_device,
        )
