"""Master-slave teleoperation planner."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch

from manaenv.pipeline.datacollection.events import MARK_SUCCESS, RESET_EPISODE
from manaenv.runtime_owners import mana_embodiment

from .teleop_planner import TeleopPlanner, TeleopPlannerCFG

_MASTER_DEVICE_REGISTRY: dict[str, tuple[str, str, str]] = {
    "ex001_6r": (
        "manaenv.teleop_devices.ex001_6r_master",
        "Ex0016RMasterCfg",
        "Ex0016RMasterTeleop",
    ),
    "ex001": (
        "manaenv.teleop_devices.ex001_master",
        "Ex001MasterCfg",
        "Ex001MasterTeleop",
    ),
}

# embodiment.name → registry key
_EMBODIMENT_DEVICE_MAP: dict[str, str] = {
    "ex001": "ex001",
    "EX001-6R": "ex001_6r",
}


@dataclass
class MasterSlaveTeleopPlannerCFG(TeleopPlannerCFG):
    """Settings for master-arm teleoperation (6R and whole-body)."""

    remote_ip: str = "127.0.0.1"
    remote_port: int = 5555
    remote_token: str = ""
    reconnect_interval: float = 2.0
    debug: bool = False
    joint_signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    joint_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    max_linear_velocity: float = 0.5
    max_angular_velocity: float = 2.0
    # Align with auto-collection / bench: r=0.078, track=0.48.
    # 0.084 / 0.458 are experimental sim-effective parameters.
    wheel_radius: float = 0.078
    wheel_track_width: float = 0.48
    lift_step: float = 0.005
    lift_lower: float = 0.0
    lift_upper: float = 0.78


class MasterSlaveTeleopPlanner(TeleopPlanner):
    """Master-arm teleoperation for joint control (6R and whole-body)."""

    CFG = MasterSlaveTeleopPlannerCFG
    device_name = "master_slave"

    def requested_action_mode(self) -> str:
        return "joint_position"

    def is_control_active(self) -> bool:
        return self._master_controller is not None and self._master_controller.has_fresh_input()

    def __init__(self):
        super().__init__()
        self._master_controller = None
        self._expected_action_dim: int | None = None

    def _init_device(self, env: Any) -> None:
        from manaenv.teleop_devices.keyboard import _get_expected_action_dim

        embodiment = mana_embodiment(env)
        if embodiment is None:
            raise RuntimeError(
                "Master-slave teleop needs a robot embodiment "
                "(set env.embodiment or build via Arena cfg.isaaclab_arena_env)."
            )
        # Env already built with JointActionsCfg + variant clamps. Do not
        # rebuild here: fresh JointActionsCfg() resets clamp_max to the
        # default G-jaw open (1.89) and would desync embodiment metadata
        # from the live ActionManager for H-jaw (open≈5.72).

        device_key = _EMBODIMENT_DEVICE_MAP.get(embodiment.name)
        if device_key is None:
            raise ValueError(
                f"No master device registered for embodiment {embodiment.name!r}. "
                f"Add it to _EMBODIMENT_DEVICE_MAP (available: {sorted(_EMBODIMENT_DEVICE_MAP)})"
            )
        entry = _MASTER_DEVICE_REGISTRY[device_key]
        mod = importlib.import_module(entry[0])
        CfgCls = getattr(mod, entry[1])
        DeviceCls = getattr(mod, entry[2])

        cfg_kwargs: dict[str, Any] = dict(
            remote_ip=self.cfg.remote_ip,
            remote_port=self.cfg.remote_port,
            remote_token=self.cfg.remote_token,
            sim_device=str(env.device),
            reconnect_interval=self.cfg.reconnect_interval,
            debug=self.cfg.debug,
            joint_signs=self.cfg.joint_signs,
            joint_offsets=self.cfg.joint_offsets,
        )
        if device_key == "ex001":
            cfg_kwargs.update(
                max_linear_velocity=self.cfg.max_linear_velocity,
                max_angular_velocity=self.cfg.max_angular_velocity,
                wheel_radius=self.cfg.wheel_radius,
                wheel_track_width=self.cfg.wheel_track_width,
                lift_step=self.cfg.lift_step,
                lift_lower=self.cfg.lift_lower,
                lift_upper=self.cfg.lift_upper,
            )
        self._master_controller = DeviceCls(CfgCls(**cfg_kwargs))

        self._expected_action_dim = _get_expected_action_dim(env)
        self.bind_controller_events(
            self._master_controller,
            {"R": RESET_EPISODE, "ENTER": MARK_SUCCESS},
        )

    def _get_device_action(self, obs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        from manaenv.teleop_devices.keyboard import _map_action_dim

        raw, record = self._master_controller.advance_with_record()
        raw_dim = int(raw.numel())
        record_dim = int(record.numel())
        if self._expected_action_dim is not None and raw_dim + record_dim <= int(self._expected_action_dim):
            raw = torch.cat([raw.reshape(-1), record.reshape(-1)], dim=0)
        if self._expected_action_dim is not None:
            action = _map_action_dim(raw, self._expected_action_dim)
        else:
            action = raw
        return action, {"device": self.device_name}

    def _reset_device(self) -> None:
        if self._master_controller is not None:
            self._master_controller.reset()


__all__ = ["MasterSlaveTeleopPlanner", "MasterSlaveTeleopPlannerCFG"]
