"""WebSocket master-arm teleop device for EX001-6R joint control."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlencode

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Ex0016RMasterCfg:
    """Configuration for the EX001-6R master-arm remote device."""

    remote_ip: str = "127.0.0.1"
    remote_port: int = 5555
    remote_token: str = ""
    sim_device: str | None = None
    reconnect_interval: float = 2.0
    debug: bool = False
    joint_signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    joint_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class Ex0016RMasterTeleop:
    """Remote master-arm device for EX001-6R master-slave teleoperation."""

    def __init__(self, cfg: Ex0016RMasterCfg):
        """Create the background receiver and keyboard callbacks."""
        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._additional_callbacks: dict[str, Callable[[], None]] = {}
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._last_received = 0.0
        self._connected = False
        self._joint_signs = np.array(cfg.joint_signs, dtype=np.float64)
        self._joint_offsets = np.array(cfg.joint_offsets, dtype=np.float64)
        self._debug_counter = 0
        self._latest_record = torch.zeros(12, dtype=torch.float32, device=self._sim_device)
        self._ws = None
        self._should_run = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        self._keyboard_sub = None
        self._input = None
        self._keyboard = None
        self._setup_keyboard_listener()

    def __del__(self) -> None:
        """Stop the background receiver thread."""
        self._should_run = False

    def close(self) -> None:
        self._should_run = False
        if self._ws is not None:
            self._ws.close()
        if self._thread.is_alive():
            self._thread.join(timeout=6)
        if self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __str__(self) -> str:
        """Return a short description for logging."""
        status = "connected" if self._connected else "disconnected"
        lines = [
            f"EX001-6R Master Teleop: {self.__class__.__name__}",
            f"\tRemote    : ws://{self.cfg.remote_ip}:{self.cfg.remote_port}",
            f"\tAuth      : {'enabled' if self.cfg.remote_token else 'disabled'}",
            f"\tStatus    : {status}",
            "\t----------------------------------------------",
            "\tR : discard episode & reset env",
            "\tEnter : mark success, export & reset env",
        ]
        return "\n".join(lines)

    def has_fresh_input(self) -> bool:
        with self._lock:
            return self._connected and self._latest is not None and time.monotonic() - self._last_received < 1.0

    def _recv_loop(self) -> None:
        """Receive remote joint states over WebSocket."""
        import websocket

        token = (
            self.cfg.remote_token
            or os.environ.get("MANAENV_MASTER_TOKEN", "")
            or os.environ.get("ARX_ARM_WS_TOKEN", "")
        )
        query = f"?{urlencode({'token': token})}" if token else ""
        url = f"ws://{self.cfg.remote_ip}:{self.cfg.remote_port}{query}"
        while self._should_run:
            try:
                ws = websocket.create_connection(url, timeout=5)
                self._ws = ws
                self._connected = True
                print(f"[EX001 Master] Connected to ws://{self.cfg.remote_ip}:{self.cfg.remote_port}")
                while self._should_run:
                    raw = ws.recv()
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        continue
                    if not all("joint_pos" in (data.get(side) or {}) for side in ("left", "right")):
                        continue
                    if not all(
                        np.asarray(data[side]["joint_pos"]).shape == (7,)
                        and bool(np.isfinite(np.asarray(data[side]["joint_pos"], dtype=float)).all())
                        for side in ("left", "right")
                    ):
                        continue
                    with self._lock:
                        self._latest = data
                        self._last_received = time.monotonic()
            except Exception as e:
                self._ws = None
                if self._connected:
                    print(f"[EX001 Master] Disconnected: {type(e).__name__}")
                self._connected = False
                with self._lock:
                    self._latest = None
                time.sleep(self.cfg.reconnect_interval)

    def _setup_keyboard_listener(self) -> None:
        """Register keyboard callbacks for episode controls."""
        try:
            import carb.input
            import omni.appwindow

            appwindow = omni.appwindow.get_default_app_window()
            input_iface = carb.input.acquire_input_interface()
            keyboard = appwindow.get_keyboard()

            def _on_keyboard_event(event, *args, **kwargs) -> bool:
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    key_name = event.input.name
                    cb = self._additional_callbacks.get(key_name)
                    if cb is not None:
                        cb()
                return True

            self._keyboard_sub = input_iface.subscribe_to_keyboard_events(
                keyboard,
                _on_keyboard_event,
            )
            self._input = input_iface
            self._keyboard = keyboard
        except Exception as e:
            logger.warning("carb.input keyboard listener unavailable: %s", e)

    def reset(self) -> None:
        """Reset device-local state between episodes."""
        pass

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        """Register a callback for a keyboard key."""
        self._additional_callbacks[key] = func

    def send_signal(self, cmd: str, **kwargs) -> None:
        """Send a control signal back to the remote bridge."""
        ws = self._ws
        if ws is None or not self._connected:
            return
        try:
            ws.send(json.dumps({"cmd": cmd, **kwargs}))
        except Exception as exc:
            logger.warning("send_signal(%s) failed: %s", cmd, exc)

    def advance(self) -> torch.Tensor:
        """Return the latest 14D follow-arm joint command."""
        action, _ = self.advance_with_record()
        return action

    def advance_with_record(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return follow-arm command and matching record-only master channels."""
        with self._lock:
            state = self._latest

        if state is None:
            return self._zero_action(), self.latest_record_action()

        left = state.get("left") or {}
        right = state.get("right") or {}
        if "joint_pos" not in left and "joint_pos" not in right:
            return self._zero_action(), self.latest_record_action()

        left_joints, left_grip = self._extract_joint_state(left)
        right_joints, right_grip = self._extract_joint_state(right)

        left_mapped = self._joint_signs * left_joints + self._joint_offsets
        right_mapped = self._joint_signs * right_joints + self._joint_offsets

        if self.cfg.debug:
            self._debug_counter += 1
            if self._debug_counter % 50 == 0:
                print(f"[Master Joint Debug #{self._debug_counter}] L_grip={left_grip:.3f} R_grip={right_grip:.3f}")

        cmd = np.concatenate([left_mapped, [left_grip], right_mapped, [right_grip]])
        action = torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)
        record = self._build_record_vector(left, right)
        with self._lock:
            self._latest_record = record
        return action, record.clone()

    def latest_record_action(self) -> torch.Tensor:
        """Return 12D record-only master EE (6+6) appended after the 14D sim command."""
        with self._lock:
            return self._latest_record.clone()

    def _zero_action(self) -> torch.Tensor:
        return torch.zeros(14, dtype=torch.float32, device=self._sim_device)

    @staticmethod
    def _extract_joint_state(side: dict) -> tuple[np.ndarray, float]:
        """Extract six arm joints and one gripper value from remote state."""
        if "joint_pos" not in side:
            return np.zeros(6, dtype=np.float64), 0.0
        jp = np.array(side["joint_pos"], dtype=np.float64)
        joints = jp[:6]
        grip = jp[6] if len(jp) > 6 else 0.0
        return joints, grip

    def _build_record_vector(self, left: dict, right: dict) -> torch.Tensor:
        """Pack master-side record-only EE channels (position(3)+rotation(3) per arm)."""
        left_ee = self._extract_ee_state(left)
        right_ee = self._extract_ee_state(right)
        record = np.concatenate([left_ee, right_ee])
        return torch.tensor(record, dtype=torch.float32, device=self._sim_device)

    @staticmethod
    def _extract_ee_state(side: dict) -> np.ndarray:
        """Extract optional master EE [position(3), rotation(3)] aligned with x2robot."""
        pos = np.array(side.get("position", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
        rot = np.array(side.get("rotation", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
        if pos.shape[0] < 3:
            pos = np.pad(pos, (0, 3 - pos.shape[0]))
        if rot.shape[0] < 3:
            rot = np.pad(rot, (0, 3 - rot.shape[0]))
        return np.concatenate([pos, rot])
