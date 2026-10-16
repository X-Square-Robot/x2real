"""Base teleoperation planner with task-success detection."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from manaenv.pipeline.datacollection.events import MARK_SUCCESS, RESET_EPISODE, CollectionEvent, CollectionEventQueue
from manaenv.pipeline.planners.planner_base import PlannerBase, PlannerCFG


@dataclass
class TeleopPlannerCFG(PlannerCFG):
    """Common settings shared by all teleop planners."""

    max_steps: int | None = None
    position_scale: float = 0.05
    rotation_scale: float = 0.5
    min_recorded_actions_for_export: int = 5


class TeleopPlanner(PlannerBase):
    """Base teleop planner with automatic task-success detection.

    Reads ``task.termination_cfg.success`` (a TerminationTermCfg)
    and calls its ``func(gym_env, **params)`` each step. When the
    condition fires, ``is_done()`` and ``is_success()`` return True
    so that the datacollection loop exports the episode.

    Subclasses override ``_init_device`` / ``_get_device_action`` /
    ``_reset_device`` to plug in their own input hardware.
    """

    CFG = TeleopPlannerCFG
    device_name = "teleop"

    def __init__(self):
        super().__init__()
        self.step = 0
        self.last_action: np.ndarray | None = None
        self.done_signal = False
        self.success_signal = False
        self._torch_device: torch.device | None = None
        self._success_term = None
        self._event_queue = CollectionEventQueue()

    def setup(self, env: Any, task: Any) -> bool:
        self._read_success_term(task)
        return True

    def prepare_episode(self, gym_env: Any, obs: dict[str, Any]) -> None:
        self._torch_device = gym_env.device

    def _step_single(
        self,
        gym_env: Any,
        obs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Advance one step: check success then poll device."""
        self.step += 1
        return self._get_device_action(obs)

    def set_action(self, action: np.ndarray | torch.Tensor) -> None:
        """Inject action from an external source."""
        if isinstance(action, torch.Tensor):
            self.last_action = action.detach().cpu().numpy()
        else:
            self.last_action = action

    def signal_done(self, success: bool = False) -> None:
        """Compatibility path for planners that still own completion state."""
        self.done_signal = True
        self.success_signal = success

    def emit_event(self, name: str, **payload: Any) -> None:
        self._event_queue.emit(name, **payload)

    def drain_events(self) -> list[CollectionEvent]:
        return self._event_queue.drain()

    def bind_controller_events(self, controller: Any, event_map: dict[str, str] | None = None) -> None:
        self._event_queue.bind_controller_callbacks(controller, event_map)

    def emit_reset_event(self) -> None:
        self.emit_event(RESET_EPISODE)

    def emit_success_event(self) -> None:
        self.emit_event(MARK_SUCCESS)

    def is_done(self) -> bool:
        return self.done_signal

    def is_success(self) -> bool:
        return self.success_signal

    def reset(self, success: bool | None = None) -> None:
        self.step = 0
        self.last_action = None
        self.done_signal = False
        self.success_signal = False
        self._mana_success_cache = None
        self._reset_device()

    def _init_device(self, env: Any) -> None:
        """Override to initialize hardware / controller."""

    def _get_device_action(
        self,
        obs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Override to return ``(action_tensor, info_dict)``."""
        if self.last_action is None:
            self.last_action = np.zeros(8)
        device = self._resolve_torch_device(obs)
        action = torch.as_tensor(
            self.last_action,
            dtype=torch.float32,
            device=device,
        ).clone()
        return action, {"device": self.device_name}

    def _reset_device(self) -> None:
        """Override to reset device state between episodes."""

    def cleanup(self) -> None:
        for name in ("_keyboard_controller", "_master_controller"):
            controller = getattr(self, name, None)
            close = getattr(controller, "close", None)
            if callable(close):
                close()

    def _read_success_term(self, task: Any) -> None:
        # Old-style tasks expose .success on the termination configclass
        term = getattr(task.get_termination_cfg(), "success", None)
        # New modular tasks expose it via get_success_term_cfg()
        if term is None and hasattr(task, "get_success_term_cfg"):
            term = task.get_success_term_cfg()
        self._success_term = term

    def _check_task_success(self, gym_env: Any) -> None:
        if self.done_signal or gym_env is None or self._success_term is None:
            return
        cached = getattr(self, "_mana_success_cache", None)
        tick = getattr(gym_env, "common_step_counter", None)
        if cached is not None and tick is not None and cached[0] == int(tick):
            # Reuse the AFTER_STEP evaluation for this tick. Stateful terms
            # (consecutive_steps / hold_seconds dwell) must run at most once
            # per physics tick or the dwell counter double-advances / resets.
            any_success = cached[1]
        else:
            any_success = bool(self._success_term.func(gym_env, **self._success_term.params).any())
        if any_success:
            self.success_signal = True
            self.done_signal = True

    # ------------------------------------------------------------------
    #  Batch-first interface (teleop is always num_envs=1)
    # ------------------------------------------------------------------

    def get_actions(self, env: Any, obs: dict[str, Any]) -> torch.Tensor:
        """``(1, act_dim)`` — teleop is single-env only."""
        result, _ = self._step_single(env, obs)
        if result is None:
            act_dim = env.action_manager.action.shape[-1]
            return torch.zeros(1, act_dim, device=env.device)
        a = result.to(device=env.device, dtype=torch.float32)
        return a.unsqueeze(0) if a.ndim == 1 else a

    def on_reset(self, env, obs, env_ids, env_id_to_success=None):
        self.reset()
        self.prepare_episode(env, obs)

    def initial_plan(self, env, settle_steps=30):
        return None

    def _resolve_torch_device(self, obs: dict[str, Any]) -> torch.device:
        if self._torch_device is not None:
            return self._torch_device
        for value in (obs or {}).values():
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")


__all__ = ["TeleopPlanner", "TeleopPlannerCFG"]
