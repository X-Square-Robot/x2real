"""Mana-specific Arena environment registration."""

from __future__ import annotations

from typing import Any

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

from manaenv.environments.physx_async_env import ManaPhysxAsyncKitAsyncEnv
from manaenv.pipeline.runtime_cfg import (
    KIT_RENDER_MODE_ASYNC_HANDSHAKE,
    KIT_RENDER_MODE_SYNC,
    KIT_RENDER_MODES,
    PHYSICS_STEP_MODE_SYNC,
    PHYSICS_STEP_MODES,
    runtime_device_type,
)
from manaenv.utils.viewport_utils import attach_task_visualizer_cfg


class ManaArenaEnvBuilder(ArenaEnvBuilder):
    """Select one environment class for each supported physics/Kit schedule."""

    def __init__(
        self,
        *args: Any,
        physics_step_mode: str = PHYSICS_STEP_MODE_SYNC,
        kit_render_mode: str = KIT_RENDER_MODE_SYNC,
        **kwargs: Any,
    ) -> None:
        if physics_step_mode not in PHYSICS_STEP_MODES:
            raise ValueError(f"Unsupported physics_step_mode: {physics_step_mode!r}")
        if kit_render_mode not in KIT_RENDER_MODES:
            raise ValueError(f"Unsupported kit_render_mode: {kit_render_mode!r}")
        if physics_step_mode != PHYSICS_STEP_MODE_SYNC and kit_render_mode == KIT_RENDER_MODE_SYNC:
            raise ValueError(
                "physics_step_mode='async' has no overlap with kit_render_mode='sync'; "
                "use sync/sync or async/async_handshake"
            )
        self.physics_step_mode = physics_step_mode
        self.kit_render_mode = kit_render_mode
        super().__init__(*args, **kwargs)
        if self.cfg.mimic and kit_render_mode != KIT_RENDER_MODE_SYNC:
            raise ValueError("kit_render_mode='async_handshake' is not supported with mimic environments")
        if kit_render_mode == KIT_RENDER_MODE_ASYNC_HANDSHAKE and runtime_device_type(self.cfg.device) != "cpu":
            raise ValueError("kit_render_mode='async_handshake' currently supports only CPU PhysX")

    def compose_manager_cfg(self) -> tuple[Any, dict[str, Any]]:
        """Compose the Arena cfg and attach Mana's native visualizer camera hint."""
        env_cfg, env_kwargs = super().compose_manager_cfg()
        attach_task_visualizer_cfg(
            env_cfg,
            getattr(getattr(self, "arena_env", None), "task", None),
            visualizer_cfg=getattr(self, "_resolved_task_visualizer_cfg", None),
        )
        return env_cfg, env_kwargs

    def get_entry_point(self):
        """Resolve the schedule once, before Gym constructs the environment."""
        if self.physics_step_mode == PHYSICS_STEP_MODE_SYNC:
            return super().get_entry_point()
        if self.kit_render_mode == KIT_RENDER_MODE_ASYNC_HANDSHAKE:
            return ManaPhysxAsyncKitAsyncEnv
        raise ValueError(f"Unsupported kit_render_mode: {self.kit_render_mode!r}")
