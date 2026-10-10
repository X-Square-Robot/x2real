"""Lazy IsaacLab visualizer adapter for protected asynchronous Kit pumping."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from isaaclab.scene_data import SceneDataProvider
    from isaaclab.sim import SimulationContext


class _RenderVisualizer(Protocol):
    """Visualizer interface consumed by the async PhysX render scheduler."""

    @property
    def is_closed(self) -> bool: ...

    def flush_startup_messages(self) -> None: ...

    def get_rendering_dt(self) -> float | None: ...

    def supports_markers(self) -> bool: ...

    def pumps_app_update(self) -> bool: ...

    def is_running(self) -> bool: ...

    def is_rendering_paused(self) -> bool: ...

    def is_training_paused(self) -> bool: ...

    def close(self) -> None: ...

    def requires_forward_before_step(self) -> bool: ...

    def step(self, dt: float) -> None: ...


@lru_cache(maxsize=1)
def _async_kit_pump_visualizer_type() -> Callable[[SimulationContext], _RenderVisualizer]:
    """Create the IsaacLab-backed adapter type only when rendering installs it."""
    from isaaclab.visualizers.base_visualizer import BaseVisualizer

    class _AsyncKitPumpVisualizer(BaseVisualizer):
        """Minimal Kit visualizer for headless camera pipelines."""

        def __init__(self, sim: SimulationContext) -> None:
            super().__init__(cfg=None)
            self._sim = sim
            self._is_initialized = True

        def initialize(self, scene_data_provider: SceneDataProvider) -> None:
            self._set_scene_data_provider(scene_data_provider)
            self._is_initialized = True

        def step(self, dt: float) -> None:
            del dt
            import omni.kit.app

            app = omni.kit.app.get_app()
            if app is None or not app.is_running():
                raise RuntimeError("Kit app is not running during asynchronous render")
            self._sim.set_setting("/app/player/playSimulations", False)
            try:
                app.update()
            finally:
                self._sim.set_setting("/app/player/playSimulations", True)
            if torch.device(self._sim.device).type == "cuda":
                torch.cuda.set_device(self._sim.device)

        def close(self) -> None:
            self._is_initialized = False
            self._is_closed = True

        def is_running(self) -> bool:
            # The adapter exists only for one protected render. Keeping it
            # active prevents fallback to an unguarded Kit app update.
            return True

        def pumps_app_update(self) -> bool:
            return True

        def requires_forward_before_step(self) -> bool:
            return False

    return _AsyncKitPumpVisualizer


def create_async_kit_pump_visualizer(sim: SimulationContext) -> _RenderVisualizer:
    """Construct the lazily imported IsaacLab visualizer adapter."""
    return _async_kit_pump_visualizer_type()(sim)
