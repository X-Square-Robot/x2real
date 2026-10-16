"""Collection camera previews and display-only gripper-tip crosshairs."""

from __future__ import annotations

import time
from typing import Any

from manaenv.pipeline.runtime.env_runner import (
    AFTER_STEP,
    BEFORE_STEP,
    EPISODE_REBUILD,
    SESSION_END,
    GymEnvRunner,
)
from manaenv.utils.camera_utils import iter_camera_cfg_items
from manaenv.utils.debug_print import manaprint
from manaenv.utils.viewport_crosshair import attach_tip_crosshairs_to_viewports
from manaenv.utils.viewport_utils import close_viewport_windows, open_extra_viewports, resolve_sensor_render_product_paths


class _FpsMeter:
    """Measure wall-clock simulation-step rate over a short rolling interval."""

    def __init__(self, *, interval_s: float = 0.5, clock=time.monotonic) -> None:
        self.interval_s = max(float(interval_s), 0.1)
        self._clock = clock
        self._started_at: float | None = None
        self._samples = 0
        self.total_steps = 0

    def tick(self) -> float | None:
        now = float(self._clock())
        self.total_steps += 1
        if self._started_at is None:
            self._started_at = now
            self._samples = 1
            return None
        self._samples += 1
        elapsed = now - self._started_at
        if elapsed < self.interval_s:
            return None
        fps = self._samples / elapsed
        self._started_at = now
        self._samples = 0
        return fps


class _EmaFpsMeter:
    """Measure env-step rate with the same EMA used by dev-6.0 teleop."""

    def __init__(self, *, report_interval_s: float = 5.0, clock=time.perf_counter) -> None:
        self.report_interval_s = max(float(report_interval_s), 0.1)
        self._clock = clock
        self._last_tick_at: float | None = None
        self._ema_dt_s: float | None = None
        self._next_report_at = 0.0
        self.total_steps = 0

    @property
    def ema_dt_s(self) -> float | None:
        return self._ema_dt_s

    @property
    def last_tick_at(self) -> float | None:
        return self._last_tick_at

    def tick(self) -> float | None:
        """Record one completed env step and return the current EMA FPS."""
        now = float(self._clock())
        self.total_steps += 1
        previous = self._last_tick_at
        self._last_tick_at = now
        if previous is None:
            return None
        dt = now - previous
        if dt <= 0.0:
            return None
        self._ema_dt_s = dt if self._ema_dt_s is None else 0.9 * self._ema_dt_s + 0.1 * dt
        return 1.0 / self._ema_dt_s

    def report_due(self) -> bool:
        """Return true at the same five-second cadence as the dev-6.0 log."""
        if self._last_tick_at is None or self._ema_dt_s is None:
            return False
        if self._last_tick_at < self._next_report_at:
            return False
        self._next_report_at = self._last_tick_at + self.report_interval_s
        return True


class _SimFpsOverlay:
    """The dev-6.0 ``Sim FPS`` Kit window used by manual collection."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        import omni.ui as ui

        stream = config if isinstance(config, dict) else {}
        raw_cfg = stream.get("fps_window", {})
        cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
        width = 230
        canvas_width = int(stream.get("canvas_width", 1920))
        self._window = ui.Window(
            "Sim FPS",
            width=width,
            height=64,
            position_x=int(cfg.get("position_x", max(canvas_width - width, 0))),
            position_y=int(cfg.get("position_y", 0)),
        )
        with self._window.frame:
            self._label = ui.Label(
                "-- FPS",
                style={"font_size": 22, "color": 0xFF00FF00},
            )

    def update(self, fps: float, dt_ms: float) -> None:
        self._label.text = f"{fps:5.1f} FPS({dt_ms:.2f}ms)"

    def destroy(self) -> None:
        window, self._window = self._window, None
        if window is not None:
            window.destroy()


def _fps_overlay_enabled() -> bool:
    """Read the same opt-in Kit setting as the dev-6.0 teleop loop."""
    try:
        import carb

        return bool(carb.settings.get_settings().get("/manaenv/stream/fps_overlay"))
    except Exception:
        # Unit-level doubles do not provide Carb; keep the display available
        # when a real Kit setting cannot be queried.
        return True


def register_collection_viewports(runner: GymEnvRunner, *, environment: Any) -> None:
    """Own camera windows until their Gym scene is rebuilt or the session ends."""
    windows: list[Any] = []
    crosshairs: list[Any] = []
    shown_env: Any = None
    fps_meter = _EmaFpsMeter()
    fps_overlay: _SimFpsOverlay | None = None
    next_overlay_write = 0.0

    def close(_ctx: Any) -> None:
        nonlocal shown_env, fps_overlay, next_overlay_write
        for handle in reversed(crosshairs):
            handle.destroy()
        crosshairs.clear()
        close_viewport_windows(windows, logger=manaprint)
        windows.clear()
        if fps_overlay is not None:
            fps_overlay.destroy()
            fps_overlay = None
        next_overlay_write = 0.0
        shown_env = None

    def show(ctx: Any, _obs: Any) -> None:
        nonlocal shown_env, fps_overlay
        gym = ctx.gym_env
        if shown_env is gym:
            return
        close(ctx)
        shown_env = gym
        visualizers = getattr(getattr(gym, "sim", None), "visualizers", ())
        if not any(
            getattr(getattr(viz, "cfg", None), "visualizer_type", None) == "kit"
            and not getattr(viz, "_runtime_headless", False)
            for viz in visualizers
        ):
            return

        bundle = getattr(environment.embodiment, "camera_config", None)
        camera_cfgs = dict(iter_camera_cfg_items(bundle))
        stream = getattr(getattr(environment, "runtime_cfg", None), "stream_viewports", None) or {}
        fps_config = stream.get("fps_window", {})
        if fps_config is not False and _fps_overlay_enabled():
            try:
                fps_overlay = _SimFpsOverlay(stream)
                manaprint("INFO: Sim FPS overlay window created")
            except Exception as exc:
                manaprint(f"WARNING: Sim FPS overlay unavailable: {exc}")
        requests = stream.get("cameras") or ["left_wrist_camera", "head_camera", "right_wrist_camera"]
        cameras: list[tuple[str, str]] = []
        sizes: list[tuple[int, int]] = []
        sensors = gym.scene.sensors
        for request in requests:
            item = {"name": request} if isinstance(request, str) else request
            name = item["name"]
            if name not in camera_cfgs or name not in sensors:
                continue
            path = str(camera_cfgs[name].prim_path).replace("{ENV_REGEX_NS}", "/World/envs/env_0")
            cameras.append((name, path))
            sizes.append((int(item.get("width", stream.get("width", 640))),
                          int(item.get("height", stream.get("height", 480)))))
        if not cameras:
            return
        canvas_width = stream.get("canvas_width")
        total_width = sum(width for width, _ in sizes)
        if canvas_width and total_width > int(canvas_width):
            scale = int(canvas_width) / total_width
            sizes = [(max(1, int(width * scale)), max(1, int(height * scale))) for width, height in sizes]

        products = resolve_sensor_render_product_paths(gym, cameras)
        if products is None:
            manaprint("WARNING: Collection camera previews require initialized sensor render products")
            return
        try:
            windows.extend(open_extra_viewports(
                count=len(cameras),
                name_prefix="Collection Camera",
                camera_paths=[path for _, path in cameras],
                render_product_paths=products,
                sizes=sizes,
                position_x=int(stream.get("position_x", 0)),
                position_y=int(stream.get("position_y", 0)),
                canvas_height=stream.get("canvas_height"),
                row_vertical_align=stream.get("row_vertical_align", "top"),
                vertical_align=stream.get("vertical_align", "top"),
                allow_legacy_fallback=False,
                logger=manaprint,
            ))
            crosshairs.extend(attach_tip_crosshairs_to_viewports(gym, windows, camera_bundle=bundle))
        except Exception as exc:
            close(ctx)
            shown_env = gym
            manaprint(f"WARNING: Collection camera preview setup failed: {exc}")

    def update_fps(ctx: Any, *_args: Any) -> None:
        nonlocal next_overlay_write
        fps = fps_meter.tick()
        if fps is None:
            return
        now = fps_meter.last_tick_at or 0.0
        if fps_overlay is not None and now >= next_overlay_write:
            dt_ms = (fps_meter.ema_dt_s or 0.0) * 1000.0
            fps_overlay.update(fps, dt_ms)
            next_overlay_write = now + 0.5
        if fps_meter.report_due():
            dt_ms = (fps_meter.ema_dt_s or 0.0) * 1000.0
            manaprint(
                f"MANA_METRIC sim_fps={fps:.2f}({dt_ms:.2f}ms) "
                f"total_frames={int(getattr(ctx, 'step_idx', fps_meter.total_steps))}"
            )

    # BEFORE_STEP runs after reset observations, including after a scene rebuild.
    runner.add_hook(BEFORE_STEP, show, priority=10)
    runner.add_hook(AFTER_STEP, update_fps, priority=110)
    runner.add_hook(EPISODE_REBUILD, close, priority=-10)
    runner.add_hook(SESSION_END, close, priority=50)
