"""Mana Arena environment with a guarded CPU PhysX / Kit scheduler.

Enable overlap with ``physics_step_mode: async`` and
``kit_render_mode: async_handshake``. ``ManaArenaEnvBuilder`` selects
``ManaPhysxAsyncKitAsyncEnv`` for that pair; CUDA PhysX and the non-overlapping
async/sync pair are rejected.

On the overlapping path, the caller publishes Fabric snapshot ``S_n`` and arms
camera tickets. The worker owns the complete decimation physics chain while the
main thread submits Kit/Hydra against frozen ``S_n``. Per-substep Fabric
writeback stays disabled, and the caller joins before camera capture, manager
work, or publication of ``S_n+1``. No native render work may cross the public
``step()`` boundary.

Camera observations and camera pre-step recorder terms consume ``S_n``;
non-camera observations and post-step state recording consume ``S_n+1``.
Kit features that may read live PhysX, such as collider/debug visualization or
main-thread recorder terms, force render-before-physics serialization instead.

``close()`` drains rendering, joins the worker, and restores Fabric before a
scene epoch clears its ``SimulationContext``. This remains an experimental
schedule without a renderer-consumed snapshot fence, so optional Kit extensions
require workload-specific validation. ``MANAENV_STEP_PHASE_PROBE=<frames>``
bounds a run and prints its timing report.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, cast

import torch
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv

from manaenv.environments._async_kit_visualizer import _RenderVisualizer, create_async_kit_pump_visualizer
from manaenv.pipeline.runtime_cfg import KIT_RENDER_MODE_ASYNC_HANDSHAKE, KIT_RENDER_MODE_SYNC, PHYSICS_STEP_MODE_ASYNC
from manaenv.utils.debug_print import manaprint
from manaenv.utils.isaac_extensions import (
    PHYSICS_COLLIDER_VISUALIZATION_SETTING,
    PHYSICS_COLLIDER_VISUALIZATION_UI_SETTING,
)
from manaenv.utils.physx_utils import ManaPhysxContractError
from manaenv.utils.tracy_zone import tracy_zone

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.envs import VecEnvObs, VecEnvStepReturn
    from isaaclab.managers import ObservationManager, RecorderTerm
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
    from isaaclab_arena.variations.variation_recorder import VariationRecorder
    from isaaclab_physx.physics import PhysxCfg

    from manaenv.pipeline.runtime.env_runner import EnvRunContext, GymEnvRunner

_WARMUP_FRAMES = 30
# Retained timing samples per series. Bounds memory and ``_report``'s sort on
# long datagen runs while keeping far more than the warm-up discard needs.
_SAMPLE_WINDOW = 20000
_FABRIC_TRANSFORM_SETTING = "/physics/fabricUpdateTransformations"
# The worker now runs a whole decimation loop, so the join bound scales with it.
_WORKER_TIMEOUT_S = 30.0
_KIT_RENDER_TIMEOUT_S = 30.0

_RenderReady = Callable[[], bool]
_RenderCapture = Callable[[], bool]
_RenderCancel = Callable[[], None]
_CameraHandoff = tuple[_RenderReady | None, _RenderCapture, _RenderCancel | None]
_RenderArmedCheck = Callable[[], bool]
_ResetParams = ParamSpec("_ResetParams")
_ResetResult = TypeVar("_ResetResult")
type _RecorderValue = torch.Tensor | dict[str, _RecorderValue]
type _RecorderHookResult = tuple[str | None, _RecorderValue | None]


class _PrepareExternalRender(Protocol):
    """Camera method that optionally arms a completion ticket before rendering."""

    def __call__(self, *, wait_for_completion: bool = False) -> bool:
        pass


class _SubstepTimings:
    """Per-step accumulators the worker fills and the caller reads after joining."""

    __slots__ = ("physics", "scene")

    def __init__(self) -> None:
        self.physics = 0.0
        self.scene = 0.0


@dataclass(slots=True)
class _PendingRender:
    """One armed or submitted Kit frame tied to its published PhysX state."""

    frame_token: int
    handoffs: list[_CameraHandoff]
    armed_checks: list[_RenderArmedCheck]
    deadline: float | None
    submitted: bool = False


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _rate_hz(period_ms: float) -> float:
    """Convert a per-frame duration to a rate, tolerating degenerate samples."""
    return 1e3 / period_ms if period_ms > 0.0 else float("inf")


def _name_this_thread(name: str) -> None:
    # Label the OS thread (PR_SET_NAME = 15) — helps `top`/`ps`/gdb.
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.prctl(15, name.encode()[:15], 0, 0, 0)
    except (AttributeError, OSError) as exc:
        logger.debug("Native worker thread naming is unavailable: %s", exc)
    # Tracy reads thread names via its own protocol, not prctl; resolve the
    # already-loaded plugin (skips cleanly when no tracy backend is active).
    try:
        so_path = None
        with open("/proc/self/maps") as maps:
            for line in maps:
                if "libcarb.profiler-tracy.plugin.so" in line:
                    so_path = line.rsplit(" ", 1)[-1].strip()
                    break
        if so_path:
            lib = ctypes.CDLL(so_path)
            fn = getattr(lib, "___tracy_set_thread_name")
            fn.argtypes = [ctypes.c_char_p]
            fn(name.encode())
    except (AttributeError, OSError, ctypes.ArgumentError) as exc:
        logger.debug("Tracy worker thread naming is unavailable: %s", exc)


class _State:
    def __init__(self) -> None:
        self.mode: str | None = None
        self.kit_render_mode = KIT_RENDER_MODE_SYNC
        self.worker_render_submit_allowed = False
        self.ready = False
        self.render: Callable[[], None] | None = None
        self.pending_render: _PendingRender | None = None
        # Diagnostic samples only. Bounded so a multi-hour datagen run neither
        # grows without limit nor makes ``_report``'s sort scale with frames;
        # the report is a steady-state summary, so the newest window suffices.
        self.step_count = 0  # every step ever run, unlike len(step_ms)
        self.step_ms: deque[float] = deque(maxlen=_SAMPLE_WINDOW)
        self.frame_ms: deque[float] = deque(maxlen=_SAMPLE_WINDOW)  # public step-entry interval
        self.last_frame_t: float | None = None
        self.upd_ts: deque[float] = deque(maxlen=_SAMPLE_WINDOW)
        self.tail_ms: deque[float] = deque(maxlen=_SAMPLE_WINDOW)  # obs/camera reads, recorders
        self.obs_ms: deque[float] = deque(maxlen=_SAMPLE_WINDOW)  # observation_manager.compute() alone
        self.publish_ms: deque[float] = deque(maxlen=_SAMPLE_WINDOW)  # Fabric publish per boundary
        self.join_ms: deque[float] = deque(maxlen=_SAMPLE_WINDOW)  # main blocked waiting on the worker
        self.physics_ms: deque[float] = deque(maxlen=_SAMPLE_WINDOW)  # worker PhysicsManager.step, per step
        self.scene_ms: deque[float] = deque(maxlen=_SAMPLE_WINDOW)  # scene.update + write_data_to_sim, per step
        self.overlapped = 0  # frames scheduled for native RTX / CPU PhysX overlap
        self.collider_sync_warned = False
        self.marker_sync_warned = False
        self.recorder_sync_warned = False
        self.render_cadence_warned = False
        self.reported = False
        # Persistent worker: main sets work_ready, worker runs work_fn, sets work_done.
        self.worker: threading.Thread | None = None
        self.work_ready = threading.Event()
        self.work_done = threading.Event()
        self.work_fn: Callable[[], None] | None = None
        self.worker_err: BaseException | None = None
        self.worker_alive = True
        # Scene-epoch identity: the SimulationContext this state is bound to.
        # ``active_only`` rebuilds clear the old instance, so a differing sim
        # means every cached handle below points at freed physics/USD state.
        self.sim: SimulationContext | None = None
        # Save the temporary GPU Fabric override for teardown restoration.
        self.fabric_setting_sim: SimulationContext | None = None
        self.fabric_setting_saved = False
        self.fabric_setting_value: bool | None = None
        # ``PhysxManager.step`` may otherwise re-enable per-substep Fabric
        # writeback via ``fabric_skip``. Async mode owns publication explicitly.
        self.fabric_skip_cfg: PhysxCfg | None = None
        self.fabric_skip_saved = False
        self.fabric_skip_value: bool | None = None
        self.fabric_guard_depth = 0


class _SnapshotVisualizerProxy:
    """Make one upstream visualizer consume the explicitly published snapshot.

    ``SimulationContext.render()`` normally calls ``forward()`` when a Kit
    visualizer requests it. This scheduler already publishes Fabric after the
    PhysX worker joins, so the proxy suppresses only that redundant forward and
    delegates the rest of the visualizer contract unchanged.
    """

    def __init__(self, visualizer: _RenderVisualizer) -> None:
        self.visualizer = visualizer
        self.error: BaseException | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self.visualizer, name)

    @property
    def is_closed(self) -> bool:
        return self.visualizer.is_closed

    def flush_startup_messages(self) -> None:
        self.visualizer.flush_startup_messages()

    def get_rendering_dt(self) -> float | None:
        return self.visualizer.get_rendering_dt()

    def supports_markers(self) -> bool:
        return self.visualizer.supports_markers()

    def pumps_app_update(self) -> bool:
        return self.visualizer.pumps_app_update()

    def is_running(self) -> bool:
        return self.visualizer.is_running()

    def is_rendering_paused(self) -> bool:
        return self.visualizer.is_rendering_paused()

    def is_training_paused(self) -> bool:
        return self.visualizer.is_training_paused()

    def close(self) -> None:
        self.visualizer.close()

    def requires_forward_before_step(self) -> bool:
        return False

    def step(self, dt: float) -> None:
        # SimulationContext deliberately removes visualizers that raise. Keep
        # this proxy installed until recording hooks finish so the protected
        # render lifecycle cannot fall back to a redundant forward(); surface
        # the original error immediately after render() returns instead.
        try:
            self.visualizer.step(dt)
        except BaseException as exc:  # noqa: BLE001 - defer across the protected render lifecycle
            self.error = exc


def _simulation_visualizers(sim: SimulationContext) -> list[_RenderVisualizer]:
    """Return the public visualizer collection, with a legacy test-double fallback."""
    visualizers = getattr(sim, "visualizers", None)
    if visualizers is not None:
        return cast(list[_RenderVisualizer], visualizers)
    return cast(list[_RenderVisualizer], sim._visualizers)


def _is_cuda_device(device: str | torch.device) -> bool:
    """Return whether a device string or ``torch.device`` targets CUDA."""
    return torch.device(device).type == "cuda"


def _worker_owns_scene(st: _State) -> bool:
    """Return whether a submitted worker task may be reading or mutating PhysX."""
    return st.work_ready.is_set() or (st.work_fn is not None and not st.work_done.is_set())


def _kit_app_is_running() -> bool:
    """Return the Kit app state without making Kit imports mandatory in unit tests."""
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        return app is not None and bool(app.is_running())
    except (ImportError, AttributeError, RuntimeError):
        return True


def _require_kit_app_running() -> None:
    """Stop an async render before it reads resources invalidated by window close."""
    if not _kit_app_is_running():
        raise ReferenceError("Kit app closed during asynchronous render")


def _assert_cpu_worker_chain(env: ManaPhysxAsyncKitAsyncEnv, sim: SimulationContext) -> None:
    """Reject known split-device controls before the async worker starts."""
    scene_cfg = getattr(getattr(env, "scene", None), "cfg", None)
    if getattr(scene_cfg, "lazy_sensor_update", True) is not True:
        raise ManaPhysxContractError(
            "kit_render_mode='async_handshake' requires scene.lazy_sensor_update=true so the physics "
            "worker cannot enter RenderContext or recompute RTX sensors"
        )
    try:
        _assert_worker_sensor_updates_are_lazy(getattr(env, "scene", None))
    except RuntimeError as exc:
        # At installation this is an unsupported environment contract. The
        # same helper remains a RuntimeError when visualization is activated
        # later while the worker is already running.
        raise ManaPhysxContractError(str(exc)) from exc

    sim_cfg = getattr(sim, "cfg", None)
    actuator_device = getattr(sim_cfg, "newton_actuator_device", None) or sim.device
    if torch.device(actuator_device).type != "cpu":
        raise ManaPhysxContractError(
            "kit_render_mode='async_handshake' requires CPU actuator execution; "
            f"resolved actuator_device={actuator_device!r}"
        )
    if getattr(sim_cfg, "newton_actuator_cuda_graph", None) is True:
        raise ManaPhysxContractError("kit_render_mode='async_handshake' cannot enable actuator CUDA Graphs")

    action_terms = getattr(getattr(env, "action_manager", None), "_terms", {})
    action_term_items = action_terms.items() if isinstance(action_terms, dict) else ()
    for name, term in action_term_items:
        control_device = getattr(term, "_control_device", None)
        if control_device is not None and torch.device(control_device).type != "cpu":
            raise ManaPhysxContractError(
                "kit_render_mode='async_handshake' requires CPU action controllers; "
                f"term {name!r} resolved control_device={control_device!r}"
            )


def _base_decimation_recorder_hook() -> Callable[[RecorderTerm], _RecorderHookResult]:
    """Return the base decimation hook required to classify worker overrides."""
    try:
        from isaaclab.managers.recorder_manager import RecorderTerm
    except Exception as exc:
        raise RuntimeError("cannot resolve RecorderTerm's decimation hook for worker-safety validation") from exc
    hook = getattr(RecorderTerm, "record_post_physics_decimation_step", None)
    if not callable(hook):
        raise TypeError("cannot resolve RecorderTerm's decimation hook for worker-safety validation")
    return cast(Callable[[RecorderTerm], _RecorderHookResult], hook)


def _worker_decimation_recorder_terms(env: ManaPhysxAsyncKitAsyncEnv) -> tuple[list[str], list[str]]:
    """Split overriding decimation recorder terms into declared and undeclared.

    ``RecorderManager.record_post_physics_decimation_step`` runs inside the
    worker's substep chain. Terms that only read physics buffers are correctly
    sequenced there; terms that touch USD, Kit or the renderer are not. Every
    override must therefore state which it is through a
    ``runs_on_physics_worker`` attribute.
    """
    manager = getattr(env, "recorder_manager", None)
    terms = getattr(manager, "_terms", None)
    if not isinstance(terms, dict) or not terms:
        return [], []
    base_hook = _base_decimation_recorder_hook()
    if base_hook is None:
        raise RuntimeError("cannot resolve RecorderTerm's decimation hook for worker-safety validation")
    main_thread_terms: list[str] = []
    undeclared: list[str] = []
    for name, term in terms.items():
        hook = getattr(type(term), "record_post_physics_decimation_step", None)
        if hook is None or hook is base_hook:
            continue
        declared = getattr(term, "runs_on_physics_worker", None)
        if declared is None:
            undeclared.append(str(name))
        elif not declared:
            main_thread_terms.append(str(name))
    return main_thread_terms, undeclared


def _assert_worker_decimation_recorder_terms_are_declared(env: ManaPhysxAsyncKitAsyncEnv) -> list[str]:
    """Return terms that opted out of the worker, rejecting undeclared overrides."""
    main_thread_terms, undeclared = _worker_decimation_recorder_terms(env)
    if undeclared:
        raise RuntimeError(
            "physics_step_mode='async' runs record_post_physics_decimation_step inside the PhysX "
            "worker's substep chain; set runs_on_physics_worker on "
            f"{sorted(undeclared)} to declare whether those terms may run there"
        )
    return main_thread_terms


def _assert_worker_sensor_updates_are_lazy(scene: InteractiveScene | None) -> None:
    """Prevent SensorBase.update() from forcing renderer work on the worker."""
    sensors = getattr(scene, "sensors", None)
    sensor_items = sensors.items() if isinstance(sensors, Mapping) else ()
    visualized = [str(name) for name, sensor in sensor_items if bool(getattr(sensor, "_is_visualizing", False))]
    if visualized:
        raise RuntimeError(
            "kit_render_mode='async_handshake' cannot update visualized sensors on the physics worker; "
            f"disable debug visualization for {visualized}"
        )


def _stop_worker(st: _State) -> None:
    """Join the physics worker so no thread holds the PxScene across teardown."""
    worker = st.worker
    st.worker_alive = False
    st.work_ready.set()  # unblock the loop's wait() so it can observe the flag
    if worker is not None:
        try:
            worker.join(timeout=10.0)
            if worker.is_alive():
                # The bounded join is only a diagnostic: a live worker may still
                # hold the outgoing PxScene, so teardown must wait for a real
                # exit before SimulationContext.clear_instance().
                manaprint("WARNING: PhysX step worker is still running; waiting before scene teardown")
                worker.join()
        except Exception as exc:  # noqa: BLE001 - teardown must not mask shutdown
            manaprint(f"WARNING: PhysX step worker join failed: {exc}; retrying without a timeout")
            try:
                worker.join()
            except Exception as join_exc:
                raise RuntimeError("PhysX step worker could not be joined safely") from join_exc
    st.worker = None
    st.work_fn = None
    st.worker_render_submit_allowed = False
    st.work_ready.clear()
    st.work_done.clear()


def _restore_fabric_setting(st: _State) -> None:
    """Restore the GPU Fabric writeback setting overridden by this runtime."""
    if not st.fabric_setting_saved:
        return
    sim = st.fabric_setting_sim
    try:
        value = st.fabric_setting_value
        if sim is not None and value is not None:
            sim.set_setting(_FABRIC_TRANSFORM_SETTING, value)
    except Exception as exc:  # noqa: BLE001 - teardown must not mask shutdown
        manaprint(f"WARNING: PhysX step executor could not restore Fabric setting: {exc}")
    finally:
        st.fabric_setting_sim = None
        st.fabric_setting_saved = False
        st.fabric_setting_value = None


def _restore_fabric_skip_config(st: _State) -> None:
    """Restore the upstream ``fabric_skip`` policy overridden by async mode."""
    if not st.fabric_skip_saved:
        return
    try:
        cfg = st.fabric_skip_cfg
        value = st.fabric_skip_value
        if cfg is not None and value is not None:
            cfg.fabric_skip = value
    finally:
        st.fabric_skip_cfg = None
        st.fabric_skip_saved = False
        st.fabric_skip_value = None


def _release_epoch_state(st: _State, *, drain_pending_render: bool = False) -> None:
    """Drop every handle that a scene-epoch rebuild invalidates.

    Mirrors ``NewtonManager.reset(soft=False)``: the render call closes over the
    old ``sim``, so it points at freed state after a rebuild.
    """
    _stop_worker(st)
    try:
        if drain_pending_render and st.pending_render is not None:
            _finish_pending_render(st, st.pending_render.frame_token)
        else:
            _cancel_pending_render(st)
    finally:
        _restore_fabric_setting(st)
        _restore_fabric_skip_config(st)
        st.fabric_guard_depth = 0
        st.ready = False
        st.render = None
        st.worker_err = None
        st.sim = None


def _worker_loop(st: _State, device: str | torch.device) -> None:
    _name_this_thread("mana-physics")
    try:
        if _is_cuda_device(device):
            torch.cuda.set_device(device)
    except BaseException as exc:  # noqa: BLE001 - relay every initialization failure to main
        st.worker_err = exc
        st.worker_alive = False
        st.work_done.set()
        return
    while st.worker_alive:
        st.work_ready.wait()
        st.work_ready.clear()
        if not st.worker_alive:
            break
        st.worker_err = None
        try:
            work_fn = st.work_fn
            if work_fn is None:
                raise RuntimeError("PhysX step worker woke without a task")
            work_fn()
        except BaseException as exc:  # noqa: BLE001 - relay every worker failure to main after join
            st.worker_err = exc
        st.work_done.set()


def _make_render(sim: SimulationContext, st: _State) -> Callable[[], None]:
    """Build a published-snapshot render through ``SimulationContext.render``.

    App-pumping visualizers are temporarily represented by proxies that suppress
    their normal pre-render ``forward()`` request. Standalone visualizers remain
    outside the Kit pump exactly as before.
    """
    injected_kit_pump = create_async_kit_pump_visualizer(sim)
    # Capture the class-bound implementation once. Reset temporarily replaces
    # ``sim.render`` with a ticket-aware wrapper; this closure must still reach
    # the underlying SimulationContext lifecycle without recursing.
    simulation_render = sim.render

    def _render() -> None:
        _require_kit_app_running()
        if _worker_owns_scene(st) and not st.worker_render_submit_allowed:
            raise RuntimeError("Cannot pump Kit app.update() while the PhysX worker owns the live scene")
        visualizers = _simulation_visualizers(sim)
        original_visualizers = list(visualizers)
        app_visualizers = [viz for viz in original_visualizers if viz.pumps_app_update()]
        delegates = app_visualizers or [injected_kit_pump]
        proxies = [_SnapshotVisualizerProxy(viz) for viz in delegates]
        proxy_originals = {id(proxy): original for proxy, original in zip(proxies, app_visualizers, strict=False)}
        proxy_ids = {id(proxy) for proxy in proxies}
        original_ids = {id(viz) for viz in original_visualizers}
        visualizers[:] = proxies
        t0 = time.perf_counter()
        try:
            simulation_render()
        finally:
            st.upd_ts.append(time.perf_counter() - t0)
            surviving_original_ids = {
                id(proxy_originals[id(proxy)]) for proxy in visualizers if id(proxy) in proxy_originals
            }
            added_visualizers = [viz for viz in visualizers if id(viz) not in proxy_ids and id(viz) not in original_ids]
            visualizers[:] = [
                viz for viz in original_visualizers if not viz.pumps_app_update() or id(viz) in surviving_original_ids
            ]
            visualizers.extend(added_visualizers)

        _require_kit_app_running()
        for proxy in proxies:
            if proxy.error is not None:
                raise proxy.error

    return _render


def _physics_collider_visualization_may_activate(sim: SimulationContext) -> bool:
    """Return whether a Kit update may read the live scene for debug drawing.

    The UI marker matters even while colliders are currently hidden: a menu
    click is processed inside ``app.update()``, after scheduling has started.
    """
    for setting in (
        PHYSICS_COLLIDER_VISUALIZATION_UI_SETTING,
        PHYSICS_COLLIDER_VISUALIZATION_SETTING,
    ):
        try:
            if int(sim.get_setting(setting) or 0) != 0:
                return True
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return False


def _marker_callbacks_may_read_physx(sim: SimulationContext) -> bool:
    """Return whether a Kit update may dispatch debug-vis marker callbacks.

    ``SimulationContext.render`` dispatches every registered
    ``vis_marker_registry`` callback on the calling thread whenever a
    marker-capable visualizer is installed. Those callbacks are registered only
    by ``set_debug_vis(True)`` owners — assets, action terms, command terms — and
    read their physics buffers, so they cannot run while the worker owns the
    live scene.
    """
    registry = getattr(sim, "vis_marker_registry", None)
    if not getattr(registry, "_callbacks", None):
        return False
    try:
        visualizers = _simulation_visualizers(sim)
    except AttributeError:
        return True
    for viz in visualizers:
        supports_markers = getattr(viz, "supports_markers", None)
        if not callable(supports_markers):
            continue
        try:
            has_marker_support = bool(supports_markers())
        except Exception:  # noqa: BLE001 - an unqueryable visualizer cannot claim markers
            has_marker_support = False
        if has_marker_support:
            return True
    return False


def _cancel_camera_handoffs(handoffs: list[_CameraHandoff]) -> None:
    """Cancel every camera ticket belonging to one submitted Kit frame."""
    for _ready, _capture, cancel in handoffs:
        if cancel is not None:
            try:
                cancel()
            except Exception as exc:  # noqa: BLE001 - cleanup must continue across all cameras
                manaprint(f"WARNING: Could not cancel an asynchronous camera render ticket: {exc}")


def _cancel_pending_render(st: _State) -> None:
    """Cancel and forget the submitted frame owned by *st*, if any."""
    pending = st.pending_render
    st.pending_render = None
    if pending is not None:
        _cancel_camera_handoffs(pending.handoffs)


def _arm_camera_handoffs(
    st: _State,
    env: ManaPhysxAsyncKitAsyncEnv,
) -> tuple[list[_CameraHandoff], list[_RenderArmedCheck], float | None]:
    """Arm camera tickets without submitting or pumping a Kit frame."""
    sensors = getattr(env.scene, "sensors", None)
    async_handshake = st.kit_render_mode == KIT_RENDER_MODE_ASYNC_HANDSHAKE
    handoffs: list[_CameraHandoff] = []
    armed_checks: list[_RenderArmedCheck] = []
    try:
        if sensors is not None:
            for sensor in sensors.values():
                prepare = getattr(sensor, "prepare_for_external_render", None)
                capture = getattr(sensor, "capture_after_external_render", None)
                if not callable(prepare) or not callable(capture):
                    continue
                ready = getattr(sensor, "external_render_is_ready", None)
                cancel = getattr(sensor, "cancel_external_render", None)
                if async_handshake and (not callable(ready) or not callable(cancel)):
                    continue
                prepare_for_render = cast(_PrepareExternalRender, prepare)
                capture_after_render = cast(_RenderCapture, capture)
                ready_check = cast(_RenderReady, ready) if callable(ready) else None
                cancel_render = cast(_RenderCancel, cancel) if callable(cancel) else None
                prepared = prepare_for_render(wait_for_completion=True) if async_handshake else prepare_for_render()
                if prepared:
                    handoffs.append((ready_check, capture_after_render, cancel_render))
                    is_armed = getattr(sensor, "external_render_is_armed", None)
                    if async_handshake and callable(is_armed):
                        armed_checks.append(cast(_RenderArmedCheck, is_armed))

        if async_handshake and not handoffs:
            raise RuntimeError(
                "kit_render_mode='async_handshake' requires at least one active ManaAsyncCamera render ticket"
            )

        deadline = time.monotonic() + _KIT_RENDER_TIMEOUT_S if async_handshake else None
        return handoffs, armed_checks, deadline
    except BaseException:
        _cancel_camera_handoffs(handoffs)
        raise


def _begin_render_with_camera_handoffs(
    st: _State,
    env: ManaPhysxAsyncKitAsyncEnv,
) -> tuple[list[_CameraHandoff], list[_RenderArmedCheck], float | None]:
    """Arm camera tickets and submit one render from the published state."""
    handoffs, armed_checks, deadline = _arm_camera_handoffs(st, env)
    render = st.render
    if render is None:
        _cancel_camera_handoffs(handoffs)
        raise RuntimeError("Simulation render callback is not installed")
    try:
        render()
    except BaseException:
        _cancel_camera_handoffs(handoffs)
        raise
    return handoffs, armed_checks, deadline


def _finish_render_with_camera_handoffs(
    st: _State,
    handoffs: list[_CameraHandoff],
    deadline: float | None,
) -> None:
    """Wait for one submitted frame, then expose it to camera observations."""
    async_handshake = st.kit_render_mode == KIT_RENDER_MODE_ASYNC_HANDSHAKE
    render = st.render
    if render is None:
        _cancel_camera_handoffs(handoffs)
        raise RuntimeError("Simulation render callback is not installed")
    try:
        _require_kit_app_running()
        if async_handshake:
            if deadline is None:
                raise RuntimeError("Async Kit render handoff is missing its completion deadline")
            ready_checks = [ready for ready, _capture, _cancel in handoffs if ready is not None]
            if len(ready_checks) != len(handoffs):
                raise RuntimeError("Async Kit render handoff is missing a completion check")
            while not all(ready() for ready in ready_checks):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out after {_KIT_RENDER_TIMEOUT_S:.3f}s pumping Kit for all render products"
                    )
                render()
        _require_kit_app_running()
        for _ready, capture, _cancel in handoffs:
            capture()
    except BaseException:
        _cancel_camera_handoffs(handoffs)
        raise


def _render_with_camera_handoffs(st: _State, env: ManaPhysxAsyncKitAsyncEnv) -> None:
    """Submit and synchronously finish one render/camera handoff."""
    handoffs, _armed_checks, deadline = _begin_render_with_camera_handoffs(st, env)
    _finish_render_with_camera_handoffs(st, handoffs, deadline)


def _begin_pending_render(st: _State, env: ManaPhysxAsyncKitAsyncEnv, frame_token: int) -> None:
    """Submit one frame from an already-published PhysX snapshot."""
    _arm_pending_render(st, env, frame_token)
    _submit_pending_render(st, frame_token)


def _arm_pending_render(st: _State, env: ManaPhysxAsyncKitAsyncEnv, frame_token: int) -> None:
    """Create one camera handoff while leaving Kit submission deferred."""
    if st.pending_render is not None:
        raise RuntimeError("Cannot submit a Kit frame while another render handoff is pending")
    handoffs, armed_checks, deadline = _arm_camera_handoffs(st, env)
    st.pending_render = _PendingRender(
        frame_token=frame_token,
        handoffs=handoffs,
        armed_checks=armed_checks,
        deadline=deadline,
    )


def _submit_pending_render(st: _State, frame_token: int, *, cancel_on_error: bool = True) -> None:
    """Submit a previously armed camera handoff for *frame_token*."""
    pending = _require_pending_render(st, frame_token)
    if pending.submitted:
        raise RuntimeError(f"Pending Kit render for PhysX frame {frame_token} was already submitted")
    render = st.render
    if render is None:
        _cancel_pending_render(st)
        raise RuntimeError("Simulation render callback is not installed")
    try:
        render()
    except BaseException:
        if cancel_on_error:
            _cancel_pending_render(st)
        raise
    pending.submitted = True


def _pending_render_is_armed(pending: _PendingRender) -> bool:
    """Return whether external sensor resets left every camera ticket intact."""
    return all(check() for check in pending.armed_checks)


def _require_pending_render(st: _State, frame_token: int) -> _PendingRender:
    """Return the pending frame only when it belongs to the current PhysX state."""
    pending = st.pending_render
    if pending is None:
        raise RuntimeError(f"Missing pending Kit render for PhysX frame {frame_token}")
    if pending.frame_token != frame_token:
        _cancel_pending_render(st)
        raise RuntimeError(
            f"Pending Kit render frame token {pending.frame_token} does not match current PhysX frame {frame_token}"
        )
    return pending


def _finish_pending_render(st: _State, frame_token: int) -> None:
    """Complete the submitted frame for *frame_token* and clear its ownership."""
    with tracy_zone("manaenv/finish_render", source="physx_async_env.py"):
        pending = _require_pending_render(st, frame_token)
        if not pending.submitted:
            _cancel_pending_render(st)
            raise RuntimeError(f"Pending Kit render for PhysX frame {frame_token} was not submitted")
        st.pending_render = None
        _finish_render_with_camera_handoffs(st, pending.handoffs, pending.deadline)


def _is_camera_recorder_term(name: str, term: RecorderTerm) -> bool:
    """Classify pre-step terms that consume a completed camera observation."""
    marker = getattr(term, "requires_rendered_camera_frame", None)
    if marker is not None:
        return bool(marker)
    return "camera" in name.lower() or "camera" in type(term).__name__.lower()


def _record_pre_step_terms(env: ManaPhysxAsyncKitAsyncEnv, *, camera_terms: bool) -> None:
    """Run one partition of RecorderManager's pre-step terms."""
    zone = "manaenv/record_pre_camera" if camera_terms else "manaenv/record_pre"
    with tracy_zone(zone, source="physx_async_env.py"):
        manager = env.recorder_manager
        terms = getattr(manager, "_terms", None)
        add_to_episodes = getattr(manager, "add_to_episodes", None)
        if not isinstance(terms, dict) or not callable(add_to_episodes):
            if not camera_terms:
                manager.record_pre_step()
            return

        for name, term in terms.items():
            if _is_camera_recorder_term(str(name), term) != camera_terms:
                continue
            key, value = term.record_pre_step()
            add_to_episodes(key, value)


def _camera_observation_groups(observation_manager: ObservationManager) -> tuple[str, ...]:
    """Return configured observation groups that consume camera products."""
    active_terms = getattr(observation_manager, "active_terms", None)
    if not isinstance(active_terms, dict):
        return ()
    return tuple(str(name) for name in active_terms if "camera" in str(name).lower())


def _cache_observations(observation_manager: ObservationManager, observations: VecEnvObs) -> None:
    """Keep ObservationManager's cache coherent when groups are computed separately."""
    if hasattr(observation_manager, "_obs_buffer"):
        observation_manager._obs_buffer = observations


def _refresh_camera_observations(env: ManaPhysxAsyncKitAsyncEnv, *, update_history: bool) -> bool:
    """Consume the completed external frame without recomputing state groups."""
    with tracy_zone("manaenv/camera_obs", source="physx_async_env.py"):
        observations = getattr(env, "obs_buf", None)
        manager = env.observation_manager
        compute_group = getattr(manager, "compute_group", None)
        groups = _camera_observation_groups(manager)
        if not isinstance(observations, dict) or not callable(compute_group) or not groups:
            return False
        for group in groups:
            observations[group] = compute_group(group, update_history=update_history)
        _cache_observations(manager, observations)
        return True


def _compute_observations_around_camera_frame(
    env: ManaPhysxAsyncKitAsyncEnv,
    *,
    update_history: bool,
    recompute_camera: bool,
) -> VecEnvObs:
    """Advance state observation groups while keeping camera history single-shot.

    Camera groups already consumed the completed external frame and pushed their
    history in ``_refresh_camera_observations``. A regular step retains those
    tensors; a reset step recomputes them from the post-reset frame but must not
    push a second history entry for the same step.
    """
    manager = env.observation_manager
    previous = getattr(env, "obs_buf", None)
    active_terms = getattr(manager, "active_terms", None)
    compute_group = getattr(manager, "compute_group", None)
    camera_groups = set(_camera_observation_groups(manager))
    if (
        not isinstance(previous, dict)
        or not isinstance(active_terms, dict)
        or not callable(compute_group)
        or not camera_groups
    ):
        return manager.compute(update_history=update_history)

    observations: VecEnvObs = {}
    for group in active_terms:
        group_name = str(group)
        if group_name not in camera_groups:
            observations[group_name] = compute_group(group_name, update_history=update_history)
        elif recompute_camera:
            observations[group_name] = compute_group(group_name, update_history=False)
        elif group_name in previous:
            observations[group_name] = previous[group_name]
        else:
            observations[group_name] = compute_group(group_name, update_history=update_history)
    _cache_observations(manager, observations)
    return observations


def _step_physics_backend(sim: SimulationContext) -> None:
    """Advance one backend step while preserving SimulationContext's counter.

    IsaacLab exposes backend-only stepping through ``PhysicsManager.step()`` but
    currently increments its render freshness counter only in
    ``SimulationContext.step()``. Calling the latter on the worker would also
    execute visualizer pause gates there and publish the new counter before
    fetch completes. Keep the unavoidable private write isolated here until
    upstream exposes a split-phase step API.
    """
    completed_steps = sim.get_physics_step_count()
    sim.physics_manager.step()
    sim._physics_step_count = completed_steps + 1


def _report_enabled(bench_frames: int) -> bool:
    """Session-end timing report is opt-in; ``MANAENV_STEP_PHASE_PROBE`` implies it."""
    override = (
        os.environ.get(
            "MANAENV_PHYSICS_STEP_REPORT",
            os.environ.get("MANAENV_ASYNC_PHYSICS_REPORT", ""),
        )
        .strip()
        .lower()
    )
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    return bool(bench_frames)


def _steady_slice(samples: deque[float], steady: int) -> list[float]:
    """Drop the leading warm-up samples, keeping everything when too few remain.

    The series are bounded deques, so a long run has already evicted the real
    warm-up frames — the discard is then a no-op on data that is all steady.
    """
    values = list(samples)
    return values[steady:] or values


def _report(st: _State) -> None:
    if st.reported or not st.step_ms:
        return
    st.reported = True
    # Warm-up discard applies to the retained window, not the whole run.
    steady = min(_WARMUP_FRAMES, len(st.step_ms) // 5)
    steady_ms = _steady_slice(st.step_ms, steady)
    discarded = len(st.step_ms) - len(steady_ms)
    w = sorted(steady_ms)
    if not w:
        return
    med = _percentile(w, 0.5)
    fast_ms = min(steady_ms)
    fast_idx = discarded + steady_ms.index(fast_ms)
    upd = sorted(list(st.upd_ts)[-len(steady_ms) :]) if st.upd_ts else []
    manaprint(f"INFO: ===== PhysX step report (mode={st.mode}) =====")
    manaprint(
        f"INFO:   frames={st.step_count} (last {len(st.step_ms)} retained, "
        f"steady from #{discarded}), overlap_scheduled={st.overlapped}"
    )
    manaprint(
        f"INFO:   per-frame wall ms: median={med:.2f} mean={sum(w) / len(w):.2f} "
        f"p95={_percentile(w, 0.95):.2f} ({_rate_hz(med):.2f} it/s)"
    )
    if upd:
        manaprint(f"INFO:   render ms: median={_percentile(upd, 0.5) * 1e3:.2f}")
    if st.tail_ms:
        tail = sorted(_steady_slice(st.tail_ms, steady))
        obs = sorted(_steady_slice(st.obs_ms, steady))
        manaprint(
            f"INFO:   post-step tail ms: median={_percentile(tail, 0.5):.2f} "
            f"(obs compute median={_percentile(obs, 0.5):.2f})"
        )
    if st.publish_ms or st.join_ms:
        pub = sorted(_steady_slice(st.publish_ms, steady)) if st.publish_ms else []
        join = sorted(_steady_slice(st.join_ms, steady)) if st.join_ms else []
        phys = sorted(_steady_slice(st.physics_ms, steady)) if st.physics_ms else []
        scene = sorted(_steady_slice(st.scene_ms, steady)) if st.scene_ms else []
        manaprint(
            f"INFO:   in-step detail ms: worker physics median={_percentile(phys, 0.5):.2f} "
            f"| scene write/update median={_percentile(scene, 0.5):.2f}"
        )
        manaprint(
            f"INFO:                      fabric publish median={_percentile(pub, 0.5):.2f} "
            f"| worker join median={_percentile(join, 0.5):.2f}"
        )
    if st.frame_ms:
        f = sorted(_steady_slice(st.frame_ms, steady))
        fmed = _percentile(f, 0.5)
        manaprint(
            f"INFO:   full-frame interval ms (A/B metric): median={fmed:.2f} "
            f"p95={_percentile(f, 0.95):.2f} ({_rate_hz(fmed):.2f} it/s)"
        )
    manaprint(
        f"INFO:   fastest steady frame: #{fast_idx} at {fast_ms:.2f}ms ({_rate_hz(fast_ms):.2f} it/s) "
        f"| slowest steady: {max(steady_ms):.2f}ms"
    )
    manaprint("INFO: ===========================================================")


def _run_post_step_managers(
    st: _State,
    env: ManaPhysxAsyncKitAsyncEnv,
    sim: SimulationContext,
    *,
    async_render: bool,
    sim_rendering: bool,
) -> None:
    """Run the manager tail after physics and the regular Fabric publication."""
    reset_happened = False

    def _ensure_current_render_pending() -> None:
        if async_render and st.pending_render is None:
            _begin_pending_render(st, env, int(sim.get_physics_step_count()))

    try:
        env.episode_length_buf += 1
        env.common_step_counter += 1
        env.reset_buf = env.termination_manager.compute()
        env.reset_terminated = env.termination_manager.terminated
        env.reset_time_outs = env.termination_manager.time_outs
        env.reward_buf = env.reward_manager.compute(dt=env.step_dt)

        if len(env.recorder_manager.active_terms) > 0:
            if env.cfg.recorder_obs_refresh:
                if async_render:
                    _ensure_current_render_pending()
                    _finish_pending_render(st, int(sim.get_physics_step_count()))
                env.obs_buf = env.observation_manager.compute()
            with tracy_zone("manaenv/record_post", source="physx_async_env.py"):
                env.recorder_manager.record_post_step()

        reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            reset_happened = True
            if env.cfg.compute_final_obs:
                if async_render:
                    _ensure_current_render_pending()
                    _finish_pending_render(st, int(sim.get_physics_step_count()))
                env.extras["final_obs"] = env.observation_manager.compute()
            elif async_render and st.pending_render is not None:
                # A native frame may still be using the frozen snapshot even
                # when final observations are not requested. Complete it before
                # reset mutates and republishes Fabric.
                _finish_pending_render(st, int(sim.get_physics_step_count()))
            env.recorder_manager.record_pre_reset(reset_env_ids)
            env._reset_idx(reset_env_ids)
            # Reset events write new root poses after this step's regular Fabric
            # publication. Flush buffered scene state and publish again before any
            # reset rerender or post-reset observation consumes the snapshot.
            env.scene.write_data_to_sim()
            if sim_rendering:
                t_pub = time.perf_counter()
                env.publish_fabric()
                st.publish_ms.append((time.perf_counter() - t_pub) * 1e3)
            if env.render_enabled and sim.is_rendering and env.has_rtx_sensors and env.cfg.num_rerenders_on_reset > 0:
                for _ in range(env.cfg.num_rerenders_on_reset):
                    if async_render:
                        _render_with_camera_handoffs(st, env)
                    else:
                        sim.render()
            env.recorder_manager.record_post_reset(reset_env_ids)

        # IsaacLab once exposed SimulationContext.consume_reset_request for
        # visualizer UI resets; 3.0 no longer has it. Skip when absent.
        consume_reset_request = getattr(sim, "consume_reset_request", None)
        if callable(consume_reset_request) and consume_reset_request():
            not_yet_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            not_yet_reset[reset_env_ids] = False
            manual_reset_ids = not_yet_reset.nonzero(as_tuple=False).squeeze(-1).int()
            if len(manual_reset_ids) > 0:
                reset_happened = True
                env.reset_terminated[manual_reset_ids] = True
                env.recorder_manager.record_pre_reset(manual_reset_ids)
                env._reset_idx(manual_reset_ids)
                # The async executor published Fabric before the manager tail;
                # publish the UI-reset poses before rendering or observations.
                env.scene.write_data_to_sim()
                if sim_rendering:
                    t_pub = time.perf_counter()
                    env.publish_fabric()
                    st.publish_ms.append((time.perf_counter() - t_pub) * 1e3)
                if (
                    env.render_enabled
                    and sim.is_rendering
                    and env.has_rtx_sensors
                    and env.cfg.num_rerenders_on_reset > 0
                ):
                    for _ in range(env.cfg.num_rerenders_on_reset):
                        if async_render:
                            _render_with_camera_handoffs(st, env)
                        else:
                            sim.render()
                env.recorder_manager.record_post_reset(manual_reset_ids)

        env.command_manager.compute(dt=env.step_dt)
        if "interval" in env.event_manager.available_modes:
            env.event_manager.apply(mode="interval", dt=env.step_dt)
        # IsaacLab 3.0 has no environment-owned video recorder collection.
        # Keep compatibility with Arena variants that still provide one.
        for recorder in getattr(env, "video_recorders", ()):
            recorder.step()
        t_obs = time.perf_counter()
        with tracy_zone("manaenv/obs_compute", source="physx_async_env.py"):
            if async_render:
                env.obs_buf = _compute_observations_around_camera_frame(
                    env, update_history=True, recompute_camera=reset_happened
                )
            else:
                env.obs_buf = env.observation_manager.compute(update_history=True)
        st.obs_ms.append((time.perf_counter() - t_obs) * 1e3)

        # Do not pre-submit another frame here. Public ``step()`` returns only
        # after both native Kit rendering and the PhysX worker are quiescent;
        # the next call submits its own snapshot immediately before starting
        # that call's worker interval.
    except BaseException:
        _cancel_pending_render(st)
        raise


def _resolve_render_cadence(
    st: _State,
    env: ManaPhysxAsyncKitAsyncEnv,
    *,
    sim_rendering: bool,
    render_ok: bool,
) -> tuple[int, int, bool, bool]:
    """Resolve IsaacLab's render boundaries and whether overlap is valid."""
    decimation = int(env.cfg.decimation)
    render_interval = int(env.cfg.sim.render_interval)
    if render_interval <= 0:
        raise ValueError(f"cfg.sim.render_interval must be positive; got {render_interval}")
    substep_render_cadence = bool(
        sim_rendering and (render_interval != decimation or env._sim_step_counter % render_interval != 0)
    )
    async_render_requested = render_ok and st.kit_render_mode == KIT_RENDER_MODE_ASYNC_HANDSHAKE
    async_render = async_render_requested and not substep_render_cadence
    if async_render_requested and substep_render_cadence and not st.render_cadence_warned:
        manaprint(
            "WARNING: cfg.sim.render_interval does not align with cfg.decimation; "
            "preserving IsaacLab's substep render cadence without Kit/PhysX overlap "
            f"(render_interval={render_interval}, decimation={decimation})"
        )
        st.render_cadence_warned = True
    return decimation, render_interval, substep_render_cadence, async_render


def _publish_current_fabric_frame(
    st: _State,
    env: ManaPhysxAsyncKitAsyncEnv,
    sim: SimulationContext,
) -> int:
    """Publish the current PhysX state and return its completed step token."""
    t_pub = time.perf_counter()
    env.publish_fabric()
    st.publish_ms.append((time.perf_counter() - t_pub) * 1e3)
    return int(sim.get_physics_step_count())


def _render_current_frame(
    st: _State,
    env: ManaPhysxAsyncKitAsyncEnv,
    sim: SimulationContext,
    *,
    render_ok: bool,
) -> None:
    """Render one configured Kit or standalone-visualizer boundary."""
    if render_ok:
        _render_with_camera_handoffs(st, env)
    else:
        sim.render(skip_app_pumping=True)


def _finish_step_render_cadence(
    st: _State,
    env: ManaPhysxAsyncKitAsyncEnv,
    sim: SimulationContext,
    *,
    sim_rendering: bool,
    render_ok: bool,
    async_render: bool,
    substep_render_cadence: bool,
    last_published_frame: int | None,
) -> None:
    """Publish the final state and render when no substep boundary did so."""
    if not sim_rendering:
        return
    current_frame = int(sim.get_physics_step_count())
    if last_published_frame != current_frame:
        _publish_current_fabric_frame(st, env, sim)
    if not substep_render_cadence and not async_render:
        _render_current_frame(st, env, sim, render_ok=render_ok)


def _raw_step(st: _State, env: ManaPhysxAsyncKitAsyncEnv, action: torch.Tensor) -> VecEnvStepReturn:
    sim = env.sim
    env.action_manager.process_action(action.to(env.device))

    # ``render_enabled`` gates only the Kit pump; headless still needs forward()
    # so Fabric and camera transforms advance.
    sim_rendering = bool(sim.is_rendering)
    render_ok = sim_rendering and bool(getattr(env, "render_enabled", True))
    # The overlap pipeline can batch the whole decimation loop only when its
    # single render lands on the same final boundary as IsaacLab's substep
    # cadence. Other layouts retain exact rendering semantics synchronously.
    decimation, render_interval, substep_render_cadence, async_render = _resolve_render_cadence(
        st,
        env,
        sim_rendering=sim_rendering,
        render_ok=render_ok,
    )
    collider_visualization_may_activate = async_render and _physics_collider_visualization_may_activate(sim)
    if collider_visualization_may_activate and not st.collider_sync_warned:
        manaprint(
            "WARNING: PhysX collider visualization is enabled; forcing synchronous Kit render and physics stepping"
        )
        st.collider_sync_warned = True
    marker_callbacks_may_read_physx = async_render and _marker_callbacks_may_read_physx(sim)
    if marker_callbacks_may_read_physx and not st.marker_sync_warned:
        manaprint(
            "WARNING: debug visualization markers read physics buffers during Kit render; "
            "forcing synchronous Kit render and physics stepping"
        )
        st.marker_sync_warned = True
    main_thread_recorder_terms = _assert_worker_decimation_recorder_terms_are_declared(env)
    if main_thread_recorder_terms and not st.recorder_sync_warned:
        manaprint(
            "WARNING: decimation recorder terms declared runs_on_physics_worker=False; "
            f"running the decimation chain on the main thread for {sorted(main_thread_recorder_terms)}"
        )
        st.recorder_sync_warned = True
    run_substeps_on_main = bool(main_thread_recorder_terms) or st.worker is None or substep_render_cadence
    serialize_kit_render = bool(
        collider_visualization_may_activate or marker_callbacks_may_read_physx or main_thread_recorder_terms
    )
    overlap_render_submit = bool(async_render and st.worker is not None and not serialize_kit_render)
    if not async_render and st.pending_render is not None:
        if sim_rendering:
            _finish_pending_render(st, st.pending_render.frame_token)
        else:
            _cancel_pending_render(st)

    physics_dt = env.physics_dt
    timings = _SubstepTimings()
    last_published_frame: int | None = None

    def _run_substeps() -> None:
        """Advance the full decimation loop — the worker owns this whole chain.

        Everything here is physics-layer: actuator models, PhysX target writes,
        the configured ``PhysicsManager.step()``, and the asset/sensor timestamp
        bookkeeping that gates their staleness. It has to stay together because
        each substep's actuator output reads the joint state the previous
        ``scene.update`` invalidated. Managers, obs/camera reads and Fabric
        publication stay on the caller.
        """
        nonlocal last_published_frame
        for _ in range(decimation):
            env._sim_step_counter += 1
            t_scene = time.perf_counter()
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            timings.scene += time.perf_counter() - t_scene

            t_phys = time.perf_counter()
            _step_physics_backend(sim)
            timings.physics += time.perf_counter() - t_phys

            env.recorder_manager.record_post_physics_decimation_step()

            if substep_render_cadence and env._sim_step_counter % render_interval == 0:
                last_published_frame = _publish_current_fabric_frame(st, env, sim)
                _render_current_frame(st, env, sim, render_ok=render_ok)

            t_scene = time.perf_counter()
            _assert_worker_sensor_updates_are_lazy(env.scene)
            env.scene.update(dt=physics_dt)
            timings.scene += time.perf_counter() - t_scene

    t0 = time.perf_counter()

    def _start_worker() -> None:
        if not st.worker_alive:
            if st.worker_err is not None:
                raise st.worker_err
            raise RuntimeError("PhysX step worker is not running")
        st.work_fn = _run_substeps
        st.work_done.clear()
        st.work_ready.set()

    def _wait_for_worker() -> None:
        with tracy_zone("manaenv/wait_worker", source="physx_async_env.py"):
            t_join = time.perf_counter()
            if not st.work_done.wait(timeout=_WORKER_TIMEOUT_S):
                raise RuntimeError("PhysX step worker timed out during the substep chain")
            st.join_ms.append((time.perf_counter() - t_join) * 1e3)
            if st.worker_err is not None:
                raise st.worker_err

    def _advance_physics() -> None:
        if st.worker is not None and not run_substeps_on_main:
            _start_worker()
            _wait_for_worker()
        else:
            _run_substeps()

    pre_step_frame = int(sim.get_physics_step_count())
    if async_render:
        if st.pending_render is not None and not _pending_render_is_armed(st.pending_render):
            raise RuntimeError(
                "A submitted Kit render ticket was canceled before completion; refusing to republish Fabric"
            )
        # The first step after install/reset may not have a frame in flight yet.
        # Publish S_n and arm its camera tickets before any worker can advance
        # it. The worker schedule defers only that initial submission until just
        # after the worker starts; serialized fallbacks submit immediately.
        if st.pending_render is None:
            t_pub = time.perf_counter()
            env.publish_fabric()
            st.publish_ms.append((time.perf_counter() - t_pub) * 1e3)
            if overlap_render_submit:
                _arm_pending_render(st, env, pre_step_frame)
            else:
                _begin_pending_render(st, env, pre_step_frame)
        _require_pending_render(st, pre_step_frame)
        _record_pre_step_terms(env, camera_terms=False)
    else:
        env.recorder_manager.record_pre_step()

    # The initial S_n submission and its native Hydra/RTX tail run while the
    # worker advances the live PhysX scene to S_n+1. The caller joins before any
    # completion pump. Camera recorder terms run only after both sides finish,
    # so RecorderManager never mutates episode buffers concurrently.
    if async_render:
        if serialize_kit_render or st.worker is None:
            pending = _require_pending_render(st, pre_step_frame)
            if not pending.submitted:
                _submit_pending_render(st, pre_step_frame)
            _finish_pending_render(st, pre_step_frame)
            _refresh_camera_observations(env, update_history=True)
            _record_pre_step_terms(env, camera_terms=True)
            _advance_physics()
        else:
            try:
                _start_worker()
                st.worker_render_submit_allowed = True
                try:
                    _submit_pending_render(st, pre_step_frame, cancel_on_error=False)
                finally:
                    st.worker_render_submit_allowed = False
                    # Even a partially submitted Kit frame cannot unwind into
                    # reset/teardown while the worker owns PxScene.
                    _wait_for_worker()
            except BaseException:
                _cancel_pending_render(st)
                raise
            _finish_pending_render(st, pre_step_frame)
            _refresh_camera_observations(env, update_history=True)
            _record_pre_step_terms(env, camera_terms=True)
            st.overlapped += 1
    else:
        _advance_physics()

    _finish_step_render_cadence(
        st,
        env,
        sim,
        sim_rendering=sim_rendering,
        render_ok=render_ok,
        async_render=async_render,
        substep_render_cadence=substep_render_cadence,
        last_published_frame=last_published_frame,
    )

    t_tail = time.perf_counter()
    st.step_count += 1
    st.step_ms.append((t_tail - t0) * 1e3)
    st.physics_ms.append(timings.physics * 1e3)
    st.scene_ms.append(timings.scene * 1e3)

    _run_post_step_managers(st, env, sim, async_render=async_render, sim_rendering=sim_rendering)

    st.tail_ms.append((time.perf_counter() - t_tail) * 1e3)
    return env.obs_buf, env.reward_buf, env.reset_terminated, env.reset_time_outs, env.extras


def _prepare_visualizers(st: _State, env: ManaPhysxAsyncKitAsyncEnv) -> None:
    """Mirror the pause/close gates normally owned by ``SimulationContext.step``."""
    sim = env.sim
    for viz in _simulation_visualizers(sim):
        try:
            while viz.is_training_paused() and viz.is_running() and not viz.is_closed:
                sim.update_scene_data_provider()
                viz.step(0.0)
        except Exception as exc:  # noqa: BLE001 - a visualizer dying mid-pause is a close, not a bug
            manaprint(f"WARNING: PhysX step executor could not pump a paused visualizer: {exc}")
    for viz in _simulation_visualizers(sim):
        try:
            closed = viz.is_closed or not viz.is_running()
        except Exception:  # noqa: BLE001, S112
            continue
        if closed:
            _cancel_pending_render(st)
            raise ReferenceError("Visualizer closed while stepping the environment")


def _install(st: _State, env: ManaPhysxAsyncKitAsyncEnv) -> None:
    """Bind the CPU PhysX/Kit overlapping executor to one live environment."""
    sim = env.sim
    if torch.device(sim.device).type != "cpu":
        raise ManaPhysxContractError("kit_render_mode='async_handshake' currently supports only CPU PhysX")
    _assert_cpu_worker_chain(env, sim)
    if st.ready and st.sim is sim:
        return
    if st.worker is not None or st.render is not None:
        _release_epoch_state(st)
    if st.worker is not None:
        raise RuntimeError("Previous PhysX step worker is still alive")
    st.mode = PHYSICS_STEP_MODE_ASYNC
    st.kit_render_mode = KIT_RENDER_MODE_ASYNC_HANDSHAKE
    st.worker_alive = True
    st.collider_sync_warned = False
    st.marker_sync_warned = False
    st.recorder_sync_warned = False
    st.render_cadence_warned = False
    st.fabric_guard_depth = 0
    pm = getattr(sim, "physics_manager", None)
    if getattr(pm, "__name__", "") != "PhysxManager":
        raise ManaPhysxContractError("physics_step_mode='async' requires the PhysX backend")
    if getattr(env, "_physics_handles_decimation", False):
        raise ManaPhysxContractError(
            "physics_step_mode='async' cannot wrap a backend that handles decimation internally"
        )
    standalone = [type(v).__name__ for v in _simulation_visualizers(sim) if not v.pumps_app_update()]
    if standalone:
        manaprint(f"WARNING: PhysX step executor excludes standalone visualizers from the Kit pump: {standalone}")
    physics_cfg = getattr(getattr(sim, "cfg", None), "physics", None)
    if physics_cfg is not None and hasattr(physics_cfg, "fabric_skip"):
        st.fabric_skip_cfg = physics_cfg
        st.fabric_skip_saved = True
        st.fabric_skip_value = physics_cfg.fabric_skip
        # Async mode publishes exactly once after joining. Letting the upstream
        # PhysX step toggle writeback on its render-interval boundary would add
        # an implicit publication between the explicit step boundaries.
        physics_cfg.fabric_skip = False
    # CPU fetches can otherwise publish an implicit intermediate state. Keeping
    # per-substep writeback off leaves every Fabric write to the caller's
    # explicit publish point.
    fabric_setting_value = sim.get_setting(_FABRIC_TRANSFORM_SETTING)
    if not isinstance(fabric_setting_value, bool):
        _restore_fabric_skip_config(st)
        raise ManaPhysxContractError(
            f"{_FABRIC_TRANSFORM_SETTING} must resolve to bool; got {type(fabric_setting_value).__name__}"
        )
    try:
        st.fabric_setting_value = fabric_setting_value
        st.fabric_setting_sim = sim
        st.fabric_setting_saved = True
        sim.set_setting(_FABRIC_TRANSFORM_SETTING, False)
    except Exception as exc:
        _restore_fabric_setting(st)
        _restore_fabric_skip_config(st)
        raise RuntimeError("Could not isolate Fabric for async mode") from exc
    st.render = _make_render(sim, st)
    st.worker = threading.Thread(target=_worker_loop, args=(st, sim.device), name="mana-physics", daemon=True)
    st.worker.start()
    st.sim = sim
    st.ready = True
    manaprint(
        f"INFO: PhysX step executor installed "
        f"(env_class={type(env).__name__}, mode={st.mode}, "
        f"kit_render_mode={st.kit_render_mode}, device={sim.device})"
    )


class ManaPhysxAsyncKitAsyncEnv(IsaacLabArenaManagerBasedRLEnv):
    """Overlap CPU PhysX with Kit's native asynchronous render handshake."""

    def __init__(
        self,
        cfg: IsaacLabArenaManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        variation_recorder: VariationRecorder | None = None,
        **kwargs: object,
    ) -> None:
        # Keep ManagerBasedEnv.__del__ safe if class-owned validation rejects
        # the schedule before the parent constructor initializes this field.
        self._is_closed = True
        device = torch.device(cfg.sim.device)
        if device.type != "cpu":
            raise ValueError(
                f"{type(self).__name__} currently supports only CPU PhysX; "
                f"GPU PhysX/Kit concurrency is disabled (got {device})"
            )
        # Gym passes registration metadata such as ``env_cfg_entry_point`` to
        # the entry point; retain the parent Arena environment's kwargs contract.
        if variation_recorder is None:
            super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        else:
            super().__init__(cfg=cfg, render_mode=render_mode, variation_recorder=variation_recorder, **kwargs)

        self.physics_step_mode = PHYSICS_STEP_MODE_ASYNC
        self.kit_render_mode = KIT_RENDER_MODE_ASYNC_HANDSHAKE
        self._mana_physx_step_state: _State | None = _State()
        self._mana_physx_step_closed = False
        self._mana_physx_bench_frames = int(os.environ.get("MANAENV_STEP_PHASE_PROBE", "0") or "0")
        try:
            _install(self._mana_physx_step_state, self)
        except Exception:
            _release_epoch_state(self._mana_physx_step_state)
            self._mana_physx_step_state = None
            super().close()
            raise

    @property
    def completed_physx_steps(self) -> int:
        state = self._mana_physx_step_state
        return state.step_count if state is not None else 0

    @property
    def physics_device(self) -> torch.device:
        """Return the CPU device supplied through ``cfg.sim.device``."""
        return torch.device(self.sim.device)

    @contextmanager
    def fabric_publication_guard(self) -> Iterator[None]:
        """Temporarily open async Fabric writeback while physics is quiescent.

        Async mode keeps automatic writeback disabled for the whole scene
        epoch. Replay and snapshot restore paths must enter this boundary
        before forwarding restored PhysX state to Fabric.
        """
        state = getattr(self, "_mana_physx_step_state", None)
        if (
            getattr(self, "_mana_physx_step_closed", False)
            or state is None
            or not state.ready
            or state.sim is not self.sim
        ):
            raise RuntimeError("Async Fabric publication requires an active ManaPhysxAsyncKitAsyncEnv scene epoch")
        if _worker_owns_scene(state):
            raise RuntimeError("Cannot publish Fabric while the async PhysX worker owns the scene")

        # This setting gates explicit publication as well as fetch-time
        # writeback. Keep it open only for the outermost caller's quiescent
        # boundary: reset entrypoints wrap the whole parent reset, and event
        # terms publish inside it.
        outermost = state.fabric_guard_depth == 0
        state.fabric_guard_depth += 1
        try:
            if outermost:
                self.sim.set_setting(_FABRIC_TRANSFORM_SETTING, True)
            yield
        finally:
            state.fabric_guard_depth -= 1
            if outermost:
                self.sim.set_setting(_FABRIC_TRANSFORM_SETTING, False)

    def publish_fabric(self) -> None:
        """Publish one completed PhysX state through the async env boundary."""
        state = self._mana_physx_step_state
        if state is not None and _worker_owns_scene(state):
            raise RuntimeError("Cannot publish Fabric while the async PhysX worker owns the scene")
        if state is not None and state.pending_render is not None:
            _finish_pending_render(state, state.pending_render.frame_token)
        with self.fabric_publication_guard():
            # Device-agnostic: ``PhysxManager.forward`` runs
            # ``update_articulations_kinematic`` then Fabric ``force_update``.
            self.sim.physics_manager.forward()

    def render_camera_frame(self) -> None:
        """Render and finish one camera frame while the async scene is quiescent."""
        state = self._mana_physx_step_state
        if self._mana_physx_step_closed or state is None or not state.ready or state.sim is not self.sim:
            raise RuntimeError("Async camera rendering requires an active scene epoch")
        if _worker_owns_scene(state):
            raise RuntimeError("Cannot render a camera frame while the async PhysX worker owns the scene")
        if state.pending_render is not None:
            _finish_pending_render(state, state.pending_render.frame_token)
        _render_with_camera_handoffs(state, self)

    def _run_parent_reset_with_fabric_writeback(
        self,
        reset_fn: Callable[_ResetParams, _ResetResult],
        *args: _ResetParams.args,
        **kwargs: _ResetParams.kwargs,
    ) -> _ResetResult:
        """Let the parent reset publish its pose writes and reset rerenders."""
        state = getattr(self, "_mana_physx_step_state", None)
        if state is None or not state.ready or state.sim is not self.sim:
            return reset_fn(*args, **kwargs)

        if _worker_owns_scene(state):
            raise RuntimeError("Cannot reset while the async PhysX worker owns the scene")
        if state.pending_render is not None:
            _finish_pending_render(state, state.pending_render.frame_token)
        # Async stepping normally keeps worker-side writeback disabled so each
        # publication is explicit. Public reset entrypoints run on the main
        # thread with physics quiescent, so the inherited forward/render sequence
        # is the correct publication boundary. While Kit's native async mode is
        # active, replace only the inherited bare render calls with the same
        # render-product completion handshake used by step().
        sim_instance_dict = getattr(self.sim, "__dict__", {})
        had_instance_render = "render" in sim_instance_dict
        original_instance_render = sim_instance_dict.get("render")
        original_render = getattr(self.sim, "render", None)

        def _reset_render(mode: int | None = None, skip_app_pumping: bool = False) -> None:
            if mode is not None or skip_app_pumping:
                if not callable(original_render):
                    raise RuntimeError("SimulationContext.render is unavailable during reset")
                original_render(mode=mode, skip_app_pumping=skip_app_pumping)
                return
            _render_with_camera_handoffs(state, self)

        patch_reset_render = state.kit_render_mode == KIT_RENDER_MODE_ASYNC_HANDSHAKE
        if patch_reset_render:
            self.sim.render = _reset_render
        try:
            with self.fabric_publication_guard():
                result = reset_fn(*args, **kwargs)
        finally:
            if patch_reset_render:
                if had_instance_render:
                    self.sim.render = original_instance_render
                else:
                    del self.sim.render
        return result

    def reset(
        self,
        seed: int | None = None,
        env_ids: Sequence[int] | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[VecEnvObs, dict[str, object]]:
        """Reset environments while allowing the parent forward to update Fabric."""
        return self._run_parent_reset_with_fabric_writeback(
            super().reset,
            seed=seed,
            env_ids=env_ids,
            options=options,
        )

    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ) -> tuple[VecEnvObs, dict[str, object]]:
        """Restore explicit scene state while allowing its forward to update Fabric."""
        return self._run_parent_reset_with_fabric_writeback(
            super().reset_to,
            state,
            env_ids,
            seed=seed,
            is_relative=is_relative,
        )

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one full RL step while preserving the public env contract."""
        state = self._mana_physx_step_state
        if self._mana_physx_step_closed or state is None:
            raise RuntimeError(f"{type(self).__name__} is closed")

        now = time.perf_counter()
        if state.last_frame_t is not None:
            state.frame_ms.append((now - state.last_frame_t) * 1e3)
        state.last_frame_t = now
        _prepare_visualizers(state, self)
        try:
            return _raw_step(state, self, action)
        except ReferenceError:
            _cancel_pending_render(state)
            raise

    def report_physx_steps(self) -> None:
        """Print the timing report if this run asked for one (idempotent).

        Both the SESSION_END probe hook and ``close()`` funnel through here so
        the opt-in decision is made once, at report time — a caller that
        registered the hook before the environment variable was set must not
        bypass it.
        """
        state = self._mana_physx_step_state
        if state is not None and _report_enabled(getattr(self, "_mana_physx_bench_frames", 0)):
            _report(state)

    def close(self) -> None:
        """Drain native rendering, then release state before Arena clears it."""
        state = getattr(self, "_mana_physx_step_state", None)
        if state is None:
            super().close()
            return
        self.report_physx_steps()
        # Keep teardown defensive if an exception left a render submitted.
        # Camera-ticket cancellation only drops the observer; it does not
        # join native Hydra work. Drain while scene/Fabric are still valid.
        # A failed drain must not leave the environment half-closed.
        try:
            _release_epoch_state(state, drain_pending_render=True)
        finally:
            self._mana_physx_step_state = None
            self._mana_physx_step_closed = True
            super().close()


class _BaselineProbe:
    """Step-interval samples for a sync-mode run, which owns no ``_State``."""

    def __init__(self) -> None:
        self.frame_ms: deque[float] = deque(maxlen=_SAMPLE_WINDOW)
        self.count = 0
        self.last: float | None = None
        self.reported = False


def register_physics_step_probe(runner: GymEnvRunner) -> None:
    """Attach optional timing/stop hooks without replacing ``env.step()``."""
    bench_frames = int(os.environ.get("MANAENV_STEP_PHASE_PROBE", "0") or "0")
    if not bench_frames and not _report_enabled(bench_frames):
        return
    from manaenv.pipeline.runtime.env_runner import AFTER_STEP, SESSION_END

    baseline = _BaselineProbe()

    def _baseline_report() -> None:
        if baseline.reported or not baseline.frame_ms:
            return
        baseline.reported = True
        steady = min(_WARMUP_FRAMES, len(baseline.frame_ms) // 5)
        values = sorted(_steady_slice(baseline.frame_ms, steady))
        median = _percentile(values, 0.5)
        manaprint("INFO: ===== baseline report (physics_step_mode=sync) =====")
        manaprint(f"INFO:   frames={baseline.count} (last {len(baseline.frame_ms)} retained, steady from #{steady})")
        manaprint(
            f"INFO:   full-frame interval ms (A/B metric): median={median:.2f} "
            f"p95={_percentile(values, 0.95):.2f} ({_rate_hz(median):.2f} it/s)"
        )
        manaprint("INFO: ===========================================================")

    def _after_step(
        ctx: EnvRunContext,
        obs: VecEnvObs,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        del obs, terminated, truncated, actions
        physx_state = getattr(ctx.gym_env, "_mana_physx_step_state", None)
        if physx_state is None:
            now = time.perf_counter()
            if baseline.last is not None:
                baseline.frame_ms.append((now - baseline.last) * 1e3)
                baseline.count += 1
            baseline.last = now
            completed = baseline.count
        else:
            completed = physx_state.step_count
        if bench_frames and completed >= bench_frames:
            if physx_state is None:
                _baseline_report()
            else:
                ctx.gym_env.report_physx_steps()
            ctx.stop_requested = True

    def _session_end(ctx: EnvRunContext) -> None:
        physx_state = getattr(ctx.gym_env, "_mana_physx_step_state", None)
        if physx_state is None:
            _baseline_report()
        else:
            ctx.gym_env.report_physx_steps()

    runner.add_hook(AFTER_STEP, _after_step, priority=4)
    runner.add_hook(SESSION_END, _session_end, priority=4)
