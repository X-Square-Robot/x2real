"""Gym-style environment step loop with per-hook callables.

Register small functions (or lambdas) for each intervention point via
:meth:`GymEnvRunner.add_hook` — no monolithic callback base class.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from manaenv.utils.debug_print import manaprint
from manaenv.utils.priority import sort_by_priority

# Stable hook names for registration and documentation.
SESSION_START = "session_start"
GYM_READY = "gym_ready"
INITIAL_OBS = "initial_obs"
LOOP_START = "loop_start"
BEFORE_STEP = "before_step"
AFTER_ACTION = "after_action"
PERFORM_STEP = "perform_step"
AFTER_STEP = "after_step"
ENV_DONES = "env_dones"
EPISODE_REBUILD = "episode_rebuild"
SHOULD_STOP = "should_stop"
SESSION_END = "session_end"

_HookFn = Callable[..., Any]
_HookEntry = tuple[int, _HookFn]

# ---------------------------------------------------------------------------
# Global runner hooks — the single interface for cross-cutting instrumentation
# (e.g. monitoring) to attach to EVERY rollout without editing individual
# factories. Hooks registered here are applied to each runner at the top of
# ``run_loop`` (the one path all rollouts pass through).
# ---------------------------------------------------------------------------
_GLOBAL_RUNNER_HOOKS: list[tuple[str, _HookFn, int]] = []


def register_global_runner_hook(hook: str, fn: _HookFn, priority: int = 100) -> None:
    """Register a hook applied to every :class:`GymEnvRunner` rollout.

    Applied at the top of :meth:`GymEnvRunner.run_loop` (once per runner), so an
    external owner (e.g. the DataCenter monitoring manager) can observe all
    rollouts from one place without touching individual factories. Idempotent
    per ``(hook, fn)``: registering the same pair twice is a no-op.
    """
    for h, f, _ in _GLOBAL_RUNNER_HOOKS:
        if h == hook and f is fn:
            return
    _GLOBAL_RUNNER_HOOKS.append((hook, fn, priority))


def clear_global_runner_hooks() -> None:
    """Remove all global runner hooks (e.g. when monitoring is torn down)."""
    _GLOBAL_RUNNER_HOOKS.clear()


def unregister_global_runner_hook(hook: str, fn: _HookFn) -> None:
    """Remove one global runner hook by ``(hook, fn)`` identity; no-op if absent.

    The targeted counterpart to :func:`clear_global_runner_hooks`. Lets an owner
    (e.g. the editor service's per-rollout stop/progress hooks, or the monitoring
    manager) detach only its own hooks without clobbering hooks another owner
    registered.
    """
    _GLOBAL_RUNNER_HOOKS[:] = [(h, f, p) for (h, f, p) in _GLOBAL_RUNNER_HOOKS if not (h == hook and f is fn)]


def _coerce_reset_obs(value: Any) -> Any:
    """Accept Gymnasium-style ``(obs, info)`` values in override paths."""

    if isinstance(value, tuple) and len(value) == 2:
        return value[0]
    return value


def _simulation_app_is_running() -> bool:
    """Return False after a Kit viewer close; True when Kit is unavailable."""
    try:
        import omni.kit.app

        return bool(omni.kit.app.get_app().is_running())
    except Exception:
        return True


def _sim_context_instance():
    """Return the active SimulationContext, or None off Isaac paths."""
    try:
        from isaaclab.sim.simulation_context import SimulationContext

        return SimulationContext.instance()
    except Exception:
        return None


def _viewer_live(viz) -> bool:
    """True while a standalone visualizer window is open and running."""
    try:
        return bool(viz.is_running() and not viz.is_closed)
    except Exception:
        return False


def _standalone_viewers_closed(sim, tracked: dict[int, Any]) -> bool:
    """Return True once all seen-live non-Kit viewers have closed."""
    if sim is None:
        return False
    try:
        visualizers = list(sim.visualizers)
    except Exception:
        visualizers = []
    for viz in visualizers:
        try:
            if not viz.pumps_app_update() and viz.is_running() and not viz.is_closed:
                tracked[id(viz)] = viz
        except Exception:
            pass
    if not tracked:
        return False
    for viz in tracked.values():
        try:
            if not viz.is_closed and viz.is_running():
                return False
        except Exception:
            return False
    return True


def _visualizer_type(viz) -> str:
    """Return the configured visualizer backend id."""
    return getattr(getattr(viz, "cfg", None), "visualizer_type", None) or ""


def _is_interactive_pause_viewer(viz) -> bool:
    """Return whether an on-screen Kit viewer exposes the Play control."""
    return _visualizer_type(viz) == "kit" and not getattr(viz, "_runtime_headless", False)


def _timeline_is_playing() -> bool:
    """Return Kit timeline play state; True if it cannot be read."""
    try:
        import omni.timeline

        return bool(omni.timeline.get_timeline_interface().is_playing())
    except Exception:
        return True


def _pause_kit_timeline() -> bool:
    """Pause Kit timeline so the Play button becomes the start signal."""
    try:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            timeline.pause()
        return True
    except Exception:
        return False


def _resolve_start_paused(gym_env: Any, assembled_environment: Any) -> bool:
    """Read runtime.start_paused from the assembled environment or gym back-ref."""
    environment = assembled_environment or getattr(gym_env, "_mana_assembled_environment", None)
    runtime_cfg = getattr(environment, "runtime_cfg", None)
    return bool(getattr(runtime_cfg, "start_paused", False))


def _apply_start_pause_gate(sim, gym_env: Any, *, start_paused: bool) -> None:
    """Render the scene but hold first physics step until Play/Resume when requested."""
    if sim is None:
        return
    try:
        visualizers = list(sim.visualizers)
    except Exception:
        visualizers = []
    interactive = [viz for viz in visualizers if _is_interactive_pause_viewer(viz)]
    if not interactive:
        return

    if not start_paused:
        return

    _pause_kit_timeline()

    manaprint("INFO: start_paused — viewer rendering; physics frozen until Play (Kit toolbar).")

    def viewer_resumed(_viz) -> bool:
        return _timeline_is_playing()

    while True:
        live = [viz for viz in interactive if _viewer_live(viz)]
        if not live or all(viewer_resumed(viz) for viz in live):
            return
        for viz in live:
            try:
                sim.update_scene_data_provider()
                viz.step(0.0)
            except Exception:
                pass


_ASSET_DISPLAY_HOOK_REGISTERED = False


def _tick_asset_displays(ctx: EnvRunContext, obs: Any, terminated: Any, truncated: Any, actions: Any) -> None:
    """AFTER_STEP: let any scene asset refresh its live display (duck-typed).

    The generic runner stays asset-agnostic: it just calls ``sync_display`` on
    whichever scene assets expose it (e.g. ``KnobA``). The concrete USD/shader
    logic lives on the asset in ``manaenv.assets.objects``.
    """
    from manaenv.runtime_owners import mana_scene

    gym_env = getattr(ctx, "gym_env", None)
    if gym_env is None:
        return
    scene = mana_scene(gym_env)
    assets = getattr(scene, "assets", None) if scene is not None else None
    if not isinstance(assets, dict):
        return
    for asset in assets.values():
        sync = getattr(asset, "sync_display", None)
        if callable(sync):
            sync(gym_env)


def ensure_asset_display_tick() -> None:
    """Register the global AFTER_STEP hook that refreshes asset displays."""
    global _ASSET_DISPLAY_HOOK_REGISTERED
    if _ASSET_DISPLAY_HOOK_REGISTERED:
        return
    register_global_runner_hook(AFTER_STEP, _tick_asset_displays, priority=80)
    _ASSET_DISPLAY_HOOK_REGISTERED = True


ensure_asset_display_tick()


@dataclass
class EnvRunContext:
    """Mutable state shared across one ``run_loop`` invocation."""

    gym_env: Any
    step_idx: int = 0
    stop_requested: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _consume_skip_step(
    ctx: EnvRunContext,
    *,
    consecutive_skips: int,
    max_consecutive_skips: int,
    phase: str,
) -> tuple[bool, int]:
    if not ctx.extra.pop("skip_step", False):
        return False, consecutive_skips

    consecutive_skips += 1
    if ctx.extra.get("allow_indefinite_skip_step", False):
        return True, 0
    if consecutive_skips >= max_consecutive_skips:
        warnings.warn(
            f"Stopping env loop after {consecutive_skips} consecutive skip_step requests at {phase}.",
            RuntimeWarning,
            stacklevel=2,
        )
        ctx.stop_requested = True
    return True, consecutive_skips


class GymEnvRunner:
    """Runs ``actor.get_actions`` → ``gym_env.step`` until stop or ``max_steps``."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[_HookEntry]] = defaultdict(list)

    def add_hook(self, hook: str, fn: _HookFn, priority: int = 100) -> GymEnvRunner:
        """Attach one callable to a hook. Lower *priority* runs earlier.

        For :data:`SESSION_END`, callables run in reverse order (teardown).

        Expected signatures (unused parameters may be omitted)::

            session_start(ctx)
            gym_ready(ctx)
            initial_obs(ctx) -> Any | None   # first non-None wins; else reset()
            loop_start(ctx, obs)
            before_step(ctx, obs)
            after_action(ctx, obs, actions)
            perform_step(ctx, obs, actions) -> (obs, reward, terminated, truncated, info) | None
            after_step(ctx, obs, terminated, truncated, actions)
            env_dones(ctx, done, obs, terminated, truncated, actions)
            episode_rebuild(ctx)              # optional full Gym/scene swap
            should_stop(ctx) -> bool         # any true stops
            session_end(ctx)

        Hooks may set ``ctx.extra["skip_step"] = True`` to keep the current
        observation and retry the loop without incrementing ``step_idx``.
        ``ctx.extra["max_consecutive_skips"]`` can override the default guard
        of 1000 consecutive skipped iterations.
        Hooks that set ``ctx.extra["skip_env_dones"] = True`` must own done-env
        recovery, including reset or stop behavior. Once recovery is complete,
        set ``ctx.extra["skip_env_dones_recovered"] = True`` to acknowledge it.
        """
        self._hooks[hook].append((priority, fn))
        return self

    def clear_hooks(self) -> None:
        self._hooks.clear()

    def _apply_global_hooks(self) -> None:
        """Attach registered global hooks once per runner.

        See :func:`register_global_runner_hook`. Applied at the top of
        :meth:`run_loop` so it re-applies on every rollout (surviving any prior
        ``clear_hooks``), while the per-runner guard avoids duplicate adds when a
        single runner runs more than once.
        """
        if getattr(self, "_global_hooks_applied", False):
            return
        self._global_hooks_applied = True
        for hook, fn, priority in list(_GLOBAL_RUNNER_HOOKS):
            self.add_hook(hook, fn, priority=priority)

    def _sorted_entries(self, hook: str) -> list[_HookEntry]:
        return sort_by_priority(self._hooks.get(hook, []), key=lambda e: e[0])

    def _call_hook(self, hook: str, *args: Any) -> None:
        for _, fn in self._sorted_entries(hook):
            fn(*args)

    def _first_hook_result(self, hook: str, *args: Any) -> Any | None:
        for _, fn in self._sorted_entries(hook):
            result = fn(*args)
            if result is not None:
                return result
        return None

    @staticmethod
    def _warn_if_override_actions_skipped(actions_overridden: bool) -> None:
        if actions_overridden:
            warnings.warn(
                "skip_step requested after override_actions; the overridden action was discarded.",
                RuntimeWarning,
                stacklevel=2,
            )

    def run_with_build(
        self,
        environment: Any,
        build_gym_kwargs: dict[str, Any],
        actor: Any,
        *,
        context: EnvRunContext | None = None,
        max_steps: int | None = None,
        max_steps_resolver: Callable[[Any], int | None] | None = None,
        use_inference_mode: bool = True,
        create_stage: bool = True,
        close_gym_env: bool = True,
        assembled_environment: Any = None,
    ) -> EnvRunContext | None:
        """Create USD stage, ``build_gym_environment``, then :meth:`run_loop`.

        Returns ``None`` when ``build_gym_environment`` returns ``None``.
        """
        if create_stage:
            from isaaclab.sim.utils import stage as stage_utils

            stage_utils.create_new_stage()

        gym_env = environment.build_gym_environment(**build_gym_kwargs)
        if gym_env is None:
            return None

        effective_max = max_steps
        if effective_max is None and max_steps_resolver is not None:
            effective_max = max_steps_resolver(gym_env)

        return self.run_loop(
            gym_env,
            actor,
            context=context,
            max_steps=effective_max,
            use_inference_mode=use_inference_mode,
            close_gym_env=close_gym_env,
            assembled_environment=assembled_environment,
        )

    def run_with_factory(
        self,
        build_gym_env_fn: Callable[[], Any],
        actor: Any,
        *,
        context: EnvRunContext | None = None,
        max_steps: int | None = None,
        use_inference_mode: bool = True,
        close_gym_env: bool = True,
        assembled_environment: Any = None,
    ) -> EnvRunContext | None:
        """Create a gym env through a caller-supplied factory, then run the loop."""
        gym_env = build_gym_env_fn()
        if gym_env is None:
            return None
        return self.run_loop(
            gym_env,
            actor,
            context=context,
            max_steps=max_steps,
            use_inference_mode=use_inference_mode,
            close_gym_env=close_gym_env,
            assembled_environment=assembled_environment,
        )

    def run_loop(  # noqa: C901
        self,
        gym_env: Any,
        actor: Any,
        *,
        context: EnvRunContext | None = None,
        max_steps: int | None = None,
        use_inference_mode: bool = True,
        close_gym_env: bool = False,
        assembled_environment: Any = None,
    ) -> EnvRunContext:
        """Execute the step loop. ``actor`` must implement ``get_actions(gym_env, obs)``."""
        ctx = context if context is not None else EnvRunContext(gym_env=gym_env)
        ctx.gym_env = gym_env
        if assembled_environment is not None:
            ctx.extra.setdefault("assembled_environment", assembled_environment)
        try:
            self._apply_global_hooks()

            self._call_hook(SESSION_START, ctx)
            self._call_hook(GYM_READY, ctx)

            obs: Any = None
            for _, fn in self._sorted_entries(INITIAL_OBS):
                candidate = fn(ctx)
                if candidate is not None:
                    obs = candidate
                    break
            if obs is None:
                obs, _ = gym_env.reset()

            self._call_hook(LOOP_START, ctx, obs)

            import torch

            ctx_mgr = torch.inference_mode() if use_inference_mode else nullcontext()
            max_consecutive_skips = max(int(ctx.extra.get("max_consecutive_skips", 1000) or 1000), 1)
            consecutive_skips = 0
            sim_ctx = _sim_context_instance()
            standalone_viewers: dict[int, Any] = {}

            try:
                _apply_start_pause_gate(
                    sim_ctx,
                    gym_env,
                    start_paused=_resolve_start_paused(gym_env, assembled_environment),
                )
            except Exception as exc:
                warnings.warn(f"start_paused gate skipped: {exc}", RuntimeWarning, stacklevel=2)
        except Exception:
            # Kit can close while initialization is inside any backend call.
            # PhysX and render-product wrappers do not share one exception
            # type after teardown, so app liveness is the shutdown authority.
            if _simulation_app_is_running():
                self._teardown_run_loop(
                    ctx,
                    gym_env,
                    close_gym_env=close_gym_env,
                    assembled_environment=assembled_environment,
                )
                raise
            ctx.stop_requested = True
            self._teardown_run_loop(
                ctx,
                gym_env,
                close_gym_env=close_gym_env,
                assembled_environment=assembled_environment,
            )
            return ctx

        try:
            with ctx_mgr:
                while True:
                    if not _simulation_app_is_running() or _standalone_viewers_closed(sim_ctx, standalone_viewers):
                        ctx.stop_requested = True
                        break
                    if max_steps is not None and ctx.step_idx >= max_steps:
                        break

                    # A reset-time hook may only mutate state inside the
                    # current USD topology.  ``active_only`` instead requests
                    # a whole scene epoch here, after the previous episode's
                    # evaluation hooks completed and before the next
                    # action is computed.
                    if ctx.extra.get("episode_rebuild_requested") or ctx.extra.get("episode_boundary") is not None:
                        previous_gym_env = gym_env
                        self._call_hook(EPISODE_REBUILD, ctx)
                        # Non-active scenes have no rebuild hook; do not carry a
                        # stale boundary into every subsequent step.
                        ctx.extra.pop("episode_rebuild_requested", None)
                        ctx.extra.pop("episode_boundary", None)
                        if ctx.stop_requested:
                            break
                        if ctx.gym_env is not previous_gym_env:
                            gym_env = ctx.gym_env
                            sim_ctx = _sim_context_instance()
                            standalone_viewers = {}
                            # Manual/reset hooks may have queued an observation
                            # from the just-destroyed Gym instance.  It cannot
                            # be consumed against the rebuilt USD topology.
                            ctx.extra.pop("override_obs", None)
                            self._call_hook(GYM_READY, ctx)
                            rebuilt_obs = self._first_hook_result(INITIAL_OBS, ctx)
                            if rebuilt_obs is None:
                                rebuilt_obs, _ = gym_env.reset()
                            obs = rebuilt_obs

                    if "override_obs" in ctx.extra:
                        obs = ctx.extra.pop("override_obs")
                    self._call_hook(BEFORE_STEP, ctx, obs)
                    if "override_obs" in ctx.extra:
                        obs = ctx.extra.pop("override_obs")
                    if ctx.stop_requested:
                        break
                    did_skip, consecutive_skips = _consume_skip_step(
                        ctx,
                        consecutive_skips=consecutive_skips,
                        max_consecutive_skips=max_consecutive_skips,
                        phase=BEFORE_STEP,
                    )
                    if did_skip:
                        if ctx.stop_requested:
                            break
                        continue

                    actions = actor.get_actions(gym_env, obs)
                    self._call_hook(AFTER_ACTION, ctx, obs, actions)
                    actions_overridden = False
                    if "override_actions" in ctx.extra:
                        actions = ctx.extra.pop("override_actions")
                        actions_overridden = True
                    if "override_obs" in ctx.extra:
                        obs = ctx.extra.pop("override_obs")
                    if ctx.stop_requested:
                        break
                    did_skip, consecutive_skips = _consume_skip_step(
                        ctx,
                        consecutive_skips=consecutive_skips,
                        max_consecutive_skips=max_consecutive_skips,
                        phase=AFTER_ACTION,
                    )
                    if did_skip:
                        self._warn_if_override_actions_skipped(actions_overridden)
                        if ctx.stop_requested:
                            break
                        continue
                    try:
                        step_result = self._first_hook_result(PERFORM_STEP, ctx, obs, actions)
                    except ReferenceError:
                        # Custom step hooks (for example state replay) may call
                        # ``sim.render()`` and then refresh scene buffers.  A Kit
                        # window close can invalidate IsaacLab's PhysX tensor
                        # views during that render, before the hook returns.  The
                        # default ``gym_env.step`` path has the same guard below;
                        # keep custom hooks on the same window-closed=clean-exit
                        # contract instead of surfacing a dangling weakref.
                        if _simulation_app_is_running():
                            raise
                        ctx.stop_requested = True
                        break
                    if step_result is None:
                        try:
                            step_result = gym_env.step(actions)
                        except ReferenceError:
                            if _simulation_app_is_running():
                                raise
                            ctx.stop_requested = True
                            break
                    obs, _, terminated, truncated, _ = step_result
                    ctx.step_idx += 1
                    consecutive_skips = 0

                    self._call_hook(AFTER_STEP, ctx, obs, terminated, truncated, actions)

                    done = terminated | truncated
                    if done.any():
                        skip_env_dones = bool(ctx.extra.pop("skip_env_dones", False))
                        skip_env_dones_recovered = bool(ctx.extra.pop("skip_env_dones_recovered", False))
                        if skip_env_dones:
                            if not skip_env_dones_recovered and not ctx.extra.get("_skip_env_dones_warned", False):
                                warnings.warn(
                                    "ENV_DONES hook skipped while at least one environment is done; "
                                    "the hook owner must reset or stop the done environment.",
                                    RuntimeWarning,
                                    stacklevel=2,
                                )
                                ctx.extra["_skip_env_dones_warned"] = True
                        else:
                            self._call_hook(ENV_DONES, ctx, done, obs, terminated, truncated, actions)
                            ctx.extra["episode_boundary"] = done

                    for _, fn in self._sorted_entries(SHOULD_STOP):
                        try:
                            if fn(ctx):
                                ctx.stop_requested = True
                                break
                        except Exception:
                            pass
                    if ctx.stop_requested:
                        break
        except Exception:
            # Closing Kit invalidates PhysX views and render tickets at several
            # backend-specific boundaries. Treat an exception as shutdown only
            # after the app reports stopped; live-runtime failures still raise.
            if _simulation_app_is_running():
                raise
            ctx.stop_requested = True
        finally:
            self._teardown_run_loop(
                ctx,
                gym_env,
                close_gym_env=close_gym_env,
                assembled_environment=assembled_environment,
            )

        return ctx

    def _teardown_run_loop(
        self,
        ctx: EnvRunContext,
        gym_env: Any,
        *,
        close_gym_env: bool,
        assembled_environment: Any,
    ) -> None:
        for _, fn in reversed(self._sorted_entries(SESSION_END)):
            try:
                fn(ctx)
            except Exception:
                pass
        if close_gym_env and gym_env is not None:
            try:
                gym_env.close()
            except Exception:
                pass
        if assembled_environment is not None and hasattr(assembled_environment, "is_running"):
            assembled_environment.is_running = False
