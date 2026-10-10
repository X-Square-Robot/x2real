"""Scene-epoch rebuild support for evaluation-only active-slot scenes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from manaenv.pipeline.runtime.env_runner import EPISODE_REBUILD, _simulation_app_is_running
from manaenv.utils.debug_print import manaprint


def register_active_only_episode_rebuild(
    runner: Any,
    *,
    environment: Any,
    planner: Any,
    state: Any,
    build_gym_kwargs: Mapping[str, Any],
    episode_count: Callable[[Any], int],
    target_count: Callable[[Any], int | None],
) -> bool:
    """Rebuild USD topology between episodes when the scene requires it."""

    requires = getattr(environment, "requires_episode_scene_rebuild", None)
    if not callable(requires) or not bool(requires()):
        return False

    scene_epoch_index = 0

    def rebuild(ctx: Any) -> None:
        nonlocal scene_epoch_index
        requested = bool(ctx.extra.pop("episode_rebuild_requested", False))
        boundary = ctx.extra.pop("episode_boundary", None)
        if not requested and boundary is None:
            return

        target = target_count(state)
        if target is not None and int(episode_count(state)) >= int(target):
            return
        if boundary is not None:
            try:
                if not bool(boundary.all().item()):
                    manaprint("WARNING: active-only rebuild waits until every vector env reaches a boundary")
                    return
            except Exception:
                pass

        prepare = getattr(environment, "prepare_next_scene_epoch", None)
        if not callable(prepare):
            ctx.stop_requested = True
            return

        cleanup = getattr(planner, "cleanup", None)
        if callable(cleanup):
            cleanup()
        scene_epoch_index += 1
        scene = prepare(episode_index=scene_epoch_index)
        if scene is None:
            manaprint("ERROR: active-only scene materialization returned no scene")
            ctx.stop_requested = True
            return

        rebuilt = environment.rebuild_scene_epoch(scene=scene, **dict(build_gym_kwargs))
        if rebuilt is None:
            if _simulation_app_is_running():
                manaprint("ERROR: failed to rebuild active-only evaluation scene")
            ctx.stop_requested = True
            return
        if not bool(planner.setup(environment, getattr(environment, "task", None))):
            manaprint("ERROR: policy setup failed after active-only scene rebuild")
            ctx.stop_requested = True
            return
        ctx.gym_env = rebuilt
        manaprint(f"INFO: evaluation scene epoch rebuilt: {scene_epoch_index}")

    runner.add_hook(EPISODE_REBUILD, rebuild, priority=0)
    return True


__all__ = ["register_active_only_episode_rebuild"]
