"""Resolve the three runtime owners — Scene, Embodiment, Task — from a gym env.

A checker / termination func receives only ``env`` (IsaacLab calls it as
``func(env, ...)``). Everything it needs lives on one of the three owners
reachable from that single handle:

* ``scene``      — the ``isaaclab_arena`` Scene (``.assets``)
* ``embodiment`` — the ``EmbodimentBase`` (e.g. ``FrankaEmbodiment``)
* ``task``       — the live ``ManaTask`` (stays alive at runtime)

They hang off the ``AssembledEnvironment`` back-reference the assembly factory
installs on the env (``env._mana_assembled_environment``), with the env-cfg route
(``env.cfg.isaaclab_arena_env``) as fallback. This module is the single place
that resolution lives — previously duplicated (slightly differently) in
``planners/motion_planner/runtime/robot_state.py``,
``scene_layout_solver/manaenv_bridge/task_runtime_consumer.py``.

No new wrapper type: callers read the owners' existing names off ``env``.
"""

from __future__ import annotations

from typing import Any


def assembled_environment(env: Any) -> Any:
    """Return the ``AssembledEnvironment`` for *env*, or ``None`` if unavailable.

    Prefers the direct back-reference installed at ``gym.make`` time; falls back
    to the env-cfg route used by the motion-planner runtime.
    """
    assembled = getattr(env, "_mana_assembled_environment", None)
    if assembled is not None:
        return assembled
    cfg = getattr(env, "cfg", None)
    return getattr(cfg, "isaaclab_arena_env", None)


def mana_scene(env: Any) -> Any:
    """The ``isaaclab_arena`` Scene (``.assets``) for *env*, or ``None``."""
    return getattr(assembled_environment(env), "scene", None)


def mana_embodiment(env: Any) -> Any:
    """The ``EmbodimentBase`` for *env*, or ``None``."""
    return getattr(assembled_environment(env), "embodiment", None)


def mana_task(env: Any) -> Any:
    """The live ``ManaTask`` for *env*, or ``None``."""
    return getattr(assembled_environment(env), "task", None)
