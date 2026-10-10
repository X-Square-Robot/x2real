"""Task template discovery for the global lazy registry.

The constraint workflow dispatches scene-bound task templates by their
CamelCase ``*Task`` symbol (``PickAndPlaceTask``, ``OpenDoorTask``, ...).
This module wires those symbols up to the convention-driven lazy
registry so :class:`ConstraintTaskGenerator` can resolve them on demand
without hand-maintained dictionaries.

Resolution order mirrors :mod:`manaenv.assets.affordances.discovery`:

* ``manaenv.tasks.<snake_case_stem>`` — preferred home for mana-owned
  templates; created on demand when the package starts shipping its own
  task templates.
* ``isaaclab_arena.tasks.<snake_case_stem>`` — fallback for the existing
  Arena task templates such as ``PickAndPlaceTask`` / ``OpenDoorTask``.

The ``CFG`` peer is resolved through the registry's standard
``<symbol>CFG`` derivation (``PickAndPlaceTask`` -> ``PickAndPlaceTaskCFG``).
"""

from __future__ import annotations

from pathlib import Path

from manaenv.pipeline.registry.convention_provider import camel_to_snake
from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = [
    "ARENA_TASK_PACKAGE",
    "MANA_TASK_PACKAGE",
    "TASK_TEMPLATE_DOMAIN",
    "camel_to_snake",
    "discover_task_template",
    "task_template_module_path",
]

TASK_TEMPLATE_DOMAIN: str = "task_template"
MANA_TASK_PACKAGE: str = "manaenv.tasks"
ARENA_TASK_PACKAGE: str = "isaaclab_arena.tasks"

_MANA_TASK_DIR = Path(__file__).resolve().parents[3] / "tasks"
try:
    import isaaclab_arena.tasks as arena_tasks
except ModuleNotFoundError:
    _REPO_ROOT = Path(__file__).resolve().parents[4]
    _LOCAL_ARENA_TASK_DIR = _REPO_ROOT / "3rd" / "isaaclabarena" / "isaaclab_arena" / "tasks"
    _ARENA_TASK_DIR: Path | None = _LOCAL_ARENA_TASK_DIR if _LOCAL_ARENA_TASK_DIR.exists() else None
else:
    _ARENA_TASK_DIR = Path(arena_tasks.__file__).resolve().parent


def task_template_module_path(symbol: str) -> str | None:
    """Return the module path that hosts a ``*Task`` template, or ``None``."""
    if not symbol.endswith("Task") or symbol == "Task":
        return None
    stem = camel_to_snake(symbol)
    candidates: list[tuple[str, Path]] = [(MANA_TASK_PACKAGE, _MANA_TASK_DIR)]
    if _ARENA_TASK_DIR is not None:
        candidates.append((ARENA_TASK_PACKAGE, _ARENA_TASK_DIR))
    for package, directory in candidates:
        if (directory / f"{stem}.py").exists():
            return f"{package}.{stem}"
    return None


def discover_task_template(symbol: str) -> ComponentSpec | None:
    module_path = task_template_module_path(symbol)
    if module_path is None:
        return None
    return ComponentSpec(module_path=module_path, class_name=symbol, symbol=symbol)


register_provider(TASK_TEMPLATE_DOMAIN, discover_task_template)
