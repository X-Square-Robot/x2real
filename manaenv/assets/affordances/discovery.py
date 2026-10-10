"""Affordance discovery for the global lazy registry.

Mana-owned affordances live under ``manaenv.assets.affordances`` because
they describe asset capabilities. Arena affordances remain supported as a
fallback so existing ``Openable`` / ``Pressable`` YAML keeps resolving to
``isaaclab_arena.affordances``.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab_arena.affordances as arena_affordances

from manaenv.pipeline.registry.convention_provider import camel_to_snake
from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = [
    "AFFORDANCE_DOMAIN",
    "ARENA_AFFORDANCE_PACKAGE",
    "MANA_AFFORDANCE_PACKAGE",
    "affordance_module_path",
    "camel_to_snake",
    "discover_affordance",
]

AFFORDANCE_DOMAIN: str = "affordance"
MANA_AFFORDANCE_PACKAGE: str = "manaenv.assets.affordances"
ARENA_AFFORDANCE_PACKAGE: str = "isaaclab_arena.affordances"
MANA_AFFORDANCE_DIR = Path(__file__).resolve().parent
ARENA_AFFORDANCE_DIR = Path(arena_affordances.__file__).resolve().parent


def affordance_module_path(symbol: str) -> str | None:
    stem = camel_to_snake(symbol)
    for package, directory in (
        (MANA_AFFORDANCE_PACKAGE, MANA_AFFORDANCE_DIR),
        (ARENA_AFFORDANCE_PACKAGE, ARENA_AFFORDANCE_DIR),
    ):
        module_file = directory / f"{stem}.py"
        if module_file.exists():
            return f"{package}.{stem}"
    return None


def discover_affordance(symbol: str) -> ComponentSpec | None:
    module_path = affordance_module_path(symbol)
    if module_path is None:
        return None
    return ComponentSpec(module_path=module_path, class_name=symbol, symbol=symbol)


register_provider(AFFORDANCE_DOMAIN, discover_affordance)
