"""Scene template discovery for the global lazy registry."""

from __future__ import annotations

from pathlib import Path

from manaenv.pipeline.registry.convention_provider import camel_to_snake
from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = [
    "SCENE_TEMPLATE_DOMAIN",
    "SCENE_TEMPLATE_PACKAGE",
    "camel_to_snake",
    "discover_scene_template",
    "scene_template_module_path",
]

SCENE_TEMPLATE_DOMAIN: str = "scene_template"
SCENE_TEMPLATE_PACKAGE: str = "manaenv.pipeline.generators.scene.templates"

_SCENE_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def scene_template_module_path(symbol: str) -> str | None:
    """Return the module path that hosts a ``*Scene`` template, or ``None``."""
    if not symbol.endswith("Scene") or symbol == "Scene":
        return None
    stem = camel_to_snake(symbol)
    if (_SCENE_TEMPLATE_DIR / f"{stem}.py").exists():
        return f"{SCENE_TEMPLATE_PACKAGE}.{stem}"
    return None


def discover_scene_template(symbol: str) -> ComponentSpec | None:
    module_path = scene_template_module_path(symbol)
    if module_path is None:
        return None
    return ComponentSpec(module_path=module_path, class_name=symbol, symbol=symbol)


register_provider(SCENE_TEMPLATE_DOMAIN, discover_scene_template)
