"""Scene component discovery for the global lazy registry.

Scene component YAML sections resolve by CFG name convention:

* ``background_component`` -> ``BackgroundComponent``
* ``objects_component`` -> ``ObjectsComponent``
* ``render_settings_component`` -> ``RenderSettingsComponent``

The lazy registry imports
``manaenv.pipeline.generators.scene.components.<snake>_component`` and
resolves the paired ``<symbol>CFG`` class on demand.
"""

from __future__ import annotations

from pathlib import Path

from manaenv.pipeline.registry.convention_provider import make_convention_domain
from manaenv.pipeline.registry.global_provider import global_registry

__all__ = [
    "SCENE_COMPONENT_DOMAIN",
    "SCENE_COMPONENT_PACKAGE",
    "component_symbol_from_section_key",
    "discover_scene_component",
    "is_scene_component_section_key",
    "resolve_scene_component_cfg",
    "scene_component_module_path",
]

SCENE_COMPONENT_DOMAIN: str = "scene_component"
SCENE_COMPONENT_PACKAGE: str = "manaenv.pipeline.generators.scene.components"

scene_component_module_path, discover_scene_component = make_convention_domain(
    SCENE_COMPONENT_DOMAIN,
    base_package=SCENE_COMPONENT_PACKAGE,
    base_dir=Path(__file__).resolve().parent / "components",
    suffix="Component",
    family_dir_suffix=None,
)


def component_symbol_from_section_key(key: str) -> str:
    if key.endswith("CFG"):
        return key[: -len("CFG")]
    if "_" in key:
        return "".join(part.capitalize() for part in key.split("_"))
    if key.endswith("Component"):
        return key
    return ""


def resolve_scene_component_cfg(key: str) -> type | None:
    from manaenv.pipeline.generators.scene.component import SceneComponentCFG

    symbol = component_symbol_from_section_key(key)
    if not symbol:
        return None
    cfg_class = global_registry.get_cfg(symbol)
    if cfg_class is not None and issubclass(cfg_class, SceneComponentCFG):
        return cfg_class
    return None


def is_scene_component_section_key(key: str) -> bool:
    return resolve_scene_component_cfg(key) is not None
