"""Translator-boundary bridge: schema-vNext scene config -> ManaEnv scene order.

A scene order authored in the schema-vNext scene format (top-level
``background`` / ``objects`` / ``layout`` / ``variants``) is not directly
reflectable into ``SceneOrderCFG`` because the public vNext fields differ from
the Scene DSL component fields the factory consumes. This module materializes a
vNext scene config into the standard scene-order shape so the existing scene
factory builds assets normally, while ``scene_generator.metadata`` carries the
``schema_vnext_scene`` / ``schema_vnext_materialization`` / ``layout`` payloads
that static layout solving and the task-side runtime-sample bridge read.

Kept import-light: the schema-vNext machinery is imported lazily so non-vNext
pipelines never pay for it.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

# Definition keys that map onto SceneGeneratorCFG fields (everything else stays
# in metadata, which the factory copies onto the produced Scene).
_SCENE_GENERATOR_PASSTHROUGH = (
    "nums",
    "sim_fps",
    "render_decremental",
    "simulation_decimation",
    "antialiasing_mode",
    "seed",
)

# Component fields copied verbatim from the materialized payload.
_COMPONENT_PASSTHROUGH = (
    "semantic_tags",
    "scale",
    "initial_position",
    "initial_orientation",
    "spawn_mass",
    "spawn_static_friction",
    "spawn_dynamic_friction",
    "kinematic",
    "collision_approximation",
    "strip_child_z_rotation",
)

_RENDER_SETTINGS_COMPONENT_PASSTHROUGH = (
    "enable_dlssg",
    "dlssg_multiplier",
    "dlss_mode",
    "carb_settings",
)


def looks_like_vnext_scene_order(config: Any) -> bool:
    """Cheap structural probe that avoids importing schema-vNext for legacy orders."""
    return (
        isinstance(config, Mapping)
        and isinstance(config.get("background"), Mapping)
        and isinstance(config.get("layout"), Mapping)
        and isinstance(config.get("variants"), Mapping)
    )


def is_vnext_scene_order(config: Any) -> bool:
    """Authoritative schema-vNext scene-config check (lazy import)."""
    if not looks_like_vnext_scene_order(config):
        return False
    from .schema_vnext import is_vnext_scene_config

    return is_vnext_scene_config(config)


def vnext_scene_order_from_config(
    config: Mapping[str, Any],
    *,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Materialize a schema-vNext scene config into a standard scene-order dict."""
    from .manaenv_bridge.asset_tags import resolve_manaenv_asset_ids_by_tag
    from .schema_vnext import materialize_manaenv_vnext_scene_config

    materialized = materialize_manaenv_vnext_scene_config(
        config,
        rng=rng,
        asset_tag_resolver=resolve_manaenv_asset_ids_by_tag,
    )

    definition = dict(materialized.get("definition") or {})
    metadata = dict(definition.get("metadata") or {})
    scene_type = str(definition.get("scene_type") or "").strip()

    scene_generator: dict[str, Any] = {"metadata": metadata}
    if scene_type:
        scene_generator["scene_type"] = scene_type
    for key in _SCENE_GENERATOR_PASSTHROUGH:
        if key in definition:
            scene_generator[key] = definition[key]

    order: dict[str, Any] = {
        "name": str(metadata.get("name") or scene_type or "scene"),
        "scene_generator": scene_generator,
    }

    objects = [_component_from_materialized(obj) for obj in (materialized.get("objects") or [])]
    if objects:
        order["objects_component"] = objects

    background = materialized.get("background")
    if isinstance(background, Mapping):
        order["background_component"] = _component_from_materialized(background)

    render_settings = materialized.get("render_settings")
    if isinstance(render_settings, Mapping):
        rs_component: dict[str, Any] = {}
        asset = render_settings.get("asset") or render_settings.get("tags")
        if asset:
            rs_component["asset"] = str(asset)
        aa_mode = render_settings.get("antialiasing_mode")
        if aa_mode is not None and str(aa_mode).strip():
            mode = str(aa_mode).strip()
            rs_component["antialiasing_mode"] = mode
            # Prefer the render_settings YAML switch over definition passthrough.
            scene_generator["antialiasing_mode"] = mode
        for key in _RENDER_SETTINGS_COMPONENT_PASSTHROUGH:
            if key in render_settings:
                rs_component[key] = render_settings[key]
        if rs_component:
            order["render_settings_component"] = rs_component

    return order


def _component_from_materialized(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Map a materialized component payload (Scene DSL v0) to Scene DSL v1 fields."""
    out: dict[str, Any] = {}
    name = payload.get("name") or payload.get("id")
    if name:
        out["id"] = str(name)
    asset = payload.get("asset") or payload.get("tags")
    if asset:
        out["asset"] = str(asset)
    for key in _COMPONENT_PASSTHROUGH:
        if key in payload:
            out[key] = payload[key]
    return out
