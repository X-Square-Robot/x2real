"""Scene-layout-solver extension for ManaEnv reflective translation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from manaenv.pipeline.data_types import GenMethod
from manaenv.pipeline.generators.task.component_discovery import is_task_component_section_key
from manaenv.pipeline.registry.global_provider import global_registry
from manaenv.pipeline.translator.yaml_reflection import reflect_yaml_dict


def reflect_scene_layout_solver_section(
    section_key: str,
    expanded: Any,
    *,
    context: str,
) -> tuple[bool, Any]:
    """Reflect scene-layout-solver public scene/task authoring sections."""
    if section_key == "scene":
        return _reflect_scene_section(expanded, context=context)
    if section_key == "task":
        return _reflect_task_section(expanded)
    return False, None


def _is_scene_entry(scene_config: Any) -> bool:
    if not isinstance(scene_config, Mapping):
        return False
    from manaenv.pipeline.generators.scene_layout_solver.schema_vnext.manaenv_scene_bridge import is_vnext_scene_config

    layout = scene_config.get("layout")
    if is_vnext_scene_config(scene_config) or (isinstance(layout, Mapping) and bool(layout.get("enabled"))):
        return True
    if "definition" in scene_config or "scene_generator" in scene_config:
        component_keys = {
            "objects_component",
            "background_component",
            "render_settings_component",
            "lighting_component",
        }
        return any(key in scene_config for key in component_keys)
    return False


def _reflect_scene_section(expanded: Any, *, context: str) -> tuple[bool, Any]:
    entries = expanded if isinstance(expanded, list) else [expanded]
    if not any(_is_scene_entry(entry) for entry in entries):
        return False, None

    from manaenv.pipeline.factories.scene_factory import SceneOrderCFG, SceneOrdersCFG
    from manaenv.pipeline.generators.scene_layout_solver.integration import materialize_manaenv_scene_config
    from manaenv.pipeline.generators.scene_layout_solver.manaenv_bridge.asset_tags import (
        resolve_manaenv_asset_ids_by_tag,
    )
    from manaenv.pipeline.generators.scene_layout_solver.schema_vnext.manaenv_scene_bridge import (
        is_vnext_scene_config,
        materialize_manaenv_vnext_scene_config,
    )

    def component_payload(payload: Any) -> Any:
        if isinstance(payload, list):
            return [component_payload(item) for item in payload]
        if not isinstance(payload, Mapping):
            return payload
        out = dict(payload)
        if "id" not in out and out.get("name"):
            out["id"] = out["name"]
        if "asset" not in out and out.get("tags"):
            out["asset"] = out["tags"]
        out.pop("gen_method", None)
        out.pop("name", None)
        out.pop("tags", None)
        return out

    def reflective_component_keys(scene_config: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(scene_config)
        if isinstance(out.get("render_settings"), Mapping):
            render_settings = dict(out["render_settings"])
            if "asset" not in render_settings:
                if render_settings.get("tags"):
                    render_settings["asset"] = render_settings["tags"]
                elif render_settings.get("asset_tag"):
                    render_settings["asset"] = render_settings["asset_tag"]
            render_settings.pop("gen_method", None)
            render_settings.pop("tags", None)
            out["render_settings"] = render_settings
        for component_key in ("background", "objects"):
            if component_key in out:
                out[component_key] = component_payload(out[component_key])
        aliases = {
            "background": "background_component",
            "objects": "objects_component",
            "render_settings": "render_settings_component",
        }
        for old_key, new_key in aliases.items():
            if old_key in out and new_key not in out:
                out[new_key] = out.pop(old_key)
        return out

    def materialize_one(scene_config: Any) -> Any:
        if not isinstance(scene_config, Mapping):
            return scene_config
        scene_dict = dict(scene_config)
        if is_vnext_scene_config(scene_dict):
            return reflective_component_keys(
                materialize_manaenv_vnext_scene_config(
                    scene_dict,
                    asset_tag_resolver=resolve_manaenv_asset_ids_by_tag,
                )
            )
        if isinstance(scene_dict.get("layout"), Mapping) and scene_dict["layout"].get("enabled"):
            return reflective_component_keys(materialize_manaenv_scene_config(scene_dict))
        return scene_config

    def reflect_component(component_type: str, payload: Any) -> Any:
        reflected = reflect_yaml_dict({component_type: payload}, context=context, ignore_unknown=True)
        return reflected.get(component_type)

    orders: list[SceneOrderCFG] = []
    for idx, scene_config in enumerate(entries):
        scene_dict = materialize_one(scene_config)
        if not isinstance(scene_dict, Mapping):
            raise TypeError(f"{context}[{idx}]: scene entry must materialize to a mapping")
        definition = scene_dict.get("scene_generator") or scene_dict.get("definition")
        if not isinstance(definition, Mapping):
            raise ValueError(f"{context}[{idx}]: scene entry is missing scene_generator/definition mapping")

        definition = dict(definition)
        top_level_layout = scene_dict.get("layout")
        if isinstance(top_level_layout, Mapping):
            metadata = dict(definition.get("metadata") or {})
            metadata["layout"] = dict(top_level_layout)
            definition["metadata"] = metadata

        scene_generator = reflect_yaml_dict(
            {"scene_generator": definition},
            context=f"{context}[{idx}].scene_generator",
        )["scene_generator"]
        scene_name = str(scene_generator.scene_type or definition.get("scene_type") or "")
        if not scene_name:
            raise ValueError(f"{context}[{idx}]: scene_generator.scene_type is required")
        order_name = str(scene_dict.get("name") or scene_name)

        order_payload: dict[str, Any] = {
            "name": order_name,
            "scene_generator": scene_generator,
        }
        for component_key in (
            "objects_component",
            "background_component",
            "render_settings_component",
            "lighting_component",
        ):
            if component_key not in scene_dict:
                continue
            reflected_component = reflect_component(component_key, scene_dict[component_key])
            if reflected_component is None:
                continue
            if component_key == "objects_component" and not isinstance(reflected_component, list):
                reflected_component = [reflected_component]
            order_payload[component_key] = reflected_component
        orders.append(SceneOrderCFG(**order_payload))

    return True, SceneOrdersCFG(orders=orders)


_TASK_GLOBAL_DEFINITION_KEYS: frozenset[str] = frozenset(
    {
        "gen_method",
        "episode_length_s",
        "prompt",
        "num_episodes",
        "max_steps_per_episode",
        "robot_constraint",
        "benchmark",
        "init_states_file",
        "seed",
        "layout",
    }
)

_TASK_RESERVED_CFG_KWARGS: frozenset[str] = frozenset(
    {"class_type", "gen_num", "metadata", "definition", "roles", "component_sections"}
)


def _canonical_definition_key(payload_key: str) -> str:
    if payload_key == "robot_constraint":
        return "robot_constraints"
    return payload_key


def _task_generator_symbol_from_gen_method(gen_method: Any) -> str:
    method = GenMethod.DIRECT if gen_method is None else GenMethod(gen_method)
    if method == GenMethod.DIRECT:
        return "DirectTaskGenerator"
    if method == GenMethod.CONSTRAINT:
        return "ConstraintTaskGenerator"
    raise ValueError(f"Unsupported task gen_method {gen_method!r}; supported values are DIRECT(0) and CONSTRAINT(2).")


def _resolve_task_generator_payload(task_block: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(task_block, Mapping):
        raise ValueError(f"task yaml must declare a mapping under 'task:', got: {task_block!r}")

    explicit_symbols = [key for key in task_block if key != "name" and str(key).endswith("_task_generator")]
    if explicit_symbols:
        if len(explicit_symbols) != 1:
            raise ValueError(f"task yaml declares multiple explicit task generator envelopes: {explicit_symbols!r}")
        symbol = str(explicit_symbols[0])
        payload = task_block[symbol]
        if not isinstance(payload, Mapping):
            raise ValueError(f"task envelope payload for {symbol!r} must be a dict, got {type(payload).__name__}")
        return symbol, dict(payload)

    if "definition" in task_block:
        payload = dict(task_block)
        definition = payload.pop("definition")
        if not isinstance(definition, Mapping):
            raise ValueError(f"task.definition must be a dict, got {type(definition).__name__}")
        gen_method = definition.get("gen_method")
        if gen_method is None and ("template" in task_block or "payload" in task_block):
            symbol = "ConstraintTaskGenerator"
        else:
            symbol = _task_generator_symbol_from_gen_method(gen_method)
        payload.update(definition)
        return symbol, payload

    if len(task_block) != 1:
        raise ValueError(
            "task yaml must declare either 'task: { definition: { gen_method: ... }, ... }' "
            "or a single 'task: { <TaskGeneratorSymbol>: { ... } }' envelope; "
            f"got: {task_block!r}"
        )

    symbol, payload = next(iter(task_block.items()))
    if not isinstance(symbol, str) or not symbol:
        raise ValueError(f"task envelope key must be a TaskGenerator class symbol string, got {symbol!r}")
    if not isinstance(payload, Mapping):
        raise ValueError(f"task envelope payload for {symbol!r} must be a dict, got {type(payload).__name__}")
    return symbol, dict(payload)


def _instantiate_task_gen_cfg(*, cfg_class: Any, payload: dict[str, Any], symbol: str, task_type: str) -> Any:
    cfg_field_names = {field_info.name for field_info in fields(cfg_class)}
    is_constraint_task = cfg_class.__name__ == "ConstraintTaskGeneratorCFG"
    component_sections: dict[str, Any] = {}
    definition: dict[str, Any] = {"task_type": task_type}
    metadata: dict[str, Any] = {}
    roles: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    template: str = ""
    template_payload: dict[str, Any] = {}
    unknown: list[str] = []

    for key, value in payload.items():
        if key == "task_type":
            definition["task_type"] = value
            continue
        if key == "definition":
            if not isinstance(value, Mapping):
                raise ValueError(f"task envelope {symbol!r} 'definition' must be a dict, got {type(value).__name__}")
            definition.update(value)
            continue
        if key == "metadata":
            if not isinstance(value, Mapping):
                raise ValueError(f"task envelope {symbol!r} 'metadata' must be a dict, got {type(value).__name__}")
            metadata.update(value)
            continue
        if key == "acceptance":
            metadata["acceptance"] = value
            continue
        if key == "roles":
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"task envelope {symbol!r} 'roles' must be a dict of <slot_name>: <tag>, "
                    f"got {type(value).__name__}"
                )
            roles = dict(value)
            continue
        if key == "component_sections":
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"task envelope {symbol!r} 'component_sections' must be a dict, got {type(value).__name__}"
                )
            component_sections.update(value)
            continue
        if is_task_component_section_key(key):
            component_sections[key] = value if isinstance(value, Mapping) else {"constraints": value}
            continue
        if key in _TASK_GLOBAL_DEFINITION_KEYS:
            definition[_canonical_definition_key(key)] = value
            continue
        if is_constraint_task and key == "template":
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"task envelope {symbol!r} (task_type={task_type!r}) field 'template' "
                    f"must be a non-empty task template string, got {value!r}."
                )
            if template and template != value:
                raise ValueError(
                    f"task envelope {symbol!r} (task_type={task_type!r}) declares conflicting task "
                    f"templates: {template!r} and {value!r}."
                )
            template = value
            continue
        if is_constraint_task and key == "payload":
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"task envelope {symbol!r} (task_type={task_type!r}) field 'payload' "
                    f"must be a dict, got {type(value).__name__}."
                )
            if template_payload:
                raise ValueError(
                    f"task envelope {symbol!r} (task_type={task_type!r}) declares multiple task template payloads."
                )
            template_payload = dict(value)
            continue
        if is_constraint_task and key == "template_payload":
            raise ValueError(
                f"task envelope {symbol!r} (task_type={task_type!r}) should use public field "
                "'payload:' with 'template:'; 'template_payload:' is internal to ConstraintTaskGeneratorCFG."
            )
        if is_constraint_task and isinstance(value, Mapping) and key.endswith("Task"):
            if global_registry.get_spec(key) is None:
                unknown.append(key)
                continue
            if template:
                raise ValueError(
                    f"task envelope {symbol!r} (task_type={task_type!r}) mixes public template field "
                    f"{template!r} with task-class section {key!r}; use only one style."
                )
            template = key
            template_payload = dict(value)
            continue
        if key in cfg_field_names and key not in _TASK_RESERVED_CFG_KWARGS:
            overrides[key] = value
            continue
        unknown.append(key)

    if is_constraint_task and not template:
        raise ValueError(
            f"task envelope {symbol!r} (task_type={task_type!r}) must declare 'template: <TaskClass>' "
            "and optional 'payload: {...}', or one task-class section such as 'ButtonContactTask: {...}'."
        )

    if unknown:
        raise ValueError(
            f"task envelope {symbol!r} (task_type={task_type!r}) has unrecognised payload keys {unknown!r}; "
            f"not a cfg field on {cfg_class.__name__}, not a known task component section discoverable by "
            f"the lazy task-component registry, and not a task-level global ({sorted(_TASK_GLOBAL_DEFINITION_KEYS)!r})."
        )

    cfg_kwargs = dict(overrides)
    if is_constraint_task:
        cfg_kwargs.update(template=template, template_payload=template_payload)

    return cfg_class(
        class_type=cfg_class.class_type,
        gen_num=1,
        metadata=metadata,
        definition=definition,
        roles=roles,
        component_sections=component_sections,
        **cfg_kwargs,
    )


def _build_task_gen_cfg_from_envelope(task_entry: Any) -> Any:
    if not isinstance(task_entry, Mapping):
        return task_entry
    task_block = task_entry.get("task", task_entry)
    symbol, payload = _resolve_task_generator_payload(task_block)

    cfg_probe = next(iter(reflect_yaml_dict({symbol: {}}, context=f"task envelope {symbol!r}").values()), None)
    if cfg_probe is None:
        raise ValueError(f"task envelope key {symbol!r} did not resolve via the lazy task registry.")
    cfg_class = type(cfg_probe)

    nested_definition = payload.get("definition") if isinstance(payload.get("definition"), Mapping) else {}
    task_type = str(task_entry.get("name") or payload.get("task_type") or nested_definition.get("task_type") or "")
    if not task_type:
        raise ValueError(
            f"task envelope {symbol!r} is missing task_type; task YAML must declare definition.task_type "
            "or task_type at the generator payload level."
        )

    return _instantiate_task_gen_cfg(cfg_class=cfg_class, payload=payload, symbol=symbol, task_type=task_type)


def _reflect_task_section(expanded: Any) -> tuple[bool, Any]:
    entries = expanded if isinstance(expanded, list) else [expanded]
    if not entries or not all(isinstance(entry, Mapping) for entry in entries):
        return False, None

    needs_bridge = any(
        "definition" in entry or "template" in entry or "payload" in entry or "task" in entry for entry in entries
    )
    if not needs_bridge:
        return False, None

    from manaenv.pipeline.factories.task_factory import TaskOrdersCFG

    return True, TaskOrdersCFG(orders=[_build_task_gen_cfg_from_envelope(entry) for entry in entries])
