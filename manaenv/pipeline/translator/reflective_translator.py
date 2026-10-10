"""Reflective translator: the single YAML -> CFG entrypoint for pipelines.

Centralizes YAML loading, ``include:`` expansion, and strict top-level
section reflection into registered CFG dataclasses. Sections whose key does
not resolve to a registered CFG symbol stay as raw YAML values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_origin, get_type_hints

from manaenv.utils.config_io import load_dict_from_yaml


def load_root_yaml(path: str | Path) -> dict[str, Any]:
    """Load a pipeline root YAML mapping."""
    return load_dict_from_yaml(path)


def _normalize_include_paths(include_value: Any) -> list[str]:
    """Normalize ``include:`` to an ordered list of non-empty relative paths.

    Accepts a single string or a non-empty list of strings. Later entries
    override earlier ones under the same merge contract as local keys.
    """
    if isinstance(include_value, str):
        if not include_value:
            raise ValueError(f"include must be a non-empty string, got {include_value!r}")
        return [include_value]
    if isinstance(include_value, list):
        if not include_value:
            raise ValueError("include list must be non-empty")
        paths: list[str] = []
        for item in include_value:
            if not isinstance(item, str) or not item:
                raise ValueError(f"include list entries must be non-empty strings, got {item!r}")
            paths.append(item)
        return paths
    raise ValueError(f"include must be a non-empty string or list of strings, got {include_value!r}")


# Top-level scene backdrop keys must replace wholesale when stacked via
# ``include:``. Deep-merging ``background.asset`` can leave stale fields such as
# ``id: green_booth_mesh`` beside a later ``type: pool`` variant, which pollutes
# random background sampling. Nested keys under these sections still deep-merge
# when authored as local overrides of an already-expanded mapping.
_WHOLESALE_REPLACE_TOP_LEVEL_KEYS = frozenset({"background", "lighting", "render_settings"})


def expand_includes(node: Any, base_dir: str | Path) -> Any:
    """Recursively expand ``include:`` mappings relative to ``base_dir``.

    The merge contract is intentionally literal: included mappings are loaded
    first and local mapping keys override included values recursively while
    lists/scalars replace wholesale. ``include`` may be a string or a list of
    strings; list entries merge left-to-right (later overrides earlier), then
    local keys override the stacked includes. Includes do not inject authoring
    sugar such as a filename-derived ``name``; YAML must spell every field
    explicitly.

    Exception: when stacking include files (and local keys over that stack),
    top-level ``background``, ``lighting``, and ``render_settings`` replace
    wholesale so a later fragment can fully swap green_booth backdrops without
    residual asset ids. ``render_settings.tags: green_booth`` OCIO packs remain
    valid when the later fragment still declares them.
    """
    root = Path(base_dir)

    def merge_dicts(
        base: Mapping[str, Any],
        overrides: Mapping[str, Any],
        *,
        wholesale_top_level: bool = False,
    ) -> dict[str, Any]:
        merged = dict(base)
        for key, value in overrides.items():
            existing = merged.get(key)
            if wholesale_top_level and key in _WHOLESALE_REPLACE_TOP_LEVEL_KEYS and isinstance(value, Mapping):
                merged[key] = value
            elif isinstance(existing, Mapping) and isinstance(value, Mapping):
                merged[key] = merge_dicts(existing, value)
            else:
                merged[key] = value
        return merged

    if isinstance(node, list):
        return [expand_includes(item, root) for item in node]

    if not isinstance(node, Mapping):
        return node

    if "include" not in node:
        return {key: expand_includes(value, root) for key, value in node.items()}

    include_paths = _normalize_include_paths(node.get("include"))
    merged: dict[str, Any] = {}
    for include_rel in include_paths:
        included = load_dict_from_yaml(root / include_rel)
        if not isinstance(included, Mapping):
            raise ValueError(f"included YAML must be a mapping: {include_rel}")
        merged = merge_dicts(merged, included, wholesale_top_level=True)
    overrides = {key: value for key, value in node.items() if key != "include"}
    merged = merge_dicts(merged, overrides, wholesale_top_level=True)
    return expand_includes(merged, root)


def _yaml_key_to_cfg_symbol(yaml_key: str) -> str:
    if "_" not in yaml_key and any(char.isupper() for char in yaml_key):
        return yaml_key
    return "".join(part.capitalize() for part in yaml_key.split("_"))


def _single_list_field_name(cfg_class: type) -> str | None:
    """Return the sole list field name for section wrappers, if one exists."""
    if not is_dataclass(cfg_class):
        return None
    try:
        resolved_types = get_type_hints(cfg_class)
    except Exception:
        resolved_types = {}

    list_fields: list[str] = []
    for field_info in fields(cfg_class):
        field_type = resolved_types.get(field_info.name, field_info.type)
        origin = get_origin(field_type)
        if origin is list or str(field_type).startswith("list["):
            list_fields.append(field_info.name)
    return list_fields[0] if len(list_fields) == 1 else None


def _reflect_top_level_section(section_key: str, expanded: Any, *, context: str) -> tuple[bool, Any]:
    """Reflect a top-level section without section-name-specific branches."""
    from manaenv.pipeline.registry.global_provider import global_registry
    from manaenv.pipeline.translator.yaml_reflection import reflect_yaml_dict

    symbol = _yaml_key_to_cfg_symbol(section_key)
    cfg_class = global_registry.get_cfg(symbol)
    if cfg_class is None:
        return False, None

    payload = expanded
    if isinstance(payload, list):
        list_field = _single_list_field_name(cfg_class)
        if list_field is None:
            return False, None
        payload = {list_field: expanded}

    reflected = reflect_yaml_dict({section_key: payload}, context=context)
    return True, reflected[section_key]


def _reflect_nested_tree(node: Any, *, context: str) -> Any:
    """Recursively reflect every CFG-addressable YAML key in a tree."""
    from manaenv.pipeline.translator.yaml_reflection import reflect_yaml_dict

    if isinstance(node, list):
        return [_reflect_nested_tree(item, context=f"{context}[{idx}]") for idx, item in enumerate(node)]

    if not isinstance(node, Mapping):
        return node

    reflected_mapping: dict[str, Any] = {}
    for key, value in node.items():
        key_context = f"{context}.{key}"
        child = _reflect_nested_tree(value, context=key_context)
        if key == "runtime":
            reflected_mapping[key] = child
            continue
        if isinstance(child, Mapping) or (isinstance(child, list) and all(isinstance(item, Mapping) for item in child)):
            reflected = reflect_yaml_dict({key: child}, context=context, ignore_unknown=True)
            if key in reflected:
                reflected_mapping[key] = reflected[key]
                continue
        reflected_mapping[key] = child
    return reflected_mapping


def _reflect_extension_section(
    section_key: str,
    expanded: Any,
    *,
    context: str,
) -> tuple[bool, Any]:
    from manaenv.pipeline.generators.scene_layout_solver.manaenv_bridge.translator_bridge import (
        reflect_scene_layout_solver_section,
    )

    return reflect_scene_layout_solver_section(section_key, expanded, context=context)


def _default_optional_section(name: str) -> Any | None:
    if name == "rollout":
        from manaenv.pipeline.factories.rollout_factory_cfg import RolloutOrdersCFG

        return RolloutOrdersCFG()
    if name == "output":
        from manaenv.pipeline.datacenter.output_orders import OutputOrdersCFG

        return OutputOrdersCFG()
    return None


def reflect_section(section_key: str, chunk: Any, *, base_dir: str | Path, context: str | None = None) -> Any:
    """Expand includes and reflect the current top-level section when known."""
    label = context or f"pipeline.{section_key}"
    expanded = expand_includes(chunk, base_dir)

    if isinstance(expanded, list) and section_key.endswith("_orders"):
        inner_section_key = section_key.removesuffix("_orders")
        reflected, value = _reflect_extension_section(
            inner_section_key,
            expanded,
            context=f"{label}.orders",
        )
        if reflected:
            return value

    if section_key != "task":
        reflected, value = _reflect_extension_section(section_key, expanded, context=label)
        if reflected:
            return value

    expanded = _reflect_nested_tree(expanded, context=label)
    reflected, value = _reflect_top_level_section(section_key, expanded, context=label)
    if reflected:
        return value
    return expanded


class ReflectiveTranslator:
    """Phase-1 sidecar translator exposing only reflected/raw section fields."""

    def __init__(self) -> None:
        from manaenv.pipeline.registry.bootstrap import bootstrap_discovery

        bootstrap_discovery()
        self.fields: dict[str, Any] = {}
        self._yaml_dir: Path | None = None

    def precompile(self, pipeline_description_yaml_path: str | Path) -> dict[str, Any]:
        """Load a root YAML and expand includes without reflecting CFGs."""
        if pipeline_description_yaml_path is None:
            raise ValueError("Pipeline description YAML path is required but not provided.")

        yaml_path = Path(pipeline_description_yaml_path)
        self._yaml_dir = yaml_path.resolve().parent
        config = load_root_yaml(yaml_path)
        return expand_includes(config, self._yaml_dir)

    def compile(self, precompiled_tree: Mapping[str, Any]) -> dict[str, Any]:
        """Reflect a precompiled YAML tree into section fields."""
        if self._yaml_dir is None:
            raise RuntimeError("precompile() must be called before compile() so include base_dir is known.")
        if not isinstance(precompiled_tree, Mapping):
            raise TypeError(f"compile() expected a mapping, got {type(precompiled_tree).__name__}")

        self.fields = {
            section_key: reflect_section(section_key, chunk, base_dir=self._yaml_dir)
            for section_key, chunk in precompiled_tree.items()
        }
        return self.fields

    def parse_pipe_instructions(self, pipeline_description_yaml_path: str) -> None:
        if pipeline_description_yaml_path is None:
            raise ValueError("Pipeline description YAML path is required but not provided.")

        precompiled_tree = self.precompile(pipeline_description_yaml_path)
        self.compile(precompiled_tree)

    def section(self, name: str, *, required: bool = False) -> Any:
        """Return one reflected/raw pipeline section from ``fields``.

        ``fields`` remains the only state; this is the single convenience
        access point for downstream code that should not index the dict
        directly.
        """
        if name not in self.fields:
            default_value = _default_optional_section(name)
            if default_value is not None:
                return default_value
            if required:
                raise KeyError(f"Missing pipeline section: {name}")
            return None
        return self.fields[name]
