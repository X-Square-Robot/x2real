"""Role-aware splitting for task component sections.

This module is intentionally independent from concrete generators and
component classes. It only inspects raw YAML-shaped payloads and answers
two questions:

* Which ``@role`` names are referenced by a payload?
* Which component sections can be generated before scene binding, and
  which must wait until roles are resolved against scene assets?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.ref_text import collect_text_refs

ROLE_REFERENCE_PREFIX = "@"
ROLE_SPLIT_CONTAINER_KEYS = frozenset({"terms", "constraints", "sensors"})


def _is_text_ref_section(key: str) -> bool:
    """Whether *key*'s component declares it carries ``@role.<attr>`` prose.

    Such sections are routed wholesale to the finalize phase whenever they contain
    any role reference (inline attr ref or whole-string ref) so the renderer sees
    scene-bound asset objects. The general param-binding path
    (``split_component_section``) is left untouched, so inline ``@`` in non-text
    params (e.g. ``user@host.com``) is never mis-detected.

    The capability is a component-type trait declared by the component itself
    (``TaskComponentCFG.renders_text_refs``) and resolved through the component
    registry — the splitter does NOT enumerate concrete component identities.
    """
    from manaenv.pipeline.generators.task.component_discovery import resolve_task_component_cfg

    cfg_class = resolve_task_component_cfg(key)
    return bool(getattr(cfg_class, "renders_text_refs", False))


@dataclass
class RoleSplitResult:
    """Result of splitting component sections by role dependency."""

    instance_sections: dict[str, Any] = field(default_factory=dict)
    finalize_sections: dict[str, Any] = field(default_factory=dict)
    referenced_roles: set[str] = field(default_factory=set)


def collect_role_references(value: Any) -> set[str]:
    """Recursively collect unresolved ``@role`` references from *value*.

    The returned names exclude the leading ``@`` so callers can merge
    them directly into a runtime ``roles`` mapping as implicit
    ``slot -> tag`` bindings.
    """

    seen: set[int] = set()
    refs: set[str] = set()

    def _walk(obj: Any) -> None:
        if isinstance(obj, str):
            if obj.startswith(ROLE_REFERENCE_PREFIX) and len(obj) > 1:
                refs.add(obj[1:])
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(obj, dict):
            for item in obj.values():
                _walk(item)
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                _walk(item)
            return
        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            try:
                attrs = vars(obj)
            except TypeError:
                return
            for item in attrs.values():
                _walk(item)

    _walk(value)
    return refs


def contains_role_reference(value: Any) -> bool:
    """Return ``True`` when *value* contains at least one ``@role`` ref."""
    return bool(collect_role_references(value))


def split_component_sections_by_role_usage(sections: dict[str, Any]) -> RoleSplitResult:
    """Split component sections into instance-time and finalize-time payloads."""
    instance_sections: dict[str, Any] = {}
    finalize_sections: dict[str, Any] = {}
    text_referenced: set[str] = set()
    for key, section_data in (sections or {}).items():
        if _is_text_ref_section(key):
            text_refs = collect_text_refs(section_data)
            if text_refs or contains_role_reference(section_data):
                finalize_sections[key] = section_data
                text_referenced |= text_refs
            else:
                instance_sections[key] = section_data
            continue
        instance_payload, finalize_payload = split_component_section(section_data)
        if instance_payload:
            instance_sections[key] = instance_payload
        if finalize_payload:
            finalize_sections[key] = finalize_payload
    return RoleSplitResult(
        instance_sections=instance_sections,
        finalize_sections=finalize_sections,
        referenced_roles=collect_role_references(finalize_sections) | text_referenced,
    )


def split_component_section(section_data: Any) -> tuple[Any, Any]:
    """Split a single component section into ``(instance_payload, finalize_payload)``."""
    if not isinstance(section_data, dict):
        if contains_role_reference(section_data):
            return None, section_data
        return section_data, None

    split_keys = [key for key in section_data if key in ROLE_SPLIT_CONTAINER_KEYS]
    if not split_keys:
        if contains_role_reference(section_data):
            return None, dict(section_data)
        return dict(section_data), None

    instance_payload: dict[str, Any] = {}
    finalize_payload: dict[str, Any] = {}
    common_payload: dict[str, Any] = {}

    for key, value in section_data.items():
        if key not in ROLE_SPLIT_CONTAINER_KEYS:
            common_payload[key] = value
            continue
        early_value, late_value = split_role_container(value)
        if early_value:
            instance_payload[key] = early_value
        if late_value:
            finalize_payload[key] = late_value

    if contains_role_reference(common_payload):
        return None, dict(section_data)
    if instance_payload:
        instance_payload.update(common_payload)
    if finalize_payload:
        finalize_payload.update(common_payload)
    return instance_payload or None, finalize_payload or None


def split_role_container(value: Any) -> tuple[Any, Any]:
    """Split a section container while preserving dict/list shape."""
    if isinstance(value, dict):
        early: dict[str, Any] = {}
        late: dict[str, Any] = {}
        for key, item in value.items():
            if contains_role_reference(item):
                late[key] = item
            else:
                early[key] = item
        return early or None, late or None
    if isinstance(value, list):
        early_list: list[Any] = []
        late_list: list[Any] = []
        for item in value:
            if contains_role_reference(item):
                late_list.append(item)
            else:
                early_list.append(item)
        return early_list or None, late_list or None
    if contains_role_reference(value):
        return None, value
    return value, None
