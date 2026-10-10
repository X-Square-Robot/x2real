from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from manaenv.pipeline.component import BaseComponent, BaseComponentCFG
from manaenv.pipeline.registry.resolver import resolve_func as _resolve_func


class TaskComponent(BaseComponent):
    """Task-specific component with helpers for func resolution and @role expansion."""

    @staticmethod
    def resolve_func(name: str, domain_module: str) -> Callable:
        """Resolve a term function through the shared lazy-compatible resolver."""
        return _resolve_func(name, domain_module)

    @staticmethod
    def resolve_roles(params: dict[str, Any], roles: dict[str, Any]) -> dict[str, Any]:
        """Replace ``@role`` references in *params* with bound role values."""
        resolved: dict[str, Any] = {}
        for k, v in params.items():
            resolved[k] = _resolve_value(v, roles)
        return resolved


def _resolve_value(value: Any, roles: dict[str, Any]) -> Any:
    """Recursively resolve ``@role`` references in a single value.

    The ``roles`` dict is populated from the same scene binder used by
    template slots, so values are generally bound assets or lists of
    assets. Components that need scene entity names should project at
    their own boundary.
    """
    if isinstance(value, str) and value.startswith("@"):
        role_name = value[1:]
        return _project_role_value(roles.get(role_name, role_name))
    if isinstance(value, list):
        return [_resolve_value(item, roles) for item in value]
    if isinstance(value, dict):
        return {k: _resolve_value(v, roles) for k, v in value.items()}
    return value


def _project_role_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_project_role_value(item) for item in value]
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return value


@dataclass
class TaskComponentCFG(BaseComponentCFG):

    class_type: type[TaskComponent] = None

    #: Whether this component's sections carry human-readable prose where
    #: ``@role.<attr>`` text interpolation applies (so role_splitter routes them
    #: wholesale to finalize when they contain any role ref). A component-type
    #: trait declared by the component itself — not enumerated in the splitter.
    renders_text_refs: ClassVar[bool] = False
