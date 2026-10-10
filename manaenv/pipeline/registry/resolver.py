"""Dynamic type / function resolution helpers used by the lazy registry.

Direct copy of the ``test_registry`` prototype (``resolver.py``); kept
intentionally minimal so other domains can rely on the exact same
contract when implementing their own discovery providers.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable


def resolve_obj(name: str, default_module: str | None = None) -> object:
    """Resolve a Python object from a short name or a full dotted path."""
    if "." in name:
        module_name, attr_name = name.rsplit(".", 1)
    else:
        if default_module is None:
            raise ValueError(f"default_module is required for short symbol names: {name}")
        module_name, attr_name = default_module, name

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def resolve_func(name: str, default_module: str | None = None) -> Callable:
    """Resolve a callable from a short name or a full dotted path."""
    resolved = resolve_obj(name, default_module)
    if not callable(resolved):
        raise TypeError(f"Resolved symbol is not callable: {name}")
    return resolved


def resolve_type(name: str, default_module: str | None = None) -> type:
    """Resolve a type from a short name or a full dotted path."""
    resolved = resolve_obj(name, default_module)
    if not isinstance(resolved, type):
        raise TypeError(f"Resolved symbol is not a type: {name}")
    return resolved
