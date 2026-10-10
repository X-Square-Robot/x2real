"""Policy remote discovery for the global lazy registry."""

from __future__ import annotations

import importlib

from manaenv.pipeline.registry.convention_provider import camel_to_snake
from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = [
    "POLICY_REMOTE_DOMAIN",
    "POLICY_REMOTE_PACKAGE",
    "discover_policy_remote",
    "policy_remote_module_path",
]

POLICY_REMOTE_DOMAIN = "policy_remote"
POLICY_REMOTE_PACKAGE = "manaenv.policy.policy_remote"


def policy_remote_module_path(symbol: str) -> str:
    stem = symbol[: -len("Remote")] if symbol.endswith("Remote") else symbol
    module_stem = camel_to_snake(stem).replace("_", "")
    return f"{POLICY_REMOTE_PACKAGE}.{module_stem}"


def discover_policy_remote(symbol: str) -> ComponentSpec | None:
    try:
        module = importlib.import_module(policy_remote_module_path(symbol))
    except ModuleNotFoundError:
        return None
    expected = camel_to_snake(symbol)
    remote_class = next(
        (
            obj
            for obj in vars(module).values()
            if isinstance(obj, type) and obj.__name__.endswith("Remote") and camel_to_snake(obj.__name__) == expected
        ),
        None,
    )
    if remote_class is None:
        return None
    return ComponentSpec(module_path=module.__name__, class_name=remote_class.__name__, symbol=symbol)


register_provider(POLICY_REMOTE_DOMAIN, discover_policy_remote)
