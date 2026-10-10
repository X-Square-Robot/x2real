"""Policy implementation discovery for the global lazy registry."""

from __future__ import annotations

import importlib

from manaenv.pipeline.registry.convention_provider import camel_to_snake
from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = [
    "POLICY_IMPLEMENTATION_DOMAIN",
    "POLICY_IMPLEMENTATION_PACKAGE",
    "discover_policy_implementation",
    "policy_implementation_module_path",
]

POLICY_IMPLEMENTATION_DOMAIN = "policy_implementation"
POLICY_IMPLEMENTATION_PACKAGE = "manaenv.policy.policy_implementation"


def policy_implementation_module_path(symbol: str) -> str:
    stem = symbol[: -len("Impl")] if symbol.endswith("Impl") else symbol
    return f"{POLICY_IMPLEMENTATION_PACKAGE}.{camel_to_snake(stem)}"


def discover_policy_implementation(symbol: str) -> ComponentSpec | None:
    try:
        module = importlib.import_module(policy_implementation_module_path(symbol))
    except ModuleNotFoundError:
        return None
    expected_symbols = [symbol]
    if not symbol.endswith("Impl"):
        expected_symbols.append(f"{symbol}Impl")
    expected_names = {camel_to_snake(candidate) for candidate in expected_symbols}
    impl_class = next(
        (
            obj
            for obj in vars(module).values()
            if isinstance(obj, type)
            and obj.__name__.endswith("Impl")
            and camel_to_snake(obj.__name__) in expected_names
        ),
        None,
    )
    if impl_class is None:
        return None
    return ComponentSpec(module_path=module.__name__, class_name=impl_class.__name__, symbol=symbol)


register_provider(POLICY_IMPLEMENTATION_DOMAIN, discover_policy_implementation)
