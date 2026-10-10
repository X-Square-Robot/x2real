"""ActionTransform discovery for the global lazy registry.

The transform classes live in ``action_transforms.py``; ``command_space``
(derived from ``embodiment.action_mode`` by the wallx adapter) maps onto these
symbols. ``create_action_transform`` resolves the class via
``global_registry.get_impl`` instead of a local class dict.
"""

from __future__ import annotations

from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = ["ACTION_TRANSFORM_DOMAIN", "discover_action_transform"]

ACTION_TRANSFORM_DOMAIN: str = "action_transform"

_MODULE: str = "manaenv.policy.action_transforms"

# symbol -> hosting module (all transforms share one module).
_ACTION_TRANSFORM_SYMBOLS: dict[str, str] = {
    "FollowPosConcatTransform": _MODULE,
    "EEAbsoluteToDeltaTransform": _MODULE,
    "EEAbsoluteIKTransform": _MODULE,
    "WholeBodyEEAbsoluteIKTransform": _MODULE,
    "RawActionTransform": _MODULE,
}


def discover_action_transform(symbol: str) -> ComponentSpec | None:
    """Resolve an ActionTransform symbol to its class."""
    module_path = _ACTION_TRANSFORM_SYMBOLS.get(symbol)
    if module_path is None:
        return None
    return ComponentSpec(module_path=module_path, class_name=symbol, symbol=symbol)


register_provider(ACTION_TRANSFORM_DOMAIN, discover_action_transform)
