"""Actuator discovery for the global lazy registry.

The YAML ``actuator`` values are stable user-facing symbols. This domain maps
those symbols to Isaac Lab / manaenv actuator CFG classes while keeping the
selection logic in ``select.py`` registry-backed.
"""

from __future__ import annotations

from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = ["ACTUATOR_DOMAIN", "ACTUATOR_SPECS", "discover_actuator"]

ACTUATOR_DOMAIN: str = "actuator"

ACTUATOR_SPECS: dict[str, tuple[str, str]] = {
    "pd": ("isaaclab.actuators", "ImplicitActuatorCfg"),
    "stablepd": ("isaaclab.actuators", "StablePDActuatorCfg"),
}


def discover_actuator(symbol: str) -> ComponentSpec | None:
    """Resolve a YAML actuator symbol to an actuator CFG class."""
    spec = ACTUATOR_SPECS.get(symbol)
    if spec is None:
        return None
    module_path, class_name = spec
    return ComponentSpec(module_path=module_path, class_name=class_name, symbol=symbol)


register_provider(ACTUATOR_DOMAIN, discover_actuator)
