"""Utility CFG discovery for process bootstrap sections."""

from __future__ import annotations

from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

APP_BOOTSTRAP_DOMAIN: str = "app_bootstrap"


def discover_app_bootstrap(symbol: str) -> ComponentSpec | None:
    """Resolve early app bootstrap CFGs."""
    if symbol != "AppBootstrap":
        return None
    return ComponentSpec(
        module_path="manaenv.utils.app_bootstrap_cfg",
        class_name="AppBootstrap",
        symbol=symbol,
    )


register_provider(APP_BOOTSTRAP_DOMAIN, discover_app_bootstrap)
