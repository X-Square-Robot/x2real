"""Pipeline runtime CFG discovery for the global lazy registry."""

from __future__ import annotations

from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = ["RUNTIME_DOMAIN", "discover_runtime"]

RUNTIME_DOMAIN: str = "runtime"


def discover_runtime(symbol: str) -> ComponentSpec | None:
    """Resolve the existing PipelineRuntimeCFG without introducing a runtime class."""
    if symbol != "PipelineRuntime":
        return None
    return ComponentSpec(
        module_path="manaenv.pipeline.runtime_cfg",
        class_name="PipelineRuntime",
        symbol=symbol,
    )


register_provider(RUNTIME_DOMAIN, discover_runtime)
