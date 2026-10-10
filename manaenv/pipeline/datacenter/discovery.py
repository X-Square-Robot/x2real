"""DataCenter discovery for the global lazy registry."""

from __future__ import annotations

from pathlib import Path

from manaenv.pipeline.registry.convention_provider import camel_to_snake, make_convention_domain
from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = [
    "DATACENTER_DOMAIN",
    "DATACENTER_PACKAGE",
    "OUTPUT_DOMAIN",
    "camel_to_snake",
    "datacenter_module_path",
    "discover_datacenter",
    "discover_output",
]

DATACENTER_DOMAIN: str = "datacenter"
DATACENTER_PACKAGE: str = "manaenv.pipeline.datacenter"
OUTPUT_DOMAIN: str = "output"

datacenter_module_path, discover_datacenter = make_convention_domain(
    DATACENTER_DOMAIN,
    base_package=DATACENTER_PACKAGE,
    base_dir=Path(__file__).resolve().parent,
    suffix="Center",
    family_dir_suffix=None,
)


def discover_output(symbol: str) -> ComponentSpec | None:
    """Resolve the top-level output section to typed output orders."""
    if symbol != "Output":
        return None
    return ComponentSpec(
        module_path="manaenv.pipeline.datacenter.output_orders",
        class_name="OutputOrders",
        symbol=symbol,
    )


register_provider(OUTPUT_DOMAIN, discover_output)
