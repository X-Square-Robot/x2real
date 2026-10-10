"""Factory discovery for the global lazy registry."""

from __future__ import annotations

from pathlib import Path

from manaenv.pipeline.registry.convention_provider import camel_to_snake, make_convention_domain
from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = [
    "FACTORY_DOMAIN",
    "FACTORY_ORDER_DOMAIN",
    "FACTORY_PACKAGE",
    "camel_to_snake",
    "discover_factory",
    "discover_factory_orders",
    "factory_module_path",
]

FACTORY_DOMAIN: str = "factory"
FACTORY_PACKAGE: str = "manaenv.pipeline.factories"
FACTORY_ORDER_DOMAIN: str = "factory_orders"

_ORDER_SPECS: dict[str, tuple[str, str]] = {
    "SceneOrders": ("manaenv.pipeline.factories.scene_factory", "SceneOrders"),
    "TaskOrders": ("manaenv.pipeline.factories.task_factory", "TaskOrders"),
    "AssemblyGenerator": ("manaenv.pipeline.factories.assembly_factory", "AssemblyOrder"),
    "AssemblyOrders": ("manaenv.pipeline.factories.assembly_factory", "AssemblyOrders"),
    # The public runtime exposes one PolicyPlanner evaluation order section.
    "Rollout": ("manaenv.pipeline.factories.rollout_factory", "RolloutOrders"),
    "RobotGenerator": ("manaenv.pipeline.factories.robot_factory", "RobotGenerator"),
    "RobotOrders": ("manaenv.pipeline.factories.robot_factory", "RobotOrders"),
}

factory_module_path, discover_factory = make_convention_domain(
    FACTORY_DOMAIN,
    base_package=FACTORY_PACKAGE,
    base_dir=Path(__file__).resolve().parent,
    suffix="Factory",
    family_dir_suffix=None,
)


def discover_factory_orders(symbol: str) -> ComponentSpec | None:
    """Resolve typed order CFGs owned by factory modules."""
    spec = _ORDER_SPECS.get(symbol)
    if spec is None:
        return None
    module_path, class_name = spec
    return ComponentSpec(
        module_path=module_path,
        class_name=class_name,
        symbol=symbol,
    )


register_provider(FACTORY_ORDER_DOMAIN, discover_factory_orders)
