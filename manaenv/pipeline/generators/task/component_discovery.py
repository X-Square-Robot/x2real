"""Task component discovery for the global lazy registry.

Task component YAML sections resolve by CFG name convention:

* ``termination_component`` -> ``TerminationComponent``
* ``domain_randomization_component`` -> ``DomainRandomizationComponent``

The lazy registry then imports
``manaenv.pipeline.generators.task.components.<snake>_component`` and resolves
the paired ``<symbol>CFG`` class on demand.
"""

from __future__ import annotations

from pathlib import Path

from manaenv.pipeline.registry.convention_provider import make_convention_domain
from manaenv.pipeline.registry.global_provider import global_registry

__all__ = [
    "TASK_COMPONENT_DOMAIN",
    "TASK_COMPONENT_PACKAGE",
    "discover_task_component",
    "is_task_component_section_key",
    "resolve_task_component_cfg",
    "task_component_module_path",
]

TASK_COMPONENT_DOMAIN: str = "task_component"
TASK_COMPONENT_PACKAGE: str = "manaenv.pipeline.generators.task.components"

task_component_module_path, discover_task_component = make_convention_domain(
    TASK_COMPONENT_DOMAIN,
    base_package=TASK_COMPONENT_PACKAGE,
    base_dir=Path(__file__).resolve().parent / "components",
    suffix="Component",
    family_dir_suffix=None,
)


def component_symbol_from_section_key(key: str) -> str:
    if key.endswith("CFG"):
        return key[: -len("CFG")]
    if "_" in key:
        return "".join(part.capitalize() for part in key.split("_"))
    if key.endswith("Component"):
        return key
    return key.capitalize() + "Component" if key else ""


def resolve_task_component_cfg(key: str) -> type | None:
    from manaenv.pipeline.generators.task.component import TaskComponentCFG

    symbol = component_symbol_from_section_key(key)
    if not symbol:
        return None
    cfg_class = global_registry.get_cfg(symbol)
    if cfg_class is not None and issubclass(cfg_class, TaskComponentCFG):
        return cfg_class
    if not symbol.endswith("Component"):
        cfg_class = global_registry.get_cfg(symbol + "Component")
        if cfg_class is not None and issubclass(cfg_class, TaskComponentCFG):
            return cfg_class
    return None


def is_task_component_section_key(key: str) -> bool:
    return resolve_task_component_cfg(key) is not None
