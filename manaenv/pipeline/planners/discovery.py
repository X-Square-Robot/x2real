"""Planner discovery for the global lazy registry.

Pilot consumer of :func:`make_convention_domain`. New domains
(component, term, ...) should mirror this declaration: pin the
domain constants, call the helper, optionally re-export
:func:`camel_to_snake` so stability tests can import it through the
domain's discovery module.
"""

from __future__ import annotations

from pathlib import Path

from manaenv.pipeline.registry.convention_provider import camel_to_snake, make_convention_domain

__all__ = [
    "PLANNER_DOMAIN",
    "PLANNER_PACKAGE",
    "camel_to_snake",
    "discover_planner",
    "planner_module_path",
]

PLANNER_DOMAIN: str = "planner"
PLANNER_PACKAGE: str = "manaenv.pipeline.planners"

planner_module_path, discover_planner = make_convention_domain(
    PLANNER_DOMAIN,
    base_package=PLANNER_PACKAGE,
    base_dir=Path(__file__).resolve().parent,
    suffix="Planner",
    family_dir_suffix="_planner",
)
