"""Task generator discovery for the global lazy registry.

Mirrors :mod:`manaenv.pipeline.planners.discovery`: task generators
are looked up by their CamelCase ``*TaskGenerator`` symbol; the
convention-driven provider imports
``manaenv.pipeline.generators.task.<snake_case_stem>`` and resolves
the impl class plus its ``<symbol>CFG`` peer on demand. New
generators only need to drop a ``foo_task_generator.py`` file in
this package — no registration boilerplate, no ``__init__`` re-export.
"""

from __future__ import annotations

from pathlib import Path

from manaenv.pipeline.registry.convention_provider import camel_to_snake, make_convention_domain

__all__ = [
    "TASK_DOMAIN",
    "TASK_PACKAGE",
    "camel_to_snake",
    "discover_task",
    "task_module_path",
]

TASK_DOMAIN: str = "task"
TASK_PACKAGE: str = "manaenv.pipeline.generators.task"

task_module_path, discover_task = make_convention_domain(
    TASK_DOMAIN,
    base_package=TASK_PACKAGE,
    base_dir=Path(__file__).resolve().parent,
    suffix="TaskGenerator",
    family_dir_suffix=None,
)
