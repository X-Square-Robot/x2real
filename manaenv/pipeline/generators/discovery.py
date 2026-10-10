"""Generator discovery for the global lazy registry."""

from __future__ import annotations

from pathlib import Path

from manaenv.pipeline.registry.convention_provider import camel_to_snake, make_convention_domain

__all__ = [
    "GENERATOR_DOMAIN",
    "GENERATOR_PACKAGE",
    "camel_to_snake",
    "discover_generator",
    "generator_module_path",
]

GENERATOR_DOMAIN: str = "generator"
GENERATOR_PACKAGE: str = "manaenv.pipeline.generators"

generator_module_path, discover_generator = make_convention_domain(
    GENERATOR_DOMAIN,
    base_package=GENERATOR_PACKAGE,
    base_dir=Path(__file__).resolve().parent,
    suffix="Generator",
    family_dir_suffix=None,
)
