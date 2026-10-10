"""Startup bootstrap for lazy discovery providers.

This module is intentionally outside ``registry.__init__`` lookup flow:
callers invoke :func:`bootstrap_discovery` during program startup, then
runtime registry lookups only search already-registered provider functions.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

DEFAULT_DISCOVERY_ROOTS: tuple[str, ...] = (
    "manaenv.pipeline",
    "manaenv.policy",
    "manaenv.actuators",
    "manaenv.utils",
)

_BOOTSTRAPPED_ROOTS: set[str] = set()
_IMPORTED_MODULES: set[str] = set()


def bootstrap_discovery(root_packages: tuple[str, ...] = DEFAULT_DISCOVERY_ROOTS) -> None:
    """Import discovery modules under *root_packages* exactly once per process."""
    for root_package in root_packages:
        if root_package in _BOOTSTRAPPED_ROOTS:
            continue
        package = importlib.import_module(root_package)
        for module_name in _iter_discovery_modules(package):
            if module_name in _IMPORTED_MODULES:
                continue
            importlib.import_module(module_name)
            _IMPORTED_MODULES.add(module_name)
        _BOOTSTRAPPED_ROOTS.add(root_package)


def _iter_discovery_modules(package: ModuleType) -> list[str]:
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return []

    module_names: set[str] = set()
    for package_path in package_paths:
        base_dir = Path(package_path)
        for path in _iter_discovery_files(base_dir):
            module_names.add(_module_name(package.__name__, base_dir, path))
    return sorted(module_names)


def _iter_discovery_files(base_dir: Path) -> list[Path]:
    candidates = list(base_dir.rglob("discovery.py")) + list(base_dir.rglob("*_discovery.py"))
    return [
        path
        for path in candidates
        if "__pycache__" not in path.parts
        and "integration_test" not in path.parts
        and "gpu_test" not in path.parts
        and not path.name.startswith("_")
    ]


def _module_name(root_package: str, base_dir: Path, path: Path) -> str:
    relative = path.relative_to(base_dir).with_suffix("")
    return ".".join((root_package, *relative.parts))
