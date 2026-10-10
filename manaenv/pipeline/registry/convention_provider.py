"""Convention-based lazy discovery for the global registry.

A deliberately small toolbox. Domains plug in by composing three
building blocks; nothing here is class-shaped because the
:class:`~manaenv.pipeline.registry.LazyTypeRegistry` already caches
resolved :class:`ComponentSpec` instances per symbol.

* :func:`camel_to_snake` -- ``CamelCase`` <-> ``snake_case`` (acronyms
  preserved, e.g. ``VRTeleopPlanner`` -> ``vr_teleop_planner``).
* :func:`expand_convention_paths` -- yields the ordered candidate
  module paths for a symbol under a base package, optionally routing
  through ``*_<family>/`` sub-package directories discovered on disk.
* :func:`resolve_first_match` -- imports candidates one by one and
  returns the first :class:`ComponentSpec` whose module declares the
  requested symbol.

For the common case ("CamelCase symbol with suffix routes into
``*_<family>/`` sub-packages") :func:`make_convention_domain` wires the
three primitives together and registers the provider; a domain only
needs to declare its constants in a single ``discovery.py`` (see the
planner pilot at :mod:`manaenv.pipeline.planners.discovery`).
"""

from __future__ import annotations

import functools
import importlib
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from .global_provider import register_provider
from .lazy import ComponentSpec

__all__ = [
    "camel_to_snake",
    "expand_convention_paths",
    "make_convention_domain",
    "resolve_first_match",
]

ModulePathFn = Callable[[str], str | None]
ProviderFn = Callable[[str], ComponentSpec | None]

# Splits CamelCase boundaries while keeping ALL-CAPS acronyms intact.
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def camel_to_snake(name: str) -> str:
    """Convert ``CamelCase`` to ``snake_case`` while preserving acronyms."""
    return _CAMEL_BOUNDARY_RE.sub("_", name).lower()


def expand_convention_paths(
    symbol: str,
    *,
    base_package: str,
    base_dir: Path,
    suffix: str,
    family_dir_suffix: str | None = None,
) -> Iterator[str]:
    """Yield candidate dotted module paths for *symbol* under *base_package*.

    * *symbol* must end with *suffix* (and not equal it); otherwise no
      paths are yielded.
    * The snake_case stem of *symbol* is the expected file stem.
    * When *family_dir_suffix* is provided, the on-disk sub-packages
      whose names end with that suffix are scanned; the family that
      actually contains a ``<stem>.py`` file wins and is yielded first.
      Family routing is therefore driven by physical file location, not
      by stem suffix matching: a file may freely live under any family
      sub-package that owns its class hierarchy (e.g.
      ``SkillTrajectoryPlanner`` lives under
      ``motion_planner/skill_trajectory_planner.py``).
    * The top-level ``base_package.<stem>`` path is always yielded
      last as the fallback for symbols that no family claims.

    Paths are yielded lazily so callers can stop at the first
    successful import.
    """
    if not symbol.endswith(suffix) or symbol == suffix:
        return
    stem = camel_to_snake(symbol)
    if family_dir_suffix is not None:
        family = _scan_family_files(base_dir, family_dir_suffix).get(stem)
        if family is not None:
            yield f"{base_package}.{family}.{stem}"
    yield f"{base_package}.{stem}"


def resolve_first_match(symbol: str, candidates: Iterable[str]) -> ComponentSpec | None:
    """Return the first candidate whose module imports & declares *symbol*.

    When the runtime module cannot import (e.g. missing IsaacSim deps in a
    preflight environment), the lightweight ``<module>_cfg`` sidecar is probed
    instead: declaring ``<symbol>CFG`` there is equally strong evidence that
    the candidate path owns the symbol, and keeps YAML reflection available
    without the heavy runtime import.

    Returns ``None`` when no candidate matches so
    :class:`~manaenv.pipeline.registry.global_provider.GlobalProvider`
    can fall through to other domain providers.
    """
    for module_path in candidates:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # Covers ModuleNotFoundError plus native-lib failures (e.g. a heavy
            # runtime module that transitively loads IsaacSim's ``libcarb.so``).
            # Fall through to the lightweight ``<module>_cfg`` sidecar probe.
            module = None
        if module is not None and symbol in vars(module):
            return ComponentSpec(module_path=module_path, class_name=symbol, symbol=symbol)
        try:
            cfg_module = importlib.import_module(f"{module_path}_cfg")
        except ImportError:
            continue
        if f"{symbol}CFG" in vars(cfg_module):
            return ComponentSpec(module_path=module_path, class_name=symbol, symbol=symbol)
    return None


def make_convention_domain(
    domain: str,
    *,
    base_package: str,
    base_dir: Path,
    suffix: str,
    family_dir_suffix: str | None = None,
) -> tuple[ModulePathFn, ProviderFn]:
    """Build & register a convention-based discovery provider for *domain*.

    Returns ``(module_path_for, discover)``:

    * ``module_path_for(symbol)`` -- first convention-derived dotted
      module path (or ``None``); used by stability tests and translator
      helpers that need the path *the symbol should live at*.
    * ``discover(symbol)`` -- the :data:`ProviderFn` registered with
      the global dispatcher, returning a :class:`ComponentSpec` only
      when the candidate path actually imports.

    Domain ``discovery.py`` files collapse to a constant block plus a
    single call (see :mod:`manaenv.pipeline.planners.discovery`).
    """

    def _candidates(symbol: str) -> Iterator[str]:
        return expand_convention_paths(
            symbol,
            base_package=base_package,
            base_dir=base_dir,
            suffix=suffix,
            family_dir_suffix=family_dir_suffix,
        )

    def module_path_for(symbol: str) -> str | None:
        return next(_candidates(symbol), None)

    def discover(symbol: str) -> ComponentSpec | None:
        return resolve_first_match(symbol, _candidates(symbol))

    register_provider(domain, discover)
    return module_path_for, discover


@functools.cache
def _scan_family_stems(base_dir: Path, suffix: str) -> tuple[str, ...]:
    """Disk-scan family sub-package names under *base_dir*, longest-first.

    Cached on ``(base_dir, suffix)`` because package layouts are
    frozen at process start; tests that mutate the layout can call
    ``_scan_family_stems.cache_clear()`` (note: also clear
    :func:`_scan_family_files`).
    """
    return tuple(
        sorted(
            (
                d.name
                for d in base_dir.iterdir()
                if d.is_dir() and d.name.endswith(suffix) and (d / "__init__.py").exists()
            ),
            key=len,
            reverse=True,
        )
    )


@functools.cache
def _scan_family_files(base_dir: Path, suffix: str) -> dict:
    """Build ``{file_stem: family_dir_name}`` for files under family sub-packages.

    A family sub-package is any directory whose name ends with *suffix*
    and contains an ``__init__.py``. Each family directory is scanned
    for top-level ``*.py`` files (excluding dunder/private files). The
    first family found to contain a given stem wins (families are
    iterated in alphabetical order to keep results deterministic when a
    stem appears in multiple families, which would itself be a bug).

    Cached on ``(base_dir, suffix)``. Tests that mutate the layout
    must clear both this and :func:`_scan_family_stems`.
    """
    mapping: dict[str, str] = {}
    families = sorted(
        (d for d in base_dir.iterdir() if d.is_dir() and d.name.endswith(suffix) and (d / "__init__.py").exists()),
        key=lambda d: d.name,
    )
    for family_dir in families:
        for py in family_dir.glob("*.py"):
            stem = py.stem
            if stem.startswith("_"):
                continue
            mapping.setdefault(stem, family_dir.name)
    return mapping
