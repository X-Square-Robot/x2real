"""Lazy type registry shared across the manaenv pipeline.

Direct port of the ``test_registry`` prototype (``lazy_registry.py``).
The registry stores one :class:`ComponentSpec` per symbol and derives
``impl`` / ``cfg`` views via :mod:`manaenv.pipeline.registry.resolver`.

The intent is that *every* domain (planner, generator, component, ...)
shares the same minimal abstraction; per-domain knowledge lives in the
*provider* callable, not in the registry itself.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from .resolver import resolve_obj, resolve_type

_NON_IDENTIFIER_CHARS = re.compile(r"[^0-9a-zA-Z]+")


@dataclass(slots=True)
class ModuleImportSpec:
    package: str
    module_name: str
    symbol: str | None = None

    @classmethod
    def from_symbol(cls, package: str, symbol: str) -> ModuleImportSpec:
        module_name = _NON_IDENTIFIER_CHARS.sub("_", symbol).strip("_").lower()
        return cls(package=package, module_name=module_name, symbol=symbol)

    @property
    def module_path(self) -> str:
        return f"{self.package}.{self.module_name}"

    def import_module(self) -> ModuleImportSpec | None:
        try:
            importlib.import_module(self.module_path)
        except ModuleNotFoundError as exc:
            if exc.name == self.module_path:
                return None
            raise
        return self

    def try_import(self) -> bool:
        return self.import_module() is not None


def type_path(klass: type) -> str:
    return f"{klass.__module__}.{klass.__qualname__}"


@dataclass(slots=True)
class ComponentSpec:
    module_path: str
    class_name: str
    symbol: str | None = None

    @property
    def impl_path(self) -> str:
        return f"{self.module_path}.{self.class_name}"

    @property
    def cfg_path(self) -> str:
        return f"{self.module_path}.{self.class_name}CFG"


class LazyTypeRegistry:
    """Store one component spec and derive cfg/impl views on demand."""

    def __init__(self, provider: Callable[[str], ComponentSpec | None] | None = None):
        self._provider = provider
        self._specs: dict[str, ComponentSpec] = {}

    def set_provider(self, provider: Callable[[str], ComponentSpec | None] | None) -> None:
        self._provider = provider

    def register(self, spec: ComponentSpec) -> ComponentSpec:
        if spec.symbol is None:
            raise ValueError("ComponentSpec.symbol is required")
        self._specs[spec.symbol] = spec
        return spec

    def _discover(self, name: str) -> ComponentSpec | None:
        if self._provider is None:
            return None
        discovered = self._provider(name)
        if discovered is None:
            return None
        return self.register(discovered)

    def get_spec(self, name: str) -> ComponentSpec | None:
        spec = self._specs.get(name)
        if spec is not None:
            return spec
        discovered = self._discover(name)
        if discovered is not None:
            return discovered

        lowered = name.lower()
        for symbol in self._specs:
            if symbol.lower() == lowered and symbol != name:
                raise KeyError(f"Unknown component '{name}'. Did you mean '{symbol}'?")
        return None

    def get_obj(self, name: str) -> object | None:
        spec = self.get_spec(name)
        if spec is None:
            return None
        return resolve_obj(spec.impl_path)

    def get_impl(self, name: str) -> type | None:
        resolved = self.get_obj(name)
        if resolved is None:
            return None
        if not isinstance(resolved, type):
            raise TypeError(f"Resolved symbol is not a type: {name}")
        return resolved

    def get_func(self, name: str) -> Callable | None:
        resolved = self.get_obj(name)
        if resolved is None:
            return None
        if not callable(resolved):
            raise TypeError(f"Resolved symbol is not callable: {name}")
        return resolved

    def get_cfg(self, name: str) -> type | None:
        spec = self.get_spec(name)
        if spec is None:
            return None
        # Prefer the lightweight ``<module>_cfg`` sidecar when present: it is
        # the canonical CFG home for components whose runtime module needs
        # heavy deps (IsaacSim/pxr) that preflight environments lack.
        sidecar_path = f"{spec.module_path}_cfg"
        try:
            sidecar = importlib.import_module(sidecar_path)
        except ModuleNotFoundError as exc:
            if exc.name != sidecar_path:
                raise
            sidecar = None
        if sidecar is not None:
            cfg_class = getattr(sidecar, f"{spec.class_name}CFG", None)
            if isinstance(cfg_class, type):
                return cfg_class
        return resolve_type(spec.cfg_path)

    def role_of(self, instance: object) -> str:
        mro_paths = {type_path(klass) for klass in type(instance).__mro__}
        for spec in self._specs.values():
            if spec.impl_path in mro_paths:
                return spec.symbol or "none"

        class_type = getattr(instance, "class_type", None)
        if isinstance(class_type, str):
            resolved = resolve_type(class_type, type(instance).__module__)
            class_type = type_path(resolved)
        elif isinstance(class_type, type):
            class_type = type_path(class_type)

        if isinstance(class_type, str):
            for spec in self._specs.values():
                if spec.impl_path == class_type:
                    return spec.symbol or "none"

        return "none"

    def tree(self) -> dict[str, dict[str, str | None]]:
        return {
            spec.symbol
            or spec.class_name: {
                "module_path": spec.module_path,
                "class_name": spec.class_name,
                "symbol": spec.symbol,
            }
            for spec in self._specs.values()
        }
