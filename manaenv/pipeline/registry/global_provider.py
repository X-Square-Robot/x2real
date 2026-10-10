"""Global provider dispatcher for the manaenv lazy registry.

A single :class:`GlobalProvider` instance plays the role of the
:class:`LazyTypeRegistry` ``provider`` callable: each registered
domain (planner, scene, task, ...) plugs its own discovery function
in here, and lookups dispatch to whichever domain claims the symbol.

There is exactly one global registry / provider per process, exposed
through the module-level :data:`global_provider` and
:data:`global_registry` symbols. Domain modules import the
:func:`register_provider` helper at import time so callers can resolve
symbols without knowing which domain owns them.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from .lazy import ComponentSpec, LazyTypeRegistry

ProviderFn = Callable[[str], ComponentSpec | None]


class GlobalProvider:
    """Route discovery requests to per-domain provider callables.

    Domains register themselves by name (e.g. ``"planner"``); on
    lookup the provider iterates registered domains in registration
    order and returns the first non-``None`` :class:`ComponentSpec`.

    The order is preserved so callers can give precedence to specific
    domains by registering them first.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderFn] = {}
        self._domain_modules: dict[str, str] = {}

    def register(
        self,
        domain: str,
        provider: ProviderFn,
        *,
        ensure_module: str | None = None,
    ) -> None:
        """Register *provider* for *domain* (idempotent on the symbol id).

        ``ensure_module``: optional dotted module path that the
        dispatcher will import on first lookup, so domain providers
        can stay registration-side-effect-driven without forcing
        eager imports at startup.
        """
        if domain in self._providers and self._providers[domain] is provider:
            return
        self._providers[domain] = provider
        if ensure_module is not None:
            self._domain_modules[domain] = ensure_module

    def unregister(self, domain: str) -> None:
        self._providers.pop(domain, None)
        self._domain_modules.pop(domain, None)

    def domains(self) -> list[str]:
        return list(self._providers)

    def __call__(self, symbol: str) -> ComponentSpec | None:
        self._ensure_domain_modules_loaded()
        for provider in self._providers.values():
            spec = provider(symbol)
            if spec is not None:
                return spec
        return None

    def lookup_with_domain(self, symbol: str) -> tuple[str, ComponentSpec] | None:
        """Return ``(domain, spec)`` for *symbol*, or ``None`` if unknown."""
        self._ensure_domain_modules_loaded()
        for domain, provider in self._providers.items():
            spec = provider(symbol)
            if spec is not None:
                return domain, spec
        return None

    def _ensure_domain_modules_loaded(self) -> None:
        if not self._domain_modules:
            return
        pending = list(self._domain_modules.items())
        for domain, module_path in pending:
            importlib.import_module(module_path)
            self._domain_modules.pop(domain, None)


global_provider: GlobalProvider = GlobalProvider()
"""Process-wide :class:`GlobalProvider` singleton."""


global_registry: LazyTypeRegistry = LazyTypeRegistry(provider=global_provider)
"""Process-wide :class:`LazyTypeRegistry` backed by :data:`global_provider`."""


def register_provider(
    domain: str,
    provider: ProviderFn,
    *,
    ensure_module: str | None = None,
) -> None:
    """Convenience wrapper around :meth:`GlobalProvider.register`."""
    global_provider.register(domain, provider, ensure_module=ensure_module)


def discover(symbol: str) -> ComponentSpec | None:
    """Resolve *symbol* through the global provider chain."""
    return global_provider(symbol)
