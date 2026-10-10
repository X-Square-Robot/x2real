from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from manaenv.utils.priority import sort_by_priority


class BaseGenerator(ABC):
    """Abstract base for all generators.

    ``generate`` is intentionally broad because each domain may need a
    different public generation interface. Generators that want the
    standard compile/instancing/finalize template should inherit
    :class:`ManaGenerator`.
    """

    def __init__(self, cfg: Any = None, **kwargs):
        self.cfg = cfg
        self.components: list[Any] = []

    @abstractmethod
    def generate(self, **kwargs) -> Any:
        pass

    def add_component(self, component: Any) -> None:
        self.components.append(component)

    def remove_component(self, component: Any) -> None:
        if component in self.components:
            self.components.remove(component)

    def sorted_components(self) -> list[Any]:
        """Return components sorted by their CFG priority (lower first)."""
        return sort_by_priority(self.components, key=lambda c: getattr(c.cfg, "priority", 2))

    @staticmethod
    def sort_components(cfgs: list[Any]) -> list[Any]:
        """Sort component CFGs by priority (lower runs first)."""
        return sort_by_priority(cfgs)


@dataclass
class GenerationCompilation:
    """Marker base for artifacts returned by ``ManaGenerator.compile``."""


class ManaGenerator(BaseGenerator):
    """Generator variant with the standard three-stage lifecycle."""

    def generate(self, **kwargs) -> Any:
        """Run compile + instancing.

        ``finalize`` is a late-binding hook and must be called
        explicitly by subclasses or factories when the needed context is
        available.
        """
        compiled = self.compile(**kwargs)
        return self.instancing(compiled, **kwargs)

    def compile(self, **kwargs) -> Any:
        """Normalize generation inputs into a domain-specific compilation.

        Args:
            **kwargs: Generation-time context supplied by the caller.
                Typical examples are raw yaml definitions, component
                cfgs, variant indices, constraints, seeds, or runtime
                overrides. Implementations should keep this stage free
                of late-bound scene/robot objects unless that context is
                already available by design.
        """
        return kwargs or self.cfg

    def instancing(self, compiled: Any, **kwargs) -> Any:
        """Create runtime objects from a compiled generation artifact.

        Args:
            compiled: The artifact returned by :meth:`compile`.
            **kwargs: The same generation-time context passed to
                :meth:`compile`, plus any caller-specific values needed
                to materialize instances.
        """
        del kwargs
        return compiled

    def finalize(self, instance: Any, **kwargs) -> Any:
        """Apply late-bound context and return the final output.

        Args:
            instance: The artifact returned by :meth:`instancing`.
            **kwargs: Late-binding context supplied when available.
                Common examples are ``scene``, ``robot``, resolved asset
                bindings, runtime cfg patches, or external clients.

        ``finalize`` is not called by :meth:`generate` by default. A
        factory or subclass should call it explicitly when the required
        context is ready.
        """
        del kwargs
        return instance
