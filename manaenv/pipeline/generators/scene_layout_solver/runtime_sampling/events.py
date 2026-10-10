from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedRuntimeEvent:
    name: str
    params: dict[str, Any]
    objects: tuple[str, ...] = ()
    reference: str | None = None


@dataclass(frozen=True)
class CompiledSamplingDomain:
    domains: tuple[Any, ...]
    parent: str | None = None
    cases: tuple[dict[str, Any], ...] = ()
    meta: dict[str, Any] | None = None
