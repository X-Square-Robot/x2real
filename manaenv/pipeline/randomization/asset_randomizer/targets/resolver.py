from __future__ import annotations

from ..backends.base import SceneBackend
from ..model import AssetHandle, TargetQuery


def resolve_targets(backend: SceneBackend, query: TargetQuery) -> tuple[AssetHandle, ...]:
    return tuple(backend.resolve_targets(query))
