from __future__ import annotations

from typing import Protocol

from ..model import ApplyReport, AssetHandle, MaterialEdit, MaterialSlot, TargetQuery


class SceneBackend(Protocol):
    def resolve_targets(self, query: TargetQuery) -> tuple[AssetHandle, ...]:
        pass

    def inspect_material_slots(self, asset: AssetHandle) -> tuple[MaterialSlot, ...]:
        pass

    def apply_material_edits(self, edits: tuple[MaterialEdit, ...]) -> ApplyReport:
        pass
