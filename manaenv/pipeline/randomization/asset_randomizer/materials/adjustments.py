from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MaterialLocalAdjustment:
    adjustment_id: str
    tags: tuple[str, ...] = ()
    family: str | None = None
    channels: Mapping[str, Any] = field(default_factory=dict, compare=False)
    identity_preserving: bool = True
    slotwise_safe: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class MaterialLocalAdjustmentLibrary:
    library_id: str
    adjustments: tuple[MaterialLocalAdjustment, ...] = ()

    def get(self, adjustment_id: str) -> MaterialLocalAdjustment | None:
        wanted = str(adjustment_id)
        for adjustment in self.adjustments:
            if adjustment.adjustment_id == wanted:
                return adjustment
        return None

    def filter_by_ids(self, adjustment_ids: tuple[str, ...]) -> tuple[MaterialLocalAdjustment, ...]:
        if not adjustment_ids:
            return self.adjustments
        wanted = set(adjustment_ids)
        return tuple(adjustment for adjustment in self.adjustments if adjustment.adjustment_id in wanted)

    def filter_by_tags_any(self, tags: tuple[str, ...]) -> tuple[MaterialLocalAdjustment, ...]:
        if not tags:
            return self.adjustments
        wanted = set(tags)
        return tuple(adjustment for adjustment in self.adjustments if wanted.intersection(adjustment.tags))

    def filter_by_tags_all(self, tags: tuple[str, ...]) -> tuple[MaterialLocalAdjustment, ...]:
        if not tags:
            return self.adjustments
        wanted = set(tags)
        return tuple(adjustment for adjustment in self.adjustments if wanted.issubset(set(adjustment.tags)))
