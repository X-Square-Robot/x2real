from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..model import MaterialAssetRef
from .adjustments import MaterialLocalAdjustmentLibrary


@dataclass(frozen=True)
class MaterialAsset:
    material_id: str
    tags: tuple[str, ...] = ()
    shader_model: str | None = None
    usd_material_ref: str | None = None
    channels: Mapping[str, Any] = field(default_factory=dict, compare=False)
    compatibility: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def as_ref(self) -> MaterialAssetRef:
        return MaterialAssetRef(
            material_id=self.material_id,
            shader_model=self.shader_model,
            usd_material_ref=self.usd_material_ref,
            tags=self.tags,
            channels=self.channels,
            compatibility=self.compatibility,
        )


@dataclass(frozen=True)
class MaterialLibrary:
    library_id: str
    materials: tuple[MaterialAsset, ...] = ()

    def get(self, material_id: str) -> MaterialAsset | None:
        wanted = str(material_id)
        for material in self.materials:
            if material.material_id == wanted:
                return material
        return None

    def filter_by_ids(self, material_ids: tuple[str, ...]) -> tuple[MaterialAsset, ...]:
        if not material_ids:
            return self.materials
        wanted = set(material_ids)
        return tuple(material for material in self.materials if material.material_id in wanted)

    def filter_by_tags_any(self, tags: tuple[str, ...]) -> tuple[MaterialAsset, ...]:
        if not tags:
            return self.materials
        wanted = set(tags)
        return tuple(material for material in self.materials if wanted.intersection(material.tags))

    def filter_by_tags_all(self, tags: tuple[str, ...]) -> tuple[MaterialAsset, ...]:
        if not tags:
            return self.materials
        wanted = set(tags)
        return tuple(material for material in self.materials if wanted.issubset(set(material.tags)))


@dataclass(frozen=True)
class RandomizationLibraries:
    material_library: MaterialLibrary | None = None
    adjustment_library: MaterialLocalAdjustmentLibrary | None = None


RandomizationLibraryInput = RandomizationLibraries | MaterialLibrary | MaterialLocalAdjustmentLibrary


def material_library_from_input(
    library: RandomizationLibraryInput | None,
) -> MaterialLibrary | None:
    if library is None:
        return None
    if isinstance(library, RandomizationLibraries):
        return library.material_library
    if isinstance(library, MaterialLibrary):
        return library
    return None


def adjustment_library_from_input(
    library: RandomizationLibraryInput | None,
) -> MaterialLocalAdjustmentLibrary | None:
    if library is None:
        return None
    if isinstance(library, RandomizationLibraries):
        return library.adjustment_library
    if isinstance(library, MaterialLocalAdjustmentLibrary):
        return library
    return None
