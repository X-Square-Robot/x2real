from __future__ import annotations

from dataclasses import dataclass

from .library import MaterialLibrary


@dataclass(frozen=True)
class MaterialRegistry:
    libraries: dict[str, MaterialLibrary]

    def get(self, library_id: str) -> MaterialLibrary | None:
        return self.libraries.get(library_id)
