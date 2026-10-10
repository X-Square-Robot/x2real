from __future__ import annotations

from typing import Protocol

from ..materials.library import RandomizationLibraryInput
from ..model import AssetHandle, MaterialSlot, PolicySample
from ..rng import RandomStream


class MaterialPolicy(Protocol):
    @property
    def policy_id(self) -> str: ...  # noqa: E704 - Black protocol stub style

    def sample(
        self,
        *,
        asset: AssetHandle,
        slots: tuple[MaterialSlot, ...],
        rng: RandomStream,
        library: RandomizationLibraryInput | None = None,
    ) -> PolicySample:
        pass
