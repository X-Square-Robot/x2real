from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..materials.library import RandomizationLibraryInput
from ..model import AssetHandle, MaterialSlot, PolicySample
from ..rng import RandomStream
from .base import MaterialPolicy


@dataclass(frozen=True)
class CompositePolicy:
    policy_id: str
    policies: tuple[MaterialPolicy, ...]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CompositePolicy:
        from . import create_policy

        items = config.get("policies") or ()
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            items = ()
        return cls(
            policy_id=str(config.get("id") or config.get("name") or "composite"),
            policies=tuple(create_policy(item) for item in items if isinstance(item, Mapping)),
        )

    def sample(
        self,
        *,
        asset: AssetHandle,
        slots: tuple[MaterialSlot, ...],
        rng: RandomStream,
        library: RandomizationLibraryInput | None = None,
    ) -> PolicySample:
        edits = []
        diagnostics = []
        for index, policy in enumerate(self.policies):
            sample = policy.sample(
                asset=asset,
                slots=slots,
                rng=rng.spawn(self.policy_id, index),
                library=library,
            )
            edits.extend(sample.edits)
            diagnostics.extend(sample.diagnostics)
        return PolicySample(edits=tuple(edits), diagnostics=tuple(diagnostics))
