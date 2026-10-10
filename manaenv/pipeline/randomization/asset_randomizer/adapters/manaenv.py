from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..api import randomize_asset_appearance
from ..backends.usd import UsdSceneBackend
from ..config import load_spec_from_dict
from ..materials.library import RandomizationLibraryInput
from ..model import RandomizationContext, RandomizationReport


def randomize_manaenv_stage(
    *,
    stage: Any,
    config: Mapping[str, Any],
    context: RandomizationContext,
    library: RandomizationLibraryInput | None = None,
) -> RandomizationReport:
    spec = load_spec_from_dict(config)
    return randomize_asset_appearance(
        backend=UsdSceneBackend(stage),
        spec=spec,
        context=context,
        library=library,
    )
