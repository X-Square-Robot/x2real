"""Background component for static scene elements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..utils.lighting_utils import collapse_cloned_dome_lights, silence_tabletop_rect_lights
from .asset_based_component import AssetBasedComponent, AssetBasedComponentCFG

if TYPE_CHECKING:
    from isaaclab_arena.assets.registries import AssetRegistry
    from pxr import Usd


class BackgroundComponent(AssetBasedComponent):
    """Component for managing static background elements."""

    def __init__(self, cfg: BackgroundComponentCFG | None = None, asset_registry: AssetRegistry = None):
        super().__init__(cfg, asset_registry)

    def post_spawn(self, stage: Usd.Stage, num_envs: int, asset_names: dict[str, str]) -> None:
        """Normalize background lighting: one global DomeLight, no desk RectLights.

        Benchmark backgrounds author their own DomeLight, so env cloning yields one
        dome per env. Domes are infinite lights that all illuminate the whole stage,
        so the clones must be reduced to a single shared light. Local
        ``tabletop_local_key`` RectLights are silenced by default so illumination
        is dome-only (opt back in with ``MANAENV_ENABLE_TABLETOP_RECT_LIGHT=1``).
        """
        del asset_names
        collapse_cloned_dome_lights(stage, num_envs=num_envs)
        silence_tabletop_rect_lights(stage)


@dataclass
class BackgroundComponentCFG(AssetBasedComponentCFG):
    """Configuration for background component."""

    class_type: type = BackgroundComponent
