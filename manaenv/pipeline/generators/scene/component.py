from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from manaenv.pipeline.component import BaseComponent, BaseComponentCFG

if TYPE_CHECKING:
    from pxr import Usd


class SceneComponent(BaseComponent):
    """Scene-specific component with post-spawn hook."""

    def generate_for_variant(self, cfg: SceneComponentCFG | None = None, *, variant_index: int = 0) -> Any:
        """Generate this component for one scene variant.

        Components that do not use variant-specific selection inherit the
        standard behavior and delegate to ``generate(cfg)``.
        """

        del variant_index
        return self.generate(cfg)

    def post_spawn(self, stage: Usd.Stage, num_envs: int, asset_names: dict[str, str]) -> None:
        """Hook called after gym.make() when the USD stage is fully composed.

        Override in subclasses that need to modify the live stage
        (e.g. deactivating embedded lights, adjusting physics params).

        Args:
            stage: The active USD stage.
            num_envs: Number of environment instances.
            asset_names: Mapping of asset name -> component type name.
        """


@dataclass
class SceneComponentCFG(BaseComponentCFG):

    class_type: type[SceneComponent] = None
    priority: int = 1

    # Scene DSL v1 public fields. CFG names intentionally match YAML names.
    id: str = ""
    asset: str = ""
    asset_tag: str = ""
    generator: str = ""

    # Initial transform
    initial_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    initial_orientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
