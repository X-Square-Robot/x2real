"""Button sequence scene template."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.scene.components.asset_based_component import _coerce_scale_triple
from manaenv.utils.debug_print import manaprint
from manaenv.utils.runtime_type_spec import prepare_runtime_asset_class
from manaenv.utils.semantic_tags import apply_semantic_tags


@dataclass
class ButtonSequenceSceneCFG:
    """Declarative payload for deterministic button sequence assets."""

    class_type: type[ButtonSequenceScene] | None = None
    asset_pool: list[str] = field(default_factory=list)
    select_count: int = 0
    shuffle_assets: bool = False
    semantic_tags: list[Any] = field(default_factory=list)
    initial_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    initial_orientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    slots: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = ButtonSequenceScene


class ButtonSequenceScene:
    """Generate a deterministic sequence of touchable button assets."""

    def __init__(self, cfg: ButtonSequenceSceneCFG):
        self.cfg = cfg

    def generate(self, *, asset_registry: Any, rng: Any) -> list[Any]:
        asset_pool = [str(name) for name in self.cfg.asset_pool or [] if str(name)]
        slots = list(self.cfg.slots or [])
        if not asset_pool or not slots:
            manaprint("WARNING: ButtonSequenceScene requires non-empty asset_pool and slots")
            return []

        select_count = int(self.cfg.select_count or min(len(asset_pool), len(slots)))
        select_count = max(0, min(select_count, len(asset_pool), len(slots)))
        if select_count == 0:
            return []

        selected_assets = (
            rng.sample(asset_pool, k=select_count) if select_count < len(asset_pool) else asset_pool[:select_count]
        )
        if self.cfg.shuffle_assets:
            rng.shuffle(selected_assets)

        return [
            asset
            for asset_name, slot in zip(selected_assets, slots[:select_count])
            if (asset := self._instantiate_asset(asset_registry, asset_name, slot)) is not None
        ]

    def _instantiate_asset(self, asset_registry: Any, asset_name: str, slot: Any) -> Any:
        from isaaclab_arena.utils.pose import Pose

        try:
            slot_data = slot if isinstance(slot, dict) else {"position": slot}
            position = list(slot_data.get("position", self.cfg.initial_position))
            orientation = list(slot_data.get("orientation", self.cfg.initial_orientation))
            asset_class = asset_registry.get_asset_by_name(asset_name)
            runtime_plan = prepare_runtime_asset_class(asset_class, self.cfg.semantic_tags)
            asset_instance = runtime_plan.asset_class(**runtime_plan.init_kwargs)
            asset_instance.set_initial_pose(Pose(position_xyz=tuple(position), rotation_xyzw=tuple(orientation)))

            cfg_scale = _coerce_scale_triple(self.cfg.scale)
            if cfg_scale != (1.0, 1.0, 1.0) and hasattr(asset_instance, "scale"):
                asset_instance.scale = cfg_scale
            if hasattr(asset_instance, "_init_object_cfg"):
                asset_instance.object_cfg = asset_instance._init_object_cfg()
            apply_semantic_tags(asset_instance, self.cfg.semantic_tags)
            return asset_instance
        except Exception as e:
            manaprint(f"ERROR: Failed to instantiate ButtonSequenceScene asset '{asset_name}': {e}")
            return None
