"""Task-local fixtures for ``move_fruit_to_basket``.

Tables use native/default USD loading (same route as woodshelf). Visual surface
anchors live in the scene YAML and must not mutate USD physics here.
"""

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from manaenv.assets.resolver import resolve_runtime_asset_reference


@register_asset
class MoveFruitToBasketGreenBooth(LibraryBackground):
    name = "move_fruit_to_basket_green_booth"
    tags = ["background", "move_fruit_to_basket"]
    usd_path = resolve_runtime_asset_reference("assets/art_assets/scenes/green_booth/green_booth_mesh_none_3x3.usd")
    initial_pose = Pose.identity()
    object_min_z = -0.2


@register_asset
class MoveFruitToBasketTable(LibraryObject):
    name = "move_fruit_to_basket_table"
    tags = ["object", "furniture", "table", "move_fruit_to_basket"]
    usd_path = resolve_runtime_asset_reference("assets/art_assets/models/props/table/table_011.usd")
    object_type = ObjectType.RIGID

    def _generate_rigid_cfg(self):
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

        return self._add_initial_pose_to_cfg(
            RigidObjectCfg(
                prim_path=self.prim_path,
                spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale),
            )
        )


def register_assets() -> None:
    """Stable preload hook used by the scene factory."""


__all__ = ["MoveFruitToBasketGreenBooth", "MoveFruitToBasketTable", "register_assets"]
