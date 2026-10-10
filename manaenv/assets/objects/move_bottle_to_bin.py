"""Task-local fixtures for ``move_bottle_to_bin``.

Loaded declaratively via scene YAML ``definition.asset_modules``. Table and bin
use native/default USD loading (same route as fruit/woodshelf). Visual surface
anchors live in the scene YAML and must not mutate USD physics here.
"""

from __future__ import annotations

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from manaenv.assets.resolver import resolve_runtime_asset_reference


@register_asset
class MoveBottleToBinGreenBooth(LibraryBackground):
    name = "move_bottle_to_bin_green_booth"
    tags = ["background", "move_bottle_to_bin"]
    usd_path = resolve_runtime_asset_reference("assets/art_assets/scenes/green_booth/green_booth_mesh_none_3x3.usd")
    initial_pose = Pose.identity()
    object_min_z = -0.2


@register_asset
class MoveBottleToBinTable(LibraryObject):
    name = "move_bottle_to_bin_table"
    tags = ["object", "furniture", "table", "move_bottle_to_bin"]
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


@register_asset
class MoveBottleToBinBin(LibraryObject):
    name = "move_bottle_to_bin_bin"
    tags = ["object", "furniture", "bin", "container", "move_bottle_to_bin"]
    # Trashcan-sourced dynamic bin geometry (trash_can_a).
    usd_path = resolve_runtime_asset_reference("assets/art_assets/models/objects/trash_can/trash_can_a.usd")
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


__all__ = [
    "MoveBottleToBinBin",
    "MoveBottleToBinGreenBooth",
    "MoveBottleToBinTable",
    "register_assets",
]
