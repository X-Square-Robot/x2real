"""Assets for the put_bottle_on_woodshelf task.

Loaded declaratively via scene YAML ``definition.asset_modules``. Import alone
is enough; the
``register_assets`` entrypoint exists for the shared asset_module preload path.
"""

from __future__ import annotations

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from manaenv.assets.resolver import resolve_runtime_asset_reference


@register_asset
class PutBottleOnWoodshelfGreenBooth(LibraryBackground):
    name = "put_bottle_on_woodshelf_green_booth"
    tags = ["background", "put_bottle_on_woodshelf"]
    usd_path = resolve_runtime_asset_reference("assets/art_assets/scenes/green_booth/green_booth_mesh_none_3x3.usd")
    initial_pose = Pose.identity()
    object_min_z = -0.2


@register_asset
class PutBottleOnWoodshelfTable(LibraryObject):
    name = "put_bottle_on_woodshelf_table"
    tags = ["object", "furniture", "table", "put_bottle_on_woodshelf"]
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
class PutBottleOnWoodshelfShelf(LibraryObject):
    name = "put_bottle_on_woodshelf_shelf"
    tags = ["object", "furniture", "shelf", "put_bottle_on_woodshelf"]
    usd_path = resolve_runtime_asset_reference("assets/art_assets/models/props/shelf/shelf_c.usd")
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
class PutBottleOnWoodshelfBottle(LibraryObject):
    name = "put_bottle_on_woodshelf_bottle"
    tags = ["object", "pickable", "bottle", "put_bottle_on_woodshelf"]
    usd_path = resolve_runtime_asset_reference("assets/art_assets/models/objects/bottle/bottle_nongfuspring.usd")
    object_type = ObjectType.RIGID

    def _generate_rigid_cfg(self):
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
            ),
        )
        return self._add_initial_pose_to_cfg(object_cfg)


@register_asset
class PutBottleOnWoodshelfBottleS(LibraryObject):
    name = "put_bottle_on_woodshelf_bottle_s"
    tags = ["object", "pickable", "bottle", "put_bottle_on_woodshelf"]
    usd_path = resolve_runtime_asset_reference("assets/art_assets/models/objects/bottle/bottle_nongfuspring_s.usd")
    object_type = ObjectType.RIGID

    def _generate_rigid_cfg(self):
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
            ),
        )
        return self._add_initial_pose_to_cfg(object_cfg)


def register_assets() -> None:
    """Scene YAML ``asset_modules`` / ``asset_module`` preload entrypoint.

    Class decorators already register on import; this exists so the shared
    preload path can call a stable function when present.
    """


__all__ = [
    "PutBottleOnWoodshelfBottle",
    "PutBottleOnWoodshelfBottleS",
    "PutBottleOnWoodshelfGreenBooth",
    "PutBottleOnWoodshelfShelf",
    "PutBottleOnWoodshelfTable",
    "register_assets",
]
