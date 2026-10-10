"""Mana-owned object library — registered via Arena's ``LibraryObject`` pattern.

Objects resolve from the user-supplied local bundle and override automatic
discovery when a concrete affordance mixin is required
(e.g. ``KnobA(LibraryObject, Turnable)``).
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

from isaaclab_arena.affordances.turnable import Turnable
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset

from manaenv.assets.config import ASSET_ROOT_URI, join_asset_uri
from manaenv.assets.resolver import is_asset_reference, resolve_runtime_asset_reference

if TYPE_CHECKING:
    from isaaclab_arena.utils.pose import Pose

KNOB_A_USD = resolve_runtime_asset_reference(
    join_asset_uri(ASSET_ROOT_URI, "art_assets/models/objects/knob/knob_a.usd")
)
USB_A_BASE_USD = resolve_runtime_asset_reference(
    join_asset_uri(ASSET_ROOT_URI, "art_assets/models/objects/usb/usb_a_base.usd")
)
DRAWER_BA_USD = resolve_runtime_asset_reference(
    join_asset_uri(ASSET_ROOT_URI, "art_assets/models/objects/drawer/drawer_ba.usd")
)
BRICK_ROOT_URL = resolve_runtime_asset_reference(
    join_asset_uri(ASSET_ROOT_URI, "objects/brick")
)

_KNOB_TEXTURE = resolve_runtime_asset_reference(
    join_asset_uri(ASSET_ROOT_URI, "art_assets/models/objects/knob/textures/num_bc.jpg")
)


@register_asset
class UsbABase(LibraryObject):
    """USB dock rigid body with raised PhysX position solver iterations."""

    name = "usb_a_base"
    tags = ["object", "destination"]
    usd_path = USB_A_BASE_USD
    object_type = ObjectType.RIGID

    def _generate_rigid_cfg(self):
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
        from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

        cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
                rigid_props=PhysxRigidBodyPropertiesCfg(
                    solver_position_iteration_count=32,
                ),
            ),
        )
        return self._add_initial_pose_to_cfg(cfg)


class _Brick3cmBase(LibraryObject):
    """Three-centimeter stacking brick with increased contact solver iterations."""

    tags = ["toy|block", "tabletop_object", "floor_object", "inside_container_candidate", "graspable"]
    object_type = ObjectType.RIGID

    def _generate_rigid_cfg(self):
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
        from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

        cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
                rigid_props=PhysxRigidBodyPropertiesCfg(
                    solver_position_iteration_count=32,
                ),
            ),
        )
        return self._add_initial_pose_to_cfg(cfg)


@register_asset
class BrickBlue3cm(_Brick3cmBase):
    name = "brick_blue_3cm"
    usd_path = f"{BRICK_ROOT_URL}/brick_blue_3cm.usd"


@register_asset
class BrickGreen3cm(_Brick3cmBase):
    name = "brick_green_3cm"
    usd_path = f"{BRICK_ROOT_URL}/brick_green_3cm.usd"


@register_asset
class BrickOrange3cm(_Brick3cmBase):
    name = "brick_orange_3cm"
    usd_path = f"{BRICK_ROOT_URL}/brick_orange_3cm.usd"


@register_asset
class BrickRed3cm(_Brick3cmBase):
    name = "brick_red_3cm"
    usd_path = f"{BRICK_ROOT_URL}/brick_red_3cm.usd"


@register_asset
class KnobA(LibraryObject, Turnable):
    """Panel knob with a two-digit MDL display driven by joint travel."""

    name = "knob_a"
    tags = ["turnable", "object"]
    usd_path = KNOB_A_USD
    object_type = ObjectType.ARTICULATION

    turnable_joint_name = "RevoluteJoint"
    min_level_angle_deg = 0.0
    max_level_angle_deg = 594.0
    # The two-digit panel exposes 0..99. The USD keeps broader physical travel;
    # task startup applies the shared physical stop while preserving 6 deg/unit.
    num_levels = 100

    digit_shader_suffixes = ("/Looks/digit0/Shader", "/Looks/digit1/Shader")
    joint_angle_min_deg = 0.0
    joint_angle_max_deg = -720.0
    display_min = 0
    display_max = 99
    degrees_per_display_step = 6.0

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(
            prim_path=prim_path,
            initial_pose=initial_pose,
            turnable_joint_name=self.turnable_joint_name,
            min_level_angle_deg=self.min_level_angle_deg,
            max_level_angle_deg=self.max_level_angle_deg,
            num_levels=self.num_levels,
        )

    def joint_angle_to_display_value(self, joint_rad: float) -> int:
        joint_deg = math.degrees(float(joint_rad))
        lo = min(self.joint_angle_min_deg, self.joint_angle_max_deg)
        hi = max(self.joint_angle_min_deg, self.joint_angle_max_deg)
        joint_deg = max(lo, min(hi, joint_deg))
        if self.joint_angle_max_deg < 0.0:
            travel_deg = abs(joint_deg)
        else:
            travel_deg = abs(joint_deg - self.joint_angle_min_deg)
        raw = int(round(travel_deg / self.degrees_per_display_step))
        return max(self.display_min, min(self.display_max, raw))

    @classmethod
    def display_value_to_joint_angle(cls, display_value: int | float) -> float:
        """Panel reading -> joint rad (inverse of :meth:`joint_angle_to_display_value`)."""
        value = int(round(float(display_value)))
        travel_deg = float(value) * cls.degrees_per_display_step
        joint_deg = -travel_deg if cls.joint_angle_max_deg < 0.0 else travel_deg
        return math.radians(joint_deg)

    def display_value_to_digit_uv(self, display_value: int) -> tuple[float, float]:
        value = max(self.display_min, min(self.display_max, int(display_value)))
        return (value // 10) * 0.1, (value % 10) * 0.1

    # ------------------------------------------------------------------
    # Live display driving (USD digit-shader UV sync). Called each step by
    # the generic asset-display tick in the env runner via ``sync_display``.
    # ------------------------------------------------------------------

    def sync_display(self, gym_env: Any, *, env_idx: int | None = None) -> None:
        """Push the live joint angle(s) to the two-digit panel shader UVs."""
        from manaenv.utils.scene_object_utils import get_articulation_joint_value

        if not self.turnable_joint_name:
            return
        num_envs = int(getattr(gym_env, "num_envs", 1) or 1)
        env_indices = [int(env_idx)] if env_idx is not None else list(range(num_envs))
        for idx in env_indices:
            joint_rad = get_articulation_joint_value(gym_env, self.name, idx, self.turnable_joint_name)
            if joint_rad is None:
                continue
            self._apply_display_uv(idx, self.joint_angle_to_display_value(joint_rad))

    def _apply_display_uv(self, env_idx: int, display_value: int) -> None:
        import omni.usd

        cache = getattr(self, "_display_cache", None)
        if cache is None:
            cache = {"textures_bound": False, "last_display": {}}
            self._display_cache = cache

        if cache["last_display"].get(env_idx) == int(display_value):
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        prim_template = getattr(self, "prim_path", None) or f"{{ENV_REGEX_NS}}/{self.name}"
        prim_root = self._resolve_env_prim_path(prim_template, env_idx)
        shader_paths = [f"{prim_root}{suffix}" for suffix in self.digit_shader_suffixes]

        if not cache["textures_bound"]:
            self._ensure_digit_textures(
                stage, shader_paths, self._resolve_num_bc_texture(getattr(self, "usd_path", None))
            )
            cache["textures_bound"] = True

        for shader_path, tx in zip(shader_paths, self.display_value_to_digit_uv(display_value), strict=False):
            shader_prim = stage.GetPrimAtPath(shader_path)
            if shader_prim.IsValid():
                self._set_texture_translate_x(shader_prim, tx)

        cache["last_display"][env_idx] = int(display_value)

    @staticmethod
    def _resolve_env_prim_path(prim_template: str, env_idx: int) -> str:
        if "{ENV_REGEX_NS}" in prim_template:
            return prim_template.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_idx}")
        return prim_template

    @staticmethod
    def _resolve_num_bc_texture(usd_path: str | None) -> str | None:
        if not usd_path:
            return _KNOB_TEXTURE
        if is_asset_reference(usd_path):
            base = usd_path.rsplit("/", 1)[0]
            candidate = resolve_runtime_asset_reference(f"{base}/textures/num_bc.jpg")
            if os.path.isfile(candidate):
                return candidate
            return _KNOB_TEXTURE
        usd_dir = os.path.dirname(os.path.abspath(usd_path))
        local = os.path.join(usd_dir, "textures", "num_bc.jpg")
        return local if os.path.isfile(local) else _KNOB_TEXTURE

    @staticmethod
    def _ensure_digit_textures(stage: Any, shader_paths: list[str], texture_path: str) -> None:
        from pxr import Sdf

        asset = Sdf.AssetPath(texture_path)
        for shader_path in shader_paths:
            shader_prim = stage.GetPrimAtPath(shader_path)
            if not shader_prim.IsValid():
                continue
            for attr_name in ("inputs:diffuse_texture", "inputs:emissive_color_texture"):
                attr = shader_prim.GetAttribute(attr_name)
                if attr and attr.IsValid():
                    attr.Set(asset)

    @staticmethod
    def _set_texture_translate_x(shader_prim: Any, tx: float) -> None:
        from pxr import Gf

        attr = shader_prim.GetAttribute("inputs:texture_translate")
        if not attr or not attr.IsValid():
            return
        ty = 0.0
        current = attr.Get()
        if current is not None:
            ty = float(current[1])
        attr.Set(Gf.Vec2f(float(tx), ty))


@register_asset
class DrawerBa(LibraryObject):
    """3-layer drawer with prismatic travel; closed init is joint pos 0."""

    name = "drawer_ba"
    tags = ["object", "openable", "drawer", "containable"]
    usd_path = DRAWER_BA_USD
    object_type = ObjectType.ARTICULATION


# ---------------------------------------------------------------------------
#  Franka + Robotiq 2F-85 (DROID) single-arm task objects
# ---------------------------------------------------------------------------

_FRANKA_ASSET_ROOT = resolve_runtime_asset_reference("assets/robots/franka")


@register_asset
class FrankaBanana(LibraryObject):
    name = "franka_banana"
    tags = ["ycb", "fruit", "pickable"]
    usd_path = f"{_FRANKA_ASSET_ROOT}/objects/ycb/banana.usd"


@register_asset
class FrankaBowl(LibraryObject):
    name = "franka_bowl"
    tags = ["ycb", "container", "destination"]
    usd_path = f"{_FRANKA_ASSET_ROOT}/objects/ycb/bowl.usd"


@register_asset
class FrankaPlate(LibraryObject):
    name = "franka_plate"
    tags = ["vomp", "surface", "destination"]
    usd_path = f"{_FRANKA_ASSET_ROOT}/objects/vomp/plate_large/plate_large.usd"


@register_asset
class FrankaRubiksCube(LibraryObject):
    name = "franka_rubiks_cube"
    tags = ["hot3d", "pickable"]
    usd_path = f"{_FRANKA_ASSET_ROOT}/objects/hot3d/rubiks_cube.usd"


@register_asset
class FrankaTableOak(LibraryObject):
    name = "franka_table_oak"
    tags = ["fixture", "table"]
    usd_path = f"{_FRANKA_ASSET_ROOT}/fixtures/table_oak.usd"
    object_type = ObjectType.RIGID


@register_asset
class FrankaTable(LibraryObject):
    name = "franka_table"
    tags = ["fixture", "table"]
    usd_path = f"{_FRANKA_ASSET_ROOT}/fixtures/franka_table.usd"
    object_type = ObjectType.BASE


@register_asset
class RackD(LibraryObject):
    """Wooden rack used as the hanging support for ``towel_to_basket``."""

    name = "rack_d"
    tags = ["source", "rackable", "object", "hang_bar"]
    usd_path = resolve_runtime_asset_reference(
        join_asset_uri(ASSET_ROOT_URI, "objects/rack/rack_d.usd")
    )
    object_type = ObjectType.RIGID


@register_asset
class RackE(LibraryObject):
    """Coin rack destination for ``hand_over_coin``."""

    name = "rack_e"
    tags = ["container", "destination", "rackable", "object"]
    usd_path = resolve_runtime_asset_reference(
        join_asset_uri(ASSET_ROOT_URI, "objects/rack/rack_e.usd")
    )
    object_type = ObjectType.RIGID


@register_asset
class CoinAa(LibraryObject):
    """Coin pickable for ``hand_over_coin``."""

    name = "coin_aa"
    tags = ["pickable", "object", "coin"]
    usd_path = resolve_runtime_asset_reference(
        join_asset_uri(ASSET_ROOT_URI, "art_assets/models/objects/coin/coin_aa.usd")
    )
    object_type = ObjectType.RIGID
