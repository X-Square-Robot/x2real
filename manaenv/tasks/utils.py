"""Task utilities – shared helpers for scene queries and object reset events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from manaenv.terms.events import (
    randomize_object_orientation,
    randomize_object_pose,
    randomize_object_pose_from_pose_range,
    randomize_object_pose_full,
    reset_scene_to_default,
    set_object_pose,
)

from .task import DomainRandomizationCfg

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg


def build_object_reset_event_term(
    asset_name: str,
    initial_pose: Any,
    domain_randomization_cfg: DomainRandomizationCfg | None = None,
    pose_range: dict | None = None,
) -> EventTermCfg:
    """Build the reset/randomization event term for a single object."""
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    position_cfg = domain_randomization_cfg.object_position if domain_randomization_cfg is not None else None
    orientation_cfg = domain_randomization_cfg.object_orientation if domain_randomization_cfg is not None else None

    if position_cfg is not None and orientation_cfg is not None:
        return EventTermCfg(
            func=randomize_object_pose_full,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "initial_pose": initial_pose,
                "position_range": position_cfg.to_range_dict(),
                "yaw_range": list(orientation_cfg.yaw_range),
            },
        )

    if position_cfg is not None:
        if position_cfg.method == "uniform_offset":
            return EventTermCfg(
                func=randomize_object_pose,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(asset_name),
                    "initial_pose": initial_pose,
                    "position_range": position_cfg.to_range_dict(),
                },
            )
        raise ValueError(f"Unsupported object_position randomization method: {position_cfg.method}")

    if orientation_cfg is not None:
        if orientation_cfg.method == "uniform_yaw":
            return EventTermCfg(
                func=randomize_object_orientation,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(asset_name),
                    "initial_pose": initial_pose,
                    "yaw_range": list(orientation_cfg.yaw_range),
                },
            )
        raise ValueError(f"Unsupported object_orientation randomization method: {orientation_cfg.method}")

    if pose_range:
        return EventTermCfg(
            func=randomize_object_pose_from_pose_range,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "initial_pose": initial_pose,
                "pose_range": pose_range,
            },
        )

    return EventTermCfg(
        func=set_object_pose,
        mode="reset",
        params={
            "pose": initial_pose,
            "asset_cfg": SceneEntityCfg(asset_name),
        },
    )


def find_assets_by_tag(scene, tag: str) -> list:
    """Return assets from *scene* whose unified ``tags`` contain *tag*."""
    assets = getattr(scene, "assets", {})
    assets_list = list(assets.values()) if isinstance(assets, dict) else list(assets)
    result = []
    for a in assets_list:
        tags = getattr(a, "tags", None) or []
        if isinstance(tags, str):
            tags = [tags]
        if tag in tags:
            result.append(a)
    return result


def find_background(scene):
    """Return the first background asset from *scene*."""
    assets = getattr(scene, "assets", {})
    assets_list = list(assets.values()) if isinstance(assets, dict) else list(assets)
    for a in assets_list:
        if getattr(a, "_component_type", "") == "BackgroundComponent":
            return a
        if hasattr(a, "object_min_z"):
            return a
    return None


def resolve_pick_and_place_assets(scene):
    """Resolve pick/place/background assets from a scene."""
    pick_candidates = find_assets_by_tag(scene, "pickable")
    place_candidates = find_assets_by_tag(scene, "destination")
    background_asset = find_background(scene)

    pick_asset = pick_candidates[0] if pick_candidates else None
    place_asset = place_candidates[0] if place_candidates else None

    assets = getattr(scene, "assets", {})
    assets_list = list(assets.values()) if isinstance(assets, dict) else list(assets)
    object_assets = [asset for asset in assets_list if asset is not background_asset]

    remaining = [asset for asset in object_assets if asset is not pick_asset and asset is not place_asset]
    if pick_asset is None and remaining:
        pick_asset = remaining.pop(0)
    if place_asset is None and remaining:
        place_asset = remaining.pop(0)

    if pick_asset is None or place_asset is None or background_asset is None:
        return None
    return pick_asset, place_asset, background_asset


def build_object_reset_events(
    objects: list,
    domain_randomization_cfg: DomainRandomizationCfg | None = None,
    pose_range: dict | None = None,
) -> list[tuple[str, type, Any]]:
    """Build configclass field tuples for per-object pose reset / randomize.

    Returns a list of ``(field_name, field_type, default_value)`` tuples
    suitable for ``make_configclass()``.  Caller should prepend any
    additional fields (e.g. ``reset_scene_to_default``) and wrap with
    ``make_configclass(name, fields)()``.
    """
    from isaaclab.managers import EventTermCfg

    fields: list[tuple[str, type, Any]] = []
    for i, obj in enumerate(objects):
        initial_pose = obj.get_initial_pose()
        if initial_pose is None:
            continue
        field_name = (
            f"randomize_object_{i}_pose"
            if domain_randomization_cfg is not None or pose_range
            else f"reset_object_{i}_pose"
        )
        fields.append(
            (
                field_name,
                EventTermCfg,
                build_object_reset_event_term(
                    asset_name=obj.name,
                    initial_pose=initial_pose,
                    domain_randomization_cfg=domain_randomization_cfg,
                    pose_range=pose_range,
                ),
            )
        )
    return fields


def build_default_events_cfg(
    objects: list,
    domain_randomization_cfg: DomainRandomizationCfg | None = None,
    pose_range: dict | None = None,
) -> Any:
    """Convenience: build a complete events configclass with scene reset + object resets."""
    from isaaclab.managers import EventTermCfg
    from isaaclab_arena.utils.configclass import make_configclass

    fields = [
        (
            "reset_scene_to_default",
            EventTermCfg,
            EventTermCfg(
                func=reset_scene_to_default,
                mode="reset",
                params={"reset_joint_targets": True},
            ),
        ),
    ]
    fields.extend(
        build_object_reset_events(objects, domain_randomization_cfg=domain_randomization_cfg, pose_range=pose_range)
    )
    return make_configclass("ObjectResetEventsCfg", fields)()
