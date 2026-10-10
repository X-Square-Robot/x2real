from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..assets.collection import asset_map
from ..common.diagnostics import error
from ..common.geometry import Pose


@dataclass(frozen=True)
class SimplePose:
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


def _make_pose(pose: Pose) -> Any:
    try:
        from isaaclab_arena.utils.pose import Pose as ArenaPose

        return ArenaPose(position_xyz=tuple(pose.position), rotation_xyzw=tuple(pose.quaternion))
    except Exception:
        return SimplePose(position_xyz=tuple(pose.position), rotation_xyzw=tuple(pose.quaternion))


def apply_solved_poses(
    assets: Mapping[str, Any] | Iterable[Any],
    poses: Mapping[str, Pose],
) -> None:
    by_name = asset_map(assets)
    for name, pose in poses.items():
        asset = by_name.get(name)
        if asset is None:
            raise error("asset_missing", "solved pose has no matching ManaEnv asset", target=name)
        arena_pose = _make_pose(pose)
        if hasattr(asset, "set_initial_pose"):
            # set_initial_pose refreshes object_cfg.init_state in place; rebuilding
            # object_cfg here would wipe spawn overrides (kinematic/mass/friction)
            # applied by the scene component's _customize_asset.
            asset.set_initial_pose(arena_pose)
            from manaenv.terms.events import bind_kinematic_pose_reset_event

            bind_kinematic_pose_reset_event(asset)
        else:
            asset.initial_pose = arena_pose
            if hasattr(asset, "_init_object_cfg"):
                try:
                    asset.object_cfg = asset._init_object_cfg()
                except Exception as exc:
                    raise error(
                        "asset_cfg_init_failed",
                        "failed to refresh ManaEnv asset object_cfg after applying solved pose",
                        target=name,
                        exception=str(exc),
                    ) from exc
                from manaenv.pipeline.generators.scene.components.objects_component import reapply_kinematic_override

                reapply_kinematic_override(asset)
