"""Pose synchronization utilities for local layout editing."""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any

import torch

from manaenv.utils.asset_utils import is_kinematic_asset
from manaenv.utils.scene_state_utils import capture_active_slot_visibility
from manaenv.utils.usd_pose_sync import set_existing_pose_xform_ops

logger = logging.getLogger(__name__)


def _concrete_prim_path(prim_path_template: str, env_idx: int) -> str:
    prim_path = prim_path_template.replace("{ENV_REGEX_NS}", f"envs/env_{env_idx}")
    return prim_path.replace("env_.*", f"env_{env_idx}")


def _rigid_body_prim_path(obj: Any, asset_prim_path: str) -> str:
    """Return the tensor-backed rigid-body prim below an asset wrapper."""
    root_view = getattr(obj, "root_view", None)
    body_paths = [str(path) for path in getattr(root_view, "prim_paths", ())]
    matches = [path for path in body_paths if path == asset_prim_path or path.startswith(f"{asset_prim_path}/")]
    return min(matches, key=len) if matches else asset_prim_path


def _is_active_pool_slot(name: str, active_slots: dict[str, dict[str, str]]) -> bool:
    logical_name = name.split("__slot_", 1)[0]
    entry = active_slots.get(logical_name)
    return entry is None or entry.get("slot") == name


def _pose_xyzw(xform: Any) -> tuple[list[float], list[float]]:
    pose_xform = xform.RemoveScaleShear()
    translation = pose_xform.ExtractTranslation()
    rotation = pose_xform.ExtractRotationQuat()
    return (
        [float(translation[0]), float(translation[1]), float(translation[2])],
        [
            float(rotation.GetImaginary()[0]),
            float(rotation.GetImaginary()[1]),
            float(rotation.GetImaginary()[2]),
            float(rotation.GetReal()),
        ],
    )


def _world_pose_xyzw(prim: Any, usd_geom: Any) -> tuple[list[float], list[float]]:
    return _pose_xyzw(usd_geom.XformCache().GetLocalToWorldTransform(prim))


def _set_frame_view_pose(
    view: Any,
    *,
    position_xyz: list[float] | tuple[float, ...],
    quaternion_xyzw: list[float] | tuple[float, ...],
    device: str,
    warp: Any,
    local: bool,
) -> None:
    positions = warp.array([list(position_xyz)], dtype=warp.float32, device=device)
    orientations = warp.array([list(quaternion_xyzw)], dtype=warp.float32, device=device)
    if local:
        view.set_local_poses(translations=positions, orientations=orientations)
    else:
        view.set_world_poses(positions=positions, orientations=orientations)


def _sync_fabric_wrapper_from_usd(prim_path: str) -> None:
    """Refresh a GUI wrapper's Fabric matrices after its USD pose changes."""
    import usdrt
    from isaaclab.sim.utils import get_current_stage_id
    from usdrt import Rt, hierarchy

    stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Fabric wrapper prim does not exist: {prim_path}")

    xformable = Rt.Xformable(prim)
    xformable.CreateFabricHierarchyWorldMatrixAttr()
    xformable.CreateFabricHierarchyLocalMatrixAttr()
    xformable.SetLocalXformFromUsd()
    xformable.SetWorldXformFromUsd()

    fabric_hierarchy = hierarchy.IFabricHierarchy().get_fabric_hierarchy(
        stage.GetFabricId(),
        stage.GetStageIdAsStageId(),
    )
    fabric_hierarchy.update_world_xforms()


def _install_matrix_scale_reader(frame_view: Any, *, warp: Any) -> None:
    """Make Fabric initialization tolerate scalar or unauthored scale ops."""
    usd_view = getattr(frame_view, "_usd_view", None)
    if usd_view is None:
        return

    from isaaclab.utils.warp import ProxyArray
    from pxr import Gf, UsdGeom

    def _get_local_scales_from_matrix(view: Any, indices: Any = None) -> Any:
        scales = []
        for prim_idx in view._resolve_indices(indices):
            local_matrix = UsdGeom.Xformable(view._prims[prim_idx]).GetLocalTransformation()
            scale = Gf.Transform(local_matrix).GetScale()
            if scale is None:
                raise RuntimeError(f"unable to resolve local scale for prim index {prim_idx}")
            scales.append([float(scale[0]), float(scale[1]), float(scale[2])])
        return ProxyArray(warp.array(scales, dtype=warp.float32, device=view._device))

    # IsaacLab's USD fallback reads xformOp:scale as Vec3 directly. Some
    # benchmark assets author it as a scalar or leave it unauthored, even
    # though their composed local matrix has a valid scale. Override only this
    # layout view's fallback reader before Fabric performs its one-time seed.
    usd_view._get_local_scales_impl = MethodType(_get_local_scales_from_matrix, usd_view)


def sync_sim_poses_to_usd(
    gym_env: Any,
    *,
    synced_prim_paths: set[str] | None = None,
    fabric_views: dict[str, Any] | None = None,
) -> int:
    """Write current PhysX rigid object poses into USD prim xformOps.

    Call this before entering layout-edit mode so that the GUI Transform
    Gizmo shows the objects at their current simulation positions (not the
    stale USD values from spawn time or last reset).

    If provided, ``synced_prim_paths`` is replaced with the prim paths that
    were safely updated during this call.

    Returns the number of objects successfully synced.
    """
    import omni.usd
    import warp as wp
    from isaaclab.sim.utils import convert_world_pose_to_local
    from isaaclab.sim.views import FrameView
    from pxr import Gf, UsdGeom

    if synced_prim_paths is not None:
        synced_prim_paths.clear()
    if fabric_views is None:
        fabric_views = {}
        gym_env._layout_edit_fabric_views = fabric_views
    else:
        fabric_views.clear()

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return 0

    active_slots_by_env = [
        capture_active_slot_visibility(gym_env, env_id=env_idx) for env_idx in range(gym_env.num_envs)
    ]
    synced = 0

    for name, obj in gym_env.scene.rigid_objects.items():
        prim_path_template = obj.cfg.prim_path
        for env_idx in range(gym_env.num_envs):
            if not _is_active_pool_slot(name, active_slots_by_env[env_idx]):
                continue

            asset_prim_path = _concrete_prim_path(prim_path_template, env_idx)
            body_prim_path = _rigid_body_prim_path(obj, asset_prim_path)

            wrapper_prim = stage.GetPrimAtPath(asset_prim_path)
            body_prim = stage.GetPrimAtPath(body_prim_path)
            if not wrapper_prim.IsValid() or not body_prim.IsValid():
                continue
            if body_prim_path == asset_prim_path:
                logger.warning(
                    "Layout edit skipped '%s': a separate non-rigid wrapper prim is required.",
                    asset_prim_path,
                )
                continue

            pos_w = obj.data.root_pos_w.torch[env_idx].cpu().tolist()
            quat_xyzw = obj.data.root_quat_w.torch[env_idx].cpu().tolist()
            local_pos, local_quat_xyzw = convert_world_pose_to_local(
                pos_w,
                quat_xyzw,
                wrapper_prim.GetParent(),
            )

            wrapper_xformable = UsdGeom.Xformable(wrapper_prim)
            original_local_pos, original_local_quat = _pose_xyzw(wrapper_xformable.GetLocalTransformation())
            device = str(gym_env.device)
            try:
                body_view = FrameView(body_prim_path, device=device, validate_xform_ops=False)
                _install_matrix_scale_reader(body_view, warp=wp)
                body_view.get_world_poses()
            except Exception:
                logger.warning(
                    "Layout edit skipped '%s': Fabric frame-view initialization failed.",
                    asset_prim_path,
                    exc_info=True,
                )
                continue

            # The wrapper is the editable USD transform. The tensor-backed body
            # stays untouched in USD; its Fabric world pose is corrected after
            # the wrapper update so the visible object does not move twice.
            if not set_existing_pose_xform_ops(
                wrapper_xformable,
                position_xyz=local_pos,
                quaternion_xyzw=local_quat_xyzw,
                gf=Gf,
                usd_geom=UsdGeom,
            ):
                try:
                    _set_frame_view_pose(
                        body_view,
                        position_xyz=pos_w,
                        quaternion_xyzw=quat_xyzw,
                        device=device,
                        warp=wp,
                        local=False,
                    )
                except Exception:
                    logger.exception("Failed to restore Fabric pose for '%s'.", asset_prim_path)
                continue

            try:
                _sync_fabric_wrapper_from_usd(asset_prim_path)
                _set_frame_view_pose(
                    body_view,
                    position_xyz=pos_w,
                    quaternion_xyzw=quat_xyzw,
                    device=device,
                    warp=wp,
                    local=False,
                )
            except Exception:
                logger.warning(
                    "Layout edit skipped '%s': Fabric pose mirroring failed.",
                    asset_prim_path,
                    exc_info=True,
                )
                set_existing_pose_xform_ops(
                    wrapper_xformable,
                    position_xyz=original_local_pos,
                    quaternion_xyzw=original_local_quat,
                    gf=Gf,
                    usd_geom=UsdGeom,
                )
                try:
                    _sync_fabric_wrapper_from_usd(asset_prim_path)
                    _set_frame_view_pose(
                        body_view,
                        position_xyz=pos_w,
                        quaternion_xyzw=quat_xyzw,
                        device=device,
                        warp=wp,
                        local=False,
                    )
                except Exception:
                    logger.exception("Failed to restore Fabric pose for '%s'.", asset_prim_path)
                continue

            fabric_views[asset_prim_path] = body_view
            synced += 1
            if synced_prim_paths is not None:
                synced_prim_paths.add(asset_prim_path)

    return synced


def sync_usd_poses_to_sim(
    gym_env: Any,
    *,
    prim_paths: set[str] | None = None,
    fabric_views: dict[str, Any] | None = None,
) -> int:
    """Read rigid object poses from USD xformOps and write them into PhysX.

    During local layout-edit mode the user can move objects via the IsaacSim
    GUI (which modifies USD prim transforms).  Since Fabric mode keeps PhysX
    tensors as the authoritative state, we need to explicitly copy the
    GUI-edited USD poses back into the physics engine so that the simulation
    continues from the new layout.

    If ``prim_paths`` is provided, only those prims are copied back. Passing
    an empty set intentionally performs no writes.

    Returns the number of objects successfully synced.
    """
    import omni.usd
    import warp as wp
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return 0

    if fabric_views is None:
        fabric_views = getattr(gym_env, "_layout_edit_fabric_views", {})

    device = gym_env.device
    synced = 0

    for _name, obj in gym_env.scene.rigid_objects.items():
        prim_path_template = obj.cfg.prim_path
        for env_idx in range(gym_env.num_envs):
            asset_prim_path = _concrete_prim_path(prim_path_template, env_idx)
            if prim_paths is not None and asset_prim_path not in prim_paths:
                continue

            wrapper_prim = stage.GetPrimAtPath(asset_prim_path)
            if not wrapper_prim.IsValid():
                continue

            body_view = fabric_views.get(asset_prim_path)
            if body_view is None:
                logger.warning("Layout edit skipped '%s': Fabric frame views are unavailable.", asset_prim_path)
                continue

            pos, quat_xyzw = _world_pose_xyzw(wrapper_prim, UsdGeom)
            wrapper_xformable = UsdGeom.Xformable(wrapper_prim)
            edited_local_pos, edited_local_quat = _pose_xyzw(wrapper_xformable.GetLocalTransformation())
            if not set_existing_pose_xform_ops(
                wrapper_xformable,
                position_xyz=(0.0, 0.0, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                gf=Gf,
                usd_geom=UsdGeom,
            ):
                continue

            try:
                _sync_fabric_wrapper_from_usd(asset_prim_path)
                _set_frame_view_pose(
                    body_view,
                    position_xyz=pos,
                    quaternion_xyzw=quat_xyzw,
                    device=str(device),
                    warp=wp,
                    local=False,
                )
            except Exception:
                logger.warning(
                    "Layout edit skipped '%s': Fabric hierarchy restore failed.",
                    asset_prim_path,
                    exc_info=True,
                )
                set_existing_pose_xform_ops(
                    wrapper_xformable,
                    position_xyz=edited_local_pos,
                    quaternion_xyzw=edited_local_quat,
                    gf=Gf,
                    usd_geom=UsdGeom,
                )
                try:
                    _sync_fabric_wrapper_from_usd(asset_prim_path)
                    _set_frame_view_pose(
                        body_view,
                        position_xyz=pos,
                        quaternion_xyzw=quat_xyzw,
                        device=str(device),
                        warp=wp,
                        local=False,
                    )
                except Exception:
                    logger.exception("Failed to restore edited Fabric pose for '%s'.", asset_prim_path)
                continue

            pose_tensor = torch.tensor(
                [pos + quat_xyzw],
                dtype=torch.float32,
                device=device,
            )

            env_ids = torch.tensor([env_idx], dtype=torch.long, device=device)
            obj.write_root_pose_to_sim_index(root_pose=pose_tensor, env_ids=env_ids)
            if not is_kinematic_asset(obj):
                obj.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros(1, 6, dtype=torch.float32, device=device),
                    env_ids=env_ids,
                )
            synced += 1
            fabric_views.pop(asset_prim_path, None)

    return synced
