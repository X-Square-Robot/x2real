"""Helpers for restoring recorded IsaacLab scene state safely."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from manaenv.utils.asset_utils import (
    is_kinematic_asset,
    pool_cache_slot_name,
    record_active_asset_slot,
    resolve_pool_cache_size,
    resolve_pool_slot_asset_id,
    scene_has_asset,
)
from manaenv.utils.usd_pose_sync import should_author_pose_owned_visibility_to_usd

ACTIVE_SLOT_VISIBILITY_ATTR = "mana_active_slot_visibility"


def publish_scene_state(env: Any) -> None:
    """Publish restored scene state through the environment-owned boundary.

    ``ManaPhysxAsyncKitAsyncEnv`` owns Fabric isolation and exposes
    ``publish_fabric``; synchronous environments retain their regular
    ``SimulationContext.forward`` behavior. Restore/replay code should use this
    helper instead of selecting a physics-step mode itself.
    """
    unwrapped = getattr(env, "unwrapped", None)
    publisher = getattr(env, "publish_fabric", None)
    if not callable(publisher) and unwrapped is not None and unwrapped is not env:
        publisher = getattr(unwrapped, "publish_fabric", None)
    if callable(publisher):
        publisher()
        return

    sim = getattr(env, "sim", None)
    if sim is None and unwrapped is not None and unwrapped is not env:
        sim = getattr(unwrapped, "sim", None)
    forward = getattr(sim, "forward", None)
    if not callable(forward):
        raise RuntimeError("Scene-state publication requires env.publish_fabric() or env.sim.forward()")
    forward()


def capture_active_slot_visibility(env: Any, *, env_id: int = 0) -> dict[str, dict[str, str]]:
    """Return the sampled schema-vNext pool slots for one environment.

    The runtime store is deliberately copied into a JSON-safe shape so the
    recorder can attach it to an HDF5 episode attribute without retaining a
    reference to mutable reset state.
    """
    state = getattr(env, "_schema_vnext_active_asset_slots", None)
    if not isinstance(state, Mapping):
        return {}

    captured: dict[str, dict[str, str]] = {}
    # ``runtime`` is the common plan id; preserve the first entry for an object
    # if several plans happened to report it during one reset.
    plan_items = sorted(state.items(), key=lambda item: (str(item[0]) != "runtime", str(item[0])))
    for _plan_id, plan_state in plan_items:
        if not isinstance(plan_state, Mapping):
            continue
        env_state = plan_state.get(int(env_id), plan_state.get(str(int(env_id))))
        if not isinstance(env_state, Mapping):
            continue
        for object_id, entry in env_state.items():
            if str(object_id) in captured:
                continue
            if isinstance(entry, Mapping):
                slot = str(entry.get("slot") or "").strip()
                asset_id = str(entry.get("asset_id") or "").strip()
            else:
                slot = str(entry or "").strip()
                asset_id = ""
            if slot:
                captured[str(object_id)] = {"slot": slot, "asset_id": asset_id}
    return captured


def apply_recorded_active_slot_visibility(
    env: Any,
    active_slots: Mapping[str, Any] | None,
    *,
    env_id: int = 0,
) -> int:
    """Restore the recorded active pool slot without competing with PhysX.

    Simulation-owned slots retain authored USD visibility because the preceding
    recorded-state restore already reinstates their active and parked poses.
    USD-only peers are hidden before the recorded slot is exposed.
    """
    if not isinstance(active_slots, Mapping):
        return 0

    restored = 0
    for object_id, raw_entry in active_slots.items():
        object_id = str(object_id).strip()
        if not object_id:
            continue
        entry = raw_entry if isinstance(raw_entry, Mapping) else {"slot": raw_entry}
        active_slot = str(entry.get("slot") or "").strip()
        if not active_slot:
            continue
        asset_id = str(entry.get("asset_id") or "").strip()

        try:
            cache_size = resolve_pool_cache_size(env, object_id)
        except (TypeError, ValueError):
            cache_size = 1

        slot_names = [pool_cache_slot_name(object_id, index) for index in range(int(cache_size))]
        if active_slot not in slot_names:
            slot_names.append(active_slot)

        slot_state_restored = False
        for slot_name in slot_names:
            if not scene_has_asset(env, slot_name):
                continue
            asset = env.scene[slot_name]
            if not should_author_pose_owned_visibility_to_usd(asset):
                slot_state_restored = True
                continue
            setter = getattr(asset, "set_visibility", None)
            if not callable(setter):
                continue
            setter(slot_name == active_slot, env_ids=[int(env_id)])
            slot_state_restored = True

        if slot_state_restored:
            record_active_asset_slot(
                env,
                plan_id="runtime",
                env_id=int(env_id),
                object_id=object_id,
                active_slot=active_slot,
                asset_id=asset_id or resolve_pool_slot_asset_id(env, object_id, active_slot),
            )
            restored += 1
    return restored


def reset_scene_to_recorded_state(
    scene: Any,
    state: dict[str, dict[str, dict[str, Any]]],
    env_ids: Sequence[int] | None = None,
    *,
    is_relative: bool = False,
) -> None:
    """Restore recorded scene state without writing velocity to kinematic bodies."""
    if env_ids is None:
        env_ids = scene._ALL_INDICES

    for asset_name, articulation in scene.articulations.items():
        asset_state = state["articulation"][asset_name]
        root_pose = asset_state["root_pose"].clone().to(scene.device)
        if is_relative:
            root_pose[:, :3] += scene.env_origins[env_ids]
        root_velocity = asset_state["root_velocity"].clone().to(scene.device)
        articulation.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        articulation.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

        joint_position = asset_state["joint_position"].clone().to(scene.device)
        joint_velocity = asset_state["joint_velocity"].clone().to(scene.device)
        articulation.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
        articulation.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
        articulation.set_joint_velocity_target_index(target=joint_velocity, env_ids=env_ids)

    for asset_name, deformable_object in scene.deformable_objects.items():
        asset_state = state["deformable_object"][asset_name]
        nodal_position = asset_state["nodal_position"].clone().to(scene.device)
        if is_relative:
            nodal_position[:, :3] += scene.env_origins[env_ids]
        nodal_velocity = asset_state["nodal_velocity"].clone().to(scene.device)
        deformable_object.write_nodal_pos_to_sim(nodal_position, env_ids=env_ids)
        deformable_object.write_nodal_velocity_to_sim(nodal_velocity, env_ids=env_ids)

    for asset_name, rigid_object in scene.rigid_objects.items():
        asset_state = state["rigid_object"][asset_name]
        root_pose = asset_state["root_pose"].clone().to(scene.device)
        if is_relative:
            root_pose[:, :3] += scene.env_origins[env_ids]
        rigid_object.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        if not is_kinematic_asset(rigid_object):
            root_velocity = asset_state["root_velocity"].clone().to(scene.device)
            rigid_object.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    for asset_name, surface_gripper in scene.surface_grippers.items():
        surface_gripper.set_grippers_command(state["gripper"][asset_name])

    scene.write_data_to_sim()
