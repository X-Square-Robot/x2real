"""Custom event functions for domain randomization.

These functions follow the IsaacLab EventTermCfg function signature:
    func(env: ManagerBasedEnv, env_ids: torch.Tensor, **params)

They are injected into the event_manager via get_events_cfg() override
in ManaEnv task classes, keeping all changes within the manaenv package.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import random as py_random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_arena.terms.events import set_object_pose as _arena_set_object_pose
from isaaclab_arena.terms.events import set_object_pose_per_env as _arena_set_object_pose_per_env
from isaaclab_arena.utils.pose import Pose

from manaenv.pipeline.generators.scene_layout_solver.manaenv_bridge.runtime_events import apply_runtime_slot_activation
from manaenv.pipeline.runtime.log_store import write_committed
from manaenv.terms.attribute_instruction import compose_attribute_instruction_with_confusion
from manaenv.terms.balance_problem_pool import balance_problem_slice, enumerate_balance_problems
from manaenv.utils.asset_utils import (
    get_scene_entity_name,
    inactive_pool_slot_position,
    is_kinematic_asset,
    pool_cache_slot_name,
    record_active_asset_slot,
    resolve_active_scene_entity_name,
    resolve_pool_cache_size,
    resolve_pool_slot_asset_id,
    scene_has_asset,
)
from manaenv.utils.debug_print import manaprint
from manaenv.utils.scene_state_utils import publish_scene_state
from manaenv.utils.tensor_utils import as_torch, to_numpy_keep_shape
from manaenv.utils.usd_pose_sync import set_existing_pose_xform_ops
from manaenv.utils.usd_pose_sync import (
    should_author_pose_owned_visibility_to_usd as _should_author_pose_owned_visibility_to_usd,
)
from manaenv.utils.usd_pose_sync import should_sync_runtime_pose_to_usd as _should_sync_runtime_pose_to_usd

_ALPHABET_WORD_STATE_ATTR = "_mana_alphabet_word_state"
_TASK_SAMPLE_RNG_ATTR = "_mana_task_sample_rngs"


def _is_kinematic_rigid_object(asset: Any) -> bool:
    """Return whether an asset is explicitly configured as a kinematic rigid body."""
    return is_kinematic_asset(asset)


def _write_root_velocity_if_dynamic(asset: Any, root_velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write root velocity only for dynamic assets."""
    if hasattr(asset, "write_root_velocity_to_sim_index") and not _is_kinematic_rigid_object(asset):
        asset.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)


def _walk_prim_subtree(root_prim: Any) -> Iterator[Any]:
    """Yield one USD prim subtree without requiring a global stage traversal."""
    stack = [root_prim]
    while stack:
        prim = stack.pop()
        yield prim
        stack.extend(reversed(list(prim.GetChildren())))


def disable_cast_shadows(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    body_names: list[str] | None = None,
    scene_wide: bool = False,
) -> None:
    """Author inherited ``primvars:doNotCastShadows`` on env roots or robot links."""
    from pxr import Sdf, UsdGeom

    stage = env.scene.stage
    env_namespace = str(getattr(env.scene, "env_ns", "/World/envs")).rstrip("/")
    resolved_env_ids = range(int(env.num_envs)) if env_ids is None else (int(value) for value in env_ids.tolist())
    requested_names = {str(name) for name in body_names or () if str(name)}
    authored = 0

    for env_id in resolved_env_ids:
        env_root = stage.GetPrimAtPath(f"{env_namespace}/env_{env_id}")
        if not env_root:
            continue
        if scene_wide:
            targets = (env_root,)
        else:
            targets = (prim for prim in _walk_prim_subtree(env_root) if prim.GetName() in requested_names)
        for prim in targets:
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
            authored += 1

    if authored == 0:
        scope = "scene roots" if scene_wide else f"body names {sorted(requested_names)!r}"
        manaprint(f"WARNING: disable_cast_shadows found no matching {scope}.")


def reset_scene_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    reset_joint_targets: bool = False,
) -> None:
    """Restore the scene default state without writing velocity to kinematic bodies."""
    for rigid_object in env.scene.rigid_objects.values():
        default_root_pose = rigid_object.data.default_root_pose.torch[env_ids].clone()
        default_root_vel = rigid_object.data.default_root_vel.torch[env_ids].clone()
        default_root_pose[:, :3] += env.scene.env_origins[env_ids]
        rigid_object.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        _write_root_velocity_if_dynamic(rigid_object, default_root_vel, env_ids)

    for articulation_asset in env.scene.articulations.values():
        default_root_pose = articulation_asset.data.default_root_pose.torch[env_ids].clone()
        default_root_vel = articulation_asset.data.default_root_vel.torch[env_ids].clone()
        default_root_pose[:, :3] += env.scene.env_origins[env_ids]
        articulation_asset.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)

        default_joint_pos = articulation_asset.data.default_joint_pos.torch[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel.torch[env_ids].clone()
        articulation_asset.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=env_ids)
        articulation_asset.write_joint_velocity_to_sim_index(velocity=default_joint_vel, env_ids=env_ids)
        if reset_joint_targets:
            articulation_asset.set_joint_position_target_index(target=default_joint_pos, env_ids=env_ids)
            articulation_asset.set_joint_velocity_target_index(target=default_joint_vel, env_ids=env_ids)

    for deformable_object in env.scene.deformable_objects.values():
        nodal_state = deformable_object.data.default_nodal_state_w.torch[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


def set_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
    velocity: Any | None = None,
) -> None:
    """Set one object's pose and only write velocity when the body is dynamic."""
    if env_ids is None:
        return
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    pose_tensor = pose.to_tensor(device=env.device).repeat(num_envs, 1)
    pose_tensor[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim_index(root_pose=pose_tensor, env_ids=env_ids)
    if velocity is None:
        root_velocity = torch.zeros(num_envs, 6, device=env.device)
    else:
        root_velocity = velocity.to_tensor(device=env.device).unsqueeze(0).expand(num_envs, -1)
    _write_root_velocity_if_dynamic(asset, root_velocity, env_ids)


def set_object_pose_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_list: list[Pose],
) -> None:
    """Set per-environment poses without writing velocity to kinematic bodies."""
    if env_ids is None:
        return
    asset = env.scene[asset_cfg.name]
    for cur_env in env_ids.tolist():
        pose_tensor = pose_list[cur_env].to_tensor(device=env.device).unsqueeze(0)
        pose_tensor[0, :3] += env.scene.env_origins[cur_env]
        cur_env_ids = torch.tensor([cur_env], device=env.device)
        asset.write_root_pose_to_sim_index(root_pose=pose_tensor, env_ids=cur_env_ids)
        _write_root_velocity_if_dynamic(asset, torch.zeros(1, 6, device=env.device), cur_env_ids)


def bind_kinematic_pose_reset_event(asset: Any) -> None:
    """Replace Arena fixed-pose reset events with kinematic-aware ManaEnv variants.

    Arena ``Object`` / ``PlaceableAsset`` store the reset term on
    ``_pose_event_cfg`` (rebuilt by ``set_initial_pose``).
    """
    if not _is_kinematic_rigid_object(asset):
        return
    _rewrite_pose_event_term(getattr(asset, "_pose_event_cfg", None))


def _rewrite_pose_event_term(event_cfg: Any) -> bool:
    """Rewrite one Arena pose-reset term to the ManaEnv kinematic-safe variant."""
    if event_cfg is None:
        return False
    func = getattr(event_cfg, "func", None)
    if func is _arena_set_object_pose:
        event_cfg.func = set_object_pose
        return True
    if func is _arena_set_object_pose_per_env:
        event_cfg.func = set_object_pose_per_env
        return True
    return False


_ARENA_POSE_RESET_REPLACEMENTS = {
    _arena_set_object_pose: set_object_pose,
    _arena_set_object_pose_per_env: set_object_pose_per_env,
}


def rewrite_arena_pose_reset_events(events_cfg: Any) -> int:
    """Rewrite Arena pose-reset terms to ManaEnv variants that skip kinematic velocity.

    This is the assembly-time root cure: even if an asset-level bind missed
    ``_pose_event_cfg`` (or ``set_initial_pose`` rebuilt Arena's callable later),
    the composed ``env_cfg.events`` still must not call
    ``PxRigidDynamic::setLinearVelocity`` on kinematic bodies such as
    ``stack_base`` / paper props.

    Returns:
        Number of event terms whose ``func`` was rewritten.
    """
    if events_cfg is None:
        return 0
    rewritten = 0
    for name in getattr(events_cfg, "__dataclass_fields__", {}) or ():
        term = getattr(events_cfg, name, None)
        func = getattr(term, "func", None)
        replacement = _ARENA_POSE_RESET_REPLACEMENTS.get(func)
        if replacement is None or func is replacement:
            continue
        term.func = replacement
        rewritten += 1
    return rewritten


def _task_sample_episode_token(env: ManagerBasedEnv, key: str, env_id: int) -> int:
    """Advance and return a per-(key, env_id) episode token for reset sampling."""

    counters = getattr(env, _TASK_SAMPLE_RNG_ATTR, None)
    if not isinstance(counters, dict):
        counters = {}
        setattr(env, _TASK_SAMPLE_RNG_ATTR, counters)
    episode_key = ("episode", str(key), int(env_id))
    token = int(counters.get(episode_key, 0))
    counters[episode_key] = token + 1
    return token


def _task_sample_rng(
    env: ManagerBasedEnv,
    key: str,
    seed: int | None = None,
    *,
    env_id: int = 0,
) -> random.Random:
    """Return a per-(key, env_id, episode) RNG for reset-time task sampling.

    With a seed (YAML override or master-derived via :mod:`master_seed`), each
    env gets an independent stable stream. Without any seed budget, each call
    draws a fresh OS-entropy RNG so parallel envs do not collapse to one draw.
    """
    from manaenv.pipeline.runtime import master_seed as master_seed_runtime

    episode_token = _task_sample_episode_token(env, key, int(env_id))
    return master_seed_runtime.event_rng(
        key=str(key),
        env_id=int(env_id),
        episode_token=int(episode_token),
        override=None if seed is None else int(seed),
    )


def capture_init_ee_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ee_frame_keys: list[str],
    init_state_attr: str = "_mana_init_ee_pos_w",
) -> None:
    """Freeze each EE frame's episode-start world position for the reset envs.

    Paired with :func:`manaenv.terms.terminations.ee_near_init_pose`, which reads
    ``env.<init_state_attr>[frame_key]`` ``[num_envs, 3]`` to score the dynamic
    ``prep`` / ``return`` stages. Capture must be a reset event (runs for all
    reset envs unconditionally) rather than lazy inside the predicate: under DAG
    gating a later stage's success func is not called until its predecessors
    complete, so a lazy capture would freeze a mid-episode pose, not the start.
    """
    if env_ids is None:
        return
    store = getattr(env, init_state_attr, None)
    if not isinstance(store, dict):
        store = {}
        setattr(env, init_state_attr, store)
    for frame_key in ee_frame_keys:
        if not scene_has_asset(env.scene, frame_key):
            continue
        frame = env.scene[frame_key]
        cur_pos_w = frame.data.target_pos_w.torch[:, 0, :]
        buf = store.get(frame_key)
        if not isinstance(buf, torch.Tensor) or buf.shape != (env.num_envs, 3):
            buf = torch.zeros((env.num_envs, 3), dtype=cur_pos_w.dtype, device=cur_pos_w.device)
            store[frame_key] = buf
        buf[env_ids] = cur_pos_w[env_ids].detach()


# ---------------------------------------------------------------------------
# ECU Assembly — 10 discrete placement scenarios
# Each entry: (rack_2_pos, block_2_pos, rack_1_pos, block_1_pos)
# Indices 0-9 correspond to user scenarios 1-10.
# ---------------------------------------------------------------------------
_ECU_SCENARIOS: list[tuple[list, list, list, list]] = [
    # 0  s1: block_1 left-center, block_2 right-center
    ([0.13, -0.49, -0.15], [0.14932, -0.49281, -0.07076], [0.07, -0.08, -0.15], [0.06279, -0.09971, -0.06384]),
    # 1  s2: block_1 right-far, block_2 right-far
    ([0.01, -0.71, -0.15], [0.01279, -0.70970, -0.06384], [-0.05, -0.48, -0.15], [-0.05721, -0.49971, -0.06384]),
    # 2  s3: block_1 left-far, block_2 right-far
    ([0.01, -0.61, -0.15], [0.01279, -0.60970, -0.06384], [-0.05, -0.08, -0.15], [-0.05721, -0.09971, -0.06384]),
    # 3  s4: block_1 right-center, block_2 right-center
    ([0.13, -0.49, -0.15], [0.14932, -0.49281, -0.07076], [0.09, -0.68, -0.15], [0.08279, -0.69971, -0.06384]),
    # 4  s5: block_1 right-far, block_2 left-center
    ([0.07, 0.07, -0.15], [0.06279, 0.07030, -0.06384], [-0.05, -0.48, -0.15], [-0.05721, -0.49971, -0.06384]),
    # 5  s6: block_1 right-far, block_2 left-near
    ([0.23, -0.09, -0.15], [0.24932, -0.09281, -0.07076], [-0.05, -0.48, -0.15], [-0.05721, -0.49971, -0.06384]),
    # 6  s7: block_1 left-center, block_2 left-center
    ([0.07, 0.07, -0.15], [0.06279, 0.07030, -0.06384], [0.07, -0.08, -0.15], [0.06279, -0.09971, -0.06384]),
    # 7  s8: block_1 left-far, block_2 left-far
    ([0.01, 0.09, -0.15], [0.01279, 0.09030, -0.06384], [-0.05, -0.08, -0.15], [-0.05721, -0.09971, -0.06384]),
    # 8  s9: block_1 right-center, block_2 left-near (distinct from s7)
    ([0.23, 0.07, -0.15], [0.24932, 0.07030, -0.07076], [0.13, -0.08, -0.15], [0.14932, -0.09971, -0.07076]),
    # 9  s10: block_1 left-far, block_2 right-center
    ([0.15, -0.49, -0.15], [0.16932, -0.49281, -0.07076], [-0.05, -0.08, -0.15], [-0.05721, -0.09971, -0.06384]),
]


def reset_ecu_random_scenario(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    scenario_idx: int = -1,
) -> None:
    """Randomly select one of 10 ECU placement scenarios and apply it on episode reset.

    Args:
        scenario_idx: -1 = random each episode; 0-9 = fixed scenario (user's s1-s10).
    """
    if len(env_ids) == 0:
        return

    if scenario_idx < 0:
        idx = random.randint(0, len(_ECU_SCENARIOS) - 1)
    else:
        idx = max(0, min(scenario_idx, len(_ECU_SCENARIOS) - 1))

    rack_2_pos, block_2_pos, rack_1_pos, block_1_pos = _ECU_SCENARIOS[idx]
    manaprint(f"INFO: ECU scenario {idx + 1}/10 → rack_2={rack_2_pos}, rack_1={rack_1_pos}")

    # Scene entity names: ecu_rack_0=rack_2, ecu_rack_1=rack_1 (order from scene YAML)
    for entity_name, new_pos in [
        ("ecu_rack_0", rack_2_pos),
        ("ecu_block_2", block_2_pos),
        ("ecu_rack_1", rack_1_pos),
        ("ecu_block_1", block_1_pos),
    ]:
        try:
            asset = env.scene[entity_name]
            sim_pose = _default_root_pose_local_from_asset(asset, env_ids)
            if sim_pose is None:
                continue
            sim_pose[:, :3] = torch.tensor(new_pos, device=env.device, dtype=sim_pose.dtype)
            # default_root_state is env-local; add env_origins to reach world frame
            # (matches the pattern used by randomize_object_pose_full etc.).
            sim_pose[:, :3] += env.scene.env_origins[env_ids]
            _write_root_pose_with_zero_velocity(asset, sim_pose, env_ids)
        except Exception as e:
            manaprint(f"WARNING: reset_ecu_random_scenario failed for '{entity_name}': {e}")


def _sample_light_scalar(range_values, default: float) -> float:
    if range_values is None:
        return float(default)
    if isinstance(range_values, (int, float)):
        return float(range_values)
    if len(range_values) < 2:
        return float(default)
    lo, hi = float(range_values[0]), float(range_values[1])
    return float(torch.empty((), device="cpu").uniform_(lo, hi).item())


def _sample_light_choice(values: list[str] | tuple[str, ...] | None) -> str | None:
    if not values:
        return None
    idx = int(torch.randint(0, len(values), (), device="cpu").item())
    return str(values[idx])


def _discover_light_prim_paths(stage) -> list[str]:
    light_type_names = {
        "DomeLight",
        "DistantLight",
        "RectLight",
        "SphereLight",
        "DiskLight",
        "CylinderLight",
    }
    paths: list[str] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() in light_type_names:
            paths.append(str(prim.GetPath()))
    return paths


def _resolve_light_prim_paths(prim_paths: list[str], env_ids: torch.Tensor) -> list[str]:
    resolved: list[str] = []
    for prim_path in prim_paths:
        path = str(prim_path)
        if "{ENV_REGEX_NS}" not in path:
            resolved.append(path)
            continue
        for env_idx in env_ids:
            concrete = path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{int(env_idx)}")
            concrete = concrete.replace("env_.*", f"env_{int(env_idx)}")
            resolved.append(concrete)
    return resolved


def _sample_orientation_quat(rotation_rpy_range: dict[str, list[float] | tuple[float, float]] | None):
    if not rotation_rpy_range:
        return None
    from pxr import Gf

    roll = _sample_light_scalar(rotation_rpy_range.get("roll"), 0.0)
    pitch = _sample_light_scalar(rotation_rpy_range.get("pitch"), 0.0)
    yaw = _sample_light_scalar(rotation_rpy_range.get("yaw"), 0.0)
    rotation = (
        Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw)
        * Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), pitch)
        * Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), roll)
    )
    return rotation.GetQuat()


def _set_light_orientation_quat(prim, orientation_quat) -> None:
    if orientation_quat is None:
        return
    try:
        from pxr import UsdGeom
    except Exception:
        return

    xformable = UsdGeom.Xformable(prim)
    orient_op = None
    for op in xformable.GetOrderedXformOps() or ():
        if op.GetOpName() == "xformOp:orient":
            orient_op = op
            break
    if orient_op is None:
        orient_op = xformable.AddOrientOp()
    orient_op.Set(orientation_quat)


def _set_light_texture_file(prim, texture_file: str | None) -> None:
    if not texture_file:
        return
    try:
        from pxr import Sdf, UsdLux

        dome_light = UsdLux.DomeLight(prim)
        if dome_light:
            dome_light.CreateTextureFileAttr(Sdf.AssetPath(texture_file), True)
            return
    except Exception:
        pass

    for attr_name in ("inputs:texture:file", "inputs:textureFile", "inputs:texture_file"):
        attr = prim.GetAttribute(attr_name)
        if attr:
            attr.Set(texture_file)
            return


def randomize_scene_lighting(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    prim_paths: list[str] | None = None,
    intensity_range: list[float] | tuple[float, float] = (500.0, 1500.0),
    exposure_range: list[float] | tuple[float, float] | None = None,
    color_range: dict[str, list[float] | tuple[float, float]] | None = None,
    rotation_rpy_range: dict[str, list[float] | tuple[float, float]] | None = None,
    texture_file_candidates: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Randomize USD light attributes at episode reset without rebuilding the stage.

    World-scoped lights such as ``/World/DomeLight`` are shared global lighting
    and are written once. Env-scoped lights may use ``{ENV_REGEX_NS}``, which is
    expanded for each reset env id.
    """
    del env

    try:
        import omni.usd
        from pxr import Gf
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    targets = _resolve_light_prim_paths(list(prim_paths or []), env_ids) or _discover_light_prim_paths(stage)
    if not targets:
        return

    intensity = _sample_light_scalar(intensity_range, 1000.0)
    exposure = _sample_light_scalar(exposure_range, 0.0) if exposure_range is not None else None
    orientation_quat = _sample_orientation_quat(rotation_rpy_range)
    texture_candidates = list(texture_file_candidates or [])
    texture_file = _sample_light_choice(texture_candidates)

    color = None
    if color_range:
        color = Gf.Vec3f(
            _sample_light_scalar(color_range.get("r"), 1.0),
            _sample_light_scalar(color_range.get("g"), 1.0),
            _sample_light_scalar(color_range.get("b"), 1.0),
        )

    for prim_path in targets:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        intensity_attr = prim.GetAttribute("inputs:intensity")
        if intensity_attr:
            intensity_attr.Set(float(intensity))
        if exposure is not None:
            exposure_attr = prim.GetAttribute("inputs:exposure")
            if exposure_attr:
                exposure_attr.Set(float(exposure))
        if color is not None:
            color_attr = prim.GetAttribute("inputs:color")
            if color_attr:
                color_attr.Set(color)
        _set_light_orientation_quat(prim, orientation_quat)
        _set_light_texture_file(prim, texture_file)


def _default_root_pose_local_from_asset(asset, env_ids: torch.Tensor) -> torch.Tensor | None:
    """Spawn pose ``[x,y,z,qx,qy,qz,qw]`` in env-local frame from asset configuration.

    Matches IsaacLab ``RigidObject`` / ``Articulation`` ``default_root_state`` (first 7).
    """
    data = getattr(asset, "data", None)
    if data is None:
        return None
    drs = as_torch(getattr(data, "default_root_state", None))
    if drs is None or drs.numel() == 0:
        return None
    return drs[env_ids, :7].clone()


def _write_root_pose_with_zero_velocity(asset, root_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write root pose and keep articulated child links coherent after teleporting the root."""
    num_envs = root_pose.shape[0]
    device = root_pose.device

    if hasattr(asset, "write_root_pose_to_sim_index"):
        asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)

        data = getattr(asset, "data", None)
        is_articulation = data is not None and hasattr(asset, "write_joint_state_to_sim") and hasattr(data, "joint_pos")
        if is_articulation:
            joint_pos = as_torch(data.joint_pos)[env_ids].clone()
            if joint_pos.numel() > 0:
                asset.write_joint_state_to_sim_index(
                    position=joint_pos, velocity=torch.zeros_like(joint_pos), env_ids=env_ids
                )
            _ = as_torch(data.body_link_pose_w)[env_ids]

        _write_root_velocity_if_dynamic(asset, torch.zeros(num_envs, 6, device=device), env_ids)
        return

    if hasattr(asset, "set_world_poses"):
        view_indices = env_ids.detach().cpu().to(torch.int32)
        asset.set_world_poses(positions=root_pose[:, :3], orientations=root_pose[:, 3:7], indices=view_indices)
        return

    raise TypeError(f"Asset {type(asset).__name__} does not support root-pose writes.")


# ---------------------------------------------------------------------------
# USD sync helper
# ---------------------------------------------------------------------------


def _sync_pose_to_usd(
    asset,
    env_ids: torch.Tensor,
    local_pose: torch.Tensor,
) -> None:
    """Write local-frame pose to USD only when simulation does not own the root.

    ``write_root_pose_to_sim`` writes to the PhysX tensor API (GPU buffers)
    but does **not** update USD prim attributes.  Tools that read via
    ``UsdGeom.XformCache`` (e.g. GraspingManager) will see stale values.

    This helper bridges the gap by writing the **local** pose (before
    env_origins offset) directly to the USD prim's xformOps.

    Args:
        asset: The IsaacLab RigidObject / Articulation.
        env_ids: Environment indices that were randomized.
        local_pose: (N, 7) tensor  [x, y, z, qx, qy, qz, qw] in the
            prim-local frame (i.e. **before** adding env_origins).
    """
    if not _should_sync_runtime_pose_to_usd(asset):
        return

    try:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()

        for i, env_idx in enumerate(env_ids):
            idx = int(env_idx)
            # Resolve prim path template → concrete path
            prim_path = asset.cfg.prim_path
            prim_path = prim_path.replace("{ENV_REGEX_NS}", f"envs/env_{idx}")
            prim_path = prim_path.replace("env_.*", f"env_{idx}")

            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue

            p = local_pose[i, :3].cpu().tolist()
            q = local_pose[i, 3:7].cpu().tolist()

            xformable = UsdGeom.Xformable(prim)
            set_existing_pose_xform_ops(
                xformable,
                position_xyz=p,
                quaternion_xyzw=q,
                gf=Gf,
                usd_geom=UsdGeom,
            )
    except Exception:
        pass  # Non-critical – simulation still works, only USD read is stale


# ---------------------------------------------------------------------------
# DR event functions
# ---------------------------------------------------------------------------


def shuffle_shelf_stock_layout(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | list,
    permute: bool = False,
    position_jitter: float = 0.0,
) -> None:
    """Reset shelf-stock assets to default shelf slots, optionally permuted and jittered."""
    if env_ids is None:
        return

    asset_cfgs = asset_cfg if isinstance(asset_cfg, list) else [asset_cfg]
    asset_cfgs = [cfg for cfg in asset_cfgs if scene_has_asset(env.scene, get_scene_entity_name(cfg))]
    if not asset_cfgs:
        return

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    slot_poses: list[torch.Tensor] = []
    assets = []
    for cfg in asset_cfgs:
        asset = env.scene[get_scene_entity_name(cfg)]
        pose = _default_root_pose_local_from_asset(asset, env_ids)
        if pose is None:
            continue
        assets.append(asset)
        slot_poses.append(pose)

    if not assets:
        return

    num_assets = len(assets)
    num_envs = len(env_ids)
    stacked_slots = torch.stack(slot_poses, dim=0)
    permutations = [
        torch.randperm(num_assets, device=env.device) if permute and num_assets > 1 else None for _ in range(num_envs)
    ]

    for asset_index, asset in enumerate(assets):
        local_pose = torch.empty_like(stacked_slots[asset_index])
        for local_env_index in range(num_envs):
            source_index = asset_index
            permutation = permutations[local_env_index]
            if permutation is not None:
                source_index = int(permutation[asset_index].item())
            local_pose[local_env_index] = stacked_slots[source_index, local_env_index]

        if position_jitter:
            jitter = torch.empty((num_envs, 2), device=env.device).uniform_(
                -float(position_jitter), float(position_jitter)
            )
            local_pose[:, :2] += jitter

        _sync_pose_to_usd(asset, env_ids, local_pose)
        sim_pose = local_pose.clone()
        sim_pose[:, :3] += env.scene.env_origins[env_ids]
        _write_root_pose_with_zero_velocity(asset, sim_pose, env_ids)


def _shelf_stock_assets(env: ManagerBasedEnv, asset_cfg: Any) -> list:
    """Resolve ``@shelf_stock`` (a SceneEntityCfg or list of them) to the present
    scene assets, skipping any that the scene does not actually contain."""
    cfgs = asset_cfg if isinstance(asset_cfg, list) else [asset_cfg]
    assets = []
    for cfg in cfgs:
        name = get_scene_entity_name(cfg)
        if name and scene_has_asset(env.scene, name):
            assets.append(env.scene[name])
    return assets


def customer_buy(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: Any = None) -> None:
    """Space-pod store flavor: a customer buys a drink → hide one visible shelf item.

    Interval event (scheduling handled by the EventManager). Best-effort narrative
    only: every branch is guarded so a term failure NEVER aborts the rollout — the
    composite scene must keep rendering regardless.
    """
    if env_ids is None or asset_cfg is None:
        return
    try:
        assets = _shelf_stock_assets(env, asset_cfg)
        if not assets:
            return
        for env_id in env_ids.tolist():
            # This flavor event has no parked-pose state; visibility is its sole owner.
            _set_asset_visibility(py_random.choice(assets), False, int(env_id))
    except Exception as exc:  # noqa: BLE001 - flavor only; never crash the rollout
        manaprint(f"WARNING: customer_buy skipped: {exc}")


def restock_shelf(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: Any = None) -> None:
    """Space-pod store flavor: restock → make one shelf item visible again.

    The visibility counterpart of :func:`customer_buy`; likewise best-effort and
    fully guarded so it can never abort the rollout.
    """
    if env_ids is None or asset_cfg is None:
        return
    try:
        assets = _shelf_stock_assets(env, asset_cfg)
        if not assets:
            return
        for env_id in env_ids.tolist():
            # Counterpart to customer_buy: restore its visibility-owned state.
            _set_asset_visibility(py_random.choice(assets), True, int(env_id))
    except Exception as exc:  # noqa: BLE001 - flavor only; never crash the rollout
        manaprint(f"WARNING: restock_shelf skipped: {exc}")


def cycle_daylight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    prim_paths: list[str] | None = None,
    period_s: float = 60.0,
    intensity_range: list[float] | tuple[float, float] = (400.0, 1400.0),
) -> None:
    """Space-pod flavor: drive dome-light intensity along a day/night cycle.

    Maps the env's elapsed sim time (mod ``period_s``) onto a cosine day curve so
    the booth visibly brightens toward "noon" and dims toward "night". Best-effort
    and fully guarded — a failure leaves the lighting untouched, never crashing.
    """
    del env_ids
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        targets = list(prim_paths or []) or _discover_light_prim_paths(stage)
        if not targets:
            return
        elapsed = float(getattr(getattr(env, "sim", None), "current_time", 0.0) or 0.0)
        period = max(float(period_s), 1e-3)
        phase = (elapsed % period) / period
        lo, hi = float(intensity_range[0]), float(intensity_range[1])
        day = 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)  # 0 at night, 1 at noon
        intensity = lo + (hi - lo) * day
        for prim_path in targets:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            attr = prim.GetAttribute("inputs:intensity")
            if attr:
                attr.Set(float(intensity))
    except Exception as exc:  # noqa: BLE001 - flavor only; never crash the rollout
        manaprint(f"WARNING: cycle_daylight skipped: {exc}")


# Letter a-z → ids 1..26; digit 0-9 → ids 27..36 (block_alphabet_{token}).
_ALPHABET_DIGIT_ID_BASE = 27


def alphabet_token_id(token: str) -> int:
    """Map one alphabet-word glyph (``a``-``z`` or ``0``-``9``) to a stable int id."""
    raw = str(token).strip()
    if len(raw) != 1 or not raw.isascii():
        raise ValueError(f"alphabet token must be a single ASCII letter or digit, got {token!r}")
    if raw.isalpha():
        return ord(raw.lower()) - ord("a") + 1
    if raw.isdigit():
        return _ALPHABET_DIGIT_ID_BASE + int(raw)
    raise ValueError(f"alphabet token must be a letter or digit, got {token!r}")


def alphabet_token_from_id(letter_id: int) -> str:
    """Inverse of :func:`alphabet_token_id`."""
    value = int(letter_id)
    if 1 <= value <= 26:
        return chr(ord("a") + value - 1)
    if _ALPHABET_DIGIT_ID_BASE <= value <= _ALPHABET_DIGIT_ID_BASE + 9:
        return str(value - _ALPHABET_DIGIT_ID_BASE)
    raise ValueError(f"unknown alphabet letter id {letter_id!r}")


def alphabet_token_catalog() -> list[str]:
    """Ordered glyph catalog matching letter-id index ``letter_id - 1``."""
    return [chr(ord("a") + offset) for offset in range(26)] + [str(digit) for digit in range(10)]


def _normalize_word_bank(word_bank: list[str] | tuple[str, ...]) -> list[str]:
    words = []
    for raw in word_bank or []:
        word = str(raw).strip().lower()
        if not word:
            continue
        if not word.isascii() or not all(ch.isalpha() or ch.isdigit() for ch in word):
            raise ValueError(f"sample_alphabet_word word_bank entries must be ASCII letters/digits, got {raw!r}")
        if len(set(word)) != len(word):
            raise ValueError(
                f"sample_alphabet_word cannot use repeated letters with one physical block per letter: {word!r}"
            )
        words.append(word)
    if not words:
        raise ValueError("sample_alphabet_word requires a non-empty word_bank")
    return words


def _sample_region_pose(
    rng: random.Random,
    region: dict[str, Any],
    *,
    occupied_xy: list[tuple[float, float]],
    min_separation: float,
) -> tuple[list[float], float]:
    x_range = region.get("x")
    y_range = region.get("y")
    z_range = region.get("z", [0.0, 0.0])
    if not isinstance(x_range, (list, tuple)) or len(x_range) < 2:
        raise ValueError("sample_alphabet_word region.x must be [min, max]")
    if not isinstance(y_range, (list, tuple)) or len(y_range) < 2:
        raise ValueError("sample_alphabet_word region.y must be [min, max]")
    if not isinstance(z_range, (list, tuple)) or len(z_range) < 2:
        raise ValueError("sample_alphabet_word region.z must be [min, max]")

    min_sep_sq = float(min_separation) * float(min_separation)
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    z0, z1 = float(z_range[0]), float(z_range[1])
    for _attempt in range(2000):
        x = rng.uniform(x0, x1)
        y = rng.uniform(y0, y1)
        if all((x - ox) * (x - ox) + (y - oy) * (y - oy) >= min_sep_sq for ox, oy in occupied_xy):
            z = rng.uniform(z0, z1)
            occupied_xy.append((x, y))
            return [x, y, z], rng.uniform(-3.141592653589793, 3.141592653589793)

    # Region too tight for hard separation: place as far as possible from peers.
    best_xy = (rng.uniform(x0, x1), rng.uniform(y0, y1))
    best_score = -1.0
    for _ in range(500):
        x = rng.uniform(x0, x1)
        y = rng.uniform(y0, y1)
        score = min(((x - ox) * (x - ox) + (y - oy) * (y - oy) for ox, oy in occupied_xy), default=float("inf"))
        if score > best_score:
            best_score = score
            best_xy = (x, y)
    occupied_xy.append(best_xy)
    return [best_xy[0], best_xy[1], rng.uniform(z0, z1)], rng.uniform(-3.141592653589793, 3.141592653589793)


def _set_asset_visibility(asset: Any, visible: bool, env_id: int) -> None:
    setter = getattr(asset, "set_visibility", None)
    if callable(setter):
        setter(bool(visible), env_ids=[int(env_id)])


def _set_pose_driven_asset_visibility(asset: Any, visible: bool, env_id: int) -> None:
    """Author visibility only when simulation pose does not own activation."""
    if not _should_author_pose_owned_visibility_to_usd(asset):
        return
    _set_asset_visibility(asset, visible, env_id)


def _write_single_env_local_pose(env: ManagerBasedEnv, asset: Any, env_id: int, local_pose: torch.Tensor) -> None:
    env_ids = torch.as_tensor([int(env_id)], dtype=torch.long, device=env.device)
    local_pose = local_pose.reshape(1, 7).to(device=env.device)
    _sync_pose_to_usd(asset, env_ids, local_pose)
    sim_pose = local_pose.clone()
    sim_pose[:, :3] += env.scene.env_origins[env_ids]
    _write_root_pose_with_zero_velocity(asset, sim_pose, env_ids)


def sample_color_classification(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    colors: list[str],
    object_prefix: str = "classify_",
    destination_prefix: str = "paper_",
    region: dict[str, Any] | None = None,
    min_separation: float = 0.05,
    object_orientation: list[float] | None = None,
    away_z: float = -10.0,
    object_log_key: str = "classify_object_indices",
    color_log_key: str = "classify_color_indices",
    seed: int | None = None,
) -> None:
    """Expose one fixed-orientation cached object for every configured color.

    Language instruction is owned by ``instruction_component`` (static
    ``get_prompt()`` fallback); this event does not write the active store.
    """
    if env_ids is None:
        return
    if region is None:
        raise ValueError("sample_color_classification requires region={x: [...], y: [...], z: [...]}")
    normalized_colors = [str(color).strip().lower() for color in colors if str(color).strip()]
    if not normalized_colors:
        raise ValueError("sample_color_classification requires at least one color")
    cache_size = _resolve_shared_color_cache_size(env, normalized_colors, object_prefix=object_prefix)
    orientation = [float(value) for value in (object_orientation or [0.0, 0.0, 0.0, 1.0])]
    if len(orientation) != 4:
        raise ValueError("sample_color_classification object_orientation must be an XYZW quaternion")

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = getattr(env, "device", ids.device)
    object_indices = torch.zeros((num_envs, len(normalized_colors)), dtype=torch.int64, device=device)
    color_indices = torch.zeros((num_envs, len(normalized_colors)), dtype=torch.int64, device=device)

    state_attr = "_mana_color_classification_state"
    state = getattr(env, state_attr, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, state_attr, state)
    labels: dict[int, str] = state.setdefault("label", {})

    all_object_names = [
        pool_cache_slot_name(f"{object_prefix}{color}", slot)
        for color in normalized_colors
        for slot in range(int(cache_size))
    ]
    scene_assets = {name: env.scene[name] for name in all_object_names}
    slot_activation: dict[str, dict[int, bool]] = {}
    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, f"{object_log_key}:{color_log_key}", seed, env_id=env_id)
        sampled_objects = [
            (color_index, rng.randrange(int(cache_size))) for color_index in range(len(normalized_colors))
        ]

        for parking_index, logical_name in enumerate(all_object_names):
            asset = scene_assets[logical_name]
            parked = torch.zeros(7, dtype=torch.float32, device=device)
            parked[:3] = torch.as_tensor(
                inactive_pool_slot_position(parking_index, isolation_z=away_z),
                dtype=torch.float32,
                device=device,
            )
            parked[6] = 1.0
            _set_pose_driven_asset_visibility(asset, False, env_id)
            _write_single_env_local_pose(env, asset, env_id, parked)
            slot_activation.setdefault(logical_name, {})[env_id] = False

        occupied_xy: list[tuple[float, float]] = []
        for stage_index, (color_index, cache_slot) in enumerate(sampled_objects):
            object_id = f"{object_prefix}{normalized_colors[color_index]}"
            logical_name = pool_cache_slot_name(object_id, cache_slot)
            asset = scene_assets[logical_name]
            position, _ = _sample_region_pose(
                rng,
                region,
                occupied_xy=occupied_xy,
                min_separation=min_separation,
            )
            pose = torch.zeros(7, dtype=torch.float32, device=device)
            pose[:3] = torch.as_tensor(position, dtype=torch.float32, device=device)
            pose[3:7] = torch.as_tensor(orientation, dtype=torch.float32, device=device)
            _set_pose_driven_asset_visibility(asset, True, env_id)
            _write_single_env_local_pose(env, asset, env_id, pose)
            slot_activation.setdefault(logical_name, {})[env_id] = True
            # Overlay Active assets reads schema-vNext slot state; this custom
            # sampler bypasses runtime_sample, so record the chosen slot here.
            record_active_asset_slot(
                env,
                plan_id="runtime",
                env_id=env_id,
                object_id=object_id,
                active_slot=logical_name,
                asset_id=resolve_pool_slot_asset_id(env, object_id, logical_name),
            )
            logical_index = all_object_names.index(logical_name)
            object_indices[env_id, stage_index] = logical_index + 1
            color_indices[env_id, stage_index] = color_index + 1

        for color in normalized_colors:
            paper = env.scene[f"{destination_prefix}{color}"]
            _set_pose_driven_asset_visibility(paper, True, env_id)

        labels[env_id] = f"Targets: {', '.join(normalized_colors)}"

    apply_runtime_slot_activation(env, scene_assets, slot_activation)
    write_committed(env, str(object_log_key), object_indices, env_ids=ids, frame_idx=-1)
    write_committed(env, str(color_log_key), color_indices, env_ids=ids, frame_idx=-1)


def _resolve_shared_color_cache_size(env: ManagerBasedEnv, colors: list[str], *, object_prefix: str) -> int:
    sizes = {color: resolve_pool_cache_size(env, f"{object_prefix}{color}") for color in colors}
    unique = set(sizes.values())
    if len(unique) != 1:
        raise ValueError(
            "sample_color_classification requires a shared scene cache_size across colors; " f"got {sizes}"
        )
    return int(next(iter(unique)))


def sample_alphabet_word(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    word_bank: list[str],
    letters_log_key: str = "target_word_letters",
    region: dict[str, Any] | None = None,
    instruction_template: str = "{word}",
    word_log_key: str = "target_word",
    block_prefix: str = "block_alphabet_",
    slot_marker_prefix: str | None = None,
    slot_marker_count: int = 0,
    slot_marker_center: list[float] | None = None,
    slot_marker_spacing: float = 0.16,
    slot_marker_orientation: list[float] | None = None,
    distractor_count: int = 5,
    min_separation: float = 0.11,
    block_yaw: float | None = None,
    away_z: float = -10.0,
    seed: int | None = None,
    exclude_repeat: bool = True,
) -> None:
    """Sample one target word per env and materialize only its letters plus distractors.

    ``block_yaw`` fixes every visible block's Z-axis rotation (radians) instead of
    sampling a random yaw; e.g. ``math.pi`` locks blocks to a 180-degree yaw.
    """
    if env_ids is None:
        return
    if region is None:
        raise ValueError("sample_alphabet_word requires region={x: [...], y: [...], z: [...]}")

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    words = _normalize_word_bank(word_bank)
    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = getattr(env, "device", ids.device)

    state = getattr(env, _ALPHABET_WORD_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _ALPHABET_WORD_STATE_ATTR, state)

    previous: dict[int, str] = state.setdefault("previous", {})
    words_by_env: dict[int, str] = state.setdefault("word", {})
    letters_by_env: dict[int, list[int]] = state.setdefault("letters", {})
    text_by_env: dict[int, str] = state.setdefault("instruction", {})
    slot_centers_by_env: dict[int, list[list[float]]] = state.setdefault("slot_centers", {})
    state["word_log_key"] = str(word_log_key)
    state["letters_log_key"] = str(letters_log_key)

    letter_tokens = [chr(ord("a") + offset) for offset in range(26)]
    digit_tokens = [str(digit) for digit in range(10)]
    all_tokens = letter_tokens + [
        token for token in digit_tokens if scene_has_asset(env.scene, f"{block_prefix}{token}")
    ]
    letters_tensor = torch.zeros((num_envs, max(len(word) for word in words)), dtype=torch.int64, device=device)

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, f"{letters_log_key}:{word_log_key}", seed, env_id=env_id)
        choices = words
        if exclude_repeat and len(words) > 1:
            prev = previous.get(env_id)
            choices = [word for word in words if word != prev] or words
        word = rng.choice(choices)
        previous[env_id] = word
        encoded = [alphabet_token_id(token) for token in word]
        letters_tensor[env_id, : len(encoded)] = torch.as_tensor(encoded, dtype=torch.int64, device=device)

        instruction = str(instruction_template).format(word=word.upper())
        write_active_instruction(env, env_id, instruction, num_envs=num_envs, reset_owned=True)
        words_by_env[env_id] = word
        letters_by_env[env_id] = encoded
        text_by_env[env_id] = instruction

        if slot_marker_prefix:
            # Tabletop center z is -0.240423; lift by 5mm so thin paper_green
            # markers sit just above the surface instead of sinking into it.
            center = slot_marker_center or [0.02, -0.268, -0.235423]
            orientation = slot_marker_orientation or [0.0, 0.0, 0.70711, 0.70711]
            active_slot_centers: list[list[float]] = []
            for slot_idx in range(1, max(0, int(slot_marker_count)) + 1):
                name = f"{slot_marker_prefix}{slot_idx}"
                if scene_has_asset(env.scene, name):
                    asset = env.scene[name]
                    visible = slot_idx <= len(word)
                    _set_pose_driven_asset_visibility(asset, visible, env_id)
                    local_pose = torch.zeros(7, dtype=torch.float32, device=device)
                    if visible:
                        slot_y = float(center[1]) + (0.5 * (len(word) - 1) - (slot_idx - 1)) * float(
                            slot_marker_spacing
                        )
                        slot_position = [float(center[0]), slot_y, float(center[2])]
                        active_slot_centers.append(slot_position)
                        local_pose[:3] = torch.as_tensor(slot_position, dtype=torch.float32, device=device)
                        local_pose[3:7] = torch.as_tensor(orientation, dtype=torch.float32, device=device)
                    else:
                        local_pose[:3] = torch.as_tensor(
                            [float(center[0]), float(center[1]), float(away_z)], dtype=torch.float32, device=device
                        )
                        local_pose[6] = 1.0
                    _write_single_env_local_pose(env, asset, env_id, local_pose)
            slot_centers_by_env[env_id] = active_slot_centers

        target_tokens = list(dict.fromkeys(word))
        # Distractors stay alphabetic so digit blocks only appear when the word needs them.
        distractor_pool = [token for token in letter_tokens if token not in set(target_tokens)]
        rng.shuffle(distractor_pool)
        visible_tokens = target_tokens + distractor_pool[: max(0, int(distractor_count))]
        visible_set = set(visible_tokens)
        occupied_xy: list[tuple[float, float]] = []

        for token in all_tokens:
            name = f"{block_prefix}{token}"
            if not scene_has_asset(env.scene, name):
                continue
            asset = env.scene[name]
            visible = token in visible_set
            _set_pose_driven_asset_visibility(asset, visible, env_id)
            local_pose = torch.zeros(7, dtype=torch.float32, device=device)
            if visible:
                position, yaw = _sample_region_pose(rng, region, occupied_xy=occupied_xy, min_separation=min_separation)
                if block_yaw is not None:
                    yaw = float(block_yaw)
                local_pose[:3] = torch.as_tensor(position, dtype=torch.float32, device=device)
                half_yaw = torch.tensor(0.5 * yaw, dtype=torch.float32, device=device)
                local_pose[5] = torch.sin(half_yaw)
                local_pose[6] = torch.cos(half_yaw)
            else:
                default_pose = _default_root_pose_local_from_asset(asset, torch.as_tensor([env_id], device=device))
                if default_pose is not None:
                    local_pose[:] = default_pose[0]
                else:
                    local_pose[6] = 1.0
                local_pose[2] = float(away_z)
            _write_single_env_local_pose(env, asset, env_id, local_pose)

    write_committed(env, str(letters_log_key), letters_tensor, env_ids=ids, frame_idx=-1)


_BUTTONS_RANDOM_ORDER_STATE_ATTR = "_mana_buttons_random_order_state"


def _normalize_button_order_bank(
    order_bank: list[Any] | tuple[Any, ...] | None,
    *,
    colors: list[str],
    num_buttons: int,
    button_prefix: str = "button_",
) -> list[list[int]] | None:
    """Normalize ``order_bank`` entries to color-index sequences into ``colors``.

    Each entry may be a list/tuple of color names or a comma/space-separated
    string. Returns ``None`` when ``order_bank`` is omitted so callers can fall
    back to sampling any permutation of ``num_buttons`` colors.
    """
    if order_bank is None:
        return None
    color_to_idx = {color: idx for idx, color in enumerate(colors)}
    normalized: list[list[int]] = []
    for raw in order_bank:
        if isinstance(raw, str):
            parts = [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
        elif isinstance(raw, (list, tuple)):
            parts = [str(part).strip() for part in raw if str(part).strip()]
        else:
            raise ValueError(f"sample_button_order order_bank entries must be lists or strings, got {raw!r}")
        if len(parts) != num_buttons:
            raise ValueError(f"sample_button_order order_bank entry must have length {num_buttons}, got {parts!r}")
        indices: list[int] = []
        for part in parts:
            color = part.lower().replace(button_prefix, "")
            if color not in color_to_idx:
                raise ValueError(f"sample_button_order order_bank color {part!r} is not in button_colors={colors}")
            indices.append(color_to_idx[color])
        if len(set(indices)) != len(indices):
            raise ValueError(f"sample_button_order order_bank entry has duplicate colors: {parts!r}")
        normalized.append(indices)
    if not normalized:
        raise ValueError("sample_button_order order_bank must be non-empty when provided")
    return normalized


def sample_button_order(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    num_buttons: int,
    button_colors: list[str],
    button_prefix: str = "button_",
    order_log_key: str = "button_press_order",
    order_log_width: int | None = None,
    order_bank: list[Any] | tuple[Any, ...] | None = None,
    require_order_bank: bool = True,
    instruction_template: str = "Press the buttons in this order: {order}, then return the arm home.",
    region: dict[str, Any] | None = None,
    min_separation: float = 0.15,
    away_z: float = -10.0,
    seed: int | None = None,
) -> None:
    """Sample a button press order per env and scatter the buttons.

    When ``order_bank`` is set, each reset picks one press-order sequence from
    that bank. By default this term requires an explicit bank so a published
    benchmark cannot silently expand to every permutation. Set
    ``require_order_bank: false`` only for exploratory tasks that deliberately
    want unconstrained permutation sampling. Active buttons are placed at random
    non-overlapping tabletop positions and inactive ones are parked below the
    table. The per-position color index is committed to
    ``order_log_key`` (1-based index into ``button_colors``, 0 padded) for the
    success term, and the order is written into the per-env instruction plus
    the overlay state.
    """
    if env_ids is None:
        return
    if region is None:
        raise ValueError("sample_button_order requires region={x: [...], y: [...], z: [...]}")

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    colors = [str(color).strip().lower().replace(button_prefix, "") for color in (button_colors or []) if color]
    if not colors:
        raise ValueError("sample_button_order requires a non-empty button_colors")
    num_buttons = int(num_buttons)
    if num_buttons < 1 or num_buttons > len(colors):
        raise ValueError(f"sample_button_order num_buttons must be in [1, {len(colors)}], got {num_buttons}")

    width = int(order_log_width) if order_log_width else len(colors)
    if width < num_buttons:
        raise ValueError("sample_button_order order_log_width must be >= num_buttons")
    bank = _normalize_button_order_bank(order_bank, colors=colors, num_buttons=num_buttons, button_prefix=button_prefix)
    if bank is None and require_order_bank:
        raise ValueError(
            "sample_button_order requires order_bank by default to preserve the published benchmark set. "
            "Set require_order_bank: false only for deliberate unconstrained sampling."
        )

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = getattr(env, "device", ids.device)

    state = getattr(env, _BUTTONS_RANDOM_ORDER_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _BUTTONS_RANDOM_ORDER_STATE_ATTR, state)
    order_text_by_env: dict[int, str] = state.setdefault("order_text", {})

    order_tensor = torch.zeros((num_envs, width), dtype=torch.int64, device=device)

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(order_log_key), seed, env_id=env_id)
        if bank is not None:
            press_order = list(rng.choice(bank))
        else:
            # random.sample draws a distinct subset already in random order.
            press_order = rng.sample(range(len(colors)), num_buttons)
        active = set(press_order)
        order_tensor[env_id, :num_buttons] = torch.as_tensor(
            [idx + 1 for idx in press_order], dtype=torch.int64, device=device
        )

        occupied_xy: list[tuple[float, float]] = []
        for canon_idx, color in enumerate(colors):
            name = f"{button_prefix}{color}"
            if not scene_has_asset(env.scene, name):
                continue
            asset = env.scene[name]
            visible = canon_idx in active
            _set_pose_driven_asset_visibility(asset, visible, env_id)
            local_pose = torch.zeros(7, dtype=torch.float32, device=device)
            if visible:
                position, _yaw = _sample_region_pose(
                    rng, region, occupied_xy=occupied_xy, min_separation=min_separation
                )
                local_pose[:3] = torch.as_tensor(position, dtype=torch.float32, device=device)
                # Buttons stay upright (identity orientation); they are pressed
                # from the top so yaw is irrelevant.
                local_pose[6] = 1.0
            else:
                default_pose = _default_root_pose_local_from_asset(asset, torch.as_tensor([env_id], device=device))
                if default_pose is not None:
                    local_pose[:] = default_pose[0]
                else:
                    local_pose[6] = 1.0
                local_pose[2] = float(away_z)
            _write_single_env_local_pose(env, asset, env_id, local_pose)

        order_words = [colors[idx] for idx in press_order]
        instruction = str(instruction_template).format(order=", ".join(order_words))
        write_active_instruction(env, env_id, instruction, num_envs=num_envs, reset_owned=True)
        order_text_by_env[env_id] = " -> ".join(word.upper() for word in order_words)
        print(f"[buttons_random_order] env={env_id} instruction={instruction!r}")

    write_committed(env, str(order_log_key), order_tensor, env_ids=ids, frame_idx=-1)


_BUTTONS_PRESS_SEQUENCE_STATE_ATTR = "_mana_buttons_press_sequence_state"


def sample_button_press_sequence(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    sequence_length: int,
    button_colors: list[str],
    button_prefix: str = "button_",
    order_log_key: str = "button_press_order",
    order_log_width: int | None = None,
    instruction_template: str = "Press the buttons in this order: {order}, then return the arm home.",
    region: dict[str, Any] | None = None,
    min_separation: float = 0.15,
    seed: int | None = None,
) -> None:
    """Sample a with-replacement button press sequence and scatter all buttons.

    Unlike :func:`sample_button_order` (distinct subset, no replacement), this
    draws ``sequence_length`` colors via ``rng.choices`` so repeats are allowed
    and length may exceed the number of physical buttons. All buttons stay on
    the table every episode; only XY is re-sampled under ``min_separation``.

    Commits 1-based color indices (into ``button_colors``) to ``order_log_key``
    and stores ``sequence_names`` on ``_mana_buttons_press_sequence_state``.
    Identity orientation is xyzw ``[0, 0, 0, 1]``.
    """
    if env_ids is None:
        return
    if region is None:
        raise ValueError("sample_button_press_sequence requires region={x: [...], y: [...], z: [...]}")

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    colors = [str(color).strip().lower().replace(button_prefix, "") for color in (button_colors or []) if color]
    if not colors:
        raise ValueError("sample_button_press_sequence requires a non-empty button_colors")
    length = int(sequence_length)
    if length < 1:
        raise ValueError(f"sample_button_press_sequence sequence_length must be >= 1, got {length}")

    width = int(order_log_width) if order_log_width else length
    if width < length:
        raise ValueError("sample_button_press_sequence order_log_width must be >= sequence_length")

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = getattr(env, "device", ids.device)

    state = getattr(env, _BUTTONS_PRESS_SEQUENCE_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _BUTTONS_PRESS_SEQUENCE_STATE_ATTR, state)
    order_text_by_env: dict[int, str] = state.setdefault("order_text", {})
    sequence_names_by_env: dict[int, list[str]] = state.setdefault("sequence_names", {})

    order_tensor = torch.zeros((num_envs, width), dtype=torch.int64, device=device)

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(order_log_key), seed, env_id=env_id)
        press_order = rng.choices(range(len(colors)), k=length)
        order_tensor[env_id, :length] = torch.as_tensor(
            [idx + 1 for idx in press_order], dtype=torch.int64, device=device
        )

        occupied_xy: list[tuple[float, float]] = []
        for color in colors:
            name = f"{button_prefix}{color}"
            if not scene_has_asset(env.scene, name):
                continue
            asset = env.scene[name]
            _set_pose_driven_asset_visibility(asset, True, env_id)
            position, _yaw = _sample_region_pose(rng, region, occupied_xy=occupied_xy, min_separation=min_separation)
            local_pose = torch.zeros(7, dtype=torch.float32, device=device)
            local_pose[:3] = torch.as_tensor(position, dtype=torch.float32, device=device)
            # Buttons stay upright (identity xyzw); they are pressed from the top.
            local_pose[6] = 1.0
            _write_single_env_local_pose(env, asset, env_id, local_pose)

        order_words = [colors[idx] for idx in press_order]
        sequence_names_by_env[env_id] = [f"{button_prefix}{word}" for word in order_words]
        instruction = str(instruction_template).format(order=", ".join(order_words))
        write_active_instruction(env, env_id, instruction, num_envs=num_envs, reset_owned=True)
        order_text_by_env[env_id] = " -> ".join(word.upper() for word in order_words)
        print(f"[buttons_press_sequence] env={env_id} instruction={instruction!r}")

    write_committed(env, str(order_log_key), order_tensor, env_ids=ids, frame_idx=-1)


def _set_joint_openness(
    env: ManagerBasedEnv,
    articulation: Any,
    joint_index: int,
    openness: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    """Write a per-env normalized openness [0, 1] to one revolute joint.

    Inverse of the Arena ``get_normalized_joint_position`` convention used by
    ``manaenv.terms.terminations._normalized_joint_position_openness``: openness 0
    is the fully-closed limit and 1 the fully-open one, with the ``j_min < 0``
    branch flipping the mapping so both directions agree with the success checker.
    """
    limits = as_torch(articulation.data.joint_pos_limits)[0, joint_index, :]
    j_min = float(limits[0].item())
    j_max = float(limits[1].item())
    raw = 1.0 - openness if j_min < 0.0 else openness
    theta = (j_min + raw * (j_max - j_min)).reshape(-1, 1)
    # PhysX Warp joint kernels expect int32 selectors (int64 fails at wp.launch).
    joint_ids = torch.tensor([int(joint_index)], device=env.device, dtype=torch.int32)
    zero_vel = torch.zeros((env_ids.shape[0], 1), device=env.device)
    # The knob joint has no drive/actuator, so a target alone does not move it;
    # write the joint state directly. The drive target is set too so a PD-actuated
    # variant would hold (no-op otherwise).
    articulation.write_joint_state_to_sim(theta, zero_vel, joint_ids=joint_ids, env_ids=env_ids)
    articulation.set_joint_position_target(theta, joint_ids=joint_ids, env_ids=env_ids)


def sample_knob_temperature(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    level_labels: list[str],
    object_name: str = "knob_a",
    joint_name: str = "RevoluteJoint",
    target_log_key: str = "target_temperature_level",
    init_log_key: str = "init_temperature_level",
    instruction_template: str = "Rotate the knob to adjust the temperature from {from_temp} to {to_temp}.",
    seed: int | None = None,
) -> None:
    """Sample a distinct (initial, target) temperature per env at reset.

    The knob's discrete display marks are modeled as ``num_levels = len(level_labels)``
    equal bins over the joint's full normalized travel. Each reset env draws an
    initial level and a distinct target level, the knob joint is set onto the
    initial level's dial mark, and the per-env instruction is rendered from the two
    labels. The target/initial level indices are committed to the live log store so
    the success checker (:func:`manaenv.terms.terminations.knob_at_target_level`) and
    the overlay/log labels read the same source of truth.
    """
    if env_ids is None:
        return

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    labels = [str(item) for item in (level_labels or [])]
    num_levels = len(labels)
    if num_levels < 2:
        raise ValueError("sample_knob_temperature requires at least 2 level_labels (initial and target must differ)")
    if not scene_has_asset(env.scene, object_name):
        return

    articulation = env.scene[object_name]
    joint_ids, _ = articulation.find_joints([joint_name])
    if not joint_ids:
        return
    joint_index = int(joint_ids[0])

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    init_tensor = torch.zeros(num_envs, dtype=torch.int64, device=device)
    target_tensor = torch.zeros(num_envs, dtype=torch.int64, device=device)
    openness = torch.zeros(ids.shape[0], dtype=torch.float32, device=device)

    for slot, env_id_t in enumerate(ids):
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(target_log_key), seed, env_id=env_id)
        init_idx = rng.randrange(num_levels)
        # Draw the target from the remaining levels so it always differs from init.
        target_idx = rng.randrange(num_levels - 1)
        if target_idx >= init_idx:
            target_idx += 1
        init_tensor[env_id] = init_idx
        target_tensor[env_id] = target_idx
        # Land exactly on the level's dial mark (marks evenly spaced over the
        # joint's full travel at i / num_levels), not the bin center, so the
        # knob's printed dial reads the sampled temperature.
        openness[slot] = init_idx / num_levels
        text = str(instruction_template).format(from_temp=labels[init_idx], to_temp=labels[target_idx])
        write_active_instruction(env, env_id, text, num_envs=num_envs, reset_owned=True)

    _set_joint_openness(env, articulation, joint_index, openness, ids)
    write_committed(env, str(target_log_key), target_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(init_log_key), init_tensor, env_ids=ids, frame_idx=-1)


_PRESS_BUTTON_N_TIMES_STATE_ATTR = "_mana_press_button_n_times_state"


def sample_press_button_n_times(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    object_name: str = "pump_bottle",
    slot_poses: list[list[float]] | None = None,
    count_choices: list[int] | None = None,
    count_log_key: str = "target_press_count",
    slot_log_key: str = "bottle_slot",
    orientation: list[float] | None = None,
    instruction_template: str = ("Hold the pump bottle steady and press the pump head {n} times, then return home."),
    seed: int | None = None,
) -> None:
    """Sample a discrete bottle slot and press-count target per env at reset.

    ``slot_poses`` are env-local ``[x, y, z]`` placements (world frame of a single
    env origin). Each reset draws a slot uniformly and a count from
    ``count_choices``, writes the bottle root pose, commits both scalars to the
    log store, and renders the per-env instruction.
    """
    if env_ids is None:
        return

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    poses = list(slot_poses or [])
    if not poses:
        raise ValueError("sample_press_button_n_times requires a non-empty slot_poses list")
    counts = [int(c) for c in (count_choices or [3, 4, 5, 6])]
    if not counts:
        raise ValueError("sample_press_button_n_times requires a non-empty count_choices list")
    if any(c < 1 for c in counts):
        raise ValueError(f"sample_press_button_n_times count_choices must be >= 1, got {counts}")
    if not scene_has_asset(env.scene, object_name):
        return

    quat = [float(v) for v in (orientation or [0.0, 0.0, 0.0, 1.0])]
    if len(quat) != 4:
        raise ValueError(f"sample_press_button_n_times orientation must be XYZW length 4, got {quat!r}")

    normalized_poses: list[list[float]] = []
    for raw in poses:
        vals = [float(v) for v in raw]
        if len(vals) == 3:
            normalized_poses.append(vals + quat)
        elif len(vals) == 7:
            normalized_poses.append(vals)
        else:
            raise ValueError(
                "sample_press_button_n_times slot_poses entries must be [x,y,z] or " f"[x,y,z,qx,qy,qz,qw], got {raw!r}"
            )

    asset = env.scene[object_name]
    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    state = getattr(env, _PRESS_BUTTON_N_TIMES_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _PRESS_BUTTON_N_TIMES_STATE_ATTR, state)
    count_text_by_env: dict[int, str] = state.setdefault("count_text", {})
    slot_text_by_env: dict[int, str] = state.setdefault("slot_text", {})
    slot_labels = ("left_mid", "center_mid", "right_mid")

    count_tensor = torch.zeros(num_envs, dtype=torch.int64, device=device)
    slot_tensor = torch.zeros(num_envs, dtype=torch.int64, device=device)

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(count_log_key), seed, env_id=env_id)
        slot_idx = rng.randrange(len(normalized_poses))
        count = counts[rng.randrange(len(counts))]
        slot_tensor[env_id] = slot_idx
        count_tensor[env_id] = count

        local_pose = torch.as_tensor(normalized_poses[slot_idx], dtype=torch.float32, device=device)
        _write_single_env_local_pose(env, asset, env_id, local_pose)

        instruction = str(instruction_template).format(n=count)
        write_active_instruction(env, env_id, instruction, num_envs=num_envs, reset_owned=True)
        count_text_by_env[env_id] = str(count)
        slot_text_by_env[env_id] = slot_labels[slot_idx] if slot_idx < len(slot_labels) else str(slot_idx)
        manaprint(
            f"INFO: [press_button_n_times] env={env_id} slot={slot_text_by_env[env_id]} " f"instruction={instruction!r}"
        )

    write_committed(env, str(count_log_key), count_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(slot_log_key), slot_tensor, env_ids=ids, frame_idx=-1)


_BALANCE_STATE_ATTR = "_mana_balance_state"
_BALANCE_POOL_ATTR = "_mana_balance_problem_pool"


def _balance_problem_pool(
    env: ManagerBasedEnv,
    denoms: list[int],
    count_per_denom: int,
    add_count: int,
    final_total_cap: int,
    same_target_pan: bool,
    empty_target_denoms: bool,
) -> list[dict[str, Any]]:
    cache = getattr(env, _BALANCE_POOL_ATTR, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(env, _BALANCE_POOL_ATTR, cache)
    key = (
        tuple(denoms),
        int(count_per_denom),
        int(add_count),
        int(final_total_cap),
        bool(same_target_pan),
        bool(empty_target_denoms),
    )
    if key not in cache:
        cache[key] = enumerate_balance_problems(
            denoms,
            count_per_denom,
            add_count,
            final_total_cap,
            same_target_pan=same_target_pan,
            empty_target_denoms=empty_target_denoms,
        )
    return cache[key]


def _pan_slot_positions(center: list[float], count: int, spacing: float) -> list[list[float]]:
    """Evenly spread ``count`` weight slots along table X across a pan center."""
    if count <= 0:
        return []
    return [
        [float(center[0]) + (i - 0.5 * (count - 1)) * float(spacing), float(center[1]), float(center[2])]
        for i in range(count)
    ]


def _select_unique_balance_distractors(
    denoms: list[int],
    add_targets: list[tuple[int, int]],
    candidates: list[tuple[int, str]],
    distractor_count: int,
    rng: random.Random,
) -> list[str]:
    """Select as many distractors as possible without creating another mass-matching denomination multiset."""
    target_counts = [0] * len(denoms)
    for denom_index, _ in add_targets:
        target_counts[denom_index] += 1
    target_mass = sum(denom * count for denom, count in zip(denoms, target_counts))

    def is_unique(combo: tuple[tuple[int, str], ...]) -> bool:
        available_counts = list(target_counts)
        for denom_index, _ in combo:
            available_counts[denom_index] += 1
        for counts in itertools.product(*(range(count + 1) for count in available_counts)):
            if list(counts) == target_counts:
                continue
            if sum(denom * count for denom, count in zip(denoms, counts)) == target_mass:
                return False
        return True

    max_count = min(max(0, int(distractor_count)), len(candidates))
    for count in range(max_count, -1, -1):
        options = list(itertools.combinations(candidates, count))
        rng.shuffle(options)
        for combo in options:
            if is_unique(combo):
                return [name for _, name in combo]
    return []


def sample_balance_problem(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    denoms: list[int],
    count_per_denom: int = 2,
    add_count: int = 1,
    final_total_cap: int = 4,
    problem_limit: int | list[Any] | tuple[Any, ...] | None = None,
    same_target_pan: bool = False,
    empty_target_denoms: bool = False,
    weight_prefix: str = "weight_",
    weight_orientation: list[float] | None = None,
    left_pan_pos: list[float] | None = None,
    right_pan_pos: list[float] | None = None,
    pan_slot_spacing: float = 0.045,
    pile_region: dict[str, Any] | None = None,
    distractor_count: int = 2,
    unique_solution: bool = False,
    min_separation: float = 0.05,
    denom_log_key: str = "balance_add_denom",
    pan_log_key: str = "balance_add_pan",
    instruction_template: str = ("Add weights from the pile to balance the two-pan scale, then return the arm home."),
    detailed_instruction_template: str | None = (
        "Add weights from the pile to balance the scale: {plan}. Then return the arm home."
    ),
    away_z: float = -10.0,
    seed: int | None = None,
) -> None:
    """Sample one solvable balance problem per env and materialize its weights.

    Preplaces the initial (unbalanced) weights on the two pans, drops the required
    add-weights plus distractors into the pile region, parks everything else, and
    commits the ordered add plan to ``denom_log_key`` / ``pan_log_key`` (both int
    tensors ``[num_envs, add_count]``; denom code = denom index + 1, pan code
    0=left/1=right; 0 denom = unused slot). ``same_target_pan`` restricts every
    addition in a sampled problem to one side. ``empty_target_denoms`` prevents
    preplaced weights from satisfying an add step with the same denomination.
    ``problem_limit`` restricts sampling over the deterministic problem pool:
    ``None`` keeps the full pool, an ``int N`` keeps ``pool[:N]``, and a slice
    ``[start, stop]`` / ``[start]`` keeps the half-open range (``stop`` may be
    ``None`` for the pool end). ``instruction_template`` is the vague policy-facing
    wording (may omit ``{plan}``); ``detailed_instruction_template`` optionally
    records the concrete plan under the detailed-instruction store. ``weight_orientation``
    fixes every visible weight to one XYZW orientation. ``unique_solution`` only
    exposes distractors that preserve a single denomination multiset whose mass
    closes the initial pan difference. The per-env instruction and overlay plan
    text are stored under :data:`_BALANCE_STATE_ATTR`.
    """
    if env_ids is None:
        return
    if left_pan_pos is None or right_pan_pos is None:
        raise ValueError("sample_balance_problem requires left_pan_pos and right_pan_pos")
    if pile_region is None:
        raise ValueError("sample_balance_problem requires pile_region={x:[...], y:[...], z:[...]}")
    orientation = [float(value) for value in (weight_orientation or [0.0, 0.0, 0.0, 1.0])]
    if len(orientation) != 4:
        raise ValueError("sample_balance_problem weight_orientation must be an XYZW quaternion with four values")

    from manaenv.pipeline.runtime.active_instruction import write_active_detailed_instruction, write_active_instruction

    denoms = [int(d) for d in denoms]
    ndim = len(denoms)
    pool = balance_problem_slice(
        _balance_problem_pool(
            env, denoms, count_per_denom, add_count, final_total_cap, same_target_pan, empty_target_denoms
        ),
        problem_limit,
    )
    if not pool:
        raise ValueError(
            f"sample_balance_problem: no solvable problems for denoms={denoms} "
            f"count_per_denom={count_per_denom} add_count={add_count} cap={final_total_cap}"
        )

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    state = getattr(env, _BALANCE_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _BALANCE_STATE_ATTR, state)
    plan_by_env: dict[int, str] = state.setdefault("plan", {})
    previous: dict[int, int] = state.setdefault("previous", {})

    denom_tensor = torch.zeros((num_envs, max(1, add_count)), dtype=torch.int64, device=device)
    pan_tensor = torch.zeros((num_envs, max(1, add_count)), dtype=torch.int64, device=device)

    pan_names = {0: "left", 1: "right"}
    pan_centers = {0: list(left_pan_pos), 1: list(right_pan_pos)}

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, f"{denom_log_key}:{add_count}", seed, env_id=env_id)
        # Pick a problem, avoiding an immediate repeat when possible.
        choices = list(range(len(pool)))
        if len(pool) > 1 and env_id in previous:
            choices = [c for c in choices if c != previous[env_id]] or choices
        pidx = rng.choice(choices)
        previous[env_id] = pidx
        problem = pool[pidx]

        # Per-denomination object queues (only assets that actually exist).
        queues: dict[int, list[str]] = {}
        for i, d in enumerate(denoms):
            names = [f"{weight_prefix}{d}g_{k}" for k in range(1, int(count_per_denom) + 1)]
            queues[i] = [nm for nm in names if scene_has_asset(env.scene, nm)]

        used: set[str] = set()

        # 1) Preplace initial weights on the pans.
        pan_slots: dict[int, list[list[float]]] = {
            0: _pan_slot_positions(pan_centers[0], sum(problem["left"]), pan_slot_spacing),
            1: _pan_slot_positions(pan_centers[1], sum(problem["right"]), pan_slot_spacing),
        }
        pan_cursor = {0: 0, 1: 0}
        for i in range(ndim):
            for pan, counts in ((0, problem["left"]), (1, problem["right"])):
                for _ in range(counts[i]):
                    if not queues[i]:
                        continue
                    name = queues[i].pop(0)
                    used.add(name)
                    slot = pan_slots[pan][pan_cursor[pan]]
                    pan_cursor[pan] += 1
                    pose = torch.zeros(7, dtype=torch.float32, device=device)
                    pose[:3] = torch.as_tensor(slot, dtype=torch.float32, device=device)
                    pose[3:7] = torch.as_tensor(orientation, dtype=torch.float32, device=device)
                    _place_balance_weight(env, name, env_id, pose)

        # 2) Drop the required add-weights into the pile, in randomized order.
        occupied_xy: list[tuple[float, float]] = []
        add_list = list(problem["add"])
        rng.shuffle(add_list)
        plan_parts: list[str] = []
        for step, (di, pan) in enumerate(add_list):
            if queues[di]:
                name = queues[di].pop(0)
                used.add(name)
                _place_balance_weight(
                    env,
                    name,
                    env_id,
                    _pile_pose(rng, pile_region, occupied_xy, min_separation, orientation, device),
                )
            denom_tensor[env_id, step] = di + 1
            pan_tensor[env_id, step] = pan
            plan_parts.append(f"a {denoms[di]}g weight on the {pan_names[pan]} pan")

        # 3) Distractors: extra unused weights shown in the pile.
        remaining = [(i, nm) for i in range(ndim) for nm in queues[i] if nm not in used]
        rng.shuffle(remaining)
        if unique_solution:
            distractors = _select_unique_balance_distractors(denoms, add_list, remaining, distractor_count, rng)
        else:
            distractors = [name for _, name in remaining[: max(0, int(distractor_count))]]
        for name in distractors:
            used.add(name)
            _place_balance_weight(
                env,
                name,
                env_id,
                _pile_pose(rng, pile_region, occupied_xy, min_separation, orientation, device),
            )

        # 4) Park everything else out of view.
        for i in range(ndim):
            for name in queues[i]:
                if name in used:
                    continue
                if not scene_has_asset(env.scene, name):
                    continue
                asset = env.scene[name]
                _set_pose_driven_asset_visibility(asset, False, env_id)
                pose = torch.zeros(7, dtype=torch.float32, device=device)
                default_pose = _default_root_pose_local_from_asset(asset, torch.as_tensor([env_id], device=device))
                if default_pose is not None:
                    pose[:] = default_pose[0]
                else:
                    pose[6] = 1.0
                pose[2] = float(away_z)
                _write_single_env_local_pose(env, asset, env_id, pose)

        plan_text = " and ".join(plan_parts) if plan_parts else "the scale is already balanced"
        plan_by_env[env_id] = plan_text
        write_active_instruction(
            env,
            env_id,
            str(instruction_template).format(plan=plan_text),
            num_envs=num_envs,
            reset_owned=True,
        )
        if detailed_instruction_template:
            write_active_detailed_instruction(
                env,
                env_id,
                str(detailed_instruction_template).format(plan=plan_text),
                num_envs=num_envs,
                reset_owned=True,
            )

    write_committed(env, str(denom_log_key), denom_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(pan_log_key), pan_tensor, env_ids=ids, frame_idx=-1)


def _pile_pose(
    rng: random.Random,
    pile_region: dict[str, Any],
    occupied_xy: list[tuple[float, float]],
    min_separation: float,
    orientation: list[float],
    device: Any,
) -> torch.Tensor:
    position, _ = _sample_region_pose(rng, pile_region, occupied_xy=occupied_xy, min_separation=min_separation)
    pose = torch.zeros(7, dtype=torch.float32, device=device)
    pose[:3] = torch.as_tensor(position, dtype=torch.float32, device=device)
    pose[3:7] = torch.as_tensor(orientation, dtype=torch.float32, device=device)
    return pose


def _place_balance_weight(env: ManagerBasedEnv, name: str, env_id: int, pose: torch.Tensor) -> None:
    if not scene_has_asset(env.scene, name):
        return
    asset = env.scene[name]
    _set_pose_driven_asset_visibility(asset, True, env_id)
    _write_single_env_local_pose(env, asset, env_id, pose)


_BRICK_ROUND_TRIP_STATE_ATTR = "_mana_brick_round_trip_state"


def sample_brick_round_trip(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    brick_names: list[str],
    front_slot_positions: list[list[float]],
    brick_colors: list[str] | None = None,
    source_log_key: str = "brick_round_trip_source_slot",
    brick_log_key: str = "brick_round_trip_brick",
    instruction_template: str = (
        "Move the {color} brick to the central plate, wait a moment, "
        "then put it back on the same plate and return the arm home."
    ),
    away_z: float = -10.0,
    seed: int | None = None,
) -> None:
    """Sample the per-episode brick color + source slot and place the active brick.

    Activates exactly one brick out of ``brick_names`` per env, drops it onto one
    randomly-chosen front slot (``front_slot_positions`` index = source slot), and
    parks the remaining bricks below the floor. The source slot index is committed
    to ``source_log_key`` and the brick index to ``brick_log_key`` (both int
    tensors ``[num_envs, 1]``) so the return stage can enforce return-to-origin.
    The per-env instruction and overlay label are stored under
    :data:`_BRICK_ROUND_TRIP_STATE_ATTR`.
    """
    if env_ids is None:
        return
    if not brick_names:
        raise ValueError("sample_brick_round_trip requires a non-empty brick_names")
    if not front_slot_positions:
        raise ValueError("sample_brick_round_trip requires a non-empty front_slot_positions")

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    brick_names = [str(name) for name in brick_names]
    slots = [list(pos) for pos in front_slot_positions]
    colors = [str(color) for color in (brick_colors or [])]

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    state = getattr(env, _BRICK_ROUND_TRIP_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _BRICK_ROUND_TRIP_STATE_ATTR, state)
    plan_by_env: dict[int, str] = state.setdefault("plan", {})
    previous: dict[int, tuple[int, int]] = state.setdefault("previous", {})

    source_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    brick_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(source_log_key), seed, env_id=env_id)
        source = rng.randrange(len(slots))
        brick_idx = rng.randrange(len(brick_names))
        # Avoid an immediate exact repeat (same brick AND same slot) when possible.
        if (brick_idx, source) == previous.get(env_id) and len(brick_names) * len(slots) > 1:
            source = rng.randrange(len(slots))
            brick_idx = rng.randrange(len(brick_names))
        previous[env_id] = (brick_idx, source)

        for i, name in enumerate(brick_names):
            if not scene_has_asset(env.scene, name):
                continue
            asset = env.scene[name]
            visible = i == brick_idx
            _set_pose_driven_asset_visibility(asset, visible, env_id)
            pose = torch.zeros(7, dtype=torch.float32, device=device)
            if visible:
                pose[:3] = torch.as_tensor(slots[source], dtype=torch.float32, device=device)
                pose[6] = 1.0
            else:
                default_pose = _default_root_pose_local_from_asset(asset, torch.as_tensor([env_id], device=device))
                if default_pose is not None:
                    pose[:] = default_pose[0]
                else:
                    pose[6] = 1.0
                pose[2] = float(away_z)
            _write_single_env_local_pose(env, asset, env_id, pose)

        source_tensor[env_id, 0] = source
        brick_tensor[env_id, 0] = brick_idx
        color = colors[brick_idx] if brick_idx < len(colors) else "target"
        write_active_instruction(
            env, env_id, str(instruction_template).format(color=color), num_envs=num_envs, reset_owned=True
        )
        plan_by_env[env_id] = f"{color} brick -> plate {source + 1}"

    write_committed(env, str(source_log_key), source_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(brick_log_key), brick_tensor, env_ids=ids, frame_idx=-1)


_FIND_DRAWER_STATE_ATTR = "_mana_find_drawer_state"
_FIND_DRAWER_LAYER_LABELS = ("top", "middle", "bottom")


def sample_find_drawer(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    num_layers: int = 3,
    layer_labels: list[str] | None = None,
    layer_log_key: str = "find_drawer_target_layer",
    instruction_template: str = (
        "Open the {layer} drawer, put the brick inside, close it, wait a few seconds, "
        "then reopen the same drawer, take the brick out, close it, and return the arm home."
    ),
    seed: int | None = None,
) -> None:
    """Sample the per-episode target drawer layer and commit it for indexed stages.

    Uniformly draws ``layer_index ∈ {0 .. num_layers-1}`` (0=top /
    ``PrismaticJoint_01`` on drawer_ba), commits it to ``layer_log_key`` as int
    ``[num_envs, 1]``, and writes the active instruction + overlay plan under
    :data:`_FIND_DRAWER_STATE_ATTR`.
    """
    if env_ids is None:
        return
    n_layers = int(num_layers)
    if n_layers <= 0:
        raise ValueError("sample_find_drawer requires num_layers > 0")

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    labels = [str(x) for x in (layer_labels or list(_FIND_DRAWER_LAYER_LABELS))]
    if len(labels) < n_layers:
        labels = labels + [f"layer_{i}" for i in range(len(labels), n_layers)]

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    state = getattr(env, _FIND_DRAWER_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _FIND_DRAWER_STATE_ATTR, state)
    plan_by_env: dict[int, str] = state.setdefault("plan", {})
    previous: dict[int, int] = state.setdefault("previous", {})

    layer_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(layer_log_key), seed, env_id=env_id)
        layer = rng.randrange(n_layers)
        if layer == previous.get(env_id) and n_layers > 1:
            layer = rng.randrange(n_layers)
        previous[env_id] = layer
        layer_tensor[env_id, 0] = layer
        label = labels[layer]
        write_active_instruction(
            env, env_id, str(instruction_template).format(layer=label), num_envs=num_envs, reset_owned=True
        )
        plan_by_env[env_id] = f"target drawer: {label}"

    write_committed(env, str(layer_log_key), layer_tensor, env_ids=ids, frame_idx=-1)


_WHACK_A_MOLE_STATE_ATTR = "_mana_whack_a_mole_state"
_WHACK_A_MOLE_PEG_COLORS = ("red", "orange", "yellow", "green", "blue", "purple")
_WHACK_A_MOLE_BASE_SLOT_LABELS = ("left_mid", "center_mid", "right_mid")
_WHACK_A_MOLE_HAMMER_SLOT_LABELS = (
    "top_left",
    "top_mid",
    "top_right",
    "mid_left",
    "mid_mid",
    "mid_right",
    "bot_left",
    "bot_mid",
    "bot_right",
)


def sample_whack_a_mole(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    mole_base_name: str = "mole_base",
    hammer_name: str = "hammer",
    mole_base_poses: list[list[float]] | None = None,
    mole_base_orientation: list[float] | None = None,
    hammer_poses: list[list[float]] | None = None,
    hammer_orientation: list[float] | None = None,
    peg_colors: list[str] | None = None,
    peg_joint_names: list[str] | None = None,
    raised_openness: float = 0.0,
    down_openness: float = 1.0,
    peg_index_log_key: str = "whack_target_peg_index",
    hammer_slot_log_key: str = "hammer_slot",
    mole_base_slot_log_key: str = "mole_base_slot",
    instruction_template: str = (
        "Pick up the wooden hammer, whack the raised {color} peg so it drops "
        "below the panel, put the hammer back on the table, then return the arm home."
    ),
    seed: int | None = None,
) -> None:
    """Sample mole-base slot, hammer 9-grid slot, and raised peg color at reset.

    Relocates ``mole_base_name`` / ``hammer_name``, sets the sampled peg joint to
    ``raised_openness`` and all other peg joints to ``down_openness``, commits
    ``peg_index_log_key`` (int ``[num_envs, 1]`` aligned with ``peg_joint_names``)
    plus optional slot indices, and writes the active instruction + overlay plan
    under :data:`_WHACK_A_MOLE_STATE_ATTR`.

    Openness follows Arena / :func:`_set_joint_openness` for ``j_min < 0`` pegs
    (toy_moel_base prismatic Z ``[-0.04, 0]``): 0 = raised (joint 0), 1 = down
    (joint -0.04).
    """
    if env_ids is None:
        return

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    base_poses = list(mole_base_poses or [])
    mallet_poses = list(hammer_poses or [])
    if not base_poses:
        raise ValueError("sample_whack_a_mole requires a non-empty mole_base_poses list")
    if not mallet_poses:
        raise ValueError("sample_whack_a_mole requires a non-empty hammer_poses list")

    colors = [str(c) for c in (peg_colors or list(_WHACK_A_MOLE_PEG_COLORS))]
    joints = [str(j) for j in (peg_joint_names or colors)]
    if not colors or len(colors) != len(joints):
        raise ValueError(f"sample_whack_a_mole peg_colors/peg_joint_names length mismatch: {colors!r} vs {joints!r}")

    base_quat = [float(v) for v in (mole_base_orientation or [0.0, 0.0, 0.0, 1.0])]
    hammer_quat = [float(v) for v in (hammer_orientation or [0.0, 0.0, 0.0, 1.0])]
    if len(base_quat) != 4 or len(hammer_quat) != 4:
        raise ValueError("sample_whack_a_mole orientations must be XYZW length 4")

    def _normalize_poses(raw_poses: list[list[float]], quat: list[float], label: str) -> list[list[float]]:
        out: list[list[float]] = []
        for raw in raw_poses:
            vals = [float(v) for v in raw]
            if len(vals) == 3:
                out.append(vals + quat)
            elif len(vals) == 7:
                out.append(vals)
            else:
                raise ValueError(
                    f"sample_whack_a_mole {label} entries must be [x,y,z] or " f"[x,y,z,qx,qy,qz,qw], got {raw!r}"
                )
        return out

    base_local = _normalize_poses(base_poses, base_quat, "mole_base_poses")
    hammer_local = _normalize_poses(mallet_poses, hammer_quat, "hammer_poses")

    if not scene_has_asset(env.scene, mole_base_name) or not scene_has_asset(env.scene, hammer_name):
        return

    mole_base = env.scene[mole_base_name]
    hammer = env.scene[hammer_name]
    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    joint_indices: list[int] = []
    if hasattr(mole_base, "find_joints"):
        found, _ = mole_base.find_joints(joints, preserve_order=True)
        if len(found) != len(joints):
            raise ValueError(
                f"sample_whack_a_mole: expected {len(joints)} joints matching {joints!r} "
                f"on {mole_base_name!r}, found {len(found)}"
            )
        joint_indices = [int(i) for i in found]

    state = getattr(env, _WHACK_A_MOLE_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _WHACK_A_MOLE_STATE_ATTR, state)
    plan_by_env: dict[int, str] = state.setdefault("plan", {})
    previous_peg: dict[int, int] = state.setdefault("previous_peg", {})

    peg_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    hammer_slot_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    base_slot_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)

    raised = float(raised_openness)
    down = float(down_openness)

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(peg_index_log_key), seed, env_id=env_id)
        peg_idx = rng.randrange(len(colors))
        if peg_idx == previous_peg.get(env_id) and len(colors) > 1:
            peg_idx = rng.randrange(len(colors))
        previous_peg[env_id] = peg_idx

        base_slot = rng.randrange(len(base_local))
        hammer_slot = rng.randrange(len(hammer_local))

        peg_tensor[env_id, 0] = peg_idx
        base_slot_tensor[env_id, 0] = base_slot
        hammer_slot_tensor[env_id, 0] = hammer_slot

        single = ids.new_tensor([env_id])
        _write_single_env_local_pose(
            env, mole_base, env_id, torch.as_tensor(base_local[base_slot], dtype=torch.float32, device=device)
        )
        _write_single_env_local_pose(
            env, hammer, env_id, torch.as_tensor(hammer_local[hammer_slot], dtype=torch.float32, device=device)
        )

        if joint_indices:
            openness = torch.full((1,), down, dtype=torch.float32, device=device)
            for joint_index in joint_indices:
                _set_joint_openness(env, mole_base, joint_index, openness, single)
            raised_t = torch.full((1,), raised, dtype=torch.float32, device=device)
            _set_joint_openness(env, mole_base, joint_indices[peg_idx], raised_t, single)

        color = colors[peg_idx]
        base_label = (
            _WHACK_A_MOLE_BASE_SLOT_LABELS[base_slot]
            if base_slot < len(_WHACK_A_MOLE_BASE_SLOT_LABELS)
            else str(base_slot)
        )
        hammer_label = (
            _WHACK_A_MOLE_HAMMER_SLOT_LABELS[hammer_slot]
            if hammer_slot < len(_WHACK_A_MOLE_HAMMER_SLOT_LABELS)
            else str(hammer_slot)
        )
        instruction = str(instruction_template).format(color=color)
        write_active_instruction(env, env_id, instruction, num_envs=num_envs, reset_owned=True)
        plan_by_env[env_id] = f"raised={color}; base={base_label}; hammer={hammer_label}"
        manaprint(
            f"INFO: [whack_a_mole] env={env_id} raised={color} base={base_label} "
            f"hammer={hammer_label} instruction={instruction!r}"
        )

    write_committed(env, str(peg_index_log_key), peg_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(hammer_slot_log_key), hammer_slot_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(mole_base_slot_log_key), base_slot_tensor, env_ids=ids, frame_idx=-1)


_STACK_BLOCKS_STATE_ATTR = "_mana_stack_blocks_state"


def sample_stack_blocks(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    num_blocks: int,
    spawn_region: dict[str, Any],
    colors: list[str] | None = None,
    slot_prefix: str = "block_",
    num_slots: int = 5,
    base_position: list[float] | None = None,
    min_separation: float = 0.09,
    count_log_key: str = "stack_blocks_count",
    instruction_template: str = ("Stack the {n} blocks into a single tower on the marker, then return the arm home."),
    away_z: float = -10.0,
    seed: int | None = None,
) -> None:
    """Activate the first ``num_blocks`` block slots, each an independent color.

    The scene declares ``num_slots`` block slots, each with one fixed object per
    color named ``{slot_prefix}{i}_{color}`` (i is 1-based). Each DAG variant sets
    ``num_blocks`` (3/4/5). Per reset this activates slots ``1..num_blocks``; for
    each active slot it independently draws a random color from ``colors``, shows
    that color's object at a random non-overlapping position inside
    ``spawn_region`` (rejection sampling on ``min_separation``), and parks every
    other object (unchosen colors of active slots and all objects of inactive
    slots) below the floor. Colors are drawn independently per slot, so repeats
    are expected and fine. The block count is committed to ``count_log_key`` and
    the per-env instruction + overlay label are stored under
    :data:`_STACK_BLOCKS_STATE_ATTR`. ``base_position`` documents the fixed tower
    footprint the stage success zones check against; it is not consumed here.
    """
    if env_ids is None:
        return
    if spawn_region is None:
        raise ValueError("sample_stack_blocks requires spawn_region={x: [...], y: [...], z: [...]}")

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    colors = [str(c) for c in (colors or ["blue", "green", "red", "orange"])]
    total_slots = max(0, int(num_slots))
    count = max(0, min(int(num_blocks), total_slots))

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    state = getattr(env, _STACK_BLOCKS_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _STACK_BLOCKS_STATE_ATTR, state)
    plan_by_env: dict[int, str] = state.setdefault("plan", {})

    count_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(count_log_key), seed, env_id=env_id)
        occupied_xy: list[tuple[float, float]] = []
        for slot in range(1, total_slots + 1):
            active = slot <= count
            chosen_color = rng.choice(colors) if active else None
            for color in colors:
                name = f"{slot_prefix}{slot}_{color}"
                if not scene_has_asset(env.scene, name):
                    continue
                asset = env.scene[name]
                visible = active and color == chosen_color
                _set_pose_driven_asset_visibility(asset, visible, env_id)
                pose = torch.zeros(7, dtype=torch.float32, device=device)
                if visible:
                    position, yaw = _sample_region_pose(
                        rng, spawn_region, occupied_xy=occupied_xy, min_separation=min_separation
                    )
                    pose[:3] = torch.as_tensor(position, dtype=torch.float32, device=device)
                    half_yaw = torch.tensor(0.5 * yaw, dtype=torch.float32, device=device)
                    pose[5] = torch.sin(half_yaw)
                    pose[6] = torch.cos(half_yaw)
                else:
                    default_pose = _default_root_pose_local_from_asset(asset, torch.as_tensor([env_id], device=device))
                    if default_pose is not None:
                        pose[:] = default_pose[0]
                    else:
                        pose[6] = 1.0
                    pose[2] = float(away_z)
                _write_single_env_local_pose(env, asset, env_id, pose)

        count_tensor[env_id, 0] = count
        instruction = str(instruction_template).format(n=count)
        write_active_instruction(env, env_id, instruction, num_envs=num_envs, reset_owned=True)
        plan_by_env[env_id] = f"stack {count} blocks"

    write_committed(env, str(count_log_key), count_tensor, env_ids=ids, frame_idx=-1)


_HANOI_STATE_ATTR = "_mana_hanoi_state"


def sample_tower_of_hanoi(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    ring_names: list[str],
    n: int,
    left_peg_xy: list[float] | tuple[float, float] | None = None,
    layer_z_center: list[float] | None = None,
    layer_z_bottom: list[float] | None = None,
    bottom_origin_rings: list[str] | None = None,
    away_z: float = -10.0,
    min_z: float = -1.0,  # unused; kept for older YAML compatibility
    instruction_template: str = (
        "Move the {n} rings from the left peg to the right peg. "
        "Move one ring at a time, and never place a larger ring on a smaller one."
    ),
    plan_log_key: str = "tower_of_hanoi_plan",
    seed: int | None = None,
) -> None:
    """Sample a random adjacent ring window, stack it on the left peg, park the rest.

    Heights are hardcoded (not layout-solved): bottom clearance 3 cm, thickness
    1.5 cm, layer step 1.8 cm. ``ring_a`` uses bottom-origin z; others center-origin.

    ``active_rings`` is ordered smallest→largest (``disk_1`` = top / smallest) so
    ``skill_binding`` ``role_prefix: disk_`` matches stage ``@disk_k`` refs.

    Adjacent windows (e.g. n=3 → abc/bcd/cde) are drawn with a per-env RNG
    seeded by ``seed`` so successive resets are reproducible but not ordered.
    """
    del min_z  # placement is authored; no longer inferred from spawn z
    if env_ids is None:
        return

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction
    from manaenv.pipeline.runtime.log_store import write_committed

    names = [str(name).strip() for name in (ring_names or []) if str(name).strip()]
    if not names:
        raise ValueError("sample_tower_of_hanoi requires non-empty ring_names")
    count = int(n)
    if count < 1:
        raise ValueError(f"sample_tower_of_hanoi requires n >= 1, got {n}")
    if count > len(names):
        raise ValueError(f"sample_tower_of_hanoi n={count} exceeds ring_names={len(names)}")

    # Fallback XY if peg_left is missing; prefer live peg_left root after surface place.
    fallback_xy = [float(v) for v in (left_peg_xy if left_peg_xy is not None else [0.155, -0.148])]
    if len(fallback_xy) != 2:
        raise ValueError("left_peg_xy must be [x, y]")
    z_center = [
        float(v) for v in (layer_z_center if layer_z_center is not None else [-0.2029 + i * 0.018 for i in range(5)])
    ]
    z_bottom = [
        float(v) for v in (layer_z_bottom if layer_z_bottom is not None else [-0.2104 + i * 0.018 for i in range(5)])
    ]
    bottom_set = {str(x).strip() for x in (bottom_origin_rings or ["ring_a"]) if str(x).strip()}

    windows = [tuple(names[i : i + count]) for i in range(0, len(names) - count + 1)]
    if not windows:
        raise ValueError(f"sample_tower_of_hanoi: no adjacent windows for n={count}")

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    state = getattr(env, _HANOI_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _HANOI_STATE_ATTR, state)
    active_by_env: dict[int, list[str]] = state.setdefault("active_rings", {})
    n_by_env: dict[int, int] = state.setdefault("n", {})
    plan_by_env: dict[int, str] = state.setdefault("plan", {})
    # Drop the old round-robin cursor if a prior run left it on the env.
    state.pop("window_cursor", None)
    state.pop("previous_window", None)

    n_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    for env_id_t in ids:
        env_id = int(env_id_t.item())
        peg_xy = list(fallback_xy)
        if scene_has_asset(env.scene, "peg_left"):
            peg = env.scene["peg_left"]
            local = peg.data.root_pos_w[env_id] - env.scene.env_origins[env_id]
            peg_xy = [float(local[0].item()), float(local[1].item())]
        rng = _task_sample_rng(env, f"{plan_log_key}:{env_id}", seed, env_id=env_id)
        window = rng.choice(windows)
        active_set = set(window)
        # disk_1 = smallest = last letter in large→small window order.
        active = list(reversed(window))

        for name in names:
            if not scene_has_asset(env.scene, name):
                continue
            asset = env.scene[name]
            visible = name in active_set
            _set_pose_driven_asset_visibility(asset, visible, env_id)
            pose = torch.zeros(7, dtype=torch.float32, device=device)
            pose[6] = 1.0  # XYZW identity (Isaac Lab 3.0+; 5.1 was WXYZ)
            if visible:
                layer = window.index(name)
                z_table = z_bottom if name in bottom_set else z_center
                if layer >= len(z_table):
                    raise ValueError(f"layer {layer} out of range for {name}")
                pose[0] = peg_xy[0]
                pose[1] = peg_xy[1]
                pose[2] = z_table[layer]
            else:
                default_pose = _default_root_pose_local_from_asset(asset, torch.as_tensor([env_id], device=device))
                if default_pose is not None:
                    pose[:] = default_pose[0]
                pose[2] = float(away_z)
            _write_single_env_local_pose(env, asset, env_id, pose)

        active_by_env[env_id] = list(active)
        n_by_env[env_id] = count
        instruction = str(instruction_template).format(n=count)
        write_active_instruction(env, env_id, instruction, num_envs=num_envs, reset_owned=True)
        plan_by_env[env_id] = f"hanoi n={count} rings={active}"
        n_tensor[env_id, 0] = count

    write_committed(env, str(plan_log_key), n_tensor, env_ids=ids, frame_idx=-1)


_STACK_HEIGHT_STATE_ATTR = "_mana_stack_height_state"
_STACK_HEIGHT_PRESERVE_ATTR = "_mana_stack_height_preserve"


def sample_stack_blocks_height(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    block_names: list[str],
    target_height_range: list[int] | tuple[int, int] = (12, 12),
    target_height_log_key: str = "stack_tower_height",
    instruction_template: str = (
        "Place a block on the green paper, then stack other blocks on top to build a tower of {n}, "
        "and return the arm home."
    ),
    seed: int | None = None,
) -> None:
    """Sample per-env tower height and pin instruction for ``stack_blocks_height``.

    Layout placement of the 15 cubes + paper is owned by ``layout.runtime_sample``.
    This event only samples the language target height. When
    ``env._mana_stack_height_preserve[env_id]`` is true (staged child segments),
    the previously pinned height/instruction are kept and re-committed so a
    snapshot restore cannot change the root goal.
    """
    if env_ids is None:
        return

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    names = [str(n).strip() for n in (block_names or []) if str(n).strip()]
    if not names:
        raise ValueError("sample_stack_blocks_height requires non-empty block_names")

    lo = int(target_height_range[0])
    hi = int(target_height_range[1] if len(target_height_range) > 1 else target_height_range[0])
    if hi < lo:
        lo, hi = hi, lo
    lo = max(1, lo)
    hi = max(lo, hi)

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    state = getattr(env, _STACK_HEIGHT_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _STACK_HEIGHT_STATE_ATTR, state)
    plan_by_env: dict[int, str] = state.setdefault("plan", {})
    target_by_env: dict[int, int] = state.setdefault("target_height", {})
    blocks_by_env: dict[int, list[str]] = state.setdefault("block_names", {})
    preserve = getattr(env, _STACK_HEIGHT_PRESERVE_ATTR, None)
    if not isinstance(preserve, dict):
        preserve = {}
        setattr(env, _STACK_HEIGHT_PRESERVE_ATTR, preserve)

    height_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(target_height_log_key), seed, env_id=env_id)
        if preserve.get(env_id) and env_id in target_by_env:
            height = int(target_by_env[env_id])
        else:
            height = max(lo, min(hi, int(rng.randint(lo, hi))))
            target_by_env[env_id] = height
            preserve[env_id] = False
        target_by_env[env_id] = height
        blocks_by_env[env_id] = list(names)
        instruction = str(instruction_template).format(n=height)
        write_active_instruction(env, env_id, instruction, num_envs=num_envs, reset_owned=True)
        plan_by_env[env_id] = f"stack tower of {height}"
        height_tensor[env_id, 0] = height

    write_committed(env, str(target_height_log_key), height_tensor, env_ids=ids, frame_idx=-1)


_SPECIFIC_POSITION_STATE_ATTR = "_mana_specific_position_state"


def _grid_cache_slot_names(env: ManagerBasedEnv, logical_name: str, *, max_slots: int = 16) -> list[str]:
    """Return the spawned prespawn cache slots for one logical pool node (base + ``__slot_N``)."""
    names: list[str] = []
    for index in range(int(max_slots)):
        name = logical_name if index == 0 else f"{logical_name}__slot_{index}"
        if scene_has_asset(env.scene, name):
            names.append(name)
        elif index:
            break
    return names


def sample_specific_position_grid(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    grid_shapes: list[list[int]],
    target_name: str = "target_bus",
    grid_member_prefix: str = "grid_bus_",
    max_objects: int = 12,
    grid_center: list[float] | None = None,
    row_spacing: float = 0.14,
    column_spacing: float = 0.15,
    object_height: float = -0.18,
    object_orientation: list[float] | None = None,
    away_z: float = -10.0,
    rows_log_key: str = "specific_position_grid_rows",
    columns_log_key: str = "specific_position_grid_columns",
    target_index_log_key: str = "specific_position_target_index",
    instruction_template: str = "Pick up the object in row {row}, column {column}, and put it in the basket, then return the arm home.",
    seed: int | None = None,
) -> None:
    """Lay the pool-randomized buses on a camera-ordered grid and pin one target cell.

    Per-episode appearance randomization and the ``@target_bus`` / ``@grid_member``
    tag -> active cache-slot routing (which lets the contact sensor and planner
    follow the randomized bus) are owned by the injected ``layout.runtime_sample``
    asset effect, exactly like pick_by_attribute; that effect runs before this reset
    event. This event only decides the per-episode grid geometry: it resolves each
    logical node to its runtime-active slot, lays the active grid members on an exact
    zero-jitter grid with ``target_bus`` forced onto the sampled cell, parks every
    other cache slot below the floor, and commits rows/columns/target-index plus the
    instruction.
    """
    if env_ids is None:
        return
    shapes = [(int(shape[0]), int(shape[1])) for shape in grid_shapes if len(shape) == 2]
    if not shapes or any(rows <= 0 or columns <= 0 for rows, columns in shapes):
        raise ValueError("sample_specific_position_grid requires positive [rows, columns] grid_shapes")
    if any(rows * columns > int(max_objects) for rows, columns in shapes):
        raise ValueError("sample_specific_position_grid grid shape exceeds max_objects")

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    center = [float(value) for value in (grid_center or [0.28, -0.268246])]
    if len(center) != 2:
        raise ValueError("sample_specific_position_grid grid_center must contain [x, y]")
    orientation = [float(value) for value in (object_orientation or [0.0, 0.0, 0.0, 1.0])]
    if len(orientation) != 4:
        raise ValueError("sample_specific_position_grid object_orientation must be an XYZW quaternion")

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device
    orientation_tensor = torch.as_tensor(orientation, dtype=torch.float32, device=device)
    rows_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    columns_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    target_index_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)

    state = getattr(env, _SPECIFIC_POSITION_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _SPECIFIC_POSITION_STATE_ATTR, state)
    plan_by_env: dict[int, str] = state.setdefault("plan", {})

    logical_names = [str(target_name)] + [
        f"{grid_member_prefix}{index}" for index in range(1, max(1, int(max_objects)))
    ]

    def parked_pose(slot_index: int) -> torch.Tensor:
        pose = torch.zeros(7, dtype=torch.float32, device=device)
        pose[0] = 0.15 * float(slot_index + 1)
        pose[2] = float(away_z) - 0.1 * float(slot_index + 1)
        pose[6] = 1.0
        return pose

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(target_index_log_key), seed, env_id=env_id)
        rows, columns = rng.choice(shapes)
        active_count = rows * columns
        target_index = rng.randrange(active_count)

        # cell -> logical node; target_bus is forced onto the sampled target cell.
        members = list(logical_names[:active_count])
        rng.shuffle(members)
        current = members.index(str(target_name))
        if current != target_index:
            members[current], members[target_index] = members[target_index], members[current]

        for logical_name in logical_names:
            active_name = resolve_active_scene_entity_name(env, logical_name, env_id=env_id)
            for slot_index, slot_name in enumerate(_grid_cache_slot_names(env, logical_name)):
                asset = env.scene[slot_name]
                _set_pose_driven_asset_visibility(asset, False, env_id)
                _write_single_env_local_pose(env, asset, env_id, parked_pose(slot_index))
            if logical_name not in members:
                continue
            cell = members.index(logical_name)
            row = cell // columns
            column = cell % columns
            pose = torch.zeros(7, dtype=torch.float32, device=device)
            pose[0] = center[0] + (0.5 * (rows - 1) - row) * float(row_spacing)
            pose[1] = center[1] + (0.5 * (columns - 1) - column) * float(column_spacing)
            pose[2] = float(object_height)
            pose[3:7] = orientation_tensor
            asset = env.scene[active_name]
            _set_pose_driven_asset_visibility(asset, True, env_id)
            _write_single_env_local_pose(env, asset, env_id, pose)

        target_row = target_index // columns + 1
        target_column = target_index % columns + 1
        rows_tensor[env_id, 0] = rows
        columns_tensor[env_id, 0] = columns
        target_index_tensor[env_id, 0] = target_index
        instruction = str(instruction_template).format(row=target_row, column=target_column)
        write_active_instruction(env, env_id, instruction, num_envs=num_envs, reset_owned=True)
        plan_by_env[env_id] = f"{rows}x{columns} grid: row {target_row}, column {target_column}"

    write_committed(env, str(rows_log_key), rows_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(columns_log_key), columns_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(target_index_log_key), target_index_tensor, env_ids=ids, frame_idx=-1)


_ATTRIBUTE_PICK_STATE_ATTR = "_mana_attribute_pick_state"
_ATTRIBUTE_CATALOG_CACHE_ATTR = "_mana_attribute_catalog_cache"


def _resolve_definition_path(path_text: str) -> Path:
    """Resolve a defination-relative asset path the same way the vNext parser does."""
    path = Path(str(path_text)).expanduser()
    if path.is_absolute():
        return path
    package_root = Path(__file__).resolve().parents[1]
    for candidate in (package_root / "defination" / path, package_root / path, Path.cwd() / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"attribute catalog file not found: {path_text}")


def _attribute_id_keys(asset_id: str) -> tuple[str, ...]:
    """Return the ``source:id`` / ``source__id`` spellings of one asset id."""
    text = str(asset_id).strip()
    if not text:
        return ()
    return tuple(dict.fromkeys((text, text.replace(":", "__"), text.replace("__", ":"))))


def _load_attribute_catalog(env: ManagerBasedEnv, catalog_file: str) -> dict[str, dict[str, str]]:
    """Build (and cache on env) an ``asset_id -> {color, shape}`` reverse map."""
    cache = getattr(env, _ATTRIBUTE_CATALOG_CACHE_ATTR, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(env, _ATTRIBUTE_CATALOG_CACHE_ATTR, cache)
    if catalog_file in cache:
        return cache[catalog_file]

    payload = json.loads(_resolve_definition_path(catalog_file).read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, str]] = {}
    for entry in payload.get("assets", []):
        asset_id = str(entry.get("id") or entry.get("asset_name") or "").strip()
        if not asset_id:
            continue
        attrs = {
            "color": str(entry.get("color") or "").strip().lower(),
            "shape": str(entry.get("shape") or "").strip().lower(),
        }
        for key in _attribute_id_keys(asset_id):
            mapping[key] = attrs
    cache[catalog_file] = mapping
    return mapping


def _read_active_asset_id(env: ManagerBasedEnv, plan_id: str, env_id: int, logical_name: str) -> str | None:
    """Return the concrete ``source:asset_id`` materialized for a pool slot this episode."""
    state = getattr(env, "_schema_vnext_active_asset_slots", None)
    if not isinstance(state, dict):
        return None
    plan_state = state.get(str(plan_id))
    if not isinstance(plan_state, dict):
        for candidate in state.values():
            if isinstance(candidate, dict) and int(env_id) in candidate:
                plan_state = candidate
                break
    if not isinstance(plan_state, dict):
        return None
    env_state = plan_state.get(int(env_id))
    if not isinstance(env_state, dict):
        return None
    entry = env_state.get(str(logical_name))
    if isinstance(entry, dict):
        asset_id = str(entry.get("asset_id") or "").strip()
        return asset_id or None
    return None


def _attribute_cond_matches(attrs: dict[str, str], cond: dict[str, Any]) -> bool:
    return all(str(attrs.get(str(key), "")) == str(value) for key, value in cond.items())


def _attribute_predicate_matches(attrs: dict[str, str], predicate: dict[str, Any]) -> bool:
    """Evaluate one object against an AND (all_of) / OR (any_of) / NOT (none_of) predicate."""
    all_of = predicate.get("all_of") or []
    any_of = predicate.get("any_of") or []
    none_of = predicate.get("none_of") or []
    if not (all_of or any_of or none_of):
        return False
    if all_of and not all(_attribute_cond_matches(attrs, cond) for cond in all_of):
        return False
    if any_of and not any(_attribute_cond_matches(attrs, cond) for cond in any_of):
        return False
    if none_of and any(_attribute_cond_matches(attrs, cond) for cond in none_of):
        return False
    return True


def sample_attribute_pick(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    object_names: list[str],
    catalog_file: str,
    predicate_bank: list[dict[str, Any]],
    target_mask_log_key: str = "attribute_target_mask",
    plan_id: str = "runtime",
    stratify_by_group: bool = False,
    confusion_phrases: list[str] | None = None,
    confusion_count: Any = 1,
    confusion_colors: list[str] | None = None,
    confusion_shapes: list[str] | None = None,
    confusion_placement: str = "random",
    default_color: str = "unknown",
    default_shape: str = "unknown",
    seed: int | None = None,
) -> None:
    """Pick a color/shape/logic predicate over the four drawn objects and commit its target mask.

    Design B reverse lookup: each slot's concrete asset was already materialized by
    the runtime sampler, so we read it back from ``_schema_vnext_active_asset_slots``
    and map ``asset_id -> (color, shape)`` via ``catalog_file``. Only predicates that
    at least one drawn object satisfies are eligible, guaranteeing >= 1 target. The
    committed ``[num_envs, num_slots]`` 0/1 mask (column i == object_names[i]) is read
    back by :func:`manaenv.terms.terminations.attribute_targets_in_basket`.

    ``stratify_by_group`` first picks a predicate ``group`` (e.g. ``and`` / ``or`` /
    ``not``) uniformly among the groups that still have a satisfiable predicate this
    episode, then a satisfiable predicate within it — so evaluation coverage stays
    balanced across logic types even though OR/NOT are satisfiable far more often
    than a specific AND.

    When ``confusion_phrases`` is non-empty, random distractor sentences are woven
    around the real predicate text (placement / count configurable). The target mask
    still follows only the selected predicate.
    """
    if env_ids is None:
        return
    if not object_names:
        raise ValueError("sample_attribute_pick requires a non-empty object_names list")
    if not predicate_bank:
        raise ValueError("sample_attribute_pick requires a non-empty predicate_bank")

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    mapping = _load_attribute_catalog(env, str(catalog_file))
    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device
    num_slots = len(object_names)
    mask_tensor = torch.zeros((num_envs, num_slots), dtype=torch.int64, device=device)

    state = getattr(env, _ATTRIBUTE_PICK_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _ATTRIBUTE_PICK_STATE_ATTR, state)
    labels: dict[int, str] = state.setdefault("label", {})

    def slot_attrs(env_id: int, logical_name: str) -> dict[str, str]:
        asset_id = _read_active_asset_id(env, plan_id, env_id, logical_name)
        if asset_id is not None:
            for key in _attribute_id_keys(asset_id):
                if key in mapping:
                    return mapping[key]
        return {"color": str(default_color), "shape": str(default_shape)}

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(target_mask_log_key), seed, env_id=env_id)
        attrs_per_slot = [slot_attrs(env_id, name) for name in object_names]

        satisfiable = [
            predicate
            for predicate in predicate_bank
            if any(_attribute_predicate_matches(attrs, predicate) for attrs in attrs_per_slot)
        ]
        if satisfiable and stratify_by_group:
            groups: dict[str, list[dict[str, Any]]] = {}
            for candidate in satisfiable:
                groups.setdefault(str(candidate.get("group", "")), []).append(candidate)
            predicate = rng.choice(groups[rng.choice(sorted(groups))])
        elif satisfiable:
            predicate = rng.choice(satisfiable)
        else:
            # Guaranteed-satisfiable fallback: this episode's first-slot color.
            fallback_color = attrs_per_slot[0]["color"]
            predicate = {
                "id": f"color_{fallback_color}",
                "text": f"Put the {fallback_color} objects into the basket.",
                "any_of": [{"color": fallback_color}],
            }

        slot_matches = [1 if _attribute_predicate_matches(attrs, predicate) else 0 for attrs in attrs_per_slot]
        for column, match in enumerate(slot_matches):
            mask_tensor[env_id, column] = int(match)

        base_instruction = str(predicate.get("text") or "")
        if confusion_phrases:

            instruction = compose_attribute_instruction_with_confusion(
                base_instruction,
                confusion_phrases=confusion_phrases,
                rng=rng,
                confusion_count=confusion_count,
                confusion_colors=confusion_colors,
                confusion_shapes=confusion_shapes,
                confusion_placement=confusion_placement,
            )
        else:
            instruction = base_instruction
        write_active_instruction(env, env_id, instruction, num_envs=num_envs, reset_owned=True)

        object_desc = ", ".join(
            f"{attrs_per_slot[i]['color']} {attrs_per_slot[i]['shape']}{'*' if slot_matches[i] else ''}"
            for i in range(num_slots)
        )
        labels[env_id] = f"{predicate.get('id', '?')}\n{instruction}\nobjects: {object_desc}"

    write_committed(env, str(target_mask_log_key), mask_tensor, env_ids=ids, frame_idx=-1)


def select_episode_grasp_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfgs: Any = None,
    store_attr: str = "_mana_episode_grasp_target",
) -> None:
    """Pin one grasp-target bottle per env for this episode — TASK/SCENE authority.

    Choosing WHICH drink to grasp belongs to the task/scene, not the collection
    planner. ``object_cfgs`` is the candidate pool, declared in YAML as the
    ``@drink`` role and resolved to concrete bottle NAMES at finalize — the SAME
    finalize-time role projection ``shuffle_shelf_stock_layout`` uses for
    ``@shelf_stock``. (A runtime ``find_assets_by_tag(env.scene, ...)`` query
    does NOT work: the runtime ``env.scene`` is IsaacLab's ``InteractiveScene``,
    which carries physics assets but none of manaenv's generation-time semantic
    tags, so the pool would come up empty.) One bottle is picked per reset env
    and pinned into ``env.<store_attr>[env_id]`` for the reach success checker
    and the (obeying) planner to read — neither recomputes it, and no shelf
    geometry is consulted here.
    """
    if env_ids is None:
        return
    from manaenv.terms.grasp_target_selection import normalize_pool, select_episode_target

    pool = normalize_pool(object_cfgs)
    if not pool:
        return

    store = getattr(env, store_attr, None)
    if not isinstance(store, dict):
        store = {}
        setattr(env, store_attr, store)

    ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
    for env_id in ids:
        prev = store.get(int(env_id))
        store[int(env_id)] = select_episode_target(pool, exclude=prev)


def randomize_object_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | str | list,
    mass_range: tuple[float, float] | list[float],
    recompute_inertia: bool = True,
) -> None:
    """Sample a uniform mass in ``mass_range`` (kg) on reset; pool-aware.

    Covers every pool cache slot (``id`` / ``id__slot_N``) so the active slot
    is always in-range; inertia rescales with mass assuming uniform density.
    """
    if env_ids is None:
        return
    if isinstance(asset_cfg, list):
        for single_cfg in asset_cfg:
            randomize_object_mass(env, env_ids, single_cfg, mass_range, recompute_inertia)
        return

    low, high = float(mass_range[0]), float(mass_range[1])
    if low < 1e-6 or high < low:
        raise ValueError(f"randomize_object_mass: invalid mass_range {mass_range}")

    logical_name = get_scene_entity_name(asset_cfg)
    slot_names = [
        name
        for name in (pool_cache_slot_name(logical_name, i) for i in range(resolve_pool_cache_size(env, logical_name)))
        if scene_has_asset(env, name)
    ] or [logical_name]

    for name in slot_names:
        asset = env.scene[name]
        body_masses = asset.data.body_mass.torch
        ids = env_ids.to(device=body_masses.device, dtype=torch.long)
        body_ids = torch.arange(body_masses.shape[1], device=body_masses.device)
        current_masses = body_masses[ids[:, None], body_ids].clone()
        sampled_masses = torch.empty_like(current_masses).uniform_(low, high)

        if recompute_inertia:
            current_inertias = asset.data.body_inertia.torch[ids[:, None], body_ids].clone()
            sampled_inertias = current_inertias * (sampled_masses / current_masses)[..., None]

        asset.set_masses_index(masses=sampled_masses, body_ids=body_ids, env_ids=ids)
        if recompute_inertia:
            asset.set_inertias_index(inertias=sampled_inertias, body_ids=body_ids, env_ids=ids)


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    initial_pose: Pose,
    position_range: dict[str, list[float]],
) -> None:
    """Randomize object position around its initial pose on reset.

    Each reset, the object is placed at initial_pose + uniform_random_offset,
    where the offset is sampled independently per axis per environment.

    Args:
        env: The IsaacLab environment instance.
        env_ids: Indices of environments being reset.
        asset_cfg: Scene entity config identifying the object.
        initial_pose: The base pose to randomize around.
        position_range: Per-axis randomization range, e.g.
            {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.0, 0.0]}
    """
    if env_ids is None:
        return

    asset = env.scene[get_scene_entity_name(asset_cfg)]
    num_envs = len(env_ids)

    # Start from initial pose (position_xyz + rotation_xyzw = 7D)
    pose_tensor = initial_pose.to_tensor(device=env.device).repeat(num_envs, 1)

    # Add uniform random offsets per axis
    x_range = position_range.get("x", [0.0, 0.0])
    y_range = position_range.get("y", [0.0, 0.0])
    z_range = position_range.get("z", [0.0, 0.0])

    pose_tensor[:, 0] += torch.empty(num_envs, device=env.device).uniform_(x_range[0], x_range[1])
    pose_tensor[:, 1] += torch.empty(num_envs, device=env.device).uniform_(y_range[0], y_range[1])
    pose_tensor[:, 2] += torch.empty(num_envs, device=env.device).uniform_(z_range[0], z_range[1])

    # Sync local pose to USD *before* adding env_origins
    _sync_pose_to_usd(asset, env_ids, pose_tensor)

    # Transform to simulation frame (add env origin offset)
    pose_tensor[:, :3] += env.scene.env_origins[env_ids]

    # Write pose and zero velocity
    _write_root_pose_with_zero_velocity(asset, pose_tensor, env_ids)


def randomize_object_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    initial_pose: Pose,
    yaw_range: list[float] | None = None,
) -> None:
    """Randomize object yaw (Z-axis rotation) around its initial pose.

    Only randomizes yaw for now; pitch/roll randomization can be added later.

    Args:
        env: The IsaacLab environment instance.
        env_ids: Indices of environments being reset.
        asset_cfg: Scene entity config identifying the object.
        initial_pose: The base pose to randomize around.
        yaw_range: [min_yaw, max_yaw] in radians. Defaults to [-pi, pi].
    """
    if env_ids is None:
        return

    if yaw_range is None:
        yaw_range = [-math.pi, math.pi]

    asset = env.scene[get_scene_entity_name(asset_cfg)]
    num_envs = len(env_ids)

    pose_tensor = initial_pose.to_tensor(device=env.device).repeat(num_envs, 1)

    # Generate random yaw angles
    yaw = torch.empty(num_envs, device=env.device).uniform_(yaw_range[0], yaw_range[1])

    # Convert yaw to quaternion (x, y, z, w) — rotation around Z axis
    half_yaw = yaw * 0.5
    pose_tensor[:, 3] = 0.0  # x
    pose_tensor[:, 4] = 0.0  # y
    pose_tensor[:, 5] = torch.sin(half_yaw)  # z
    pose_tensor[:, 6] = torch.cos(half_yaw)  # w

    # Sync local pose to USD *before* adding env_origins
    _sync_pose_to_usd(asset, env_ids, pose_tensor)

    # Transform to simulation frame
    pose_tensor[:, :3] += env.scene.env_origins[env_ids]

    _write_root_pose_with_zero_velocity(asset, pose_tensor, env_ids)


def randomize_object_pose_full(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | list,
    initial_pose: Pose | None = None,
    position_range: dict[str, list[float]] | None = None,
    yaw_range: list[float] | None = None,
) -> None:
    """Randomize both position and yaw of an object around its initial pose.

    Combines position randomization and yaw randomization in a single event.

    Args:
        env: The IsaacLab environment instance.
        env_ids: Indices of environments being reset.
        asset_cfg: Scene entity config identifying the object, or a list of configs.
        initial_pose: The base pose to randomize around.  When ``None``,
            the pose is read from the asset's ``default_root_state``
            (i.e. the ``init_state`` specified in the scene config).
        position_range: Per-axis position offsets, e.g.
            {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.0, 0.0]}
        yaw_range: [min_yaw, max_yaw] in radians. None means no yaw randomization.
    """
    if env_ids is None:
        return

    if isinstance(asset_cfg, list):
        for single_cfg in asset_cfg:
            randomize_object_pose_full(
                env,
                env_ids,
                single_cfg,
                initial_pose,
                position_range,
                yaw_range,
            )
        return

    asset = env.scene[get_scene_entity_name(asset_cfg)]
    num_envs = len(env_ids)

    if initial_pose is not None:
        pose_tensor = initial_pose.to_tensor(device=env.device).repeat(num_envs, 1)
    else:
        default_state = asset.data.default_root_state.torch  # (num_envs, 13)
        pose_tensor = default_state[0, :7].unsqueeze(0).repeat(num_envs, 1)

    # Position randomization
    if position_range is not None:
        x_range = position_range.get("x", [0.0, 0.0])
        y_range = position_range.get("y", [0.0, 0.0])
        z_range = position_range.get("z", [0.0, 0.0])

        pose_tensor[:, 0] += torch.empty(num_envs, device=env.device).uniform_(x_range[0], x_range[1])
        pose_tensor[:, 1] += torch.empty(num_envs, device=env.device).uniform_(y_range[0], y_range[1])
        pose_tensor[:, 2] += torch.empty(num_envs, device=env.device).uniform_(z_range[0], z_range[1])

    # Yaw randomization
    if yaw_range is not None:
        half_yaw = torch.empty(num_envs, device=env.device).uniform_(yaw_range[0], yaw_range[1]) * 0.5
        pose_tensor[:, 3] = 0.0  # x
        pose_tensor[:, 4] = 0.0  # y
        pose_tensor[:, 5] = torch.sin(half_yaw)  # z
        pose_tensor[:, 6] = torch.cos(half_yaw)  # w

    # Sync local pose to USD *before* adding env_origins
    _sync_pose_to_usd(asset, env_ids, pose_tensor)

    # Transform to simulation frame
    pose_tensor[:, :3] += env.scene.env_origins[env_ids]

    _write_root_pose_with_zero_velocity(asset, pose_tensor, env_ids)


def randomize_pickable_xy_align_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | list,
    reference_cfg: SceneEntityCfg,
    z_above_reference: list[float] | tuple[float, float],
    pose_range: dict[str, list[float] | tuple[float, float]] | None = None,
    yaw_range: list[float] | tuple[float, float] | None = None,
    x_offset_range: list[float] | tuple[float, float] | None = None,
    y_offset_range: list[float] | tuple[float, float] | None = None,
) -> None:
    """Randomize a reference asset and place one or more assets around it.

    Intended to run after ``mdp.reset_scene_to_default`` so default heights and
    orientations are valid. The reference keeps its original z/rotation while
    its x/y are randomized by ``pose_range``.

    When ``pose_range`` is omitted or ``None``, the reference's current sim
    position is used as-is (useful when the reference was already placed by an
    earlier event with ``min_separation``).

    ``asset_cfg`` may be a single asset or a list of assets. Each asset is then
    spawned relative to the randomized reference, with per-asset x/y offsets and
    z offsets sampled independently.
    """
    if env_ids is None:
        return

    ref = env.scene[get_scene_entity_name(reference_cfg)]
    asset_cfgs = asset_cfg if isinstance(asset_cfg, list) else [asset_cfg]
    objs = [env.scene[get_scene_entity_name(single_cfg)] for single_cfg in asset_cfgs]
    num_envs = len(env_ids)
    device = env.device

    ref_pos_local = ref.data.root_pos_w.torch[env_ids] - env.scene.env_origins[env_ids]
    ref_quat = ref.data.root_quat_w.torch[env_ids]

    if pose_range is not None:
        xr = pose_range["x"]
        yr = pose_range["y"]
        new_x = torch.empty(num_envs, device=device).uniform_(float(xr[0]), float(xr[1]))
        new_y = torch.empty(num_envs, device=device).uniform_(float(yr[0]), float(yr[1]))
    else:
        new_x = ref_pos_local[:, 0]
        new_y = ref_pos_local[:, 1]
    new_z_ref = ref_pos_local[:, 2]

    ref_pose = torch.zeros(num_envs, 7, device=device)
    ref_pose[:, 0] = new_x
    ref_pose[:, 1] = new_y
    ref_pose[:, 2] = new_z_ref
    ref_pose[:, 3:7] = ref_quat

    z_lo, z_hi = float(z_above_reference[0]), float(z_above_reference[1])
    x_offset_range = x_offset_range or (0.0, 0.0)
    y_offset_range = y_offset_range or (0.0, 0.0)
    x_lo, x_hi = float(x_offset_range[0]), float(x_offset_range[1])
    y_lo, y_hi = float(y_offset_range[0]), float(y_offset_range[1])

    zero_vel = torch.zeros(num_envs, 6, device=device)
    if pose_range is not None:
        for asset, pose_local in ((ref, ref_pose),):
            _sync_pose_to_usd(asset, env_ids, pose_local.clone())
            pose_sim = pose_local.clone()
            pose_sim[:, :3] += env.scene.env_origins[env_ids]
            asset.write_root_pose_to_sim_index(root_pose=pose_sim, env_ids=env_ids)
            _write_root_velocity_if_dynamic(asset, zero_vel, env_ids)

    for obj in objs:
        obj_quat = obj.data.root_quat_w.torch[env_ids]
        obj_pose = ref_pose.clone()
        obj_pose[:, 0] += torch.empty(num_envs, device=device).uniform_(x_lo, x_hi)
        obj_pose[:, 1] += torch.empty(num_envs, device=device).uniform_(y_lo, y_hi)
        obj_pose[:, 2] = new_z_ref + torch.empty(num_envs, device=device).uniform_(z_lo, z_hi)

        if yaw_range is not None:
            yaw_lo, yaw_hi = float(yaw_range[0]), float(yaw_range[1])
            yaw = torch.empty(num_envs, device=device).uniform_(yaw_lo, yaw_hi)
            half_yaw = yaw * 0.5
            obj_pose[:, 3] = 0.0
            obj_pose[:, 4] = 0.0
            obj_pose[:, 5] = torch.sin(half_yaw)
            obj_pose[:, 6] = torch.cos(half_yaw)
        else:
            obj_pose[:, 3:7] = obj_quat

        _sync_pose_to_usd(obj, env_ids, obj_pose.clone())
        obj_pose[:, :3] += env.scene.env_origins[env_ids]
        obj.write_root_pose_to_sim_index(root_pose=obj_pose, env_ids=env_ids)
        _write_root_velocity_if_dynamic(obj, zero_vel, env_ids)


def _normalize_interval_spec(spec: list[float] | tuple[float, float] | list[list[float]]) -> list[tuple[float, float]]:
    """Normalize one-axis range spec into a list of [lo, hi] intervals."""
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ValueError(f"Range tuple must have length 2, got {spec!r}")
        return [(float(spec[0]), float(spec[1]))]
    if not isinstance(spec, list) or len(spec) == 0:
        raise ValueError(f"Range spec must be a non-empty list/tuple, got {spec!r}")

    if len(spec) == 2 and not isinstance(spec[0], (list, tuple)):
        return [(float(spec[0]), float(spec[1]))]

    intervals: list[tuple[float, float]] = []
    for item in spec:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Multi-interval entries must be [lo, hi], got {item!r}")
        intervals.append((float(item[0]), float(item[1])))
    return intervals


def _sample_range_value(
    num_envs: int,
    device: torch.device | str,
    spec: list[float] | tuple[float, float] | list[list[float]],
) -> torch.Tensor:
    """Sample one value per env from single or multi-interval range specs."""
    intervals = _normalize_interval_spec(spec)
    if len(intervals) == 1:
        lo, hi = intervals[0]
        return torch.empty(num_envs, device=device).uniform_(lo, hi)

    lo = torch.tensor([it[0] for it in intervals], dtype=torch.float32, device=device)
    hi = torch.tensor([it[1] for it in intervals], dtype=torch.float32, device=device)
    picks = torch.randint(low=0, high=len(intervals), size=(num_envs,), device=device)
    u = torch.rand(num_envs, device=device)
    return lo[picks] + u * (hi[picks] - lo[picks])


def randomize_object_pose_from_pose_range(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | list,
    initial_pose: Pose | None,
    pose_range: dict[str, list[float] | tuple[float, float]],
    min_separation: float = 0.0,
    max_sample_tries: int = 1000,
) -> None:
    """Randomize a single object's absolute local pose on reset.

    Position keys (``x/y/z``) are interpreted as absolute local-frame ranges.
    Orientation keys (``roll/pitch/yaw``) are interpreted as absolute Euler
    ranges in radians. Any missing orientation key keeps the base orientation.

    When ``initial_pose`` is omitted or ``None``, the base pose is read from the
    asset's configured spawn state (``default_root_state``), not identity. That
    preserves scene YAML orientation (e.g. upside-down) while only randomizing
    axes listed in ``pose_range``.

    ``asset_cfg`` may be a list; each entry is randomized independently.
    When ``min_separation`` > 0 and ``asset_cfg`` is a list, rejection sampling
    ensures all objects are at least ``min_separation`` apart (XY distance).

    Args:
        min_separation: Minimum XY Euclidean distance between any two objects.
            Only effective when ``asset_cfg`` is a list with more than one entry.
        max_sample_tries: Maximum rejection-sampling attempts per object before
            accepting the last sample regardless.
    """
    from isaaclab.utils import math as math_utils
    from isaaclab_arena.utils.pose import Pose

    if env_ids is None:
        return

    if isinstance(asset_cfg, list):
        if min_separation > 0 and len(asset_cfg) > 1:
            _randomize_multi_pose_with_separation(
                env,
                env_ids,
                asset_cfg,
                initial_pose,
                pose_range,
                min_separation,
                max_sample_tries,
            )
            return
        for single_cfg in asset_cfg:
            randomize_object_pose_from_pose_range(env, env_ids, single_cfg, initial_pose, pose_range)
        return

    asset = env.scene[get_scene_entity_name(asset_cfg)]
    num_envs = len(env_ids)
    if initial_pose is None:
        pose_tensor = _default_root_pose_local_from_asset(asset, env_ids)
        if pose_tensor is None:
            pose_tensor = Pose.identity().to_tensor(device=env.device).repeat(num_envs, 1)
        else:
            pose_tensor = pose_tensor.to(device=env.device)
    else:
        pose_tensor = initial_pose.to_tensor(device=env.device).repeat(num_envs, 1)

    for coord_idx, key in enumerate(("x", "y", "z")):
        if key in pose_range:
            pose_tensor[:, coord_idx] = _sample_range_value(num_envs, env.device, pose_range[key])

    base_roll, base_pitch, base_yaw = math_utils.euler_xyz_from_quat(pose_tensor[:, 3:7])
    roll = base_roll.clone()
    pitch = base_pitch.clone()
    yaw = base_yaw.clone()

    if "roll" in pose_range:
        roll = _sample_range_value(num_envs, env.device, pose_range["roll"])
    if "pitch" in pose_range:
        pitch = _sample_range_value(num_envs, env.device, pose_range["pitch"])
    if "yaw" in pose_range:
        yaw = _sample_range_value(num_envs, env.device, pose_range["yaw"])

    pose_tensor[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    _sync_pose_to_usd(asset, env_ids, pose_tensor)
    pose_tensor[:, :3] += env.scene.env_origins[env_ids]

    asset.write_root_pose_to_sim_index(root_pose=pose_tensor, env_ids=env_ids)
    _write_root_velocity_if_dynamic(asset, torch.zeros(num_envs, 6, device=env.device), env_ids)


def _randomize_multi_pose_with_separation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg_list: list,
    initial_pose: Pose | None,
    pose_range: dict[str, list[float] | tuple[float, float]],
    min_separation: float,
    max_sample_tries: int,
) -> None:
    """Place multiple objects via rejection sampling to enforce minimum XY separation.

    For each environment, objects are placed one by one. Each new object is
    sampled repeatedly (up to ``max_sample_tries``) until its XY distance to
    every already-placed object exceeds ``min_separation``.
    """
    from isaaclab.utils import math as math_utils
    from isaaclab_arena.utils.pose import Pose

    num_envs = len(env_ids)
    device = env.device
    n_assets = len(asset_cfg_list)

    # Build per-axis sampling ranges ------------------------------------------------
    xy_keys = ("x", "y", "z")
    ranges = {}
    for key in xy_keys:
        if key in pose_range:
            ranges[key] = pose_range[key]

    ori_keys = ("roll", "pitch", "yaw")
    ori_ranges = {}
    for key in ori_keys:
        if key in pose_range:
            ori_ranges[key] = pose_range[key]

    # accepted_xy: (num_envs, asset_idx, 2) — stores accepted XY for distance checks
    accepted_xy = torch.zeros(num_envs, n_assets, 2, device=device)

    for obj_idx, single_cfg in enumerate(asset_cfg_list):
        asset = env.scene[get_scene_entity_name(single_cfg)]

        # Start from default or identity pose
        if initial_pose is None:
            pose_tensor = _default_root_pose_local_from_asset(asset, env_ids)
            if pose_tensor is None:
                pose_tensor = Pose.identity().to_tensor(device=device).repeat(num_envs, 1)
            else:
                pose_tensor = pose_tensor.to(device=device)
        else:
            pose_tensor = initial_pose.to_tensor(device=device).repeat(num_envs, 1)

        # Sample positions with rejection per env --------------------------------
        # For efficiency, sample all envs in batch and retry only those that fail.
        need_resample = torch.ones(num_envs, dtype=torch.bool, device=device)

        for _attempt in range(max_sample_tries):
            if not need_resample.any():
                break
            n_resample = need_resample.sum().item()

            # Sample candidate XYZ
            for coord_idx, key in enumerate(xy_keys):
                if key in ranges:
                    pose_tensor[need_resample, coord_idx] = _sample_range_value(n_resample, device, ranges[key])

            # Check XY distances against all previously accepted objects
            if obj_idx > 0:
                candidate_xy = pose_tensor[need_resample, :2]  # (n_resample, 2)
                prev_xy = accepted_xy[need_resample, :obj_idx, :]  # (n_resample, obj_idx, 2)
                # Compute pairwise distances: (n_resample, obj_idx)
                dists = torch.norm(candidate_xy.unsqueeze(1) - prev_xy, dim=2)
                too_close = (dists < min_separation).any(dim=1)  # (n_resample,)
                # Update need_resample: only those still too close
                resample_indices = need_resample.nonzero(as_tuple=True)[0]
                need_resample[resample_indices] = too_close
            else:
                need_resample[:] = False

        # Sample orientation
        base_roll, base_pitch, base_yaw = math_utils.euler_xyz_from_quat(pose_tensor[:, 3:7])
        roll, pitch, yaw = base_roll.clone(), base_pitch.clone(), base_yaw.clone()
        if "roll" in ori_ranges:
            roll = _sample_range_value(num_envs, device, ori_ranges["roll"])
        if "pitch" in ori_ranges:
            pitch = _sample_range_value(num_envs, device, ori_ranges["pitch"])
        if "yaw" in ori_ranges:
            yaw = _sample_range_value(num_envs, device, ori_ranges["yaw"])
        pose_tensor[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # Record accepted XY
        accepted_xy[:, obj_idx, :] = pose_tensor[:, :2]

        # Write to sim
        _sync_pose_to_usd(asset, env_ids, pose_tensor)
        pose_tensor[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim_index(root_pose=pose_tensor, env_ids=env_ids)
        _write_root_velocity_if_dynamic(asset, torch.zeros(num_envs, 6, device=device), env_ids)


def randomize_object_pose_from_list(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | list,
    positions: list[list[float]],
    orientations: list[list[float]] | None = None,
    position_range: dict[str, list[float] | tuple[float, float]] | None = None,
    yaw_range: list[float] | tuple[float, float] | None = None,
) -> None:
    """Randomly select position and orientation from discrete lists on reset.

    Each reset, one position is uniformly sampled from ``positions`` and one
    orientation from ``orientations`` independently per environment.

    When ``position_range`` is provided, a uniform random offset is added to the
    sampled position per axis (x/y/z).

    When ``yaw_range`` is provided, a uniform random yaw offset is applied on top
    of the sampled/default orientation.

    ``asset_cfg`` may be a list; each entry is randomized independently.

    When ``orientations`` is omitted or ``None``, the asset's default
    orientation (from the scene YAML ``initial_orientation``) is preserved.

    Args:
        env: The IsaacLab environment instance.
        env_ids: Indices of environments being reset.
        asset_cfg: Scene entity config identifying the object.
        positions: List of [x, y, z] candidate positions.
        orientations: List of [qx, qy, qz, qw] candidate orientations.
            ``None`` keeps the asset's default orientation.
        position_range: Optional per-axis offset ranges added to the sampled
            position, e.g. {"x": [-0.01, 0.01], "y": [-0.02, 0.02]}.
        yaw_range: Optional yaw offset range in radians.
    """
    from isaaclab.utils import math as math_utils

    if env_ids is None:
        return

    if isinstance(asset_cfg, list):
        n_assets = len(asset_cfg)
        n_pos = len(positions)
        if n_assets <= n_pos:
            # Assign unique positions via random permutation (no overlap).
            pos_t = torch.tensor(positions, dtype=torch.float32, device=env.device)
            num_envs = len(env_ids)
            # Generate a random permutation of position indices per env.
            perms = torch.stack([torch.randperm(n_pos, device=env.device) for _ in range(num_envs)])
            for i, single_cfg in enumerate(asset_cfg):
                asset = env.scene[get_scene_entity_name(single_cfg)]
                idx = perms[:, i]
                pose_tensor = torch.zeros(num_envs, 7, device=env.device)
                pose_tensor[:, :3] = pos_t[idx]
                if position_range is not None:
                    xr = position_range.get("x", (0.0, 0.0))
                    yr = position_range.get("y", (0.0, 0.0))
                    zr = position_range.get("z", (0.0, 0.0))
                    pose_tensor[:, 0] += torch.empty(num_envs, device=env.device).uniform_(float(xr[0]), float(xr[1]))
                    pose_tensor[:, 1] += torch.empty(num_envs, device=env.device).uniform_(float(yr[0]), float(yr[1]))
                    pose_tensor[:, 2] += torch.empty(num_envs, device=env.device).uniform_(float(zr[0]), float(zr[1]))
                if orientations is not None:
                    ori_t = torch.tensor(orientations, dtype=torch.float32, device=env.device)
                    ori_idx = torch.randint(0, len(orientations), (num_envs,), device=env.device)
                    pose_tensor[:, 3:7] = ori_t[ori_idx]
                else:
                    default_pose = _default_root_pose_local_from_asset(asset, env_ids)
                    if default_pose is not None:
                        pose_tensor[:, 3:7] = default_pose[:, 3:7]
                    else:
                        pose_tensor[:, 6] = 1.0
                if yaw_range is not None:
                    roll, pitch, base_yaw = math_utils.euler_xyz_from_quat(pose_tensor[:, 3:7])
                    yaw_offset = torch.empty(num_envs, device=env.device).uniform_(
                        float(yaw_range[0]), float(yaw_range[1])
                    )
                    pose_tensor[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, base_yaw + yaw_offset)
                _sync_pose_to_usd(asset, env_ids, pose_tensor)
                pose_tensor[:, :3] += env.scene.env_origins[env_ids]
                _write_root_pose_with_zero_velocity(asset, pose_tensor, env_ids)
        else:
            for single_cfg in asset_cfg:
                randomize_object_pose_from_list(
                    env,
                    env_ids,
                    single_cfg,
                    positions,
                    orientations,
                    position_range=position_range,
                    yaw_range=yaw_range,
                )
        return

    asset = env.scene[get_scene_entity_name(asset_cfg)]
    num_envs = len(env_ids)

    pos_t = torch.tensor(positions, dtype=torch.float32, device=env.device)
    pos_idx = torch.randint(0, len(positions), (num_envs,), device=env.device)

    pose_tensor = torch.zeros(num_envs, 7, device=env.device)
    pose_tensor[:, :3] = pos_t[pos_idx]
    if position_range is not None:
        xr = position_range.get("x", (0.0, 0.0))
        yr = position_range.get("y", (0.0, 0.0))
        zr = position_range.get("z", (0.0, 0.0))
        pose_tensor[:, 0] += torch.empty(num_envs, device=env.device).uniform_(float(xr[0]), float(xr[1]))
        pose_tensor[:, 1] += torch.empty(num_envs, device=env.device).uniform_(float(yr[0]), float(yr[1]))
        pose_tensor[:, 2] += torch.empty(num_envs, device=env.device).uniform_(float(zr[0]), float(zr[1]))

    if orientations is not None:
        ori_t = torch.tensor(orientations, dtype=torch.float32, device=env.device)
        ori_idx = torch.randint(0, len(orientations), (num_envs,), device=env.device)
        pose_tensor[:, 3:7] = ori_t[ori_idx]
    else:
        default_pose = _default_root_pose_local_from_asset(asset, env_ids)
        if default_pose is not None:
            pose_tensor[:, 3:7] = default_pose[:, 3:7]
        else:
            pose_tensor[:, 6] = 1.0  # identity quaternion

    if yaw_range is not None:
        roll, pitch, base_yaw = math_utils.euler_xyz_from_quat(pose_tensor[:, 3:7])
        yaw_offset = torch.empty(num_envs, device=env.device).uniform_(float(yaw_range[0]), float(yaw_range[1]))
        pose_tensor[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, base_yaw + yaw_offset)

    _sync_pose_to_usd(asset, env_ids, pose_tensor)
    pose_tensor[:, :3] += env.scene.env_origins[env_ids]

    _write_root_pose_with_zero_velocity(asset, pose_tensor, env_ids)


def randomize_destination_pose_from_pose_range(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | list,
    initial_pose: Pose | None,
    pose_range: dict[str, list[float] | tuple[float, float]],
) -> None:
    """Second reset event name for ``@destination``; delegates to ``randomize_object_pose_from_pose_range``."""
    randomize_object_pose_from_pose_range(env, env_ids, asset_cfg, initial_pose, pose_range)


def randomize_destination_pose_from_list(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | list,
    positions: list[list[float]],
    orientations: list[list[float]] | None = None,
    position_range: dict[str, list[float] | tuple[float, float]] | None = None,
    yaw_range: list[float] | tuple[float, float] | None = None,
) -> None:
    """Second reset event name for ``@destination``; delegates to ``randomize_object_pose_from_list``."""
    randomize_object_pose_from_list(
        env,
        env_ids,
        asset_cfg,
        positions,
        orientations,
        position_range=position_range,
        yaw_range=yaw_range,
    )


def randomize_object_orientation_from_list(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | list,
    orientations: list[list[float]],
) -> None:
    """Randomly select orientation from a discrete list on reset, preserving position.

    Each reset, one orientation (XYZW quaternion) is uniformly sampled from
    ``orientations`` per environment and applied to the asset. The asset's
    current position is preserved unchanged.

    When ``asset_cfg`` is a list, each asset is randomized independently.

    Args:
        env: The IsaacLab environment instance.
        env_ids: Indices of environments being reset.
        asset_cfg: Scene entity config identifying the object(s).
        orientations: List of [qx, qy, qz, qw] candidate orientations (XYZW).
    """
    if env_ids is None:
        return

    if isinstance(asset_cfg, list):
        for single_cfg in asset_cfg:
            randomize_object_orientation_from_list(env, env_ids, single_cfg, orientations)
        return

    asset = env.scene[get_scene_entity_name(asset_cfg)]
    num_envs = len(env_ids)

    ori_t = torch.tensor(orientations, dtype=torch.float32, device=env.device)
    ori_idx = torch.randint(0, len(orientations), (num_envs,), device=env.device)

    pose_tensor = asset.data.root_pos_w.torch[env_ids, :].clone()
    pose_tensor -= env.scene.env_origins[env_ids]
    local_pose = torch.zeros(num_envs, 7, device=env.device)
    local_pose[:, :3] = pose_tensor[:, :3]
    local_pose[:, 3:7] = ori_t[ori_idx]

    _sync_pose_to_usd(asset, env_ids, local_pose)
    local_pose[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim_index(root_pose=local_pose, env_ids=env_ids)
    _write_root_velocity_if_dynamic(asset, torch.zeros(num_envs, 6, device=env.device), env_ids)


def _openness_to_joint_target(openness, j_min, j_max):
    """Map normalized openness in ``[0, 1]`` to an absolute joint position.

    Mirrors :func:`manaenv.terms.terminations.joint_above_threshold`: when the
    lower limit is negative the joint is treated as inverted (0 = fully open at
    ``j_min``), otherwise 0 = closed at ``j_min``. ``openness`` may be a python
    float or a per-env tensor; ``j_min`` / ``j_max`` are 0-dim tensors, so the
    result broadcasts to the shape of ``openness``.
    """
    span = j_max - j_min
    if float(j_min.item()) < 0.0:
        return j_min + (1.0 - openness) * span
    return j_min + openness * span


def set_articulation_joint_openness_map(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | str,
    joint_openness: dict[str, float] | None = None,
    joint_openness_range: dict[str, list[float]] | None = None,
) -> None:
    """Set per-joint reset openness for an articulation, with optional randomization.

    Different openness value per joint (unlike a single scalar for all joints).
    Authored as a task ``event_component`` reset term referencing the articulation
    via ``@role`` (same pattern as ``randomize_object_pose_*``), e.g.::

        set_articulation_joint_openness_map:
          mode: reset
          asset_cfg: '@containable'
          joint_openness: {drawer_003_joint: 0.75}
          joint_openness_range: {drawer_003_joint: [0.5, 0.9]}

    Args:
        joint_openness: ``{joint_regex: openness}`` fixed openness in ``[0, 1]``.
        joint_openness_range: ``{joint_regex: [lo, hi]}`` uniform range sampled
            **per env, per reset**. Overrides ``joint_openness`` for the same
            joint. Enables domain randomization of the initial articulation pose.

    Each key is a regex matched with ``re.fullmatch`` against
    ``articulation.joint_names``. Run **after** ``mdp.reset_scene_to_default`` so
    the USD default is overridden for the listed joints.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    if not joint_openness and not joint_openness_range:
        return

    entity_name = get_scene_entity_name(asset_cfg)
    articulation = env.scene[entity_name]
    if not hasattr(articulation, "find_joints") or not hasattr(articulation, "write_joint_state_to_sim"):
        return

    joint_pos = as_torch(articulation.data.joint_pos)[env_ids].clone()
    device, dtype = joint_pos.device, joint_pos.dtype
    num_env_ids = joint_pos.shape[0]
    changed = False

    def _resolve(pattern: str) -> list[int]:
        indices, _ = articulation.find_joints([str(pattern).strip()], preserve_order=True)
        return indices

    for pattern, value in (joint_openness or {}).items():
        o = float(max(0.0, min(1.0, float(value))))
        for joint_index in _resolve(pattern):
            limits = as_torch(articulation.data.joint_pos_limits)[0, joint_index, :]
            target = _openness_to_joint_target(o, limits[0], limits[1])
            joint_pos[:, joint_index] = target.to(device=device, dtype=dtype)
            changed = True

    for pattern, rng in (joint_openness_range or {}).items():
        lo, hi = float(max(0.0, min(1.0, float(rng[0])))), float(max(0.0, min(1.0, float(rng[1]))))
        if hi < lo:
            lo, hi = hi, lo
        for joint_index in _resolve(pattern):
            sampled = torch.rand(num_env_ids, device=device, dtype=dtype) * (hi - lo) + lo
            limits = as_torch(articulation.data.joint_pos_limits)[0, joint_index, :]
            j_min = limits[0].to(device=device, dtype=dtype)
            j_max = limits[1].to(device=device, dtype=dtype)
            joint_pos[:, joint_index] = _openness_to_joint_target(sampled, j_min, j_max)
            changed = True

    if not changed:
        return

    articulation.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
    if hasattr(articulation.data, "body_link_pose_w"):
        _ = as_torch(articulation.data.body_link_pose_w)[env_ids]


def set_articulation_joint_position_limit(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg | str,
    joint_name: str,
    lower_limit_deg: float,
    upper_limit_deg: float = 0.0,
) -> None:
    """Set one articulation joint hard limit once at environment startup."""
    del env_ids  # Startup events apply to every cloned environment.

    entity_name = get_scene_entity_name(asset_cfg)
    articulation = env.scene[entity_name]
    if not hasattr(articulation, "find_joints") or not hasattr(articulation, "write_joint_position_limit_to_sim_index"):
        raise TypeError(f"{entity_name!r} does not support articulation joint limits")

    pattern = str(joint_name).strip()
    joint_indices, _ = articulation.find_joints([pattern], preserve_order=True)
    if not joint_indices:
        raise ValueError(f"joint {pattern!r} was not found on articulation {entity_name!r}")

    lower_rad = math.radians(float(lower_limit_deg))
    upper_rad = math.radians(float(upper_limit_deg))
    if lower_rad >= upper_rad:
        raise ValueError("lower_limit_deg must be less than upper_limit_deg")

    joint_pos = as_torch(articulation.data.joint_pos)
    all_env_ids = torch.arange(joint_pos.shape[0], device=joint_pos.device, dtype=torch.long)
    limits = torch.empty((len(all_env_ids), 1, 2), device=joint_pos.device, dtype=joint_pos.dtype)
    limits[..., 0] = lower_rad
    limits[..., 1] = upper_rad
    articulation.write_joint_position_limit_to_sim_index(
        limits=limits,
        joint_ids=[int(joint_indices[0])],
        env_ids=all_env_ids,
        warn_limit_violation=False,
    )


def sample_turnable_twist_goal(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | str,
    joint_name: str = "RevoluteJoint",
    initial_temperature_range: list[int] | None = None,
    target_delta_temperature_range: list[int] | None = None,
    min_delta_temperature: int = 3,
    temperature_max: int = 99,
    seed: int | None = None,
    store_attr: str = "EPISODE_TURNABLE_TWIST_GOAL_ATTR",
) -> None:
    """Sample initial + target knob panel temperature per reset.

    Pins ``TurnableTwistGoal`` plus a rendered ``TurnableTwistEpisodeRecord``
    (task + subtask instructions with concrete temperatures) for evaluation.
    Applies the sampled **initial** joint pose after
    ``mdp.reset_scene_to_default``.

    Samples an initial panel temperature and a target within
    ``target_delta_temperature_range`` of that initial reading. Optional
    ``seed`` overrides the run master seed. Otherwise the shared
    reproducibility contract derives an independent stream per env and episode.
    Instruction templates are read from the bound task's ``instruction`` / ``subtasks``
    runtime cfg (authored under ``instruction_component`` in the graph YAML) and
    rendered with ``{initial}`` / ``{target}`` placeholders.
    """

    from manaenv.assets.objects.object_library import KnobA
    from manaenv.pipeline.runtime.active_instruction import write_active_instruction
    from manaenv.pipeline.skills.twist_runtime.twist_goal import (
        EPISODE_TURNABLE_TWIST_GOAL_ATTR,
        TurnableTwistGoal,
        build_episode_record,
        resolve_task_instruction_templates,
        sample_twist_goal_pair_temperature,
        write_episode_record_store,
        write_goal_store,
    )

    if env_ids is None or len(env_ids) == 0:
        return

    resolved_store = store_attr
    if resolved_store == "EPISODE_TURNABLE_TWIST_GOAL_ATTR":
        resolved_store = EPISODE_TURNABLE_TWIST_GOAL_ATTR

    entity_name = get_scene_entity_name(asset_cfg)
    articulation = env.scene[entity_name]
    if not hasattr(articulation, "find_joints") or not hasattr(articulation, "write_joint_state_to_sim"):
        return

    pattern = str(joint_name or "RevoluteJoint").strip()
    joint_indices, _ = articulation.find_joints([pattern], preserve_order=True)
    if not joint_indices:
        return
    joint_index = int(joint_indices[0])

    init_temp_rng = tuple(initial_temperature_range or [10, 50])
    delta_temp_rng = tuple(target_delta_temperature_range or [-10, 10])
    joint_pos = as_torch(articulation.data.joint_pos)[env_ids].clone()
    device, dtype = joint_pos.device, joint_pos.dtype
    task_template, node_templates = resolve_task_instruction_templates(env)

    ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
    for local_row, env_id in enumerate(ids):
        rng = _task_sample_rng(
            env,
            f"turnable_twist_goal:{resolved_store}:{entity_name}:{pattern}",
            seed,
            env_id=int(env_id),
        )
        initial_temp, target_temp = sample_twist_goal_pair_temperature(
            initial_temperature_range=(int(init_temp_rng[0]), int(init_temp_rng[1])),
            target_delta_temperature_range=(int(delta_temp_rng[0]), int(delta_temp_rng[1])),
            min_delta_temperature=int(min_delta_temperature),
            temperature_max=int(temperature_max),
            rng=rng,
        )
        goal = TurnableTwistGoal(
            joint_name=pattern,
            initial_temperature=int(initial_temp),
            target_temperature=int(target_temp),
            temperature_max=int(temperature_max),
        )
        write_goal_store(env, int(env_id), goal, store_attr=resolved_store)
        episode_record = build_episode_record(
            goal,
            task_template=task_template,
            node_templates=node_templates,
        )
        write_episode_record_store(env, int(env_id), episode_record)
        # reset_owned: planner/reset clears non-owned active-instruction slots.
        write_active_instruction(
            env,
            int(env_id),
            episode_record.task_instruction,
            num_envs=int(env.num_envs),
            reset_owned=True,
        )

        joint_pos[local_row, joint_index] = torch.tensor(
            KnobA.display_value_to_joint_angle(initial_temp),
            device=device,
            dtype=dtype,
        )

    articulation.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
    if hasattr(articulation.data, "body_link_pose_w"):
        _ = as_torch(articulation.data.body_link_pose_w)[env_ids]


def set_arm_joint_pose_to_sim(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Write a fixed joint pose straight into the sim on reset.

    Replaces the contrib ``set_default_joint_pose`` + ``randomize_joint_by_gaussian_offset``
    pair for ManaEnv scripted evaluation. Those two split the work
    (``set_default_joint_pose`` only updates the ``default_joint_pos`` buffer;
    ``randomize_joint_by_gaussian_offset`` is the only one that writes joints to
    sim), and the randomize half does not land the buffered pose under ManaEnv's
    env stack — so the arm always reset to ~0 (USD spawn pose) regardless of YAML
    ``initial_joint_positions``. This single event writes ``default_pose``
    directly via ``write_joint_state_to_sim_index``, so the configured pose
    actually reaches the sim.

    Args:
        env: The environment.
        env_ids: Indices of environments being reset (``None`` -> all envs).
        default_pose: Joint positions ordered to match the articulation's
            ``joint_names`` (panda_joint1..7, panda_finger_joint1/2 for Franka).
        asset_cfg: Scene entity config identifying the articulation.
    """
    name = get_scene_entity_name(asset_cfg)
    asset = env.scene[name]
    # Duck-type rather than isinstance: the runtime asset is a backend-specific
    # Articulation (e.g. isaaclab_physx...Articulation) whose class name matches
    # but is not the isaaclab.assets.Articulation factory class, so isinstance
    # would falsely reject it. Match how other manaenv events probe articulations.
    if not (hasattr(asset, "write_joint_state_to_sim_index") and hasattr(asset, "joint_names")):
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device).reshape(-1)
    if env_ids.numel() == 0:
        return

    num_envs = env_ids.shape[0]
    device = env.device
    dof = len(asset.joint_names)
    pose = torch.as_tensor(default_pose, dtype=torch.float32, device=device)
    if pose.numel() < dof:
        # Pad short specs with zeros so we never index past the provided values.
        pose = torch.cat([pose, torch.zeros(dof - pose.numel(), device=device)])
    pos = pose[:dof].unsqueeze(0).expand(num_envs, -1).clone()
    vel = torch.zeros_like(pos)

    # Write the physics state AND the PD targets. State alone is not enough: under
    # the high-PD Franka (stiffness=400) the controller would pull the arm back to
    # its stale target (~0) on the next physics step. The contrib randomize event
    # set both; this single event must too.
    asset.write_joint_state_to_sim_index(position=pos, velocity=vel, env_ids=env_ids)
    asset.set_joint_position_target_index(target=pos, env_ids=env_ids)
    asset.set_joint_velocity_target_index(target=vel, env_ids=env_ids)


_TURNTABLE_PICK_STATE_ATTR = "_mana_pick_item_from_turntable_state"


def sample_pick_item_from_turntable(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    object_names: list[str],
    object_colors: list[str] | None = None,
    turntable_name: str = "turntable",
    joint_name: str = "RevoluteJoint",
    index_log_key: str = "turntable_target_index",
    instruction_template: str = (
        "Pick the {color} brick from the turntable and place it on the plate, then return the arm home."
    ),
    seed: int | None = None,
) -> None:
    """Sample one target brick index per env and reset the turntable joint to zero.

    Commits ``index_log_key`` as int ``[num_envs, 1]`` aligned with ``object_names``,
    writes the active instruction, and stores an overlay plan under
    :data:`_TURNTABLE_PICK_STATE_ATTR`. All bricks stay on the turntable; only the
    instruction / success terms narrow to the sampled target.
    """
    if env_ids is None:
        return
    names = [str(name) for name in object_names]
    if not names:
        raise ValueError("sample_pick_item_from_turntable requires a non-empty object_names")

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    colors = [str(c) for c in (object_colors or [])]
    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    state = getattr(env, _TURNTABLE_PICK_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _TURNTABLE_PICK_STATE_ATTR, state)
    plan_by_env: dict[int, str] = state.setdefault("plan", {})
    previous: dict[int, int] = state.setdefault("previous", {})

    index_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(index_log_key), seed, env_id=env_id)
        idx = rng.randrange(len(names))
        if idx == previous.get(env_id) and len(names) > 1:
            idx = (idx + 1 + rng.randrange(len(names) - 1)) % len(names)
        previous[env_id] = idx
        index_tensor[env_id, 0] = idx
        color = colors[idx] if idx < len(colors) else "target"
        write_active_instruction(
            env,
            env_id,
            str(instruction_template).format(color=color),
            num_envs=num_envs,
            reset_owned=True,
        )
        plan_by_env[env_id] = f"pick {color} brick"

    write_committed(env, str(index_log_key), index_tensor, env_ids=ids, frame_idx=-1)

    if scene_has_asset(env.scene, str(turntable_name)):
        turntable = env.scene[str(turntable_name)]
        if hasattr(turntable, "find_joints") and hasattr(turntable, "write_joint_state_to_sim"):
            pattern = str(joint_name or "RevoluteJoint").strip()
            joint_ids, _ = turntable.find_joints([pattern], preserve_order=True)
            if joint_ids:
                joint_index = int(joint_ids[0])
                zero = torch.zeros((ids.shape[0], 1), device=device, dtype=torch.float32)
                # PhysX Warp joint kernels expect int32 selectors (int64 fails at wp.launch).
                jids = torch.tensor([joint_index], device=device, dtype=torch.int32)
                turntable.write_joint_state_to_sim(zero, zero, joint_ids=jids, env_ids=ids)
                if hasattr(turntable, "set_joint_position_target"):
                    turntable.set_joint_position_target(zero, joint_ids=jids, env_ids=ids)


_DEFAULT_PLATE_TURNTABLE_OMEGA_ATTR = "_mana_plate_turntable_omega"


def sample_turntable_angular_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    angular_velocity_ranges: list[list[float]] | list[tuple[float, float]],
    state_attr: str = _DEFAULT_PLATE_TURNTABLE_OMEGA_ATTR,
    turntable_name: str | None = None,
    joint_name: str = "RevoluteJoint",
    randomize_sign: bool = False,
    seed: int | None = None,
) -> None:
    """Sample per-env turntable angular velocity from one or more ranges.

    Each env first picks a range uniformly, then samples ``uniform(lo, hi)``
    inside it. When ``randomize_sign`` is true, the sampled magnitude is
    negated with probability 1/2 (CW / CCW). Values are stored on ``env`` as a
    float tensor ``[num_envs]`` under ``state_attr`` for :func:`drive_turntable`
    to consume. When ``turntable_name`` is set, that articulation's revolute
    joint is reset to zero (same as :func:`sample_pick_item_from_turntable`).
    """
    if env_ids is None:
        return
    ranges: list[tuple[float, float]] = []
    for raw in angular_velocity_ranges or []:
        if raw is None or len(raw) < 2:
            raise ValueError(f"sample_turntable_angular_velocity: invalid range {raw!r}")
        lo, hi = float(raw[0]), float(raw[1])
        if hi < lo:
            lo, hi = hi, lo
        ranges.append((lo, hi))
    if not ranges:
        raise ValueError("sample_turntable_angular_velocity requires non-empty angular_velocity_ranges")

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    omega = getattr(env, str(state_attr), None)
    if not isinstance(omega, torch.Tensor) or omega.numel() != num_envs:
        omega = torch.zeros(num_envs, dtype=torch.float32, device=device)
        setattr(env, str(state_attr), omega)
    elif omega.device != device:
        omega = omega.to(device=device)
        setattr(env, str(state_attr), omega)

    flip_sign = bool(randomize_sign)
    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(state_attr), seed, env_id=env_id)
        lo, hi = ranges[rng.randrange(len(ranges))]
        value = float(rng.uniform(lo, hi))
        if flip_sign and rng.random() < 0.5:
            value = -value
        omega[env_id] = value

    if turntable_name and scene_has_asset(env.scene, str(turntable_name)):
        turntable = env.scene[str(turntable_name)]
        if hasattr(turntable, "find_joints") and hasattr(turntable, "write_joint_state_to_sim"):
            pattern = str(joint_name or "RevoluteJoint").strip()
            joint_ids, _ = turntable.find_joints([pattern], preserve_order=True)
            if joint_ids:
                joint_index = int(joint_ids[0])
                zero = torch.zeros((ids.shape[0], 1), device=device, dtype=torch.float32)
                # PhysX Warp joint kernels expect int32 selectors (int64 fails at wp.launch).
                jids = torch.tensor([joint_index], device=device, dtype=torch.int32)
                turntable.write_joint_state_to_sim(zero, zero, joint_ids=jids, env_ids=ids)
                if hasattr(turntable, "set_joint_position_target"):
                    turntable.set_joint_position_target(zero, joint_ids=jids, env_ids=ids)


def drive_turntable(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    object_name: str = "turntable",
    joint_name: str = "RevoluteJoint",
    angular_velocity: float = 0.4,
    angular_velocity_attr: str | None = None,
    carry_object_names: list[str] | None = None,
    carry_release_height: float = 0.05,
    carry_radius: float | None = None,
) -> None:
    """Advance the turntable revolute joint each interval tick (continuous spin).

    The USD joint typically has no actuator drive, so this writes joint state
    directly (``angular_velocity * step_dt``). When ``angular_velocity_attr`` is
    set, per-env omegas are read from that env attribute (see
    :func:`sample_turntable_angular_velocity`) and the scalar
    ``angular_velocity`` is ignored. When ``carry_object_names`` is set, free
    rigid bodies on the disk are kinematically orbited by the same delta so
    they stay with the platform until lifted above ``carry_release_height`` over
    the turntable root z. Optional ``carry_radius`` restricts carry to objects
    whose XY distance to the turntable center is within that radius.
    """

    # IsaacLab EventManager calls interval terms with env_ids=None when
    # ``is_global_time: true`` (see event_manager.apply interval branch). Expand
    # to all envs — same contract as customer_buy / restock_shelf.
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    if env_ids.numel() == 0:
        return
    if not scene_has_asset(env.scene, str(object_name)):
        return
    articulation = env.scene[str(object_name)]
    if not hasattr(articulation, "find_joints") or not hasattr(articulation, "write_joint_state_to_sim"):
        return

    ids = env_ids
    pattern = str(joint_name or "RevoluteJoint").strip()
    joint_ids, _ = articulation.find_joints([pattern], preserve_order=True)
    if not joint_ids:
        return
    joint_index = int(joint_ids[0])
    # PhysX Warp joint kernels expect int32 selectors (int64 fails at wp.launch).
    jids = torch.tensor([joint_index], device=env.device, dtype=torch.int32)

    step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
    if step_dt <= 0.0:
        cfg = getattr(env, "cfg", None)
        sim = getattr(cfg, "sim", None) if cfg is not None else None
        sim_dt = getattr(sim, "dt", None) if sim is not None else None
        decimation = getattr(cfg, "decimation", None) if cfg is not None else None
        if isinstance(sim_dt, (int, float)) and sim_dt > 0:
            step_dt = float(sim_dt) * (
                float(decimation) if isinstance(decimation, (int, float)) and decimation > 0 else 1.0
            )
    if step_dt <= 0.0:
        step_dt = 0.02

    if angular_velocity_attr:
        omega_buf = getattr(env, str(angular_velocity_attr), None)
        if not isinstance(omega_buf, torch.Tensor) or omega_buf.numel() < int(env.num_envs):
            omega_per_env = torch.full((ids.shape[0],), float(angular_velocity), device=env.device, dtype=torch.float32)
        else:
            omega_per_env = omega_buf.to(device=env.device, dtype=torch.float32)[ids]
    else:
        omega_per_env = torch.full((ids.shape[0],), float(angular_velocity), device=env.device, dtype=torch.float32)

    dtheta_per_env = omega_per_env * step_dt
    cur = articulation.data.joint_pos.torch[ids, joint_index].reshape(-1, 1)
    new_pos = cur + dtheta_per_env.reshape(-1, 1)
    vel = omega_per_env.reshape(-1, 1)
    articulation.write_joint_state_to_sim(new_pos, vel, joint_ids=jids, env_ids=ids)
    if hasattr(articulation, "set_joint_position_target"):
        articulation.set_joint_position_target(new_pos, joint_ids=jids, env_ids=ids)
    if hasattr(articulation, "set_joint_velocity_target"):
        articulation.set_joint_velocity_target(vel, joint_ids=jids, env_ids=ids)

    carry = [str(n) for n in (carry_object_names or []) if str(n)]
    if not carry:
        return

    origins = env.scene.env_origins
    turn_pos_w = articulation.data.root_pos_w.torch
    release = float(carry_release_height)
    radius = float(carry_radius) if carry_radius is not None else None
    radius_sq = radius * radius if radius is not None else None

    for name in carry:
        if not scene_has_asset(env.scene, name):
            continue
        asset = env.scene[name]
        root_w = asset.data.root_pos_w.torch
        root_q = asset.data.root_quat_w.torch
        for i, env_id_t in enumerate(ids):
            env_id = int(env_id_t.item())
            dtheta = float(dtheta_per_env[i].item())
            if abs(dtheta) < 1e-9:
                continue
            center = turn_pos_w[env_id, :2]
            # Release once the object is clearly lifted off the disk.
            if float(root_w[env_id, 2] - turn_pos_w[env_id, 2]) > release:
                continue
            dx = float(root_w[env_id, 0] - center[0])
            dy = float(root_w[env_id, 1] - center[1])
            if radius_sq is not None and (dx * dx + dy * dy) > radius_sq:
                continue
            cos_t = math.cos(dtheta)
            sin_t = math.sin(dtheta)
            new_x = float(center[0]) + cos_t * dx - sin_t * dy
            new_y = float(center[1]) + sin_t * dx + cos_t * dy
            local = torch.zeros(7, dtype=torch.float32, device=env.device)
            local[0] = new_x - float(origins[env_id, 0])
            local[1] = new_y - float(origins[env_id, 1])
            local[2] = float(root_w[env_id, 2] - origins[env_id, 2])
            local[3:] = root_q[env_id]
            _write_single_env_local_pose(env, asset, env_id, local)


_TRACK_OBJECT_UNDER_CUP_STATE_ATTR = "_mana_track_object_under_cup_state"
# Slot index 0..8 row-major. This booth: smaller x = nearer = front;
# larger y = left; smaller y = right (operator view).
_TRACK_OBJECT_UNDER_CUP_SLOT_LABELS = (
    "front_left",
    "front_center",
    "front_right",
    "mid_left",
    "mid_center",
    "mid_right",
    "back_left",
    "back_center",
    "back_right",
)


def sample_track_object_under_cup(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    cup_target_name: str = "cup_target",
    cup_distractor_name: str = "cup_distractor",
    hidden_object_name: str = "hidden_object",
    grid_center: list[float] | None = None,
    row_spacing: float = 0.12,
    column_spacing: float = 0.12,
    # Inverted cup root ≈ table_z (-0.240423) + cup height (~0.0813).
    cup_height: float = -0.159,
    hidden_height: float = -0.225423,
    cup_orientation: list[float] | None = None,
    hidden_orientation: list[float] | None = None,
    count_choices: list[int] | None = None,
    count_log_key: str = "target_swap_count",
    target_slot_log_key: str = "target_cup_slot",
    distractor_slot_log_key: str = "distractor_cup_slot",
    hidden_slot_log_key: str = "hidden_slot",
    away_z: float = -10.0,
    instruction_template: str = (
        "Cover the dice with the target cup, swap the two cups once, lift "
        "the cup that still covers the dice, then return home."
    ),
    seed: int | None = None,
) -> None:
    """Sample 3x3 grid slots for two cups + hidden dice and a swap-count target.

    Per reset: draws three distinct cells on a 3x3 grid (row0=near/smaller-x=
    front, row2=far/larger-x=back; col0=larger-y=left), places cups upside-down
    and the hidden object upright, commits slot indices and ``count_choices``,
    and writes the active instruction. Inactive cup-pool slots are parked below
    the floor.
    """
    if env_ids is None:
        return

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    counts = [int(c) for c in (count_choices or [1])]
    if not counts:
        raise ValueError("sample_track_object_under_cup requires a non-empty count_choices list")
    if any(c < 1 for c in counts):
        raise ValueError(f"sample_track_object_under_cup count_choices must be >= 1, got {counts}")

    # Under this booth placement, smaller env-x is nearer the operator / robot.
    center = [float(value) for value in (grid_center or [0.12, -0.268246])]
    if len(center) != 2:
        raise ValueError("sample_track_object_under_cup grid_center must contain [x, y]")
    # Default upside-down (XYZW), matching inverted_triangle_cups.
    cup_quat = [float(v) for v in (cup_orientation or [0.0, 1.0, 0.0, 0.0])]
    hidden_quat = [float(v) for v in (hidden_orientation or [0.0, 0.0, 0.0, 1.0])]
    if len(cup_quat) != 4 or len(hidden_quat) != 4:
        raise ValueError("sample_track_object_under_cup orientations must be XYZW length 4")

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device
    rows = columns = 3
    cell_count = rows * columns

    def cell_xy(slot: int) -> tuple[float, float]:
        # Row 0 = near robot (smaller x) = front; row 2 = far = back.
        # Col 0 = larger y = left; col 2 = smaller y = right.
        row = int(slot) // columns
        column = int(slot) % columns
        x = center[0] + (row - 0.5 * (rows - 1)) * float(row_spacing)
        y = center[1] + (0.5 * (columns - 1) - column) * float(column_spacing)
        return x, y

    state = getattr(env, _TRACK_OBJECT_UNDER_CUP_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _TRACK_OBJECT_UNDER_CUP_STATE_ATTR, state)
    target_cup_by_env: dict[int, str] = state.setdefault("target_cup", {})
    distractor_cup_by_env: dict[int, str] = state.setdefault("distractor_cup", {})
    count_text_by_env: dict[int, str] = state.setdefault("count_text", {})

    count_tensor = torch.zeros(num_envs, dtype=torch.int64, device=device)
    target_slot_tensor = torch.zeros(num_envs, dtype=torch.int64, device=device)
    distractor_slot_tensor = torch.zeros(num_envs, dtype=torch.int64, device=device)
    hidden_slot_tensor = torch.zeros(num_envs, dtype=torch.int64, device=device)

    logical_names = [str(cup_target_name), str(cup_distractor_name), str(hidden_object_name)]

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(count_log_key), seed, env_id=env_id)
        slots = rng.sample(range(cell_count), 3)
        target_slot, distractor_slot, hidden_slot = slots[0], slots[1], slots[2]
        count = counts[rng.randrange(len(counts))]

        count_tensor[env_id] = count
        target_slot_tensor[env_id] = target_slot
        distractor_slot_tensor[env_id] = distractor_slot
        hidden_slot_tensor[env_id] = hidden_slot

        placements = {
            str(cup_target_name): (*cell_xy(target_slot), float(cup_height), *cup_quat),
            str(cup_distractor_name): (*cell_xy(distractor_slot), float(cup_height), *cup_quat),
            str(hidden_object_name): (*cell_xy(hidden_slot), float(hidden_height), *hidden_quat),
        }

        for logical_name in logical_names:
            slot_names = _grid_cache_slot_names(env, logical_name)
            if not slot_names:
                if not scene_has_asset(env.scene, logical_name):
                    continue
                slot_names = [logical_name]
            active_name = resolve_active_scene_entity_name(env, logical_name, env_id=env_id)
            pose_vals = placements[logical_name]
            for slot_index, slot_name in enumerate(slot_names):
                asset = env.scene[slot_name]
                if slot_name == active_name or (
                    active_name == logical_name and slot_index == 0 and len(slot_names) == 1
                ):
                    pose = torch.as_tensor(pose_vals, dtype=torch.float32, device=device)
                    _set_pose_driven_asset_visibility(asset, True, env_id)
                    _write_single_env_local_pose(env, asset, env_id, pose)
                else:
                    parked = torch.zeros(7, dtype=torch.float32, device=device)
                    parked[0] = 0.15 * float(slot_index + 1)
                    parked[2] = float(away_z) - 0.1 * float(slot_index + 1)
                    parked[6] = 1.0
                    _set_pose_driven_asset_visibility(asset, False, env_id)
                    _write_single_env_local_pose(env, asset, env_id, parked)

        instruction = str(instruction_template).format(n=count)
        write_active_instruction(env, env_id, instruction, num_envs=num_envs, reset_owned=True)
        t_label = _TRACK_OBJECT_UNDER_CUP_SLOT_LABELS[target_slot]
        d_label = _TRACK_OBJECT_UNDER_CUP_SLOT_LABELS[distractor_slot]
        h_label = _TRACK_OBJECT_UNDER_CUP_SLOT_LABELS[hidden_slot]
        count_text_by_env[env_id] = str(count)
        target_cup_by_env[env_id] = t_label
        distractor_cup_by_env[env_id] = d_label
        manaprint(
            f"INFO: [track_object_under_cup] env={env_id} swaps={count} "
            f"target={t_label} distractor={d_label} hidden={h_label}"
        )

    write_committed(env, str(count_log_key), count_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(target_slot_log_key), target_slot_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(distractor_slot_log_key), distractor_slot_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(hidden_slot_log_key), hidden_slot_tensor, env_ids=ids, frame_idx=-1)


def _find_particle_cloth_mesh_paths(stage: Any, root_path: str) -> list[str]:
    """Return prim paths under ``root_path`` that carry PhysxParticleClothAPI."""
    try:
        from pxr import Usd
    except Exception:
        return []

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []
    paths: list[str] = []
    for prim in Usd.PrimRange(root):
        schemas = set(prim.GetAppliedSchemas()) if hasattr(prim, "GetAppliedSchemas") else set()
        if prim.HasAPI("PhysxParticleClothAPI") or "PhysxParticleClothAPI" in schemas:
            paths.append(str(prim.GetPath()))
    if not paths:
        for rel in ("towel", "towel_b_01", "World/towel", "cloth", "World/cloth"):
            candidate = f"{root_path}/{rel}"
            prim = stage.GetPrimAtPath(candidate)
            if prim and prim.IsValid():
                paths.append(candidate)
                break
    return paths


def _gf_matrix_to_rt(delta: Any) -> tuple[Any, Any]:
    """Convert a Gf.Matrix4d rigid transform to (R_3x3 list, t_3 list)."""
    rot = delta.ExtractRotationMatrix()
    tr = delta.ExtractTranslation()
    r = [[float(rot[i][j]) for j in range(3)] for i in range(3)]
    t = [float(tr[0]), float(tr[1]), float(tr[2])]
    return r, t


def _world_pose_to_gf_matrix(pos_xyz: Any, quat_xyzw: Any) -> Any:
    """Build a Gf.Matrix4d from world translation + XYZW quaternion."""

    from pxr import Gf

    x, y, z, w = (float(quat_xyzw[i]) for i in range(4))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n > 1e-12:
        w, x, y, z = w / n, x / n, y / n, z / n
    rot = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    return Gf.Matrix4d(rot, Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])))


def _publish_cloth_scene_state(env: ManagerBasedEnv) -> bool:
    """Publish teleported cloth through the environment-owned boundary.

    Async environments own Fabric isolation via ``publish_fabric``; synchronous
    environments retain ``SimulationContext.forward``. Event terms must not
    select either backend directly.
    """
    try:
        publish_scene_state(env)
        return True
    except Exception as exc:
        manaprint(f"WARNING: scene-state publication failed: {exc}")
        return False


def _strip_cloth_usd_xy_keep_particles(env: ManagerBasedEnv, stage: Any, cloth_path: str) -> tuple[int, bool]:
    """Pin cloth USD to scene-default XY/yaw without permanently moving PhysX particles.

    Hang reset authors shared ``dx/dy`` on the cloth USD root (needed so PhysX
    does not yank during the particle delta). Fabric/Hydra can then re-apply that
    translate on top of particle-driven mesh — PhysX contact stays on the
    bar while the *visible* towel is shifted. Snapshot → USD default → restore
    particles strips the render bias without changing contact.
    """
    defaults = _cloth_usd_defaults_cache(env)
    base = _ensure_cloth_usd_default(stage, cloth_path, defaults)
    if base is None:
        return 0, False
    snap = _snapshot_particle_cloth_world_positions(stage, cloth_path)
    if not snap:
        return 0, False
    bp = base["pos"]
    _set_cloth_usd_absolute_pose(
        stage,
        cloth_path,
        x=float(bp[0]),
        y=float(bp[1]),
        z=float(bp[2]),
        yaw_rad=0.0,
        base=base,
        move_particles=False,
    )
    n_restored = _restore_particle_cloth_world_positions(stage, cloth_path, snap)
    fabric_ok = _publish_cloth_scene_state(env)
    return n_restored, fabric_ok


def _move_particle_cloth_by_rt(stage: Any, root_path: str, r_list: Any, t_list: Any) -> int:
    """Apply ``p' = R @ p + t`` to live PhysX particle cloth under ``root_path``."""
    moved_meshes = 0
    for mesh_path in _find_particle_cloth_mesh_paths(stage, root_path):
        try:
            from isaacsim.core.prims import ClothPrim

            view = ClothPrim(
                prim_paths_expr=mesh_path,
                name=f"mana_cloth_{mesh_path.replace('/', '_')}_{id(stage) & 0xFFFF}",
                reset_xform_properties=False,
            )
            try:
                view.initialize()
            except Exception as exc:
                manaprint(f"WARNING: ClothPrim.initialize failed for {mesh_path}: {exc}")
                continue
            if not bool(getattr(view, "is_physics_handle_valid", lambda: False)()):
                manaprint(f"WARNING: ClothPrim physics handle invalid for {mesh_path}")
                continue
            if not hasattr(view, "_max_particles_per_cloth"):
                manaprint(f"WARNING: ClothPrim not fully initialized for {mesh_path}")
                continue
            pos = view.get_world_positions()
            if pos is None:
                manaprint(f"WARNING: ClothPrim.get_world_positions returned None for {mesh_path}")
                continue
            pos = as_torch(pos)
            if not isinstance(pos, torch.Tensor):
                pos = torch.as_tensor(pos, dtype=torch.float32)
            else:
                pos = pos.detach()
            device = pos.device
            flat = pos.reshape(-1, 3).to(dtype=torch.float32)
            if flat.shape[0] < 1:
                continue
            r = torch.tensor(r_list, dtype=torch.float32, device=device)
            t = torch.tensor(t_list, dtype=torch.float32, device=device)
            # p' = R @ p + t  (column-vector convention) → rows: p @ R.T + t
            moved = flat @ r.T + t
            view.set_world_positions(moved.reshape(pos.shape[0], -1, 3) if pos.ndim == 3 else moved.reshape(1, -1, 3))
            moved_meshes += 1
        except Exception as exc:
            manaprint(f"WARNING: particle cloth move failed for {mesh_path}: {exc}")
            continue
    return moved_meshes


def _z_yaw_from_quat_xyzw(quat_xyzw: Any) -> float:
    """Extract Z yaw (radians) from an XYZW quaternion."""

    x, y, z, w = (float(quat_xyzw[i]) for i in range(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle_rad(a: float) -> float:

    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _move_particle_cloth_about_pivot(
    stage: Any,
    root_path: str,
    *,
    old_pivot_xyz: tuple[float, float, float],
    new_pivot_xyz: tuple[float, float, float],
    dyaw_rad: float,
) -> int:
    """Move particles with hang-centric rigid motion: ``p' = Rz(dyaw)(p-old)+new``.

    Unlike ``new_xf * inv(old_xf)`` about a cached cloth/bar matrix (whose
    translation often does not match the live particle cloud), this rotates
    about the hang pivot so a draped towel stays on the bar under shared yaw.
    """

    dyaw = _wrap_angle_rad(dyaw_rad)
    c, s = math.cos(dyaw), math.sin(dyaw)
    r_list = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    ox, oy, oz = (float(v) for v in old_pivot_xyz)
    nx, ny, nz = (float(v) for v in new_pivot_xyz)
    # t = new_pivot - R @ old_pivot
    t_list = [
        nx - (c * ox - s * oy),
        ny - (s * ox + c * oy),
        nz - oz,
    ]
    return _move_particle_cloth_by_rt(stage, root_path, r_list, t_list)


def _snapshot_particle_cloth_world_positions(stage: Any, root_path: str) -> dict[str, Any]:
    """Capture live PhysX particle world positions per cloth mesh path."""
    out: dict[str, Any] = {}
    for mesh_path in _find_particle_cloth_mesh_paths(stage, root_path):
        try:
            from isaacsim.core.prims import ClothPrim

            view = ClothPrim(
                prim_paths_expr=mesh_path,
                name=f"mana_cloth_snap_{mesh_path.replace('/', '_')}_{id(stage) & 0xFFFF}",
                reset_xform_properties=False,
            )
            try:
                view.initialize()
            except Exception as exc:
                manaprint(f"WARNING: ClothPrim.initialize failed for snapshot {mesh_path}: {exc}")
                continue
            if not bool(getattr(view, "is_physics_handle_valid", lambda: False)()):
                continue
            pos = as_torch(view.get_world_positions())
            if pos is None:
                continue
            if isinstance(pos, torch.Tensor):
                out[mesh_path] = pos.detach().clone()
            else:
                out[mesh_path] = torch.as_tensor(pos, dtype=torch.float32).clone()
        except Exception as exc:
            manaprint(f"ERROR: particle cloth snapshot failed for {mesh_path}: {exc}")
            continue
    return out


def _restore_particle_cloth_world_positions(stage: Any, root_path: str, snap: dict[str, Any]) -> int:
    """Write back particle world positions from ``_snapshot_particle_cloth_world_positions``."""
    if not snap:
        return 0
    restored = 0
    for mesh_path, pos in snap.items():
        try:
            from isaacsim.core.prims import ClothPrim

            view = ClothPrim(
                prim_paths_expr=mesh_path,
                name=f"mana_cloth_restore_{mesh_path.replace('/', '_')}_{id(stage) & 0xFFFF}",
                reset_xform_properties=False,
            )
            try:
                view.initialize()
            except Exception:
                continue
            if not bool(getattr(view, "is_physics_handle_valid", lambda: False)()):
                continue
            pos = as_torch(pos)
            if isinstance(pos, torch.Tensor):
                pos_t = pos
            else:
                pos_t = torch.as_tensor(pos, dtype=torch.float32)
            view.set_world_positions(pos_t)
            # Drop leftover velocities from the previous episode if the API exists.
            try:
                flat = pos_t.reshape(-1, 3)
                zeros = torch.zeros_like(flat)
                if hasattr(view, "set_velocities"):
                    view.set_velocities(
                        zeros.reshape(pos_t.shape[0], -1, 3) if pos_t.ndim == 3 else zeros.reshape(1, -1, 3)
                    )
                elif hasattr(view, "set_world_velocities"):
                    view.set_world_velocities(
                        zeros.reshape(pos_t.shape[0], -1, 3) if pos_t.ndim == 3 else zeros.reshape(1, -1, 3)
                    )
            except Exception:
                pass
            restored += 1
        except Exception as exc:
            manaprint(f"WARNING: particle cloth restore failed for {mesh_path}: {exc}")
            continue
    return restored


def _reset_particle_cloth_from_usd_rest(stage: Any, root_path: str) -> int:
    """Reset live PhysX particles to USD Mesh rest points × current L2W.

    This is the authored "above the hang bar" / flat sheet state before gravity
    drapes the cloth. ``reset_scene_to_default`` does not restore particles; call
    this explicitly on hang reset so episodes do not inherit prior deformation.
    """
    try:
        from pxr import Usd, UsdGeom
    except Exception:
        return 0

    reset_meshes = 0
    for mesh_path in _find_particle_cloth_mesh_paths(stage, root_path):
        prim = stage.GetPrimAtPath(mesh_path)
        if not prim or not prim.IsValid():
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if points is None or len(points) < 8:
            continue
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rows = [[float(mat.Transform(p)[0]), float(mat.Transform(p)[1]), float(mat.Transform(p)[2])] for p in points]
        try:
            from isaacsim.core.prims import ClothPrim

            view = ClothPrim(
                prim_paths_expr=mesh_path,
                name=f"mana_cloth_rest_{mesh_path.replace('/', '_')}_{id(stage) & 0xFFFF}",
                reset_xform_properties=False,
            )
            try:
                view.initialize()
            except Exception as exc:
                manaprint(f"WARNING: ClothPrim.initialize failed for rest reset {mesh_path}: {exc}")
                continue
            if not bool(getattr(view, "is_physics_handle_valid", lambda: False)()):
                continue
            pos = torch.as_tensor(rows, dtype=torch.float32).reshape(1, -1, 3)
            # Match device of live buffer when possible.
            try:
                live = as_torch(view.get_world_positions())
                if isinstance(live, torch.Tensor) and live.numel() > 0:
                    if live.reshape(-1, 3).shape[0] != pos.reshape(-1, 3).shape[0]:
                        manaprint(
                            f"WARNING: rest reset vertex count mismatch for {mesh_path}: "
                            f"usd={pos.reshape(-1, 3).shape[0]} phys={live.reshape(-1, 3).shape[0]}"
                        )
                        continue
                    pos = pos.to(device=live.device, dtype=live.dtype)
            except Exception as exc:
                manaprint(f"WARNING: particle cloth live-buffer inspection failed for {mesh_path}: {exc}")
            view.set_world_positions(pos)
            try:
                zeros = torch.zeros_like(pos)
                if hasattr(view, "set_velocities"):
                    view.set_velocities(zeros)
                elif hasattr(view, "set_world_velocities"):
                    view.set_world_velocities(zeros)
            except Exception:
                pass
            reset_meshes += 1
        except Exception as exc:
            manaprint(f"WARNING: particle cloth rest reset failed for {mesh_path}: {exc}")
            continue
    return reset_meshes


def _relocate_particle_cloth_from_rest_by_root_delta(
    env: ManagerBasedEnv,
    stage: Any,
    cloth_path: str,
    *,
    default_xf: Any,
    target_xf: Any,
) -> tuple[int, int, bool]:
    """Shared cloth relocate: USD-rest particles at ``default_xf``, then δ to ``target_xf``.

    Contract for particle-cloth randomization (fold / hang / tshirt):

    1. Cloth **USD root stays at the scene default** (``default_xf``) so Fabric does
       not double-apply a root xform on top of world-frame particles.
    2. PhysX particles are rebuilt from USD rest mesh × that default L2W, then
       moved by ``target_xf * inv(default_xf)``.
    3. PhysX→Fabric is force-flushed so the viewport follows particles.

    Returns ``(n_rest_meshes, n_moved_meshes, fabric_ok)``.
    """
    n_rest = _reset_particle_cloth_from_usd_rest(stage, cloth_path)
    n_moved = _move_particle_cloth_by_delta(stage, cloth_path, default_xf, target_xf)
    fabric_ok = _publish_cloth_scene_state(env)
    return n_rest, n_moved, fabric_ok


def _particle_snap_centroid_xy(snap: dict[str, Any]) -> tuple[float, float] | None:
    """Mean XY of snapped particle world positions, or None if empty."""
    if not snap:
        return None

    chunks = []
    for pos in snap.values():
        if pos is None:
            continue
        arr = to_numpy_keep_shape(pos)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
        if arr.shape[0] > 0:
            chunks.append(arr)
    if not chunks:
        return None
    all_pts = np.concatenate(chunks, axis=0)
    return float(all_pts[:, 0].mean()), float(all_pts[:, 1].mean())


def _move_particle_cloth_by_delta(stage: Any, root_path: str, old_xf: Any, new_xf: Any) -> int:
    """Move PhysX particle cloth by the same world delta as a USD root xform change.

    Does **not** rebuild from rest mesh (that heuristic was placing the towel
    off-table). Only applies ``delta = new_xf * inv(old_xf)`` to live particle
    world positions.

    Returns:
        Number of cloth meshes whose particles were successfully moved. ``0``
        means PhysX particles were left behind while USD may already have moved
        — that is the classic "invisible contact vs rendered mesh" desync.
    """
    try:
        from pxr import Gf
    except Exception:
        return 0

    delta = new_xf * old_xf.GetInverse()
    # Near-identity → nothing to do.
    eye = Gf.Matrix4d(1.0)
    try:
        if Gf.IsClose(delta, eye, 1e-9):
            return 0
    except Exception:
        pass

    r_list, t_list = _gf_matrix_to_rt(delta)
    return _move_particle_cloth_by_rt(stage, root_path, r_list, t_list)


def clear_particle_cloth_view_cache() -> None:
    """Drop any legacy ClothPrim view cache (handles die across resets)."""
    if hasattr(_move_particle_cloth_by_delta, "_views"):
        try:
            setattr(_move_particle_cloth_by_delta, "_views", {})
        except Exception:
            pass


def _set_prim_collision_enabled(prim: Any, enabled: bool) -> None:
    try:
        from pxr import UsdPhysics
    except Exception:
        return
    try:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        api = UsdPhysics.CollisionAPI(prim)
        attr = api.GetCollisionEnabledAttr()
        if attr:
            attr.Set(bool(enabled))
    except Exception:
        pass
    try:
        attr = prim.GetAttribute("physics:collisionEnabled")
        if attr:
            attr.Set(bool(enabled))
    except Exception:
        pass


def _ensure_thin_hang_rod_collision(stage: Any, rack_path: str) -> str | None:
    """Replace square top-plate contact with a thin hang rod along rack local +Y.

    ``rack_d`` authors ``collision_03`` as a nearly-square horizontal plate
    (``aabb≈0.27×0.27×0.03``). Particle cloth then drapes on that plate and the
    crease follows the towel long-axis / plate PCA — which is **not** the thin
    visual hang tube. Disable the plate (and base pad) collision and install a
    kinematic capsule along local +Y (matches authored child +90° tube axis =
    ``eff+X`` in world).
    """
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except Exception:
        return None

    root = stage.GetPrimAtPath(rack_path)
    if not root or not root.IsValid():
        return None

    # Disable flat pad colliders; keep the vertical post (collision_02).
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name not in ("collision_03", "collision_04"):
            continue
        path = str(prim.GetPath())
        if "/collision/" not in path:
            continue
        _set_prim_collision_enabled(prim, False)
        manaprint(f"INFO: hang rod: disabled plate collision on {path}")

    capsule_path = f"{rack_path}/hang_rod_capsule"
    existing = stage.GetPrimAtPath(capsule_path)
    if existing and existing.IsValid():
        _set_prim_collision_enabled(existing, True)
        return capsule_path

    # Local +Y capsule. World hang height ≈0.074 with rack root z≈-0.226 → local z≈0.30.
    local_z = 0.30
    radius = 0.012
    height = 0.28

    capsule = UsdGeom.Capsule.Define(stage, capsule_path)
    capsule.CreateRadiusAttr(radius)
    capsule.CreateHeightAttr(height)
    capsule.CreateAxisAttr("Y")
    xform = UsdGeom.Xformable(capsule.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, local_z))
    xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    prim = capsule.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    _set_prim_collision_enabled(prim, True)
    manaprint(f"INFO: hang rod: created thin capsule {capsule_path} " f"axis=Y z={local_z:.3f} r={radius} h={height}")
    return capsule_path


def _cloth_mesh_local_aabb(stage: Any, root_path: str) -> tuple[Any, Any, int, int, int] | None:
    """Return ``(lo, hi, long_i, short_i, thick_i)`` in asset-local frame, or None."""
    try:
        from pxr import UsdGeom
    except Exception:
        return None

    mesh_paths = _find_particle_cloth_mesh_paths(stage, root_path)
    if not mesh_paths:
        try:
            from pxr import Usd

            root = stage.GetPrimAtPath(root_path)
            if root and root.IsValid():
                for prim in Usd.PrimRange(root):
                    if prim.IsA(UsdGeom.Mesh):
                        mesh_paths.append(str(prim.GetPath()))
                        break
        except Exception:
            return None
    if not mesh_paths:
        return None

    mesh_prim = stage.GetPrimAtPath(mesh_paths[0])
    if not mesh_prim or not mesh_prim.IsValid():
        return None
    points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
    if points is None or len(points) < 4:
        return None

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    lo = (min(xs), min(ys), min(zs))
    hi = (max(xs), max(ys), max(zs))
    extent = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
    order = sorted(range(3), key=lambda i: extent[i], reverse=True)
    return lo, hi, order[0], order[1], order[2]


def _asset_short_edge_midpoints_world(
    stage: Any, root_path: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Both short-edge midpoints in world, from asset-local AABB + current L2W."""
    try:
        from pxr import Gf, Usd, UsdGeom
    except Exception:
        return None

    aabb = _cloth_mesh_local_aabb(stage, root_path)
    if aabb is None:
        return None
    lo, hi, long_i, short_i, thick_i = aabb

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    mesh_paths = _find_particle_cloth_mesh_paths(stage, root_path)
    xf_prim = stage.GetPrimAtPath(mesh_paths[0]) if mesh_paths else root
    if not xf_prim or not xf_prim.IsValid():
        xf_prim = root
    l2w = UsdGeom.Xformable(xf_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    mid_short = 0.5 * (lo[short_i] + hi[short_i])
    mid_thick = 0.5 * (lo[thick_i] + hi[thick_i])

    def _mid(long_coord: float) -> tuple[float, float, float]:
        p = [0.0, 0.0, 0.0]
        p[long_i] = long_coord
        p[short_i] = mid_short
        p[thick_i] = mid_thick
        wp = l2w.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        return (float(wp[0]), float(wp[1]), float(wp[2]))

    return _mid(lo[long_i]), _mid(hi[long_i])


def _se2_rt_about_pivot(
    *,
    dx: float,
    dy: float,
    dz: float,
    yaw_rad: float,
    pivot_xyz: tuple[float, float, float],
) -> tuple[list[list[float]], list[float]]:
    """Rigid SE2: ``p' = Rz(yaw)(p - pivot) + pivot + (dx,dy,dz)`` as ``R,t`` for ``Rp+t``."""

    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    px, py, pz = (float(pivot_xyz[0]), float(pivot_xyz[1]), float(pivot_xyz[2]))
    # t = pivot + d - R @ pivot
    rpx = c * px - s * py
    rpy = s * px + c * py
    t = [px + float(dx) - rpx, py + float(dy) - rpy, pz + float(dz) - pz]
    return r, t


def _apply_se2_to_point(
    point: tuple[float, float, float],
    *,
    dx: float,
    dy: float,
    dz: float,
    yaw_rad: float,
    pivot_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:

    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    px, py, pz = pivot_xyz
    x, y, z = point
    rx = x - px
    ry = y - py
    return (
        c * rx - s * ry + px + float(dx),
        s * rx + c * ry + py + float(dy),
        z + float(dz),
    )


def _tabletop_particle_cloth_se2(
    env: ManagerBasedEnv,
    stage: Any,
    cloth_path: str,
    *,
    dx: float,
    dy: float,
    dz: float,
    yaw_rad: float,
    pivot_xyz: tuple[float, float, float],
) -> tuple[int, int, bool]:
    """Keep cloth USD at default; rebuild particles from rest; apply SE2 about pivot.

    Does **not** author cloth USD (avoids viewport flash + Fabric double-apply).
    Pivot should be the default cloth-root world position (USD already at default).
    """
    n_rest = _reset_particle_cloth_from_usd_rest(stage, cloth_path)
    r_list, t_list = _se2_rt_about_pivot(dx=dx, dy=dy, dz=dz, yaw_rad=yaw_rad, pivot_xyz=pivot_xyz)
    n_moved = _move_particle_cloth_by_rt(stage, cloth_path, r_list, t_list)
    fabric_ok = _publish_cloth_scene_state(env)
    return n_rest, n_moved, fabric_ok


def _point_in_reach_box(point: tuple[float, float, float], reach: dict[str, list[float]]) -> bool:
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        lo_hi = reach.get(axis)
        if lo_hi is None or len(lo_hi) < 2:
            continue
        v = point[idx]
        if v < float(lo_hi[0]) or v > float(lo_hi[1]):
            return False
    return True


def randomize_cloth_pose_usd(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    prim_path_expr: str,
    position_range: dict[str, list[float]] | None = None,
    yaw_range: list[float] | None = None,
    *,
    require_short_edges_in_reach: bool = False,
    reach_position_range: dict[str, list[float]] | None = None,
    max_sample_tries: int = 48,
) -> None:
    """Randomize tabletop particle cloth without touching cloth USD.

    Particle cloth has no rigid-body root state, so ``reset_scene_to_default`` does
    **not** restore it. Contract (aligned with hang):

    1. Sample SE2 offsets in math only (no cloth USD writes → no viewport flash).
    2. Cloth USD stays at the **scene default** (Fabric will not double-apply).
    3. Rebuild PhysX particles from USD rest × default L2W, apply SE2 about the
       default root world pivot, Fabric flush.

    Prefer placing this **after** ``mdp.reset_scene_to_default``.

    Args:
        prim_path_expr: Path to the cloth root prim.  Use
            ``{ENV_REGEX_NS}`` as a placeholder for the per-env namespace,
            e.g. ``"{ENV_REGEX_NS}/tshirt_a"``.
        position_range: Per-axis offset from the prim's **default** translation,
            e.g. ``{"x": [-0.05, 0.05], "y": [-0.08, 0.08]}``.
        yaw_range: ``[min_rad, max_rad]`` for Z-axis rotation.  ``None``
            skips yaw randomisation.
    """

    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    # Persist scene-default xforms across resets (do not accumulate offsets).
    defaults: dict = getattr(env, "_cloth_pose_usd_defaults", None)
    if defaults is None:
        defaults = {}
        setattr(env, "_cloth_pose_usd_defaults", defaults)

    se2_store: dict = getattr(env, "_cloth_particle_se2", None)
    if se2_store is None:
        se2_store = {}
        setattr(env, "_cloth_particle_se2", se2_store)

    reach = reach_position_range or {}
    check_reach = bool(require_short_edges_in_reach) and bool(reach)

    # ClothPrim views die across resets; never reuse a stale handle cache.
    clear_particle_cloth_view_cache()

    for env_idx in env_ids.tolist():
        prim_path = prim_path_expr.replace("{ENV_REGEX_NS}", f"/World/envs/env_{int(env_idx)}")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        base = _ensure_cloth_usd_default(stage, prim_path, defaults)
        if base is None:
            continue

        # Force USD to scene default without moving particles (clears any prior sample).
        bp = base["pos"]
        _set_cloth_usd_absolute_pose(
            stage,
            prim_path,
            x=float(bp[0]),
            y=float(bp[1]),
            z=float(bp[2]),
            yaw_rad=0.0,
            base=base,
            move_particles=False,
        )
        default_xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pivot = (
            float(default_xf.ExtractTranslation()[0]),
            float(default_xf.ExtractTranslation()[1]),
            float(default_xf.ExtractTranslation()[2]),
        )

        # Default short-edge mids (USD at default) for optional reach checks.
        base_mids = _asset_short_edge_midpoints_world(stage, prim_path) if check_reach else None

        tries = max(int(max_sample_tries), 1) if check_reach else 1
        accepted = False
        last_xyzyaw = (0.0, 0.0, 0.0, 0.0)
        for attempt in range(tries):
            x_range = (position_range or {}).get("x", [0.0, 0.0])
            y_range = (position_range or {}).get("y", [0.0, 0.0])
            z_range = (position_range or {}).get("z", [0.0, 0.0])
            dx = random.uniform(x_range[0], x_range[1]) if position_range else 0.0
            dy = random.uniform(y_range[0], y_range[1]) if position_range else 0.0
            dz = random.uniform(z_range[0], z_range[1]) if position_range else 0.0
            yaw_rad = random.uniform(yaw_range[0], yaw_range[1]) if yaw_range is not None else 0.0
            last_xyzyaw = (float(dx), float(dy), float(dz), float(yaw_rad))

            if not check_reach:
                accepted = True
                break

            if base_mids is None:
                accepted = True
                break
            a0, b0 = base_mids
            a = _apply_se2_to_point(a0, dx=dx, dy=dy, dz=dz, yaw_rad=yaw_rad, pivot_xyz=pivot)
            b = _apply_se2_to_point(b0, dx=dx, dy=dy, dz=dz, yaw_rad=yaw_rad, pivot_xyz=pivot)
            if _point_in_reach_box(a, reach) and _point_in_reach_box(b, reach):
                accepted = True
                manaprint(
                    f"[randomize_cloth_pose_usd] {prim_path}: short-edge reach OK "
                    f"on try {attempt + 1}/{tries} "
                    f"yaw={last_xyzyaw[3]:.3f} rad "
                    f"xy=({last_xyzyaw[0]:.3f},{last_xyzyaw[1]:.3f})"
                )
                break

        if check_reach and not accepted:
            manaprint(
                f"[WARN] randomize_cloth_pose_usd {prim_path}: no sample with both "
                f"short-edge midpoints in reach after {tries} tries; keeping last "
                f"offset=({last_xyzyaw[0]:.3f},{last_xyzyaw[1]:.3f}) yaw={last_xyzyaw[3]:.3f}"
            )

        n_rest, n_moved, fabric_ok = _tabletop_particle_cloth_se2(
            env,
            stage,
            prim_path,
            dx=last_xyzyaw[0],
            dy=last_xyzyaw[1],
            dz=last_xyzyaw[2],
            yaw_rad=last_xyzyaw[3],
            pivot_xyz=pivot,
        )
        se2_store[prim_path] = {
            "dx": last_xyzyaw[0],
            "dy": last_xyzyaw[1],
            "dz": last_xyzyaw[2],
            "yaw": last_xyzyaw[3],
            "pivot": pivot,
        }
        try:
            import numpy as np

            pts = _read_particle_cloth_world_points(stage, prim_path)
            if pts is not None:
                arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
                manaprint(
                    f"INFO: randomize_cloth_pose_usd {prim_path}: "
                    f"dx={last_xyzyaw[0]:.3f} dy={last_xyzyaw[1]:.3f} "
                    f"yaw_deg={math.degrees(last_xyzyaw[3]):.1f} "
                    f"rest={n_rest} moved={n_moved} fabric={fabric_ok} usd=untouched "
                    f"phys_centroid=({arr[:, 0].mean():.3f},{arr[:, 1].mean():.3f},"
                    f"{arr[:, 2].mean():.3f}) pivot=({pivot[0]:.3f},{pivot[1]:.3f},{pivot[2]:.3f})"
                )
            elif n_rest <= 0 or n_moved <= 0:
                manaprint(
                    f"WARNING: randomize_cloth_pose_usd {prim_path}: "
                    f"rest={n_rest} moved={n_moved} fabric={fabric_ok}; particles may be stale"
                )
        except Exception:
            if n_rest <= 0 or n_moved <= 0:
                manaprint(
                    f"WARNING: randomize_cloth_pose_usd {prim_path}: "
                    f"rest={n_rest} moved={n_moved} fabric={fabric_ok}"
                )


def _cloth_usd_defaults_cache(env: ManagerBasedEnv) -> dict:
    defaults: dict = getattr(env, "_cloth_pose_usd_defaults", None)
    if defaults is None:
        defaults = {}
        setattr(env, "_cloth_pose_usd_defaults", defaults)
    return defaults


def _ensure_cloth_usd_default(stage: Any, prim_path: str, defaults: dict) -> dict | None:
    """Cache and return the cloth root's scene-default translate / rotation."""
    try:
        from pxr import Gf, UsdGeom
    except Exception:
        return None

    if prim_path in defaults:
        return defaults[prim_path]

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    xformable = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps() or ()}
    translate_op0 = ops.get("xformOp:translate") or ops.get("xformOp:translation")
    rotate_xyz_op0 = ops.get("xformOp:rotateXYZ")
    orient_op0 = ops.get("xformOp:orient")
    base_pos = translate_op0.Get() if translate_op0 is not None else None
    base_rot = rotate_xyz_op0.Get() if rotate_xyz_op0 is not None else None
    base_orient = orient_op0.Get() if orient_op0 is not None else None
    defaults[prim_path] = {
        "pos": (
            Gf.Vec3d(float(base_pos[0]), float(base_pos[1]), float(base_pos[2]))
            if base_pos is not None
            else Gf.Vec3d(0, 0, 0)
        ),
        "rot": (
            Gf.Vec3f(float(base_rot[0]), float(base_rot[1]), float(base_rot[2]))
            if base_rot is not None
            else Gf.Vec3f(0, 0, 0)
        ),
        "orient": base_orient,
        "has_rotate_xyz": rotate_xyz_op0 is not None,
        "has_orient": orient_op0 is not None,
    }
    return defaults[prim_path]


def _set_cloth_usd_absolute_pose(
    stage: Any,
    prim_path: str,
    *,
    x: float,
    y: float,
    z: float,
    yaw_rad: float,
    base: dict,
    move_particles: bool = True,
) -> bool:
    """Set cloth root USD translate/yaw to an absolute local pose.

    Yaw is applied as an offset on top of the cached scene-default rotation
    (``rotateXYZ`` Z or composed onto default ``orient``).

    When ``move_particles`` is True, also applies the USD root delta to live
    PhysX particles. Hang-bar reset should set this False and move particles
    with the hang-bar world delta instead (``reset_scene_to_default`` does not
    restore particle cloth, so cloth-USD delta can desync after the first R).
    """

    try:
        from pxr import Gf, Usd, UsdGeom
    except Exception:
        return False

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False

    xformable = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps() or ()}
    old_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    translate_op = ops.get("xformOp:translate") or ops.get("xformOp:translation")
    if translate_op is not None:
        translate_op.Set(Gf.Vec3d(float(x), float(y), float(z)))

    yaw_deg = math.degrees(float(yaw_rad))
    rotate_xyz_op = ops.get("xformOp:rotateXYZ")
    orient_op = ops.get("xformOp:orient")
    if rotate_xyz_op is not None:
        br = base["rot"]
        rotate_xyz_op.Set(Gf.Vec3f(float(br[0]), float(br[1]), float(br[2]) + yaw_deg))
    elif orient_op is not None:
        half = float(yaw_rad) * 0.5
        yaw_q = Gf.Quatd(math.cos(half), 0.0, 0.0, math.sin(half))
        base_orient = base.get("orient")
        if base_orient is not None:
            try:
                # Compose default * delta so authored base orientation is preserved.
                orient_op.Set(Gf.Quatd(base_orient) * yaw_q)
            except Exception:
                orient_op.Set(yaw_q)
        else:
            orient_op.Set(yaw_q)

    if not move_particles:
        return True

    new_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    n_moved = _move_particle_cloth_by_delta(stage, prim_path, old_xf, new_xf)
    if n_moved <= 0:
        # USD root moved but PhysX particles did not — contact follows particles.
        manaprint(
            f"WARNING: cloth USD pose updated but 0 particle meshes moved ({prim_path}); "
            "expect PhysX contact vs rendered cloth desync"
        )
    return True


def _cloth_basket_required_separation(
    dx: float,
    dy: float,
    yaw_rad: float,
    *,
    cloth_half_length: float,
    cloth_half_width: float,
    basket_radius: float,
    margin: float,
) -> float:
    """Minimum center distance so cloth OBB and basket circle do not overlap in XY."""

    dist = math.hypot(dx, dy)
    if dist < 1e-8:
        return float("inf")
    ux, uy = dx / dist, dy / dist
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    cloth_support = abs(ux * c + uy * s) * float(cloth_half_length) + abs(-ux * s + uy * c) * float(cloth_half_width)
    return cloth_support + float(basket_radius) + float(margin)


def randomize_cloth_basket_layout_usd(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cloth_prim_path_expr: str,
    basket_cfg: SceneEntityCfg,
    cloth_pose_range: dict[str, list[float]] | None = None,
    cloth_yaw_range: list[float] | None = None,
    min_separation: float = 0.30,
    max_sample_tries: int = 48,
    cloth_half_length: float = 0.18,
    cloth_half_width: float = 0.14,
    basket_radius: float = 0.12,
    separation_margin: float = 0.03,
) -> None:
    """Randomize cloth on the right half of the table; keep basket fixed.

    Basket stays at the scene default restored by ``mdp.reset_scene_to_default``.
    Cloth XY is sampled only inside ``cloth_pose_range`` (right-half band) with
    yaw from ``cloth_yaw_range``, rejecting samples that are too close to the
    fixed basket.

    Uses tabletop cloth contract: cloth USD stays at scene default; PhysX
    particles rest + SE2 about the default root pivot (no cloth USD writes).

    Must run **after** ``mdp.reset_scene_to_default``.
    """

    if env_ids is None or len(env_ids) == 0:
        return

    cloth_pose_range = cloth_pose_range or {}
    x_cloth = cloth_pose_range.get("x", [0.12, 0.22])
    y_cloth = cloth_pose_range.get("y", [-0.50, -0.34])
    cloth_yaw_range = cloth_yaw_range if cloth_yaw_range is not None else [-math.pi, math.pi]
    min_sep_floor = float(min_separation)
    max_tries = max(int(max_sample_tries), 1)

    try:
        import omni.usd
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    basket = env.scene[get_scene_entity_name(basket_cfg)]
    defaults = _cloth_usd_defaults_cache(env)
    env_origins = env.scene.env_origins
    clear_particle_cloth_view_cache()

    for env_idx in env_ids.tolist():
        env_i = int(env_idx)
        cloth_path = cloth_prim_path_expr.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_i}")
        base = _ensure_cloth_usd_default(stage, cloth_path, defaults)
        if base is None:
            continue

        basket_xy = basket.data.root_pos_w.torch[env_i, :2] - env_origins[env_i, :2]
        basket_x = float(basket_xy[0].item())
        basket_y = float(basket_xy[1].item())
        cloth_z = float(base["pos"][2])

        chosen_x = random.uniform(float(x_cloth[0]), float(x_cloth[1]))
        chosen_y = random.uniform(float(y_cloth[0]), float(y_cloth[1]))
        chosen_yaw = random.uniform(float(cloth_yaw_range[0]), float(cloth_yaw_range[1]))
        best_x, best_y, best_yaw = chosen_x, chosen_y, chosen_yaw
        best_score = float("-inf")
        found = False

        for _ in range(max_tries):
            cx = random.uniform(float(x_cloth[0]), float(x_cloth[1]))
            cy = random.uniform(float(y_cloth[0]), float(y_cloth[1]))
            yaw = random.uniform(float(cloth_yaw_range[0]), float(cloth_yaw_range[1]))
            dx = basket_x - cx
            dy = basket_y - cy
            dist = math.hypot(dx, dy)
            need = _cloth_basket_required_separation(
                dx,
                dy,
                yaw,
                cloth_half_length=cloth_half_length,
                cloth_half_width=cloth_half_width,
                basket_radius=basket_radius,
                margin=separation_margin,
            )
            need = max(need, min_sep_floor)
            score = dist - need
            if score > best_score:
                best_score = score
                best_x, best_y, best_yaw = cx, cy, yaw
            if dist >= need:
                chosen_x, chosen_y, chosen_yaw = cx, cy, yaw
                found = True
                break

        if not found:
            chosen_x, chosen_y, chosen_yaw = best_x, best_y, best_yaw

        # Keep USD at default; particle SE2 about default root world pivot.
        from pxr import Usd, UsdGeom

        bp = base["pos"]
        _set_cloth_usd_absolute_pose(
            stage,
            cloth_path,
            x=float(bp[0]),
            y=float(bp[1]),
            z=float(bp[2]),
            yaw_rad=0.0,
            base=base,
            move_particles=False,
        )
        xf_prim = stage.GetPrimAtPath(cloth_path)
        default_xf = UsdGeom.Xformable(xf_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pivot = (
            float(default_xf.ExtractTranslation()[0]),
            float(default_xf.ExtractTranslation()[1]),
            float(default_xf.ExtractTranslation()[2]),
        )
        dx = float(chosen_x) - float(bp[0])
        dy = float(chosen_y) - float(bp[1])
        dz = float(cloth_z) - float(bp[2])
        n_rest, n_moved, fabric_ok = _tabletop_particle_cloth_se2(
            env,
            stage,
            cloth_path,
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_rad=float(chosen_yaw),
            pivot_xyz=pivot,
        )
        if n_rest <= 0 or n_moved <= 0:
            manaprint(
                f"WARNING: randomize_cloth_basket_layout_usd {cloth_path}: "
                f"rest={n_rest} moved={n_moved} fabric={fabric_ok}"
            )


def randomize_hang_bar_and_cloth_usd(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cloth_prim_path_expr: str,
    hang_bar_cfg: SceneEntityCfg,
    position_range: dict[str, list[float]] | None = None,
    yaw_range: list[float] | None = None,
) -> None:
    """Reset cloth to authored rest (above the bar), then place it over the new hang.

    Scope (intentionally narrow):

    1. Randomize the hang-bar root pose (yaw / XY vs scene default).
    2. Clear hang ``/Aligned`` child xforms so collision matches the visible rod.
    3. Same particle-cloth contract as fold/tshirt: rebuild from USD rest, relocate
       particles only (here about the hang pivot), leave cloth USD at scene default,
       flush Fabric; gravity re-settles onto the bar.

    Must run **after** ``mdp.reset_scene_to_default``.
    """
    from isaaclab.utils import math as math_utils

    if env_ids is None or len(env_ids) == 0:
        return

    position_range = position_range or {}
    x_range = position_range.get("x", [0.0, 0.0])
    y_range = position_range.get("y", [0.0, 0.0])
    if yaw_range is None:
        yaw_range = [-math.pi, math.pi]

    try:
        import omni.usd
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    hang_bar = env.scene[get_scene_entity_name(hang_bar_cfg)]
    # ClothPrim views die across resets; never reuse a stale handle cache.
    clear_particle_cloth_view_cache()
    # Fixed-pose scenes never go through schema-vNext runtime_sample / pose
    # bridge, so clear package ``/Aligned`` here each reset.
    _asset_prim_path_for_env = None
    try:
        from manaenv.pipeline.generators.scene_layout_solver.manaenv_bridge.runtime_events import (
            _asset_prim_path_for_env as _asset_prim_path_for_env_fn,
        )
        from manaenv.pipeline.generators.scene_layout_solver.manaenv_bridge.runtime_events import _ignore_aligned_xforms

        _asset_prim_path_for_env = _asset_prim_path_for_env_fn
        _ignore_aligned_xforms(hang_bar, env_ids)
    except Exception as exc:
        manaprint(f"WARNING: randomize_hang_bar_and_cloth_usd: aligned setup failed: {exc}")

    env_origins = env.scene.env_origins

    bar_defaults = _default_root_pose_local_from_asset(hang_bar, env_ids)
    if bar_defaults is None:
        manaprint("WARNING: randomize_hang_bar_and_cloth_usd: hang bar has no default_root_state")
        return

    # Particles are NOT restored by reset_scene_to_default. Always rebuild from
    # USD rest (above-bar), then relocate onto the new hang pivot.
    hang_pivot_cache: dict[int, tuple[tuple[float, float, float], float]] = (
        getattr(env, "_hang_particle_pivot_cache", None) or {}
    )
    if not isinstance(hang_pivot_cache, dict):
        hang_pivot_cache = {}

    pose_tensor = bar_defaults.clone()
    num = len(env_ids)
    for i, env_idx in enumerate(env_ids.tolist()):
        env_i = int(env_idx)
        dx = random.uniform(float(x_range[0]), float(x_range[1]))
        dy = random.uniform(float(y_range[0]), float(y_range[1]))
        yaw = random.uniform(float(yaw_range[0]), float(yaw_range[1]))

        pose_tensor[i, 0] = float(bar_defaults[i, 0].item()) + dx
        pose_tensor[i, 1] = float(bar_defaults[i, 1].item()) + dy
        roll, pitch, base_yaw = math_utils.euler_xyz_from_quat(bar_defaults[i : i + 1, 3:7])
        pose_tensor[i, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, base_yaw + yaw)[0]

    _sync_pose_to_usd(hang_bar, env_ids, pose_tensor)
    sim_pose = pose_tensor.clone()
    sim_pose[:, :3] += env_origins[env_ids]
    _write_root_pose_with_zero_velocity(hang_bar, sim_pose, env_ids)

    rack_paths: dict[int, str] = {}
    if _asset_prim_path_for_env is not None:
        try:
            for env_idx in env_ids.tolist():
                rp = _asset_prim_path_for_env(hang_bar, env_idx=int(env_idx), local_index=0)
                if rp:
                    rack_paths[int(env_idx)] = rp
        except Exception:
            pass

    for i, env_idx in enumerate(env_ids.tolist()):
        env_i = int(env_idx)
        cloth_path = cloth_prim_path_expr.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_i}")
        prim = stage.GetPrimAtPath(cloth_path)
        if not prim or not prim.IsValid():
            continue

        rack_path = rack_paths.get(env_i)

        # Hang pivot = bar root XY + collision top Z (fallback root_z+0.30).
        new_yaw = _z_yaw_from_quat_xyzw(sim_pose[i, 3:7].detach().cpu())
        new_xy = (
            float(sim_pose[i, 0].item()),
            float(sim_pose[i, 1].item()),
        )
        hang_z = float(sim_pose[i, 2].item()) + 0.30
        if rack_path:
            try:
                import numpy as np

                hpts = _hang_collision_top_points(stage, rack_path)
                if hpts is not None:
                    arr = np.asarray(hpts, dtype=np.float64).reshape(-1, 3)
                    if arr.shape[0] >= 4:
                        hang_z = float(np.quantile(arr[:, 2], 0.85))
            except Exception:
                pass
        new_pivot = (new_xy[0], new_xy[1], hang_z)

        # 1) Back to authored rest (flat / above default hang).
        n_rest = _reset_particle_cloth_from_usd_rest(stage, cloth_path)
        # 2) Relocate that rest sheet from default hang → new hang.
        default_pos = bar_defaults[i, :3].detach().cpu() + env_origins[env_ids[i]].detach().cpu()
        old_yaw = _z_yaw_from_quat_xyzw(bar_defaults[i, 3:7].detach().cpu())
        old_pivot = (float(default_pos[0]), float(default_pos[1]), hang_z)
        dyaw = _wrap_angle_rad(new_yaw - old_yaw)
        n_moved = _move_particle_cloth_about_pivot(
            stage,
            cloth_path,
            old_pivot_xyz=old_pivot,
            new_pivot_xyz=new_pivot,
            dyaw_rad=dyaw,
        )
        # Visual follows particles only — do not touch cloth USD.
        _publish_cloth_scene_state(env)
        hang_pivot_cache[env_i] = (new_pivot, new_yaw)
        if n_rest <= 0:
            manaprint(
                f"WARNING: hang reset env={env_i}: USD rest particle reset failed; "
                "cloth may keep previous-episode deformation"
            )
        if n_moved <= 0:
            manaprint(
                f"WARNING: hang reset env={env_i}: 0 particle meshes moved; "
                "PhysX cloth may desync from rendered mesh (grasp state vs rendered state)"
            )

    setattr(env, "_hang_particle_pivot_cache", hang_pivot_cache)

    manaprint(
        f"INFO: randomize_hang_bar_and_cloth_usd: relocated cloth onto bar for {num} env(s) "
        f"yaw_range=[{float(yaw_range[0]):.2f}, {float(yaw_range[1]):.2f}] "
        f"pos_range={{x:[{float(x_range[0]):.2f}, {float(x_range[1]):.2f}], "
        f"y:[{float(y_range[0]):.2f}, {float(y_range[1]):.2f}]}}"
    )


def _mesh_world_points_under(stage: Any, root_path: str, *, subtree: str) -> Any:
    """Collect USD mesh points under ``root/subtree`` transformed to world."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    rows: list[tuple[float, float, float]] = []
    needle = f"/{subtree}/"
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if needle not in path:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if points is None or len(points) < 3:
            continue
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for p in points:
            wp = mat.Transform(p)
            rows.append((float(wp[0]), float(wp[1]), float(wp[2])))
    if len(rows) < 8:
        return None
    return np.asarray(rows, dtype=np.float64)


def _iter_mesh_world_points_by_prim(stage: Any, root_path: str, *, subtree: str):
    """Yield ``(prim_path, Nx3 world points)`` for each Mesh under ``root/subtree``."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    needle = f"/{subtree}/"
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if needle not in path:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if points is None or len(points) < 3:
            continue
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rows = []
        for p in points:
            wp = mat.Transform(p)
            rows.append((float(wp[0]), float(wp[1]), float(wp[2])))
        if len(rows) < 3:
            continue
        yield path, np.asarray(rows, dtype=np.float64)


def _read_particle_cloth_world_points(stage: Any, cloth_path: str) -> Any | None:
    """Return Nx3 numpy world positions for the first valid particle cloth mesh."""

    for mesh_path in _find_particle_cloth_mesh_paths(stage, cloth_path):
        pts = None
        try:
            from isaacsim.core.prims import ClothPrim

            view = ClothPrim(
                prim_paths_expr=mesh_path,
                name=f"mana_cloth_read_{mesh_path.replace('/', '_')}_{id(stage) & 0xFFFF}",
                reset_xform_properties=False,
            )
            try:
                view.initialize()
            except Exception as exc:
                manaprint(f"WARNING: ClothPrim.initialize failed for point read {mesh_path}: {exc}")
                continue
            if not bool(getattr(view, "is_physics_handle_valid", lambda: False)()):
                continue
            pts = view.get_world_positions()
            if pts is None:
                continue
            converted = to_numpy_keep_shape(pts)
            if converted is None:
                continue
            arr = np.asarray(converted, dtype=np.float64).reshape(-1, 3)
            if arr.shape[0] >= 8:
                return arr
        except Exception as exc:
            manaprint(
                f"ERROR: particle cloth point read failed for {mesh_path}: "
                f"type={type(pts).__name__} device={getattr(pts, 'device', 'n/a')}: {exc}"
            )
            continue
    return None


def _hang_collision_top_points(stage: Any, rack_path: str) -> Any | None:
    """Prefer collision_03 top surface points; else all collision meshes."""
    try:
        for path, pts in _iter_mesh_world_points_by_prim(stage, rack_path, subtree="collision"):
            if path.rsplit("/", 1)[-1] == "collision_03":
                return pts
        return _mesh_world_points_under(stage, rack_path, subtree="collision")
    except Exception:
        return None


_CLOTH_DIFFUSE_TEXTURE_INPUTS = (
    "diffuse_texture",
    "diffuseColor",
    "file",
    "inputs:file",
    "inputs:diffuse_texture",
    "inputs:diffuseColor",
)


def _set_shader_texture_asset(shader: Any, texture_uri: str) -> bool:
    """Set common diffuse / UsdUVTexture file inputs on a UsdShade.Shader."""
    try:
        from pxr import Sdf
    except Exception:
        return False

    asset = Sdf.AssetPath(str(texture_uri))
    wrote = False
    for name in _CLOTH_DIFFUSE_TEXTURE_INPUTS:
        try:
            inp = shader.GetInput(name)
            if not inp:
                # Only create known OmniPBR / UsdUVTexture slots when already present
                # elsewhere on this shader family — skip creating unrelated inputs.
                continue
            try:
                attr = inp.GetAttr()
                if attr is not None:
                    attr.ClearConnections()
            except Exception:
                pass
            shader.CreateInput(name, Sdf.ValueTypeNames.Asset).Set(asset)
            wrote = True
        except Exception:
            continue
    # UsdUVTexture often only exposes ``file``; create it when shader id matches.
    try:
        shader_id = shader.GetIdAttr().Get() if shader.GetIdAttr() else None
    except Exception:
        shader_id = None
    if not wrote and shader_id and "UVTexture" in str(shader_id):
        try:
            from pxr import Sdf

            shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(asset)
            wrote = True
        except Exception:
            pass
    if not wrote:
        # Last resort for OmniPBR-like shaders: create diffuse_texture.
        try:
            from pxr import Sdf

            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(asset)
            wrote = True
        except Exception:
            pass
    return wrote


def _apply_diffuse_texture_under_prim(stage: Any, root_path: str, texture_uri: str) -> int:
    """Walk shaders under ``root_path`` and set diffuse/file texture assets."""
    try:
        from pxr import Usd, UsdShade
    except Exception:
        return 0

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0

    n = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdShade.Shader):
            continue
        if _set_shader_texture_asset(UsdShade.Shader(prim), texture_uri):
            n += 1
    return n


def _is_direct_cloth_texture_ref(texture_ref: str) -> bool:
    ref = str(texture_ref or "").strip()
    return "://" in ref or ref.startswith("/") or ref.startswith("~")


def _normalize_cloth_texture_filename(texture_name: str) -> str:
    suffix = str(texture_name).rsplit("/", 1)[-1].strip()
    if not suffix:
        return ""
    if "." not in suffix:
        suffix = f"{suffix}.jpg"
    return suffix


def _sdf_asset_path_to_str(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "path"):
        try:
            path = value.path
            if path:
                return str(path)
        except Exception:
            pass
    text = str(value).strip()
    return text


def _existing_cloth_texture_dir(stage: Any, root_path: str) -> str | None:
    """Return the directory of an existing diffuse texture under ``root_path``."""
    try:
        from pxr import Usd, UsdShade
    except Exception:
        return None

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        for name in _CLOTH_DIFFUSE_TEXTURE_INPUTS:
            try:
                inp = shader.GetInput(name)
                if not inp:
                    continue
                uri = _sdf_asset_path_to_str(inp.Get())
                if uri and ("/" in uri or "\\" in uri):
                    return uri.rsplit("/", 1)[0]
            except Exception:
                continue
    return None


def _sibling_cloth_texture_uri(usd_path: str, texture_name: str) -> str:
    base = str(usd_path).rsplit("/", 1)[0]
    suffix = _normalize_cloth_texture_filename(texture_name)
    return f"{base}/textures/{suffix}"


def _resolve_cloth_texture_uri(
    texture_ref: str,
    *,
    stage: Any,
    cloth_prim_path: str,
    object_asset_id: str | None = None,
) -> str | None:
    """Resolve a cloth texture ref to an absolute URI.

    Accepts full local paths, or short filenames / stems that are
    resolved relative to the cloth asset's ``textures/`` directory.

    Cloth USDs often author ``./textures/*.jpg``. After composition under
    ``/World/envs/...`` those relative paths no longer resolve for Hydra, so
    short names must be rewritten to absolute local paths.
    """
    ref = str(texture_ref or "").strip()
    if not ref:
        return None
    if _is_direct_cloth_texture_ref(ref):
        return ref

    filename = _normalize_cloth_texture_filename(ref)
    if not filename:
        return None

    # Prefer asset-package absolute path first. Authored shader refs are often
    # ``./textures/...`` and must not be reused as the join base.
    asset_id = str(object_asset_id or "").strip() or cloth_prim_path.rstrip("/").rsplit("/", 1)[-1]
    if asset_id:
        try:
            from manaenv.assets.local_discovery import resolve_usd_path

            return _sibling_cloth_texture_uri(resolve_usd_path(asset_id), filename)
        except Exception:
            pass

    existing_dir = _existing_cloth_texture_dir(stage, cloth_prim_path)
    if existing_dir and _is_direct_cloth_texture_ref(existing_dir):
        return f"{existing_dir.rstrip('/')}/{filename}"
    return None


def randomize_cloth_diffuse_texture(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    prim_path_expr: str,
    texture_uris: list[str] | tuple[str, ...] | None = None,
    object_asset_id: str | None = None,
) -> None:
    """Randomize the cloth albedo / diffuse texture from a URI / name list.

    Intended for particle-cloth assets whose OmniPBR (or UsdUVTexture) shaders
    live under the cloth root prim. Samples one entry per env and writes it to
    common diffuse texture inputs (``diffuse_texture``, UsdUVTexture ``file``, …).

    Args:
        prim_path_expr: Cloth root path with ``{ENV_REGEX_NS}``, e.g.
            ``"{ENV_REGEX_NS}/towel_b"``.
        texture_uris: Absolute local texture paths, **or** short
            filenames / stems (e.g. ``towel_a_bc_b.jpg`` / ``towel_a_bc_b``)
            resolved against the cloth asset's ``textures/`` directory.
        object_asset_id: Optional local asset id used when resolving short
            names (defaults to the cloth prim basename, e.g. ``towel_b``).
    """

    refs = [str(u).strip() for u in (texture_uris or []) if str(u).strip()]
    if not refs:
        return

    try:
        import omni.usd
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    for env_idx in env_ids.tolist():
        prim_path = prim_path_expr.replace("{ENV_REGEX_NS}", f"/World/envs/env_{int(env_idx)}")
        texture_uri = _resolve_cloth_texture_uri(
            random.choice(refs),
            stage=stage,
            cloth_prim_path=prim_path,
            object_asset_id=object_asset_id,
        )
        if not texture_uri:
            continue
        _apply_diffuse_texture_under_prim(stage, prim_path, texture_uri)


_PLUG_IN_CHARGER_STATE_ATTR = "_mana_plug_in_charger_state"


def _set_usd_attr(prim: Any, name: str, value: Any, type_name: Any = None) -> None:
    """Set an existing USD attr, or create it when ``type_name`` is given."""
    try:
        if name in set(prim.GetPropertyNames()):
            prim.GetAttribute(name).Set(value)
            return
        if type_name is not None:
            prim.CreateAttribute(name, type_name).Set(value)
    except Exception:
        return


def apply_cloth_particle_params_usd(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    prim_path_expr: str,
    *,
    friction: float | None = None,
    cohesion: float | None = None,
    particle_friction_scale: float | None = None,
    damping: float | None = None,
    drag: float | None = None,
    lift: float | None = None,
    adhesion: float | None = None,
    contact_offset: float | None = None,
    rest_offset: float | None = None,
    particle_contact_offset: float | None = None,
    solid_rest_offset: float | None = None,
    fluid_rest_offset: float | None = None,
    solver_position_iteration_count: int | None = None,
    enable_ccd: bool | None = None,
    stretch_stiffness: float | None = None,
    bend_stiffness: float | None = None,
    shear_stiffness: float | None = None,
    spring_damping: float | None = None,
    apply_once: bool = True,
) -> None:
    """Override PhysX particle-cloth params under one asset root.

    Works for any particle-cloth asset (towel, tshirt, …): only walks
    ``prim_path_expr`` (e.g. ``{ENV_REGEX_NS}/towel_a``), so other cloths are
    untouched. Pass only the fields you want to change; ``None`` skips.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    try:
        import omni.usd
        from pxr import Sdf, Usd
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    applied_raw = getattr(env, "_mana_cloth_particle_params_applied", None)
    # Fingerprint map: re-apply when yaml params change (old set-cache silently
    # kept adhesion=0.8 after we lowered it to 0).
    if isinstance(applied_raw, dict):
        applied = applied_raw
    else:
        applied = {}
        setattr(env, "_mana_cloth_particle_params_applied", applied)

    material_attrs = {
        "physxPBDMaterial:friction": friction,
        "physxPBDMaterial:cohesion": cohesion,
        "physxPBDMaterial:particleFrictionScale": particle_friction_scale,
        "physxPBDMaterial:damping": damping,
        "physxPBDMaterial:drag": drag,
        "physxPBDMaterial:lift": lift,
        "physxPBDMaterial:adhesion": adhesion,
    }
    system_attrs = {
        "contactOffset": contact_offset,
        "restOffset": rest_offset,
        "particleContactOffset": particle_contact_offset,
        "solidRestOffset": solid_rest_offset,
        "fluidRestOffset": fluid_rest_offset,
    }
    spring_attrs = {
        "physxAutoParticleCloth:springStretchStiffness": stretch_stiffness,
        "physxAutoParticleCloth:springBendStiffness": bend_stiffness,
        "physxAutoParticleCloth:springShearStiffness": shear_stiffness,
        "physxAutoParticleCloth:springDamping": spring_damping,
    }
    fingerprint = repr(
        (
            sorted((k, v) for k, v in material_attrs.items() if v is not None),
            sorted((k, v) for k, v in system_attrs.items() if v is not None),
            sorted((k, v) for k, v in spring_attrs.items() if v is not None),
            solver_position_iteration_count,
            enable_ccd,
        )
    )

    for env_idx in env_ids.tolist():
        prim_path = prim_path_expr.replace("{ENV_REGEX_NS}", f"/World/envs/env_{int(env_idx)}")
        if apply_once and applied.get(prim_path) == fingerprint:
            continue
        root = stage.GetPrimAtPath(prim_path)
        if not root or not root.IsValid():
            continue

        n_mat = n_sys = n_cloth = 0
        for prim in Usd.PrimRange(root):
            schemas = set(prim.GetAppliedSchemas()) if hasattr(prim, "GetAppliedSchemas") else set()
            if prim.HasAPI("PhysxPBDMaterialAPI") or "PhysxPBDMaterialAPI" in schemas:
                for name, value in material_attrs.items():
                    if value is not None:
                        _set_usd_attr(prim, name, float(value), Sdf.ValueTypeNames.Float)
                n_mat += 1
            if prim.HasAPI("PhysxParticleSystemAPI") or "PhysxParticleSystemAPI" in schemas:
                for name, value in system_attrs.items():
                    if value is not None:
                        _set_usd_attr(prim, name, float(value), Sdf.ValueTypeNames.Float)
                if solver_position_iteration_count is not None:
                    _set_usd_attr(
                        prim,
                        "solverPositionIterationCount",
                        int(solver_position_iteration_count),
                        Sdf.ValueTypeNames.Int,
                    )
                if enable_ccd is not None:
                    _set_usd_attr(prim, "enableCCD", bool(enable_ccd), Sdf.ValueTypeNames.Bool)
                n_sys += 1
            if prim.HasAPI("PhysxParticleClothAPI") or "PhysxParticleClothAPI" in schemas:
                for name, value in spring_attrs.items():
                    if value is not None:
                        _set_usd_attr(prim, name, float(value), Sdf.ValueTypeNames.Float)
                n_cloth += 1

        applied[prim_path] = fingerprint
        manaprint(
            f"[apply_cloth_particle_params_usd] {prim_path}: "
            f"mat={n_mat} sys={n_sys} cloth={n_cloth} friction={friction} adhesion={adhesion}"
        )


def sample_usb_insertion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    usb_name: str = "usb",
    base_name: str = "usb_base",
    port_labels: list[str] | None = None,
    face_labels: list[str] | None = None,
    port_log_key: str = "target_port",
    face_log_key: str = "usb_face",
    instruction_template: str = (
        "Identify the {port} USB port on the hub, pick up the USB stick, "
        "flip it if needed so the face is correct, align and insert it fully "
        "into the {port} port, then return the arm home."
    ),
    seed: int | None = None,
) -> None:
    """Sample per-episode USB hub port + face, commit logs, and optionally flip the stick.

    Uniformly draws ``port_index ∈ {0,1,2}`` (left/middle/right) and
    ``face_index ∈ {0,1}`` (face up / face down). Commits both as int
    ``[num_envs, 1]`` log keys, writes the active instruction, and stores an
    overlay plan under :data:`_PLUG_IN_CHARGER_STATE_ATTR`.

    When ``face_index == 1``, rotates the USB 180 deg about its model +X long
    axis (standing upright: model +X aligns with world +Z) so the opposite face
    points forward. Also snapshots the hub root position for the
    "base must not move" insert check.
    """
    from isaaclab.utils import math as math_utils

    if env_ids is None:
        return

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    ports = [str(x) for x in (port_labels or ["left", "middle", "right"])]
    faces = [str(x) for x in (face_labels or ["face up", "face down"])]
    n_ports = len(ports)
    n_faces = len(faces)
    if n_ports <= 0:
        raise ValueError("sample_usb_insertion requires non-empty port_labels")
    if n_faces <= 0:
        raise ValueError("sample_usb_insertion requires non-empty face_labels")

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    state = getattr(env, _PLUG_IN_CHARGER_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _PLUG_IN_CHARGER_STATE_ATTR, state)
    plan_by_env: dict[int, str] = state.setdefault("plan", {})
    base_pos = state.get("base_pos")
    if not isinstance(base_pos, torch.Tensor) or base_pos.shape != (num_envs, 3):
        base_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        state["base_pos"] = base_pos

    port_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    face_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)

    has_usb = scene_has_asset(env.scene, usb_name)
    has_base = scene_has_asset(env.scene, base_name)
    usb = env.scene[usb_name] if has_usb else None
    base = env.scene[base_name] if has_base else None

    # 180 deg about model +X (XYZW).
    flip_x = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(port_log_key), seed, env_id=env_id)
        port_idx = rng.randrange(n_ports)
        face_idx = rng.randrange(n_faces)
        port_tensor[env_id, 0] = port_idx
        face_tensor[env_id, 0] = face_idx

        if base is not None:
            base_pos[env_id] = (base.data.root_pos_w.torch[env_id] - env.scene.env_origins[env_id]).detach()

        if usb is not None and face_idx == 1:
            # Flip standing USB about its long axis after layout placement.
            root = usb.data.root_pos_w.torch[env_id].clone()
            quat = usb.data.root_quat_w.torch[env_id].clone().unsqueeze(0)
            flipped = math_utils.quat_mul(quat, flip_x.unsqueeze(0))
            local_pose = torch.zeros(7, dtype=torch.float32, device=device)
            local_pose[:3] = root - env.scene.env_origins[env_id]
            local_pose[3:] = flipped[0]
            _write_single_env_local_pose(env, usb, env_id, local_pose)

        port_label = ports[port_idx]
        face_label = faces[face_idx]
        plan_by_env[env_id] = f"{port_label} port | {face_label}"
        text = str(instruction_template).format(port=port_label, face=face_label)
        write_active_instruction(env, env_id, text, num_envs=num_envs, reset_owned=True)

    write_committed(env, str(port_log_key), port_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(face_log_key), face_tensor, env_ids=ids, frame_idx=-1)


_USB_PLUG_STATE_ATTR = "_mana_usb_plug_state"


def sample_usb_dock_slot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    slot_prims: list[str] | None = None,
    slot_labels: list[str] | None = None,
    slot_log_key: str = "target_dock_slot",
    instruction_template: str = (
        "Pick up the USB stick and insert it into dock slot {slot} ({slot_prim}), " "then return the arm home."
    ),
    seed: int | None = None,
) -> None:
    """Sample per-episode dock child slot, commit index log, write instruction.

    Uniformly draws an index into ``slot_prims`` (default ``slot_a|b|c``). Commits
    the index as int ``[num_envs, 1]`` under ``slot_log_key`` for
    :func:`manaenv.terms.terminations.object_near_nearest_reference_child_xy`
    (with ``target_child_log_key``). Renders the active instruction with human
    ``slot_labels`` (default A/B/C) and stores an overlay plan under
    :data:`_USB_PLUG_STATE_ATTR`.
    """
    if env_ids is None:
        return

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    prims = [str(x).strip().lstrip("/") for x in (slot_prims or ["slot_a", "slot_b", "slot_c"]) if str(x).strip()]
    labels = [str(x) for x in (slot_labels or ["A", "B", "C"])]
    if not prims:
        raise ValueError("sample_usb_dock_slot requires non-empty slot_prims")
    if len(labels) < len(prims):
        labels = labels + [prims[i] for i in range(len(labels), len(prims))]

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    state = getattr(env, _USB_PLUG_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _USB_PLUG_STATE_ATTR, state)
    plan_by_env: dict[int, str] = state.setdefault("plan", {})

    slot_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    n_slots = len(prims)
    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(slot_log_key), seed, env_id=env_id)
        slot_idx = rng.randrange(n_slots)
        slot_tensor[env_id, 0] = slot_idx
        slot_prim = prims[slot_idx]
        slot_label = labels[slot_idx]
        plan_by_env[env_id] = f"dock slot {slot_label} ({slot_prim})"
        text = str(instruction_template).format(slot=slot_label, slot_prim=slot_prim, index=slot_idx + 1)
        write_active_instruction(env, env_id, text, num_envs=num_envs, reset_owned=True)

    write_committed(env, str(slot_log_key), slot_tensor, env_ids=ids, frame_idx=-1)


_MATCH_SHAPE_TO_SLOT_STATE_ATTR = "_mana_match_shape_to_slot_state"


def _resolve_asset_prim_path(asset: Any, env_id: int) -> str | None:
    """Resolve IsaacLab asset ``prim_path`` template for one env index."""
    cfg = getattr(asset, "cfg", None)
    prim_path = getattr(cfg, "prim_path", None) if cfg is not None else None
    if not prim_path:
        return None
    path = str(prim_path)
    path = path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}")
    path = path.replace("env_.*", f"env_{env_id}")
    if not path.startswith("/"):
        # Common template form: /World/{ENV_REGEX_NS}/Object — already absolute after replace.
        # Some assets use "envs/env_N/..." without leading /World.
        if path.startswith("envs/"):
            path = f"/World/{path}"
    return path


def _set_diffuse_tint_on_asset(
    asset: Any,
    env_id: int,
    tint_rgb: list[float] | tuple[float, float, float],
    *,
    tint_shader_suffix: str = "/Looks/color/Shader",
) -> bool:
    """Set OmniPBR ``diffuse_tint`` on ``{prim}{tint_shader_suffix}`` when present."""
    try:
        import omni.usd
        from pxr import Gf, Sdf, UsdShade
    except Exception:
        return False

    root = _resolve_asset_prim_path(asset, env_id)
    if not root:
        return False
    suffix = str(tint_shader_suffix or "/Looks/color/Shader")
    if not suffix.startswith("/"):
        suffix = "/" + suffix
    shader_path = f"{root.rstrip('/')}{suffix}"
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return False
    prim = stage.GetPrimAtPath(shader_path)
    if not prim or not prim.IsValid():
        # Fallback: first Shader under Looks that exposes diffuse_tint.
        looks = stage.GetPrimAtPath(f"{root.rstrip('/')}/Looks")
        if not looks or not looks.IsValid():
            return False
        from pxr import Usd

        prim = None
        for candidate in Usd.PrimRange(looks):
            if not candidate.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(candidate)
            if shader.GetInput("diffuse_tint") or shader.GetInput("inputs:diffuse_tint"):
                prim = candidate
                break
        if prim is None:
            return False

    shader = UsdShade.Shader(prim)
    rgb = [float(tint_rgb[0]), float(tint_rgb[1]), float(tint_rgb[2])]
    vec = Gf.Vec3f(rgb[0], rgb[1], rgb[2])
    for name in ("diffuse_tint", "inputs:diffuse_tint"):
        try:
            inp = shader.GetInput(name)
            if inp:
                inp.Set(vec)
                return True
        except Exception:
            continue
    try:
        shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(vec)
        return True
    except Exception:
        return False


def _match_shape_parse_orientations(
    orientations: list[dict[str, Any]] | None,
    *,
    num_pieces: int,
) -> list[dict[str, Any]]:
    """Normalize orientation bank entries (yaw + L/R piece indices/labels)."""
    # piece_names default order: triangle=0, square=1, round=2, trapezoid=3.
    # yaw L/R pairs confirmed in sim: 0 tri/sq, 90 round/tri, 180 trap/round, 270 sq/trap.
    default = [
        {"yaw_deg": 0, "left_idx": 0, "right_idx": 1, "left_label": "triangle", "right_label": "square"},
        {"yaw_deg": 90, "left_idx": 2, "right_idx": 0, "left_label": "round", "right_label": "triangle"},
        {"yaw_deg": 180, "left_idx": 3, "right_idx": 2, "left_label": "trapezoid", "right_label": "round"},
        {"yaw_deg": 270, "left_idx": 1, "right_idx": 3, "left_label": "square", "right_label": "trapezoid"},
    ]
    raw = list(orientations) if orientations else default
    if not raw:
        raise ValueError("sample_match_shape_to_slot requires a non-empty orientations bank")
    parsed: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"orientation entry must be a mapping, got {type(entry)}")
        left_idx = int(entry["left_idx"])
        right_idx = int(entry["right_idx"])
        if not (0 <= left_idx < num_pieces and 0 <= right_idx < num_pieces):
            raise ValueError(f"orientation piece idx out of range: left={left_idx} right={right_idx}")
        if left_idx == right_idx:
            raise ValueError("orientation left_idx and right_idx must differ")
        parsed.append(
            {
                "yaw_deg": float(entry.get("yaw_deg", 0.0)),
                "left_idx": left_idx,
                "right_idx": right_idx,
                "left_label": str(entry.get("left_label", f"shape_{left_idx}")),
                "right_label": str(entry.get("right_label", f"shape_{right_idx}")),
            }
        )
    return parsed


def _match_shape_face_vec3_map(
    payload: dict[str, Any] | None,
    *,
    defaults: dict[str, list[float]],
    what: str,
) -> dict[str, list[float]]:
    merged = {str(k): [float(x) for x in v] for k, v in defaults.items()}
    if payload:
        for key, value in payload.items():
            vec = [float(x) for x in value]
            if len(vec) != 3:
                raise ValueError(f"{what}[{key!r}] must be length-3")
            merged[str(key)] = vec
    return merged


def _match_shape_face_quat_map(
    payload: dict[str, Any] | None,
    *,
    defaults: dict[str, list[float]],
    what: str,
) -> dict[str, list[float]]:
    merged = {str(k): [float(x) for x in v] for k, v in defaults.items()}
    if payload:
        for key, value in payload.items():
            quat = [float(x) for x in value]
            if len(quat) != 4:
                raise ValueError(f"{what}[{key!r}] must be XYZW length-4")
            merged[str(key)] = quat
    return merged


def sample_match_shape_to_slot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    base_name: str = "shape_base",
    piece_names: list[str] | None = None,
    orientations: list[dict[str, Any]] | None = None,
    face_slot_offsets: dict[str, list[float]] | None = None,
    face_slot_rel_orientations: dict[str, list[float]] | None = None,
    left_slot_name: str = "slot_left",
    right_slot_name: str = "slot_right",
    left_grid_positions: list[list[float]] | None = None,
    right_grid_positions: list[list[float]] | None = None,
    piece_orientation: list[float] | None = None,
    away_z: float = -10.0,
    tint_shader_suffix: str = "/Looks/color/Shader",
    color_names: list[str] | None = None,
    color_tints: list[list[float]] | None = None,
    left_shape_log_key: str = "left_shape",
    right_shape_log_key: str = "right_shape",
    left_color_log_key: str = "left_color",
    right_color_log_key: str = "right_color",
    orientation_log_key: str = "base_orientation",
    distractor_log_key: str = "distractor_shape",
    num_distractors: int = 1,
    instruction_template: str = (
        "Identify the left-front and right-front shape holes and their matching "
        "pieces, insert the left-front {left_shape} piece into the left-front hole, "
        "insert the right-front {right_shape} piece into the right-front hole, "
        "then return the arm home."
    ),
    seed: int | None = None,
) -> None:
    """Sample base yaw orientation, sync L/R pieces onto grids, tint, re-anchor slots.

    Uniformly draws one entry from ``orientations`` (yaw delta + L/R piece indices),
    rotates ``base_name`` about world Z, places the two active pieces on the left /
    right 9-grids, exposes ``num_distractors`` unused pieces on free grid cells,
    parks the rest at ``away_z``, samples colors, re-anchors slot markers to the
    active face holes, and commits log keys for stages/overlay.
    """
    from isaaclab.utils import math as math_utils

    if env_ids is None:
        return

    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    pieces = [str(n) for n in (piece_names or ["piece_triangle", "piece_cube", "piece_round", "piece_trapezoid"])]
    if len(pieces) < 2:
        raise ValueError("sample_match_shape_to_slot requires at least two piece_names")
    orient_bank = _match_shape_parse_orientations(orientations, num_pieces=len(pieces))

    offset_defaults = {
        "triangle": [0.035, -0.045, 0.055],
        "square": [0.035, 0.045, 0.055],
        "round": [-0.035, -0.045, 0.055],
        "trapezoid": [-0.035, 0.045, 0.055],
    }
    rel_q_defaults = {
        "triangle": [0.3827, 0.0, 0.0, 0.9239],
        "square": [-0.3827, 0.0, 0.0, 0.9239],
        "round": [0.3827, 0.0, 0.0, 0.9239],
        "trapezoid": [-0.3827, 0.0, 0.0, 0.9239],
    }
    face_offsets = _match_shape_face_vec3_map(face_slot_offsets, defaults=offset_defaults, what="face_slot_offsets")
    face_rel_qs = _match_shape_face_quat_map(
        face_slot_rel_orientations, defaults=rel_q_defaults, what="face_slot_rel_orientations"
    )

    left_grid = [[float(x) for x in p] for p in (left_grid_positions or [])]
    right_grid = [[float(x) for x in p] for p in (right_grid_positions or [])]
    if not left_grid or not right_grid:
        raise ValueError("sample_match_shape_to_slot requires left_grid_positions and right_grid_positions")
    if any(len(p) != 3 for p in left_grid + right_grid):
        raise ValueError("grid positions must be length-3 XYZ lists")

    piece_quat = [float(v) for v in (piece_orientation or [0.7071, 0.0, 0.0, 0.7071])]
    if len(piece_quat) != 4:
        raise ValueError("piece_orientation must be XYZW length-4")

    names = [str(x) for x in (color_names or ["red", "yellow", "blue", "green"])]
    default_tints = [
        [1.0, 0.0, 0.0],
        [1.0, 0.88030887, 0.0],
        [0.0, 0.14675382, 0.82625484],
        [0.021496434, 0.39768338, 0.0],
    ]
    tints = list(color_tints) if color_tints is not None else default_tints
    if len(names) == 0 or len(names) != len(tints):
        raise ValueError("sample_match_shape_to_slot requires matching non-empty color_names and color_tints")

    ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    num_envs = int(getattr(env, "num_envs", len(ids)))
    device = env.device

    state = getattr(env, _MATCH_SHAPE_TO_SLOT_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, _MATCH_SHAPE_TO_SLOT_STATE_ATTR, state)
    plan_by_env: dict[int, str] = state.setdefault("plan", {})
    base_pos = state.get("base_pos")
    if not isinstance(base_pos, torch.Tensor) or base_pos.shape != (num_envs, 3):
        base_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        state["base_pos"] = base_pos

    left_shape_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    right_shape_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    left_color_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    right_color_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    orient_tensor = torch.zeros((num_envs, 1), dtype=torch.int64, device=device)
    distractor_tensor = torch.full((num_envs, 1), -1, dtype=torch.int64, device=device)
    n_distractors = max(0, int(num_distractors))

    base = env.scene[base_name] if scene_has_asset(env.scene, base_name) else None
    left_slot = env.scene[left_slot_name] if scene_has_asset(env.scene, left_slot_name) else None
    right_slot = env.scene[right_slot_name] if scene_has_asset(env.scene, right_slot_name) else None
    piece_assets = {name: env.scene[name] for name in pieces if scene_has_asset(env.scene, name)}
    piece_quat_t = torch.tensor(piece_quat, dtype=torch.float32, device=device)
    # Stable label fallback for overlay when orientations omit left/right labels.
    piece_labels = [str(n).removeprefix("piece_") for n in pieces]

    for env_id_t in ids:
        env_id = int(env_id_t.item())
        rng = _task_sample_rng(env, str(orientation_log_key), seed, env_id=env_id)
        orient = orient_bank[rng.randrange(len(orient_bank))]
        left_idx = int(orient["left_idx"])
        right_idx = int(orient["right_idx"])
        left_label = str(orient["left_label"])
        right_label = str(orient["right_label"])
        yaw_deg = float(orient["yaw_deg"])

        left_ci = rng.randrange(len(names))
        right_ci = rng.randrange(len(names))
        left_shape_tensor[env_id, 0] = left_idx
        right_shape_tensor[env_id, 0] = right_idx
        left_color_tensor[env_id, 0] = left_ci
        right_color_tensor[env_id, 0] = right_ci
        orient_tensor[env_id, 0] = int(round(yaw_deg)) % 360

        if base is not None:
            # Keep lateral position from runtime_sample; apply yaw_delta about world Z.
            pos = (base.data.root_pos_w.torch[env_id] - env.scene.env_origins[env_id]).detach()
            base_quat = base.data.root_quat_w.torch[env_id].detach()
            if abs(yaw_deg) > 1.0e-6:
                yaw_q = math_utils.quat_from_euler_xyz(
                    torch.zeros(1, device=device),
                    torch.zeros(1, device=device),
                    torch.tensor([yaw_deg * 3.141592653589793 / 180.0], device=device),
                )[0]
                base_quat = math_utils.quat_mul(yaw_q.unsqueeze(0), base_quat.unsqueeze(0))[0]
            base_pose = torch.zeros(7, dtype=torch.float32, device=device)
            base_pose[:3] = pos
            base_pose[3:] = base_quat
            _write_single_env_local_pose(env, base, env_id, base_pose)
            base_pos[env_id] = pos

            left_off = face_offsets.get(left_label) or offset_defaults["triangle"]
            right_off = face_offsets.get(right_label) or offset_defaults["square"]
            left_rel = face_rel_qs.get(left_label) or rel_q_defaults["triangle"]
            right_rel = face_rel_qs.get(right_label) or rel_q_defaults["square"]
            left_off_t = torch.tensor(left_off, dtype=torch.float32, device=device)
            right_off_t = torch.tensor(right_off, dtype=torch.float32, device=device)
            left_rel_t = torch.tensor(left_rel, dtype=torch.float32, device=device)
            right_rel_t = torch.tensor(right_rel, dtype=torch.float32, device=device)

            if left_slot is not None:
                world_off = math_utils.quat_apply(base_quat.unsqueeze(0), left_off_t.unsqueeze(0))[0]
                slot_quat = math_utils.quat_mul(base_quat.unsqueeze(0), left_rel_t.unsqueeze(0))[0]
                slot_pose = torch.zeros(7, dtype=torch.float32, device=device)
                slot_pose[:3] = pos + world_off
                slot_pose[3:] = slot_quat
                _write_single_env_local_pose(env, left_slot, env_id, slot_pose)
            if right_slot is not None:
                world_off = math_utils.quat_apply(base_quat.unsqueeze(0), right_off_t.unsqueeze(0))[0]
                slot_quat = math_utils.quat_mul(base_quat.unsqueeze(0), right_rel_t.unsqueeze(0))[0]
                slot_pose = torch.zeros(7, dtype=torch.float32, device=device)
                slot_pose[:3] = pos + world_off
                slot_pose[3:] = slot_quat
                _write_single_env_local_pose(env, right_slot, env_id, slot_pose)

        left_cell = left_grid[rng.randrange(len(left_grid))]
        right_cell = right_grid[rng.randrange(len(right_grid))]
        unused = [i for i in range(len(pieces)) if i not in (left_idx, right_idx)]
        rng.shuffle(unused)
        distractor_idxs = unused[: min(n_distractors, len(unused))]
        primary_distractor = distractor_idxs[0] if distractor_idxs else -1
        distractor_tensor[env_id, 0] = int(primary_distractor)

        # Free cells on each side for distractors (avoid landing on target cells).
        left_free = [c for c in left_grid if c != left_cell] or list(left_grid)
        right_free = [c for c in right_grid if c != right_cell] or list(right_grid)
        distractor_cells: dict[int, list[float]] = {}
        for d_idx in distractor_idxs:
            use_left = rng.random() < 0.5
            pool = left_free if use_left else right_free
            if not pool:
                pool = right_free if use_left else left_free
            cell = pool.pop(rng.randrange(len(pool)))
            distractor_cells[d_idx] = cell

        distractor_labels: list[str] = []
        for piece_i, piece_name in enumerate(pieces):
            asset = piece_assets.get(piece_name)
            if asset is None:
                continue
            pose = torch.zeros(7, dtype=torch.float32, device=device)
            pose[3:] = piece_quat_t
            if piece_i == left_idx:
                pose[:3] = torch.tensor(left_cell, dtype=torch.float32, device=device)
                _set_pose_driven_asset_visibility(asset, True, env_id)
                _set_diffuse_tint_on_asset(asset, env_id, tints[left_ci], tint_shader_suffix=tint_shader_suffix)
            elif piece_i == right_idx:
                pose[:3] = torch.tensor(right_cell, dtype=torch.float32, device=device)
                _set_pose_driven_asset_visibility(asset, True, env_id)
                _set_diffuse_tint_on_asset(asset, env_id, tints[right_ci], tint_shader_suffix=tint_shader_suffix)
            elif piece_i in distractor_cells:
                pose[:3] = torch.tensor(distractor_cells[piece_i], dtype=torch.float32, device=device)
                _set_pose_driven_asset_visibility(asset, True, env_id)
                d_ci = rng.randrange(len(names))
                _set_diffuse_tint_on_asset(asset, env_id, tints[d_ci], tint_shader_suffix=tint_shader_suffix)
                distractor_labels.append(piece_labels[piece_i] if piece_i < len(piece_labels) else piece_name)
            else:
                pose[0] = 0.05 * float(piece_i + 1)
                pose[1] = -0.5 - 0.05 * float(piece_i)
                pose[2] = float(away_z) - 0.05 * float(piece_i)
                _set_pose_driven_asset_visibility(asset, False, env_id)
            _write_single_env_local_pose(env, asset, env_id, pose)

        distractor_txt = f" | distractor {','.join(distractor_labels)}" if distractor_labels else ""
        plan_by_env[env_id] = f"yaw{int(yaw_deg)} | {left_label} left | {right_label} right{distractor_txt}"
        text = str(instruction_template).format(left_shape=left_label, right_shape=right_label)
        write_active_instruction(env, env_id, text, num_envs=num_envs, reset_owned=True)

    write_committed(env, str(left_shape_log_key), left_shape_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(right_shape_log_key), right_shape_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(left_color_log_key), left_color_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(right_color_log_key), right_color_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(orientation_log_key), orient_tensor, env_ids=ids, frame_idx=-1)
    write_committed(env, str(distractor_log_key), distractor_tensor, env_ids=ids, frame_idx=-1)


# ---------------------------------------------------------------------------
# Mobile-manip case bank applicator (compiled bank only; no reset-time YAML IO)
# ---------------------------------------------------------------------------


def apply_case_bank_layout(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    compiled_bank: Any,
    robot_asset: str = "robot",
    object_asset: str | None = None,
    context_namespace: str = "mobile_manip",
    context_plan_id: str,
    seed: int | None = None,
    bank_path: str | None = None,
    required_zone_keys: Any = None,
) -> None:
    """Apply a compile-time ``compiled_bank`` layout on reset.

    Poses are sampled in env-local coordinates. Writes follow the generic pose
    event path: USD sync of local pose → clone → add ``env_origins`` →
    ``_write_root_pose_with_zero_velocity``, then
    ``sync_dynamic_root_default_state`` so ``default_root_state`` /
    ``default_root_pose`` match the teleported env-local pose. Sampling uses
    the shared ``_task_sample_episode_token`` / master-seed event RNG contract
    so each reset advances an episode token. Does not read YAML, touch joint
    defaults, ``embodiment.initial_pose``, or handle arm/head/lift.
    """

    del bank_path, required_zone_keys  # diagnostics / compile-only; never used for IO here
    if not isinstance(compiled_bank, dict) or not compiled_bank:
        raise RuntimeError(
            "apply_case_bank_layout requires params['compiled_bank'] from task finalize; "
            "refusing missing/empty bank (no runtime file IO)"
        )

    from manaenv.pipeline.runtime import master_seed as master_seed_runtime
    from manaenv.runtime_context import set_runtime_context

    ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
    if not ids:
        return

    robot_name = str(robot_asset or "robot")
    object_name = None if object_asset in (None, "", False) else str(object_asset)
    namespace = str(context_namespace or "mobile_manip")
    plan_id = str(context_plan_id or "").strip()
    if not plan_id:
        raise RuntimeError("apply_case_bank_layout requires context_plan_id")

    zone_sets = compiled_bank.get("operation_zone_sets") or {}
    if not isinstance(zone_sets, dict) or not zone_sets:
        raise RuntimeError("apply_case_bank_layout: compiled_bank.operation_zone_sets missing")

    sample_key = f"apply_case_bank_layout:{plan_id}"
    payloads: dict[int, dict[str, Any]] = {}
    robot_poses: list[torch.Tensor] = []
    object_poses_by_asset: dict[str, list[torch.Tensor]] = {}
    robot_env_ids: list[int] = []
    object_env_ids_by_asset: dict[str, list[int]] = {}

    for env_id in ids:
        eid = int(env_id)
        episode_token = _task_sample_episode_token(env, sample_key, eid)
        rng = master_seed_runtime.event_rng(
            key=sample_key,
            env_id=eid,
            episode_token=int(episode_token),
            override=None if seed is None else int(seed),
        )
        case = _sample_case_bank_instance(compiled_bank, rng=rng, episode_token=int(episode_token))
        robot_pose = _maybe_face_live_pickable_yaw(
            env,
            compiled_bank,
            case,
            env_id=eid,
            plan_id=plan_id,
        )
        robot_poses.append(_case_bank_root_pose_tensor(robot_pose, device=env.device))
        robot_env_ids.append(eid)

        obj_pose = case.get("object_root_pose")
        if object_name and obj_pose is not None:
            active_object_name = resolve_active_scene_entity_name(env, object_name, env_id=eid)
            object_poses_by_asset.setdefault(active_object_name, []).append(
                _case_bank_root_pose_tensor(obj_pose, device=env.device)
            )
            object_env_ids_by_asset.setdefault(active_object_name, []).append(eid)

        place_target_pose = case.get("place_target_root_pose")
        roles = compiled_bank.get("roles") if isinstance(compiled_bank.get("roles"), Mapping) else {}
        place_target_name = roles.get("place_target") if isinstance(roles, Mapping) else None
        if place_target_pose is not None and place_target_name and scene_has_asset(env, str(place_target_name)):
            active_place_target_name = resolve_active_scene_entity_name(env, str(place_target_name), env_id=eid)
            object_poses_by_asset.setdefault(active_place_target_name, []).append(
                _case_bank_root_pose_tensor(place_target_pose, device=env.device)
            )
            object_env_ids_by_asset.setdefault(active_place_target_name, []).append(eid)

        zones_payload: dict[str, Any] = {}
        for zone_key, entry in zone_sets.items():
            zone = entry.get("zone") if isinstance(entry, dict) else None
            if not isinstance(zone, dict):
                raise RuntimeError(f"apply_case_bank_layout: compiled zone {zone_key!r} missing .zone")
            zones_payload[str(zone_key)] = _flat_compiled_zone(zone, roles=roles)

        payloads[eid] = {
            "case_id": str(case["case_id"]),
            "split": "id",
            "episode_token": int(episode_token),
            "robot_start_id": case.get("robot_start_id"),
            "object_position_id": case.get("object_position_id"),
            "zones": zones_payload,
        }

    robot = env.scene[robot_name]
    _write_case_bank_local_root_poses(env, robot, robot_poses, robot_env_ids)

    for active_object_name, object_poses in object_poses_by_asset.items():
        obj = env.scene[active_object_name]
        _write_case_bank_local_root_poses(
            env,
            obj,
            object_poses,
            object_env_ids_by_asset[active_object_name],
        )

    set_runtime_context(env, namespace, plan_id, robot_env_ids, payloads)


def sync_dynamic_root_default_state(
    asset: Any,
    local_poses: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    """Sync per-env default root caches to env-local poses (XYZW).

    Updates only buffers that exist on ``asset.data``:
    ``default_root_state`` and/or ``default_root_pose``. Pose rows are written
    as env-local ``[x,y,z,qx,qy,qz,qw]``; when ``default_root_state`` has
    velocity columns (width >= 13), those six dims are zeroed. Does not modify
    ``embodiment.initial_pose`` and does not write PhysX/USD (callers reuse
    ``_sync_pose_to_usd`` / ``_write_root_pose_with_zero_velocity``).

    Fail-clear on missing ``asset.data``, missing both default buffers, or
    incompatible tensor ranks/widths.
    """

    if not isinstance(local_poses, torch.Tensor) or local_poses.ndim != 2 or local_poses.shape[-1] < 7:
        raise RuntimeError(
            "sync_dynamic_root_default_state: local_poses must be Tensor[N,7+] env-local XYZW; "
            f"got shape={getattr(local_poses, 'shape', None)}"
        )
    if not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1:
        raise RuntimeError(
            "sync_dynamic_root_default_state: env_ids must be 1-D long Tensor; "
            f"got shape={getattr(env_ids, 'shape', None)}"
        )
    if int(local_poses.shape[0]) != int(env_ids.shape[0]):
        raise RuntimeError(
            "sync_dynamic_root_default_state: local_poses/env_ids length mismatch: "
            f"{tuple(local_poses.shape)} vs {tuple(env_ids.shape)}"
        )

    data = getattr(asset, "data", None)
    if data is None:
        raise RuntimeError(f"sync_dynamic_root_default_state: asset {type(asset).__name__} has no .data")

    pose = local_poses[:, :7].to(dtype=torch.float32)
    synced = False
    for attr_name in ("default_root_state", "default_root_pose"):
        raw = getattr(data, attr_name, None)
        if raw is None:
            continue
        buf = as_torch(raw)
        if not isinstance(buf, torch.Tensor) or buf.ndim != 2 or buf.shape[-1] < 7:
            raise RuntimeError(
                f"sync_dynamic_root_default_state: {attr_name} must be Tensor[num_envs,7+]; "
                f"got shape={getattr(buf, 'shape', None)}"
            )
        max_env = int(env_ids.max().item()) if env_ids.numel() else -1
        if max_env >= int(buf.shape[0]):
            raise RuntimeError(
                f"sync_dynamic_root_default_state: env_id {max_env} out of range for "
                f"{attr_name} rows={buf.shape[0]}"
            )
        pose_dev = pose.to(device=buf.device, dtype=buf.dtype)
        ids = env_ids.to(device=buf.device, dtype=torch.long)
        buf[ids, :7] = pose_dev
        if attr_name == "default_root_state" and buf.shape[-1] >= 13:
            buf[ids, 7:13] = 0
        synced = True

    if not synced:
        raise RuntimeError(
            f"sync_dynamic_root_default_state: asset {type(asset).__name__} missing "
            "default_root_state and default_root_pose"
        )


def _write_case_bank_local_root_poses(
    env: ManagerBasedEnv,
    asset: Any,
    local_poses: list[torch.Tensor],
    env_id_list: list[int],
) -> None:
    """USD-sync local poses, write world poses, sync default root caches.

    P6C/P6D: robot and object case-bank teleports both sync
    ``default_root_state`` / ``default_root_pose`` because
    ``reset_scene_to_default`` and generic pose events consume those buffers
    for any teleported asset (same consumer evidence as robot).
    """

    if not local_poses:
        return
    env_ids_t = torch.tensor(env_id_list, device=env.device, dtype=torch.long)
    local_pose = torch.stack(local_poses, dim=0)
    _sync_pose_to_usd(asset, env_ids_t, local_pose)
    sim_pose = local_pose.clone()
    origins = getattr(env.scene, "env_origins", None)
    if not isinstance(origins, torch.Tensor):
        raise RuntimeError("apply_case_bank_layout: env.scene.env_origins tensor is required")
    sim_pose[:, :3] += origins[env_ids_t]
    _write_root_pose_with_zero_velocity(asset, sim_pose, env_ids_t)
    sync_dynamic_root_default_state(asset, local_pose, env_ids_t)


def _maybe_face_live_pickable_yaw(
    env: ManagerBasedEnv,
    compiled_bank: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    env_id: int,
    plan_id: str,
) -> list[float]:
    """Rewrite robot yaw to face live pickable when sampler had no object_xy.

    Bottle already faces sampled ``object_root_pose`` inside ``_sample_disk_case``.
    Fruit (and any disk case with ``yaw_policy: face_pickable`` but no table-local
    object placement) faces the active pickable after ``runtime_sample``.
    """

    pose = list(case["robot_root_pose"])
    robot_start = compiled_bank.get("robot_start") or {}
    yaw_policy = str(robot_start.get("yaw_policy") or "")
    if yaw_policy != "face_pickable":
        return pose
    # Sampler already faced authored/sampled object_xy.
    if case.get("object_root_pose") is not None:
        return pose

    face_xy = _live_pickable_xy_env_local(env, compiled_bank, env_id=env_id, plan_id=plan_id)
    if face_xy is None:
        raise RuntimeError(
            "apply_case_bank_layout: yaw_policy=face_pickable requires a live pickable "
            f"pose after runtime_sample (plan_id={plan_id!r}, env_id={env_id})"
        )
    yaw = math.atan2(face_xy[1] - float(pose[1]), face_xy[0] - float(pose[0]))
    pose[3:7] = _yaw_to_quat_xyzw(yaw)
    return pose


def _live_pickable_xy_env_local(
    env: ManagerBasedEnv,
    compiled_bank: Mapping[str, Any],
    *,
    env_id: int,
    plan_id: str,
) -> tuple[float, float] | None:
    """Read env-local XY of the active pickable role (pool slot aware)."""

    del plan_id  # slots are keyed by schema-vNext plan; search all like terminations.
    roles = compiled_bank.get("roles") if isinstance(compiled_bank.get("roles"), Mapping) else {}
    logical = roles.get("pickable") if isinstance(roles, Mapping) else None
    if logical in (None, ""):
        return None
    # Do not pass plan_id: active slots may be recorded under the schema-vNext
    # runtime plan key, not context_plan_id (same contract as object_tilt_in_range).
    entity_name = resolve_active_scene_entity_name(env, str(logical), env_id=int(env_id))
    try:
        asset = env.scene[entity_name]
    except Exception:
        try:
            asset = env.scene[str(logical)]
        except Exception:
            return None
    data = getattr(asset, "data", None)
    root_pos = getattr(data, "root_pos_w", None) if data is not None else None
    if root_pos is None:
        return None
    pos = as_torch(root_pos)
    if not isinstance(pos, torch.Tensor) or pos.ndim < 1:
        return None
    if pos.ndim == 1:
        world_xy = pos[:2]
    else:
        world_xy = pos[int(env_id), :2]
    origins = getattr(env.scene, "env_origins", None)
    if isinstance(origins, torch.Tensor):
        world_xy = world_xy - as_torch(origins)[int(env_id), :2]
    return float(world_xy[0].item()), float(world_xy[1].item())


def _sample_case_bank_instance(
    compiled_bank: dict[str, Any],
    *,
    rng: random.Random,
    episode_token: int,
) -> dict[str, Any]:
    case_axes = compiled_bank.get("case_axes") if isinstance(compiled_bank.get("case_axes"), dict) else None
    if case_axes:
        return _sample_woodshelf_case(compiled_bank, rng=rng)
    return _sample_disk_case(compiled_bank, rng=rng, episode_token=episode_token)


def _sample_disk_case(
    compiled_bank: dict[str, Any],
    *,
    rng: random.Random,
    episode_token: int,
) -> dict[str, Any]:
    robot_start = compiled_bank.get("robot_start") or {}
    mode = str(robot_start.get("mode") or "")
    if mode == "disk":
        center = robot_start["center"]
        radius = float(robot_start["radius_m"])
        rx, ry = _sample_disk_xy(float(center[0]), float(center[1]), radius, rng=rng)
        case_prefix = "disk"
    elif mode == "rectangle":
        rx = rng.uniform(float(robot_start["x_range"][0]), float(robot_start["x_range"][1]))
        ry = rng.uniform(float(robot_start["y_range"][0]), float(robot_start["y_range"][1]))
        case_prefix = "rectangle"
    else:
        raise RuntimeError("apply_case_bank_layout: expected compiled robot_start.mode == 'disk' or 'rectangle'")

    object_root = None
    object_xy = None
    placement = compiled_bank.get("object_placement")
    if isinstance(placement, dict) and placement.get("mode") == "table_local_xy":
        lx = rng.uniform(float(placement["x_range"][0]), float(placement["x_range"][1]))
        ly = rng.uniform(float(placement["y_range"][0]), float(placement["y_range"][1]))
        support = placement["support_frame"]
        wx, wy, wz = _support_local_to_world(
            support, lx, ly, z=float(support.get("object_root_z", placement["load_pose_world"][2]))
        )
        object_xy = (wx, wy)
        object_root = [wx, wy, wz, 0.0, 0.0, 0.0, 1.0]

    yaw_policy = str(robot_start.get("yaw_policy") or "")
    if yaw_policy == "face_pickable" and object_xy is not None:
        yaw = math.atan2(object_xy[1] - ry, object_xy[0] - rx)
    else:
        yaw = rng.uniform(-math.pi, math.pi)

    quat = _yaw_to_quat_xyzw(yaw)
    out: dict[str, Any] = {
        "case_id": f"disk_e{int(episode_token)}",
        "robot_start_id": None,
        "object_position_id": None,
        "robot_root_pose": [rx, ry, 0.0, *quat],
    }
    if object_root is not None:
        out["object_root_pose"] = object_root
    place_target_root = _sample_place_target_root_pose(compiled_bank, rng=rng)
    if place_target_root is not None:
        out["place_target_root_pose"] = place_target_root
    out["case_id"] = f"{case_prefix}_e{int(episode_token)}"
    return out


def _sample_place_target_root_pose(compiled_bank: Mapping[str, Any], *, rng: random.Random) -> list[float] | None:
    placement = compiled_bank.get("place_target_placement")
    if not isinstance(placement, Mapping):
        return None
    x_range = placement.get("x_range")
    y_range = placement.get("y_range")
    if not isinstance(x_range, (list, tuple)) or not isinstance(y_range, (list, tuple)):
        raise RuntimeError("apply_case_bank_layout: compiled place_target_placement ranges missing")
    orientation_wxyz = placement.get("orientation_wxyz")
    if orientation_wxyz is None:
        quat_xyzw = [0.0, 0.0, 0.0, 1.0]
    else:
        if not isinstance(orientation_wxyz, (list, tuple)) or len(orientation_wxyz) != 4:
            raise RuntimeError("apply_case_bank_layout: compiled place_target orientation_wxyz invalid")
        quat_xyzw = [
            float(orientation_wxyz[1]),
            float(orientation_wxyz[2]),
            float(orientation_wxyz[3]),
            float(orientation_wxyz[0]),
        ]
    return [
        rng.uniform(float(x_range[0]), float(x_range[1])),
        rng.uniform(float(y_range[0]), float(y_range[1])),
        float(placement.get("z", 0.0)),
        *quat_xyzw,
    ]


def _flat_compiled_zone(zone: Mapping[str, Any], *, roles: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Copy a compiled zone into reset context, resolving circle roles."""
    shape = str(zone.get("shape") or "aabb")
    payload: dict[str, Any] = {"frame": str(zone.get("frame") or "world"), "shape": shape}
    if shape == "circle":
        center_role = str(zone["center_role"])
        resolved_center = roles.get(center_role) if isinstance(roles, Mapping) else None
        payload["center_role"] = str(resolved_center or center_role)
        payload["radius_m"] = float(zone["radius_m"])
        return payload
    payload["x_range"] = list(zone["x_range"])
    payload["y_range"] = list(zone["y_range"])
    payload["yaw_deg_range"] = list(zone["yaw_deg_range"])
    return payload


def _sample_woodshelf_case(compiled_bank: dict[str, Any], *, rng: random.Random) -> dict[str, Any]:
    axes = compiled_bank["case_axes"]
    robot_ids = list(axes["robot_start_ids"])
    object_ids = list(axes["object_position_ids"])
    robot_id = str(robot_ids[rng.randrange(len(robot_ids))])
    object_id = str(object_ids[rng.randrange(len(object_ids))])

    anchors = compiled_bank["robot_start_anchors"]
    obj_anchors = compiled_bank["object_anchors"]
    support = compiled_bank["support_frames"]["pick_support"]
    if robot_id not in anchors:
        raise RuntimeError(f"apply_case_bank_layout: missing robot_start_anchors[{robot_id!r}]")
    if object_id not in obj_anchors:
        raise RuntimeError(f"apply_case_bank_layout: missing object_anchors[{object_id!r}]")

    r_anchor = anchors[robot_id]
    ox, oy = _sample_disk_xy(
        float(r_anchor["position"][0]),
        float(r_anchor["position"][1]),
        float(r_anchor["radius_m"]),
        rng=rng,
    )
    yaw_jitter = float(r_anchor["yaw_jitter_deg"])
    yaw_deg = float(r_anchor["yaw_deg"]) + rng.uniform(-yaw_jitter, yaw_jitter)
    yaw = math.radians(yaw_deg)
    robot_pose = [ox, oy, float(r_anchor["position"][2]), *_yaw_to_quat_xyzw(yaw)]

    o_anchor = obj_anchors[object_id]
    lx = float(o_anchor["local_xy"][0])
    ly = float(o_anchor["local_xy"][1])
    if float(o_anchor["radius_m"]) > 0:
        jx, jy = _sample_disk_xy(0.0, 0.0, float(o_anchor["radius_m"]), rng=rng)
        lx += jx
        ly += jy
    wx, wy, wz = _support_local_to_world(support, lx, ly, z=float(support["object_root_z"]))
    obj_yaw = math.radians(float(o_anchor["yaw_deg"]))
    object_pose = [wx, wy, wz, *_yaw_to_quat_xyzw(obj_yaw)]

    return {
        "case_id": f"{robot_id}__{object_id}",
        "robot_start_id": robot_id,
        "object_position_id": object_id,
        "robot_root_pose": robot_pose,
        "object_root_pose": object_pose,
    }


def _case_bank_root_pose_tensor(pose: Any, *, device: torch.device) -> torch.Tensor:
    if not isinstance(pose, (list, tuple)) or len(pose) != 7:
        raise RuntimeError("apply_case_bank_layout: invalid root pose (need 7 values)")
    return torch.as_tensor(pose, dtype=torch.float32, device=device)


def _sample_disk_xy(cx: float, cy: float, radius: float, *, rng: random.Random) -> tuple[float, float]:
    if radius <= 0:
        return cx, cy
    u1 = rng.random()
    u2 = rng.random()
    r = radius * math.sqrt(u1)
    theta = 2.0 * math.pi * u2
    return cx + r * math.cos(theta), cy + r * math.sin(theta)


def _support_local_to_world(
    support: Mapping[str, Any], lx: float, ly: float, *, z: float
) -> tuple[float, float, float]:
    center = support["center"]
    x_axis = support["x_axis"]
    y_axis = support["y_axis"]
    wx = float(center[0]) + float(lx) * float(x_axis[0]) + float(ly) * float(y_axis[0])
    wy = float(center[1]) + float(lx) * float(x_axis[1]) + float(ly) * float(y_axis[1])
    return wx, wy, float(z)


def _yaw_to_quat_xyzw(yaw: float) -> list[float]:
    half = 0.5 * float(yaw)
    return [0.0, 0.0, math.sin(half), math.cos(half)]
