from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from typing import Any

_POSE_FRAME_ATTR = "_mana_schema_vnext_canonical_to_actor_pose"
_SOURCE_RIGID_MATRIX_TOLERANCE = 1.0e-5
_SPAWN_SCALE_UNIFORM_TOLERANCE = 1.0e-6
_ACTOR_ROTATION_IDENTITY_TOLERANCE = 1.0e-6
_SOURCE_POSE_CACHE: dict[tuple[Any, ...], tuple[float, ...]] = {}

LOGGER = logging.getLogger(__name__)


def initialize_schema_vnext_asset_pose_frames(
    env: Any,
    env_ids: Any,
    asset_names: Iterable[str] = (),
) -> None:
    """Capture each schema-vNext asset's canonical-to-actor transform.

    Scene layout poses address the spawned asset's outer/default prim. IsaacLab
    root-pose APIs address the descendant rigid-body actor. The two frames differ
    when a package authors a transform such as ``Aligned`` between those prims.

    This startup event runs after PhysX initialization and before the first
    reset. It reads the fixed transform from the source USD package rather than
    the live actor state, which may already have moved because of contact, and
    patches ``default_root_state`` in place so IsaacLab's generic default reset
    continues to target the actor frame.
    """

    del env_ids  # Startup events receive None; every cloned environment is captured.
    missing: list[str] = []
    unsupported_articulations: list[str] = []
    for name in sorted({str(value).strip() for value in asset_names if str(value).strip()}):
        try:
            asset = env.scene[name]
        except Exception:
            missing.append(name)
            continue
        if not asset_supports_sim_root_pose(asset):
            continue
        if not asset_supports_canonical_pose_bridge(asset):
            unsupported_articulations.append(name)
            continue
        _initialize_asset_pose_frame(env, asset, asset_name=name)
    if missing:
        raise RuntimeError(f"schema_vnext_asset_pose_frame_missing_assets: {missing}")
    if unsupported_articulations:
        LOGGER.warning(
            "schema-vNext canonical actor-pose bridge does not yet support articulations; "
            "preserving legacy direct-root behavior for assets: %s",
            unsupported_articulations,
        )


initialize_schema_vnext_asset_pose_frames._schema_vnext_asset_pose_frame_event = True


def asset_supports_sim_root_pose(asset: Any) -> bool:
    return callable(getattr(asset, "write_root_pose_to_sim", None)) and callable(
        getattr(asset, "write_root_velocity_to_sim", None)
    )


def asset_supports_canonical_pose_bridge(asset: Any) -> bool:
    """Return whether the current bridge supports this physical asset."""

    return asset_supports_sim_root_pose(asset) and not _asset_is_articulation(asset)


def _asset_is_articulation(asset: Any) -> bool:
    if callable(getattr(asset, "write_joint_state_to_sim", None)):
        return True
    view = getattr(asset, "root_view", None)
    if view is None:
        view = getattr(asset, "root_physx_view", None)
    return getattr(view, "link_paths", None) is not None


def canonical_pose_to_actor_pose(
    env: Any,
    asset: Any,
    canonical_local_pose: Any,
    env_ids: Any,
) -> Any:
    """Convert env-local outer/default-prim poses to world-frame actor poses."""

    canonical = _as_pose_batch(canonical_local_pose, label="canonical_pose")
    offsets = _asset_pose_frame_rows(asset, env_ids, expected_rows=canonical.shape[0])
    actor_local = _compose_pose(canonical, offsets)
    actor_world = actor_local.clone()
    actor_world[:, :3] += _env_origin_rows(env, env_ids, expected_rows=canonical.shape[0])
    return actor_world


def actor_pose_to_canonical_pose(
    env: Any,
    asset: Any,
    actor_world_pose: Any,
    env_ids: Any,
) -> Any:
    """Convert world-frame actor poses to env-local outer/default-prim poses."""

    actor_local = _as_pose_batch(actor_world_pose, label="actor_pose").clone()
    actor_local[:, :3] -= _env_origin_rows(env, env_ids, expected_rows=actor_local.shape[0])
    offsets = _asset_pose_frame_rows(asset, env_ids, expected_rows=actor_local.shape[0])
    return _compose_pose(actor_local, _inverse_pose(offsets))


def _initialize_asset_pose_frame(env: Any, asset: Any, *, asset_name: str) -> None:
    if getattr(asset, _POSE_FRAME_ATTR, None) is not None:
        return

    live_actor_world = _root_link_pose_w(asset).detach().clone()
    live_actor_world = _as_pose_batch(live_actor_world, label=f"{asset_name}.root_link_pose_w")
    row_count = int(live_actor_world.shape[0])
    offsets = _authored_canonical_to_actor_poses(
        asset,
        asset_name=asset_name,
        row_count=row_count,
        reference=live_actor_world,
    )
    canonical = _configured_canonical_pose(asset, row_count=row_count, reference=live_actor_world)
    actor_local = _compose_pose(canonical, offsets)

    from manaenv.utils.tensor_utils import as_torch

    default_state = as_torch(getattr(getattr(asset, "data", None), "default_root_state", None))
    if default_state is None or getattr(default_state, "ndim", 0) != 2 or default_state.shape[0] < row_count:
        raise RuntimeError(f"schema_vnext_asset_default_root_state_missing: {asset_name}")
    if default_state.shape[1] < 7:
        raise RuntimeError(f"schema_vnext_asset_default_root_state_invalid: {asset_name}")
    default_state[:row_count, :7].copy_(actor_local)
    setattr(asset, _POSE_FRAME_ATTR, offsets.detach().clone())


def _authored_canonical_to_actor_poses(
    asset: Any,
    *,
    asset_name: str,
    row_count: int,
    reference: Any,
) -> Any:
    """Read the fixed outer-to-actor frame from the source USD package."""

    import torch

    if row_count <= 0:
        raise RuntimeError(f"schema_vnext_asset_actor_pose_rows_missing: {asset_name}")
    actor_paths = _actor_prim_paths(asset)
    relative_actor_paths: list[str] = []
    for env_index in range(row_count):
        outer_path = _outer_prim_path_for_env(asset, env_index)
        actor_path = _actor_prim_path_for_env(
            asset,
            outer_path=outer_path,
            env_index=env_index,
            actor_paths=actor_paths,
        )
        relative_actor_paths.append(actor_path[len(outer_path) :])
    relative_actor_path = relative_actor_paths[0]
    if any(path != relative_actor_path for path in relative_actor_paths[1:]):
        raise RuntimeError(f"schema_vnext_asset_actor_prim_path_inconsistent: {asset_name}")

    if not relative_actor_path:
        # IsaacLab 3.0 / ManaEnv pose rows are XYZW.
        row = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    else:
        row = _source_actor_pose(
            asset,
            asset_name=asset_name,
            relative_actor_path=relative_actor_path,
        )
    return (
        torch.tensor(
            row,
            dtype=reference.dtype,
            device=reference.device,
        )
        .unsqueeze(0)
        .repeat(int(row_count), 1)
    )


def _outer_prim_path_for_env(asset: Any, env_index: int) -> str:
    cfg = getattr(asset, "cfg", None)
    raw = str(getattr(cfg, "prim_path", "") or getattr(asset, "prim_path", "") or "")
    if not raw:
        raise RuntimeError("schema_vnext_asset_outer_prim_path_missing")
    raw = raw.replace("{ENV_REGEX_NS}", f"/World/envs/env_{int(env_index)}")
    raw = raw.replace("/World/envs/env_.*", f"/World/envs/env_{int(env_index)}")
    raw = raw.replace("env_.*", f"env_{int(env_index)}")
    return raw if raw.startswith("/") else f"/{raw}"


def _actor_prim_paths(asset: Any) -> tuple[str, ...]:
    view = getattr(asset, "root_view", None)
    prim_paths = getattr(view, "prim_paths", None)
    if prim_paths is None:
        prim_paths = getattr(view, "link_labels", None)
    if prim_paths is None:
        legacy_view = getattr(asset, "root_physx_view", None)
        prim_paths = getattr(legacy_view, "prim_paths", None)
    if prim_paths is None:
        return ()
    try:
        return tuple(str(path) for path in prim_paths if path)
    except TypeError:
        return ()


def _actor_prim_path_for_env(
    asset: Any,
    *,
    outer_path: str,
    env_index: int,
    actor_paths: Sequence[str] | None = None,
) -> str:
    candidates = _actor_prim_paths(asset) if actor_paths is None else actor_paths
    prefix = f"{outer_path}/"
    if 0 <= int(env_index) < len(candidates):
        indexed_path = candidates[int(env_index)]
        if indexed_path == outer_path or indexed_path.startswith(prefix):
            return indexed_path
    matches = sorted({path for path in candidates if path == outer_path or path.startswith(prefix)})
    if len(matches) == 1:
        return matches[0]

    relative_matches: set[str] = set()
    for candidate_index, path in enumerate(candidates):
        candidate_outer_path = _outer_prim_path_for_env(asset, candidate_index)
        candidate_prefix = f"{candidate_outer_path}/"
        if path == candidate_outer_path or path.startswith(candidate_prefix):
            relative_matches.add(path[len(candidate_outer_path) :])
    if len(relative_matches) == 1:
        return f"{outer_path}{relative_matches.pop()}"

    raise RuntimeError("schema_vnext_asset_actor_prim_path_invalid: " f"outer={outer_path} matches={matches}")


def _source_actor_pose(
    asset: Any,
    *,
    asset_name: str,
    relative_actor_path: str,
) -> tuple[float, ...]:
    from pxr import Gf, Usd, UsdGeom

    spawn = getattr(getattr(asset, "cfg", None), "spawn", None)
    usd_path = str(getattr(spawn, "usd_path", "") or "")
    if not usd_path:
        raise RuntimeError(f"schema_vnext_asset_source_usd_missing: {asset_name}")
    scale = _spawn_scale(spawn, asset_name=asset_name)
    variants = _variant_selections(getattr(spawn, "variants", None))
    cache_key = (usd_path, relative_actor_path, scale, variants)
    cached = _SOURCE_POSE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"schema_vnext_asset_source_usd_open_failed: {asset_name}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError(f"schema_vnext_asset_source_default_prim_missing: {asset_name}")
    _apply_variant_selections(default_prim, variants, asset_name=asset_name)
    actor_path = f"{default_prim.GetPath()}{relative_actor_path}"
    actor_prim = stage.GetPrimAtPath(actor_path)
    if not actor_prim.IsValid():
        raise RuntimeError("schema_vnext_asset_source_actor_prim_missing: " f"{asset_name} path={actor_path}")

    default_matrix = UsdGeom.Xformable(default_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    actor_matrix = UsdGeom.Xformable(actor_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    relative_matrix = actor_matrix * default_matrix.GetInverse()
    rigid_matrix = Gf.Matrix4d(relative_matrix)
    if not rigid_matrix.Orthonormalize():
        raise RuntimeError(f"schema_vnext_asset_source_actor_transform_invalid: {asset_name}")
    _validate_source_rigid_matrix(relative_matrix, rigid_matrix, asset_name=asset_name)
    _validate_spawn_scale_actor_rotation(scale, rigid_matrix, asset_name=asset_name)

    translation = relative_matrix.ExtractTranslation()
    translation = Gf.Vec3d(
        float(translation[0]) * scale[0],
        float(translation[1]) * scale[1],
        float(translation[2]) * scale[2],
    )
    quaternion = rigid_matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    # Store as XYZW to match IsaacLab 3.0 / ManaEnv layout pose rows.
    row = (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(quaternion.GetReal()),
    )
    quaternion_norm = math.sqrt(sum(value * value for value in row[3:7]))
    if quaternion_norm <= 1.0e-9 or not all(math.isfinite(value) for value in row):
        raise RuntimeError(f"schema_vnext_asset_source_actor_pose_invalid: {asset_name}")
    normalized = (*row[:3], *(value / quaternion_norm for value in row[3:7]))
    _SOURCE_POSE_CACHE[cache_key] = normalized
    return normalized


def _validate_source_rigid_matrix(matrix: Any, rigid_matrix: Any, *, asset_name: str) -> None:
    linear_error = max(
        abs(float(matrix[row][column]) - float(rigid_matrix[row][column])) for row in range(3) for column in range(3)
    )
    affine_error = max(
        abs(float(matrix[0][3])),
        abs(float(matrix[1][3])),
        abs(float(matrix[2][3])),
        abs(float(matrix[3][3]) - 1.0),
    )
    if max(linear_error, affine_error) > _SOURCE_RIGID_MATRIX_TOLERANCE:
        raise RuntimeError(
            "schema_vnext_asset_source_actor_transform_nonrigid: "
            f"{asset_name} linear_error={linear_error:.6g} affine_error={affine_error:.6g}"
        )


def _validate_spawn_scale_actor_rotation(
    scale: tuple[float, float, float],
    rigid_matrix: Any,
    *,
    asset_name: str,
) -> None:
    scale_extent = max(scale) - min(scale)
    scale_reference = max(1.0, max(abs(value) for value in scale))
    if scale_extent <= _SPAWN_SCALE_UNIFORM_TOLERANCE * scale_reference:
        return
    rotation_error = max(
        abs(float(rigid_matrix[row][column]) - (1.0 if row == column else 0.0))
        for row in range(3)
        for column in range(3)
    )
    if rotation_error > _ACTOR_ROTATION_IDENTITY_TOLERANCE:
        raise RuntimeError(
            "schema_vnext_asset_spawn_scale_rotated_actor_unsupported: "
            f"{asset_name} scale={scale} rotation_error={rotation_error:.6g}; "
            "pre-bake the non-uniform scale into the asset package"
        )


def _spawn_scale(spawn: Any, *, asset_name: str) -> tuple[float, float, float]:
    raw = getattr(spawn, "scale", None)
    if raw is None:
        scale = (1.0, 1.0, 1.0)
    elif isinstance(raw, (int, float)):
        scale = (float(raw),) * 3
    else:
        try:
            scale = tuple(float(value) for value in raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"schema_vnext_asset_spawn_scale_invalid: {asset_name}") from exc
    if len(scale) != 3 or not all(math.isfinite(value) and value > 0.0 for value in scale):
        raise RuntimeError(f"schema_vnext_asset_spawn_scale_invalid: {asset_name}")
    return scale


def _variant_selections(raw: Any) -> tuple[tuple[str, str], ...]:
    if raw is None:
        return ()
    values = raw if isinstance(raw, dict) else raw.to_dict()
    return tuple(sorted((str(name), str(selection)) for name, selection in values.items()))


def _apply_variant_selections(
    default_prim: Any,
    variants: tuple[tuple[str, str], ...],
    *,
    asset_name: str,
) -> None:
    variant_sets = default_prim.GetVariantSets()
    for name, selection in variants:
        if not variant_sets.HasVariantSet(name):
            raise RuntimeError("schema_vnext_asset_source_variant_missing: " f"{asset_name} variant={name}")
        variant_set = variant_sets.GetVariantSet(name)
        if not variant_set.SetVariantSelection(selection):
            raise RuntimeError(
                "schema_vnext_asset_source_variant_selection_failed: "
                f"{asset_name} variant={name} selection={selection}"
            )


def _configured_canonical_pose(asset: Any, *, row_count: int, reference: Any) -> Any:
    import torch

    cfg = getattr(asset, "cfg", None)
    init_state = getattr(cfg, "init_state", None)
    if init_state is None:
        raise RuntimeError("schema_vnext_asset_init_state_missing")
    position = tuple(getattr(init_state, "pos", ()) or ())
    rotation = tuple(getattr(init_state, "rot", ()) or ())
    if len(position) != 3 or len(rotation) != 4:
        raise RuntimeError("schema_vnext_asset_init_state_pose_invalid")
    row = torch.tensor(
        (*position, *rotation),
        dtype=reference.dtype,
        device=reference.device,
    )
    pose = row.unsqueeze(0).repeat(int(row_count), 1)
    pose[:, 3:7] = _normalize_quaternion(pose[:, 3:7], label="configured_canonical_pose")
    return pose


def _root_link_pose_w(asset: Any) -> Any:
    from manaenv.utils.tensor_utils import as_torch

    data = getattr(asset, "data", None)
    for field_name in ("root_link_pose_w", "root_pose_w"):
        try:
            value = getattr(data, field_name)
        except Exception:
            continue
        if value is not None:
            return as_torch(value)
    raise RuntimeError("schema_vnext_asset_root_link_pose_missing")


def _asset_pose_frame_rows(asset: Any, env_ids: Any, *, expected_rows: int) -> Any:
    offsets = getattr(asset, _POSE_FRAME_ATTR, None)
    if offsets is None:
        raise RuntimeError(
            "schema_vnext_asset_pose_frame_uninitialized: "
            "the startup pose-frame event must run before reset-time pose conversion"
        )
    rows = offsets if env_ids is None else offsets[env_ids]
    rows = _as_pose_batch(rows, label="canonical_to_actor_pose")
    if int(rows.shape[0]) != int(expected_rows):
        raise RuntimeError(
            "schema_vnext_asset_pose_frame_row_mismatch: " f"expected={int(expected_rows)} actual={int(rows.shape[0])}"
        )
    return rows


def _env_origin_rows(env: Any, env_ids: Any, *, expected_rows: int) -> Any:
    origins = env.scene.env_origins
    rows = origins if env_ids is None else origins[env_ids]
    if getattr(rows, "ndim", 0) == 1:
        rows = rows.unsqueeze(0)
    if int(rows.shape[0]) < int(expected_rows):
        raise RuntimeError(
            "schema_vnext_env_origin_row_mismatch: " f"expected={int(expected_rows)} actual={int(rows.shape[0])}"
        )
    return rows[:expected_rows]


def _as_pose_batch(value: Any, *, label: str) -> Any:
    if getattr(value, "ndim", 0) == 1:
        value = value.unsqueeze(0)
    if getattr(value, "ndim", 0) != 2 or int(value.shape[1]) != 7:
        raise RuntimeError(f"schema_vnext_{label}_shape_invalid")
    return value


def _compose_pose(first: Any, second: Any) -> Any:
    first = _as_pose_batch(first, label="compose_first")
    second = _as_pose_batch(second, label="compose_second")
    first_quaternion = _normalize_quaternion(first[:, 3:7], label="compose_first")
    second_quaternion = _normalize_quaternion(second[:, 3:7], label="compose_second")
    position = first[:, :3] + _quat_apply(first_quaternion, second[:, :3])
    quaternion = _quat_multiply(
        first_quaternion,
        second_quaternion,
    )
    return _join_pose(position, quaternion)


def _inverse_pose(pose: Any) -> Any:
    pose = _as_pose_batch(pose, label="inverse_pose")
    quaternion = _quat_inverse(_normalize_quaternion(pose[:, 3:7], label="inverse_pose"))
    position = _quat_apply(quaternion, -pose[:, :3])
    return _join_pose(position, quaternion)


def _join_pose(position: Any, quaternion: Any) -> Any:
    import torch

    return torch.cat((position, _normalize_quaternion(quaternion, label="pose_result")), dim=-1)


def _normalize_quaternion(quaternion: Any, *, label: str) -> Any:
    import torch

    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool(torch.any(norm <= 1.0e-9)):
        raise RuntimeError(f"schema_vnext_{label}_quaternion_invalid")
    return quaternion / norm


def _quat_inverse(quaternion: Any) -> Any:
    # XYZW conjugate: negate xyz, keep w.
    inverse = quaternion.clone()
    inverse[:, :3] *= -1.0
    return inverse


def _quat_multiply(first: Any, second: Any) -> Any:
    import torch

    first_xyz, first_w = first[:, :3], first[:, 3:4]
    second_xyz, second_w = second[:, :3], second[:, 3:4]
    scalar = first_w * second_w - torch.sum(first_xyz * second_xyz, dim=-1, keepdim=True)
    vector = first_w * second_xyz + second_w * first_xyz + torch.cross(first_xyz, second_xyz, dim=-1)
    return torch.cat((vector, scalar), dim=-1)


def _quat_apply(quaternion: Any, vector: Any) -> Any:
    import torch

    quaternion_xyz = quaternion[:, :3]
    twice_cross = 2.0 * torch.cross(quaternion_xyz, vector, dim=-1)
    return vector + quaternion[:, 3:4] * twice_cross + torch.cross(quaternion_xyz, twice_cross, dim=-1)
