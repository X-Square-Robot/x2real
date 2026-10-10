"""Camera-related utility helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from manaenv.pipeline.generators.scene_layout_solver.common.geometry import quat_mul, quat_normalize
from manaenv.utils.camera_constants import (
    SIDECAR_DEPTH_METERS_TO_MM,
    SIDECAR_DEPTH_UINT16_MAX_MM,
    SIDECAR_SEMANTIC_BACKGROUND_CLASS_ID,
    SIDECAR_SEMANTIC_BACKGROUND_RGB,
    SIDECAR_SEMANTIC_HASH_RGB_ADD,
    SIDECAR_SEMANTIC_HASH_RGB_MUL,
    SIDECAR_SEMANTIC_UINT8_MOD,
)

_OPENCV_LENS_DISTORTION_MODELS = frozenset({"opencvFisheye", "opencvPinhole"})
_NOMINAL_OFFSET_ATTR = "_mana_nominal_camera_offset_xyzw"
_XYZW_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def normalize_opencv_camera_intrinsics_to_render_resolution(prim: Any, *, width: int, height: int) -> bool:
    """Reauthor complete OpenCV intrinsics into the rendered pixel space.

    The operation is idempotent: a prim whose authored ``imageSize`` already
    matches the target is left untouched. Distortion coefficients are not
    changed because OpenCV's normalized-coordinate distortion model is
    independent of image resolution.

    Returns:
        ``True`` when the prim was normalized, otherwise ``False``.
    """
    try:
        target_width = int(width)
        target_height = int(height)
    except (TypeError, ValueError):
        return False
    if target_width <= 0 or target_height <= 0:
        return False

    get_attribute = getattr(prim, "GetAttribute", None)
    if not callable(get_attribute):
        return False

    model_attribute = get_attribute("omni:lensdistortion:model")
    get_model = getattr(model_attribute, "Get", None)
    model = get_model() if callable(get_model) else None
    if str(model) not in _OPENCV_LENS_DISTORTION_MODELS:
        return False

    prefix = f"omni:lensdistortion:{model}:"
    attributes = {name: get_attribute(prefix + name) for name in ("fx", "fy", "cx", "cy", "imageSize")}
    if any(
        not callable(getattr(attribute, "Get", None)) or not callable(getattr(attribute, "Set", None))
        for attribute in attributes.values()
    ):
        return False

    try:
        source_image_size = attributes["imageSize"].Get()
        source_width = int(source_image_size[0])
        source_height = int(source_image_size[1])
        intrinsics = {name: float(attributes[name].Get()) for name in ("fx", "fy", "cx", "cy")}
    except (IndexError, TypeError, ValueError):
        return False
    if source_width <= 0 or source_height <= 0 or not all(np.isfinite(value) for value in intrinsics.values()):
        return False
    if (source_width, source_height) == (target_width, target_height):
        return False

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    attributes["fx"].Set(intrinsics["fx"] * scale_x)
    attributes["cx"].Set(intrinsics["cx"] * scale_x)
    attributes["fy"].Set(intrinsics["fy"] * scale_y)
    attributes["cy"].Set(intrinsics["cy"] * scale_y)
    try:
        target_image_size = type(source_image_size)(target_width, target_height)
    except TypeError:
        target_image_size = (target_width, target_height)
    attributes["imageSize"].Set(target_image_size)
    return True


def semantic_segmentation_to_rgb_uint8(frame: Any) -> np.ndarray:
    """Colorize semantic id map ``(H,W)`` or ``(H,W,C)`` to RGB uint8 for MP4 sidecars."""
    arr = np.asarray(frame)
    if arr.ndim >= 3 and arr.shape[-1] >= 1:
        ids = arr[..., 0].astype(np.int64, copy=False)
    else:
        ids = arr.astype(np.int64, copy=False)
    h, w = ids.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    bg_id = SIDECAR_SEMANTIC_BACKGROUND_CLASS_ID
    r_m, g_m, b_m = SIDECAR_SEMANTIC_HASH_RGB_MUL
    r_a, g_a, b_a = SIDECAR_SEMANTIC_HASH_RGB_ADD
    mod = int(SIDECAR_SEMANTIC_UINT8_MOD)
    rgb[ids == bg_id] = SIDECAR_SEMANTIC_BACKGROUND_RGB
    for uid in np.unique(ids):
        if uid == bg_id:
            continue
        mask = ids == uid
        u = int(uid)
        rgb[mask, 0] = (u * r_m + r_a) % mod
        rgb[mask, 1] = (u * g_m + g_a) % mod
        rgb[mask, 2] = (u * b_m + b_a) % mod
    return rgb


def depth_to_uint16_mm_numpy(frame: Any) -> np.ndarray:
    """Convert Isaac depth in **meters** (float) or an existing mm map to ``uint16`` mm.

    Works for ``distance_to_image_plane`` (Z-depth) and ``distance_to_camera`` (ray depth);
    callers choose the sensor ``data_type`` in the embodiment config.
    """
    depth = np.asarray(frame)
    if depth.ndim >= 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if np.issubdtype(depth.dtype, np.floating):
        scale = float(SIDECAR_DEPTH_METERS_TO_MM)
        hi = float(SIDECAR_DEPTH_UINT16_MAX_MM)
        mm = np.clip(depth.astype(np.float32) * scale, 0.0, hi)
        return mm.astype(np.uint16)
    return depth.astype(np.uint16)


def disable_embodiment_camera_config(
    embodiment: Any,
    *,
    clear_camera_config: bool = True,
) -> bool:
    """Disable embodiment-owned camera sensors before env config assembly."""
    if embodiment is None:
        return False

    changed = False
    if hasattr(embodiment, "enable_cameras") and getattr(embodiment, "enable_cameras", None) is not False:
        embodiment.enable_cameras = False
        changed = True
    if clear_camera_config and getattr(embodiment, "camera_config", None) is not None:
        embodiment.camera_config = None
        changed = True
    return changed


def deactivate_robot_camera_prims(
    *,
    robot_path_token: str = "/Robot/",
    logger: Callable[[str], None] | None = None,
) -> int:
    """Deactivate USD Camera prims under the robot path token.

    Returns:
        Number of camera prims successfully deactivated.
    """
    try:
        from isaaclab.sim.utils import stage as stage_utils
    except Exception as exc:
        if logger is not None:
            logger(f"WARNING: Cannot import stage utils to disable cameras: {exc}")
        return 0

    stage = stage_utils.get_current_stage()
    if stage is None:
        return 0

    disabled = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Camera":
            continue
        prim_path = str(prim.GetPath())
        if robot_path_token not in prim_path:
            continue
        try:
            prim.SetActive(False)
            disabled += 1
        except Exception:
            continue
    return disabled


def resolve_camera_env_ids(env) -> tuple[int, ...] | None:
    """Infer environment indices from a Gym environment when available."""
    if env is None:
        return None

    unwrapped = getattr(env, "unwrapped", env)
    num_envs = getattr(unwrapped, "num_envs", None)
    if num_envs is None:
        scene = getattr(unwrapped, "scene", None)
        num_envs = getattr(scene, "num_envs", None)
    if num_envs is None:
        return None

    return tuple(range(int(num_envs)))


def normalize_camera_env_ids(env_ids: Any) -> tuple[int, ...] | None:
    """Normalize reset env ids to the sequence type expected by camera USD helpers."""
    if env_ids is None:
        return None
    if hasattr(env_ids, "detach"):
        env_ids = env_ids.detach().cpu()
    if hasattr(env_ids, "tolist"):
        env_ids = env_ids.tolist()
    if isinstance(env_ids, int):
        return (int(env_ids),)
    return tuple(int(env_id) for env_id in env_ids)


def iter_camera_cfg_items(camera_bundle: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, cfg)`` pairs for camera configs stored on a bundle object."""
    if camera_bundle is None:
        return

    for name, camera_cfg in vars(camera_bundle).items():
        if name.startswith("_"):
            continue
        if not hasattr(camera_cfg, "prim_path") or not hasattr(camera_cfg, "offset"):
            continue
        yield name, camera_cfg


def iter_matching_camera_prims(camera_cfg: Any, env_ids: Sequence[int] | None = None) -> Iterator[Any]:
    """Yield USD camera prims that correspond to a camera config."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    env_id_set = {int(env_id) for env_id in env_ids} if env_ids is not None else None
    prim_path = camera_cfg.prim_path
    suffix = prim_path.split("{ENV_REGEX_NS}", 1)[1] if "{ENV_REGEX_NS}" in prim_path else prim_path

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "{ENV_REGEX_NS}" in prim_path:
            if not path.endswith(suffix):
                continue
            if env_id_set is not None:
                matched = False
                for env_id in env_id_set:
                    if f"/env_{env_id}/" in path:
                        matched = True
                        break
                if not matched:
                    continue
        elif path != prim_path:
            continue

        yield prim


def capture_nominal_camera_offset_xyzw(camera_cfg: Any) -> dict[str, Any]:
    """Return and cache the authored nominal offset, with ``rot`` read as XYZW.

    ManaEnv 6.0 authors ``CameraCfg.OffsetCfg.rot`` as XYZW (see franka_cameras /
    EX001 head). ``convention`` is passed through untouched so downstream USD
    re-authoring keeps interpreting the offset exactly as before.
    """
    cached = getattr(camera_cfg, _NOMINAL_OFFSET_ATTR, None)
    if isinstance(cached, dict) and "pos" in cached and "rot_xyzw" in cached:
        return cached
    offset = getattr(camera_cfg, "offset", None)
    if offset is None:
        nominal = {"pos": (0.0, 0.0, 0.0), "rot_xyzw": _XYZW_IDENTITY, "convention": "opengl"}
    else:
        nominal = {
            "pos": tuple(float(v) for v in offset.pos),
            "rot_xyzw": quat_normalize(tuple(float(v) for v in offset.rot)),
            "convention": str(getattr(offset, "convention", "opengl") or "opengl"),
        }
    setattr(camera_cfg, _NOMINAL_OFFSET_ATTR, nominal)
    return nominal


def apply_camera_offset_override(
    camera_cfg: Any,
    *,
    position_delta_m: Sequence[float],
    rotation_delta_xyzw: Sequence[float],
    env: Any = None,
    env_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compose XYZW noise onto the nominal offset, write ``cfg.offset``, re-author USD.

    Writing ``cfg.offset`` is required: collection loops re-apply ``cfg.offset``
    after reset and would otherwise clobber a USD-only perturbation.
    Callers that touch several cameras should reset sensors / warm RTX once after
    the batch (see ``refresh_camera_bundle_after_reset``).
    """
    if camera_cfg is None:
        return {"updated": 0}
    nominal = capture_nominal_camera_offset_xyzw(camera_cfg)
    dpos = tuple(float(v) for v in position_delta_m)
    drot = quat_normalize(tuple(float(v) for v in rotation_delta_xyzw))
    new_pos = tuple(float(base) + delta for base, delta in zip(nominal["pos"], dpos, strict=True))
    # delta ⊗ nominal (XYZW): left-multiply applies the small noise in parent frame.
    new_rot = quat_normalize(quat_mul(drot, tuple(nominal["rot_xyzw"])))
    camera_cfg.offset.pos = new_pos
    camera_cfg.offset.rot = new_rot
    return {
        "updated": int(reapply_camera_offset_to_usd(camera_cfg, env=env, env_ids=env_ids)),
        "pos": list(new_pos),
        "rot_xyzw": list(new_rot),
        "convention": str(nominal["convention"]),
        "position_delta_m": list(dpos),
        "rotation_delta_xyzw": list(drot),
    }


def reapply_camera_offset_to_usd(
    camera_cfg,
    env=None,
    env_ids: Sequence[int] | None = None,
) -> int:
    """Re-author a camera's configured local offset back onto its USD prim.

    ManaEnv 6.0 stores ``OffsetCfg.rot`` as XYZW. When ``convention == "opengl"``
    (the common EX001 path) IsaacLab's converter is a no-op and the XYZW tuple
    is written straight to USD. For other conventions the converter still runs
    on the stored tuple.
    """
    if camera_cfg is None:
        return 0

    try:
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.utils.math import convert_camera_frame_orientation_convention
    except Exception:
        return 0

    resolved_env_ids = tuple(env_ids) if env_ids is not None else resolve_camera_env_ids(env)
    local_pos = tuple(float(v) for v in camera_cfg.offset.pos)
    rot_tensor = torch.tensor(camera_cfg.offset.rot, dtype=torch.float32).unsqueeze(0)
    local_rot = convert_camera_frame_orientation_convention(
        rot_tensor,
        origin=camera_cfg.offset.convention,
        target="opengl",
    )
    local_rot_tuple = tuple(float(v) for v in local_rot.squeeze(0).cpu().tolist())

    updated_count = 0
    for prim in iter_matching_camera_prims(camera_cfg, env_ids=resolved_env_ids):
        if sim_utils.standardize_xform_ops(prim, translation=local_pos, orientation=local_rot_tuple):
            updated_count += 1

    return updated_count


def reapply_camera_offsets_to_usd(
    camera_bundle: Any,
    env=None,
    env_ids: Sequence[int] | None = None,
) -> int:
    """Re-author all configured camera offsets from a camera bundle onto USD prims."""
    updated_count = 0
    for _, camera_cfg in iter_camera_cfg_items(camera_bundle):
        updated_count += reapply_camera_offset_to_usd(camera_cfg, env=env, env_ids=env_ids)
    return updated_count


def camera_bundle_from_gym_env(gym_env: Any) -> Any:
    """Return the assembled embodiment camera bundle attached to a Gym env."""
    assembled = getattr(gym_env, "_mana_assembled_environment", None)
    embodiment = getattr(assembled, "embodiment", None) if assembled is not None else None
    return getattr(embodiment, "camera_config", None) if embodiment is not None else None


def _get_scene_sensor(sensors: Any, name: str) -> Any:
    if sensors is None:
        return None
    get = getattr(sensors, "get", None)
    if callable(get):
        return get(name)
    try:
        return sensors[name]
    except (KeyError, TypeError, AttributeError):
        return None


def reset_configured_camera_sensors(gym_env: Any, camera_bundle: Any) -> int:
    """Reset scene sensors matching the configured camera bundle."""
    scene = getattr(gym_env, "scene", None)
    sensors = getattr(scene, "sensors", None)
    reset_count = 0
    for name, _ in iter_camera_cfg_items(camera_bundle):
        sensor = _get_scene_sensor(sensors, name)
        if sensor is None:
            continue
        reset = getattr(sensor, "reset", None)
        if callable(reset):
            reset()
            reset_count += 1
    return reset_count


def warmup_rtx_camera_sensors_after_reset(gym_env: Any, warmup_frames: int | None = None) -> int:
    """Render enough frames for RTX camera products to pick up updated USD poses."""
    sim = getattr(gym_env, "sim", None)
    if sim is None:
        return 0
    # Whether RTX camera sensors are active. Isaac Lab 3.0 removed
    # ``sim.has_rtx_sensors`` and tracks this via the ``/isaaclab/render/rtx_sensors``
    # setting (see manager_based_rl_env.py:96/256, which drives the same
    # num_rerenders_on_reset warmup). Fall back to the old method/attribute when
    # running against pre-3.0 Isaac Lab.
    has_rtx_sensors = None
    get_setting = getattr(sim, "get_setting", None)
    if callable(get_setting):
        try:
            has_rtx_sensors = bool(get_setting("/isaaclab/render/rtx_sensors"))
        except Exception:
            has_rtx_sensors = None
    if has_rtx_sensors is None:
        legacy = getattr(sim, "has_rtx_sensors", None)
        has_rtx_sensors = legacy() if callable(legacy) else legacy
    if not has_rtx_sensors:
        return 0

    if warmup_frames is None:
        cfg = getattr(gym_env, "cfg", None)
        warmup_frames = getattr(cfg, "num_rerenders_on_reset", 1)
    frames = max(int(warmup_frames or 1), 1)
    render_frame = getattr(gym_env, "render_camera_frame", None)
    if not callable(render_frame):
        render_frame = sim.render
    for _ in range(frames):
        render_frame()
    return frames


def refresh_camera_bundle_after_reset(
    gym_env: Any,
    camera_bundle: Any = None,
    env_ids: Any = None,
) -> int:
    """Re-apply camera USD offsets, reset sensors, and warm up RTX render products."""
    if camera_bundle is None:
        camera_bundle = camera_bundle_from_gym_env(gym_env)
    if camera_bundle is None:
        return 0

    normalized_env_ids = normalize_camera_env_ids(env_ids)
    refreshed = reapply_camera_offsets_to_usd(camera_bundle, env=gym_env, env_ids=normalized_env_ids)
    reset_configured_camera_sensors(gym_env, camera_bundle)
    if refreshed > 0:
        warmup_rtx_camera_sensors_after_reset(gym_env)
    return refreshed
