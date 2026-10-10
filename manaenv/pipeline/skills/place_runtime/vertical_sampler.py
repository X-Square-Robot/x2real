"""Place candidate samplers (vertical-bbox / center / explicit-range strategies)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from manaenv.utils.scene_object_utils import get_rigid_object_env_pose

_STRATEGY_BBOX_RATIO = "bbox_ratio"
_STRATEGY_DESTINATION_CENTER = "destination_center"
_STRATEGY_CENTROID_OF_OBJECTS = "centroid_of_objects"
_STRATEGY_XY_RANGE = "xy_range"

_ALLOWED_STRATEGIES = frozenset(
    {
        _STRATEGY_BBOX_RATIO,
        _STRATEGY_DESTINATION_CENTER,
        _STRATEGY_CENTROID_OF_OBJECTS,
        _STRATEGY_XY_RANGE,
    }
)


@dataclass
class PlaceTargetCfg:
    """Where the place skill samples its xy target in env frame."""

    strategy: str = _STRATEGY_BBOX_RATIO
    object_name: str = ""
    objects: list[str] = field(default_factory=list)
    x_ratio_range: tuple[float, float] = (0.4, 0.6)
    y_ratio_range: tuple[float, float] = (0.4, 0.6)
    x_range: tuple[float, float] = (0.0, 0.0)
    y_range: tuple[float, float] = (0.0, 0.0)
    z: float = 0.0
    avoid_objects: list[str] = field(default_factory=list)
    min_xy_clearance: float = 0.0


@dataclass
class VerticalPlaceCandidates:
    pre_place_pos: torch.Tensor
    place_pos: torch.Tensor
    target_bbox_min: torch.Tensor
    target_bbox_max: torch.Tensor


def resolve_place_candidates(
    *,
    env: Any,
    target_cfg: PlaceTargetCfg,
    env_idx: int,
    env_origin: torch.Tensor | None,
    batch_size: int,
    pre_place_z_offset: float,
    place_z_offset: float,
    fallback_size: tuple[float, float, float],
    device: torch.device,
) -> VerticalPlaceCandidates:
    """Dispatch to the configured target strategy."""
    strategy = str(target_cfg.strategy or "").strip().lower() or _STRATEGY_BBOX_RATIO
    if strategy not in _ALLOWED_STRATEGIES:
        raise ValueError(f"Unknown place target strategy {strategy!r}. Allowed: {sorted(_ALLOWED_STRATEGIES)}.")
    count = max(int(batch_size), 1)

    if strategy == _STRATEGY_CENTROID_OF_OBJECTS:
        return _resolve_centroid_of_objects(
            env=env,
            target_cfg=target_cfg,
            env_idx=env_idx,
            env_origin=env_origin,
            count=count,
            pre_place_z_offset=pre_place_z_offset,
            place_z_offset=place_z_offset,
            fallback_size=fallback_size,
            device=device,
        )
    if strategy == _STRATEGY_XY_RANGE:
        return _resolve_xy_range(
            env=env,
            target_cfg=target_cfg,
            env_idx=env_idx,
            env_origin=env_origin,
            count=count,
            pre_place_z_offset=pre_place_z_offset,
            place_z_offset=place_z_offset,
            device=device,
        )

    object_name = str(target_cfg.object_name or "").strip()
    if not object_name:
        raise ValueError(
            "PlaceTargetCfg.object_name must be set for strategies " "'bbox_ratio' and 'destination_center'."
        )
    bbox_min, bbox_max = resolve_object_bbox_env(
        env=env,
        object_name=object_name,
        env_idx=env_idx,
        env_origin=env_origin,
        fallback_size=fallback_size,
        device=device,
    )
    if strategy == _STRATEGY_DESTINATION_CENTER:
        center_x = (bbox_min[0] + bbox_max[0]) * 0.5
        center_y = (bbox_min[1] + bbox_max[1]) * 0.5
        x = center_x.repeat(count)
        y = center_y.repeat(count)
    else:  # bbox_ratio
        x_ratio = _sample_ratio(count, target_cfg.x_ratio_range, device=device)
        y_ratio = _sample_ratio(count, target_cfg.y_ratio_range, device=device)
        x = bbox_min[0] + x_ratio * (bbox_max[0] - bbox_min[0])
        y = bbox_min[1] + y_ratio * (bbox_max[1] - bbox_min[1])

    pre_z = torch.full((count,), float(bbox_max[2] + pre_place_z_offset), dtype=torch.float32, device=device)
    place_z = torch.full((count,), float(bbox_max[2] + place_z_offset), dtype=torch.float32, device=device)
    pre_place = torch.stack([x, y, pre_z], dim=-1)
    place = torch.stack([x, y, place_z], dim=-1)
    return VerticalPlaceCandidates(
        pre_place_pos=pre_place,
        place_pos=place,
        target_bbox_min=bbox_min,
        target_bbox_max=bbox_max,
    )


def _resolve_centroid_of_objects(
    *,
    env: Any,
    target_cfg: PlaceTargetCfg,
    env_idx: int,
    env_origin: torch.Tensor | None,
    count: int,
    pre_place_z_offset: float,
    place_z_offset: float,
    fallback_size: tuple[float, float, float],
    device: torch.device,
) -> VerticalPlaceCandidates:
    names = [str(n).strip() for n in (target_cfg.objects or []) if str(n).strip()]
    if len(names) < 2:
        raise ValueError(
            "PlaceTargetCfg.objects must list at least 2 object names for "
            "strategy 'centroid_of_objects'; "
            f"got {target_cfg.objects!r}."
        )
    mins: list[torch.Tensor] = []
    maxs: list[torch.Tensor] = []
    for name in names:
        bbox_min, bbox_max = resolve_object_bbox_env(
            env=env,
            object_name=name,
            env_idx=env_idx,
            env_origin=env_origin,
            fallback_size=fallback_size,
            device=device,
        )
        mins.append(bbox_min)
        maxs.append(bbox_max)
    stacked_min = torch.stack(mins, dim=0)
    stacked_max = torch.stack(maxs, dim=0)
    center_x = ((stacked_min[:, 0] + stacked_max[:, 0]) * 0.5).mean()
    center_y = ((stacked_min[:, 1] + stacked_max[:, 1]) * 0.5).mean()
    top_z = stacked_max[:, 2].max()
    overall_min = stacked_min.min(dim=0).values
    overall_max = stacked_max.max(dim=0).values

    x = center_x.repeat(count)
    y = center_y.repeat(count)
    pre_z = torch.full((count,), float(top_z + pre_place_z_offset), dtype=torch.float32, device=device)
    place_z = torch.full((count,), float(top_z + place_z_offset), dtype=torch.float32, device=device)
    pre_place = torch.stack([x, y, pre_z], dim=-1)
    place = torch.stack([x, y, place_z], dim=-1)
    return VerticalPlaceCandidates(
        pre_place_pos=pre_place,
        place_pos=place,
        target_bbox_min=overall_min,
        target_bbox_max=overall_max,
    )


def _resolve_xy_range(
    *,
    env: Any,
    target_cfg: PlaceTargetCfg,
    env_idx: int,
    env_origin: torch.Tensor | None,
    count: int,
    pre_place_z_offset: float,
    place_z_offset: float,
    device: torch.device,
) -> VerticalPlaceCandidates:
    """Sample target positions from explicit env-frame x/y ranges."""
    x, y = _sample_xy_with_clearance(
        env=env,
        target_cfg=target_cfg,
        env_idx=env_idx,
        env_origin=env_origin,
        count=count,
        device=device,
    )
    z0 = float(target_cfg.z)
    pre_z = torch.full((count,), z0 + float(pre_place_z_offset), dtype=torch.float32, device=device)
    place_z = torch.full((count,), z0 + float(place_z_offset), dtype=torch.float32, device=device)
    pre_place = torch.stack([x, y, pre_z], dim=-1)
    place = torch.stack([x, y, place_z], dim=-1)
    bbox_min = torch.tensor(
        [min(float(v) for v in target_cfg.x_range), min(float(v) for v in target_cfg.y_range), z0],
        dtype=torch.float32,
        device=device,
    )
    bbox_max = torch.tensor(
        [max(float(v) for v in target_cfg.x_range), max(float(v) for v in target_cfg.y_range), z0],
        dtype=torch.float32,
        device=device,
    )
    return VerticalPlaceCandidates(
        pre_place_pos=pre_place,
        place_pos=place,
        target_bbox_min=bbox_min,
        target_bbox_max=bbox_max,
    )


def _sample_xy_with_clearance(
    *,
    env: Any,
    target_cfg: PlaceTargetCfg,
    env_idx: int,
    env_origin: torch.Tensor | None,
    count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    avoid_names = [str(name).strip() for name in (target_cfg.avoid_objects or []) if str(name).strip()]
    min_clearance = max(float(getattr(target_cfg, "min_xy_clearance", 0.0) or 0.0), 0.0)
    if not avoid_names or min_clearance <= 0.0:
        return (
            _sample_ratio(count, target_cfg.x_range, device=device),
            _sample_ratio(count, target_cfg.y_range, device=device),
        )

    avoid_xy: list[torch.Tensor] = []
    for name in avoid_names:
        pose = get_rigid_object_env_pose(env, name, env_idx, env_origin=env_origin)
        if pose is None:
            continue
        pos, _ = pose
        avoid_xy.append(pos[:2].to(device=device, dtype=torch.float32))
    if not avoid_xy:
        return (
            _sample_ratio(count, target_cfg.x_range, device=device),
            _sample_ratio(count, target_cfg.y_range, device=device),
        )

    # Oversample, then keep points far enough from avoided objects. If the
    # requested clearance is too strict for the range, fall back to the farthest
    # candidates rather than failing the whole skill.
    candidate_count = max(count * 64, 128)
    x_candidates = _sample_ratio(candidate_count, target_cfg.x_range, device=device)
    y_candidates = _sample_ratio(candidate_count, target_cfg.y_range, device=device)
    xy_candidates = torch.stack([x_candidates, y_candidates], dim=-1)
    avoid = torch.stack(avoid_xy, dim=0)
    distances = torch.linalg.norm(xy_candidates[:, None, :] - avoid[None, :, :], dim=-1)
    min_distances = distances.min(dim=1).values
    valid = min_distances >= min_clearance
    if int(valid.sum().item()) >= count:
        selected = xy_candidates[valid][:count]
    else:
        selected = xy_candidates[torch.argsort(min_distances, descending=True)[:count]]
    return selected[:, 0], selected[:, 1]


def _sample_ratio(count: int, value_range: Sequence[float], *, device: torch.device) -> torch.Tensor:
    lo, hi = float(value_range[0]), float(value_range[1])
    if hi < lo:
        lo, hi = hi, lo
    return torch.empty(count, dtype=torch.float32, device=device).uniform_(lo, hi)


def resolve_object_bbox_env(
    *,
    env: Any,
    object_name: str,
    env_idx: int,
    env_origin: torch.Tensor | None,
    fallback_size: tuple[float, float, float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose = get_rigid_object_env_pose(env, object_name, env_idx, env_origin=env_origin)
    bbox = _try_usd_bbox_env(env, object_name, env_idx, env_origin, device=device)
    if bbox is not None:
        if pose is not None:
            center, _ = pose
            center = center.to(device=device, dtype=torch.float32)
            extent = (bbox[1] - bbox[0]).to(device=device, dtype=torch.float32)
            half = extent * 0.5
            return center - half, center + half
        return bbox
    if pose is None:
        raise RuntimeError(f"PlaceSkill could not resolve bbox or pose for place object '{object_name}'.")
    center, _ = pose
    half = torch.as_tensor(fallback_size, dtype=torch.float32, device=device).reshape(3) * 0.5
    center = center.to(device=device, dtype=torch.float32)
    return center - half, center + half


def _try_usd_bbox_env(
    env: Any,
    object_name: str,
    env_idx: int,
    env_origin: torch.Tensor | None,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception:
        return None

    scene = getattr(env, "scene", None)
    rigid_objects = getattr(scene, "rigid_objects", None) if scene is not None else None
    obj = None
    if rigid_objects is not None:
        try:
            obj = rigid_objects[object_name]
        except Exception:
            obj = None
    prim_path = str(getattr(getattr(obj, "cfg", None), "prim_path", "") or "")
    if not prim_path:
        return None
    prim_path = _resolve_env_prim_path(prim_path, env_idx)

    try:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        bbox_min = torch.tensor(list(aligned.GetMin()), dtype=torch.float32, device=device)
        bbox_max = torch.tensor(list(aligned.GetMax()), dtype=torch.float32, device=device)
        if isinstance(env_origin, torch.Tensor):
            origin = env_origin.to(device=device, dtype=torch.float32)
            bbox_min = bbox_min - origin
            bbox_max = bbox_max - origin
        return bbox_min, bbox_max
    except Exception:
        return None


def _resolve_env_prim_path(prim_path: str, env_idx: int) -> str:
    concrete_env = f"env_{int(env_idx)}"
    resolved = prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/{concrete_env}")
    return re.sub(r"env_\.\*", concrete_env, resolved, count=1)
