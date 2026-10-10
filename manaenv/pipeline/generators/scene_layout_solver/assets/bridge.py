from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..common.diagnostics import error
from ..common.geometry import Bounds3D, Pose, quat_normalize
from ..scene_spec.schema import LayoutObjectSpec
from .collection import asset_map
from .collision import CollisionProxy, proxy_from_geometry
from .geometry_resolver import ObjectGeometryResolver


@dataclass(frozen=True)
class LayoutAssetRecord:
    name: str
    usd_path: str
    scale: tuple[float, float, float]
    local_bounds: Bounds3D
    collision_proxy: CollisionProxy
    default_pose: Pose | None = None
    tags: tuple[str, ...] = ()
    source_asset: Any = None


def _scale_tuple(raw: object) -> tuple[float, float, float]:
    if raw is None:
        return (1.0, 1.0, 1.0)
    if isinstance(raw, (int, float)):
        s = float(raw)
        return (s, s, s)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    raise error("asset_scale_invalid", "asset scale must be a number or 3-value list/tuple")


def _asset_pose(asset: Any) -> Pose | None:
    """Return catalog/class rest pose used as layout ``base_quaternion``.

    Instance ``initial_pose`` is mutated by ``apply_solved_poses`` to the full
    spawn pose (including a static yaw sample). Reading that instance pose here
    would bake yaw into ``base_quaternion`` and double-apply it when runtime
    sampling multiplies ``yaw_quat(yaw) * base_quaternion``.
    """
    pose = vars(type(asset)).get("initial_pose")
    if pose is None:
        return None
    position = getattr(pose, "position_xyz", None)
    rotation = getattr(pose, "rotation_xyzw", None)
    if position is None or rotation is None:
        return None
    return Pose(
        position=(float(position[0]), float(position[1]), float(position[2])),
        quaternion=quat_normalize(rotation),
    )


def build_asset_records(
    assets: Mapping[str, Any] | Iterable[Any],
    object_specs: Iterable[LayoutObjectSpec],
    *,
    geometry_resolver: ObjectGeometryResolver | None = None,
) -> dict[str, LayoutAssetRecord]:
    by_name = asset_map(assets)
    records: dict[str, LayoutAssetRecord] = {}
    resolver = geometry_resolver or ObjectGeometryResolver()
    for obj in object_specs:
        asset = by_name.get(obj.name)
        if asset is None:
            raise error("asset_missing", "layout object has no matching ManaEnv asset", target=obj.name)
        geometry = resolver.resolve(asset, inline_geometry=obj.geometry)
        if geometry is None:
            raise error(
                "asset_geometry_missing",
                "layout object requires real asset-local geometry from metadata, generated cache, USD extraction, or seed catalog",
                target=obj.name,
            )
        scale = _scale_tuple(obj.scale if obj.scale is not None else getattr(asset, "scale", None))
        scaled_bounds = geometry.local_bounds.scaled(scale)
        proxy_boxes = [box.scaled(scale) for box in geometry.proxy_boxes]
        proxy = proxy_from_geometry(scaled_bounds, proxy_boxes)
        tags = tuple(str(t) for t in getattr(asset, "tags", []) or [])
        usd_path = str(getattr(asset, "usd_path", "") or getattr(asset, "model_path", "") or "").strip()
        if not usd_path:
            raise error(
                "asset_reference_missing",
                "layout object asset must provide usd_path or model_path",
                target=obj.name,
            )
        records[obj.name] = LayoutAssetRecord(
            name=obj.name,
            usd_path=usd_path,
            scale=scale,
            local_bounds=scaled_bounds,
            collision_proxy=proxy,
            default_pose=_asset_pose(asset),
            tags=tags,
            source_asset=asset,
        )
    return records
