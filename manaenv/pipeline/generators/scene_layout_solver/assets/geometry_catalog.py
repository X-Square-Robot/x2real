from __future__ import annotations

from typing import Any

from ..common.geometry import Bounds3D


def asset_local_bounds_from_metadata(asset: Any) -> Bounds3D | None:
    """Read explicit asset-local bounds from an instantiated asset object.

    This only accepts already-materialized local bounds. It intentionally does
    not infer bounds from scale, size, or any object pose.
    """

    raw = getattr(asset, "local_bounds", None)
    if raw is not None:
        return _bounds_from_raw(raw)
    geometry = getattr(asset, "geometry", None)
    if isinstance(geometry, dict):
        raw = geometry.get("local_bounds")
        if raw is not None:
            return _bounds_from_raw(raw)
    layout_geometry = getattr(asset, "layout_geometry", None)
    if isinstance(layout_geometry, dict):
        raw = layout_geometry.get("local_bounds")
        if raw is not None:
            return _bounds_from_raw(raw)
    return None


def _asset_catalog_paths(asset: Any) -> list[str]:
    out: list[str] = []
    for attr in ("geometry_catalog_usd_path", "catalog_usd_path", "source_usd_path", "usd_path", "model_path"):
        value = str(getattr(asset, attr, "") or "").strip()
        if value and value not in out:
            out.append(value)
    return out


def _bounds_from_raw(raw: Any) -> Bounds3D:
    if isinstance(raw, Bounds3D):
        return raw
    if isinstance(raw, dict):
        return Bounds3D.from_dict(raw)
    if isinstance(raw, (list, tuple)):
        return Bounds3D.from_list(raw)
    raise TypeError("local_bounds must be a bounds dict or 6-value list")
