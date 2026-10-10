from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common.geometry import Bounds3D
from .geometry_catalog import _asset_catalog_paths, _bounds_from_raw, asset_local_bounds_from_metadata

SCHEMA_VERSION = 1
DEFAULT_GENERATED_CACHE_PATH = (
    Path(os.environ.get("SCENE_LAYOUT_OBJECT_GEOMETRY_CACHE", ""))
    if os.environ.get("SCENE_LAYOUT_OBJECT_GEOMETRY_CACHE")
    else Path.home() / ".cache" / "manaenv_scene_layout_solver" / "object_geometry_cache.json"
)


@dataclass(frozen=True)
class ObjectGeometryRecord:
    local_bounds: Bounds3D
    proxy_boxes: tuple[Bounds3D, ...] = ()
    source: str = "unknown"
    source_usd_path: str = ""
    source_hash: str = ""
    weak_cache_match: bool = False


class ObjectGeometryResolver:
    """Resolve asset-local geometry for layout solving.

    The resolver treats generated cache records as an optimization, not as the
    authority. A content hash match is strict. A path-only match is weak and is
    only used when extraction cannot run or when no better source exists.
    """

    def __init__(
        self,
        *,
        generated_cache_path: str | Path | None = None,
        enable_dynamic_extraction: bool = True,
    ) -> None:
        self.generated_cache_path = (
            Path(generated_cache_path).expanduser() if generated_cache_path else DEFAULT_GENERATED_CACHE_PATH
        )
        self.enable_dynamic_extraction = bool(enable_dynamic_extraction)
        self._generated_cache: dict[str, Any] | None = None

    def resolve(self, asset: Any, *, inline_geometry: Mapping[str, Any] | None = None) -> ObjectGeometryRecord | None:
        inline = _record_from_inline_geometry(inline_geometry or {})
        if inline is not None:
            return inline

        extract_usd_path = _extract_usd_path(asset)
        lookup_paths = _lookup_usd_paths(asset)

        metadata_bounds = asset_local_bounds_from_metadata(asset)
        if metadata_bounds is not None:
            return ObjectGeometryRecord(
                local_bounds=metadata_bounds,
                proxy_boxes=(),
                source="asset_metadata",
                source_usd_path=extract_usd_path,
            )

        source_hash = _content_hash(extract_usd_path)
        cached = self._record_from_generated_cache(lookup_paths=lookup_paths, source_hash=source_hash)
        if cached is not None and not cached.weak_cache_match:
            return cached

        if self.enable_dynamic_extraction and extract_usd_path:
            extracted = self._extract_and_cache(
                asset,
                usd_path=extract_usd_path,
                source_hash=source_hash,
                lookup_paths=lookup_paths,
            )
            if extracted is not None:
                return extracted

        if cached is not None:
            return cached

        return None

    def _record_from_generated_cache(
        self, *, lookup_paths: tuple[str, ...], source_hash: str
    ) -> ObjectGeometryRecord | None:
        cache = self._load_generated_cache()
        if source_hash:
            by_hash = cache.get("by_hash")
            if isinstance(by_hash, dict):
                record = by_hash.get(source_hash)
                if isinstance(record, dict):
                    return _record_from_cache_payload(
                        record,
                        source="generated_cache_hash",
                        source_usd_path=_first_path(lookup_paths),
                        source_hash=source_hash,
                        weak=False,
                    )
        by_path = cache.get("by_path")
        if isinstance(by_path, dict):
            for usd_path in lookup_paths:
                record = by_path.get(usd_path)
                if isinstance(record, dict):
                    record_hash = str(record.get("source_hash") or "")
                    if source_hash and record_hash and record_hash != source_hash:
                        continue
                    return _record_from_cache_payload(
                        record,
                        source="generated_cache_path",
                        source_usd_path=usd_path,
                        source_hash=record_hash or source_hash,
                        weak=True,
                    )
        return None

    def _extract_and_cache(
        self,
        asset: Any,
        *,
        usd_path: str,
        source_hash: str,
        lookup_paths: tuple[str, ...],
    ) -> ObjectGeometryRecord | None:
        try:
            payload = extract_usd_geometry_payload(usd_path)
        except Exception:
            return None
        payload["name"] = str(getattr(asset, "name", "") or "")
        payload["usd_path"] = usd_path
        payload["source_hash"] = source_hash
        payload["schema_version"] = SCHEMA_VERSION
        self._write_generated_record(payload, source_hash=source_hash, lookup_paths=lookup_paths)
        return _record_from_cache_payload(
            payload,
            source="dynamic_usd_extract",
            source_usd_path=usd_path,
            source_hash=source_hash,
            weak=False,
        )

    def _load_generated_cache(self) -> dict[str, Any]:
        if self._generated_cache is not None:
            return self._generated_cache
        path = self.generated_cache_path
        if not path.is_file():
            self._generated_cache = {"schema_version": SCHEMA_VERSION, "by_hash": {}, "by_path": {}}
            return self._generated_cache
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("schema_version", SCHEMA_VERSION)
        payload.setdefault("by_hash", {})
        payload.setdefault("by_path", {})
        self._generated_cache = payload
        return payload

    def _write_generated_record(
        self, record: dict[str, Any], *, source_hash: str, lookup_paths: tuple[str, ...]
    ) -> None:
        cache = self._load_generated_cache()
        by_hash = cache.setdefault("by_hash", {})
        by_path = cache.setdefault("by_path", {})
        if source_hash:
            by_hash[source_hash] = dict(record)
        for usd_path in lookup_paths:
            by_path[usd_path] = dict(record)
        path = self.generated_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_usd_geometry_payload(usd_path: str) -> dict[str, Any]:
    """Extract root-local bounds from a USD asset.

    This function is intentionally small and optional: importing the solver
    does not require pxr. If pxr or the target asset is unavailable, callers
    should fall back to cache/seed records or fail with asset_geometry_missing.
    """

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed_to_open_usd: {usd_path}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        children = [child for child in stage.GetPseudoRoot().GetChildren() if child.IsValid()]
        if len(children) != 1:
            raise RuntimeError(f"asset_root_prim_ambiguous: {usd_path}")
        root = children[0]
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
    aligned = bbox_cache.ComputeLocalBound(root).ComputeAlignedRange()
    if aligned.IsEmpty():
        raise RuntimeError(f"asset_bounds_empty: {usd_path}")
    return {
        "local_bounds": _range_payload(aligned),
        "root_prim_path": str(root.GetPath()),
        "proxy_boxes": [],
        "proxy_method": "aabb",
    }


def _range_payload(rng: Any) -> dict[str, list[float]]:
    mn = rng.GetMin()
    mx = rng.GetMax()
    return {
        "x": [float(mn[0]), float(mx[0])],
        "y": [float(mn[1]), float(mx[1])],
        "z": [float(mn[2]), float(mx[2])],
    }


def _record_from_inline_geometry(geometry: Mapping[str, Any]) -> ObjectGeometryRecord | None:
    raw = geometry.get("local_bounds")
    if raw is None:
        return None
    return ObjectGeometryRecord(
        local_bounds=_bounds_from_raw(raw),
        proxy_boxes=tuple(_proxy_boxes_from_payload(geometry)),
        source="scene_inline",
    )


def _record_from_cache_payload(
    payload: Mapping[str, Any],
    *,
    source: str,
    source_usd_path: str,
    source_hash: str,
    weak: bool,
) -> ObjectGeometryRecord | None:
    raw = payload.get("local_bounds")
    if raw is None:
        return None
    return ObjectGeometryRecord(
        local_bounds=_bounds_from_raw(raw),
        proxy_boxes=tuple(_proxy_boxes_from_payload(payload)),
        source=source,
        source_usd_path=source_usd_path,
        source_hash=source_hash,
        weak_cache_match=weak,
    )


def _proxy_boxes_from_payload(payload: Mapping[str, Any]) -> list[Bounds3D]:
    boxes = payload.get("proxy_boxes") or payload.get("collision_boxes") or []
    out: list[Bounds3D] = []
    if not isinstance(boxes, list):
        return out
    for item in boxes:
        if isinstance(item, Mapping):
            out.append(Bounds3D.from_dict(dict(item)))
        elif isinstance(item, (list, tuple)):
            out.append(Bounds3D.from_list(item))
    return out


def _extract_usd_path(asset: Any) -> str:
    for attr in ("usd_path", "model_path", "source_usd_path", "geometry_catalog_usd_path", "catalog_usd_path"):
        value = str(getattr(asset, attr, "") or "").strip()
        if value:
            return value
    return ""


def _lookup_usd_paths(asset: Any) -> tuple[str, ...]:
    out: list[str] = []
    for path in (_extract_usd_path(asset), *_asset_catalog_paths(asset)):
        if path and path not in out:
            out.append(path)
    return tuple(out)


def _first_path(paths: tuple[str, ...]) -> str:
    return paths[0] if paths else ""


def _content_hash(path_or_url: str) -> str:
    if not path_or_url or "://" in path_or_url:
        return ""
    path = Path(path_or_url).expanduser()
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
