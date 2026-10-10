from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from manaenv.utils.math_utils import quat_wxyz_to_xyzw
from manaenv.assets.resolver import resolve_asset_path

from ..common.diagnostics import error
from ..common.geometry import QuatXYZW, Vec3, quat_normalize, quat_rotate, vec_add

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[1] / "catalogs" / "background_surface_catalog.json"


def load_background_surface_catalog(path: str | Path | None = None) -> dict[str, Any]:
    catalog_ref = str(path or DEFAULT_CATALOG_PATH)
    try:
        payload = json.loads(_read_catalog_text(path))
    except FileNotFoundError as exc:
        raise error(
            "surface_catalog_missing", "background surface catalog file is missing", target=catalog_ref
        ) from exc
    except json.JSONDecodeError as exc:
        raise error("surface_catalog_invalid_json", str(exc), target=catalog_ref) from exc
    if not isinstance(payload, dict):
        raise error("surface_catalog_invalid", "background surface catalog must be a JSON object", target=catalog_ref)
    return payload


def _read_catalog_text(path: str | Path | None = None) -> str:
    if path is None:
        return DEFAULT_CATALOG_PATH.read_text(encoding="utf-8")
    return resolve_asset_path(path, preferred_root=DEFAULT_CATALOG_PATH.parent).read_text(encoding="utf-8")


def expand_surface_catalog_refs(
    payload: dict[str, Any], *, background_poses: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Materialize layout.surface_catalog_refs into layout.surfaces.

    The solver consumes concrete manual surfaces. Scene YAMLs can keep only a
    stable reference to a background surface catalog; this resolver expands
    those refs at load time without requiring ManaEnv scene files to duplicate
    surface bounds. Catalog surfaces are authored in the background asset local
    frame; the adaptor supplies the per-scene background pose so the expanded
    records become world-frame manual surfaces for the solver.
    """

    out = dict(payload or {})
    refs = out.get("surface_catalog_refs") or out.get("surface_catalog_ref") or []
    if not refs:
        return out
    if isinstance(refs, dict):
        refs = [refs]
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list):
        raise error("surface_catalog_refs_invalid", "surface_catalog_refs must be a list, dict, or string")

    pose_map = _background_pose_map(out.get("background_poses"), background_poses=background_poses)
    default_background = _default_background(pose_map)
    catalog = load_background_surface_catalog(out.get("surface_catalog_path"))
    backgrounds = catalog.get("backgrounds")
    if not isinstance(backgrounds, dict):
        raise error("surface_catalog_backgrounds_missing", "surface catalog requires a backgrounds mapping")

    surfaces = [dict(item) for item in out.get("surfaces", []) if isinstance(item, dict)]
    existing_ids = {str(item.get("id") or "") for item in surfaces}
    expanded_meta: list[dict[str, Any]] = []
    for index, raw_ref in enumerate(refs):
        ref = _parse_surface_ref(raw_ref, index=index, default_background=default_background)
        background = str(ref["background"])
        surface_key = str(ref["surface"])
        background_entry = backgrounds.get(background)
        if not isinstance(background_entry, dict):
            raise error(
                "surface_catalog_background_unknown", "unknown background in surface catalog", target=background
            )
        background_surfaces = background_entry.get("surfaces")
        if not isinstance(background_surfaces, dict):
            raise error(
                "surface_catalog_surfaces_missing",
                "background catalog entry has no surfaces mapping",
                target=background,
            )
        surface_payload = background_surfaces.get(surface_key)
        if not isinstance(surface_payload, dict):
            raise error(
                "surface_catalog_surface_unknown",
                "unknown surface in background surface catalog",
                target=f"{background}.{surface_key}",
            )
        _validate_surface_provenance(surface_payload, background=background, surface=surface_key)
        materialized = _materialize_catalog_surface(
            surface_payload,
            background=background,
            pose_map=pose_map,
        )
        materialized["id"] = str(ref.get("as") or materialized.get("id") or surface_key)
        provenance = dict(materialized.get("provenance") or {})
        provenance.update(
            {
                "catalog_background": background,
                "catalog_surface": surface_key,
                "catalog_ref": f"{background}.{surface_key}",
                "expanded_frame": "world",
            }
        )
        materialized["provenance"] = provenance
        surface_id = str(materialized["id"])
        if surface_id in existing_ids:
            raise error(
                "surface_catalog_ref_duplicate_id",
                "expanded catalog surface duplicates an existing surface id",
                target=surface_id,
            )
        existing_ids.add(surface_id)
        surfaces.append(materialized)
        expanded_meta.append({"background": background, "surface": surface_key, "as": surface_id})

    out["surfaces"] = surfaces
    out.setdefault("surface_catalog_expanded", expanded_meta)
    return out


def _validate_surface_provenance(surface_payload: dict[str, Any], *, background: str, surface: str) -> None:
    provenance = surface_payload.get("provenance")
    source = ""
    if isinstance(provenance, dict):
        source = str(provenance.get("source") or "").strip()
    if source == "manual_initial_manaenv_examples_v1":
        raise error(
            "surface_catalog_surface_unverified",
            "background support surface is an unverified migration placeholder; replace it with manually confirmed or offline-extracted surface metadata",
            target=f"{background}.{surface}",
        )


def _background_pose_map(
    raw: Any, *, background_poses: dict[str, Any] | None = None
) -> dict[str, tuple[Vec3, QuatXYZW]]:
    merged: dict[str, Any] = {}
    if isinstance(raw, dict):
        merged.update(raw)
    if isinstance(background_poses, dict):
        merged.update(background_poses)
    out: dict[str, tuple[Vec3, QuatXYZW]] = {}
    for key, value in merged.items():
        if not isinstance(value, dict):
            continue
        position = value.get("position", value.get("initial_position", value.get("translation")))
        orientation = value.get("orientation", value.get("initial_orientation", value.get("quaternion")))
        if orientation is None and value.get("rotation_wxyz") is not None:
            orientation = quat_wxyz_to_xyzw(value["rotation_wxyz"])
        if position is None or orientation is None:
            continue
        out[str(key)] = (_vec3(position, target=f"background_pose:{key}:position"), quat_normalize(orientation))
    return out


def _materialize_catalog_surface(
    surface_payload: dict[str, Any],
    *,
    background: str,
    pose_map: dict[str, tuple[Vec3, QuatXYZW]],
) -> dict[str, Any]:
    materialized = deepcopy(surface_payload)
    frame = str(materialized.get("frame") or "background_local").strip().lower()
    if frame in {"world", "manual_world"}:
        materialized["frame"] = "world"
        return materialized
    if frame not in {"background_local", "local"}:
        raise error(
            "surface_catalog_frame_unsupported", "unsupported catalog surface frame", target=background, frame=frame
        )
    pose = pose_map.get(background)
    if pose is None:
        raise error(
            "surface_catalog_background_pose_missing",
            "background-local catalog surfaces require the scene background pose",
            target=background,
        )
    position, orientation = pose
    materialized["center"] = list(
        _transform_point(
            position, orientation, _vec3(materialized.get("center"), target=f"{background}:surface_center")
        )
    )
    materialized["x_axis"] = list(
        _transform_vector(orientation, _vec3(materialized.get("x_axis"), target=f"{background}:surface_x_axis"))
    )
    materialized["y_axis"] = list(
        _transform_vector(orientation, _vec3(materialized.get("y_axis"), target=f"{background}:surface_y_axis"))
    )
    if materialized.get("normal") is not None:
        materialized["normal"] = list(
            _transform_vector(orientation, _vec3(materialized["normal"], target=f"{background}:surface_normal"))
        )
    materialized["frame"] = "world"
    provenance = dict(materialized.get("provenance") or {})
    provenance["catalog_frame"] = frame
    provenance["background_pose"] = {"position": list(position), "orientation": list(orientation)}
    materialized["provenance"] = provenance
    return materialized


def _transform_point(position: Vec3, orientation: QuatXYZW, point: Vec3) -> Vec3:
    return vec_add(position, quat_rotate(orientation, point))


def _transform_vector(orientation: QuatXYZW, vector: Vec3) -> Vec3:
    return quat_rotate(orientation, vector)


def _vec3(values: Any, *, target: str) -> Vec3:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise error("invalid_vec3", "expected a 3-value list", target=target)
    return (float(values[0]), float(values[1]), float(values[2]))


def _default_background(pose_map: dict[str, tuple[Vec3, QuatXYZW]]) -> str:
    if len(pose_map) == 1:
        return next(iter(pose_map))
    return ""


def _parse_surface_ref(raw_ref: Any, *, index: int, default_background: str = "") -> dict[str, str]:
    if isinstance(raw_ref, str):
        parts = [part.strip() for part in raw_ref.split(".") if part.strip()]
        if len(parts) == 1 and default_background:
            return {"background": default_background, "surface": parts[0], "as": parts[0]}
        if len(parts) != 2:
            raise error(
                "surface_catalog_ref_invalid",
                "string refs must be '<background>.<surface>' or '<surface>' when the scene has one background",
                target=f"ref_{index}",
            )
        return {"background": parts[0], "surface": parts[1], "as": parts[1]}
    if not isinstance(raw_ref, dict):
        raise error(
            "surface_catalog_ref_invalid", "surface catalog ref must be a dict or string", target=f"ref_{index}"
        )
    background = str(raw_ref.get("background") or raw_ref.get("background_tag") or default_background).strip()
    surface = str(raw_ref.get("surface") or raw_ref.get("surface_id") or raw_ref.get("name") or "").strip()
    alias = str(raw_ref.get("as") or raw_ref.get("alias") or surface).strip()
    if not background or not surface:
        raise error(
            "surface_catalog_ref_missing_field",
            "surface catalog ref requires background and surface",
            target=f"ref_{index}",
        )
    return {"background": background, "surface": surface, "as": alias}
