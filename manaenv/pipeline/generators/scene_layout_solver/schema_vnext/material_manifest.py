from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from manaenv.assets.resolver import resolve_asset_path

from ..common.diagnostics import error

_SUPPORTED_TEXTURE_CHANNELS = ("base_color", "roughness", "metallic", "opacity", "normal", "orm")
_ACCEPTED_STATUS = {"", "keep", "accepted", "final_keep", "pass"}


def load_material_template_manifest(
    manifest_uri: str,
    *,
    target: str = "background.surface_material_pool",
) -> dict[str, dict[str, Any]]:
    """Load a release material manifest and expose candidates as asset_randomizer templates."""

    text = _read_manifest_text(manifest_uri, target=target)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise error(
            "schema_vnext_material_manifest_invalid_json", "material manifest is not valid JSON", target=target
        ) from exc
    materials = payload.get("materials") if isinstance(payload, Mapping) else None
    if not isinstance(materials, list):
        raise error(
            "schema_vnext_material_manifest_invalid",
            "material manifest requires a materials list",
            target=target,
        )
    templates: dict[str, dict[str, Any]] = {}
    for index, raw_material in enumerate(materials):
        if not isinstance(raw_material, Mapping):
            raise error(
                "schema_vnext_material_manifest_entry_invalid",
                "material manifest entries must be mappings",
                target=f"{target}.materials[{index}]",
            )
        material_id = str(raw_material.get("material_id") or raw_material.get("id") or "").strip()
        if not material_id:
            raise error(
                "schema_vnext_material_manifest_entry_invalid",
                "material manifest entry requires material_id",
                target=f"{target}.materials[{index}]",
            )
        status = str(raw_material.get("status") or "").strip()
        if status and status not in _ACCEPTED_STATUS:
            continue
        templates[material_id] = material_manifest_entry_to_template(
            raw_material,
            manifest_uri=manifest_uri,
        )
    if not templates:
        raise error(
            "schema_vnext_material_manifest_empty",
            "material manifest has no accepted materials",
            target=target,
        )
    return templates


def material_manifest_entry_to_template(
    raw_material: Mapping[str, Any],
    *,
    manifest_uri: str,
) -> dict[str, Any]:
    material_id = str(raw_material.get("material_id") or raw_material.get("id") or "").strip()
    texture_refs = raw_material.get("texture_refs")
    if not isinstance(texture_refs, Mapping):
        texture_refs = {}

    pbr_set: dict[str, Any] = {}
    for channel in _SUPPORTED_TEXTURE_CHANNELS:
        uri = str(texture_refs.get(channel) or "").strip()
        if not uri:
            continue
        pbr_set[channel] = {"uri": uri}
    if not pbr_set:
        raise error(
            "schema_vnext_material_manifest_entry_no_supported_textures",
            "material manifest entry has no supported PBR texture channels",
            target=f"material:{material_id}",
        )
    if "normal" in pbr_set:
        pbr_set["normal_format"] = _normal_format(texture_refs.get("normal"))

    uv_tile_size_m = _positive_float(raw_material.get("uv_tile_size_m"), default=1.0)
    texture_scale = 1.0 / uv_tile_size_m
    family = str(raw_material.get("family") or "").strip()
    source_group = str(raw_material.get("source_group") or "").strip()
    tags = ["schema_vnext_tabletop", "tabletop_surface", material_id]
    if family:
        tags.append(family)
    if source_group:
        tags.append(source_group)
    return {
        "id": material_id,
        "family": family,
        "tags": tags,
        "sources": {"pbr_set": pbr_set},
        "compatibility": {
            "uv_required": True,
            "uv_tile_size_m": uv_tile_size_m,
            "texture_scale": texture_scale,
            "source_group": source_group,
            "manifest_uri": manifest_uri,
            "pbr_root": str(raw_material.get("pbr_root") or ""),
        },
    }


def _read_manifest_text(manifest_uri: str, *, target: str) -> str:
    uri = str(manifest_uri or "").strip()
    if not uri:
        raise error("schema_vnext_material_manifest_missing", "material manifest path is required", target=target)
    try:
        path = resolve_asset_path(uri, preferred_root=Path(__file__).resolve().parent)
    except FileNotFoundError as exc:
        raise error(
            "schema_vnext_material_manifest_not_found",
            "material manifest file does not exist in the local asset bundle",
            target=target,
            manifest=uri,
        ) from exc
    return path.read_text(encoding="utf-8")


def _normal_format(uri: Any) -> str:
    text = str(uri or "").lower()
    if "dx" in text or "directx" in text:
        return "dx"
    return "gl"


def _positive_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0.0 else default
