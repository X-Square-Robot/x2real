from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from manaenv.assets.resolver import resolve_asset_path

from ..common.diagnostics import error

SUPPORTED_HDRI_MANIFEST_SCHEMA = "schema_vnext_hdri_runtime_pool_v1"
SUPPORTED_PHOTOMETRIC_PROFILE_SCHEMA = "schema_vnext_hdri_photometric_profile_v1"


def load_lighting_texture_manifest(
    manifest_uri: str,
    *,
    target: str = "scene.lighting.dome.texture.manifest",
) -> dict[str, dict[str, Any]]:
    text = _read_manifest_text(manifest_uri, target=target)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise error(
            "schema_vnext_lighting_manifest_invalid_json",
            "lighting texture manifest is not valid JSON",
            target=target,
        ) from exc
    if not isinstance(payload, Mapping):
        raise error(
            "schema_vnext_lighting_manifest_invalid",
            "lighting texture manifest must be a mapping",
            target=target,
        )
    if payload.get("schema_version") != SUPPORTED_HDRI_MANIFEST_SCHEMA:
        raise error(
            "schema_vnext_lighting_manifest_schema_unsupported",
            "lighting texture manifest schema_version is unsupported",
            target=target,
            schema_version=str(payload.get("schema_version") or ""),
        )
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise error(
            "schema_vnext_lighting_manifest_invalid",
            "lighting texture manifest requires a candidates list",
            target=target,
        )

    pool: dict[str, dict[str, Any]] = {}
    for index, raw_candidate in enumerate(candidates):
        entry_target = f"{target}.candidates[{index}]"
        if not isinstance(raw_candidate, Mapping):
            raise error(
                "schema_vnext_lighting_manifest_entry_invalid",
                "lighting texture manifest candidates must be mappings",
                target=entry_target,
            )
        candidate_id = str(raw_candidate.get("id") or "").strip()
        uri = str(raw_candidate.get("uri") or "").strip()
        if not candidate_id or not uri:
            raise error(
                "schema_vnext_lighting_manifest_entry_invalid",
                "lighting texture candidate requires id and uri",
                target=entry_target,
            )
        if candidate_id in pool:
            raise error(
                "schema_vnext_duplicate_id",
                "lighting texture candidate ids must be unique",
                target=target,
                ref=candidate_id,
            )
        pool[candidate_id] = dict(raw_candidate)
    if not pool:
        raise error(
            "schema_vnext_lighting_manifest_empty",
            "lighting texture manifest has no candidates",
            target=target,
        )
    declared_count = payload.get("candidate_count")
    if declared_count is not None and (not isinstance(declared_count, int) or declared_count != len(pool)):
        raise error(
            "schema_vnext_lighting_manifest_count_mismatch",
            "lighting texture manifest candidate_count does not match candidates",
            target=target,
        )
    return pool


def load_lighting_photometric_profile(
    manifest_uri: str,
    *,
    target: str = "scene.lighting.photometric_profile.manifest",
) -> dict[str, Any]:
    text = _read_manifest_text(manifest_uri, target=target)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise error(
            "schema_vnext_lighting_profile_invalid_json",
            "lighting photometric profile is not valid JSON",
            target=target,
        ) from exc
    if not isinstance(payload, Mapping):
        raise error(
            "schema_vnext_lighting_profile_invalid",
            "lighting photometric profile must be a mapping",
            target=target,
        )
    if payload.get("schema_version") != SUPPORTED_PHOTOMETRIC_PROFILE_SCHEMA:
        raise error(
            "schema_vnext_lighting_profile_schema_unsupported",
            "lighting photometric profile schema_version is unsupported",
            target=target,
            schema_version=str(payload.get("schema_version") or ""),
        )
    profile_id = str(payload.get("profile_id") or "").strip()
    if not profile_id:
        raise error(
            "schema_vnext_lighting_profile_id_missing",
            "lighting photometric profile requires profile_id",
            target=target,
        )
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise error(
            "schema_vnext_lighting_profile_invalid",
            "lighting photometric profile requires a candidates list",
            target=target,
        )
    pool: dict[str, dict[str, Any]] = {}
    for index, raw_candidate in enumerate(candidates):
        entry_target = f"{target}.candidates[{index}]"
        if not isinstance(raw_candidate, Mapping):
            raise error(
                "schema_vnext_lighting_profile_entry_invalid",
                "lighting photometric candidates must be mappings",
                target=entry_target,
            )
        candidate_id = str(raw_candidate.get("id") or "").strip()
        uri = str(raw_candidate.get("uri") or "").strip()
        intensity = raw_candidate.get("dome_intensity")
        if not candidate_id or not uri or not isinstance(intensity, Mapping):
            raise error(
                "schema_vnext_lighting_profile_entry_invalid",
                "photometric candidate requires id, uri, and dome_intensity",
                target=entry_target,
            )
        if candidate_id in pool:
            raise error(
                "schema_vnext_duplicate_id",
                "lighting photometric candidate ids must be unique",
                target=target,
                ref=candidate_id,
            )
        value_range = intensity.get("range")
        nominal = _finite_number(intensity.get("nominal"))
        if not isinstance(value_range, list) or len(value_range) != 2:
            raise error(
                "schema_vnext_lighting_profile_intensity_invalid",
                "photometric intensity requires a two-value range",
                target=entry_target,
            )
        low = _finite_number(value_range[0])
        high = _finite_number(value_range[1])
        if nominal is None or low is None or high is None or not 0.0 < low <= nominal <= high:
            raise error(
                "schema_vnext_lighting_profile_intensity_invalid",
                "photometric intensity must satisfy 0 < low <= nominal <= high",
                target=entry_target,
            )
        if intensity.get("distribution") != "log_uniform":
            raise error(
                "schema_vnext_lighting_profile_distribution_invalid",
                "photometric intensity distribution must be log_uniform",
                target=entry_target,
            )
        pool[candidate_id] = {
            "id": candidate_id,
            "uri": uri,
            "dome_intensity": {
                "nominal": nominal,
                "range": [low, high],
                "distribution": "log_uniform",
            },
        }
    if not pool:
        raise error(
            "schema_vnext_lighting_profile_empty",
            "lighting photometric profile has no candidates",
            target=target,
        )
    declared_count = payload.get("candidate_count")
    if declared_count is not None and (not isinstance(declared_count, int) or declared_count != len(pool)):
        raise error(
            "schema_vnext_lighting_profile_count_mismatch",
            "lighting photometric candidate_count does not match candidates",
            target=target,
        )
    contract = payload.get("lighting_contract")
    if not isinstance(contract, Mapping):
        raise error(
            "schema_vnext_lighting_profile_contract_invalid",
            "lighting photometric profile requires lighting_contract",
            target=target,
        )
    dome = contract.get("dome") if isinstance(contract.get("dome"), Mapping) else {}
    rect = contract.get("tabletop_rect") if isinstance(contract.get("tabletop_rect"), Mapping) else {}
    recommendation = contract.get("recommendation") if isinstance(contract.get("recommendation"), Mapping) else {}
    if str(dome.get("role") or "") != "global_fill" or _finite_number(dome.get("exposure")) is None:
        raise error(
            "schema_vnext_lighting_profile_dome_contract_invalid",
            "photometric profile requires the global_fill DomeLight contract",
            target=target,
        )
    rect_intensity = _finite_number(rect.get("intensity"))
    if (
        str(rect.get("role") or "") != "tabletop_local_key"
        or rect_intensity is None
        or rect_intensity <= 0.0
        or rect.get("randomized") is not False
    ):
        raise error(
            "schema_vnext_lighting_profile_rect_contract_invalid",
            "photometric profile requires one fixed tabletop_local_key RectLight",
            target=target,
        )
    if recommendation.get("default_mode") not in {"range", "nominal"}:
        raise error(
            "schema_vnext_lighting_profile_recommendation_invalid",
            "photometric profile default_mode must be range or nominal",
            target=target,
        )
    if recommendation.get("range_distribution") != "candidate_declared":
        raise error(
            "schema_vnext_lighting_profile_recommendation_invalid",
            "photometric profile range_distribution must be candidate_declared",
            target=target,
        )
    if _finite_number(recommendation.get("default_ev_offset")) is None:
        raise error(
            "schema_vnext_lighting_profile_recommendation_invalid",
            "photometric profile default_ev_offset must be finite",
            target=target,
        )
    source = payload.get("source", {})
    if not isinstance(source, Mapping):
        raise error(
            "schema_vnext_lighting_profile_source_invalid",
            "lighting photometric profile source must be a mapping",
            target=target,
        )
    return {
        "profile_id": profile_id,
        "source": dict(source),
        "lighting_contract": dict(contract),
        "candidates": pool,
    }


def _read_manifest_text(manifest_uri: str, *, target: str) -> str:
    uri = str(manifest_uri or "").strip()
    if not uri:
        raise error(
            "schema_vnext_lighting_manifest_missing",
            "lighting texture manifest path is required",
            target=target,
        )
    try:
        path = resolve_asset_path(uri, preferred_root=Path(__file__).resolve().parents[1])
    except FileNotFoundError as exc:
        raise error(
            "schema_vnext_lighting_manifest_not_found",
            "lighting manifest file does not exist in the local asset bundle",
            target=target,
            manifest=uri,
        ) from exc
    return path.read_text(encoding="utf-8")


def _finite_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
