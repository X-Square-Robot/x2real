"""Read optional per-asset metadata from the local evaluation bundle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from manaenv.assets.resolver import is_asset_reference, resolve_asset_path

METADATA_BASENAME = "metadata.json"


def _usd_path_to_asset_dir(usd_path: str) -> str:
    """Return the directory containing an asset's metadata file."""
    clean = str(usd_path or "").strip().rstrip("/")
    if not clean:
        raise ValueError("usd_path must be non-empty")
    if clean.endswith(".usd"):
        return clean.rsplit("/", 1)[0]
    return clean


def metadata_path_for_asset_dir(asset_dir: str | Path, *, root: str | Path | None = None) -> Path:
    """Resolve ``metadata.json`` beside a local USD asset.

    ``assets/...`` and legacy ``asset://`` references are resolved from the
    configured local asset root. Explicit local paths remain accepted; no
    network or manifest lookup is attempted.
    """

    directory = _usd_path_to_asset_dir(str(asset_dir))
    if directory.endswith(METADATA_BASENAME):
        reference = directory
    elif is_asset_reference(directory) or directory.startswith("file://"):
        reference = f"{directory.rstrip('/')}/{METADATA_BASENAME}"
    else:
        reference = str(Path(directory).expanduser() / METADATA_BASENAME)
    return resolve_asset_path(reference, root=root, must_exist=False)


def load_asset_metadata(
    usd_path: str,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Load adjacent local ``metadata.json`` when present.

    Metadata is optional for public evaluation assets.  A missing or malformed
    file therefore yields an empty mapping so instruction rendering can fall
    back to the logical asset id without requiring a service connection.
    """

    try:
        metadata_path = metadata_path_for_asset_dir(usd_path, root=root)
    except (OSError, ValueError):
        return {}
    if not metadata_path.is_file():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


# Modifiers commonly present in taxonomy_tags alongside the object kind.
_TAXONOMY_MODIFIER_STOPWORDS = frozenset(
    {
        "white",
        "black",
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "grey",
        "silver",
        "gold",
        "golden",
        "transparent",
        "translucent",
        "matte",
        "glossy",
        "small",
        "large",
        "medium",
        "round",
        "square",
        "plastic",
        "metal",
        "wooden",
        "wood",
        "ceramic",
        "glass",
    }
)

# Coarse family leaves — never prefer these over display_name / raw tags.
_GENERIC_TAXONOMY_LEAVES = frozenset(
    {
        "fruit_vegetable",
        "fruit vegetable",
        "fruitvegetable",
        "food",
        "object",
        "item",
        "asset",
        "prop",
    }
)

# Process / scan / category-alias tags that are never a spoken object kind.
_TAXONOMY_PROCESS_STOPWORDS = frozenset(
    {
        "3d_scan",
        "3d scan",
        "scan",
        "model",
        "mesh",
        "usd",
        "asset",
        "4k",
        "4k_texture",
        "4k texture",
        "texture",
        "photoscan",
    }
)

_TAXONOMY_PROCESS_FRAGMENTS = frozenset({"photoscan", "texture", "4k", "scan", "model", "mesh", "usd"})


def _normalize_kind_label(label: str) -> str:
    return str(label or "").strip().replace("_", " ")


def _is_process_taxonomy_tag(tag: str) -> bool:
    """True for scan/texture/process aliases that should not become kinds."""
    compact = str(tag or "").strip().lower().replace(" ", "_")
    if not compact:
        return True
    if compact in _TAXONOMY_PROCESS_STOPWORDS:
        return True
    if _normalize_kind_label(tag).lower() in _GENERIC_TAXONOMY_LEAVES:
        return True
    parts = [part for part in compact.split("_") if part]
    return any(part in _TAXONOMY_PROCESS_FRAGMENTS for part in parts)


def _pick_kind_tag(kind_tags: list[str]) -> str:
    """Prefer a short species token over compound scan aliases."""
    ranked: list[tuple[tuple[int, int, int], str]] = []
    for index, tag in enumerate(kind_tags):
        label = _normalize_kind_label(tag)
        token_count = len([part for part in str(tag).lower().replace(" ", "_").split("_") if part])
        ranked.append(((token_count, len(label), index), label))
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def taxonomy_kind(meta: Mapping[str, Any] | None, *, fallback: str = "object") -> str:
    """Return a short object-kind label from local asset metadata."""
    if not isinstance(meta, Mapping):
        return str(fallback)

    category = meta.get("category") if isinstance(meta.get("category"), Mapping) else {}
    raw_tags = category.get("taxonomy_tags") if isinstance(category, Mapping) else None
    tags: list[str] = []
    if isinstance(raw_tags, (list, tuple)):
        tags = [str(item).strip() for item in raw_tags if str(item).strip()]

    kind_tags = [
        tag for tag in tags if tag.lower() not in _TAXONOMY_MODIFIER_STOPWORDS and not _is_process_taxonomy_tag(tag)
    ]
    if kind_tags:
        return _pick_kind_tag(kind_tags)

    species_only = [tag for tag in tags if not _is_process_taxonomy_tag(tag)]
    if species_only:
        return _pick_kind_tag(species_only)

    identity = meta.get("identity") if isinstance(meta.get("identity"), Mapping) else {}
    display_name = str(identity.get("display_name") or "").strip() if isinstance(identity, Mapping) else ""
    if display_name:
        parts = display_name.split()
        if len(parts) >= 2 and parts[-1].isdigit():
            return parts[0].lower()
        return display_name

    primary = str(category.get("primary") or "").strip() if isinstance(category, Mapping) else ""
    if primary:
        leaf = _normalize_kind_label(primary.rsplit(".", 1)[-1])
        if leaf and leaf.lower() not in _GENERIC_TAXONOMY_LEAVES:
            return leaf

    return str(fallback)


__all__ = ["METADATA_BASENAME", "load_asset_metadata", "metadata_path_for_asset_dir", "taxonomy_kind"]
