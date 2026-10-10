from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

# Optional caller-provided catalog. The public ID runtime resolves asset IDs
# from scene definitions/pools and does not bundle the broad evaluation index.
DEFAULT_STRICT_SEMANTIC_ASSET_INDEX_PATH = Path(__file__).resolve().parents[1] / "catalogs" / "strict_semantic_asset_index.json"


@dataclass(frozen=True)
class SemanticAssetRecord:
    semantic_id: str
    asset_id: str
    usd_path: str = ""
    source: str = ""

    def to_json(self) -> dict[str, Any]:
        out = {
            "semantic_id": self.semantic_id,
            "asset_id": self.asset_id,
        }
        if self.usd_path:
            out["usd_path"] = self.usd_path
        if self.source:
            out["source"] = self.source
        return out


class StrictSemanticAssetIndex:
    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        data = dict(payload or {})
        self.version = int(data.get("version", 1) or 1)
        self.policy = str(data.get("policy") or "strict").strip()
        self.records = _records_from_payload(data.get("assets"))
        self.by_usd_path = {record.usd_path: record for record in self.records.values() if record.usd_path}

    def __len__(self) -> int:
        return len(self.records)

    def resolve(self, semantic_id: str = "", *, usd_path: str = "") -> SemanticAssetRecord | None:
        key = str(semantic_id or "").strip()
        if key:
            record = self.records.get(key)
            if record is not None:
                return record
        path = str(usd_path or "").strip()
        if path:
            return self.by_usd_path.get(path)
        return None

    def contains(self, semantic_id: str = "", *, usd_path: str = "") -> bool:
        return self.resolve(semantic_id, usd_path=usd_path) is not None


def _records_from_payload(raw_assets: Any) -> dict[str, SemanticAssetRecord]:
    records: dict[str, SemanticAssetRecord] = {}
    if not isinstance(raw_assets, Mapping):
        return records
    for raw_semantic_id, raw_entry in raw_assets.items():
        semantic_id = str(raw_semantic_id or "").strip()
        if not semantic_id:
            continue
        if isinstance(raw_entry, str):
            entry = {"asset_id": raw_entry, "strict": True}
        elif isinstance(raw_entry, Mapping):
            entry = dict(raw_entry)
        else:
            continue
        if not _is_strict_entry(entry):
            continue
        asset_id = str(entry.get("asset_id") or entry.get("id") or semantic_id).strip()
        if not asset_id:
            continue
        records[semantic_id] = SemanticAssetRecord(
            semantic_id=semantic_id,
            asset_id=asset_id,
            usd_path=str(entry.get("usd_path") or "").strip(),
            source=str(entry.get("source") or "").strip(),
        )
    return records


def _is_strict_entry(entry: Mapping[str, Any]) -> bool:
    if entry.get("strict") is False:
        return False
    status = str(entry.get("status") or "strict").strip().lower()
    if status in {"weak", "approximate", "placeholder", "fallback", "non_strict"}:
        return False
    marker_keys = {
        "approximate",
        "placeholder",
        "weak_alias",
        "allow_weak_alias",
        "fallback",
        "alias_of",
        "substitute_for",
    }
    if any(bool(entry.get(key)) for key in marker_keys):
        return False
    return True


@lru_cache(maxsize=8)
def _load_index_file(path: str) -> StrictSemanticAssetIndex:
    index_path = Path(path)
    if not index_path.is_file():
        return StrictSemanticAssetIndex({})
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return StrictSemanticAssetIndex(payload if isinstance(payload, Mapping) else {})


def load_strict_semantic_asset_index(path: str | Path | None = None) -> StrictSemanticAssetIndex:
    index_path = Path(path).expanduser().resolve() if path else DEFAULT_STRICT_SEMANTIC_ASSET_INDEX_PATH
    return _load_index_file(str(index_path))


def is_non_strict_asset_reference(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if payload.get("strict") is False:
        return True
    marker_keys = {
        "approximate",
        "placeholder",
        "weak_alias",
        "allow_weak_alias",
        "fallback",
        "alias_of",
        "substitute_for",
    }
    if any(bool(payload.get(key)) for key in marker_keys):
        return True
    status = str(payload.get("status") or "").strip().lower()
    return status in {"weak", "approximate", "placeholder", "fallback", "non_strict"}


def semantic_ref_from_asset_pool_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, Mapping):
        for key in ("semantic_id", "asset_id", "id", "tags", "asset", "name"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
    return ""
