from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def asset_map(assets: Mapping[str, Any] | Iterable[Any]) -> dict[str, Any]:
    if isinstance(assets, Mapping):
        return dict(assets)
    out: dict[str, Any] = {}
    for asset in assets:
        name = getattr(asset, "name", None)
        if name:
            out[str(name)] = asset
    return out
