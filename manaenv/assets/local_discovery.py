"""Index and register USD/texture files from the configured local asset root."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from manaenv.assets.config import ASSET_ROOT_URI
from manaenv.assets.resolver import (
    asset_root,
    is_asset_reference,
    iter_asset_files,
    resolve_runtime_asset_reference,
)
from manaenv.utils.debug_print import manaprint

if TYPE_CHECKING:
    from isaaclab_arena.assets.registries import AssetRegistry

_USD_SUFFIXES = {".usd", ".usda", ".usdc"}
_TEXTURE_SUFFIXES = {".hdr", ".exr", ".jpg", ".jpeg", ".png", ".webp"}
_usd_paths: dict[str, str] = {}
_texture_paths: dict[str, str] = {}


def _asset_name(path: Path) -> str:
    """Return the registry name used by the source asset package."""

    if path.stem == "asset" and path.parent.name:
        return path.parent.name
    return path.stem


def _aliases(path: Path) -> set[str]:
    root = asset_root()
    relative = path.relative_to(root).with_suffix("").as_posix()
    return {_asset_name(path), path.stem, relative, relative.replace("/", "_")}


def _index() -> None:
    if _usd_paths or _texture_paths:
        return
    for path in iter_asset_files(ASSET_ROOT_URI, suffixes=_USD_SUFFIXES | _TEXTURE_SUFFIXES):
        target = _usd_paths if path.suffix.lower() in _USD_SUFFIXES else _texture_paths
        for alias in _aliases(path):
            target.setdefault(alias, str(path))


def resolve_usd_path(name: str) -> str:
    raw = str(name).strip()
    if is_asset_reference(raw) or raw.startswith("file://") or Path(raw).is_absolute():
        return resolve_runtime_asset_reference(raw, must_exist=True)
    _index()
    try:
        return _usd_paths[raw]
    except KeyError as exc:
        expected = asset_root() / raw
        raise KeyError(f"local USD asset {raw!r} is unavailable under {expected.parent}") from exc


def resolve_texture_paths(
    *,
    gen_method: int = 1,
    tags: str | list[str] | None = None,
    category: str = "skybox",
) -> list[str]:
    _index()
    if gen_method == 0:
        return [_texture_paths[str(tags)]] if isinstance(tags, str) and str(tags) in _texture_paths else []
    required = {str(item).lower() for item in ([tags] if isinstance(tags, str) else tags or [])}
    return sorted(
        {
            path
            for alias, path in _texture_paths.items()
            if (not required or all(tag in alias.lower() for tag in required))
            and (category == "object" or category.lower() in Path(path).as_posix().lower() or category == "skybox")
        }
    )


def discover_and_register_local_assets(asset_registry: AssetRegistry | None = None) -> int:
    """Register local objects/backgrounds; robot classes resolve their own USD."""

    from isaaclab_arena.assets.background_library import LibraryBackground
    from isaaclab_arena.assets.object_library import LibraryObject
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.utils.pose import Pose

    registry = asset_registry or AssetRegistry()
    _index()
    registered = 0
    root = asset_root()
    default_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    unique_paths = sorted(set(_usd_paths.values()))
    for raw_path in unique_paths:
        path = Path(raw_path)
        relative_parts = {part.lower() for part in path.relative_to(root).parts}
        if "robots" in relative_parts:
            continue
        name = _asset_name(path)
        if name in registry._components:
            continue
        is_background = bool(relative_parts & {"background", "backgrounds", "scene", "scenes"})
        tags = sorted(relative_parts - {path.name.lower()})
        base = LibraryBackground if is_background else LibraryObject
        attributes: dict[str, Any] = {
            "name": name,
            "tags": (["background"] if is_background else []) + tags,
            "usd_path": raw_path,
        }
        if is_background:
            attributes.update(initial_pose=default_pose, object_min_z=-0.4)
        else:
            attributes.update(object_type=None, scale=(1.0, 1.0, 1.0))
        cls_name = "".join(part.capitalize() for part in name.split("_")) or "LocalAsset"
        registry.register(type(cls_name, (base,), attributes), key=name)
        registered += 1
    manaprint(f"INFO: registered {registered} local evaluation asset(s) from {root}")
    return registered


__all__ = [
    "discover_and_register_local_assets",
    "resolve_texture_paths",
    "resolve_usd_path",
]
