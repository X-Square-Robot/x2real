"""Resolve evaluation assets from one local directory."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ASSET_ROOT = _REPO_ROOT / "assets"
_REPO_ASSET_PREFIX = "assets"
_LEGACY_ASSET_SCHEME = "asset://"


class AssetResolutionError(FileNotFoundError):
    """A logical evaluation asset is not present in the local bundle."""

    def __init__(self, reference: str, expected_path: Path, reason: str = "asset is missing") -> None:
        self.reference = str(reference)
        self.expected_path = Path(expected_path)
        super().__init__(f"{reason}: {reference} (expected {self.expected_path})")


def default_asset_root() -> Path:
    configured = os.environ.get("MANAENV_ASSET_ROOT", "").strip()
    return Path(configured).expanduser().resolve() if configured else _DEFAULT_ASSET_ROOT


def asset_root(root: str | Path | None = None) -> Path:
    return (Path(root).expanduser() if root is not None else default_asset_root()).resolve()


def repository_asset_root() -> Path:
    """Return the canonical repository-relative asset directory."""

    return _DEFAULT_ASSET_ROOT.resolve()


def is_asset_reference(reference: str | Path) -> bool:
    """Return whether *reference* uses the canonical or legacy asset form."""

    raw = str(reference).strip().replace("\\", "/")
    return raw == _REPO_ASSET_PREFIX or raw.startswith(f"{_REPO_ASSET_PREFIX}/") or raw.startswith(
        _LEGACY_ASSET_SCHEME
    )


def _repository_asset_relative(reference: str) -> Path:
    normalized = str(reference).strip().replace("\\", "/")
    if normalized == _REPO_ASSET_PREFIX:
        return Path()
    prefix = f"{_REPO_ASSET_PREFIX}/"
    relative = normalized[len(prefix) :] if normalized.startswith(prefix) else ""
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"invalid repository asset path: {reference!r}")
    return candidate


def _legacy_asset_relative(reference: str) -> Path:
    parsed = urlparse(reference)
    relative = "/".join(part for part in (parsed.netloc, parsed.path.lstrip("/")) if part)
    relative = unquote(relative).replace("\\", "/")
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"invalid asset URI: {reference!r}")
    return candidate


def _inside(root: Path, relative: Path) -> Path:
    """Join an asset-relative path while preserving snapshot symlinks.

    Hugging Face cache snapshots expose files as symlinks into a sibling
    ``blobs/`` directory. Resolving the final path would both reject that
    layout and make USD-relative dependencies resolve beside the blob instead
    of beside the snapshot entry.
    """

    root = root.resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"asset path escapes root: {relative}")
    candidate = root / relative
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"asset path escapes root: {relative}") from exc
    return candidate


def resolve_asset_path(
    reference: str | Path,
    *,
    root: str | Path | None = None,
    preferred_root: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve ``assets/...``, legacy ``asset://``, or an explicit local path.

    Canonical ``assets/...`` and legacy ``asset://`` references are relative
    to ``MANAENV_ASSET_ROOT`` when it is configured, otherwise to the
    repository's ``assets/`` directory. An explicit *root* overrides both.
    Network schemes are rejected.
    """

    raw = str(reference).strip()
    bundle = asset_root(root)
    if not raw:
        raise AssetResolutionError(raw, bundle, "empty asset reference")
    if raw == _REPO_ASSET_PREFIX or raw.replace("\\", "/").startswith(f"{_REPO_ASSET_PREFIX}/"):
        path = _inside(bundle, _repository_asset_relative(raw))
    elif raw.startswith(_LEGACY_ASSET_SCHEME):
        path = _inside(bundle, _legacy_asset_relative(raw))
    elif raw.startswith("file://"):
        path = Path(unquote(urlparse(raw).path)).expanduser().resolve()
    elif "://" in raw:
        raise AssetResolutionError(raw, bundle, "remote asset schemes are disabled")
    else:
        source = Path(raw).expanduser()
        if source.is_absolute():
            path = source.resolve()
        else:
            bases = [Path(preferred_root).expanduser().resolve()] if preferred_root else []
            bases.extend((_REPO_ROOT, bundle))
            existing = next(((base / source).resolve() for base in bases if (base / source).exists()), None)
            path = existing or (bases[0] / source).resolve()
    if must_exist and not path.is_file():
        raise AssetResolutionError(raw, path)
    return path


def resolve_asset_directory(
    reference: str | Path,
    *,
    root: str | Path | None = None,
    preferred_root: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    path = resolve_asset_path(
        reference,
        root=root,
        preferred_root=preferred_root,
        must_exist=False,
    )
    if must_exist and not path.is_dir():
        raise AssetResolutionError(str(reference), path, "asset directory is missing")
    return path


def resolve_asset_reference(reference: str | Path, **kwargs: Any) -> str:
    return str(resolve_asset_path(reference, **kwargs))


def resolve_runtime_asset_reference(reference: str | Path, *, must_exist: bool = False) -> str:
    return str(resolve_asset_path(reference, must_exist=must_exist))


def iter_asset_files(
    reference: str | Path = "assets",
    *,
    suffixes: Iterable[str] | None = None,
    root: str | Path | None = None,
) -> list[Path]:
    directory = asset_root(root) if str(reference).strip() == _LEGACY_ASSET_SCHEME else resolve_asset_directory(
        reference,
        root=root,
        must_exist=False,
    )
    if not directory.is_dir():
        return []
    wanted = {str(suffix).lower() for suffix in suffixes or ()}
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and (not wanted or path.suffix.lower() in wanted)
    )


def read_asset_text(reference: str | Path, **kwargs: Any) -> str:
    return resolve_asset_path(reference, **kwargs).read_text(encoding="utf-8")


def load_asset_json(reference: str | Path, **kwargs: Any) -> dict[str, Any]:
    value = json.loads(read_asset_text(reference, **kwargs))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON asset must contain an object: {reference}")
    return dict(value)


__all__ = [
    "AssetResolutionError",
    "asset_root",
    "default_asset_root",
    "iter_asset_files",
    "is_asset_reference",
    "load_asset_json",
    "read_asset_text",
    "repository_asset_root",
    "resolve_asset_directory",
    "resolve_asset_path",
    "resolve_asset_reference",
    "resolve_runtime_asset_reference",
]
