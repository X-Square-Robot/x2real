"""Avoid RTX texture resource collisions for reproducible archives with epoch mtimes.

Some Kit/RTX versions render distinct epoch-dated images as the same texture.
Keep archive files intact; author content-addressed, byte-identical texture copies
in the session layer before the first environment reset.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import tempfile

_LOG = logging.getLogger(__name__)
_IMAGE_SUFFIXES = frozenset({'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr', '.hdr', '.bmp', '.tga', '.dds'})


def cache_epoch_texture(source: Path, cache_dir: Path) -> Path:
    """Return a verified content-addressed copy, without preserving epoch metadata."""
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (digest + source.suffix.lower())
    if target.is_file() and target.stat().st_mtime_ns > 0:
        if hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            return target
    fd, temporary = tempfile.mkstemp(prefix='.texture-', dir=cache_dir)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        # Do not use copy2: preserving mtime=0 reproduces the renderer failure.
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def repair_epoch_texture_resources(stage, cache_dir: Path | None = None) -> dict[str, int]:
    """Redirect epoch-dated image assets in a USD stage; leave UVs/materials intact."""
    from pxr import Sdf, Usd

    if cache_dir is None:
        explicit = os.environ.get('MANAENV_TEXTURE_CACHE_DIR')
        cache_dir = Path(explicit).expanduser() if explicit else Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'manaenv' / 'textures'
    copied: dict[Path, Path] = {}
    edits = []
    # Shader prims in instance prototypes cannot be edited. Normal material prims
    # (including shared materials outside instances) are traversed here.
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if not value or not value.resolvedPath:
                continue
            source = Path(value.resolvedPath)
            if source.suffix.lower() not in _IMAGE_SUFFIXES or not source.is_file():
                continue
            if source.stat().st_mtime_ns > 0:
                continue
            if source not in copied:
                copied[source] = cache_epoch_texture(source, cache_dir)
            edits.append((attr, copied[source]))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for attr, target in edits:
            attr.Set(Sdf.AssetPath(str(target.resolve())))
    summary = {'textures': len(copied), 'attributes': len(edits)}
    if edits:
        _LOG.info('RTX epoch-texture workaround: %d image files, %d references; cache=%s', len(copied), len(edits), cache_dir)
    return summary


def repair_active_stage_epoch_textures() -> None:
    """Run only when a Kit USD context is available."""
    import sys

    if 'omni.usd' not in sys.modules:
        return
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is not None:
        repair_epoch_texture_resources(stage)
