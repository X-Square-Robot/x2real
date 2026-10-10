from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

_TAG_HIERARCHY_SEP = "|"


def resolve_manaenv_asset_ids_by_tag(tag: str) -> list[str]:
    """Resolve a ManaEnv asset tag into concrete asset ids.

    The AssetRegistry remains the source of truth. This helper only adapts its
    asset-class return value into the string ids consumed by the vNext asset
    sampler.

    Unlike ``AssetRegistry.get_assets_by_tag`` (exact set-membership match), this
    resolver matches the ``|``-delimited tag HIERARCHY: a requested tag ``T``
    hits an asset that carries either the exact tag ``T`` or any descendant
    ``T|<leaf>``. Local assets are tagged with their full leaf path (e.g.
    ``food|fruit_vegetable|apple``), so a family/branch request like
    ``food|fruit_vegetable`` resolves the whole fruit set instead of only the
    handful tagged at that exact level. A comma-separated ``tag`` is treated as
    an AND of independently-hierarchical requirements (e.g.
    ``food|fruit_vegetable,graspable``).
    """

    _ensure_local_assets_registered()
    from isaaclab_arena.assets.registries import AssetRegistry

    required = [part.strip() for part in str(tag).split(",") if part.strip()]
    if not required:
        return []

    registry = AssetRegistry()
    # Passing an empty tag makes the required-set empty, so every tagged asset
    # is returned; we then apply hierarchical matching ourselves.
    candidates = registry.get_assets_by_tag("")

    matched: set[str] = set()
    for asset in candidates:
        asset_tags = getattr(asset, "tags", None) or ()
        if all(_tag_hierarchy_hit(req, asset_tags) for req in required):
            name = str(getattr(asset, "name", "")).strip()
            if name:
                matched.add(name)
    return sorted(matched)


def _tag_hierarchy_hit(required_tag: str, asset_tags) -> bool:
    """True if any asset tag equals ``required_tag`` or descends from it.

    Descent uses the ``|`` hierarchy separator, so ``food|fruit_vegetable``
    matches ``food|fruit_vegetable`` and ``food|fruit_vegetable|apple`` but not
    ``food|fruit_orange``.
    """

    prefix = required_tag + _TAG_HIERARCHY_SEP
    return any(str(existing) == required_tag or str(existing).startswith(prefix) for existing in asset_tags)


def _ensure_local_assets_registered() -> None:
    """Populate the registry with locally discovered assets before tag lookup.

    vNext tag resolution runs at YAML-translation time, which is earlier than the
    pipeline warmup step. Without this, the registry would only hold the
    statically imported library objects and a local tag (e.g.
    ``food|fruit_vegetable``) could resolve to an empty set.
    """

    try:
        from manaenv.assets.local_discovery import discover_and_register_local_assets

        discover_and_register_local_assets()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Local asset discovery unavailable during tag resolution: %s", exc)
