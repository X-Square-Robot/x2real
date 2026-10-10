"""Structured semantic tags (Phase 1) for scene assets.

Canonical YAML form — block-style list, one entry per line. Every
entry is one of:

* a bare scalar (no attribute payload)::

    semantic_tags:
      - pickable
      - graspable

* a single-key mapping carrying a structured attribute payload that a
  later phase will materialize into a runtime type (``Openable`` /
  ``Pressable``)::

    semantic_tags:
      - target_door
      - Openable:
          openable_joint_name: door_joint_a
          openable_threshold: 0.6

Both shapes may appear in the same list. Flow-style ``[a, b]`` and
quoted scalars (``"pickable"``) are tolerated by the YAML parser but
are not the canonical form; write block-style with bare scalars
instead. When an asset has no semantic tags, omit the field rather
than writing ``semantic_tags: []``.

This module normalizes the parsed Python value (a ``list`` of
strings and/or single-key mappings) into :class:`SemanticTagCFG` and
applies them onto an asset by:

* appending each declared name to ``asset.tags`` (the dynamic merged
  view that already includes USD and class-static tags);
* writing the same names to ``asset.semantic_tags`` (the canonical
  semantic-tag list that downstream readers query);
    * stashing structured declarations under ``asset.semantic_tag_attrs``
      for later runtime-type materialization.

:class:`SemanticTagCFG` carries the ``CFG`` suffix to follow the
project rule for YAML-parsed structures; it is a data record, not a
factory CFG (no ``class_type``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SemanticTagCFG:
    """Normalized form of one ``semantic_tags`` entry.

    Attributes:
        name: Tag name. Appended to ``asset.tags`` and
            ``asset.semantic_tags``.
        attrs: Optional attribute payload retained for later runtime
            type materialization. Empty when the entry was a bare
            string.
        structured: Whether the source yaml entry used mapping syntax.
            This lets ``Openable: {}`` trigger runtime validation while
            keeping bare scalar ``Openable`` a plain metadata tag.
    """

    name: str
    attrs: Mapping[str, Any] = field(default_factory=dict)
    structured: bool = False


def normalize_semantic_tags(raw: Any) -> list[SemanticTagCFG]:
    """Parse a YAML ``semantic_tags`` value into typed entries.

    Args:
        raw: Yaml-loaded value (a ``list``, or ``None`` for the
            absent / empty case). Each entry is either a bare scalar
            (becomes a :class:`SemanticTagCFG` with empty ``attrs``)
            or a single-key mapping (becomes a
            :class:`SemanticTagCFG` carrying the structured payload).

    Returns:
        Ordered list of :class:`SemanticTagCFG`. Empty when ``raw`` is
        ``None`` or an empty sequence.

    Raises:
        ValueError: when an entry shape is not supported (multi-key
            mapping, non-string key, non-mapping attrs, etc.).
    """
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"semantic_tags must be a list, got {type(raw).__name__}: {raw!r}")

    out: list[SemanticTagCFG] = []
    for entry in raw:
        out.append(_normalize_entry(entry))
    return out


def _normalize_entry(entry: Any) -> SemanticTagCFG:
    if isinstance(entry, SemanticTagCFG):
        return entry

    if isinstance(entry, str):
        name = entry.strip()
        if not name:
            raise ValueError("semantic_tags string entry must be non-empty")
        return SemanticTagCFG(name=name, attrs={}, structured=False)

    if isinstance(entry, Mapping):
        if len(entry) != 1:
            raise ValueError(
                "semantic_tags structured entry must be a single-key mapping "
                f"like {{Openable: {{...}}}}, got keys {list(entry)!r}"
            )
        ((raw_name, attrs_raw),) = entry.items()
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"semantic_tags structured entry key must be a non-empty string, got {raw_name!r}")
        if attrs_raw is None:
            attrs: dict[str, Any] = {}
        elif isinstance(attrs_raw, Mapping):
            attrs = dict(attrs_raw)
        else:
            raise ValueError(
                f"semantic_tags entry {raw_name!r} attrs must be a mapping or null, "
                f"got {type(attrs_raw).__name__}: {attrs_raw!r}"
            )
        return SemanticTagCFG(name=raw_name.strip(), attrs=attrs, structured=True)

    raise ValueError(
        f"semantic_tags entry must be a string or single-key mapping, " f"got {type(entry).__name__}: {entry!r}"
    )


def _merge_unique(asset: Any, attr_name: str, names: Iterable[str]) -> None:
    """Append *names* to ``asset.<attr_name>`` deduplicated, preserving order."""
    existing = list(getattr(asset, attr_name, None) or [])
    seen = set(existing)
    for name in names:
        if name not in seen:
            existing.append(name)
            seen.add(name)
    setattr(asset, attr_name, existing)


def merge_semantic_tag_names(asset: Any, tags: Iterable[SemanticTagCFG]) -> None:
    """Append declared names to ``asset.tags`` and ``asset.semantic_tags``.

    ``asset.tags`` is the dynamic merged view (USD + class
    static + yaml). ``asset.semantic_tags`` is the canonical
    semantic-tag list that downstream readers query. Both receive the
    same names, deduplicated, with pre-existing entries kept in their
    original order.
    """
    names = [t.name for t in tags]
    _merge_unique(asset, "tags", names)
    _merge_unique(asset, "semantic_tags", names)


def attach_semantic_tag_attrs(asset: Any, tags: Iterable[SemanticTagCFG]) -> None:
    """Stash structured attribute payloads under ``asset.semantic_tag_attrs``.

    Bare scalar entries are skipped. Structured entries are retained
    even when their attrs are empty so runtime-type validation can
    catch missing required parameters. When the same name appears more
    than once, the later structured occurrence wins.
    The attribute is only assigned back to the asset when there is at
    least one payload to record, so plain-only inputs do not pollute
    the asset namespace.
    """
    existing = getattr(asset, "semantic_tag_attrs", None)
    store: dict[str, dict[str, Any]] = dict(existing) if isinstance(existing, dict) else {}
    for t in tags:
        if t.structured or t.attrs:
            store[t.name] = dict(t.attrs)
    if store:
        asset.semantic_tag_attrs = store


def apply_semantic_tags(asset: Any, semantic_tags: Any = None) -> list[SemanticTagCFG]:
    """One-shot helper used by scene components.

    Normalizes the ``semantic_tags`` YAML value (entries are bare
    scalars and/or single-key mappings), merges declared names into
    ``asset.tags`` + ``asset.semantic_tags`` deduplicated, and stashes
    any structured attribute payload on ``asset.semantic_tag_attrs``.

    Returns:
        The list of normalized :class:`SemanticTagCFG`. Empty when the
        input is empty.
    """
    cfgs = normalize_semantic_tags(semantic_tags)
    if not cfgs:
        return []
    merge_semantic_tag_names(asset, cfgs)
    attach_semantic_tag_attrs(asset, cfgs)
    return cfgs
