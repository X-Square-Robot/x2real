"""Inline ``@role.<attr>`` interpolation for human-readable task text fields.

This is the *text* projection of a ``@role`` reference and is deliberately
distinct from the param-binding path in ``component._resolve_value`` (where a
whole-string ``@role`` projects to the asset ``.name``). Inside prose fields
(instruction / topic / subtasks, ``definition.prompt``, log ``value``) a
reference must carry an explicit attribute chain, e.g. ``@drink.name`` or
``@drink.initial_pose.position_xyz``. The chain is resolved by repeated
``getattr`` against the bound role value and stringified; a role bound to a
list of assets (a group ref) projects each element and joins with ``", "``.

Semantics:

* ``@role.<attr>[.<attr>...]`` — recognised reference. ``role`` is looked up in
  the roles mapping; the attribute chain is walked with ``getattr`` and the
  result is ``str``-ified. Unknown ``role`` (not in the mapping) is left
  verbatim so stray ``@`` text / emails / version tokens survive untouched.
* bare ``@role`` (no attribute) — if ``role`` is a known role this is an
  authoring error (the reference is an object and cannot render as text);
  otherwise it is left verbatim.

Pure ``re`` only: no IsaacLab / registry imports, so the stability suite can
exercise it directly.
"""

from __future__ import annotations

import re
from typing import Any

# role: leading identifier; attrs: zero or more ``.identifier`` segments. Scoped
# role keys carry ``__node_<id>`` suffixes which are plain word chars, so the
# ``\w`` class already covers them. Attribute segments must start with a letter
# or underscore so numeric tokens (``@v1.0.0``) are not mistaken for refs.
TEXT_REF_PATTERN = re.compile(r"@(?P<role>[A-Za-z_]\w*)(?P<attrs>(?:\.[A-Za-z_]\w*)+)?")


def collect_text_refs(value: Any) -> set[str]:
    """Collect role names from inline ``@role.<attr>`` references in *value*.

    Only attribute-qualified tokens are collected: those are the references
    that need a scene-bound asset object at finalize. Bare ``@role`` tokens are
    ignored here (they either name an already-declared role or are stray text).
    Recurses through dict / list / tuple containers AND plain objects' ``__dict__``
    — the latter so a section already reflected into a component CFG dataclass
    (e.g. graph sub-generator nodes) is still scanned, not just raw YAML dicts.
    """
    refs: set[str] = set()
    seen: set[int] = set()

    def _walk(obj: Any) -> None:
        if isinstance(obj, str):
            for match in TEXT_REF_PATTERN.finditer(obj):
                if match.group("attrs"):
                    refs.add(match.group("role"))
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(obj, dict):
            for item in obj.values():
                _walk(item)
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                _walk(item)
            return
        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            try:
                attrs = vars(obj)
            except TypeError:
                return
            for item in attrs.values():
                _walk(item)

    _walk(value)
    return refs


def _walk_attrs(value: Any, attrs: list[str], token: str) -> Any:
    """Walk an attribute chain via repeated ``getattr`` on a single object."""
    current = value
    for attr in attrs:
        if not hasattr(current, attr):
            raise AttributeError(
                f"text ref {token!r}: {type(current).__name__} has no attribute {attr!r}; "
                "check the attribute name (e.g. use '.name' for the asset name)."
            )
        current = getattr(current, attr)
    return current


def render_ref_text(text: Any, roles: dict[str, Any]) -> Any:
    """Render inline ``@role.<attr>`` references in *text* against *roles*.

    Non-string input and strings without ``@`` are returned unchanged. Each
    recognised reference is resolved by walking its attribute chain against the
    bound role value and stringifying the result. When the *role* itself is
    bound to a list of assets (a group ref), the chain is applied to each
    element and the results join with ``", "`` — a scalar attribute that itself
    returns a tuple/list (e.g. ``position_xyz``) is stringified as-is. A bare
    ``@role`` naming a known role raises ``ValueError``; unknown roles are left
    verbatim.
    """
    if not isinstance(text, str) or "@" not in text:
        return text

    def _sub(match: re.Match[str]) -> str:
        role = match.group("role")
        if role not in roles:
            return match.group(0)
        attrs_raw = match.group("attrs") or ""
        attrs = [seg for seg in attrs_raw.split(".") if seg]
        if not attrs:
            raise ValueError(
                f"text ref '@{role}' needs an explicit attribute (e.g. '@{role}.name'); "
                "a bare '@role' references the asset object, which cannot render as text."
            )
        bound = roles[role]
        token = match.group(0)
        if isinstance(bound, (list, tuple)):
            return ", ".join(str(_walk_attrs(element, attrs, token)) for element in bound)
        return str(_walk_attrs(bound, attrs, token))

    return TEXT_REF_PATTERN.sub(_sub, text)


def render_refs_in(value: Any, roles: dict[str, Any]) -> Any:
    """Recursively render ``@role.<attr>`` refs in str / list / dict *value*."""
    if isinstance(value, str):
        return render_ref_text(value, roles)
    if isinstance(value, list):
        return [render_refs_in(item, roles) for item in value]
    if isinstance(value, dict):
        return {key: render_refs_in(item, roles) for key, item in value.items()}
    return value
