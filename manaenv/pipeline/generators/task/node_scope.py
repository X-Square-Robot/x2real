"""Per-node role scoping for multi-node task graphs.

Before N atomic-task nodes merge into one flat task, each node's declared
role keys and the matching ``@role`` references inside its component
sections must be suffixed by node id, so two nodes both naming ``drink``
resolve to independent role slots (the right scene asset per phase).

This rewrites raw YAML-shaped payloads at compile time, BEFORE component
sections are compiled — orthogonal to the configclass-field namespacing in
``namespace_merge`` (which scopes term *field* names on built cfg instances).
Both are needed: roles here, term fields there.

Scoping boundary — only the node's **declared** ``roles:`` keys are scoped.
Referenced-but-undeclared roles (and dynamic roles published at compile time
by ``_roles_update``, e.g. a ``scene_constraint`` sensor whose
``output_role`` is a plain string *value*, not a ``@ref``) are left in the
shared namespace. This avoids desyncing a ``@ref`` from its role-defining
string value: we only rewrite a ``@ref`` when its role name is in the scoped
(declared) set, so key and references always move together.

Pure Python, stability-testable.
"""

from __future__ import annotations

from typing import Any

from manaenv.pipeline.generators.task.namespace_merge import node_suffix
from manaenv.pipeline.generators.task.ref_text import TEXT_REF_PATTERN


def scope_role_key(role_key: str, node_id: str) -> str:
    """Suffix a single role key by node id."""
    return f"{role_key}{node_suffix(node_id)}"


def scope_node_roles(roles: dict[str, Any], node_id: str) -> dict[str, Any]:
    """Suffix every declared role KEY by node id (values — scene tags — unchanged)."""
    return {scope_role_key(key, node_id): value for key, value in (roles or {}).items()}


def rewrite_role_references(value: Any, node_id: str, scoped_roles: set[str]) -> Any:
    """Rewrite ``@role`` -> ``@role__node_{id}`` for roles in *scoped_roles*.

    Recurses through dicts/lists/tuples. Both whole-string param refs
    (``@drink``) and inline text refs with an attribute chain
    (``@drink.name`` inside a log/instruction string) are rewritten, keeping
    the attribute chain intact (``@drink__node_{id}.name``). Only tokens whose
    role name is in *scoped_roles* (the node's declared slots) are touched, so
    shared/dynamic roles and stray ``@`` text are left alone. The input is not
    mutated (rewritten containers are rebuilt; immutable leaves shared). Raw
    YAML payloads are acyclic, so no cycle guard is needed.
    """
    suffix = node_suffix(node_id)

    def _rewrite_string(text: str) -> str:
        def _sub(match: Any) -> str:
            role = match.group("role")
            if role not in scoped_roles:
                return match.group(0)
            attrs = match.group("attrs") or ""
            return f"@{role}{suffix}{attrs}"

        return TEXT_REF_PATTERN.sub(_sub, text)

    def _walk(obj: Any) -> Any:
        if isinstance(obj, str):
            return _rewrite_string(obj)
        if isinstance(obj, dict):
            return {key: _walk(item) for key, item in obj.items()}
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(item) for item in obj)
        return obj

    return _walk(value)


def scope_node(
    roles: dict[str, Any],
    component_sections: dict[str, Any],
    node_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Scope one node's declared roles + its sections' references by node id.

    Returns ``(scoped_roles, scoped_sections)``. The scoped role-key set is
    taken from ``roles`` (the node's declared slots); only ``@ref``\\ s naming
    those slots are rewritten, keeping declared roles fully isolated per node
    while shared/dynamic roles stay in the common namespace.
    """
    declared = set((roles or {}).keys())
    scoped_roles = scope_node_roles(roles, node_id)
    scoped_sections = rewrite_role_references(component_sections or {}, node_id, declared)
    return scoped_roles, scoped_sections
