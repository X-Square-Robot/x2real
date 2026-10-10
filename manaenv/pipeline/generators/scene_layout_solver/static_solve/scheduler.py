from __future__ import annotations

from collections.abc import Callable

from ..common.ids import DomainId
from .ir import GroupSpec, NodeSpec
from .scopes.model import SupportSurfaceScopePlan


def schedule_execution_sequence(
    *,
    node_specs: dict[str, NodeSpec],
    group_specs: dict[str, GroupSpec],
    grouped_objects: set[str],
    object_domain_ids: dict[str, DomainId],
    ordering_edges: list[tuple[str, str, str]],
    group_surface: Callable[[str], str | None],
    local_surface_parent: Callable[[str], str | None],
    node_dependency_nodes: Callable[[str], list[str]] | None = None,
    group_dependency_nodes: Callable[[str], list[str]] | None = None,
    support_scope_plan: SupportSurfaceScopePlan | None = None,
) -> list[DomainId]:
    """Build an explicit domain scheduling policy for the static solver."""

    # Solve constrained group domains before free support-on nodes. Explicit
    # dependency edges below still move parent/local-surface nodes ahead of any
    # dependent group. Same-surface free nodes that are not explicit parents are
    # intentionally not treated as topo dependencies here; that belongs to a
    # support-surface joint scope, not an implicit ordering shortcut.
    free_nodes = [
        node_id
        for node_id in _ordered_free_nodes(node_specs, grouped_objects, ordering_edges)
        if node_id not in grouped_objects and node_specs[node_id].domain_providers
    ]
    base_domains = [DomainId.group(group_id) for group_id in sorted(group_specs)]
    base_domains.extend(DomainId.node(node_id) for node_id in free_nodes)
    domains = _replace_transactional_components(base_domains, support_scope_plan)
    aliases = support_scope_plan.component_domain_aliases if support_scope_plan is not None else {}
    deps = _domain_dependencies(
        domains,
        node_specs=node_specs,
        object_domain_ids=object_domain_ids,
        group_surface=group_surface,
        local_surface_parent=local_surface_parent,
        node_dependency_nodes=node_dependency_nodes,
        group_dependency_nodes=group_dependency_nodes,
        domain_aliases=aliases,
    )
    if support_scope_plan is not None:
        _add_support_component_dependencies(
            deps,
            domains=domains,
            object_domain_ids=object_domain_ids,
            support_scope_plan=support_scope_plan,
            domain_aliases=aliases,
        )
    return _toposort_domains(domains, deps)


def _replace_transactional_components(
    base_domains: list[DomainId],
    support_scope_plan: SupportSurfaceScopePlan | None,
) -> list[DomainId]:
    if support_scope_plan is None or not support_scope_plan.transactional_components:
        return base_domains
    transactional = set(support_scope_plan.transactional_components)
    emitted: set[str] = set()
    out: list[DomainId] = []
    for domain in base_domains:
        component_id = support_scope_plan.domain_to_component.get(domain)
        if component_id in transactional:
            if component_id not in emitted:
                out.append(DomainId("support_component", component_id))
                emitted.add(component_id)
            continue
        out.append(domain)
    return out


def _ordered_free_nodes(
    node_specs: dict[str, NodeSpec],
    grouped_objects: set[str],
    ordering_edges: list[tuple[str, str, str]],
) -> list[str]:
    free = sorted(name for name in node_specs if name not in grouped_objects)
    if not ordering_edges:
        return free
    free_set = set(free)
    indeg = {name: 0 for name in free}
    adj = {name: [] for name in free}
    for pre, post, _axis in ordering_edges:
        if pre not in free_set or post not in free_set:
            continue
        adj[pre].append(post)
        indeg[post] += 1
    queue = sorted(name for name in free if indeg[name] == 0)
    ordered: list[str] = []
    while queue:
        cur = queue.pop(0)
        ordered.append(cur)
        for nxt in sorted(adj[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
        queue.sort()
    return ordered if len(ordered) == len(free) else free


def _domain_dependencies(
    domains: list[DomainId],
    *,
    node_specs: dict[str, NodeSpec],
    object_domain_ids: dict[str, DomainId],
    group_surface: Callable[[str], str | None],
    local_surface_parent: Callable[[str], str | None],
    node_dependency_nodes: Callable[[str], list[str]] | None,
    group_dependency_nodes: Callable[[str], list[str]] | None,
    domain_aliases: dict[DomainId, DomainId] | None = None,
) -> dict[DomainId, set[DomainId]]:
    domain_set = set(domains)
    aliases = dict(domain_aliases or {})
    deps: dict[DomainId, set[DomainId]] = {domain: set() for domain in domains}
    for domain in domains:
        for surface in _domain_surfaces(domain, node_specs=node_specs, group_surface=group_surface):
            parent = local_surface_parent(surface)
            if not parent:
                continue
            owner = object_domain_ids.get(parent)
            if owner is None or owner == domain:
                continue
            scheduled_owner = aliases.get(owner, owner)
            if scheduled_owner in domain_set and scheduled_owner != domain:
                deps[domain].add(scheduled_owner)
        dependency_nodes: list[str] = []
        if domain.kind == "node" and node_dependency_nodes is not None:
            dependency_nodes = node_dependency_nodes(domain.id)
        elif domain.kind == "group" and group_dependency_nodes is not None:
            dependency_nodes = group_dependency_nodes(domain.id)
        for parent in dependency_nodes:
            owner = object_domain_ids.get(parent)
            if owner is None or owner == domain:
                continue
            scheduled_owner = aliases.get(owner, owner)
            if scheduled_owner in domain_set and scheduled_owner != domain:
                deps[domain].add(scheduled_owner)
    return deps


def _add_support_component_dependencies(
    deps: dict[DomainId, set[DomainId]],
    *,
    domains: list[DomainId],
    object_domain_ids: dict[str, DomainId],
    support_scope_plan: SupportSurfaceScopePlan,
    domain_aliases: dict[DomainId, DomainId],
) -> None:
    domain_set = set(domains)
    components = {
        component.component_id: component for scope in support_scope_plan.scopes for component in scope.components
    }
    for domain in domains:
        if domain.kind != "support_component":
            continue
        component = components.get(domain.id)
        if component is None:
            continue
        for parent in component.dependency_nodes:
            owner = object_domain_ids.get(parent)
            if owner is None:
                continue
            scheduled_owner = domain_aliases.get(owner, owner)
            if scheduled_owner in domain_set and scheduled_owner != domain:
                deps.setdefault(domain, set()).add(scheduled_owner)


def _domain_surfaces(
    domain: DomainId,
    *,
    node_specs: dict[str, NodeSpec],
    group_surface: Callable[[str], str | None],
) -> list[str]:
    if domain.kind == "node":
        node = node_specs.get(domain.id)
        return [node.support] if node and node.support else []
    if domain.kind == "group":
        surface = group_surface(domain.id)
        return [surface] if surface else []
    return []


def _toposort_domains(domains: list[DomainId], deps: dict[DomainId, set[DomainId]]) -> list[DomainId]:
    rank = {domain: idx for idx, domain in enumerate(domains)}
    remaining = list(domains)
    out: list[DomainId] = []
    emitted: set[DomainId] = set()
    while remaining:
        ready = [domain for domain in remaining if deps.get(domain, set()).issubset(emitted)]
        if not ready:
            return domains
        ready.sort(key=lambda domain: rank[domain])
        chosen = ready[0]
        out.append(chosen)
        emitted.add(chosen)
        remaining.remove(chosen)
    return out
