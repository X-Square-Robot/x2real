from __future__ import annotations

from collections.abc import Callable

from ...common.ids import DomainId
from ..domains.placement_providers import SupportOnDomainProvider
from ..ir import GroupSpec, NodeSpec
from ..problem import LayoutProblem
from .geometry import domain_geometry_for_ref, refs_may_interact, relation_edges_for_refs
from .model import (
    ScopeDomainRef,
    ScopeInteractionEdge,
    SupportSurfaceComponent,
    SupportSurfaceScope,
    SupportSurfaceScopePlan,
)


class SupportSurfaceScopePlanner:
    def __init__(self, problem: LayoutProblem) -> None:
        self.problem = problem

    def plan(
        self,
        *,
        node_specs: dict[str, NodeSpec],
        group_specs: dict[str, GroupSpec],
        grouped_objects: set[str],
        object_domain_ids: dict[str, DomainId],
        group_surface: Callable[[str], str | None],
        local_surface_parent: Callable[[str], str | None],
    ) -> SupportSurfaceScopePlan:
        refs_by_surface: dict[str, list[ScopeDomainRef]] = {}
        for ref in self.collect_domain_refs(
            node_specs=node_specs,
            group_specs=group_specs,
            grouped_objects=grouped_objects,
            object_domain_ids=object_domain_ids,
            group_surface=group_surface,
            local_surface_parent=local_surface_parent,
        ):
            refs_by_surface.setdefault(ref.surface_id, []).append(ref)

        scopes: list[SupportSurfaceScope] = []
        domain_to_component: dict[DomainId, str] = {}
        transactional_components: list[str] = []
        transaction_domains: set[DomainId] = set()
        for surface_id in sorted(refs_by_surface):
            refs = sorted(refs_by_surface[surface_id], key=lambda ref: (ref.domain_id.kind, ref.domain_id.id))
            components = self.build_surface_components(
                surface_id,
                refs,
                node_specs=node_specs,
                group_specs=group_specs,
            )
            for component in components:
                for ref in component.domain_refs:
                    domain_to_component[ref.domain_id] = component.component_id
                if len(component.domain_refs) > 1 and _has_transactional_edge(component):
                    transactional_components.append(component.component_id)
                    transaction_domains.update(ref.domain_id for ref in component.domain_refs)
            scopes.append(
                SupportSurfaceScope(
                    scope_id=f"surface:{surface_id}",
                    surface_id=surface_id,
                    components=tuple(components),
                    domain_refs=tuple(refs),
                    node_ids=tuple(sorted({node_id for ref in refs for node_id in ref.node_ids})),
                    dependency_nodes=tuple(sorted({dep for ref in refs for dep in ref.dependency_nodes})),
                    evidence={"domain_ref_count": len(refs)},
                )
            )
        return SupportSurfaceScopePlan(
            scopes=tuple(scopes),
            domain_to_component=domain_to_component,
            transactional_components=tuple(transactional_components),
            transaction_domains=frozenset(transaction_domains),
            evidence={"surface_count": len(scopes), "transactional_component_count": len(transactional_components)},
        )

    def collect_domain_refs(
        self,
        *,
        node_specs: dict[str, NodeSpec],
        group_specs: dict[str, GroupSpec],
        grouped_objects: set[str],
        object_domain_ids: dict[str, DomainId],
        group_surface: Callable[[str], str | None],
        local_surface_parent: Callable[[str], str | None],
    ) -> list[ScopeDomainRef]:
        refs: list[ScopeDomainRef] = []
        for group_id, group in sorted(group_specs.items()):
            surface_id = group_surface(group_id)
            if not surface_id or group_id.startswith("volume:"):
                continue
            kind = "scatter_group" if group_id.startswith("scatter:") else "region_group"
            parent = local_surface_parent(surface_id)
            refs.append(
                ScopeDomainRef(
                    domain_id=DomainId.group(group_id),
                    kind=kind,
                    surface_id=surface_id,
                    node_ids=tuple(group.members),
                    dependency_nodes=(parent,) if parent else (),
                    evidence={"source": "group"},
                )
            )
        for node_id, node in sorted(node_specs.items()):
            if node_id in grouped_objects or not node.support:
                continue
            provider = node.domain_providers[0] if node.domain_providers else None
            if not isinstance(provider, SupportOnDomainProvider):
                continue
            domain_id = object_domain_ids.get(node_id, DomainId.node(node_id))
            parent = local_surface_parent(node.support)
            refs.append(
                ScopeDomainRef(
                    domain_id=domain_id,
                    kind="node",
                    surface_id=node.support,
                    node_ids=(node_id,),
                    dependency_nodes=(parent,) if parent else (),
                    evidence={"source": "support_on_node"},
                )
            )
        return refs

    def build_surface_components(
        self,
        surface_id: str,
        refs: list[ScopeDomainRef],
        *,
        node_specs: dict[str, NodeSpec],
        group_specs: dict[str, GroupSpec],
    ) -> list[SupportSurfaceComponent]:
        if not refs:
            return []
        geometries = {
            ref.domain_id: domain_geometry_for_ref(
                problem=self.problem,
                ref=ref,
                node_specs=node_specs,
                group_specs=group_specs,
            )
            for ref in refs
        }
        edges: list[ScopeInteractionEdge] = []
        for idx, left in enumerate(refs):
            for right in refs[idx + 1 :]:
                edge = refs_may_interact(geometries[left.domain_id], geometries[right.domain_id])
                if edge is not None:
                    edges.append(edge)
        edges.extend(relation_edges_for_refs(self.problem, refs, surface_id=surface_id))
        return _connected_components(surface_id, refs, edges)


def _connected_components(
    surface_id: str,
    refs: list[ScopeDomainRef],
    edges: list[ScopeInteractionEdge],
) -> list[SupportSurfaceComponent]:
    adjacency: dict[DomainId, set[DomainId]] = {ref.domain_id: set() for ref in refs}
    edge_by_pair: dict[tuple[DomainId, DomainId], list[ScopeInteractionEdge]] = {}
    for edge in edges:
        adjacency.setdefault(edge.left, set()).add(edge.right)
        adjacency.setdefault(edge.right, set()).add(edge.left)
        key = _pair_key(edge.left, edge.right)
        edge_by_pair.setdefault(key, []).append(edge)
    refs_by_domain = {ref.domain_id: ref for ref in refs}
    seen: set[DomainId] = set()
    components: list[SupportSurfaceComponent] = []
    for ref in refs:
        if ref.domain_id in seen:
            continue
        stack = [ref.domain_id]
        domains: list[DomainId] = []
        seen.add(ref.domain_id)
        while stack:
            cur = stack.pop()
            domains.append(cur)
            for nxt in sorted(adjacency.get(cur, set()), key=lambda item: (item.kind, item.id)):
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        component_refs = tuple(
            refs_by_domain[domain] for domain in sorted(domains, key=lambda item: (item.kind, item.id))
        )
        component_edges: list[ScopeInteractionEdge] = []
        for idx, left in enumerate(component_refs):
            for right in component_refs[idx + 1 :]:
                component_edges.extend(edge_by_pair.get(_pair_key(left.domain_id, right.domain_id), []))
        component_id = f"surface:{surface_id}:component:{len(components):03d}"
        components.append(
            SupportSurfaceComponent(
                component_id=component_id,
                surface_id=surface_id,
                domain_refs=component_refs,
                node_ids=tuple(sorted({node_id for item in component_refs for node_id in item.node_ids})),
                dependency_nodes=tuple(sorted({dep for item in component_refs for dep in item.dependency_nodes})),
                interaction_edges=tuple(component_edges),
                evidence={"domain_ref_count": len(component_refs)},
            )
        )
    return components


def _pair_key(left: DomainId, right: DomainId) -> tuple[DomainId, DomainId]:
    left_key = (left.kind, left.id)
    right_key = (right.kind, right.id)
    return (left, right) if left_key <= right_key else (right, left)


def _has_transactional_edge(component: SupportSurfaceComponent) -> bool:
    return any(edge.reason in {"range_overlap", "explicit_collision_risk"} for edge in component.interaction_edges)
