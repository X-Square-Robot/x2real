from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ...common.ids import DomainId, NodeId

ScopeDomainKind = Literal["node", "region_group", "scatter_group"]
InteractionReason = Literal[
    "range_overlap",
    "relative_direction",
    "explicit_collision_risk",
    "shared_group",
]


@dataclass(frozen=True)
class ScopeDomainRef:
    domain_id: DomainId
    kind: ScopeDomainKind
    surface_id: str
    node_ids: tuple[NodeId, ...]
    dependency_nodes: tuple[NodeId, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeObstacleRef:
    obstacle_id: str
    source_domain_id: DomainId | None
    surface_id: str
    node_ids: tuple[NodeId, ...]
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeInteractionEdge:
    left: DomainId
    right: DomainId
    reason: InteractionReason
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportSurfaceComponent:
    component_id: str
    surface_id: str
    domain_refs: tuple[ScopeDomainRef, ...]
    node_ids: tuple[NodeId, ...]
    dependency_nodes: tuple[NodeId, ...]
    interaction_edges: tuple[ScopeInteractionEdge, ...] = ()
    obstacles: tuple[ScopeObstacleRef, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportSurfaceScope:
    scope_id: str
    surface_id: str
    components: tuple[SupportSurfaceComponent, ...]
    domain_refs: tuple[ScopeDomainRef, ...]
    node_ids: tuple[NodeId, ...]
    dependency_nodes: tuple[NodeId, ...]
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportSurfaceScopePlan:
    scopes: tuple[SupportSurfaceScope, ...] = ()
    domain_to_component: dict[DomainId, str] = field(default_factory=dict)
    transactional_components: tuple[str, ...] = ()
    transaction_domains: frozenset[DomainId] = frozenset()
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def transactional_component_set(self) -> set[str]:
        return set(self.transactional_components)

    @property
    def component_domain_aliases(self) -> dict[DomainId, DomainId]:
        out: dict[DomainId, DomainId] = {}
        transactional = self.transactional_component_set
        for domain_id, component_id in self.domain_to_component.items():
            if component_id in transactional:
                out[domain_id] = DomainId("support_component", component_id)
        return out
