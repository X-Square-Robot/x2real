from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..assets.bridge import LayoutAssetRecord
from ..common.diagnostics import LayoutDiagnostic
from ..common.geometry import Bounds3D
from ..common.ids import DomainId, GroupId, NodeId
from .constraints.base import HardFactor
from .domains.base import DomainProvider
from .scopes.model import SupportSurfaceComponent, SupportSurfaceScope


@dataclass
class NodeSpec:
    node_id: NodeId
    asset: LayoutAssetRecord
    domain_providers: list[DomainProvider] = field(default_factory=list)
    hard_factors: list[HardFactor] = field(default_factory=list)
    support: str | None = None
    local_bounds: Bounds3D | None = None


@dataclass
class GroupSpec:
    group_id: GroupId
    members: list[NodeId]
    domain_providers: list[DomainProvider] = field(default_factory=list)
    hard_factors: list[HardFactor] = field(default_factory=list)


@dataclass
class CompiledLayoutIR:
    node_specs: dict[NodeId, NodeSpec]
    group_specs: dict[GroupId, GroupSpec]
    execution_sequence: list[DomainId]
    ordering_edges: list[tuple[NodeId, NodeId, str]] = field(default_factory=list)
    support_surface_scopes: dict[str, SupportSurfaceScope] = field(default_factory=dict)
    support_surface_components: dict[str, SupportSurfaceComponent] = field(default_factory=dict)
    unsupported: list[LayoutDiagnostic] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def node_spec(self, node_id: NodeId) -> NodeSpec | None:
        return self.node_specs.get(node_id)

    def group_spec(self, group_id: GroupId) -> GroupSpec | None:
        return self.group_specs.get(group_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "node_count": len(self.node_specs),
            "group_count": len(self.group_specs),
            "execution_sequence": [item.to_json() for item in self.execution_sequence],
            "ordering_edges": [{"pre": pre, "post": post, "axis": axis} for pre, post, axis in self.ordering_edges],
            "support_surface_scopes": {
                scope_id: {
                    "surface_id": scope.surface_id,
                    "component_count": len(scope.components),
                    "node_count": len(scope.node_ids),
                    "domain_refs": [ref.domain_id.to_json() for ref in scope.domain_refs],
                    "transactional_components": [
                        component.component_id
                        for component in scope.components
                        if component.component_id in self.support_surface_components
                    ],
                }
                for scope_id, scope in self.support_surface_scopes.items()
            },
            "unsupported": [diag.to_json() for diag in self.unsupported],
            "evidence": dict(self.evidence),
        }
