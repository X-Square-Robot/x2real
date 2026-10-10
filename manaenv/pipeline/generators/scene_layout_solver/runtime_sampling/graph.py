from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..common.ids import DomainId
from ..static_solve.ir import CompiledLayoutIR
from ..static_solve.problem import LayoutProblem


@dataclass(frozen=True)
class RuntimeCompileNode:
    node_id: str
    domain: DomainId | None
    support: str | None = None
    placement_owner: str = ""
    factors: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "domain": self.domain.to_json() if self.domain is not None else None,
            "support": self.support,
            "placement_owner": self.placement_owner,
            "factors": list(self.factors),
        }


@dataclass(frozen=True)
class RuntimeCompileScope:
    scope_id: str
    domain: DomainId
    kind: str
    members: tuple[str, ...]
    surfaces: tuple[str, ...] = ()
    dependency_nodes: tuple[str, ...] = ()
    factors: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.scope_id,
            "domain": self.domain.to_json(),
            "kind": self.kind,
            "members": list(self.members),
            "surfaces": list(self.surfaces),
            "dependency_nodes": list(self.dependency_nodes),
            "factors": list(self.factors),
        }


@dataclass(frozen=True)
class RuntimeCompileGraph:
    nodes: dict[str, RuntimeCompileNode] = field(default_factory=dict)
    scopes: list[RuntimeCompileScope] = field(default_factory=list)
    ordering_edges: tuple[tuple[str, str, str], ...] = ()
    unsupported: tuple[dict[str, Any], ...] = ()

    def domain_for_node(self, node_id: str) -> DomainId | None:
        node = self.nodes.get(str(node_id))
        return node.domain if node is not None else None

    def scope_for_domain(self, domain: DomainId) -> RuntimeCompileScope | None:
        for scope in self.scopes:
            if scope.domain == domain:
                return scope
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "nodes": {name: node.to_json() for name, node in sorted(self.nodes.items())},
            "scopes": [scope.to_json() for scope in self.scopes],
            "ordering_edges": [{"pre": pre, "post": post, "axis": axis} for pre, post, axis in self.ordering_edges],
            "unsupported": [dict(item) for item in self.unsupported],
        }


class RuntimeCompileGraphBuilder:
    """Build the runtime compile view of the solve-side IR.

    The graph records transaction/scope boundaries for runtime plan generation.
    It does not define placement semantics; semantics remain owned by the
    solve-side domain providers and hard factors.
    """

    def __init__(self, problem: LayoutProblem, ir: CompiledLayoutIR) -> None:
        self.problem = problem
        self.ir = ir

    def build(self) -> RuntimeCompileGraph:
        node_domains = self._node_domains()
        nodes = {
            name: RuntimeCompileNode(
                node_id=name,
                domain=node_domains.get(name),
                support=node.support,
                placement_owner=self.problem.objects[name].placement_owner if name in self.problem.objects else "",
                factors=tuple(_factor_names(node.hard_factors)),
            )
            for name, node in self.ir.node_specs.items()
        }
        scopes = [self._scope_for_domain(domain) for domain in self.ir.execution_sequence]
        scopes = [scope for scope in scopes if scope is not None]
        return RuntimeCompileGraph(
            nodes=nodes,
            scopes=scopes,
            ordering_edges=tuple(self.ir.ordering_edges),
            unsupported=tuple(diag.to_json() for diag in self.ir.unsupported),
        )

    def _node_domains(self) -> dict[str, DomainId]:
        out: dict[str, DomainId] = {}
        for group_id, group in self.ir.group_specs.items():
            domain = DomainId.group(group_id)
            for member in group.members:
                out[member] = domain
        for node_id, node in self.ir.node_specs.items():
            if node_id not in out and node.domain_providers:
                out[node_id] = DomainId.node(node_id)
        return out

    def _scope_for_domain(self, domain: DomainId) -> RuntimeCompileScope | None:
        if domain.kind == "group":
            group = self.ir.group_spec(domain.id)
            if group is None:
                return None
            surfaces = tuple(
                sorted({surface for surface in (self._surface_for_member(name) for name in group.members) if surface})
            )
            return RuntimeCompileScope(
                scope_id=domain.id,
                domain=domain,
                kind=_group_kind(domain.id),
                members=tuple(group.members),
                surfaces=surfaces,
                dependency_nodes=tuple(self._dependency_nodes(surfaces)),
                factors=tuple(_factor_names(group.hard_factors)),
            )
        node = self.ir.node_spec(domain.id)
        if node is None:
            return None
        surfaces = (node.support,) if node.support else ()
        return RuntimeCompileScope(
            scope_id=domain.id,
            domain=domain,
            kind="node",
            members=(domain.id,),
            surfaces=surfaces,
            dependency_nodes=tuple(self._dependency_nodes(surfaces)),
            factors=tuple(_factor_names(node.hard_factors)),
        )

    def _surface_for_member(self, node_id: str) -> str | None:
        obj = self.problem.objects.get(node_id)
        if obj is None:
            return None
        for region in self.problem.spec.regions:
            if node_id in region.objects:
                return region.surface
        for group in self.problem.spec.scatter_groups:
            if node_id in group.objects:
                return group.surface
        return obj.support

    def _dependency_nodes(self, surfaces: tuple[str, ...]) -> list[str]:
        out: list[str] = []
        for surface in surfaces:
            parent = self.problem.local_surface_parent(surface)
            if parent and parent not in out:
                out.append(parent)
        return out


def _factor_names(factors: list[Any]) -> list[str]:
    return [factor.__class__.__name__ for factor in factors]


def _group_kind(group_id: str) -> str:
    if ":" in group_id:
        return group_id.split(":", 1)[0]
    return "group"
