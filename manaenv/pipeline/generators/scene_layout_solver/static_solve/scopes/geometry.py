from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common.constants import GEOM_EPS
from ...common.geometry import Bounds2D, Range1D
from ...common.ids import NodeId
from ..domains.providers import SupportOnProvider, object_scene_yaw_range
from ..ir import GroupSpec, NodeSpec
from ..problem import LayoutProblem
from .model import ScopeDomainRef, ScopeInteractionEdge


@dataclass(frozen=True)
class DomainGeometry:
    domain_ref: ScopeDomainRef
    center_bounds_by_node: dict[NodeId, Bounds2D]
    swept_bounds: Bounds2D | None
    evidence: dict[str, Any] = field(default_factory=dict)


def domain_geometry_for_ref(
    *,
    problem: LayoutProblem,
    ref: ScopeDomainRef,
    node_specs: dict[str, NodeSpec],
    group_specs: dict[str, GroupSpec],
) -> DomainGeometry:
    provider = SupportOnProvider(problem)
    center_bounds: dict[NodeId, Bounds2D] = {}
    swept: list[Bounds2D] = []
    relation = _group_relation(ref, group_specs)
    group_surface_range = getattr(relation, "surface_range", None) if relation is not None else None
    relation_yaw = getattr(relation, "yaw", None) if relation is not None else None
    authored_range = group_surface_range is not None
    for node_id in ref.node_ids:
        obj = problem.objects.get(node_id)
        if obj is not None and (obj.surface_range is not None or obj.override_surface_range is not None):
            authored_range = True
        yaw_range = object_scene_yaw_range(problem, node_id)
        if yaw_range is None and relation_yaw is not None:
            yaw_range = (float(relation_yaw[0]), float(relation_yaw[1]))
        bounds, meta = provider.center_bounds(
            node_id,
            surface_override=ref.surface_id,
            group_surface_range=group_surface_range,
            yaw_range=yaw_range,
        )
        if bounds is None:
            continue
        center_bounds[node_id] = bounds
        surface = provider.surface(ref.surface_id)
        if surface is None:
            swept.append(bounds)
            continue
        footprint = provider.footprint_bounds(node_id, surface, yaw_range=yaw_range)
        swept.append(
            Bounds2D(
                Range1D(bounds.x.min + footprint.x.min, bounds.x.max + footprint.x.max),
                Range1D(bounds.y.min + footprint.y.min, bounds.y.max + footprint.y.max),
            )
        )
    return DomainGeometry(
        domain_ref=ref,
        center_bounds_by_node=center_bounds,
        swept_bounds=_union_bounds(swept) if swept else None,
        evidence={"node_count": len(ref.node_ids), "authored_range": authored_range},
    )


def refs_may_interact(
    left: DomainGeometry, right: DomainGeometry, *, eps: float = GEOM_EPS
) -> ScopeInteractionEdge | None:
    if left.swept_bounds is None or right.swept_bounds is None:
        return None
    if not left.evidence.get("authored_range") or not right.evidence.get("authored_range"):
        return None
    if not _bounds2d_overlap(left.swept_bounds, right.swept_bounds, eps=eps):
        return None
    return ScopeInteractionEdge(
        left=left.domain_ref.domain_id,
        right=right.domain_ref.domain_id,
        reason="range_overlap",
        evidence={
            "left_swept": _bounds_to_json(left.swept_bounds),
            "right_swept": _bounds_to_json(right.swept_bounds),
        },
    )


def relation_edges_for_refs(
    problem: LayoutProblem, refs: list[ScopeDomainRef], *, surface_id: str
) -> list[ScopeInteractionEdge]:
    owner_by_node: dict[str, ScopeDomainRef] = {}
    for ref in refs:
        for node_id in ref.node_ids:
            owner_by_node[node_id] = ref
    out: list[ScopeInteractionEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for rel in problem.relation_index.relative_by_surface.get(surface_id, []):
        left = owner_by_node.get(rel.node_1)
        right = owner_by_node.get(rel.node_2)
        if left is None or right is None or left.domain_id == right.domain_id:
            continue
        key = tuple(sorted((left.domain_id.id, right.domain_id.id)) + [str(rel.direction)])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            ScopeInteractionEdge(
                left=left.domain_id,
                right=right.domain_id,
                reason="relative_direction",
                evidence={"relation": f"{rel.node_1}->{rel.node_2}", "direction": rel.direction},
            )
        )
    return out


def _group_relation(ref: ScopeDomainRef, group_specs: dict[str, GroupSpec]):
    if ref.domain_id.kind != "group":
        return None
    spec = group_specs.get(ref.domain_id.id)
    provider = spec.domain_providers[0] if spec and spec.domain_providers else None
    return getattr(provider, "relation", None)


def _bounds2d_overlap(a: Bounds2D, b: Bounds2D, *, eps: float) -> bool:
    return a.x.min < b.x.max - eps and b.x.min < a.x.max - eps and a.y.min < b.y.max - eps and b.y.min < a.y.max - eps


def _union_bounds(bounds: list[Bounds2D]) -> Bounds2D:
    return Bounds2D(
        Range1D(min(item.x.min for item in bounds), max(item.x.max for item in bounds)),
        Range1D(min(item.y.min for item in bounds), max(item.y.max for item in bounds)),
    )


def _bounds_to_json(bounds: Bounds2D) -> dict[str, list[float]]:
    return {"x": [bounds.x.min, bounds.x.max], "y": [bounds.y.min, bounds.y.max]}
