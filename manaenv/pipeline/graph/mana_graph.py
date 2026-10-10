"""ManaGraph: a general-purpose directed dependency graph.

Generalizes the topological solver from ``PipelineDagCFG.solve()``
(``manaenv/pipeline/pipeline_cfg.py``) into a reusable structure for
task-relationship graphs — and, later, for pipeline-stage orchestration.

A node carries an opaque ``payload`` (an atomic-task reference, runtime cfgs,
…) plus a free-form ``attrs`` dict. An edge declares a base dependency
relation — ``dst`` depends on ``src`` (``src`` must complete first) — plus an
open ``attrs`` dict for custom payloads (GNN edge features, phase semantics).
The custom-payload contract is intentionally open for now; it may be frozen
into a typed extension later.

Pure Python: no IsaacSim / IsaacLab import, so it is usable at compile time
and exercised by the stability suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ManaNode:
    """One graph node: stable ``id`` + opaque ``payload`` + feature ``attrs``."""

    id: str
    payload: Any = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ManaEdge:
    """A directed dependency edge: ``dst`` depends on ``src``."""

    src: str
    dst: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ManaGraphSolution:
    """Solved views of a :class:`ManaGraph`.

    * ``order`` — a full topological order (dependencies before dependents).
    * ``levels`` — successive ready-frontiers; nodes in one level have all
      dependencies satisfied by earlier levels and may run in parallel.
    * ``roots`` — nodes with no incoming dependency edge.
    * ``leaves`` — nodes with no outgoing dependency edge.
    """

    order: list[str]
    levels: list[list[str]]
    roots: list[str]
    leaves: list[str]


class ManaGraph:
    """A directed dependency graph of :class:`ManaNode` + :class:`ManaEdge`.

    Insertion order of nodes is preserved so that solving is deterministic
    (ties within a ready-frontier break by node insertion order).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, ManaNode] = {}
        self._edges: list[ManaEdge] = []

    # -- construction / mutation --------------------------------------

    def add_node(self, node_id: str, payload: Any = None, **attrs: Any) -> ManaNode:
        """Add a node; raise on duplicate id."""
        if node_id in self._nodes:
            raise ValueError(f"ManaGraph: duplicate node id {node_id!r}")
        node = ManaNode(id=node_id, payload=payload, attrs=dict(attrs))
        self._nodes[node_id] = node
        return node

    def add_edge(self, src: str, dst: str, **attrs: Any) -> ManaEdge:
        """Add a dependency edge ``dst`` depends on ``src``.

        Both endpoints must already exist; self-loops are rejected.
        """
        if src not in self._nodes:
            raise ValueError(f"ManaGraph: edge src {src!r} is not a known node")
        if dst not in self._nodes:
            raise ValueError(f"ManaGraph: edge dst {dst!r} is not a known node")
        if src == dst:
            raise ValueError(f"ManaGraph: self-loop on node {src!r} is not allowed")
        edge = ManaEdge(src=src, dst=dst, attrs=dict(attrs))
        self._edges.append(edge)
        return edge

    # -- queries ------------------------------------------------------

    @property
    def nodes(self) -> list[ManaNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[ManaEdge]:
        return list(self._edges)

    def __len__(self) -> int:
        return len(self._nodes)

    def get_node(self, node_id: str) -> ManaNode:
        return self._nodes[node_id]

    def is_single_node(self) -> bool:
        """An atomic task is a one-node, zero-edge graph (identity merge)."""
        return len(self._nodes) == 1 and not self._edges

    # -- solving ------------------------------------------------------

    def solve(self) -> ManaGraphSolution:
        """Topologically solve the graph into order + parallel levels.

        Generalizes ``PipelineDagCFG.solve()`` (Kahn's algorithm), with edges
        given explicitly as ``(src -> dst)`` dependencies instead of per-node
        ``after`` lists. Ties within a ready-frontier break by node insertion
        order for determinism. Raises ``ValueError`` on a cycle.
        """
        # incoming[node] = set of nodes that must complete before it.
        incoming: dict[str, set[str]] = {node_id: set() for node_id in self._nodes}
        outgoing: dict[str, set[str]] = {node_id: set() for node_id in self._nodes}
        for edge in self._edges:
            # Duplicate edges collapse naturally via set semantics.
            incoming[edge.dst].add(edge.src)
            outgoing[edge.src].add(edge.dst)

        roots = [node_id for node_id in self._nodes if not incoming[node_id]]
        leaves = [node_id for node_id in self._nodes if not outgoing[node_id]]

        # Work on a mutable copy so the graph itself is not consumed.
        pending: dict[str, set[str]] = {node_id: set(deps) for node_id, deps in incoming.items()}
        ready = [node_id for node_id in self._nodes if not pending[node_id]]
        order: list[str] = []
        levels: list[list[str]] = []

        while ready:
            level = list(ready)
            levels.append(level)
            order.extend(level)
            solved = set(order)
            for solved_id in level:
                for node_id in outgoing[solved_id]:
                    pending[node_id].discard(solved_id)
            # Form the next frontier by scanning nodes in insertion order so
            # ties within a level are deterministic (not set-iteration order).
            ready = [node_id for node_id in self._nodes if node_id not in solved and not pending[node_id]]

        if len(order) != len(self._nodes):
            cycle_nodes = sorted(node_id for node_id, deps in pending.items() if deps)
            raise ValueError(f"ManaGraph contains a cycle involving: {cycle_nodes}")

        return ManaGraphSolution(order=order, levels=levels, roots=roots, leaves=leaves)

    # -- serialization (GNN-friendly) ---------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Stable adjacency + feature serialization.

        Node/edge ordering is preserved (nodes by insertion, edges by add
        order) so the output is deterministic and suitable as a GNN feed.
        ``payload`` is included verbatim; callers needing JSON must ensure
        payloads are serializable (graph topology + attrs always are).
        """
        return {
            "nodes": [{"id": n.id, "payload": n.payload, "attrs": dict(n.attrs)} for n in self._nodes.values()],
            "edges": [{"src": e.src, "dst": e.dst, "attrs": dict(e.attrs)} for e in self._edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManaGraph:
        """Rebuild a graph from :meth:`to_dict` output."""
        graph = cls()
        for node in data.get("nodes", []):
            graph.add_node(node["id"], payload=node.get("payload"), **dict(node.get("attrs", {})))
        for edge in data.get("edges", []):
            graph.add_edge(edge["src"], edge["dst"], **dict(edge.get("attrs", {})))
        return graph
