"""Parse a task-graph spec (nodes + edges) into a :class:`ManaGraph`.

Pure / sim-free: a node's payload is the **reflected sub-generator CFG**
(a ``TaskGeneratorCFG`` dataclass — reflection imports no IsaacSim). Only
*running* those sub-generators (the P2 generator glue) needs sim. This module
just builds the dependency-graph structure, so it is exercised by the
stability suite.

Node shape (after include-expansion by the translator):
    ``{id: <str>, <one *_task_generator block>, **node_attrs}``
Edge shape:
    ``{src: <id>, dst: <id>, **edge_attrs}``  (``dst`` depends on ``src``)

Atomic tasks (Direct / Template) are the degenerate single-node, zero-edge
case; the same parser handles them uniformly.
"""

from __future__ import annotations

from typing import Any

from manaenv.pipeline.generators.task_generator import TaskGeneratorCFG
from manaenv.pipeline.graph.mana_graph import ManaGraph
from manaenv.pipeline.translator.yaml_reflection import reflect_yaml_dict

GENERATOR_KEY_SUFFIX = "_task_generator"


def _reflect_node_generator_cfg(node_id: str, block_key: str, block: Any) -> TaskGeneratorCFG:
    # Some translator paths pre-reflect the nested block to a CFG already
    # (mirrors TaskOrdersCFG._normalize_task_order_item); accept either shape.
    if isinstance(block, TaskGeneratorCFG):
        return block
    reflected = reflect_yaml_dict({block_key: block}, context=f"graph node {node_id!r}")
    cfg = reflected.get(block_key)
    if not isinstance(cfg, TaskGeneratorCFG):
        raise TypeError(
            f"graph node {node_id!r}: {block_key!r} did not reflect to a TaskGeneratorCFG "
            f"(got {type(cfg).__name__})."
        )
    return cfg


def parse_node(node: Any) -> tuple[str, TaskGeneratorCFG, dict[str, Any]]:
    """Return ``(node_id, sub_generator_cfg, node_attrs)`` for one node spec."""
    if not isinstance(node, dict):
        raise TypeError(f"graph node must be a mapping, got {type(node).__name__}")
    node_id = node.get("id")
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("graph node requires a non-empty string 'id'")
    gen_items = [(key, value) for key, value in node.items() if str(key).endswith(GENERATOR_KEY_SUFFIX)]
    if len(gen_items) != 1:
        raise ValueError(
            f"graph node {node_id!r} must declare exactly one *{GENERATOR_KEY_SUFFIX} block, "
            f"found {len(gen_items)}: {[k for k, _ in gen_items]}"
        )
    block_key, block = gen_items[0]
    cfg = _reflect_node_generator_cfg(node_id, block_key, block)
    attrs = {k: v for k, v in node.items() if k != "id" and not str(k).endswith(GENERATOR_KEY_SUFFIX)}
    return node_id, cfg, attrs


def parse_edge(edge: Any) -> tuple[str, str, dict[str, Any]]:
    """Return ``(src, dst, edge_attrs)`` for one edge spec (``dst`` depends on ``src``)."""
    if not isinstance(edge, dict):
        raise TypeError(f"graph edge must be a mapping, got {type(edge).__name__}")
    src = edge.get("src")
    dst = edge.get("dst")
    if not isinstance(src, str) or not src or not isinstance(dst, str) or not dst:
        raise ValueError("graph edge requires non-empty string 'src' and 'dst'")
    attrs = {k: v for k, v in edge.items() if k not in ("src", "dst")}
    return src, dst, attrs


def parse_graph_spec(nodes: list[Any] | None, edges: list[Any] | None) -> ManaGraph:
    """Build a :class:`ManaGraph` from node + edge specs.

    Node payloads are the reflected sub-generator CFGs; remaining node keys
    become ``ManaNode.attrs``. Edge endpoint existence and acyclicity are
    enforced downstream by :meth:`ManaGraph.add_edge` / :meth:`ManaGraph.solve`.
    """
    graph = ManaGraph()
    for node in nodes or []:
        node_id, cfg, attrs = parse_node(node)
        graph.add_node(node_id, payload=cfg, **attrs)
    for edge in edges or []:
        src, dst, attrs = parse_edge(edge)
        graph.add_edge(src, dst, **attrs)
    return graph
