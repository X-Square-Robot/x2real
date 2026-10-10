"""Graph-composition: merge N graph-node ``runtime_cfgs`` into one flat task.

P1 (``namespace_merge``) provides the per-section primitives; this module
decides the per-section *strategy* and applies it across a whole
``runtime_cfgs`` dict. It is agnostic to how the nodes were authored (YAML
graph schema + generator wiring is P2b), so it is a pure, stability-testable
function over ``[(node_id, runtime_cfgs), ...]``.

Single-node (atomic) graph → **identity**: the lone node's ``runtime_cfgs`` is
returned (shallow copy, same value objects), byte-identical in content to today,
so every existing atomic task and the whole downstream consumer table is
untouched.

Multi-node graph → **namespace-merge**:
  * configclass sections (``termination_terms``, ``event_terms``,
    ``observation_cfg``, …) → field names suffixed by node id, then combined,
    so two nodes each declaring ``success`` / ``time_out`` do not collide.
  * ``metrics`` → per-node suffix, ``success_rate`` deduped (one composite).
  * ``scene_constraints`` / ``subtasks`` → concatenated.
  * ``_term_combine_modes`` → dict-merged.
  * ``instruction`` / ``topic`` / ``default_visualizer_cfg`` → first node's (composite-level
    scalar; multi-node aggregation semantics TBD).
  * ``success_term_cfg`` / ``evaluation_term_cfg`` → **not**
    aggregated here. These are composite-level semantics owned by P3 (DAG-success
    termination term). Left as ``None`` so the env
    still builds; the per-node ``success__node_X`` terms remain inside the merged
    ``termination_terms`` for P3's DAG-success func to address.
"""

from __future__ import annotations

from typing import Any

from manaenv.pipeline.generators.task.namespace_merge import merge_configclass_sections, merge_metric_lists

CONFIGCLASS_SECTIONS: frozenset[str] = frozenset(
    {
        "termination_terms",
        "event_terms",
        "observation_cfg",
        "rewards_cfg",
        "curriculum_cfg",
        "commands_cfg",
        "domain_randomization_cfg",
    }
)
LIST_CONCAT_SECTIONS: frozenset[str] = frozenset({"scene_constraints", "subtasks"})
DICT_MERGE_SECTIONS: frozenset[str] = frozenset({"_term_combine_modes"})
SCALAR_FIRST_SECTIONS: frozenset[str] = frozenset({"instruction", "topic", "default_visualizer_cfg"})
# Composite-level semantics deferred to P3; set to None so env build still works.
# ``failure_term_cfg`` mirrors ``success_term_cfg``: the composite computes its
# gated failure term at P3 via ``get_failure_term_cfg``, not by aggregating leaves.
DEFER_TO_P3_SECTIONS: frozenset[str] = frozenset(
    {"success_term_cfg", "failure_term_cfgs", "evaluation_term_cfg"}
)
# ``planner_skills`` is authored per graph node and read back off each live
# ``_composite_children`` entry by the SkillTrajectoryPlanner at setup. The
# composite's own merged ``runtime_cfgs`` must NOT aggregate it (a list, not a
# configclass), so it is dropped from the merged dict (left as None).
PER_NODE_ONLY_SECTIONS: frozenset[str] = frozenset({"planner_skills"})
METRIC_SECTION = "metrics"
SHARED_METRIC_NAMES: tuple[str, ...] = ("success_rate",)


def _first_non_none(per_node: list[tuple[str, Any]]) -> Any:
    for _nid, value in per_node:
        if value is not None:
            return value
    return None


def _merge_section(key: str, per_node: list[tuple[str, Any]]) -> Any:
    """Apply the per-section merge strategy for a multi-node graph."""
    if key in DEFER_TO_P3_SECTIONS or key in PER_NODE_ONLY_SECTIONS:
        return None
    if key in CONFIGCLASS_SECTIONS:
        return merge_configclass_sections(per_node, name=_section_cls_name(key), namespace=True)
    if key == METRIC_SECTION:
        return merge_metric_lists(per_node, namespace=True, shared_names=SHARED_METRIC_NAMES)
    if key in LIST_CONCAT_SECTIONS:
        merged: list[Any] = []
        for _nid, value in per_node:
            merged.extend(value or [])
        return merged
    if key in DICT_MERGE_SECTIONS:
        merged_dict: dict[str, Any] = {}
        for _nid, value in per_node:
            if isinstance(value, dict):
                merged_dict.update(value)
        return merged_dict
    # SCALAR_FIRST_SECTIONS and any unknown key: take the first node's value.
    return _first_non_none(per_node)


def _section_cls_name(key: str) -> str:
    """Stable configclass name for a merged section (e.g. termination_terms -> TerminationTermsCfg)."""
    return "".join(part.capitalize() for part in key.split("_")) + "Cfg"


def compose_graph_runtime_cfgs(
    named_runtime_cfgs: list[tuple[str, dict[str, Any]]],
    *,
    namespace: bool,
) -> dict[str, Any]:
    """Merge per-node ``runtime_cfgs`` dicts into one flat ``runtime_cfgs``.

    ``namespace`` is decided once by the caller from the graph:
    ``namespace = not graph.is_single_node()``. ``False`` returns the lone
    node's dict (shallow copy, identity semantics); ``True`` applies the
    per-section strategy. Section key order follows first appearance across
    nodes for determinism.
    """
    present = [(nid, rc or {}) for nid, rc in named_runtime_cfgs]
    if not namespace:
        non_empty = [rc for _nid, rc in present if rc]
        if len(non_empty) > 1:
            raise ValueError(
                f"compose_graph_runtime_cfgs: namespace=False (single-node) but "
                f"{len(non_empty)} nodes carry runtime_cfgs."
            )
        return dict(non_empty[0]) if non_empty else {}

    ordered_keys: list[str] = []
    for _nid, rc in present:
        for key in rc:
            if key not in ordered_keys:
                ordered_keys.append(key)

    merged: dict[str, Any] = {}
    for key in ordered_keys:
        per_node = [(nid, rc.get(key)) for nid, rc in present]
        merged[key] = _merge_section(key, per_node)
    return merged
