"""Namespace-merge tooling for composing N graph-node task cfgs into one flat task.

Generalizes the field-renaming transforms from
``isaaclab_arena.tasks.sequential_task_base`` — which is currently
**un-importable**: it references two helpers that do not exist anywhere in the
repo (``transform_configclass_instance`` / ``check_configclass_field_duplicates``)
and is dormant scaffolding (no generator / CFG / class_type, zero YAML use).

This module re-implements those two missing helpers plus the suffix/remove
transforms, keyed on a **stable node id** instead of a list index, so N atomic
task cfgs can merge into one flat ``ManaTask.runtime_cfgs`` without field-name
collisions (two nodes each declaring ``success`` / ``time_out``, or roles both
named ``drink``).

Single-node graphs must merge to a **byte-identical** result (no suffix) so an
atomic task keeps its bare ``success`` / ``time_out`` term names — the whole
downstream consumer table and the ``success_term_cfg`` lookup depend on it.
The per-instance helpers here therefore namespace unconditionally; the
single-node identity short-circuit lives at the orchestration layer (P2), which
simply skips namespacing when the graph ``is_single_node()``.

Pure Python + plain dataclasses (no IsaacSim import), so it is exercised by
the stability suite.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from typing import Any

from manaenv.pipeline.generators.task.section_merge import make_section_dataclass

FieldTuple = tuple  # (name, type, value)


def node_suffix(node_id: str) -> str:
    """Canonical field-name / role-key suffix for a graph node."""
    return f"__node_{node_id}"


def get_configclass_field_tuples(instance: Any) -> list[FieldTuple]:
    """Reflect a configclass/dataclass instance into ``(name, type, value)`` tuples."""
    return [(f.name, f.type, getattr(instance, f.name)) for f in dataclasses.fields(instance)]


def combine_configclass_instances(name: str, *instances: Any) -> Any:
    """Combine dataclass/configclass instances without importing IsaacLab."""
    field_tuples: list[FieldTuple] = []
    for inst in instances:
        if inst is None:
            continue
        field_tuples.extend(get_configclass_field_tuples(inst))
    if not field_tuples:
        return None
    return make_section_dataclass(name, field_tuples)()


def add_suffix_configclass_transform(fields: list[FieldTuple], suffix: str) -> list[FieldTuple]:
    """Append ``suffix`` to every field name."""
    return [(f"{name}{suffix}", ftype, value) for name, ftype, value in fields]


def remove_configclass_transform(fields: list[FieldTuple], exclude_fields: set[str]) -> list[FieldTuple]:
    """Drop fields whose name is in ``exclude_fields``."""
    return [(name, ftype, value) for name, ftype, value in fields if name not in exclude_fields]


def transform_configclass_instance(
    instance: Any,
    transform: Callable[[list[FieldTuple]], list[FieldTuple]],
    *,
    name: str | None = None,
) -> Any:
    """Reflect → transform field-tuples → rebuild a configclass instance.

    Returns ``None`` when the transform yields no fields, so empty cfgs drop out
    of a downstream ``combine_configclass_instances`` (which filters ``None``).
    """
    if instance is None:
        return None
    new_fields = transform(get_configclass_field_tuples(instance))
    if not new_fields:
        return None
    cls_name = name or type(instance).__name__
    return make_section_dataclass(cls_name, new_fields)()


def check_configclass_field_duplicates(*instances: Any) -> set[str]:
    """Return the set of field names that appear on more than one instance."""
    seen: set[str] = set()
    dupes: set[str] = set()
    for inst in instances:
        if inst is None:
            continue
        for f in dataclasses.fields(inst):
            if f.name in seen:
                dupes.add(f.name)
            seen.add(f.name)
    return dupes


# ---------------------------------------------------------------------------
# Section-level merge (graph-level identity rule via the ``namespace`` flag)
# ---------------------------------------------------------------------------
#
# The caller (P2 orchestration) decides ``namespace`` ONCE from the graph:
# ``namespace = not graph.is_single_node()``. In a single-node (atomic) graph
# nothing is suffixed, so the flat result is byte-identical to today. In a
# multi-node graph every node's contribution is suffixed by node id — even a
# single-contributor section — so the DAG-success term (P3) can address each
# node's success signal by a stable, predictable name.


def merge_configclass_sections(
    named_instances: list[tuple[str, Any]],
    *,
    name: str,
    namespace: bool,
) -> Any:
    """Merge per-node configclass instances of one ``runtime_cfgs`` section.

    ``named_instances`` is ``[(node_id, instance | None), ...]``. With
    ``namespace=False`` (single-node graph) the lone non-None instance is
    returned unchanged (identity); ``namespace=True`` suffixes each instance's
    fields by node id and combines them. Returns ``None`` when nothing to merge.
    """
    present = [(nid, inst) for nid, inst in named_instances if inst is not None]
    if not present:
        return None
    if not namespace:
        if len(present) > 1:
            raise ValueError(
                f"merge_configclass_sections({name!r}): namespace=False but {len(present)} "
                "contributors — single-node identity merge expects at most one."
            )
        return present[0][1]
    suffixed = [
        transform_configclass_instance(
            inst, lambda fields, nid=nid: add_suffix_configclass_transform(fields, node_suffix(nid))
        )
        for nid, inst in present
    ]
    return combine_configclass_instances(name, *suffixed)


def merge_roles(named_roles: list[tuple[str, dict[str, Any]]], *, namespace: bool) -> dict[str, Any]:
    """Merge per-node ``roles`` dicts.

    ``namespace=False`` returns the lone node's roles unchanged; ``namespace=True``
    suffixes every role key by node id so two nodes both naming ``drink`` stay
    distinct. The matching ``@role`` references inside each node's term params
    must be rewritten to the suffixed key by the caller (P2) before merge.
    """
    present = [(nid, roles or {}) for nid, roles in named_roles]
    non_empty = [(nid, roles) for nid, roles in present if roles]
    if not namespace:
        if len(non_empty) > 1:
            raise ValueError(f"merge_roles: namespace=False but {len(non_empty)} contributors with roles.")
        return dict(non_empty[0][1]) if non_empty else {}
    merged: dict[str, Any] = {}
    for nid, roles in present:
        for key, value in roles.items():
            merged[f"{key}{node_suffix(nid)}"] = value
    return merged


def merge_metric_lists(
    named_metrics: list[tuple[str, list[Any]]],
    *,
    namespace: bool,
    shared_names: tuple[str, ...] = (),
) -> list[Any]:
    """Merge per-node metric lists, suffixing each metric's ``.name``.

    Metrics whose ``name`` is in ``shared_names`` (e.g. ``success_rate``) are
    left un-suffixed and deduplicated across nodes — there is one composite
    success for the whole graph. ``namespace=False`` returns the lone node's
    list unchanged.
    """
    if not namespace:
        flat = [m for _nid, metrics in named_metrics for m in (metrics or [])]
        return flat
    merged: list[Any] = []
    seen_shared: set[str] = set()
    for nid, metrics in named_metrics:
        for metric in metrics or []:
            mname = getattr(metric, "name", None)
            if mname in shared_names:
                if mname in seen_shared:
                    continue
                seen_shared.add(mname)
                merged.append(metric)
                continue
            clone = copy.copy(metric)
            suffix = node_suffix(nid)
            if hasattr(clone, "name") and clone.name is not None:
                clone.name = f"{clone.name}{suffix}"
            merged.append(clone)
    return merged
