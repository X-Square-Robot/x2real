"""Dedup/coexist/raise merge for independent composite-task sections.

This is the merge model the user fixed for composite ManaTasks: graph nodes
are **fully independent** tasks; a non-gating section field shared across nodes
is **deduplicated when structurally identical**, differently-named fields
**coexist**, and the same field name with **different content raises** (explicit
resolution required — no silent last-wins, no auto-suffix).

It deliberately supersedes the earlier suffix-keyed approach
(``namespace_merge`` / ``node_scope`` / ``graph_compose``): there is no
``__node_X`` renaming. Temporal/gated success lives in the DAG-success machine
(``dag_success``), never here.

Structural identity uses ``repr`` as a fingerprint: cfgs built by including the
same YAML fragment resolve to the same registry func objects and equal params,
so their reprs match; genuinely different params differ. Errs toward raising
(the conservative direction the user chose).

Pure Python + ``isaaclab_arena.utils.configclass`` (no IsaacSim import) →
stability-testable.
"""

from __future__ import annotations

import copy
import dataclasses
import keyword
import types
from typing import Any


def _fingerprint(value: Any) -> str:
    """Structural fingerprint for dedup/conflict detection.

    ``repr`` is the fingerprint for any value whose type defines its own
    ``__repr__`` (configclasses, dataclasses, builtins, containers) — those
    repr to value-stable strings, so identical content matches.

    Objects that inherit the default identity-based ``object.__repr__`` (e.g.
    plain metric instances like ``SuccessRateMetric``, which subclasses an
    ABC with no ``__repr__``) would otherwise fingerprint by memory address:
    two structurally-identical per-node instances (the shared ``success_rate``
    metric each stage builds from the same base) would read as "different
    content" and wrongly raise. For those, fall back to a structural
    fingerprint built from the type and instance state, recursively.
    """
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{','.join(_fingerprint(v) for v in value)}]"
    if isinstance(value, dict):
        items = ",".join(
            f"{_fingerprint(k)}:{_fingerprint(v)}" for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))
        )
        return f"dict{{{items}}}"
    if type(value).__repr__ is not object.__repr__:
        return repr(value)
    # Identity-repr object: fingerprint by type + recursively-fingerprinted
    # instance state so structurally-identical instances match.
    state = vars(value) if hasattr(value, "__dict__") else {}
    items = ",".join(f"{k}={_fingerprint(v)}" for k, v in sorted(state.items()))
    return f"{type(value).__module__}.{type(value).__qualname__}({items})"


def make_section_dataclass(cls_name: str, fields: list[tuple]) -> type:
    """Build a dynamic dataclass without importing IsaacLab/pxr."""
    seen = set()
    annotations = {}
    defaults = {}
    for item in fields:
        if isinstance(item, str):
            field_name = item
            field_type = Any
        elif len(item) == 2:
            field_name, field_type = item
        elif len(item) == 3:
            field_name, field_type, spec = item
            defaults[field_name] = dataclasses.field(default_factory=_default_factory(spec))
        else:
            raise TypeError(f"Invalid field: {item!r}")

        if not isinstance(field_name, str) or not field_name.isidentifier():
            raise TypeError(f"Field names must be valid identifiers: {field_name!r}")
        if keyword.iskeyword(field_name):
            raise TypeError(f"Field names must not be keywords: {field_name!r}")
        if field_name in seen:
            raise TypeError(f"Field name duplicated: {field_name!r}")

        seen.add(field_name)
        annotations[field_name] = field_type

    def exec_body_callback(ns):
        ns.update(defaults)
        ns["__annotations__"] = annotations

    cls = types.new_class(cls_name, (), {}, exec_body_callback)
    return dataclasses.dataclass(cls)


def _default_factory(value: Any):
    def _factory():
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    return _factory


def combine_dedup_sections(
    named_instances: list[tuple[str, Any]],
    *,
    name: str,
    exclude_fields: frozenset[str] = frozenset(),
    exclude_prefixes: tuple[str, ...] = (),
    extra_fields: list[tuple] | None = None,
) -> Any:
    """Combine per-node configclass instances of one shared, non-gating section.

    ``named_instances`` is ``[(node_id, instance | None), ...]``. A field on
    multiple nodes is kept once when identical, raises when nodes disagree, and
    differently-named fields coexist. Returns ``None`` when nothing to combine.

    ``exclude_fields`` drops named fields from every node before merging (used
    to strip each child's gated ``success`` term, which is injected separately).
    ``exclude_prefixes`` drops any field whose name starts with one of these
    (used to strip each child's ``failure_*`` terms, folded into the one stage
    machine and injected as the composite ``failure``). ``extra_fields`` is a
    list of ``(name, type, value)`` specs appended to the result. When
    ``extra_fields`` is given the result is built even if all nodes are empty.
    """
    present = [(nid, inst) for nid, inst in named_instances if inst is not None]
    extras = list(extra_fields or [])
    if not present and not extras:
        return None

    # field name -> (owner_node_id, fingerprint, (name, type, value))
    chosen: dict[str, tuple[str, str, tuple]] = {}
    for node_id, inst in present:
        for f in dataclasses.fields(inst):
            if f.name in exclude_fields or (exclude_prefixes and f.name.startswith(exclude_prefixes)):
                continue
            value = getattr(inst, f.name)
            fp = _fingerprint(value)
            if f.name in chosen:
                prev_node, prev_fp, _ = chosen[f.name]
                if prev_fp != fp:
                    raise ValueError(
                        f"composite section {name!r}: nodes {prev_node!r} and {node_id!r} both define "
                        f"field {f.name!r} with different content. Independent nodes must rename one "
                        f"or make them identical (no silent merge)."
                    )
                continue  # identical → dedup
            chosen[f.name] = (node_id, fp, (f.name, f.type, value))

    field_specs = [spec for _node, _fp, spec in chosen.values()]
    for spec in extras:
        field_specs.append(spec)
    return make_section_dataclass(name, field_specs)()


def dedup_metric_list(named_metrics: list[tuple[str, list[Any]]]) -> list[Any]:
    """Dedup metric instances across nodes by ``name`` (same rule as sections).

    Same metric name + identical content → kept once; same name + different
    content → raise; distinct names → coexist (in first-seen order).
    """
    chosen: dict[str, tuple[str, str, Any]] = {}
    order: list[str] = []
    for node_id, metrics in named_metrics:
        for metric in metrics or []:
            mname = getattr(metric, "name", None)
            if mname is None:
                # Unnamed metric — cannot dedup; keep under a positional key.
                key = f"__unnamed_{len(order)}"
                chosen[key] = (node_id, "", metric)
                order.append(key)
                continue
            fp = _fingerprint(metric)
            if mname in chosen:
                prev_node, prev_fp, _ = chosen[mname]
                if prev_fp != fp:
                    raise ValueError(
                        f"composite metrics: nodes {prev_node!r} and {node_id!r} both define metric "
                        f"{mname!r} with different content. Rename one or make them identical."
                    )
                continue
            chosen[mname] = (node_id, fp, metric)
            order.append(mname)
    return [chosen[key][2] for key in order]
