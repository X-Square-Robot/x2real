"""Dynamic composite-task type forging (Affordance-style).

Mirrors the affordance type-synthesis mechanism (``type(name, bases, ns)``,
see ``manaenv/utils/runtime_type_spec.py``) for tasks: a graph of atomic-task
nodes is composed at COMPILE TIME into a freshly-forged ``ManaTask`` subclass
named after the graph.

The forged class inherits ManaTask's getters unchanged — the namespace-merged
``runtime_cfgs`` (built by ``graph_compose``) supplies the combined per-node
data, so no getter is overridden. The dynamic type exists to carry graph
IDENTITY: a stable class name plus ``__graph_task__`` / ``__node_types__``
metadata for introspection / GNN, and (on the instance) the ``ManaGraph``.

The ManaTask base is INJECTED (``base_cls``) so this module never imports the
sim-coupled ``task_base`` and is exercised by the stability suite with a stub
base. The owning generator (which already imports ManaTask) passes the real
base in.
"""

from __future__ import annotations

import keyword
import re
from typing import Any

_GRAPH_TASK_SUFFIX = "GraphTask"
_INVALID_IDENT_CHARS = re.compile(r"[^0-9A-Za-z]+")


def forged_class_name(graph_name: str) -> str:
    """Derive a valid CamelCase ``<Graph>GraphTask`` class name from *graph_name*.

    Non-identifier chars split words; words are capitalised and joined. A
    leading digit or empty/keyword result falls back to a safe prefix so the
    name is always a valid, non-keyword Python identifier.
    """
    parts = [p for p in _INVALID_IDENT_CHARS.split(graph_name or "") if p]
    stem = "".join(word[:1].upper() + word[1:] for word in parts)
    if not stem or stem[0].isdigit():
        stem = f"Graph{stem}"
    name = f"{stem}{_GRAPH_TASK_SUFFIX}"
    if keyword.iskeyword(name):  # pragma: no cover - defensive
        name = f"_{name}"
    return name


def forge_graph_task_type(
    graph_name: str,
    base_cls: type,
    *,
    node_types: tuple[str, ...] = (),
    mixins: tuple[type, ...] = (),
) -> type:
    """``type()``-forge a ``base_cls`` (ManaTask) subclass for a composed graph.

    EVERY task is a graph task; a task that declares no stages simply carries an empty
    graph. So ``RecursiveCompositeTaskMixin`` is injected unconditionally, placed BEFORE
    ``base_cls`` in the MRO: with children its recursive-delegation getters compose and
    gate, and with none they fall back to the leaf answer — unchanged behaviour, whether
    that answer comes from ``base_cls`` or (for the getters Arena's base never had) from
    the mixin itself. That single rule is what removes the second task type: nothing
    downstream ever has to ask "is this a graph task?", because there is nothing else to be.

    The forged class also carries graph IDENTITY metadata (``__graph_task__`` /
    ``__node_types__``) for introspection / GNN. A fresh type is forged per composition
    (task generation runs once per order, so there is no caching concern).
    """
    cls_name = forged_class_name(graph_name)
    namespace: dict[str, Any] = {
        "__doc__": f"Dynamically forged composite graph task {cls_name!r} "
        f"composed from atomic nodes {tuple(node_types)!r}.",
        "__graph_task__": True,
        "__graph_name__": graph_name,
        "__node_types__": tuple(node_types),
    }
    bases = (*mixins, base_cls)
    return type(cls_name, bases, namespace)
