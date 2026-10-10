"""The ONE failure-reason vocabulary — a torch-free string helper.

Extracted from ``dag_success`` so the serving layer (OD's ``analysis.py``) can clean a stored
report's labels at read time without importing torch. ``dag_success`` re-exports it, so the stage
machine and the eval runtime keep deriving reasons from the same single function — the whole point
is that "the machine said ``time_out``" and "the report shows ``mdp__time_out``" can never diverge.
"""

from __future__ import annotations

import re


def failure_reason_label(field_name: str) -> str:
    """The reason a non-success termination names, from its config FIELD name.

    Field names carry two kinds of plumbing this sheds: a YAML dotted key is flattened with
    ``__`` (``mdp.time_out`` -> ``mdp__time_out``), and an atomic term expanded per asset gains an
    index suffix (``object_dropped__0``). Neither belongs in front of a human. Idempotent: an
    already-clean label passes through unchanged, so it is safe to re-apply at serve time.
    """
    name = re.sub(r"__\d+$", "", str(field_name))
    if name == "failure":
        return "failure"
    if name.startswith("failure_"):
        name = name[len("failure_") :]
    if "__" in name:
        name = name.rsplit("__", 1)[-1]  # drop a module prefix, e.g. mdp__time_out
    return name


__all__ = ["failure_reason_label"]
