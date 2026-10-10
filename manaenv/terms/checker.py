"""Checker expressions — the one boolean-composition primitive for task logic.

A *checker* is the atom: ``func(env, **params) -> bool tensor`` of shape
``[num_envs]``. Leaf checkers are the predicates in :mod:`manaenv.terms.terminations`,
resolved by name through discovery. The three operators **AND / OR / NOT**
combine checkers, and a combined checker is *itself* a checker (same
``func(env) -> bool`` shape) — so expressions nest arbitrarily and **any** task
signal ("success", "failure", an evaluation aggregate, a graph-stage predicate)
is just a *named checker expression*, never a privileged, separately-discovered
thing.

This module is the single source of truth for that composition. It collapses
the three former duplicate AND/OR combiners — ``terminations.merged``,
``TerminationComponent._make_combined_term``, and
``EvaluationRuntime._collapse_to_combined_leaf`` — into one evaluator, and adds
the previously-missing NOT.

Discovery is for **leaves only**: the operators here are structural (referenced
by direct function object, never resolved by name), keeping the discovery
namespace clean — exactly one kind of thing is discovered, the checker.

Pure torch; no IsaacLab import → stability-testable with stub carriers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

# The complete, fixed operator set. Boolean-complete (NOT + AND/OR) — any
# relation between checkers is expressible; three names (not the minimal basis)
# because tasks are described in AND/OR/NOT, readability over minimality.
AND = "and"
OR = "or"
NOT = "not"
OPS = (AND, OR, NOT)

# Sentinel for "score/label not provided" — distinct from any falsy real value.
_UNSET: Any = object()


@dataclass
class CheckerSpec:
    """The typed face of a checker term — *(subject, state, requirement)*, parsed once.

    ``func_name`` names the state predicate (a leaf checker resolved by name);
    ``params`` carries its requirement args — the subject ``@role`` reference(s)
    plus thresholds. ``op`` (one of :data:`OPS`) together with ``children``
    expresses composition: a *leaf* has ``children is None``; a *composite*
    combines its children under ``op``. ``time_out`` and ``score`` / ``label`` are
    term metadata that ride alongside.

    This is what the termination component reads after a single parse, instead of
    poking an untyped ``Dict[str, Any]`` of magic keys term-by-term.
    """

    name: str
    func_name: str
    params: dict = field(default_factory=dict)
    time_out: bool = False
    op: str = AND
    children: list[CheckerSpec] | None = None
    score: Any = _UNSET
    label: Any = _UNSET
    latch: bool = False

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    def meta_params(self) -> dict:
        """``_score`` / ``_label`` annotations to ride on a built carrier's params."""
        out: dict = {}
        if self.score is not _UNSET:
            out["_score"] = self.score
        if self.label is not _UNSET:
            out["_label"] = self.label
        return out


def _eval(env: Any, checker: Any) -> torch.Tensor:
    """Evaluate one checker carrier (duck-typed ``.func`` / ``.params``) to bool."""
    return checker.func(env, **getattr(checker, "params", {})).to(dtype=torch.bool).reshape(-1)


def combine(env: Any, *, op: str, children: list, **_meta: Any) -> torch.Tensor:
    """Evaluate an AND/OR/NOT node over child checkers — a checker itself.

    ``children`` are checker carriers (leaves or further ``combine`` nodes), so
    this recurses naturally into arbitrarily nested boolean expressions. Extra
    ``_meta`` (e.g. ``_score`` / ``_label``) rides along on the carrier params
    for downstream scoring and is ignored here.
    """
    if op == NOT:
        if len(children) != 1:
            raise ValueError(f"NOT checker takes exactly one child, got {len(children)}")
        return ~_eval(env, children[0])
    if op == OR:
        out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for child in children:
            out = out | _eval(env, child)
        return out
    if op == AND:
        out = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for child in children:
            out = out & _eval(env, child)
        return out
    raise ValueError(f"unknown checker op {op!r}; expected one of {OPS}")


def make_checker(term_cls: type, op: str, children: list, **meta: Any) -> Any:
    """Build a combined-checker carrier — closed: the result is itself a checker.

    ``term_cls`` (injected, e.g. ``TerminationTermCfg``) keeps this module
    sim-free. ``meta`` carries optional ``_score`` / ``_label`` annotations.
    """
    if op not in OPS:
        raise ValueError(f"unknown checker op {op!r}; expected one of {OPS}")
    if op == NOT and len(children) != 1:
        raise ValueError(f"NOT checker takes exactly one child, got {len(children)}")
    return term_cls(func=combine, params={"op": op, "children": list(children), **meta})
