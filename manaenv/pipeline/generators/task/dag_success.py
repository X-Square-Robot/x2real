"""DAG stage state machine — gated, recursive composite-task Success AND Failure.

IsaacLab's ``TerminationManager.compute()`` is a flat unconditional OR over all
terms each step — it cannot express "evaluate stage N only after stage N-1 is
done". The only way to express ordered/gated outcomes is a single **stateful**
termination term whose func holds a per-env frontier over the task graph and
recursively calls each child stage's predicate.

This module is that state machine, kept **pure** (torch only, no IsaacLab
import) and **predicate-agnostic**: it calls whatever child success/failure
callables it is given, gated by graph edges. The per-stage *meaning* lives in
the child tasks' own predicates (authored per node), not here — so this is
stability-testable now with stub child funcs.

**The stage is the unit; Success and Failure are two faces of it, on ONE shared
frontier.** At each frontier stage the machine races that stage's ``success``
(→ latch & advance) against its ``failure`` (→ terminate + record which stage,
why). Composite success = all sink nodes latched (and no failure); composite
failure = any frontier stage failed. Per-stage timeout is just a Failure kind: a
machine-native dwell budget (steps a stage has been the active frontier without
latching). Success wins if both fire the same step. There is exactly ONE machine
and ONE per-env state transition — success and failure never live in two boxes
that sync a shared matrix.

Generalizes Arena's ``sequential_task_success_func`` (a linear cursor) to a DAG
frontier: a node becomes evaluable only once all its predecessors are latched
done; composite success = all nodes latched done. This gating also prevents the
false early-latch that a flat ``mode: and`` aggregate suffers (e.g. a
"return-home" stage looking satisfied at episode start) — a stage is not even
evaluated until its predecessors complete.

State lives on ``env`` keyed by a per-machine uuid (configclass deep-copies the
term defaults, so the success term, failure term and reset event each get an
independent *copy* of the machine sharing the same uuid → all resolve to the one
env state slot and advance/reset in lockstep). The machine is built at env-build
time (after the assembly deepcopy-clone), so no initialized state is ever cloned.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

import torch

# ONE vocabulary, one function — re-exported from a torch-free module so the serving layer can
# clean stored-report labels without importing torch. The machine and the eval runtime derive
# reasons from this same call, so "machine said time_out" and "report shows mdp__time_out" can
# never diverge. See manaenv/pipeline/generators/task/failure_labels.py.
from manaenv.pipeline.generators.task.failure_labels import failure_reason_label  # noqa: F401 (re-export)
from manaenv.pipeline.runtime.runtime_state import runtime_state

# A child predicate callable takes the env and returns a bool tensor [num_envs].
ChildPredicateFunc = Callable[[object], torch.Tensor]
ChildSuccessFunc = ChildPredicateFunc  # back-compat alias


def build_predecessor_indices(node_ids: list[str], edges: list[tuple[str, str]]) -> list[list[int]]:
    """For each node (by position in *node_ids*), the indices of its predecessors.

    An edge ``(src, dst)`` means ``dst`` depends on ``src`` (src must complete
    first), so ``src`` is a predecessor of ``dst``.
    """
    index_of = {node_id: i for i, node_id in enumerate(node_ids)}
    preds: list[list[int]] = [[] for _ in node_ids]
    for src, dst in edges:
        if src not in index_of or dst not in index_of:
            raise ValueError(f"DAG edge ({src!r},{dst!r}) references an unknown node")
        preds[index_of[dst]].append(index_of[src])
    return preds


def build_sink_indices(predecessor_indices: list[list[int]]) -> list[int]:
    """Indices of the sink nodes — those that are no other node's predecessor.

    Composite completion keys off the **sinks** (no-outgoing-edge nodes), not
    "all nodes". This is the correct, minimal condition: in a DAG every node is
    an ancestor of some sink, and gating latches a sink only after all its
    ancestors, so "all sinks done" already implies the whole graph is done —
    while leaving room for future optional/branch nodes that a blanket
    "all nodes" condition would wrongly force. An empty result means every node
    is someone's predecessor, i.e. a cycle (not a DAG).
    """
    is_predecessor: set[int] = set()
    for preds in predecessor_indices:
        is_predecessor.update(preds)
    return [i for i in range(len(predecessor_indices)) if i not in is_predecessor]


class DagStageMachine:
    """Per-env gated frontier over a task graph; the stage is the unit, Success
    and Failure are two faces evaluated on ONE shared frontier.

    ``child_funcs[i]`` is the success callable for node ``node_ids[i]``;
    ``failure_funcs[i]`` (optional) its failure callable; ``stage_timeouts[i]``
    (optional int) a machine-native dwell budget whose expiry is a Failure;
    ``predecessor_indices[i]`` are the indices that must be latched done before
    node ``i`` is evaluated. Composite success = "all **sink** nodes latched
    done AND not failed" (see :func:`build_sink_indices`); composite failure =
    "any frontier stage failed". Bind :meth:`evaluate` as the composite
    ``success`` termination func, :meth:`evaluate_failure` as the composite
    ``failure`` termination func, and :meth:`reset` as a ``reset``-mode event
    func — all three share this one machine (state on ``env`` keyed by uuid).

    When no ``failure_funcs`` and no ``stage_timeouts`` are given the machine is
    a pure success machine, byte-for-byte identical to the pre-failure behavior.

    State lives in the env's unified :class:`RuntimeStateRegistry`
    (``manaenv.pipeline.runtime.runtime_state``) under ``task.dag.<uuid>.*`` slots
    — one home for reset / introspection / monitoring, shared with terms and (in
    future) scene state. The uuid is copied-by-value through the configclass
    deepcopy, so all machine copies address the same registry slots.
    """

    _TIMEOUT_REASON = "timeout"

    def __init__(
        self,
        node_ids: list[str],
        child_funcs: list[ChildSuccessFunc],
        predecessor_indices: list[list[int]],
        *,
        failure_specs: list[list[tuple[str, ChildPredicateFunc]] | None] | None = None,
        stage_timeouts: list[int | None] | None = None,
        extras_key: str = "dag_stage_done",
        failed_extras_key: str = "dag_failed_stage",
        reason_extras_key: str = "dag_failed_reason",
        active_extras_key: str = "dag_stage_active",
    ) -> None:
        if not (len(node_ids) == len(child_funcs) == len(predecessor_indices)):
            raise ValueError("DagStageMachine: node_ids / child_funcs / predecessor_indices length mismatch")
        if not node_ids:
            raise ValueError("DagStageMachine requires at least one node")
        self.node_ids = list(node_ids)
        self.child_funcs = list(child_funcs)
        self.predecessor_indices = [list(p) for p in predecessor_indices]
        self.sink_indices = build_sink_indices(self.predecessor_indices)
        if not self.sink_indices:
            raise ValueError("DagStageMachine: no sink nodes (graph has a cycle?)")
        n = len(self.node_ids)
        # ``failure_specs[i]`` = an optional list of (reason_label, predicate) for
        # node i. A "global" failure (e.g. object_dropped) is simply declared on
        # every stage (via the shared base) — it is then evaluated at whichever
        # stage is the frontier when it fires, so attribution to the active stage
        # is automatic with no separate machine-level global concept.
        self.failure_specs: list[list[tuple[str, ChildPredicateFunc]]] = (
            [list(s) if s else [] for s in failure_specs] if failure_specs is not None else [[] for _ in range(n)]
        )
        self.stage_timeouts: list[int | None] = list(stage_timeouts) if stage_timeouts is not None else [None] * n
        if len(self.failure_specs) != n or len(self.stage_timeouts) != n:
            raise ValueError("DagStageMachine: failure_specs / stage_timeouts length mismatch")
        # Reason registry: index 0 reserved for the machine-native dwell timeout,
        # then every distinct failure-spec label. ``extras[reason_extras_key]``
        # publishes a per-env index into this list (-1 = no failure).
        self._reason_labels: list[str] = [self._TIMEOUT_REASON]
        for specs in self.failure_specs:
            for label, _func in specs:
                if label not in self._reason_labels:
                    self._reason_labels.append(label)
        self._reason_index: dict[str, int] = {label: i for i, label in enumerate(self._reason_labels)}
        self._has_failure = any(specs for specs in self.failure_specs) or any(
            t is not None for t in self.stage_timeouts
        )
        self.extras_key = extras_key
        self.failed_extras_key = failed_extras_key
        self.reason_extras_key = reason_extras_key
        # Per-step "stage i is enterable" (all predecessors latched done) matrix,
        # published each step so ``trigger: enter`` subtask logs can key off its
        # rising edge (the moment a stage is entered).
        self.active_extras_key = active_extras_key
        # uuid is copied-by-value through configclass deepcopy → all machine
        # copies key the same env state slot.
        self._state_uuid = uuid.uuid4().hex

    # -- env-side state slots (RuntimeStateRegistry, keyed by machine uuid) --

    def _slot_key(self, suffix: str) -> str:
        return f"task.dag.{self._state_uuid}.{suffix}"

    def _slot_keys(self) -> set[str]:
        return {self._slot_key(s) for s in ("done", "failed", "dwell", "compfail", "reason")}

    def _ensure_state(self, env) -> torch.Tensor:
        return runtime_state(env).tensor(
            self._slot_key("done"),
            shape=(env.num_envs, len(self.node_ids)),
            dtype=torch.bool,
            fill=False,
            owner="task.dag",
            monitor=True,
        )

    def _ensure_failure_state(self, env, done: torch.Tensor):
        reg = runtime_state(env)
        n = done.shape[1]
        failed = reg.tensor(
            self._slot_key("failed"),
            shape=(env.num_envs, n),
            dtype=torch.bool,
            fill=False,
            owner="task.dag",
            monitor=True,
        )
        dwell = reg.tensor(self._slot_key("dwell"), shape=(env.num_envs, n), dtype=torch.long, fill=0, owner="task.dag")
        compfail = reg.tensor(
            self._slot_key("compfail"), shape=(env.num_envs,), dtype=torch.bool, fill=False, owner="task.dag"
        )
        reason = reg.tensor(
            self._slot_key("reason"), shape=(env.num_envs,), dtype=torch.long, fill=-1, owner="task.dag", monitor=True
        )
        return failed, dwell, compfail, reason

    def _peek_state(self, env) -> torch.Tensor | None:
        return runtime_state(env).get_tensor(self._slot_key("done"))

    def active_mask(self, env) -> torch.Tensor:
        """Return ready, unfinished frontier nodes without evaluating predicates.

        This is the planner-facing read surface. It may allocate empty runtime
        slots, but it never calls child success/failure functions, increments
        dwell counters, latches completion, or advances the DAG.
        """
        done = self._ensure_state(env)
        active = torch.empty_like(done)
        ones = torch.ones(env.num_envs, dtype=torch.bool, device=done.device)
        if self._has_failure:
            failed, _dwell, _compfail, _reason = self._ensure_failure_state(env, done)
            any_failed = failed.any(dim=1)
        else:
            any_failed = torch.zeros(env.num_envs, dtype=torch.bool, device=done.device)
        for index, predecessors in enumerate(self.predecessor_indices):
            predecessors_done = done[:, predecessors].all(dim=1) if predecessors else ones
            active[:, index] = predecessors_done & ~done[:, index] & ~any_failed
        return active

    # -- evaluation -------------------------------------------------------

    def evaluate(self, env) -> torch.Tensor:
        """Advance the frontier one step; return composite success [num_envs].

        Composite success = all **sink** nodes latched done AND no stage failed.
        Gating guarantees a sink latches only after all its ancestors, so this
        already implies the whole graph is done for the current (fully-required)
        graphs. When failure is configured, the SAME frontier also races each
        stage's failure predicate + dwell-timeout; ``success`` wins on ties.

        Signature is ``(env)`` only — NO ``**kwargs``: IsaacLab's
        ``TerminationManager`` validator (``manager_base._resolve_common_term_cfg``)
        counts ``**kwargs`` as a phantom mandatory parameter and rejects the term
        at env-build time. The machine takes no params (state lives on ``env``),
        so bare ``(env)`` is both correct and the only signature that validates.
        """
        done = self._ensure_state(env)
        # Per-step enterable matrix — derived + published each step (NOT latched
        # state, so no registry slot). Sources are always enterable.
        active = torch.empty_like(done)
        ones = torch.ones(env.num_envs, dtype=torch.bool, device=done.device)
        failure_state = self._ensure_failure_state(env, done) if self._has_failure else None
        if failure_state is not None:
            failed, dwell, compfail, reason = failure_state
            any_failed = failed.any(dim=1)
        else:
            any_failed = torch.zeros(env.num_envs, dtype=torch.bool, device=done.device)
        for i in range(len(self.node_ids)):
            preds = self.predecessor_indices[i]
            preds_done = done[:, preds].all(dim=1) if preds else ones
            active[:, i] = preds_done
            # A failed env freezes: it stops being ``ready`` for any stage.
            ready = preds_done & ~done[:, i] & ~any_failed
            if not bool(ready.any()):
                continue
            result = self.child_funcs[i](env).to(dtype=torch.bool, device=done.device).reshape(-1)
            new_success = ready & result
            if failure_state is not None:
                # Each cause is (reason_index, bool[num_envs]); processed in order
                # so the FIRST cause a fresh env hits wins its reason. Timeout is
                # reason 0. ``success`` wins on ties (``~new_success``); a stage
                # that already failed this step freezes (``~any_failed``).
                causes: list[tuple[int, torch.Tensor]] = []
                budget = self.stage_timeouts[i]
                if budget is not None:
                    # Dwell = steps this stage has been the active frontier
                    # (only ``ready`` envs tick), so the timeout is per-stage,
                    # per-env correct for any num_envs.
                    dwell[:, i] = torch.where(ready, dwell[:, i] + 1, dwell[:, i])
                    causes.append((self._reason_index[self._TIMEOUT_REASON], dwell[:, i] >= int(budget)))
                for label, func in self.failure_specs[i]:
                    fired = func(env).to(dtype=torch.bool, device=done.device).reshape(-1)
                    causes.append((self._reason_index[label], fired))
                for reason_idx, cause in causes:
                    newly = ready & cause & ~new_success & ~any_failed
                    if bool(newly.any()):
                        failed[:, i] = failed[:, i] | newly
                        reason[newly] = int(reason_idx)
                        any_failed = failed.any(dim=1)
            done[:, i] = done[:, i] | new_success
        # Expose per-stage latched state for diagnostic per-stage success rates
        # + the labels (graph node ids, in matrix-column order) so monitoring can
        # name columns without re-deriving order.
        extras = getattr(env, "extras", None)
        if isinstance(extras, dict):
            extras[self.extras_key] = done.clone()
            extras[f"{self.extras_key}_labels"] = list(self.node_ids)
            if self.active_extras_key:
                extras[self.active_extras_key] = active.clone()
                extras[f"{self.active_extras_key}_labels"] = list(self.node_ids)
            if failure_state is not None:
                # First failed stage index per env (-1 = none) + reason index for
                # attribution ("where + why").
                has = failed.any(dim=1)
                first = torch.argmax(failed.to(torch.int8), dim=1).to(torch.long)
                extras[self.failed_extras_key] = torch.where(has, first, torch.full_like(first, -1))
                extras[f"{self.failed_extras_key}_labels"] = list(self.node_ids)
                extras[self.reason_extras_key] = reason.clone()
                extras[f"{self.reason_extras_key}_labels"] = list(self._reason_labels)
        if failure_state is not None:
            compfail[:] = failed.any(dim=1)
            return done[:, self.sink_indices].all(dim=1) & ~compfail
        return done[:, self.sink_indices].all(dim=1)

    def evaluate_failure(self, env) -> torch.Tensor:
        """Composite failure [num_envs] = any frontier stage has failed.

        Pure reader of the failure state that :meth:`evaluate` (the always-run
        ``success`` term) computes each step; ``reset`` clears it per episode, so
        a fresh episode never inherits a stale failure. At worst one step behind
        :meth:`evaluate` (harmless: delays termination by ≤1 step, never a false
        positive). Returns all-False when failure is not configured / no state.
        """
        if not self._has_failure:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        compfail = runtime_state(env).get_tensor(self._slot_key("compfail"))
        if not isinstance(compfail, torch.Tensor) or compfail.shape[0] != env.num_envs:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return compfail.clone()

    def reset(self, env, env_ids=None, **kwargs) -> None:
        """Clear this machine's latched state (done / failure / dwell / reason)
        for *env_ids* (all envs if None) on episode reset — via the unified
        registry, resetting only this machine's own slots."""
        del kwargs
        runtime_state(env).reset(env_ids, keys=self._slot_keys())


def _bind_child_predicate(term_cfg) -> ChildPredicateFunc:
    """Wrap a child's success/failure TerminationTermCfg into a ``func(env)``.

    Duck-typed on ``.func`` / ``.params`` so no sim-coupled cfg type import is
    needed. A child that is itself composite returns its own gated term here, so
    recursion composes naturally (graph-in-graph).
    """
    func = getattr(term_cfg, "func", None)
    if func is None:
        raise ValueError("child term cfg has no .func")
    params = dict(getattr(term_cfg, "params", {}) or {})

    def _child(env):
        return func(env, **params)

    return _child


# Back-compat alias (was the only binder before failure existed).
_bind_child_success = _bind_child_predicate


def build_dag_success_machine(
    node_ids: list[str],
    child_success_cfgs: list,
    edges: list[tuple[str, str]],
    *,
    child_failure_specs: list | None = None,
    stage_timeouts: list[int | None] | None = None,
    extras_key: str = "dag_stage_done",
    failed_extras_key: str = "dag_failed_stage",
    reason_extras_key: str = "dag_failed_reason",
    active_extras_key: str = "dag_stage_active",
) -> DagStageMachine:
    """Assemble a :class:`DagStageMachine` from child success (+ optional
    labeled failure) term cfgs, edges, and optional per-stage timeout budgets.

    ``child_success_cfgs[i]`` is node ``node_ids[i]``'s ``success`` TerminationTermCfg
    (from ``ManaTask.get_success_term_cfg()``); a ``None`` entry is rejected
    (every gated stage needs a success predicate). ``child_failure_specs[i]`` is
    that node's optional list of ``(reason_label, failure TerminationTermCfg)``
    (``None``/``[]`` = no failure predicate — a "global" failure like
    object_dropped is simply present on every stage's list via the shared base);
    ``stage_timeouts[i]`` its optional dwell budget in steps.
    """
    if len(node_ids) != len(child_success_cfgs):
        raise ValueError("build_dag_success_machine: node_ids / child_success_cfgs length mismatch")
    child_funcs: list[ChildSuccessFunc] = []
    for node_id, cfg in zip(node_ids, child_success_cfgs):
        if cfg is None:
            raise ValueError(f"graph node {node_id!r} has no success term; every gated stage needs one")
        child_funcs.append(_bind_child_predicate(cfg))

    failure_specs: list | None = None
    if child_failure_specs is not None:
        if len(child_failure_specs) != len(node_ids):
            raise ValueError("build_dag_success_machine: node_ids / child_failure_specs length mismatch")
        failure_specs = [
            [(str(label), _bind_child_predicate(cfg)) for label, cfg in (specs or [])] for specs in child_failure_specs
        ]

    preds = build_predecessor_indices(node_ids, edges)
    return DagStageMachine(
        node_ids,
        child_funcs,
        preds,
        failure_specs=failure_specs,
        stage_timeouts=stage_timeouts,
        extras_key=extras_key,
        failed_extras_key=failed_extras_key,
        reason_extras_key=reason_extras_key,
        active_extras_key=active_extras_key,
    )


def assemble_dag_success_terms(
    node_ids: list[str],
    child_success_cfgs: list,
    edges: list[tuple[str, str]],
    *,
    termination_term_cls,
    event_term_cls,
    extras_key: str = "dag_stage_done",
):
    """Build the composite ``success`` termination term + its reset event term.

    Success-only assembler (no failure); kept for back-compat. Returns
    ``(machine, success_term, reset_event_term)``. For the Success+Failure
    stage machine use :func:`assemble_dag_stage_terms`.
    """
    machine = build_dag_success_machine(node_ids, child_success_cfgs, edges, extras_key=extras_key)
    success_term = termination_term_cls(func=machine.evaluate, params={})
    reset_event_term = event_term_cls(func=machine.reset, mode="reset", params={})
    return machine, success_term, reset_event_term


def assemble_dag_stage_terms(
    node_ids: list[str],
    child_success_cfgs: list,
    edges: list[tuple[str, str]],
    *,
    termination_term_cls,
    event_term_cls,
    child_failure_specs: list | None = None,
    stage_timeouts: list[int | None] | None = None,
    extras_key: str = "dag_stage_done",
    failed_extras_key: str = "dag_failed_stage",
    reason_extras_key: str = "dag_failed_reason",
    active_extras_key: str = "dag_stage_active",
):
    """Build the composite ``success`` + ``failure`` termination terms and the
    reset event, all from ONE machine (shared per-env frontier state).

    Returns ``(machine, success_term, failure_term, reset_event_term)``.
    ``failure_term`` is ``None`` when neither any ``child_failure_specs`` nor any
    ``stage_timeouts`` is configured (pure success graph). ``termination_term_cls``
    / ``event_term_cls`` are injected (``TerminationTermCfg`` / ``EventTermCfg``)
    so this assembly is sim-free and unit-testable with stub cfg classes.
    """
    machine = build_dag_success_machine(
        node_ids,
        child_success_cfgs,
        edges,
        child_failure_specs=child_failure_specs,
        stage_timeouts=stage_timeouts,
        extras_key=extras_key,
        failed_extras_key=failed_extras_key,
        reason_extras_key=reason_extras_key,
        active_extras_key=active_extras_key,
    )
    success_term = termination_term_cls(func=machine.evaluate, params={})
    failure_term = termination_term_cls(func=machine.evaluate_failure, params={}) if machine._has_failure else None
    reset_event_term = event_term_cls(func=machine.reset, mode="reset", params={})
    return machine, success_term, failure_term, reset_event_term


# Back-compat alias (renamed from DagSuccessMachine → stage machine does Success+Failure).
DagSuccessMachine = DagStageMachine
