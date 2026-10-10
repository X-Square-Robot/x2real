"""Recursive composite-task getters — the runtime form of "ManaTask is a graph".

A composite ManaTask holds live child ManaTask objects (via its ``ManaGraph``)
and answers the Arena env-builder getters by recursively delegating to the
children. Because children are live ManaTask objects, subclass identity is
preserved and a child may itself be composite (graph-in-graph).

Composition model (user-fixed): graph nodes are **fully independent** tasks.
Non-gating sections (scene/observation/events/rewards/...) merge by
**dedup / coexist / raise** (``section_merge``): a field shared across nodes is
kept once when identical, differently-named fields coexist, and the same field
name with different content raises (explicit resolution — no auto-suffix, no
silent last-wins). Temporal/gated **success** is NOT merged here; it lives in
the DAG-success machine (wired in the gating getters, next), because even
identical predicates must not collapse when separated by graph ordering.

This mixin lives manaenv-side (not the Arena base ManaTask): the merge needs
manaenv-layer primitives and Arena never imports manaenv. The forged composite
type (``graph_task_forge``) mixes this in over ManaTask.

Children are read via ``_composite_children`` — ``list[(node_id, child)]`` in
graph-solve order — so the mixin is unit-testable with stub children (no sim).
"""

from __future__ import annotations

from typing import Any

from manaenv.pipeline.generators.task.dag_success import assemble_dag_stage_terms
from manaenv.pipeline.generators.task.section_merge import combine_dedup_sections, dedup_metric_list

# Tags a composite child's single gated failure, which has no reason-name of its own — the
# parent names it after the node it occupies. A sentinel object rather than a string, so a
# leaf that declares a term literally named ``failure`` can never collide with it.
_GRAPH_FAILURE = object()


class RecursiveCompositeTaskMixin:
    """Recursive-delegation getters for a composite ManaTask.

    Requires ``self._composite_children: list[tuple[str, ManaTask]]`` (node-id,
    child) in graph-solve order and ``self._composite_edges: list[tuple[str, str]]``
    (dependency edges). Place BEFORE ManaTask in the MRO so these override the
    base single-task getters.
    """

    _composite_children: list[tuple[str, Any]]
    _composite_edges: list[tuple[str, str]] = []
    # Per-node evaluation weights from graph YAML ``score:`` attrs.
    _composite_node_scores: dict[str, float] = {}
    _dag_extras_key: str = "dag_stage_done"
    _dag_active_extras_key: str = "dag_stage_active"
    _DAG_RESET_FIELD = "dag_success_reset"
    # Single graph-level runtime-context consumer event; child copies are stripped.
    _TASK_RUNTIME_CONSUMER_FIELD = "task_runtime__consumer"

    # -- internals ----------------------------------------------------

    def _children(self) -> list[tuple[str, Any]]:
        return list(getattr(self, "_composite_children", None) or [])

    def _is_leaf(self) -> bool:
        """An EMPTY graph — the recursion's base case, not a lesser kind of task.

        Every ManaTask carries a graph; a task that declares no stages simply carries an
        empty one. That is what lets this mixin sit on EVERY task: with children it recurses
        (composing, gating, projecting); with none, each getter yields the leaf's own answer,
        byte for byte as before. So there is no second task type to branch on — here or
        downstream.

        Two sources for that leaf answer, and the split is not arbitrary: where the Arena
        base already implements the getter, we defer to ``super()``. Where it never had one
        (``get_failure_term_cfg``, ``get_stage_timeout_steps``, ``get_log_producers``,
        ``make_runtime_context_consumer``) the mixin carries the leaf semantics itself,
        reading the same ``runtime_cfgs`` / ``definition`` slots the old duck-typed probes
        used to reach around for. ``super()`` there would raise AttributeError.
        """
        return not self._children()

    def _dedup_combine(self, getter_name: str, combined_name: str) -> Any:
        if self._is_leaf():
            return getattr(super(), getter_name)()
        per_node = [(node_id, getattr(child, getter_name)()) for node_id, child in self._children()]
        return combine_dedup_sections(per_node, name=combined_name)

    def _termination_term_cls(self):  # overridable for sim-free tests
        from isaaclab.managers import TerminationTermCfg

        return TerminationTermCfg

    def _event_term_cls(self):  # overridable for sim-free tests
        from isaaclab.managers import EventTermCfg

        return EventTermCfg

    @staticmethod
    def _child_failure_specs(node_id: str, child) -> list:
        """This child's labeled failure predicates as ``[(reason_label, cfg), ...]``.

        A COMPOSITE child contributes the one gated failure its own machine races; only the
        parent knows which node it occupies, so only the parent can name it. A LEAF
        contributes its own labeled predicates verbatim — "global" failures (object_dropped,
        …) sit on every stage via the shared base include, so they attribute to whichever
        stage is the frontier when they fire.

        Both answers come from ``child._failure_specs()``: every child is a graph task, so
        there is nothing to duck-type here.
        """
        return [
            ((f"{node_id}_failure" if label is _GRAPH_FAILURE else label), cfg) for label, cfg in child._failure_specs()
        ]

    def _dag_terms(self):
        """Lazily build + cache ``(machine, success_term, failure_term, reset_event)``.

        Built lazily (not at instancing) so the assembly factory's deepcopy of
        the composite never clones an initialized machine. success_term,
        failure_term and reset_event share ONE machine → the frontier advances +
        resets in lockstep, and Success/Failure are two faces of the same stage
        state (never two boxes syncing a matrix). Recurses naturally: a composite
        child's ``get_success_term_cfg`` / ``get_failure_term_cfg`` return its own
        gated terms. ``failure_term`` is None when no stage declares a failure
        predicate or a timeout.
        """
        cached = getattr(self, "_dag_terms_cache", None)
        if cached is not None:
            return cached
        children = self._children()
        node_ids = [node_id for node_id, _ in children]
        child_success_cfgs = [child.get_success_term_cfg() for _node_id, child in children]
        child_failure_specs = [self._child_failure_specs(node_id, child) for node_id, child in children]
        stage_timeouts = [child.get_stage_timeout_steps() for _node_id, child in children]
        edges = list(getattr(self, "_composite_edges", []) or [])
        cached = assemble_dag_stage_terms(
            node_ids,
            child_success_cfgs,
            edges,
            termination_term_cls=self._termination_term_cls(),
            event_term_cls=self._event_term_cls(),
            child_failure_specs=child_failure_specs,
            stage_timeouts=stage_timeouts,
            extras_key=self._dag_extras_key,
            active_extras_key=self._dag_active_extras_key,
        )
        self._dag_terms_cache = cached
        return cached

    # -- non-gating getters (recursive delegate, dedup/coexist/raise) -

    def get_scene_cfg(self) -> Any:
        return self._dedup_combine("get_scene_cfg", "SceneCfg")

    def get_observation_cfg(self) -> Any:
        return self._dedup_combine("get_observation_cfg", "ObservationCfg")

    def get_events_cfg(self) -> Any:
        # Children's events (dedup/coexist) PLUS this level's DAG-success reset
        # event so the gated frontier clears each episode in lockstep with the
        # success term. The reset field name is suffixed with the machine's unique
        # id: a NESTED composite's child events already carry their own
        # ``dag_success_reset_*`` field, and every level needs its OWN reset to fire
        # (each level has an independent frontier), so the names must not collide.
        # PLUS graph-level runtime events (vNext, from ``runtime_cfgs``) and exclude
        # child runtime-context-consumer copies (one graph-level consumer replaces them).
        if self._is_leaf():
            return super().get_events_cfg()
        machine, _success_term, _failure_term, reset_event = self._dag_terms()
        reset_field = f"{self._DAG_RESET_FIELD}_{machine._state_uuid[:8]}"
        per_node = [(node_id, child.get_events_cfg()) for node_id, child in self._children()]
        extra_fields: list[tuple] = [(reset_field, type(reset_event), reset_event)]
        extra_fields.extend(self._own_event_term_fields())
        return combine_dedup_sections(
            per_node,
            name="EventsCfg",
            exclude_fields=frozenset({self._TASK_RUNTIME_CONSUMER_FIELD}),
            extra_fields=extra_fields,
        )

    def _own_event_term_fields(self) -> list[tuple]:
        """Extract ``(name, type, value)`` specs from the composite's own events.

        Graph-level vNext events live in ``self.runtime_cfgs["event_terms"]`` after
        finalize. Missing events return an empty list.
        """
        runtime_cfgs = getattr(self, "runtime_cfgs", None)
        if not isinstance(runtime_cfgs, dict):
            return []
        event_terms = runtime_cfgs.get("event_terms")
        if event_terms is None:
            return []
        dataclass_fields = getattr(event_terms, "__dataclass_fields__", None)
        names = list(dataclass_fields) if isinstance(dataclass_fields, dict) else list(vars(event_terms))
        out: list[tuple] = []
        seen: set[str] = set()
        for raw_name in names:
            field_name = str(raw_name)
            if field_name in seen or not hasattr(event_terms, field_name):
                continue
            value = getattr(event_terms, field_name)
            out.append((field_name, type(value), value))
            seen.add(field_name)
        return out

    def get_rewards_cfg(self) -> Any:
        return self._dedup_combine("get_rewards_cfg", "RewardsCfg")

    def get_curriculum_cfg(self) -> Any:
        return self._dedup_combine("get_curriculum_cfg", "CurriculumCfg")

    def get_commands_cfg(self) -> Any:
        return self._dedup_combine("get_commands_cfg", "CommandsCfg")

    def get_domain_randomization_cfg(self) -> Any:
        return self._dedup_combine("get_domain_randomization_cfg", "DomainRandomizationCfg")

    def get_metrics(self) -> list[Any]:
        if self._is_leaf():
            return super().get_metrics()
        per_node = [(node_id, child.get_metrics() or []) for node_id, child in self._children()]
        return dedup_metric_list(per_node)

    # -- composite-level scalars --------------------------------------

    def get_prompt(self) -> str:
        # The DAG owns its prompt: the composite's own ``cfg.prompt`` (a
        # Compile->Instance->Finalize product carried from ``definition.prompt``)
        # is the task-level goal and wins. Only when the composite declares no
        # prompt do we fall back to the first non-empty child prompt (a 1-node /
        # node-authored degenerate case).
        if self._is_leaf():
            return super().get_prompt()
        own = getattr(getattr(self, "cfg", None), "prompt", "") or ""
        if own:
            return own
        for _node_id, child in self._children():
            prompt = child.get_prompt()
            if prompt:
                return prompt
        return ""

    def get_episode_length_s(self) -> float | None:
        # Same precedence as ``get_prompt``: an explicit composite-level length
        # (task YAML, or a rollout order override) is the task-level budget and
        # wins. Only when the composite declares none do we derive it from the
        # longest child stage.
        if self._is_leaf():
            return super().get_episode_length_s()
        own = getattr(getattr(self, "cfg", None), "episode_length_s", None)
        if own is not None:
            return float(own)
        lengths = [child.get_episode_length_s() for _node_id, child in self._children()]
        lengths = [length for length in lengths if length is not None]
        return max(lengths) if lengths else None

    def get_default_visualizer_cfg(self) -> Any:
        """Return the task camera hint using IsaacLab's native visualizer config."""
        if self._is_leaf():
            visualizer_cfg = self.runtime_cfgs.get("default_visualizer_cfg")
            return None if isinstance(visualizer_cfg, dict) else visualizer_cfg
        _node_id, first = self._children()[0]
        get_visualizer_cfg = getattr(first, "get_default_visualizer_cfg", None)
        if callable(get_visualizer_cfg):
            return get_visualizer_cfg()
        runtime_cfgs = getattr(first, "runtime_cfgs", None)
        visualizer_cfg = runtime_cfgs.get("default_visualizer_cfg") if isinstance(runtime_cfgs, dict) else None
        return None if isinstance(visualizer_cfg, dict) else visualizer_cfg

    def modify_env_cfg(self, env_cfg: Any) -> Any:
        # Apply each child's modify_env_cfg in solve order (identity by default).
        if self._is_leaf():
            return super().modify_env_cfg(env_cfg)
        for _node_id, child in self._children():
            env_cfg = child.modify_env_cfg(env_cfg)
        return env_cfg

    # -- gating getters (DAG-success composite term) ------------------
    # Temporal success across nodes is gated by graph edges into a single
    # stateful DAG-success term (forced by IsaacLab's flat TerminationManager).
    # Each child's own `success` field is stripped and replaced by the composite
    # `success`; the children's non-success terminations (e.g. time_out,
    # object_dropped) are merged by dedup/coexist/raise.

    def get_termination_cfg(self) -> Any:
        if self._is_leaf():
            return super().get_termination_cfg()
        _machine, success_term, failure_term, _reset = self._dag_terms()
        per_node = [(node_id, child.get_termination_cfg()) for node_id, child in self._children()]
        # Strip each child's own gated `success`/`failure` (they are folded into
        # the ONE stage machine) and inject the composite terms. `failure` is
        # injected only when some stage declared a failure predicate or timeout.
        extra_fields = [("success", type(success_term), success_term)]
        if failure_term is not None:
            extra_fields.append(("failure", type(failure_term), failure_term))
        return combine_dedup_sections(
            per_node,
            name="TerminationsCfg",
            exclude_fields=frozenset({"success", "failure"}),
            exclude_prefixes=("failure_",),
            extra_fields=extra_fields,
        )

    def get_success_term_cfg(self) -> Any:
        if self._is_leaf():
            return super().get_success_term_cfg()
        _machine, success_term, _failure_term, _reset = self._dag_terms()
        return success_term

    def get_failure_term_cfg(self) -> Any:
        # Symmetric to get_success_term_cfg: the composite's gated failure term
        # (None when no stage declares a failure/timeout), so a PARENT graph can
        # embed this composite as a stage and race its success vs failure.
        #
        # A leaf has no single gated term — it has a LIST of labeled predicates
        # (``_failure_specs``), which is a different shape and cannot be squeezed
        # through this getter. Its failures still terminate it: they stay in its own
        # ``get_termination_cfg`` as ``failure_*`` terms.
        if self._is_leaf():
            return None
        _machine, _success_term, failure_term, _reset = self._dag_terms()
        return failure_term

    def _failure_specs(self) -> list[tuple[Any, Any]]:
        """This task's failure predicates as ``[(reason_label, cfg), ...]``, for a PARENT's machine.

        A LEAF's list is what the task factory collected from its non-success termination terms
        (``failure_term_cfgs``) — plain string labels that become the reported failure reason.
        A COMPOSITE contributes the ONE gated failure its own machine races, tagged with the
        :data:`_GRAPH_FAILURE` sentinel because it has no reason-name of its own: the parent
        names it after the node it occupies. The sentinel is an object, not the string
        ``"failure"``, so a leaf that legitimately declares a term literally named ``failure``
        can never be mistaken for a composite.
        """
        if not self._is_leaf():
            term = self.get_failure_term_cfg()
            return [(_GRAPH_FAILURE, term)] if term is not None else []
        runtime_cfgs = getattr(self, "runtime_cfgs", None)
        specs = runtime_cfgs.get("failure_term_cfgs") if isinstance(runtime_cfgs, dict) else None
        if specs:
            return [(str(label), cfg) for label, cfg in specs]
        # Direct task generation may retain the failure-prefixed fields only in
        # the assembled termination config. Recover those fields here so a
        # composite folds them into its single DAG failure machine rather than
        # leaving a parallel flat termination term behind.
        try:
            termination_cfg = super().get_termination_cfg()
        except AttributeError:
            return []
        names = getattr(termination_cfg, "__dataclass_fields__", {})
        return [
            (str(field_name).removeprefix("failure_"), getattr(termination_cfg, field_name))
            for field_name in names
            if str(field_name).startswith("failure_") and hasattr(termination_cfg, field_name)
        ]

    def get_stage_timeout_steps(self) -> Any:
        # A composite stage has no single dwell budget — its internal stages carry
        # their own. Its failure is expressed via get_failure_term_cfg instead.
        # A leaf's budget is the ``stage_timeout_steps`` scalar on its definition.
        if not self._is_leaf():
            return None
        definition = getattr(self, "definition", None)
        if isinstance(definition, dict) and definition.get("stage_timeout_steps") is not None:
            return int(definition["stage_timeout_steps"])
        return None

    def get_evaluation_term_cfg(self) -> Any:
        # An empty graph declares no stages, so there is nothing to project: the task's
        # OWN evaluation terms are its milestones and are returned untouched. (This is
        # what keeps a stage-less task's rubric — e.g. sort_blocks' four scored placements,
        # 10 points — from being replaced by a single 1-point "the task finished" leaf.)
        if self._is_leaf():
            return super().get_evaluation_term_cfg()
        # Project each graph stage into one scored evaluation-rubric leaf, scoring
        # the SAME gated per-stage truth the DAG-success term publishes
        # (``extras["dag_stage_done"][:, i]``) rather than each stage's raw
        # predicate. This is load-bearing for stages like "retract" whose raw
        # predicate (arms-at-home) is already true at episode start: scoring the
        # gated column means a stage only counts once its graph predecessors have
        # latched. ``EvaluationRuntime`` latches each leaf's hit + first_hit_frame
        # per episode (labelled by node id) and ``build_summary`` aggregates them
        # into per-stage pass rates (到位率/抓取率/回收率). Column ``i`` matches
        # ``self._children()[i]`` — both derive from the same node-id order the
        # machine is built with in ``_dag_terms``.
        from manaenv.terms import terminations

        term_cls = self._termination_term_cls()
        sub_defs = []
        node_scores = getattr(self, "_composite_node_scores", None) or {}
        for stage_index, (node_id, _child) in enumerate(self._children()):
            leaf = term_cls(
                func=terminations.dag_stage_latched,
                params={
                    "stage_index": stage_index,
                    "extras_key": self._dag_extras_key,
                    "_label": node_id,
                    "_score": float(node_scores.get(node_id, 1.0)),
                },
            )
            sub_defs.append({"cfg": leaf})
        if not sub_defs:
            return None
        return term_cls(func=terminations.evaluation, params={"_sub_term_defs": sub_defs})

    # -- subtask logging (graph + per-stage) --------------------------

    def get_log_producers(self) -> list[dict[str, Any]]:
        """Collect graph-level and per-stage log producers.

        Stage triggers are rewritten onto DAG extras because child ``success`` terms
        are replaced by the composite DAG-success term: ``trigger: success`` keys off
        the latched ``dag_stage_done`` column, ``enter`` off the ``dag_stage_active``
        rising edge.
        """
        own_specs = [dict(spec) for spec in (self._own_log_producer_specs())]
        if self._is_leaf():
            return own_specs
        specs: list[dict[str, Any]] = own_specs
        for stage_index, (_node_id, child) in enumerate(self._children()):
            for spec in child.get_log_producers():
                rewritten = dict(spec)
                if rewritten.get("trigger") == "success":
                    rewritten.pop("trigger", None)
                    rewritten["stage_index"] = stage_index
                    rewritten["extras_key"] = self._dag_extras_key
                elif rewritten.pop("enter", False):
                    rewritten["stage_index"] = stage_index
                    rewritten["extras_key"] = self._dag_active_extras_key
                specs.append(rewritten)
        return specs

    def _own_log_producer_specs(self) -> list[dict[str, Any]]:
        """This task's OWN declared log producers (graph-level for a composite, the task's
        own for a leaf) — the same ``runtime_cfgs`` slot in both cases."""
        runtime_cfgs = getattr(self, "runtime_cfgs", None)
        if not isinstance(runtime_cfgs, dict):
            return []
        return list(runtime_cfgs.get("log_producers") or [])

    # -- runtime-context consumer (schema-vNext reset dispatch) -------

    def make_runtime_context_consumer(self) -> Any:
        """Aggregate this task's and its children's runtime-context consumers into one event.

        A leaf has no children to aggregate, so it is simply its own consumer when it declares
        the hooks — which is the rule the composite applies to itself anyway. Same rule, one node.
        """
        from manaenv.pipeline.generators.scene_layout_solver.manaenv_bridge.task_runtime_consumer import (
            resolve_task_runtime_consumers,
        )

        consumers: list[Any] = []
        seen: set[int] = set()
        # Only real reset/step hooks make a task its own consumer.
        if any(callable(getattr(self, name, None)) for name in ("on_reset", "on_benchmark_step")):
            consumers.append(self)
            seen.add(id(self))
        ri_consumer = self._runtime_instruction_consumer()
        if ri_consumer is not None:
            consumers.append(ri_consumer)
            seen.add(id(ri_consumer))
        for _node_id, child in self._children():
            for consumer in resolve_task_runtime_consumers(child):
                key = id(consumer)
                if key in seen:
                    continue
                consumers.append(consumer)
                seen.add(key)
        return tuple(consumers) if consumers else None

    def _runtime_instruction_consumer(self) -> Any:
        """Publish ``definition.runtime_instruction`` once place_beside (etc.) is sampled."""
        definition = getattr(getattr(self, "cfg", None), "definition", None) or {}
        if not isinstance(definition, dict):
            return None
        spec = definition.get("runtime_instruction")
        if not isinstance(spec, dict) or not str(spec.get("template") or "").strip():
            return None
        from manaenv.pipeline.runtime.runtime_instruction import RuntimeInstructionConsumer

        return RuntimeInstructionConsumer(plan_id=str(spec.get("plan_id") or "runtime"))
