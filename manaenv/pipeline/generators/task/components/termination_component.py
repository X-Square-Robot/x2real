from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG
from manaenv.pipeline.generators.task.term_discovery import resolve_task_term

# The boolean operators ``_combine`` (and the aggregator ``mode`` it maps onto)
# may take — kept inline so this module stays import-light (the canonical set
# lives in ``manaenv.terms.checker.OPS``; this mirrors it for fail-fast authoring).
_VALID_COMBINE_OPS = ("and", "or", "not")


def _validate_combine_op(value: Any) -> str:
    if value not in _VALID_COMBINE_OPS:
        raise ValueError(f"_combine must be one of {_VALID_COMBINE_OPS}, got {value!r}")
    return value


# DAG getters look up ``success`` / ``evaluation`` by unsuffixed name.
_SINGULAR_TERM_NAMES = frozenset({"success", "evaluation"})


def _collapse_pool_window(value: Any) -> Any:
    """``['fruit', 'fruit__slot_1']`` → ``'fruit'``; other values unchanged.

    A schema-vNext cache window is one runtime object. Collapsing it keeps
    pool-aware predicates on the active slot instead of AND-ing parked preloads.
    Distinct names such as ``['bottle_a', 'bottle_b']`` stay a list.
    """
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        return value
    bases = {name.split("__slot_", 1)[0] for name in value}
    if len(bases) != 1:
        return value
    base = next(iter(bases))
    if all(name == base or name.startswith(f"{base}__slot_") for name in value):
        return base
    return value


class TerminationComponent(TaskComponent):
    """Compiles termination YAML into ``TerminationTermCfg`` carriers.

    Each YAML term is parsed *once* into a typed :class:`~manaenv.terms.checker.CheckerSpec`
    (the *(subject, state, requirement)* face of a checker) by :meth:`_parse_spec`,
    then built into carriers by :meth:`_build_term`. Composition uses the shared
    AND/OR/NOT algebra in :mod:`manaenv.terms.checker`; leaf state predicates are
    resolved by name. The reserved keys (``time_out`` / ``_combine`` / ``_func`` /
    ``terms`` / ``_score`` / ``_label``) are read only in the parser; everything
    downstream reads typed ``CheckerSpec`` fields, not the raw dict.

    YAML example::

        termination:
          terms:
            mdp.time_out:
              time_out: true
            success:
              terms:
                - object_position_in_zone:
                    object_cfg: "@pickable"
                    reference_cfg: "@destination"
                    distance_threshold: 0.04
                - object_contact_destination:
                    contact_sensor_cfgs: "@pick_dest_contact_sensor"
                    force_threshold: 1.0
                    consecutive_steps: 5
    """

    EVALUATION_TERM_KEY: str = "evaluation"

    # Magic keys consumed by the parser; everything else is forwarded as params.
    _RESERVED_KEYS = ("time_out", "_combine", "_func", "_score", "_label", "_latch", "terms")

    def generate(self, cfg: TerminationComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        from isaaclab_arena.utils.configclass import make_configclass

        roles = roles or {}
        term_fields: list = []
        combine_modes: dict[str, str] = {}  # term name -> "and"/"or"/"not"
        evaluation_term_cfg: Any | None = None

        for name, raw in cfg.terms.items():
            spec = self._parse_spec(name, raw)
            if name == self.EVALUATION_TERM_KEY and spec.latch:
                # EvaluationRuntime already latches each rubric leaf (first_hit_frame)
                # and consumes the cfg's _sub_term_defs directly, so wrapping the
                # evaluation term would both be redundant and break that structure.
                raise ValueError("termination.terms.evaluation does not support _latch (rubric latching is built in)")
            combine_modes[name] = spec.op
            built_terms = self._build_term(spec, roles)
            if name == self.EVALUATION_TERM_KEY:
                # ``evaluation`` shares the success compile path (sub-terms
                # become _sub_term_defs) but never enters the
                # TerminationManager — EvaluationRuntime consumes the cfg
                # directly via ``ManaTask.get_evaluation_term_cfg``.
                if len(built_terms) != 1:
                    raise ValueError(
                        "termination.terms.evaluation must compile to exactly one cfg "
                        f"(got {len(built_terms)} entries)."
                    )
                evaluation_term_cfg = built_terms[0][1]
                continue
            for key, tc in built_terms:
                field_name = key.replace(".", "__")
                term_fields.append((field_name, type(tc), tc))

        TermCfgCls = make_configclass("TerminationCfg", term_fields)
        termination_terms = TermCfgCls()
        result = {
            "termination_terms": termination_terms,
            "success_term_cfg": getattr(termination_terms, "success", None),
            "evaluation_term_cfg": evaluation_term_cfg,
        }
        # Persist per-term combine modes for post-finalize expansion.
        result["_term_combine_modes"] = combine_modes
        return result

    @classmethod
    def _parse_spec(cls, name: str, raw: dict[str, Any] | None) -> Any:
        """Parse one YAML term (the only place the reserved magic keys are read)."""
        from manaenv.terms import checker

        raw = raw or {}
        children = cls._parse_children(raw["terms"]) if "terms" in raw else None
        func_name = raw.get("_func", "merged" if children is not None else name)
        return checker.CheckerSpec(
            name=name,
            func_name=func_name,
            params={k: v for k, v in raw.items() if k not in cls._RESERVED_KEYS},
            time_out=raw.get("time_out", False),
            op=cls._parse_op(raw, is_group=children is not None),
            children=children,
            score=raw["_score"] if "_score" in raw else checker._UNSET,
            label=raw["_label"] if "_label" in raw else checker._UNSET,
            latch=bool(raw.get("_latch", False)),
        )

    def _make_latched_term(self, child_cfg: Any, combine_mode: str) -> Any:
        """Wrap one momentary term cfg in a first-hit-and-hold carrier."""
        from isaaclab.managers import TerminationTermCfg

        func = resolve_task_term("termination", "latched")
        return TerminationTermCfg(
            func=func,
            params={
                "_sub_term_defs": [{"cfg": child_cfg, "combine_mode": _validate_combine_op(combine_mode)}],
                "state_key": uuid.uuid4().hex,
            },
            time_out=getattr(child_cfg, "time_out", False),
        )

    @classmethod
    def _parse_op(cls, raw: dict[str, Any], *, is_group: bool) -> str:
        """Resolve a term's fold operator from ``_combine``, or a group's ``mode``.

        ``_combine`` wins when present. A group may also spell the operator as the
        aggregator's own ``mode`` param; leaves never do, since ``mode`` there is an
        ordinary predicate arg.
        """
        from manaenv.terms import checker

        if "_combine" in raw:
            return _validate_combine_op(raw["_combine"])
        if is_group and raw.get("mode") in _VALID_COMBINE_OPS:
            return str(raw["mode"])
        return checker.AND

    @classmethod
    def _parse_children(cls, term_specs: Any) -> list[Any]:
        if not isinstance(term_specs, list) or not term_specs:
            raise ValueError("Inline termination 'terms' must be a non-empty list.")
        children: list[Any] = []
        for index, entry in enumerate(term_specs):
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(
                    "Each inline termination entry must be a single-key mapping, " f"but entry #{index} is: {entry!r}"
                )
            child_name, child_raw = next(iter(entry.items()))
            if child_raw is None:
                child_raw = {}
            if not isinstance(child_raw, dict):
                raise ValueError(f"Inline termination '{child_name}' must map to a dict, got: {type(child_raw)!r}")
            children.append(cls._parse_spec(child_name, child_raw))
        return children

    def _build_term(self, spec: Any, roles: dict[str, Any]) -> list[tuple[str, Any]]:
        """Build carriers for one parsed term: a composite -> one carrier, a leaf
        -> one (or several, when an ``@role`` list-param expands per asset)."""
        if not spec.is_leaf:
            term_cfg = self._build_compound(spec, roles)
            if spec.latch:
                term_cfg = self._make_latched_term(term_cfg, spec.op)
            return [(spec.name, term_cfg)]
        built_terms, _ = self._build_leaf_term_cfgs(spec.name, spec.func_name, spec.params, roles, spec.time_out)
        carriers = [cfg for _, cfg in built_terms]
        if (spec.latch or spec.name in _SINGULAR_TERM_NAMES) and carriers:
            child = carriers[0] if len(carriers) == 1 else self._make_combined_term(carriers, mode=spec.op)
            if spec.latch:
                child = self._make_latched_term(child, spec.op)
            return [(spec.name, child)]
        return built_terms

    def _build_compound(self, spec: Any, roles: dict[str, Any]) -> Any:
        from isaaclab.managers import TerminationTermCfg

        func = resolve_task_term("termination", spec.func_name)
        params = self.resolve_roles(spec.params, roles) if roles else dict(spec.params)
        params.update(spec.meta_params())
        # A group's operator folds its own sub-terms, so it must reach the
        # aggregator as ``mode``. Without this, aggregators default to AND and an
        # authored ``_combine: or`` / ``not`` group is silently dropped.
        if "mode" in inspect.signature(func).parameters:
            params["mode"] = spec.op
        params["_sub_term_defs"] = [
            {"cfg": self._compile_child(child, roles), "combine_mode": child.op} for child in spec.children
        ]
        return TerminationTermCfg(func=func, params=params, time_out=spec.time_out)

    def _compile_child(self, spec: Any, roles: dict[str, Any]) -> Any:
        """Compile one inline child into a single carrier, combining the per-asset
        carriers of an expanded leaf under the child's ``op`` and riding any
        ``_score`` / ``_label`` meta on the result."""
        carriers = [cfg for _, cfg in self._build_term(spec, roles)]
        if len(carriers) == 1:
            cfg = carriers[0]
            cfg.params.update(spec.meta_params())
            return cfg
        combined = self._make_combined_term(carriers, mode=spec.op)
        combined.params.update(spec.meta_params())
        return combined

    def _compile_inline_term_cfg(self, func_name: str, spec: dict[str, Any], roles: dict[str, Any]) -> Any:
        """Compile one inline term from its raw ``(name, dict)`` form.

        Thin adapter over :meth:`_parse_spec` + :meth:`_compile_child` for callers
        that still hold the unparsed YAML mapping.
        """
        return self._compile_child(self._parse_spec(func_name, spec), roles)

    def _build_leaf_term_cfgs(
        self,
        term_name: str,
        func_name: str,
        raw_params: dict[str, Any],
        roles: dict[str, Any],
        time_out: bool,
    ) -> tuple[list[tuple[str, Any]], bool]:
        from isaaclab.managers import TerminationTermCfg

        resolved = self.resolve_roles(raw_params, roles) if roles else dict(raw_params)
        for key, raw_value in raw_params.items():
            if isinstance(raw_value, str) and raw_value.startswith("@") and not key.endswith("_cfgs"):
                resolved[key] = _collapse_pool_window(resolved.get(key))
        expandable = {
            key: resolved[key]
            for key, raw_value in raw_params.items()
            if isinstance(raw_value, str)
            and raw_value.startswith("@")
            and isinstance(resolved.get(key), list)
            and resolved[key]
            and not key.endswith("_cfgs")
        }
        func = resolve_task_term("termination", func_name)
        leaf_params = dict(resolved)
        for meta_key in ("_score", "_label"):
            leaf_params.pop(meta_key, None)

        if not expandable:
            return [
                (
                    term_name,
                    TerminationTermCfg(
                        func=func,
                        params=leaf_params,
                        time_out=time_out,
                    ),
                )
            ], False

        expand_count = self._infer_expand_count(func_name, expandable)
        built_terms: list[tuple[str, Any]] = []
        for index in range(expand_count):
            expanded_params = dict(resolved)
            for key, values in expandable.items():
                expanded_params[key] = self._select_expanded_param_value(key, values, index)
            for meta_key in ("_score", "_label"):
                expanded_params.pop(meta_key, None)
            built_terms.append(
                (
                    term_name if expand_count == 1 else f"{term_name}__{index}",
                    TerminationTermCfg(
                        func=func,
                        params=expanded_params,
                        time_out=time_out,
                    ),
                )
            )
        return built_terms, True

    @staticmethod
    def _infer_expand_count(func_name: str, expandable: dict[str, list]) -> int:
        non_singleton_lengths = {len(values) for values in expandable.values() if len(values) > 1}
        if len(non_singleton_lengths) > 1:
            role_lengths = {key: len(values) for key, values in expandable.items()}
            raise ValueError(f"Termination term '{func_name}' has incompatible list-role lengths: {role_lengths}")
        if non_singleton_lengths:
            return non_singleton_lengths.pop()
        return len(next(iter(expandable.values())))

    @staticmethod
    def _select_expanded_param_value(key: str, values: list, index: int) -> Any:
        selected = values[0] if len(values) == 1 else values[index]
        if key.endswith("_cfgs"):
            return selected if isinstance(selected, list) else [selected]
        return selected

    @staticmethod
    def _make_combined_term(term_cfgs: list, mode: str = "and") -> Any:
        """Combine multiple term cfgs into one checker via the shared algebra.

        Thin wrapper over :func:`manaenv.terms.checker.make_checker` (the single
        composition primitive). ``mode`` is ``and`` / ``or`` / ``not``; the
        result is itself a checker carrier, so it nests like any leaf. Callers
        may add ``_score`` / ``_label`` onto the returned ``params`` afterward.
        """
        from isaaclab.managers import TerminationTermCfg

        from manaenv.terms import checker

        return checker.make_checker(TerminationTermCfg, mode, term_cfgs)

    @classmethod
    def expand_runtime_term_cfg(cls, term_cfg: Any) -> tuple[list[Any], bool]:
        from isaaclab.managers import TerminationTermCfg

        if not isinstance(term_cfg, TerminationTermCfg) or not isinstance(getattr(term_cfg, "params", None), dict):
            return [term_cfg], False

        params = dict(term_cfg.params)
        changed = False

        sub_term_defs = params.get("_sub_term_defs")
        if sub_term_defs is not None:
            rebuilt_sub_term_defs: list[dict[str, Any]] = []
            for sub_term_def in sub_term_defs:
                if not isinstance(sub_term_def, dict) or "cfg" not in sub_term_def:
                    raise ValueError(f"Invalid nested termination definition: {sub_term_def!r}")
                child_cfgs, child_changed = cls.expand_runtime_term_cfg(sub_term_def["cfg"])
                child_mode = sub_term_def.get("combine_mode", "and")
                if len(child_cfgs) == 1:
                    rebuilt_child_cfg = child_cfgs[0]
                else:
                    rebuilt_child_cfg = cls._make_combined_term(child_cfgs, mode=child_mode)
                    for meta_key in ("_score", "_label"):
                        if meta_key in getattr(sub_term_def["cfg"], "params", {}):
                            rebuilt_child_cfg.params[meta_key] = sub_term_def["cfg"].params[meta_key]
                    child_changed = True
                rebuilt_sub_term_defs.append(
                    {
                        **sub_term_def,
                        "cfg": rebuilt_child_cfg,
                    }
                )
                changed = changed or child_changed
            params["_sub_term_defs"] = rebuilt_sub_term_defs

        sequence_order_present = bool(params.get("sequence_order"))
        # object_contact_destination / object_no_contact_destination OR/AND a
        # whole sensor list themselves (pool cache-slot expansion mounts one
        # sensor per slot). Do not explode them into N ANDed terms here.
        func_name = getattr(getattr(term_cfg, "func", None), "__name__", "") or ""
        keep_contact_sensor_list_intact = func_name in {
            "object_contact_destination",
            "object_no_contact_destination",
        }
        # Expand singular ``_cfg`` params (one term per object, e.g. failure terms
        # object_dropped/drink_fallen) and the ``contact_sensor_cfgs`` sensor list.
        # Whole-list ``_cfgs`` params (e.g. ``object_cfgs`` on the pin-narrowing
        # reach/grasp success) stay INTACT — the predicate evaluates the single
        # episode-pinned target itself, not a broad OR/AND over the @drink pool.
        # (Mirrors the generate-time rule in _build_leaf_term_cfgs.)
        skip_contact_expand = sequence_order_present or keep_contact_sensor_list_intact
        for key, value in params.items():
            if key.startswith("_") or not (key.endswith("_cfg") or key == "contact_sensor_cfgs"):
                continue
            if key == "contact_sensor_cfgs" and skip_contact_expand:
                continue
            collapsed = _collapse_pool_window(value)
            if collapsed is not value:
                params[key] = collapsed
                changed = True
        expandable = {
            key: value
            for key, value in params.items()
            if not key.startswith("_")
            and (key.endswith("_cfg") or key == "contact_sensor_cfgs")
            and isinstance(value, list)
            and value
            and not (skip_contact_expand and key == "contact_sensor_cfgs")
        }
        if not expandable:
            if not changed:
                return [term_cfg], False
            rebuilt_term_cfg = TerminationTermCfg(
                func=term_cfg.func,
                params=params,
                time_out=getattr(term_cfg, "time_out", False),
            )
            return [rebuilt_term_cfg], True

        non_singleton_lengths = {len(values) for values in expandable.values() if len(values) > 1}
        if len(non_singleton_lengths) > 1:
            expanded_lengths = {key: len(values) for key, values in expandable.items()}
            raise ValueError("Termination term has incompatible runtime-expanded list lengths: " f"{expanded_lengths}")

        expand_count = non_singleton_lengths.pop() if non_singleton_lengths else len(next(iter(expandable.values())))
        expanded_term_cfgs: list[TerminationTermCfg] = []
        for index in range(expand_count):
            expanded_params = dict(params)
            for key, values in expandable.items():
                expanded_params[key] = cls._select_expanded_param_value(key, values, index)
            for meta_key in ("_score", "_label"):
                expanded_params.pop(meta_key, None)
            expanded_term_cfgs.append(
                TerminationTermCfg(
                    func=term_cfg.func,
                    params=expanded_params,
                    time_out=getattr(term_cfg, "time_out", False),
                )
            )
        return expanded_term_cfgs, True


@dataclass
class TerminationComponentCFG(TaskComponentCFG):
    """Flat termination configuration — function name is the key.

    YAML example::

        termination:
          terms:
            mdp.time_out:
              time_out: true
            success:
              terms:
                - object_position_in_zone:
                    object_cfg: "@pickable"
                    reference_cfg: "@destination"
                    distance_threshold: 0.04
            mdp.root_height_below_minimum:
              minimum_height: -0.05
              asset_cfg: "pick_object"

    Reserved keys — ``time_out``, ``_combine``, ``_func``, ``terms``, ``_score``,
    ``_label`` and ``_latch`` — are consumed by the parser; everything else is
    forwarded as plain params.

    ``_combine`` (``"and"`` default / ``"or"`` / ``"not"``) means **how this node's
    children fold**, uniformly at every depth:

    - on a **leaf** (a predicate with an ``@role`` that expands to N assets) it
      folds those N expanded copies (e.g. "near any tabletop" → ``or``);
    - on a **group** (a node carrying nested ``terms``) it folds the sub-terms —
      it is mapped onto the aggregator func's ``mode`` so ``success: {_combine: or,
      terms: [...]}`` really ORs (previously the top-level ``_combine`` was dropped
      and silently stuck on the func default).

    A group node whose key is just a readable label (no ``_func``) defaults its
    aggregator to ``merged``, so ``_combine`` groups nest into an arbitrary
    AND/OR/NOT tree::

        success:
          _combine: or                 # whole task succeeds if EITHER branch holds
          terms:
            - both_grippers:
                _combine: and          # ... both grippers reach (interior group)
                terms:
                  - left:  { _func: robot_body_near_object_xy, object_cfg: "@a" }
                  - right: { _func: robot_body_near_object_xy, object_cfg: "@b" }
            - shortcut: { _func: object_contact_destination, contact_sensor_cfgs: "@s" }

    ``_func`` optionally overrides the implementation function name so multiple
    aliased terms can reuse the same termination function. ``mode`` is the
    low-level func param ``_combine`` resolves onto and still works if written
    directly (``_combine`` wins when both are present).

    ``_score`` / ``_label`` annotate evaluation-rubric sub-terms.

    ``_latch`` (default ``false``) selects the term's temporal semantics, orthogonal
    to ``_func``/``_combine`` and composable with them:

    - ``false`` (default) = **momentary** — true only while the condition currently holds;
    - ``true`` = **latch** — held true for the rest of the episode once first hit.

    It wraps the built term (a leaf or a whole ``_combine`` group) in a generic
    ``latched`` carrier (per-episode buffer, self-resetting at episode start), so any
    term can opt into hold-after-hit without a bespoke ``*_once`` function. Not
    allowed on the reserved ``evaluation`` term (its rubric latches via
    ``first_hit_frame``)::

        success:
          _func: object_contact_destination
          contact_sensor_cfgs: "@s"
          _latch: true              # first contact latches success for the episode
    """

    class_type: type = TerminationComponent
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)
