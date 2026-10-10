"""Runtime helpers for public task evaluation.

Schema (``rules/04-domain-contracts.md`` CFG-Instance contract): the
evaluation rubric lives under ``termination.terms.evaluation`` in task yaml —
exactly the same compile path as ``termination.terms.success`` (sub-terms become
``_sub_term_defs`` on a single ``TerminationTermCfg``). The cfg is
extracted by name in :class:`TerminationComponent` and routed here via
``ManaTask.get_evaluation_term_cfg()``; it never enters the
``TerminationManager``.

Each leaf sub-term is a binary milestone latched once per episode. By
default a hit contributes 1.0 points; a leaf may declare a per-term
weight via the reserved ``_score: <float>`` meta-field in its yaml params
(analogous to the ``_combine`` meta-field used by ``success``). The
runtime pops ``_score`` before invoking the term function, so the meta
key never reaches ``func(env, **params)``. Total possible score equals
the sum of leaf ``_score`` values.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from manaenv.pipeline.generators.task.dag_success import failure_reason_label
from manaenv.pipeline.runtime.env_runner import (
    AFTER_ACTION,
    AFTER_STEP,
    BEFORE_STEP,
    ENV_DONES,
    SESSION_END,
    SESSION_START,
)
from manaenv.pipeline.runtime.log_store import LOG_STORE_KEY, commit_pending, ensure_log_store, flush_entries
from manaenv.pipeline.runtime.runtime_state import runtime_state
from manaenv.utils.debug_print import manaprint
from manaenv.utils.tensor_utils import as_torch

_TermFn = Callable[..., Any]


@dataclass
class EvaluationTerm:
    name: str
    func: _TermFn
    params: dict[str, Any]
    score: float = 1.0
    label: str = ""
    runtime_param_name: str | None = None


class EvaluationRuntime:
    """Latch evaluation terms over an episode and emit per-episode summaries."""

    def __init__(
        self,
        evaluation_term_cfg: Any | None,
        env_obj: Any,
        log_producers: list[Any] | None = None,
        overlay_entries: list[dict[str, Any]] | None = None,
        overlay_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.env_obj = env_obj
        self.evaluation_term_cfg = evaluation_term_cfg
        self._terms = self._build_terms_from_term_cfg(evaluation_term_cfg)
        self._log_producers = list(log_producers or [])
        self._overlay_entries = list(overlay_entries or [])
        self._overlay_cfg = dict(overlay_cfg or {})
        self.max_possible_score = round(sum(term.score for term in self._terms), 4)
        self._state_num_envs: int | None = None
        self._term_hits: dict[str, torch.Tensor] = {}
        self._term_first_hit_frames: dict[str, torch.Tensor] = {}
        self._fallback_frame_buf: torch.Tensor | None = None
        self._last_observed_frames: torch.Tensor | None = None
        self._episode_frame_counter: torch.Tensor | None = None
        self._episode_wall_started_at: list[float] = []
        # Per-env wording in force at episode start. Isaac ``step()`` auto-resets
        # done envs before ``ENV_DONES``, and reset events overwrite the live
        # store with the *next* episode; finalize must read this snapshot.
        self._episode_start_instruction: list[str] = []
        self._episode_records: list[dict[str, Any]] = []
        self._episode_index = 0
        self._step_dt_s = 0.0
        self._warned_missing_step_dt = False
        # Optional sink for per-episode streaming (e.g. evaluation_report.jsonl).
        self._on_finalized_records: Any | None = None

    @property
    def step_dt_s(self) -> float:
        return self._step_dt_s

    @property
    def overlay_entries(self) -> list[dict[str, Any]]:
        """Declared overlay display items (empty when no ``overlay_component``)."""
        return list(self._overlay_entries)

    @property
    def overlay_evaluation_enabled(self) -> bool | None:
        """Task-declared toggle for the built-in evaluation section.

        ``None`` when the task has no ``overlay_component`` (so the caller falls
        back to the rollout/loop ``show_evaluation_overlay`` flag).
        """
        if "evaluation" not in self._overlay_cfg:
            return None
        return bool(self._overlay_cfg.get("evaluation"))

    def overlay_snapshot(self, env_id: int = 0) -> list[dict[str, Any]]:
        """Resolve each overlay entry to its current per-env value for live display.

        Read-only: walks the declared ``overlay_entries`` in order, looks up the
        matching log producer by ``log_key`` and reads its :meth:`snapshot`. Values
        that are not yet available surface as ``None`` (the UI renders a placeholder).
        """
        if not self._overlay_entries:
            return []
        by_key: dict[str, Any] = {}
        for producer in self._log_producers:
            key = getattr(producer, "key", None)
            if key is not None:
                by_key.setdefault(str(key), producer)
        rows: list[dict[str, Any]] = []
        for entry in self._overlay_entries:
            log_key = str(entry.get("log_key", ""))
            producer = by_key.get(log_key)
            value = producer.snapshot(env_id) if producer is not None else None
            label = entry.get("label")
            rows.append(
                {
                    "label": log_key if label is None else str(label),
                    "value": value,
                    "font_size": entry.get("font_size"),
                    "font_color": entry.get("font_color"),
                }
            )
        return rows

    @property
    def episode_elapsed_frames(self) -> torch.Tensor | None:
        """Per-env frame counter maintained by evaluation runtime, unaffected by env resets."""
        return self._episode_frame_counter

    def register_hooks(self, runner: Any, *, extra_key: str = "evaluation_episode_records") -> None:
        def _init_state(ctx: Any) -> None:
            self._ensure_state(ctx.gym_env)
            # Seed episode 0's section choices (e.g. instruction variant) for
            # every env; no-op when no section producers are registered.
            env_ids = list(range(int(self._state_num_envs or 0)))
            self._seed_sections(ctx.gym_env, env_ids)
            self._capture_episode_start_instruction(ctx.gym_env, env_ids)

        def _reset_if_override(ctx: Any, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            if "override_obs" in ctx.extra and not ctx.extra.get("allow_indefinite_skip_step", False):
                env_ids = list(range(int(getattr(ctx.gym_env, "num_envs", 1))))
                self._reset_env_ids(env_ids, ctx.gym_env)
                self._seed_sections(ctx.gym_env, env_ids)
                self._capture_episode_start_instruction(ctx.gym_env, env_ids)

        def _update_state(ctx: Any, obs: Any, terminated: Any, truncated: Any, actions: Any) -> None:
            del obs, terminated, truncated, actions
            self._ensure_state(ctx.gym_env)
            if "override_obs" in ctx.extra and not ctx.extra.get("allow_indefinite_skip_step", False):
                env_ids = list(range(int(getattr(ctx.gym_env, "num_envs", 1))))
                self._reset_env_ids(env_ids, ctx.gym_env)
                self._seed_sections(ctx.gym_env, env_ids)
                self._capture_episode_start_instruction(ctx.gym_env, env_ids)
            else:
                self._update_term_hits(ctx.gym_env)

        def _finalize_done(ctx: Any, done: Any, obs: Any, terminated: Any, truncated: Any, actions: Any) -> None:
            del obs, terminated, truncated, actions
            env_ids = done.nonzero(as_tuple=False).squeeze(-1).tolist()
            if env_ids:
                # finalize_env_ids flushes + clears these envs (incl. each
                # producer's per-env state); seed the next episode's section
                # choices afterwards (flush-before-seed) so the new instruction
                # variant is in force before the next step feeds the policy.
                # Isaac already auto-reset these envs inside ``step()``, so the
                # live store is the *next* episode; the record uses the start
                # snapshot. Recapture after seed for the episode that just began.
                ctx.extra[extra_key] = self.finalize_env_ids(ctx.gym_env, env_ids)
                self._seed_sections(ctx.gym_env, env_ids)
                self._capture_episode_start_instruction(ctx.gym_env, env_ids)

        def _clear_state(ctx: Any) -> None:
            del ctx
            self._state_num_envs = None
            self._term_hits = {}
            self._term_first_hit_frames = {}
            self._fallback_frame_buf = None
            self._last_observed_frames = None
            self._episode_wall_started_at = []
            self._episode_start_instruction = []

        runner.add_hook(SESSION_START, _init_state, priority=15)
        runner.add_hook(BEFORE_STEP, _reset_if_override, priority=15)
        runner.add_hook(AFTER_ACTION, _reset_if_override, priority=15)
        runner.add_hook(AFTER_STEP, _update_state, priority=15)
        runner.add_hook(ENV_DONES, _finalize_done, priority=15)
        runner.add_hook(SESSION_END, _clear_state, priority=15)

    def episode_records(self) -> list[dict[str, Any]]:
        return list(self._episode_records)

    def restore_episode_records(self, records: list[dict[str, Any]]) -> None:
        """Restore finalized public-report rows before runner hooks are registered."""

        if self._state_num_envs is not None or self._episode_records or self._episode_index:
            raise RuntimeError("evaluation episode records can only be restored before a session starts")

        restored: list[dict[str, Any]] = []
        for expected_index, raw_record in enumerate(records):
            if not isinstance(raw_record, dict):
                raise TypeError(
                    f"restored evaluation episode #{expected_index} must be an object, got {type(raw_record).__name__}"
                )
            record = dict(raw_record)
            if record.get("episode_index") != expected_index:
                raise ValueError(
                    "restored evaluation episode indices must be contiguous: "
                    f"expected {expected_index}, got {record.get('episode_index')!r}"
                )
            score = record.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError(f"restored evaluation episode #{expected_index} has no numeric score")
            success = record.get("success")
            if not isinstance(success, bool):
                raise TypeError(f"restored evaluation episode #{expected_index} has no boolean success value")
            stored_max = record.get("max_possible_score")
            if isinstance(stored_max, bool) or not isinstance(stored_max, (int, float)):
                raise TypeError(
                    f"restored evaluation episode #{expected_index} has no numeric max_possible_score"
                )
            if abs(float(stored_max) - self.max_possible_score) > 1e-9:
                raise ValueError(
                    f"restored evaluation episode #{expected_index} has max_possible_score="
                    f"{stored_max!r}; current task expects {self.max_possible_score:g}"
                )
            restored.append(record)

        self._episode_records = restored
        self._episode_index = len(restored)

    def build_summary(self) -> dict[str, Any]:
        episode_count = len(self._episode_records)
        scores = [float(item["score"]) for item in self._episode_records]
        total_score = sum(scores)
        avg_score = total_score / episode_count if episode_count > 0 else 0.0
        max_score = max(scores) if scores else 0.0
        success_count = sum(1 for item in self._episode_records if bool(item.get("success", False)))
        success_rate = float(success_count) / float(episode_count) if episode_count > 0 else 0.0
        return {
            "num_episodes": episode_count,
            "success_count": success_count,
            "success_rate": round(success_rate, 4),
            "total_score": round(total_score, 4),
            "avg_score": round(avg_score, 4),
            "max_score": round(max_score, 4),
            "max_possible_score": self.max_possible_score,
            "per_stage_pass_rate": self._per_stage_pass_rate(),
            "episodes": self._episode_breakdown(),
        }

    def _episode_breakdown(self) -> list[dict[str, Any]]:
        """One compact row per episode, carrying the identity needed to pair two runs.

        Aggregates hide the only structure a head-to-head comparison can exploit: the same scene
        seed replays the same episode for both sides, so a comparison paired on episode identity
        cancels scene difficulty entirely. The pairing key is ``(env_id, ordinal)`` — the n-th
        episode of a given parallel env — because the global ``episode_index`` is assigned in
        completion order and reorders across runs when ``num_parallel_envs > 1``.

        ``last_milestone_frame`` is the frame at which the LAST rubric leaf latched. It is set on
        failed episodes too — an episode that earns partial credit still latched something, and how
        fast it got there is exactly the efficiency margin that separates two models the coarse
        rubric scores identically. It is emphatically not the frame the episode "succeeded" at.
        """
        ordinals: dict[int, int] = {}
        rows: list[dict[str, Any]] = []
        for record in self._episode_records:
            env_id = int(record.get("env_id", 0))
            ordinal = ordinals.get(env_id, 0)
            ordinals[env_id] = ordinal + 1
            frames = [
                term.get("first_hit_frame")
                for term in (record.get("terms") or {}).values()
                if term.get("hit") and term.get("first_hit_frame") is not None
            ]
            rows.append(
                {
                    "env_id": env_id,
                    "ordinal": ordinal,
                    "episode_index": record.get("episode_index"),
                    "score": float(record.get("score", 0.0)),
                    "max_possible_score": record.get("max_possible_score"),
                    "success": bool(record.get("success", False)),
                    "outcome": record.get("outcome"),
                    "failed_stage": record.get("failed_stage"),
                    "failed_reason": record.get("failed_reason"),
                    "last_milestone_frame": max(frames) if frames else None,
                    # How long the trajectory ran, on every episode — success, failure, or
                    # timeout alike. A failed episode's length is the measurement that
                    # separates "dropped it immediately" from "spent the whole budget trying".
                    "elapsed_frames": record.get("elapsed_frames"),
                    "duration_s": record.get("duration_s"),
                    "wall_duration_s": record.get("wall_duration_s"),
                    # Which milestones latched, and on which FRAME. Dropped until now, which
                    # left a report able to say only "it failed at stage X" — never how far
                    # into X it got, nor which of the scored milestones it did reach. These
                    # frames are also the only cut boundaries a per-stage video clip can use.
                    "terms": [
                        {
                            "label": term.get("label"),
                            "hit": bool(term.get("hit")),
                            "first_hit_frame": term.get("first_hit_frame"),
                            "awarded_points": term.get("awarded_points"),
                            "max_points": term.get("max_points"),
                        }
                        for term in (record.get("terms") or {}).values()
                    ],
                }
            )
        return rows

    def _per_stage_pass_rate(self) -> dict[str, float]:
        """Fraction of episodes in which each rubric leaf (stage) latched a hit.

        Groups the per-episode latched ``terms`` by their human-facing ``label``
        (the graph node id for composite tasks, e.g. reach/grasp/retract) and
        averages the boolean ``hit`` across episodes. Yields the diagnostic
        per-stage pass rates (到位率/抓取率/回收率) the staged-eval feature exists
        for. Empty for non-staged tasks (one rubric leaf per ``_label``); harmless
        passthrough into the benchmark report.
        """
        if not self._episode_records:
            return {}
        hit_counts: dict[str, int] = {}
        seen_counts: dict[str, int] = {}
        order: list[str] = []
        for record in self._episode_records:
            for term_state in (record.get("terms") or {}).values():
                label = str(term_state.get("label", ""))
                if label not in seen_counts:
                    seen_counts[label] = 0
                    hit_counts[label] = 0
                    order.append(label)
                seen_counts[label] += 1
                if bool(term_state.get("hit", False)):
                    hit_counts[label] += 1
        return {label: round(hit_counts[label] / seen_counts[label], 4) for label in order if seen_counts[label]}

    def live_snapshot(self, gym_env: Any) -> dict[str, Any]:
        """Per-stage rollout progress for live monitoring (read-only, cheap).

        Reuses the runtime's own latched state (``_term_hits``), which is already
        label-aligned to ``_terms`` (graph node ids reach/grasp/retract) and reset
        per-env in lockstep with episode boundaries — the same gated truth as
        ``env.extras["dag_stage_done"]``, but pre-grouped by label. Reports, for the
        in-flight episodes, how many active envs have latched each stage and each
        env's current frontier stage, alongside the cumulative finished-episode
        pass rates (到位率/抓取率/回收率). Empty for tasks with no evaluation rubric.

        Falls back to ``env.extras["dag_stage_done"]`` only before the first
        ``_update_term_hits`` populates ``_term_hits``; both hold the same columns.
        """
        if not self._terms:
            return {}
        stages = [term.label for term in self._terms]
        empty = {
            "stages": stages,
            "num_envs": 0,
            "in_flight_latched": [0] * len(stages),
            "current_stage_per_env": [],
            "completed_episodes": len(self._episode_records),
            "completed_pass_rate": self._per_stage_pass_rate(),
        }
        if self._term_hits:
            latched = torch.stack([self._term_hits[term.name].to(dtype=torch.bool) for term in self._terms], dim=1)
        else:
            extras = getattr(gym_env, "extras", None)
            extras_key = self._terms[0].params.get("extras_key", "dag_stage_done")
            done = extras.get(extras_key) if isinstance(extras, dict) else None
            if not isinstance(done, torch.Tensor) or done.ndim != 2:
                return empty
            latched = done.to(dtype=torch.bool)
        num_envs, in_flight, current_stage_per_env = _stage_frontier(latched, stages)
        return {
            "stages": stages,
            "num_envs": num_envs,
            "in_flight_latched": in_flight,
            "current_stage_per_env": current_stage_per_env,
            "completed_episodes": len(self._episode_records),
            "completed_pass_rate": self._per_stage_pass_rate(),
        }

    @staticmethod
    def _build_terms_from_term_cfg(term_cfg: Any | None) -> list[EvaluationTerm]:
        if term_cfg is None:
            return []
        params = getattr(term_cfg, "params", None) or {}
        sub_term_defs = params.get("_sub_term_defs") or []
        if not sub_term_defs:
            return []

        leaf_specs = EvaluationRuntime._flatten_sub_term_defs(sub_term_defs)
        built: list[EvaluationTerm] = []
        for index, (name, sub_cfg) in enumerate(leaf_specs):
            func = getattr(sub_cfg, "func", None)
            if func is None:
                raise ValueError(f"evaluation sub-term #{index} missing func ({sub_cfg!r})")
            sub_params = dict(getattr(sub_cfg, "params", {}) or {})
            score = float(sub_params.pop("_score", 1.0))
            label = str(sub_params.pop("_label", "") or name)
            runtime_param_name = None
            try:
                import inspect

                signature = inspect.signature(func)
                for candidate in ("evaluation_runtime", "score_runtime"):
                    if candidate in signature.parameters:
                        runtime_param_name = candidate
                        break
            except (TypeError, ValueError):
                runtime_param_name = None
            built.append(
                EvaluationTerm(
                    name=name,
                    func=func,
                    params=sub_params,
                    score=score,
                    label=label,
                    runtime_param_name=runtime_param_name,
                )
            )
        return built

    @staticmethod
    def _flatten_sub_term_defs(sub_term_defs: list[Any]) -> list[tuple[str, Any]]:
        """Each top-level rubric entry is exactly one scored named checker.

        Credit assignment is orthogonal to boolean composition (the user's
        rule): an entry's signal may be a leaf checker OR an AND/OR/NOT
        expression — its ``func`` already evaluates the whole expression to a
        single bool via the shared checker primitive — but it earns its
        ``_score`` as one unit. We do NOT descend into a composite to mint
        extra scored leaves; weights live only on the named entries here.
        """
        leaves: list[tuple[str, Any]] = []
        for index, raw in enumerate(sub_term_defs):
            sub_cfg = raw["cfg"] if isinstance(raw, dict) else raw
            func = getattr(sub_cfg, "func", None)
            func_name = getattr(func, "__name__", f"term_{index}")
            leaves.append((f"{func_name}__{index}", sub_cfg))
        return leaves

    def _ensure_state(self, gym_env: Any) -> None:
        num_envs = int(getattr(gym_env, "num_envs", 1))
        # Use an explicit env-count sentinel rather than ``self._term_hits`` so
        # log-only runtimes (no evaluation terms) still init their frame
        # counters exactly once instead of re-zeroing them every step.
        if self._state_num_envs == num_envs:
            if self._step_dt_s <= 0.0:
                self._step_dt_s = self._resolve_step_dt_s(gym_env)
            return

        self._state_num_envs = num_envs
        self._term_hits = {
            term.name: torch.zeros(num_envs, dtype=torch.bool, device=gym_env.device) for term in self._terms
        }
        self._term_first_hit_frames = {
            term.name: torch.full((num_envs,), -1, dtype=torch.int64, device=gym_env.device) for term in self._terms
        }
        self._fallback_frame_buf = torch.zeros((num_envs,), dtype=torch.int64, device=gym_env.device)
        self._last_observed_frames = torch.zeros((num_envs,), dtype=torch.int64, device=gym_env.device)
        self._episode_frame_counter = torch.zeros((num_envs,), dtype=torch.int64, device=gym_env.device)
        self._episode_wall_started_at = [time.monotonic()] * num_envs
        self._episode_start_instruction = [""] * num_envs
        for producer in self._log_producers:
            producer.ensure_state(num_envs, gym_env.device)
        # Live store backing runtime-readable (non-append, tensor) log cells.
        ensure_log_store(gym_env, num_envs, gym_env.device)
        self._step_dt_s = self._resolve_step_dt_s(gym_env)
        if self._step_dt_s <= 0.0 and not self._warned_missing_step_dt:
            manaprint("WARNING: [Evaluation] runtime could not resolve step_dt; time-window terms will stay false.")
            self._warned_missing_step_dt = True

    def _resolve_step_dt_s(self, gym_env: Any) -> float:
        step_dt = getattr(gym_env, "step_dt", None)
        if isinstance(step_dt, (int, float)) and step_dt > 0:
            return float(step_dt)

        scene = getattr(self.env_obj, "scene", None) if self.env_obj is not None else None
        sim_fps = getattr(scene, "sim_fps", None)
        simulation_decimation = getattr(scene, "simulation_decimation", None)
        if sim_fps and simulation_decimation:
            return float(simulation_decimation) / float(sim_fps)
        render_decremental = getattr(scene, "render_decremental", None)
        if sim_fps and render_decremental:
            return float(render_decremental) / float(sim_fps)
        return 0.0

    def _current_frame_tensor(self, gym_env: Any) -> torch.Tensor:
        if hasattr(gym_env, "episode_length_buf"):
            return gym_env.episode_length_buf.to(device=gym_env.device, dtype=torch.int64).clone()

        assert self._fallback_frame_buf is not None
        self._fallback_frame_buf += 1
        return self._fallback_frame_buf.clone()

    def _peek_frame_tensor(self, gym_env: Any) -> torch.Tensor:
        if self._last_observed_frames is not None:
            return self._last_observed_frames.clone()
        if hasattr(gym_env, "episode_length_buf"):
            return gym_env.episode_length_buf.to(device=gym_env.device, dtype=torch.int64).clone()
        if self._fallback_frame_buf is not None:
            return self._fallback_frame_buf.clone()
        num_envs = int(getattr(gym_env, "num_envs", 1))
        return torch.zeros((num_envs,), dtype=torch.int64, device=gym_env.device)

    def _update_term_hits(self, gym_env: Any) -> None:
        current_frames = self._current_frame_tensor(gym_env)
        self._last_observed_frames = current_frames.clone()
        # Isaac Lab resets episode_length_buf (and joint positions) inside
        # step() for done envs *before* AFTER_STEP fires.  This mask flags the
        # freshly-reset envs so neither the frame counter nor log producers
        # carry/latch stale values across the episode boundary.
        reset_mask = gym_env.episode_length_buf == 0 if hasattr(gym_env, "episode_length_buf") else None
        if self._episode_frame_counter is not None:
            # Isaac resets ``episode_length_buf`` inside step() before AFTER_STEP.
            # Do not reset our independent counter here: ENV_DONES still has to
            # finalize the episode that just ended. ``_reset_env_ids`` clears it
            # after the record is captured.
            self._episode_frame_counter += 1
            term_frames = self._episode_frame_counter.clone()
        else:
            term_frames = current_frames
        for term in self._terms:
            if term.runtime_param_name is not None:
                hit = term.func(gym_env, **term.params, **{term.runtime_param_name: self})
            else:
                hit = term.func(gym_env, **term.params)
            if not isinstance(hit, torch.Tensor):
                hit = torch.as_tensor(hit, dtype=torch.bool, device=gym_env.device)
            hit = hit.to(dtype=torch.bool, device=gym_env.device)
            new_hit = hit & (~self._term_hits[term.name])
            self._term_first_hit_frames[term.name][new_hit] = term_frames[new_hit]
            self._term_hits[term.name] |= hit
        for producer in self._log_producers:
            producer.update(gym_env, current_frames, reset_mask)
        # Single writer commits after every reader has run this frame; the new
        # values become readable from the next frame (1-frame latch delay).
        commit_pending(gym_env)
        # SESSION_START can snapshot the unfilled ``{initial}/{target}`` template
        # before the reset event writes real temps. Refresh on a live frame only
        # — skip Isaac's auto-reset step, where the live store is already the next episode.
        self._refresh_unfilled_episode_start_instruction(gym_env, reset_mask)

    def set_on_finalized_records(self, callback: Any | None) -> None:
        """Register a callback invoked with each batch of finalized episode records."""
        self._on_finalized_records = callback

    def finalize_env_ids(
        self,
        gym_env: Any,
        env_ids: list[int],
        *,
        success_overrides: dict[int, bool] | None = None,
    ) -> list[dict[str, Any]]:
        if not env_ids:
            return []
        self._ensure_state(gym_env)
        success_flags = self._resolve_success_flags(gym_env, env_ids)
        if success_overrides:
            success_flags = [
                bool(success_overrides[env_id]) if env_id in success_overrides else flag
                for env_id, flag in zip(env_ids, success_flags)
            ]
        finalized_records: list[dict[str, Any]] = []
        wall_finished_at = time.monotonic()
        for env_id, success_hit in zip(env_ids, success_flags):
            term_states: dict[str, dict[str, Any]] = {}
            score = 0.0
            for term in self._terms:
                hit = bool(self._term_hits[term.name][env_id].item())
                awarded = term.score if hit else 0.0
                score += awarded
                first_hit_frame = int(self._term_first_hit_frames[term.name][env_id].item())
                term_states[term.name] = {
                    "label": term.label,
                    "hit": hit,
                    "awarded_points": round(awarded, 4),
                    "max_points": round(term.score, 4),
                    "first_hit_frame": first_hit_frame,
                }

            failed_stage = None if bool(success_hit) else self._resolve_failed_stage(gym_env, int(env_id))
            failed_reason = None if bool(success_hit) else self._resolve_failed_reason(gym_env, int(env_id))

            log_payload: dict[str, list[dict[str, Any]]] = {}
            for producer in self._log_producers:
                collected = producer.collect(env_id)
                if collected is not None:
                    key, entries = collected
                    if entries:
                        log_payload.setdefault(key, []).extend(entries)
            # Non-append tensor cells live in the per-env store (not in any
            # producer's _entries); flush their final per-env value as a single
            # entry, keyed the same as append/ledger entries.
            for key, entry in flush_entries(gym_env, env_id).items():
                log_payload.setdefault(key, []).append(entry)
            # Read the counters BEFORE ``_reset_env_ids`` below zeroes them for the next episode.
            elapsed_frames = (
                int(self._episode_frame_counter[env_id].item()) if self._episode_frame_counter is not None else 0
            )
            wall_started_at = (
                self._episode_wall_started_at[env_id]
                if 0 <= env_id < len(self._episode_wall_started_at)
                else wall_finished_at
            )
            episode_record = {
                "episode_index": self._episode_index,
                "env_id": int(env_id),
                "success": bool(success_hit),
                # Wall-clock length of the trajectory in simulated time. Frames are the only
                # monotonic per-env clock here (a parallel env's episodes interleave), so
                # duration comes from frames × step_dt rather than from a timestamp delta.
                "elapsed_frames": elapsed_frames,
                "duration_s": round(elapsed_frames * self._step_dt_s, 3) if self._step_dt_s > 0.0 else None,
                # Real operator-visible elapsed time. Keep it separate from
                # simulated duration: remote inference, rendering and recorder
                # work advance this clock but not ``duration_s``.
                "wall_duration_s": round(max(0.0, wall_finished_at - wall_started_at), 3),
                # Success|Failure binary outcome + attribution: WHICH stage failed
                # and WHY (reason label). Both ``None`` on success.
                "outcome": "success" if bool(success_hit) else "failure",
                "failed_stage": failed_stage,
                "failed_reason": failed_reason,
                "score": round(score, 4),
                "max_possible_score": self.max_possible_score,
                # Policy-facing wording for this episode. Prefer the snapshot taken
                # at episode start: Isaac auto-reset already overwrote the live
                # store with the next episode before ENV_DONES reaches here.
                "instruction": self._instruction_for_finalized_episode(gym_env, int(env_id)),
                "terms": term_states,
                "log": log_payload,
            }
            self._episode_records.append(episode_record)
            finalized_records.append(dict(episode_record))
            self._episode_index += 1

        self._reset_env_ids(env_ids, gym_env)
        callback = self._on_finalized_records
        if callback is not None and finalized_records:
            callback(finalized_records)
        return finalized_records

    # Every episode is Success or Failure(stage, reason) — a failed episode is
    # never reasonless. Attribution has two independent sources, because two
    # different layers hold the two halves of the answer:
    #
    #   WHERE  the stage machine (``dag_*`` extras) — it alone tracks the graph.
    #   WHY    the TerminationManager — it alone knows which term ended the run.
    #
    # ``TaskFactory._collect_failure_term_cfgs`` folds every non-success termination
    # into the machine, so normally BOTH come from the machine. The fallbacks below
    # exist for the runs where they cannot: a report replayed from before that
    # classification existed, or a task with no stage machine at all. They must not
    # guess — naming a dropped object a "timeout" is worse than saying nothing.
    _UNKNOWN_REASON = "unknown"
    _NON_CAUSE_TERMS = frozenset({"success", "failure"})

    @classmethod
    def _resolve_failed_stage(cls, gym_env: Any, env_id: int) -> Any:
        """WHERE *env_id* died: the stage NAME, or ``None`` if not locatable.

        Prefers the machine's attributed ``extras["dag_failed_stage"]``; falls back
        to the frontier stage the env was stalled in, which is defined for every
        graph — including the pure-success ones that publish no failure extras.
        """
        extras = getattr(gym_env, "extras", None)
        if not isinstance(extras, dict):
            return None
        idx = cls._extras_index(extras, "dag_failed_stage", env_id)
        if idx is not None:
            return cls._label_at(extras, "dag_failed_stage_labels", idx)
        return cls._resolve_stalled_stage(extras, env_id)

    @classmethod
    def _resolve_failed_reason(cls, gym_env: Any, env_id: int) -> Any:
        """WHY *env_id* died: the machine's reason label, else the termination term
        that actually fired ("object_dropped", "time_out", …), else ``unknown``."""
        extras = getattr(gym_env, "extras", None)
        if isinstance(extras, dict):
            idx = cls._extras_index(extras, "dag_failed_reason", env_id)
            if idx is not None:
                return cls._label_at(extras, "dag_failed_reason_labels", idx)
        return cls._resolve_fired_termination(gym_env, env_id) or cls._UNKNOWN_REASON

    @classmethod
    def _resolve_fired_termination(cls, gym_env: Any, env_id: int) -> str | None:
        """Name of the termination term that ended this episode for *env_id*.

        A genuine cause outranks the ``time_out`` backstop: when an episode is cut
        short by e.g. ``object_dropped`` on the very step its budget also expires,
        the drop is what to report. ``success``/``failure`` are excluded — success
        is not a failure cause, and ``failure`` is the machine's own composite,
        already read (with a finer per-stage label) by the caller.

        The name is passed through :func:`failure_reason_label`, the SAME function the
        stage machine registers its reasons with. A manager term is named after its config
        field, so returning it raw leaked plumbing straight to the report and the UI: a
        reader saw ``mdp__time_out`` here and ``time_out`` from the machine, with no way to
        know they were one reason.
        """
        manager = getattr(gym_env, "termination_manager", None)
        if manager is None:
            return None
        fired_timeout: str | None = None
        for name in getattr(manager, "active_terms", ()) or ():
            if name in cls._NON_CAUSE_TERMS or not cls._term_fired(manager, name, env_id):
                continue
            if cls._is_time_out_term(manager, name):
                fired_timeout = name
                continue
            return failure_reason_label(name)
        return failure_reason_label(fired_timeout) if fired_timeout is not None else None

    @staticmethod
    def _term_fired(manager: Any, name: str, env_id: int) -> bool:
        try:
            term = manager.get_term(name)
        except (KeyError, AttributeError, ValueError):
            return False
        if not isinstance(term, torch.Tensor):
            return False
        term = term.reshape(-1)
        if not 0 <= env_id < term.shape[0]:
            return False
        return bool(term[env_id].item())

    @staticmethod
    def _is_time_out_term(manager: Any, name: str) -> bool:
        """Whether *name* is IsaacLab's truncation-flavored backstop (``time_out=True``)."""
        try:
            cfg = manager.get_term_cfg(name)
        except (KeyError, AttributeError, ValueError):
            return False
        return bool(getattr(cfg, "time_out", False))

    @staticmethod
    def _resolve_stalled_stage(extras: dict, env_id: int) -> Any:
        """The frontier stage *env_id* was stalled in: enterable but not latched.

        ``dag_stage_active`` ("all predecessors done") and ``dag_stage_done`` are
        published unconditionally — for pure-success graphs too — so this works
        where the failure extras do not exist. ``active & ~done`` is exactly the
        machine's ``ready`` frontier; the first such column is the stage the env
        never got out of.
        """
        active = extras.get("dag_stage_active")
        done = extras.get("dag_stage_done")
        if not isinstance(active, torch.Tensor) or not isinstance(done, torch.Tensor):
            return None
        if active.shape != done.shape or not (0 <= env_id < active.shape[0]):
            return None
        frontier = active[env_id].to(torch.bool) & ~done[env_id].to(torch.bool)
        hits = torch.nonzero(frontier, as_tuple=False).reshape(-1)
        if hits.numel() == 0:
            return None
        idx = int(hits[0].item())
        return EvaluationRuntime._label_at(extras, "dag_stage_active_labels", idx)

    @staticmethod
    def _extras_index(extras: dict, key: str, env_id: int) -> int | None:
        """Per-env index published under *key*, or ``None`` when absent/unset (-1)."""
        idx_tensor = extras.get(key)
        if not isinstance(idx_tensor, torch.Tensor) or not (0 <= env_id < idx_tensor.shape[0]):
            return None
        idx = int(idx_tensor[env_id].item())
        return idx if idx >= 0 else None

    @staticmethod
    def _label_at(extras: dict, labels_key: str, idx: int) -> Any:
        """Name at *idx* in the published label list; the raw index if unnamed."""
        labels = extras.get(labels_key)
        if isinstance(labels, (list, tuple)) and 0 <= idx < len(labels):
            return str(labels[idx])
        return idx

    @staticmethod
    def _resolve_success_flags(gym_env: Any, env_ids: list[int]) -> list[bool]:
        termination_manager = getattr(gym_env, "termination_manager", None)
        active_terms = getattr(termination_manager, "active_terms", ()) if termination_manager is not None else ()
        if termination_manager is None or "success" not in active_terms:
            return [False for _ in env_ids]

        success_term = termination_manager.get_term("success")
        if not isinstance(success_term, torch.Tensor):
            success_term = torch.as_tensor(success_term, dtype=torch.bool, device=gym_env.device)
        success_tensor = success_term.to(dtype=torch.bool, device=gym_env.device).reshape(-1)
        return [
            bool(success_tensor[env_id].item()) if 0 <= env_id < success_tensor.shape[0] else False
            for env_id in env_ids
        ]

    @staticmethod
    def _is_unfilled_instruction(text: str) -> bool:
        """True when *text* is empty or still has format placeholders."""
        if not isinstance(text, str) or not text.strip():
            return True
        return "{" in text and "}" in text

    def _capture_episode_start_instruction(self, gym_env: Any, env_ids: list[int]) -> None:
        """Snapshot the live instruction now in force as this episode's wording.

        Call after the episode-start write (Isaac reset event and/or section
        seed). Finalize later reads this snapshot because ``gym_env.step()``
        auto-resets done envs before ``ENV_DONES``.
        """
        if not env_ids:
            return
        self._ensure_state(gym_env)
        for env_id in env_ids:
            idx = int(env_id)
            if 0 <= idx < len(self._episode_start_instruction):
                self._episode_start_instruction[idx] = self._resolve_episode_instruction(gym_env, idx)

    def _refresh_unfilled_episode_start_instruction(self, gym_env: Any, reset_mask: Any | None) -> None:
        """Replace a template snapshot with the filled live wording mid-episode.

        Do not refresh on a freshly reset env: that live store belongs to the next episode.
        """
        if not self._episode_start_instruction:
            return
        env_ids: list[int] = []
        for env_id, cached in enumerate(self._episode_start_instruction):
            if reset_mask is not None and 0 <= env_id < int(reset_mask.shape[0]) and bool(reset_mask[env_id]):
                continue
            if self._is_unfilled_instruction(cached):
                env_ids.append(env_id)
        self._capture_episode_start_instruction(gym_env, env_ids)

    def _instruction_for_finalized_episode(self, gym_env: Any, env_id: int) -> str:
        """Return the wording that was in force when this episode started.

        Empty or still-templated snapshots fall back to the live store so a
        direct ``sync_and_finalize`` (before reset) still records the current wording.
        The snapshot is consumed so the next episode cannot reuse it.
        """
        idx = int(env_id)
        if 0 <= idx < len(self._episode_start_instruction):
            cached = self._episode_start_instruction[idx]
            self._episode_start_instruction[idx] = ""
            if isinstance(cached, str) and cached.strip() and not self._is_unfilled_instruction(cached):
                return cached
        return self._resolve_episode_instruction(gym_env, idx)

    def _resolve_episode_instruction(self, gym_env: Any, env_id: int) -> str:
        """Return the policy-facing instruction currently in the live store."""
        try:
            from manaenv.pipeline.runtime.active_instruction import resolve_active_instruction

            active = resolve_active_instruction(gym_env, int(env_id))
            if isinstance(active, str) and active.strip():
                return " ".join(active.split())
        except Exception as exc:
            manaprint(f"WARNING: Failed to resolve active instruction for episode record: {exc}")
        get_prompt = getattr(self.env_obj, "get_prompt", None)
        if callable(get_prompt):
            try:
                return " ".join(str(get_prompt() or "").split())
            except Exception as exc:
                manaprint(f"WARNING: Failed to resolve task prompt for episode record: {exc}")
        return ""

    def _reset_env_ids(self, env_ids: list[int], gym_env: Any | None = None) -> None:
        if not env_ids:
            return
        for hits in self._term_hits.values():
            hits[env_ids] = False
        for first_hit_frames in self._term_first_hit_frames.values():
            first_hit_frames[env_ids] = -1
        if self._fallback_frame_buf is not None:
            self._fallback_frame_buf[env_ids] = 0
        if self._last_observed_frames is not None:
            self._last_observed_frames[env_ids] = 0
        if self._episode_frame_counter is not None:
            self._episode_frame_counter[env_ids] = 0
        reset_at = time.monotonic()
        for env_id in env_ids:
            if 0 <= env_id < len(self._episode_wall_started_at):
                self._episode_wall_started_at[env_id] = reset_at
        for producer in self._log_producers:
            producer.reset(env_ids)
        if gym_env is not None:
            runtime_state(gym_env).reset(env_ids, keys={LOG_STORE_KEY})

    def _seed_sections(self, gym_env: Any, env_ids: list[int]) -> None:
        """Run reset-driven section producers for *env_ids* (e.g. pick this
        episode's instruction variant and write it to the per-env store). No-op
        for trigger-driven producers."""
        if not env_ids:
            return
        for producer in self._log_producers:
            producer.seed_reset(gym_env, env_ids)


def _resolve_overlay_spec(task: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the task's overlay display spec from ``runtime_cfgs``.

    Both composite (graph) and atomic tasks write ``overlay_entries`` /
    ``overlay_cfg`` into ``runtime_cfgs`` (the composite via
    ``_render_composite_runtime_sections``, the atomic via the generic component
    result merge), so a single read covers both shapes.
    """
    runtime_cfgs = getattr(task, "runtime_cfgs", None)
    if not isinstance(runtime_cfgs, dict):
        return [], {}
    entries = runtime_cfgs.get("overlay_entries") or []
    cfg = runtime_cfgs.get("overlay_cfg") or {}
    entries_list = [dict(entry) for entry in entries if isinstance(entry, dict)]
    cfg_dict = dict(cfg) if isinstance(cfg, dict) else {}
    return entries_list, cfg_dict


def _resolve_log_producer_specs(task: Any) -> list[Any]:
    """This task's log-producer specs.

    One call, no fallback: every task is a graph task, so ``get_log_producers`` always
    answers — a composite rewrites its stages' triggers onto the DAG extras, a leaf returns
    its own ``runtime_cfgs["log_producers"]``. The old two-branch probe existed only to tell
    the two kinds apart, and there are no longer two kinds.
    """
    getter = getattr(task, "get_log_producers", None)
    return list(getter() or []) if callable(getter) else []


def _stage_frontier(latched: torch.Tensor, stages: list[str]) -> tuple[int, list[int], list[int]]:
    """Shared per-stage frontier math over a latched ``(num_envs, num_stages)`` bool matrix.

    Returns ``(num_envs, in_flight_latched, current_stage_per_env)`` where
    ``in_flight_latched[i]`` counts envs that have latched stage ``i`` and
    ``current_stage_per_env[e]`` is env ``e``'s frontier index (count of latched
    stages). Reused by :meth:`EvaluationRuntime.live_snapshot` and
    :func:`stage_snapshot_from_env`.
    """
    num_envs = int(latched.shape[0])
    n_cols = int(latched.shape[1])
    in_flight = [int(latched[:, i].sum().item()) if i < n_cols else 0 for i in range(len(stages))]
    current_stage_per_env = latched.to(dtype=torch.int64).sum(dim=1).tolist()
    return num_envs, in_flight, current_stage_per_env


def stage_snapshot_from_env(
    gym_env: Any,
    labels: list[str] | None = None,
    extras_key: str = "dag_stage_done",
) -> dict[str, Any]:
    """Build a per-stage live frontier snapshot directly from the env's published matrix.

    The eval-runtime-free path used by the global monitoring hook: it needs no
    :class:`EvaluationRuntime` instance, only the env's published gated-stage
    truth ``gym_env.extras[extras_key]`` (shape ``(num_envs, num_stages)``, bool)
    — the same tensor ``live_snapshot`` falls back to. ``labels`` names the stage
    columns (reach/grasp/retract); when omitted, generic ``stage_i`` labels are
    used. Returns ``{}`` when no stage matrix is published (non-staged rollouts).
    """
    extras = getattr(gym_env, "extras", None)
    done = extras.get(extras_key) if isinstance(extras, dict) else None
    if not isinstance(done, torch.Tensor) or done.ndim != 2:
        return {}
    latched = done.to(dtype=torch.bool)
    n_cols = int(latched.shape[1])
    # Prefer the column labels the DAG machine publishes (authoritative, in
    # matrix-column order); fall back to caller-supplied labels, then generic.
    published = extras.get(f"{extras_key}_labels") if isinstance(extras, dict) else None
    if isinstance(published, (list, tuple)) and len(published) == n_cols:
        stages = [str(s) for s in published]
    elif labels and len(labels) == n_cols:
        stages = list(labels)
    else:
        stages = [f"stage_{i}" for i in range(n_cols)]
    num_envs, in_flight, current_stage_per_env = _stage_frontier(latched, stages)
    return {
        "stages": stages,
        "num_envs": num_envs,
        "in_flight_latched": in_flight,
        "current_stage_per_env": current_stage_per_env,
        "latched_matrix": latched.to(dtype=torch.int64).tolist(),
    }


def ee_init_miss_from_env(gym_env: Any, init_state_attr: str = "_mana_init_ee_pos_w") -> dict[str, float]:
    """Per-EE-frame max distance (over envs) from the captured episode-init pose.

    Best-effort readout for the ``return``/``retract`` stage, whose success
    (:func:`manaenv.terms.terminations.ee_near_init_pose`) requires every arm back
    within a tolerance of ``env.<init_state_attr>``. Surfacing the live miss makes
    it observable *why* a return stalls (which arm is short, and by how much) —
    e.g. a deep low-shelf grasp whose arm cannot get home in time. Returns ``{}``
    for tasks that never captured init poses (non-return tasks).
    """
    store = getattr(gym_env, init_state_attr, None)
    if not isinstance(store, dict) or not store:
        return {}
    out: dict[str, float] = {}
    for frame_key, init_pos in store.items():
        if not isinstance(init_pos, torch.Tensor):
            continue
        try:
            frame = gym_env.scene[frame_key]
            cur_pos_w = frame.data.target_pos_w.torch[:, 0, :]
            dist = torch.linalg.norm(init_pos - cur_pos_w, dim=-1)
            out[str(frame_key)] = float(dist.max().item())
        except Exception:
            continue
    return out


def grasp_target_force_from_env(
    gym_env: Any,
    pin_attr: str = "_mana_episode_grasp_target",
    sensor_suffix: str = "gripper_contact_sensor",
) -> dict[str, float]:
    """Gripper contact force (N) on each env's episode-pinned grasp-target bottle.

    Best-effort readout for the ``grasp`` stage, whose success
    (:func:`manaenv.terms.terminations.grasp_target_lifted`) needs the pinned
    bottle's own gripper-filtered contact force to clear a threshold. Surfacing it
    live disambiguates a grasp stall: ~0 N = the gripper never contacted the
    bottle (a pose/approach miss), a value just under the threshold = a marginal
    squeeze. Mirrors ``grasp_target_lifted``'s ``force_matrix_w`` read. Returns
    ``{}`` for tasks with no pinned grasp target.
    """
    pin = getattr(gym_env, pin_attr, None)
    if not isinstance(pin, dict) or not pin:
        return {}
    out: dict[str, float] = {}
    for env_id, name in pin.items():
        try:
            sensor = gym_env.scene[f"{name}_{sensor_suffix}"]
            fmat = as_torch(getattr(sensor.data, "force_matrix_w", None))
            if fmat is None or fmat.numel() == 0:
                continue
            norm = torch.linalg.norm(fmat, dim=-1).flatten(start_dim=1).sum(dim=-1)
            out[str(name)] = float(norm[int(env_id)].item())
        except Exception:
            continue
    return out


def ee_to_grasp_target_from_env(
    gym_env: Any,
    pin_attr: str = "_mana_episode_grasp_target",
    ee_frame_keys: tuple[str, ...] = ("left_ee_frame", "right_ee_frame"),
) -> dict[str, float]:
    """Min over arms of the EE→pinned-bottle world distance (arrival truth).

    Answers "did the gripper physically get to the bottle?" independent of the
    grasp checker: paired with :func:`grasp_target_force_from_env` it separates a
    grasp stall into *never arrived* (distance large, force 0) vs *arrived but did
    not grip* (distance small, force 0) vs *judgment error* (force high yet the
    checker said no). Returns ``{}`` for tasks with no pinned grasp target.
    """
    pin = getattr(gym_env, pin_attr, None)
    if not isinstance(pin, dict) or not pin:
        return {}
    out: dict[str, float] = {}
    for env_id, name in pin.items():
        try:
            obj_pos = gym_env.scene[name].data.root_pos_w.torch[int(env_id)]
        except Exception:
            continue
        dmin: float | None = None
        for frame_key in ee_frame_keys:
            try:
                ee = gym_env.scene[frame_key].data.target_pos_w.torch[int(env_id), 0, :]
                d = float(torch.linalg.norm(obj_pos - ee).item())
                dmin = d if dmin is None else min(dmin, d)
            except Exception:
                continue
        if dmin is not None:
            out[str(name)] = dmin
    return out


def planner_sequence_done_from_env(gym_env: Any, attr: str = "_mana_planner_sequence_done") -> bool | None:
    """Whether the scripted planner ran its skill sequence to completion (execution truth).

    A stage failure with ``sequence_done=True`` means curobo finished every phase
    but the motion was insufficient (wrong target / open-loop miss); with
    ``False`` it means the sequence never completed (plan failure / truncation /
    timeout). Returns ``None`` for non-scripted rollouts (no flag published).
    """
    flag = getattr(gym_env, attr, None)
    if isinstance(flag, torch.Tensor):
        try:
            return bool(flag.any().item())
        except Exception:
            return None
    return None


def _format_stage_trace(snapshot: dict[str, Any], step: int) -> str:
    stages = snapshot.get("stages") or []
    num_envs = snapshot.get("num_envs", 0)
    in_flight = snapshot.get("in_flight_latched") or []
    rates = snapshot.get("completed_pass_rate") or {}
    done_n = snapshot.get("completed_episodes", 0)
    live = " ".join(f"{s}={in_flight[i] if i < len(in_flight) else 0}/{num_envs}" for i, s in enumerate(stages))
    rate_str = " ".join(f"{s}={float(rates.get(s, 0.0)):.2f}" for s in stages)
    line = f"INFO: [StageMonitor] step={step} {live} | done n={done_n} rates {rate_str}"
    miss = snapshot.get("ee_init_miss") or {}
    if miss:
        line += " | ee_init_miss " + " ".join(f"{k}={v:.3f}" for k, v in miss.items())
    force = snapshot.get("grasp_force") or {}
    if force:
        line += " | grasp_force " + " ".join(f"{k}={v:.1f}N" for k, v in force.items())
    ee2tgt = snapshot.get("ee_to_target") or {}
    if ee2tgt:
        line += " | ee2tgt " + " ".join(f"{k}={v:.3f}" for k, v in ee2tgt.items())
    seq_done = snapshot.get("seq_done")
    if seq_done is not None:
        line += f" | seq_done={int(bool(seq_done))}"
    return line


def attach_stage_progress_logger(runner: Any, evaluation_runtime: EvaluationRuntime, interval_steps: int) -> None:
    """Stream a one-line per-stage progress trace during a (possibly headless) rollout.

    Registers an ``AFTER_STEP`` hook throttled to every ``interval_steps`` and an
    ``ENV_DONES`` hook (one line per completion wave) that ``manaprint`` the
    :meth:`EvaluationRuntime.live_snapshot` so the live per-stage frontier
    (到位率/抓取率/回收率) is observable mid-rollout — not only at episode end. UI-free
    (no omni.ui import), so it works where the overlay cannot attach.
    """
    if interval_steps <= 0:
        return
    counter_key = "_stage_progress_step"

    def _on_after_step(ctx: Any, obs: Any, terminated: Any, truncated: Any, actions: Any) -> None:
        del obs, terminated, truncated, actions
        step = int(ctx.extra.get(counter_key, 0)) + 1
        ctx.extra[counter_key] = step
        if step % interval_steps != 0:
            return
        snapshot = evaluation_runtime.live_snapshot(ctx.gym_env)
        if snapshot:
            snapshot["ee_init_miss"] = ee_init_miss_from_env(ctx.gym_env)
            snapshot["grasp_force"] = grasp_target_force_from_env(ctx.gym_env)
            snapshot["ee_to_target"] = ee_to_grasp_target_from_env(ctx.gym_env)
            snapshot["seq_done"] = planner_sequence_done_from_env(ctx.gym_env)
            manaprint(_format_stage_trace(snapshot, step))

    def _on_done(ctx: Any, done: Any, obs: Any, terminated: Any, truncated: Any, actions: Any) -> None:
        del done, obs, terminated, truncated, actions
        snapshot = evaluation_runtime.live_snapshot(ctx.gym_env)
        if snapshot:
            manaprint(_format_stage_trace(snapshot, int(ctx.extra.get(counter_key, 0))))

    runner.add_hook(AFTER_STEP, _on_after_step, priority=60)
    runner.add_hook(ENV_DONES, _on_done, priority=60)
    manaprint(f"INFO: Stage progress logger attached (every {interval_steps} steps)")


def create_evaluation_runtime(env_obj: Any) -> EvaluationRuntime | None:
    from manaenv.pipeline.evaluation.log_runtime import build_log_producers

    task = getattr(env_obj, "task", None)
    if task is None:
        return None
    get_term_cfg = getattr(task, "get_evaluation_term_cfg", None)
    term_cfg = get_term_cfg() if callable(get_term_cfg) else None
    log_producers = build_log_producers(_resolve_log_producer_specs(task))
    overlay_entries, overlay_cfg = _resolve_overlay_spec(task)
    # Activation is decoupled from evaluation: a task with only log producers
    # (no evaluation rubric) still gets a record channel.
    if term_cfg is None and not log_producers:
        return None
    runtime = EvaluationRuntime(
        evaluation_term_cfg=term_cfg,
        env_obj=env_obj,
        log_producers=log_producers,
        overlay_entries=overlay_entries,
        overlay_cfg=overlay_cfg,
    )
    if not runtime._terms and not runtime._log_producers:
        return None
    return runtime
