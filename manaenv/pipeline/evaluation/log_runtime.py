"""Per-env subtask log producers, sharing EvaluationRuntime's record channel.

The log schema appended onto each ``episode_record`` is intentionally minimal::

    episode_record["log"] = { <key>: [ {value, frame_idx}, ... ] }

There is no ``owner`` field: same-key entries simply append into one list and
are distinguished by their ``value``.

Each subtask producer fires on the rising edge of its trigger (a clean
not-hit -> hit transition, so a sticky/latched predicate counts once per event,
not once per frame). ``update_mode`` decides how repeated (per-frame) writes to
the same key fold into the stored value, and splits the key into two views:

* ``append`` (legacy ``every``) — a growing event list = the ledger/history
  view, recorded on each rising edge. **Not runtime-readable** (no single value
  to vectorize); only drained to ``episode_record["log"]`` at finalize.
* non-append (``first``/``last``/``max``/``min``/``and``/``or``) — a single
  current value (cell). If the value is a **vectorizable tensor** it is published
  every frame into the per-env live store
  (:mod:`manaenv.pipeline.runtime.log_store`) folded by ``update_mode`` and is
  **runtime-readable** by terms one frame later; the cell's final per-env value
  is flushed to ``episode_record["log"]`` at finalize. A non-tensor (e.g. a
  ``@ref`` asset name) only keeps a single ledger entry (``first``/``last``) and
  is not readable.

``mode``/``every`` are accepted as deprecated aliases for
``update_mode``/``append``; the default stays ``first``.

``LogProducer`` is a general base so reset-driven *section* log can plug into the
very same ``runtime_cfgs["log_producers"]`` channel and the same
``episode_record["log"]`` shape — it reserves a ``seed_reset`` hook that subtask
producers leave as a no-op. Two families are implemented:

* ``SubtaskLogProducer`` (trigger-driven) — fires on the rising edge of its
  trigger and accumulates entries by ``mode``;
* ``SectionInstructionProducer`` (reset-driven) — picks one instruction wording
  per env at episode reset, writes the rendered text into the per-env
  active-instruction store (:mod:`manaenv.pipeline.runtime.active_instruction`),
  and records the chosen variant label under ``frame_idx == -1``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import torch

from manaenv.pipeline.runtime.log_store import write_pending

_TriggerFn = Callable[[Any], Any]
_ValueFn = Callable[[Any], Sequence[Any]]


class _Unset:
    """Sentinel distinguishing "no static value" from a real ``None`` value."""


_UNSET = _Unset()


def _to_python(value: Any) -> Any:
    """Convert a tensor (scalar or row) into a plain Python value for the ledger."""
    if isinstance(value, torch.Tensor):
        return value.item() if value.dim() == 0 else value.tolist()
    return value


class LogProducer:
    """Base contract for a single per-episode log key.

    Two families share this contract so they can live in one runtime and one
    record channel:

    * trigger-driven (``SubtaskLogProducer``) — evaluated every ``AFTER_STEP``;
    * reset-driven section log (future) — seeded at episode reset via
      :meth:`seed_reset`, ``frame_idx`` fixed to ``-1``.
    """

    key: str

    def ensure_state(self, num_envs: int, device: Any) -> None:
        raise NotImplementedError

    def update(self, env: Any, current_frames: torch.Tensor, reset_mask: torch.Tensor | None) -> None:
        """Trigger-driven step hook. No-op for reset-driven producers."""
        del env, current_frames, reset_mask

    def seed_reset(self, env: Any, env_ids: Sequence[int]) -> None:
        """Reset-driven seed hook for section log. No-op for subtask producers."""
        del env, env_ids

    def collect(self, env_id: int) -> tuple[str, list[dict[str, Any]]] | None:
        """Return ``(key, entries)`` accumulated this episode, or ``None`` if empty.

        ``entries`` is a list so a single producer can contribute multiple log
        entries (``update_mode="append"``); the runtime extends ``log[key]`` with
        them.
        """
        raise NotImplementedError

    def snapshot(self, env_id: int) -> Any | None:
        """Return this env's current log value for live display, or ``None``.

        Read-only overlay hook: the single Python value most representative of the
        key *right now* (mid-episode), independent of the finalize-time ledger.
        Base default is ``None`` so producers that carry no runtime-readable value
        simply do not surface in the overlay.
        """
        del env_id
        return None

    def reset(self, env_ids: Sequence[int]) -> None:
        raise NotImplementedError


_VALID_UPDATE_MODES = frozenset({"first", "last", "append", "max", "min", "and", "or"})
_LEGACY_MODE_ALIASES = {"every": "append"}


def normalize_update_mode(update_mode: Any, legacy_mode: Any = None) -> str:
    """Resolve the effective ``update_mode`` from new/legacy fields.

    ``update_mode`` wins; otherwise the deprecated ``mode`` field is mapped
    (``every`` -> ``append``). Defaults to ``first``.
    """
    raw = update_mode if update_mode is not None else legacy_mode
    if raw is None:
        return "first"
    resolved = _LEGACY_MODE_ALIASES.get(str(raw), str(raw))
    if resolved not in _VALID_UPDATE_MODES:
        raise ValueError(f"update_mode must be one of {sorted(_VALID_UPDATE_MODES)}, got {raw!r}")
    return resolved


class SubtaskLogProducer(LogProducer):
    """Record per-env ``value`` for ``key``, folded by ``update_mode``.

    A rising edge is a per-env not-hit -> hit transition (tracked via
    ``_prev_hit``), so a sticky/latched predicate records one event, not one per
    frame. ``value`` comes either from a fixed ``static_value`` (e.g. a ``@ref``
    asset name resolved at finalize) or a per-env ``value_fn(env)``.

    Two views split on ``update_mode``:

    * ``append`` — append every rising-edge event into ``_entries`` (the ledger
      history; emitted by :meth:`collect`). Never enters the live store.
    * non-append + tensor ``value_fn`` — every frame the value is published into
      the per-env live store (:mod:`manaenv.pipeline.runtime.log_store`) folded by
      ``update_mode``, so terms can read it one frame later; the final per-env
      value is flushed to the episode log by the runtime (not :meth:`collect`).
    * non-append + non-tensor value — keep a single ledger entry (``first`` keeps
      the first rising-edge event, anything else overwrites with the latest).
    """

    def __init__(
        self,
        key: str,
        trigger_fn: _TriggerFn,
        *,
        static_value: Any = _UNSET,
        value_fn: _ValueFn | None = None,
        update_mode: str = "first",
    ) -> None:
        self.key = key
        self._trigger_fn = trigger_fn
        self._static_value = static_value
        self._value_fn = value_fn
        self.update_mode = normalize_update_mode(update_mode)
        self._prev_hit: torch.Tensor | None = None
        self._entries: list[list[dict[str, Any]]] = []

    def ensure_state(self, num_envs: int, device: Any) -> None:
        if self._prev_hit is not None and self._prev_hit.shape[0] == num_envs:
            return
        self._prev_hit = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._entries = [[] for _ in range(num_envs)]

    def update(self, env: Any, current_frames: torch.Tensor, reset_mask: torch.Tensor | None) -> None:
        assert self._prev_hit is not None
        device = current_frames.device
        hit = self._trigger_fn(env)
        if not isinstance(hit, torch.Tensor):
            hit = torch.as_tensor(hit, dtype=torch.bool, device=device)
        hit = hit.to(dtype=torch.bool, device=device).reshape(-1)
        reset_bool = None
        if reset_mask is not None:
            reset_bool = reset_mask.to(dtype=torch.bool, device=device).reshape(-1)
            # Skip envs IsaacLab just reset inside step(): their fresh initial
            # state must not record an event belonging to the next episode.
            hit = hit & (~reset_bool)
        rising = hit & (~self._prev_hit)
        self._prev_hit = hit

        values = self._value_fn(env) if self._value_fn is not None else None

        if self.update_mode != "append" and isinstance(values, torch.Tensor):
            # Live store path: publish the per-frame tensor (folded by update_mode)
            # so terms can read the latched value one frame later.

            write_mask = (~reset_bool) if reset_bool is not None else None
            write_pending(env, self.key, values, self.update_mode, current_frames, write_mask=write_mask)
            return

        if bool(rising.any()):
            frames = current_frames.to(device=device, dtype=torch.int64)
            for env_id in rising.nonzero(as_tuple=False).reshape(-1).tolist():
                value = values[env_id] if values is not None else self._resolve_static()
                self._record(env_id, {"value": _to_python(value), "frame_idx": int(frames[env_id].item())})

    def _record(self, env_id: int, entry: dict[str, Any]) -> None:
        bucket = self._entries[env_id]
        if self.update_mode == "append":
            bucket.append(entry)
        elif self.update_mode == "first":
            if not bucket:
                bucket.append(entry)
        else:  # last / max / min / and / or on a non-tensor value: keep the latest
            bucket[:] = [entry]

    def _resolve_static(self) -> Any:
        return None if isinstance(self._static_value, _Unset) else self._static_value

    def collect(self, env_id: int) -> tuple[str, list[dict[str, Any]]] | None:
        if not self._entries or not self._entries[env_id]:
            return None
        return self.key, list(self._entries[env_id])

    def snapshot(self, env_id: int) -> Any | None:
        """Latest recorded ledger value for this env (``append`` returns the last
        event; ``first``/``last`` the single current value). ``None`` before the
        trigger fires or for tensor-only live cells not tracked in ``_entries``."""
        if not self._entries or not (0 <= env_id < len(self._entries)):
            return None
        bucket = self._entries[env_id]
        if not bucket:
            return None
        return bucket[-1].get("value")

    def reset(self, env_ids: Sequence[int]) -> None:
        if self._prev_hit is None or not env_ids:
            return
        idx = torch.as_tensor(list(env_ids), dtype=torch.int64, device=self._prev_hit.device)
        self._prev_hit[idx] = False
        for env_id in env_ids:
            self._entries[int(env_id)] = []


class SectionInstructionProducer(LogProducer):
    """Reset-driven instruction selector: pick one wording per env at reset.

    A section's fact is decided once per episode at reset. Each option carries a
    log ``value`` (the variant label, e.g. ``"polite"``) and a ``rendered``
    payload (the finalize-rendered instruction text). On :meth:`seed_reset` each
    reset env draws an option from a single persistent RNG (seedable for
    reproducibility) and:

    * writes the rendered text into the per-env active-instruction store on the
      env (the single source of truth recorder / policy read), and
    * records ``{value, frame_idx: -1}`` so the chosen variant rides into
      ``episode_record["log"][key]`` for traceability.

    ``update`` is a no-op (nothing is sampled mid-episode); :meth:`collect`
    returns the chosen variant label as the episode's single log entry.
    """

    def __init__(
        self,
        key: str,
        options: Sequence[dict[str, Any]],
        *,
        seed: int | None = None,
        tensor_key: str | None = None,
    ) -> None:
        if not options:
            raise ValueError(f"SectionInstructionProducer {key!r} requires at least one option")
        self.key = key
        self._options = list(options)
        self._seed = seed
        self._tensor_key = str(tensor_key).strip() if tensor_key else None
        self._tensor_width = self._infer_tensor_width(self._options, tensor_key=self._tensor_key)
        self._rng: random.Random | None = None
        self._num_envs = 0
        # Per-env chosen (value, rendered); None until this env is seeded.
        self._chosen: list[tuple[Any, Any] | None] = []
        self._chosen_tensor_values: list[Sequence[int] | None] = []

    def ensure_state(self, num_envs: int, device: Any) -> None:
        del device
        # One persistent RNG so successive resets draw *different* variants (not
        # the same one every episode); reproducible when seeded.
        if self._rng is None:
            self._rng = random.Random(self._seed)
        if self._num_envs != num_envs:
            self._num_envs = num_envs
            self._chosen = [None] * num_envs
            self._chosen_tensor_values = [None] * num_envs

    def seed_reset(self, env: Any, env_ids: Sequence[int]) -> None:
        from manaenv.pipeline.runtime.active_instruction import RESET_OWNED_ATTR, write_active_instruction

        if self._rng is None:
            self._rng = random.Random(self._seed)
        num_envs = int(getattr(env, "num_envs", self._num_envs or (max(env_ids) + 1 if env_ids else 1)))
        # Do not overwrite runtime_instruction / twist goals (reset_owned=True).
        owned = getattr(env, RESET_OWNED_ATTR, None)
        owned = owned if isinstance(owned, set) else set()
        for env_id in env_ids:
            index = self._rng.randrange(len(self._options))
            option = self._options[index]
            value = option.get("value")
            rendered = option.get("rendered")
            if 0 <= int(env_id) < len(self._chosen):
                self._chosen[int(env_id)] = (value, rendered)
                self._chosen_tensor_values[int(env_id)] = option.get("tensor_value")
            if int(env_id) in owned:
                continue
            write_active_instruction(env, int(env_id), rendered, num_envs=num_envs)

    def update(self, env: Any, current_frames: torch.Tensor, reset_mask: torch.Tensor | None) -> None:
        if not self._tensor_key or self._tensor_width <= 0:
            return

        device = current_frames.device
        values = torch.zeros((self._num_envs, self._tensor_width), dtype=torch.int64, device=device)
        for env_id, tensor_value in enumerate(self._chosen_tensor_values):
            if tensor_value is None:
                continue
            values[env_id] = torch.as_tensor(tensor_value, dtype=torch.int64, device=device)
        write_mask = None
        if reset_mask is not None:
            write_mask = ~reset_mask.to(dtype=torch.bool, device=device).reshape(-1)
        write_pending(env, self._tensor_key, values, "last", current_frames, write_mask=write_mask)

    def collect(self, env_id: int) -> tuple[str, list[dict[str, Any]]] | None:
        if not self._chosen or env_id >= len(self._chosen):
            return None
        chosen = self._chosen[env_id]
        if chosen is None:
            return None
        value, _rendered = chosen
        return self.key, [{"value": value, "frame_idx": -1}]

    def snapshot(self, env_id: int) -> Any | None:
        """This env's reset-chosen variant value, or ``None`` before it is seeded."""
        if not self._chosen or not (0 <= env_id < len(self._chosen)):
            return None
        chosen = self._chosen[env_id]
        if chosen is None:
            return None
        value, _rendered = chosen
        return value

    def reset(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            if 0 <= int(env_id) < len(self._chosen):
                self._chosen[int(env_id)] = None
                self._chosen_tensor_values[int(env_id)] = None

    @staticmethod
    def _infer_tensor_width(options: Sequence[dict[str, Any]], *, tensor_key: str | None) -> int:
        if not tensor_key:
            return 0
        widths = set()
        for option in options:
            value = option.get("tensor_value")
            if value is None:
                raise ValueError(
                    f"SectionInstructionProducer tensor_key {tensor_key!r} requires tensor_value on every option"
                )
            widths.add(len(value))
        if len(widths) != 1:
            raise ValueError(f"SectionInstructionProducer tensor_key {tensor_key!r} requires fixed-width tensor values")
        return next(iter(widths))


def _read_termination_term(env: Any, term_name: str) -> torch.Tensor:
    """Read a named active termination term as a bool tensor (all-False if absent)."""
    num_envs = int(getattr(env, "num_envs", 1))
    device = getattr(env, "device", "cpu")
    manager = getattr(env, "termination_manager", None)
    active = getattr(manager, "active_terms", ()) if manager is not None else ()
    if manager is None or term_name not in active:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    value = manager.get_term(term_name)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.bool, device=device)
    return value.to(dtype=torch.bool, device=device).reshape(-1)


def _resolve_trigger(spec: dict[str, Any]) -> _TriggerFn:
    """Build the trigger predicate from a (possibly declarative) spec.

    Precedence: explicit ``trigger_fn`` callable (test/advanced escape hatch) →
    gated composite column (``stage_index``) → subtask entry (``enter``) → named
    termination term (``trigger``) → self-contained term function (``func`` + ``params``).

    A composite ``trigger: enter`` is rewritten upstream into a ``stage_index``
    against the ``dag_stage_active`` column, so it resolves through the
    ``stage_index`` branch; a standalone task's ``enter`` reaches the ``enter``
    branch here and is always-active, firing on its (frame-0) rising edge once per
    episode.
    """
    trigger_fn = spec.get("trigger_fn")
    if callable(trigger_fn):
        return trigger_fn
    if "stage_index" in spec:
        from manaenv.terms import terminations

        stage_index = int(spec["stage_index"])
        extras_key = spec.get("extras_key", "dag_stage_done")
        return lambda env: terminations.dag_stage_latched(env, stage_index=stage_index, extras_key=extras_key)
    if spec.get("enter"):
        return lambda env: torch.ones(
            int(getattr(env, "num_envs", 1)), dtype=torch.bool, device=getattr(env, "device", "cpu")
        )
    if spec.get("trigger"):
        term_name = str(spec["trigger"])
        return lambda env: _read_termination_term(env, term_name)
    if spec.get("func"):
        from manaenv.pipeline.generators.task.term_discovery import resolve_task_term

        func = resolve_task_term("termination", str(spec["func"]))
        params = dict(spec.get("params") or {})
        return lambda env: func(env, **params)
    raise ValueError(
        f"log producer {spec.get('key')!r} needs a trigger (trigger_fn / stage_index / enter / trigger / func)"
    )


def _resolve_value(spec: dict[str, Any]) -> tuple[_ValueFn | None, Any]:
    """Build ``(value_fn, static_value)`` from a (possibly declarative) spec.

    A fixed ``static_value`` (e.g. a ``@ref`` asset name resolved at finalize)
    takes precedence; otherwise a per-env ``value_func`` term function or an
    explicit ``value_fn`` callable; otherwise no value (records ``None``).
    """
    if "static_value" in spec:
        return None, spec["static_value"]
    if callable(spec.get("value_fn")):
        return spec["value_fn"], _UNSET
    if spec.get("value_func"):
        from manaenv.pipeline.generators.task.term_discovery import resolve_task_term

        func = resolve_task_term("termination", str(spec["value_func"]))
        params = dict(spec.get("params") or {})
        return (lambda env: func(env, **params)), _UNSET
    return None, _UNSET


def build_log_producers(specs: Sequence[dict[str, Any]] | None) -> list[LogProducer]:
    """Compile log-producer spec dicts into runtime producers.

    Specs are plain (declarative) dicts so they survive the assembly deepcopy
    and are materialized at runtime, post-clone. Dispatch on ``kind``:
    ``"subtask"`` (trigger-driven) and ``"section"`` (reset-driven instruction
    selection). Triggers/values may be declared (``trigger`` / ``func`` /
    ``value`` / ``value_func`` / ``stage_index``) or passed as ready callables
    (``trigger_fn`` / ``value_fn``) for programmatic use.
    """
    if not specs:
        return []
    producers: list[LogProducer] = []
    for spec in specs:
        kind = spec.get("kind", "subtask")
        if kind == "subtask":
            value_fn, static_value = _resolve_value(spec)
            producers.append(
                SubtaskLogProducer(
                    key=spec["key"],
                    trigger_fn=_resolve_trigger(spec),
                    static_value=static_value,
                    value_fn=value_fn,
                    update_mode=normalize_update_mode(spec.get("update_mode"), spec.get("mode")),
                )
            )
        elif kind == "section":
            producers.append(
                SectionInstructionProducer(
                    key=spec["key"],
                    options=spec["options"],
                    seed=spec.get("seed"),
                    tensor_key=spec.get("tensor_key"),
                )
            )
        else:
            raise ValueError(f"unsupported log producer kind: {kind!r}")
    return producers
