from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG
from manaenv.pipeline.generators.task.ref_text import render_ref_text


class LogComponent(TaskComponent):
    """Compiles per-episode subtask log declarations into log-producer specs.

    YAML example (under ``component_sections``)::

        log_component:
          entries:
            entered:
              # no `trigger`/`func` → defaults to `trigger: enter` (on subtask entry)
              value: "place @target.name"
            picked_cup:
              trigger: success       # trigger = this task's `success` term
              value: "@target.name"  # @role.<attr> rendered to str at finalize
            peak_force:
              trigger: block_placed  # trigger term
              func: read_peak_force  # per-env value function sampled at trigger

    Each entry becomes a ``kind="subtask"`` spec appended to
    ``runtime_cfgs["log_producers"]``; :func:`manaenv.pipeline.evaluation.log_runtime.build_log_producers`
    materializes the triggers/values at runtime. ``update_mode`` decides how
    repeated (per-frame) writes to the same key fold, and splits the key into two
    views:

    * ``append`` (legacy ``every``) — a growing event list (ledger history),
      recorded on each rising edge; only drained to the episode log at finalize,
      **not** runtime-readable.
    * non-append (``first``/``last``/``max``/``min``/``and``/``or``, default
      ``first``) — a single current value (cell). If its ``value_func`` returns a
      vectorizable tensor the cell is published into the per-env live store and is
      **runtime-readable** by terms one frame later; otherwise it is just a single
      ledger entry. No ``readable`` flag — readability is automatic.

    The ``trigger`` is either ``enter`` (fire on entering the subtask — a
    composite stage's predecessors all latched, or episode start for a standalone
    task), or a named termination term. When neither ``trigger`` nor ``func`` is
    given it defaults to ``enter``. (``trigger`` is used instead of ``on`` because
    YAML 1.1 coerces an unquoted ``on`` key to boolean ``True``.) Value is
    optional (``value`` static, or ``func`` / ``value_func`` per-env); absent →
    records ``None`` at the trigger frame. ``mode``/``every`` are accepted as
    deprecated aliases for ``update_mode``/``append``.
    """

    _RESERVED = frozenset({"trigger", "func", "value", "value_func", "update_mode", "mode", "params"})
    _VALID_UPDATE_MODES = frozenset({"first", "last", "append", "max", "min", "and", "or"})
    _LEGACY_MODE_ALIASES = {"every": "append"}

    def generate(self, cfg: LogComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        roles = roles or {}
        specs: list[dict[str, Any]] = []
        for key, raw in cfg.entries.items():
            specs.append(self._compile_entry(key, raw or {}, roles))
        return {"log_producers": specs}

    def _resolve_update_mode(self, key: str, raw: dict[str, Any]) -> str:
        legacy = raw.get("mode")
        update_mode = raw.get("update_mode")
        resolved = update_mode if update_mode is not None else legacy
        if resolved is None:
            return "first"
        resolved = self._LEGACY_MODE_ALIASES.get(str(resolved), str(resolved))
        if resolved not in self._VALID_UPDATE_MODES:
            raise ValueError(
                f"log entry {key!r}: update_mode must be one of {sorted(self._VALID_UPDATE_MODES)} (got {resolved!r})"
            )
        return resolved

    def _compile_entry(self, key: str, raw: dict[str, Any], roles: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError(f"log entry {key!r} must map to a dict, got {type(raw).__name__}")
        update_mode = self._resolve_update_mode(key, raw)

        spec: dict[str, Any] = {"kind": "subtask", "key": key, "update_mode": update_mode}
        trigger = raw.get("trigger")
        func = raw.get("func")
        if trigger == "enter":
            # `enter` fires when the subtask is entered (rising edge of its active
            # state); composite tasks repoint it to the per-stage `dag_stage_active`
            # column, standalone tasks fire it at episode start.
            spec["enter"] = True
            if func:
                spec["value_func"] = func
        elif trigger:
            spec["trigger"] = trigger
            # With a separate `trigger`, `func` acts as the per-env value source.
            if func:
                spec["value_func"] = func
        elif func:
            spec["func"] = func
        else:
            # Default trigger: log on entering the subtask.
            spec["enter"] = True

        params = raw.get("params")
        if params:
            spec["params"] = dict(params)

        if "value" in raw:
            value = raw["value"]
            spec["static_value"] = render_ref_text(value, roles) if isinstance(value, str) else value
        elif raw.get("value_func"):
            spec["value_func"] = raw["value_func"]
        return spec


@dataclass
class LogComponentCFG(TaskComponentCFG):
    """Subtask log configuration — ``entries`` maps log key → trigger/value spec.

    See :class:`LogComponent` for the entry schema. ``entries`` keys are the
    log keys that appear in ``episode_record["log"]``.
    """

    class_type: type = LogComponent
    renders_text_refs = True  # log ``value`` fields carry @role.<attr> interpolation
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
