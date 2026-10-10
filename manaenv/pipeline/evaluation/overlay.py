"""Real-time evaluation score overlay for Isaac Sim viewport sidebar.

Displays the task instruction, per-term hit status, and cumulative score in an
omni.ui docked window during benchmark sessions.
"""

from __future__ import annotations

from typing import Any

from manaenv.pipeline.evaluation.runtime import EvaluationRuntime
from manaenv.pipeline.runtime.env_runner import AFTER_ACTION, AFTER_STEP, ENV_DONES, SESSION_END
from manaenv.utils.debug_print import manaprint


class EvaluationOverlay:
    """omni.ui sidebar window showing the task instruction and live scores."""

    _HIT_COLOR = 0xFF00CC00
    _MISS_COLOR = 0xFFAAAAAA
    _SCORE_COLOR = 0xFFFFDD44
    _EDIT_MODE_COLOR = 0xFF00CCFF
    _EDIT_DONE_COLOR = 0xFF44FF44
    _IDLE_HINT_COLOR = 0xFFFF8844

    _CUSTOM_COLOR = 0xFFFFDD44

    def __init__(
        self,
        evaluation_runtime: EvaluationRuntime,
        env_id: int = 0,
        overlay_entries: list[dict[str, Any]] | None = None,
        show_evaluation: bool = True,
        window_config: dict[str, Any] | None = None,
    ) -> None:
        import omni.ui as ui

        self._runtime = evaluation_runtime
        self._env_id = env_id
        self._gym_env: Any = None
        self._labels: list[Any] = []
        self._stage_rows: list[Any] = []
        self._score_label: Any = None
        self._status_label: Any = None
        self._task_label: Any = None
        self._window: Any = None
        self._overlay_entries = list(overlay_entries or [])
        self._custom_labels: list[Any] = []
        self._show_evaluation = bool(show_evaluation)
        task_instruction = self._resolve_task_instruction()

        num_terms = len(evaluation_runtime._terms) if self._show_evaluation else 0
        # Runtime-filled instructions are long; reserve wrap room so shape/color
        # wording is not clipped by the fixed sidebar height.
        task_block = 96 if task_instruction else 72
        height = (
            40
            + task_block
            + 28 * len(self._overlay_entries)
            + (70 + 28 * num_terms * 2 if self._show_evaluation else 0)
        )
        window_cfg = window_config if isinstance(window_config, dict) else {}
        window_width = int(window_cfg.get("width", 420))
        position_x = int(window_cfg.get("position_x", 0))
        if window_cfg.get("anchor") == "top_right":
            position_x = max(int(ui.Workspace.get_main_window_width()) - window_width, 0)
        self._window = ui.Window(
            "Evaluation Score",
            width=window_width,
            height=height,
            visible=True,
            position_x=position_x,
            position_y=int(window_cfg.get("position_y", 0)),
        )
        with self._window.frame:
            with ui.VStack(spacing=4, height=0):
                self._status_label = ui.Label("", style={"font_size": 13}, visible=False)
                self._task_label = ui.Label(
                    f"Task: {task_instruction}" if task_instruction else "Task: -",
                    word_wrap=True,
                    style={"color": self._CUSTOM_COLOR, "font_size": 18},
                )
                if self._overlay_entries or self._show_evaluation:
                    ui.Separator(height=6)
                if self._overlay_entries:
                    for entry in self._overlay_entries:
                        label = str(entry.get("label") or "")
                        placeholder = f"{label}: -" if label else "-"
                        clbl = ui.Label(
                            placeholder,
                            style={
                                "color": entry.get("font_color") or self._CUSTOM_COLOR,
                                "font_size": entry.get("font_size") or 14,
                            },
                        )
                        self._custom_labels.append(clbl)
                    if self._show_evaluation:
                        ui.Separator(height=6)
                if self._show_evaluation:
                    ui.Label(
                        f"Max Score: {evaluation_runtime.max_possible_score:g}",
                        style={"color": self._SCORE_COLOR, "font_size": 14},
                    )
                    ui.Separator(height=4)
                    for term in evaluation_runtime._terms:
                        lbl = ui.Label(
                            f"[ ] {term.label}  (0 / {term.score:g})",
                            style={"color": self._MISS_COLOR, "font_size": 13},
                        )
                        self._labels.append(lbl)
                    ui.Separator(height=6)
                    ui.Label("Stage progress (all envs)", style={"color": self._SCORE_COLOR, "font_size": 13})
                    for term in evaluation_runtime._terms:
                        srow = ui.Label(f"{term.label}: -", style={"color": self._MISS_COLOR, "font_size": 13})
                        self._stage_rows.append(srow)
                    ui.Separator(height=6)
                    self._score_label = ui.Label(
                        f"Score: 0 / {evaluation_runtime.max_possible_score:g}",
                        style={"color": self._SCORE_COLOR, "font_size": 15},
                    )

    def _resolve_task_instruction(self) -> str:
        # Prefer the per-reset active instruction (shape/color filled); fall back
        # to the static task prompt used before the first runtime sample reset.
        gym_env = self._gym_env
        if gym_env is not None:
            try:
                from manaenv.pipeline.runtime.active_instruction import resolve_active_instruction

                active = resolve_active_instruction(gym_env, int(self._env_id))
                if active:
                    return str(active).strip()
            except Exception as exc:
                manaprint(f"WARNING: Failed to resolve active instruction for evaluation overlay: {exc}")

        env_obj = getattr(self._runtime, "env_obj", None)
        get_prompt = getattr(env_obj, "get_prompt", None)
        if not callable(get_prompt):
            return ""
        try:
            return str(get_prompt() or "").strip()
        except Exception as exc:
            manaprint(f"WARNING: Failed to resolve task instruction for evaluation overlay: {exc}")
            return ""

    def _refresh_task_instruction(self) -> None:
        if self._task_label is None:
            return
        text = self._resolve_task_instruction()
        self._task_label.text = f"Task: {text}" if text else "Task: -"

    def refresh(self) -> None:
        self._refresh_task_instruction()
        self._refresh_custom_entries()
        if self._task_label is not None:
            text = self._resolve_task_instruction()
            if text:
                self._task_label.text = f"Task: {text}"
        if not self._show_evaluation:
            return
        if not self._runtime._term_hits:
            return
        total = 0.0
        for i, term in enumerate(self._runtime._terms):
            hits_buf = self._runtime._term_hits.get(term.name)
            if hits_buf is None:
                continue
            hit = bool(hits_buf[self._env_id].item())
            pts = term.score if hit else 0.0
            total += pts
            mark = "[X]" if hit else "[ ]"
            color = self._HIT_COLOR if hit else self._MISS_COLOR
            self._labels[i].text = f"{mark} {term.label}  ({pts:g} / {term.score:g})"
            self._labels[i].style = {"color": color, "font_size": 13}
        if self._score_label is not None:
            self._score_label.text = f"Score: {total:g} / {self._runtime.max_possible_score:g}"

        if self._stage_rows and self._gym_env is not None:
            snapshot = self._runtime.live_snapshot(self._gym_env)
            stages = snapshot.get("stages") or []
            in_flight = snapshot.get("in_flight_latched") or []
            num_envs = snapshot.get("num_envs", 0)
            rates = snapshot.get("completed_pass_rate") or {}
            for i, label in enumerate(stages):
                if i >= len(self._stage_rows):
                    break
                active = in_flight[i] if i < len(in_flight) else 0
                rate = float(rates.get(label, 0.0))
                color = self._HIT_COLOR if active else self._MISS_COLOR
                self._stage_rows[i].text = f"{label}: {active}/{num_envs} active · {rate:.0%} done"
                self._stage_rows[i].style = {"color": color, "font_size": 13}

    def _refresh_custom_entries(self) -> None:
        if not self._custom_labels:
            return
        rows = self._runtime.overlay_snapshot(self._env_id)
        for i, row in enumerate(rows):
            if i >= len(self._custom_labels):
                break
            value = row.get("value")
            shown = "-" if value is None else str(value)
            value = str(shown)
            # Multiline values start on their own row; inline otherwise.
            sep = ":\n" if "\n" in value else ": "
            label = str(row.get("label") or "")
            self._custom_labels[i].text = f"{label}{sep}{value}" if label else value

    def set_status(self, text: str, editing: bool = False, idle: bool = False) -> None:
        """Update the status line at the top of the overlay window."""
        if self._status_label is None:
            return
        self._status_label.text = text
        self._status_label.visible = bool(text)
        if idle:
            color = self._IDLE_HINT_COLOR
        elif editing:
            color = self._EDIT_MODE_COLOR
        else:
            color = self._EDIT_DONE_COLOR
        self._status_label.style = {"color": color, "font_size": 13}

    def focus(self) -> None:
        """Raise the score overlay above camera preview windows."""
        if self._window is not None:
            self._window.focus()

    def destroy(self) -> None:
        if self._window is not None:
            self._window.visible = False
            self._window.destroy()
            self._window = None
        self._labels.clear()
        self._custom_labels.clear()
        self._score_label = None
        self._status_label = None
        self._task_label = None


def attach_evaluation_overlay(
    runner: Any,
    evaluation_runtime: EvaluationRuntime,
    env_id: int = 0,
    show_evaluation: bool = True,
    window_config: dict[str, Any] | None = None,
) -> EvaluationOverlay | None:
    """Create an EvaluationOverlay and wire it into the runner lifecycle.

    ``show_evaluation`` toggles the built-in evaluation section; the task's
    declared ``overlay_entries`` (custom log fields) render regardless. Returns
    the overlay instance so callers can access set_status().
    """
    try:
        overlay = EvaluationOverlay(
            evaluation_runtime,
            env_id=env_id,
            overlay_entries=evaluation_runtime.overlay_entries,
            show_evaluation=show_evaluation,
            window_config=window_config,
        )
    except Exception as exc:
        manaprint(f"WARNING: Failed to create EvaluationOverlay: {exc}")
        return None

    def _refresh_on_override(ctx: Any, obs: Any, actions: Any) -> None:
        del obs, actions
        if "override_obs" in ctx.extra and not ctx.extra.get("allow_indefinite_skip_step", False):
            overlay._gym_env = ctx.gym_env
            overlay.refresh()

    def _refresh_after_step(ctx: Any, obs: Any, terminated: Any, truncated: Any, actions: Any) -> None:
        del obs, terminated, truncated, actions
        overlay._gym_env = ctx.gym_env
        overlay.refresh()

    def _refresh_on_done(ctx: Any, done: Any, obs: Any, terminated: Any, truncated: Any, actions: Any) -> None:
        del done, obs, terminated, truncated, actions
        overlay._gym_env = ctx.gym_env
        overlay.refresh()

    def _destroy_overlay(ctx: Any) -> None:
        del ctx
        overlay.destroy()

    runner.add_hook(AFTER_ACTION, _refresh_on_override, priority=50)
    runner.add_hook(AFTER_STEP, _refresh_after_step, priority=50)
    runner.add_hook(ENV_DONES, _refresh_on_done, priority=50)
    runner.add_hook(SESSION_END, _destroy_overlay, priority=99)
    manaprint("INFO: Evaluation score overlay attached")
    return overlay
