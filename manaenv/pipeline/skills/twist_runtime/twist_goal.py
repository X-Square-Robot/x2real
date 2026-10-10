"""Per-episode turnable twist goal — TASK/SCENE authority.

Each reset samples an initial knob panel temperature and a distinct target
temperature. The twist skill reads the pin to plan the wrist arc; the success
checker compares the live joint against the pinned target.

Panel<->joint unit conversion lives on
:class:`manaenv.assets.objects.object_library.KnobA`. This module only samples
integer temperatures and carries per-episode metadata.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

EPISODE_TURNABLE_TWIST_GOAL_ATTR = "_mana_episode_turnable_twist_goal"
EPISODE_TURNABLE_EPISODE_RECORD_ATTR = "_mana_episode_turnable_twist_record"


@dataclass(frozen=True)
class TurnableTwistGoal:
    """Pinned twist goal for one env / one episode."""

    joint_name: str
    initial_temperature: int
    target_temperature: int
    temperature_max: int = 99

    @property
    def initial_display(self) -> int:
        return int(self.initial_temperature)

    @property
    def target_display(self) -> int:
        return int(self.target_temperature)


@dataclass
class TurnableTwistEpisodeRecord:
    """Per-episode twist metadata for instructions, HDF5 env_args, and staged export."""

    joint_name: str
    initial_temperature: int
    target_temperature: int
    task_instruction: str
    node_instructions: tuple[str, ...]
    result: str = ""

    def node_instruction(self, node_index: int | None) -> str:
        if node_index is None:
            return ""
        idx = int(node_index)
        if 0 <= idx < len(self.node_instructions):
            return str(self.node_instructions[idx])
        return ""

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "initial_temperature": int(self.initial_temperature),
            "target_temperature": int(self.target_temperature),
            "result": str(self.result or ""),
            "task_instruction": str(self.task_instruction),
            "node_instructions": list(self.node_instructions),
        }


def render_twist_instruction_template(
    template: str,
    *,
    initial: int,
    target: int,
    result: str = "",
) -> str:
    """Fill ``{initial}``, ``{target}``, ``{result}`` placeholders in task prose."""
    text = str(template or "")
    return (
        text.replace("{initial}", str(int(initial)))
        .replace("{target}", str(int(target)))
        .replace("{result}", str(result or ""))
    )


def resolve_task_instruction_templates(env: Any) -> tuple[str, list[str]]:
    """Read graph-level task + subtask templates from the bound Mana task."""
    assembled = getattr(env, "_mana_assembled_environment", None)
    task = getattr(assembled, "task", None) if assembled is not None else None
    runtime_cfgs = getattr(task, "runtime_cfgs", None) if task is not None else None
    subtasks = [str(item) for item in list((runtime_cfgs or {}).get("subtasks") or []) if str(item).strip()]
    prompt = ""
    if assembled is not None:
        getter = getattr(assembled, "get_prompt", None)
        if callable(getter):
            prompt = str(getter() or "").strip()
    if not prompt and task is not None:
        prompt = str(getattr(getattr(task, "cfg", None), "prompt", "") or "").strip()
    if not prompt and runtime_cfgs is not None:
        prompt = str(runtime_cfgs.get("instruction") or "").strip()
    return prompt, subtasks


def build_episode_record(
    goal: TurnableTwistGoal,
    *,
    task_template: str,
    node_templates: list[str],
) -> TurnableTwistEpisodeRecord:
    initial = int(goal.initial_temperature)
    target = int(goal.target_temperature)
    node_instructions = tuple(
        render_twist_instruction_template(text, initial=initial, target=target) for text in node_templates
    )
    return TurnableTwistEpisodeRecord(
        joint_name=str(goal.joint_name),
        initial_temperature=initial,
        target_temperature=target,
        task_instruction=render_twist_instruction_template(task_template, initial=initial, target=target),
        node_instructions=node_instructions,
    )


def write_episode_record_store(env: Any, env_id: int, record: TurnableTwistEpisodeRecord) -> None:
    store = getattr(env, EPISODE_TURNABLE_EPISODE_RECORD_ATTR, None)
    if not isinstance(store, dict):
        store = {}
        setattr(env, EPISODE_TURNABLE_EPISODE_RECORD_ATTR, store)
    store[int(env_id)] = record


def read_episode_record(env: Any, env_idx: int) -> TurnableTwistEpisodeRecord | None:
    store = getattr(env, EPISODE_TURNABLE_EPISODE_RECORD_ATTR, None)
    if not isinstance(store, dict):
        return None
    record = store.get(int(env_idx))
    return record if isinstance(record, TurnableTwistEpisodeRecord) else None


def set_episode_result(env: Any, env_idx: int, *, success: bool) -> None:
    record = read_episode_record(env, int(env_idx))
    if record is None:
        return
    record.result = "passed" if success else "failed"


def sample_twist_goal_pair_temperature(
    *,
    initial_temperature_range: tuple[int, int] = (10, 50),
    target_delta_temperature_range: tuple[int, int] = (-10, 10),
    min_delta_temperature: int = 3,
    temperature_max: int = 99,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """Sample ``(initial_temperature, target_temperature)`` in panel units.

    Initial temperature is uniform in ``initial_temperature_range``. Target
    temperature is ``initial + delta`` where ``delta`` is uniform in
    ``target_delta_temperature_range`` and ``|delta| >= min_delta_temperature``.
    Both are clamped to ``[0, temperature_max]``.
    """
    lo_i, hi_i = int(initial_temperature_range[0]), int(initial_temperature_range[1])
    lo_d, hi_d = int(target_delta_temperature_range[0]), int(target_delta_temperature_range[1])
    lo_i, hi_i = min(lo_i, hi_i), max(lo_i, hi_i)
    lo_d, hi_d = min(lo_d, hi_d), max(lo_d, hi_d)
    lo_i = max(0, min(int(temperature_max), lo_i))
    hi_i = max(0, min(int(temperature_max), hi_i))
    min_delta = max(1, int(min_delta_temperature))
    picker = rng or random

    initial_temp = picker.randint(lo_i, hi_i)
    delta = 0
    for _ in range(32):
        candidate = picker.randint(lo_d, hi_d)
        if abs(candidate) >= min_delta:
            delta = candidate
            break
    if abs(delta) < min_delta:
        delta = min_delta if picker.random() >= 0.5 else -min_delta

    target_temp = max(0, min(int(temperature_max), initial_temp + delta))
    if abs(target_temp - initial_temp) < min_delta:
        for fallback in (initial_temp + min_delta, initial_temp - min_delta):
            clamped = max(0, min(int(temperature_max), fallback))
            if abs(clamped - initial_temp) >= min_delta:
                target_temp = clamped
                break

    return int(initial_temp), int(target_temp)


def write_goal_store(env: Any, env_id: int, goal: TurnableTwistGoal, *, store_attr: str) -> None:
    store = getattr(env, store_attr, None)
    if not isinstance(store, dict):
        store = {}
        setattr(env, store_attr, store)
    store[int(env_id)] = goal


def read_pinned_twist_goal(
    env: Any, env_idx: int, *, store_attr: str = EPISODE_TURNABLE_TWIST_GOAL_ATTR
) -> TurnableTwistGoal | None:
    store = getattr(env, store_attr, None)
    if not isinstance(store, dict):
        return None
    goal = store.get(int(env_idx))
    return goal if isinstance(goal, TurnableTwistGoal) else None
