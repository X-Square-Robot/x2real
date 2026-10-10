"""Per-env active-instruction stores for the wording in force this episode.

Two parallel stores live on the gym env:

* ``env._mana_active_instruction`` — vague / policy-facing instruction
* ``env._mana_active_detailed_instruction`` — optional detailed wording (e.g. a
  concrete plan) recorded alongside the demo but not fed as the model prompt

A reset-driven selector (the instruction ``kind="section"`` log producer in
:mod:`manaenv.pipeline.evaluation.log_runtime`) or a task reset event writes
these stores. Consumers:

* the recorder writes them as the demo's ``language_instruction`` /
  ``detailed_instruction``;
* the policy reads only the vague active instruction;
* ``get_prompt``-style fallbacks read the vague store when unset.

Absent / short stores degrade gracefully to ``None``.
"""

from __future__ import annotations

from typing import Any

from manaenv.pipeline.runtime.runtime_state import runtime_state

STORE_ATTR = "_mana_active_instruction"
SUBTASKS_ATTR = "_mana_active_subtasks"
PLACEHOLDERS_ATTR = "_mana_active_instruction_placeholders"
RESET_OWNED_ATTR = "_mana_active_instruction_reset_owned"
SUBTASKS_RESET_OWNED_ATTR = "_mana_active_subtasks_reset_owned"
ACTIVE_INSTRUCTION_KEY = "instruction.active"
ACTIVE_SUBTASKS_KEY = "instruction.active_subtasks"
ACTIVE_PLACEHOLDERS_KEY = "instruction.active_placeholders"

DETAILED_STORE_ATTR = "_mana_active_detailed_instruction"
DETAILED_RESET_OWNED_ATTR = "_mana_active_detailed_instruction_reset_owned"
ACTIVE_DETAILED_INSTRUCTION_KEY = "instruction.detailed"


def _ensure_store(env: Any, num_envs: int, store_attr: str, register) -> list[Any]:
    store = getattr(env, store_attr, None)
    if not isinstance(store, list) or len(store) != num_envs:
        store = [None] * num_envs
        setattr(env, store_attr, store)
        register(env)
    return store


def _register_slot(
    env: Any,
    *,
    store_attr: str,
    reset_owned_attr: str,
    key: str,
) -> None:
    def _reset(ids: Any, _env: Any = env) -> None:
        store = getattr(_env, store_attr, None)
        if not isinstance(store, list):
            return
        reset_owned = getattr(_env, reset_owned_attr, None)
        reset_owned = reset_owned if isinstance(reset_owned, set) else set()
        for env_id in ids.tolist():
            if int(env_id) in reset_owned:
                continue
            if 0 <= int(env_id) < len(store):
                store[int(env_id)] = None

    def _snapshot(_env: Any = env) -> Any:
        store = getattr(_env, store_attr, None)
        return list(store) if isinstance(store, list) else None

    runtime_state(env).register(key, reset=_reset, snapshot=_snapshot, owner="instruction", monitor=True)


def ensure_active_instruction_store(env: Any, num_envs: int) -> list[Any]:
    """Return the env's active-instruction store, creating/resizing as needed."""
    return _ensure_store(
        env,
        num_envs,
        STORE_ATTR,
        lambda e: _register_slot(
            e, store_attr=STORE_ATTR, reset_owned_attr=RESET_OWNED_ATTR, key=ACTIVE_INSTRUCTION_KEY
        ),
    )


def ensure_active_detailed_instruction_store(env: Any, num_envs: int) -> list[Any]:
    """Return the env's detailed-instruction store, creating/resizing as needed."""
    return _ensure_store(
        env,
        num_envs,
        DETAILED_STORE_ATTR,
        lambda e: _register_slot(
            e,
            store_attr=DETAILED_STORE_ATTR,
            reset_owned_attr=DETAILED_RESET_OWNED_ATTR,
            key=ACTIVE_DETAILED_INSTRUCTION_KEY,
        ),
    )


def _write_store(
    env: Any,
    env_id: int,
    text: Any,
    num_envs: int | None,
    *,
    reset_owned: bool,
    ensure,
    reset_owned_attr: str,
) -> None:
    if num_envs is None:
        num_envs = int(getattr(env, "num_envs", env_id + 1))
    store = ensure(env, num_envs)
    if 0 <= env_id < len(store):
        store[env_id] = text
        owned = getattr(env, reset_owned_attr, None)
        if not isinstance(owned, set):
            owned = set()
            setattr(env, reset_owned_attr, owned)
        if reset_owned:
            owned.add(int(env_id))
        else:
            owned.discard(int(env_id))


def write_active_instruction(
    env: Any,
    env_id: int,
    text: Any,
    num_envs: int | None = None,
    *,
    reset_owned: bool = False,
) -> None:
    """Write *text* as env *env_id*'s active instruction for this episode."""
    _write_store(
        env,
        env_id,
        text,
        num_envs,
        reset_owned=reset_owned,
        ensure=ensure_active_instruction_store,
        reset_owned_attr=RESET_OWNED_ATTR,
    )


def write_active_detailed_instruction(
    env: Any,
    env_id: int,
    text: Any,
    num_envs: int | None = None,
    *,
    reset_owned: bool = False,
) -> None:
    """Write *text* as env *env_id*'s detailed instruction for this episode."""
    _write_store(
        env,
        env_id,
        text,
        num_envs,
        reset_owned=reset_owned,
        ensure=ensure_active_detailed_instruction_store,
        reset_owned_attr=DETAILED_RESET_OWNED_ATTR,
    )


def resolve_active_instruction(env: Any, env_id: int) -> str | None:
    """Read env *env_id*'s active instruction, or ``None`` if unset/absent."""
    store = getattr(env, STORE_ATTR, None)
    if isinstance(store, (list, tuple)) and 0 <= env_id < len(store):
        value = store[env_id]
        if value:
            return str(value)
    return None


def resolve_active_detailed_instruction(env: Any, env_id: int) -> str | None:
    """Read env *env_id*'s detailed instruction, or ``None`` if unset/absent."""
    store = getattr(env, DETAILED_STORE_ATTR, None)
    if isinstance(store, (list, tuple)) and 0 <= env_id < len(store):
        value = store[env_id]
        if value:
            return str(value)
    return None


def read_active_instructions(env: Any) -> list[Any] | None:
    """Return a copy of the whole per-env store, or ``None`` if absent/empty."""
    store = getattr(env, STORE_ATTR, None)
    if isinstance(store, (list, tuple)) and any(value for value in store):
        return list(store)
    return None


def read_active_detailed_instructions(env: Any) -> list[Any] | None:
    """Return a copy of the detailed per-env store, or ``None`` if absent/empty."""
    store = getattr(env, DETAILED_STORE_ATTR, None)
    if isinstance(store, (list, tuple)) and any(value for value in store):
        return list(store)
    return None


def ensure_active_subtasks_store(env: Any, num_envs: int) -> list[Any]:
    """Return the env's per-env active-subtasks store, creating/resizing as needed."""
    return _ensure_store(
        env,
        num_envs,
        SUBTASKS_ATTR,
        lambda e: _register_slot(
            e,
            store_attr=SUBTASKS_ATTR,
            reset_owned_attr=SUBTASKS_RESET_OWNED_ATTR,
            key=ACTIVE_SUBTASKS_KEY,
        ),
    )


def write_active_subtasks(
    env: Any,
    env_id: int,
    subtasks: list[str] | tuple[str, ...] | None,
    num_envs: int | None = None,
    *,
    reset_owned: bool = False,
) -> None:
    """Write rendered subtask strings for env *env_id* this episode."""
    if num_envs is None:
        num_envs = int(getattr(env, "num_envs", env_id + 1))
    store = ensure_active_subtasks_store(env, num_envs)
    if 0 <= env_id < len(store):
        cleaned = [str(item).strip() for item in list(subtasks or []) if str(item).strip()]
        store[env_id] = cleaned or None
        owned = getattr(env, SUBTASKS_RESET_OWNED_ATTR, None)
        if not isinstance(owned, set):
            owned = set()
            setattr(env, SUBTASKS_RESET_OWNED_ATTR, owned)
        if reset_owned:
            owned.add(int(env_id))
        else:
            owned.discard(int(env_id))


def resolve_active_subtasks(env: Any, env_id: int) -> list[str] | None:
    """Read env *env_id*'s active subtasks, or ``None`` if unset/absent."""
    store = getattr(env, SUBTASKS_ATTR, None)
    if isinstance(store, (list, tuple)) and 0 <= env_id < len(store):
        value = store[env_id]
        if isinstance(value, (list, tuple)) and value:
            return [str(item) for item in value if str(item).strip()]
    return None


def read_active_subtasks(env: Any) -> list[Any] | None:
    """Return a copy of the whole per-env subtasks store, or ``None`` if empty."""
    store = getattr(env, SUBTASKS_ATTR, None)
    if isinstance(store, (list, tuple)) and any(value for value in store):
        return list(store)
    return None


def ensure_active_instruction_placeholders_store(env: Any, num_envs: int) -> list[Any]:
    """Return the per-env instruction placeholder dict store."""
    return _ensure_store(
        env,
        num_envs,
        PLACEHOLDERS_ATTR,
        lambda e: _register_slot(
            e,
            store_attr=PLACEHOLDERS_ATTR,
            reset_owned_attr=RESET_OWNED_ATTR,
            key=ACTIVE_PLACEHOLDERS_KEY,
        ),
    )


def write_active_instruction_placeholders(
    env: Any,
    env_id: int,
    placeholders: dict[str, str] | None,
    num_envs: int | None = None,
) -> None:
    """Write rendered ``{nut}`` / ``{nut_color}`` placeholder values for log text."""
    if num_envs is None:
        num_envs = int(getattr(env, "num_envs", env_id + 1))
    store = ensure_active_instruction_placeholders_store(env, num_envs)
    if 0 <= env_id < len(store):
        if isinstance(placeholders, dict) and placeholders:
            store[env_id] = {str(k): str(v) for k, v in placeholders.items() if str(k).strip()}
        else:
            store[env_id] = None


def resolve_active_instruction_placeholders(env: Any, env_id: int) -> dict[str, str] | None:
    """Read env *env_id*'s instruction placeholders, or ``None`` if unset."""
    store = getattr(env, PLACEHOLDERS_ATTR, None)
    if isinstance(store, (list, tuple)) and 0 <= env_id < len(store):
        value = store[env_id]
        if isinstance(value, dict) and value:
            return {str(k): str(v) for k, v in value.items()}
    return None
