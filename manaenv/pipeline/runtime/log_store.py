"""Per-env live log store — the runtime-readable backend for non-append subtask
log cells.

The subtask log producer (:mod:`manaenv.pipeline.evaluation.log_runtime`) stays
the only writer: every frame it publishes a non-append cell's tensor value here
via :func:`write_pending`, folded by the key's ``update_mode``. Terms (and any
``value_func``) are readers via :func:`read_log`. Mirrors
:mod:`manaenv.pipeline.runtime.active_instruction`: the store lives on the gym
env as ``env._mana_log_store`` so any holder of the env can read it.

Cross-frame visibility is the whole point. The single writer (producers) runs
strictly after every reader inside one step (the ``TerminationManager`` fires
earlier in ``step()``; evaluation sub-terms run before the producer loop), so a
two-dict ``pending``/``committed`` buffer is enough: readers always see
``committed`` (last frame's snapshot), writes land in ``pending``, and the
runtime calls :func:`commit_pending` at the end of its term-hit update. A value
written on frame N is therefore readable from frame N+1 — an accepted 1-frame
latch delay, with no extra swap hooks.

Folding identity is handled by a per-key/per-env ``last_frame`` sentinel
(``-1`` == not yet written this episode): the first write of an episode takes
the current value verbatim for every ``update_mode`` (so ``committed`` can stay
zero-initialized and reset cheaply to zeros), and subsequent writes fold against
the committed snapshot.
"""

from __future__ import annotations

from typing import Any

import torch

from manaenv.pipeline.runtime.runtime_state import runtime_state

STORE_ATTR = "_mana_log_store"
LOG_STORE_KEY = "log.store"


def ensure_log_store(env: Any, num_envs: int, device: Any) -> dict[str, Any]:
    """Return the env's live log store, creating/resizing as needed."""
    store = getattr(env, STORE_ATTR, None)
    if isinstance(store, dict) and store.get("num_envs") == num_envs:
        return store
    store = {
        "num_envs": int(num_envs),
        "device": device,
        "committed": {},
        "pending": {},
        "last_frame": {},
        "update_mode": {},
        "reset_owned": set(),
    }
    setattr(env, STORE_ATTR, store)
    _register_log_store_slot(env)
    return store


def _register_log_store_slot(env: Any) -> None:
    """Register the log store on the unified runtime-state registry.

    The store's tensors still live on ``env._mana_log_store``; the registry owns its
    reset (so the loop-boundary funnel clears it like every other per-env slot) and
    lists it under ``introspect()``. Reset zeroes the named envs and rewinds
    ``last_frame`` to ``-1`` — the same semantics the old ``reset_log_store`` had.
    """

    def _reset(ids: torch.Tensor, _env: Any = env) -> None:
        store = getattr(_env, STORE_ATTR, None)
        if not isinstance(store, dict) or ids.numel() == 0:
            return
        idx = ids.to(dtype=torch.int64, device=store["device"])
        reset_owned = store.get("reset_owned")
        reset_owned = reset_owned if isinstance(reset_owned, set) else set()
        for key in store["committed"]:
            if key in reset_owned:
                continue
            store["committed"][key][idx] = 0
            store["pending"][key][idx] = 0
            store["last_frame"][key][idx] = -1

    runtime_state(env).register(LOG_STORE_KEY, reset=_reset, owner="log")


def _ensure_key(store: dict[str, Any], key: str, template: torch.Tensor, update_mode: str) -> None:
    if key in store["committed"]:
        return
    num_envs = int(store["num_envs"])
    device = template.device
    store["committed"][key] = torch.zeros_like(template)
    store["pending"][key] = torch.zeros_like(template)
    store["last_frame"][key] = torch.full((num_envs,), -1, dtype=torch.int64, device=device)
    store["update_mode"][key] = update_mode


def _fold(update_mode: str, committed: torch.Tensor, cur: torch.Tensor, first_write: torch.Tensor) -> torch.Tensor:
    """Fold ``cur`` into ``committed`` by ``update_mode``; first writes take ``cur``.

    ``first_write`` is a ``[num_envs]`` bool mask of envs that have not been
    written yet this episode (``last_frame == -1``); for those, no value has
    accumulated so the result is ``cur`` regardless of mode.
    """
    if update_mode == "first":
        base = committed
    elif update_mode == "last":
        base = cur
    elif update_mode == "or":
        base = committed | cur
    elif update_mode == "and":
        base = committed & cur
    elif update_mode == "max":
        base = torch.maximum(committed, cur)
    elif update_mode == "min":
        base = torch.minimum(committed, cur)
    else:
        raise ValueError(f"log_store cannot fold update_mode {update_mode!r}")
    first_write_b = first_write.reshape((-1,) + (1,) * (cur.dim() - 1))
    return torch.where(first_write_b, cur, base)


def write_pending(
    env: Any,
    key: str,
    tensor: torch.Tensor,
    update_mode: str,
    current_frames: torch.Tensor,
    *,
    write_mask: torch.Tensor | None = None,
) -> None:
    """Fold *tensor* into *key*'s pending cell for the envs in *write_mask*.

    Reads the committed snapshot as the fold base, writes the folded result into
    ``pending`` (to be promoted by :func:`commit_pending`), and stamps each
    written env's ``last_frame``. Envs outside *write_mask* keep their committed
    value (so the deferred commit is a no-op for them).
    """
    num_envs = int(tensor.shape[0])
    store = ensure_log_store(env, num_envs, tensor.device)
    cur = tensor.detach().clone()
    _ensure_key(store, key, cur, update_mode)

    committed = store["committed"][key]
    last_frame = store["last_frame"][key]
    first_write = last_frame < 0
    folded = _fold(update_mode, committed, cur, first_write)

    if write_mask is not None:
        mask = write_mask.to(dtype=torch.bool, device=committed.device).reshape(-1)
        mask_b = mask.reshape((-1,) + (1,) * (folded.dim() - 1))
        folded = torch.where(mask_b, folded, committed)
    else:
        mask = torch.ones(num_envs, dtype=torch.bool, device=committed.device)
    store["pending"][key] = folded

    frames = current_frames.to(device=last_frame.device, dtype=torch.int64).reshape(-1)
    # ``first`` stamps the frame only on the initial write; other modes stamp the
    # most recent write so flush reports the last-update frame.
    stamp_mask = mask & first_write if update_mode == "first" else mask
    store["last_frame"][key] = torch.where(stamp_mask, frames, last_frame)


def write_committed(
    env: Any,
    key: str,
    tensor: torch.Tensor,
    *,
    env_ids: list[int] | torch.Tensor | None = None,
    frame_idx: int = -1,
) -> None:
    """Write a reset-owned tensor cell directly into the committed store.

    Reset events run before ``ENV_DONES`` finalization, so these cells are owned
    by reset itself: each reset overwrites the previous episode's value, and the
    loop-boundary reset must not clear them afterwards.
    """
    num_envs = int(tensor.shape[0])
    store = ensure_log_store(env, num_envs, tensor.device)
    cur = tensor.detach().clone()
    _ensure_key(store, key, cur, "last")
    reset_owned = store.setdefault("reset_owned", set())
    if isinstance(reset_owned, set):
        reset_owned.add(key)

    committed = store["committed"][key]
    pending = store["pending"][key]
    last_frame = store["last_frame"][key]
    if env_ids is None:
        committed[:] = cur
        pending[:] = cur
        last_frame[:] = int(frame_idx)
        return
    idx = torch.as_tensor(env_ids, dtype=torch.int64, device=committed.device).reshape(-1)
    committed[idx] = cur[idx]
    pending[idx] = cur[idx]
    last_frame[idx] = int(frame_idx)


def commit_pending(env: Any, env_ids: list[int] | None = None) -> None:
    """Promote ``pending`` -> ``committed`` so the next frame's readers see it."""
    store = getattr(env, STORE_ATTR, None)
    if not isinstance(store, dict):
        return
    if env_ids is None:
        for key, pending in store["pending"].items():
            store["committed"][key] = pending.clone()
        return
    if not env_ids:
        return
    idx = torch.as_tensor(list(env_ids), dtype=torch.int64, device=store["device"])
    for key, pending in store["pending"].items():
        store["committed"][key][idx] = pending[idx]


def read_log(env: Any, key: str) -> torch.Tensor | None:
    """Return a clone of *key*'s committed cell (last frame's snapshot), or ``None``.

    ``None`` means the key was never written by any producer; a consumer should
    degrade to "not present" (e.g. exclude nothing).
    """
    store = getattr(env, STORE_ATTR, None)
    if not isinstance(store, dict):
        return None
    committed = store["committed"].get(key)
    return committed.clone() if committed is not None else None


def flush_entries(env: Any, env_id: int) -> dict[str, dict[str, Any]]:
    """Return per-env final cell values as ``{key: {value, frame_idx}}`` entries.

    Keys never written this episode (``last_frame == -1``) are skipped. Tensor
    cells are converted to plain Python (scalar or list) for the episode log.
    """
    store = getattr(env, STORE_ATTR, None)
    if not isinstance(store, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    reset_owned = store.get("reset_owned")
    reset_owned = reset_owned if isinstance(reset_owned, set) else set()
    for key, committed in store["committed"].items():
        if key in reset_owned:
            continue
        last = int(store["last_frame"][key][env_id].item())
        if last < 0:
            continue
        value_t = committed[env_id]
        value = value_t.item() if value_t.dim() == 0 else value_t.tolist()
        out[key] = {"value": value, "frame_idx": last}
    return out
