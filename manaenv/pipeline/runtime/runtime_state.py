"""Unified per-env runtime-state registry.

The IsaacLab runtime env (``isaaclab.envs.ManagerBasedRLEnv``) is vendored and
cannot be subclassed to add fields, so per-env mutable state (termination
buffers, the stage machine's done/failed/dwell/reason tensors, planner pins,
future scene ledgers) is attached dynamically as ``_mana_*`` attributes. Today
each owner does its own ``setattr`` + ad-hoc reset, with no single place to
reset / introspect / monitor.

This module is that single place. State still LIVES on the env (device /
num_envs / episode-reset lifecycle — the termination func only gets ``env``);
this registry is the *registration + reset + introspection + monitoring-snapshot*
surface over it. It generalizes the ``_register_tensor_episode_state`` /
``reset_stateful_termination_terms`` seed that lived in ``terms/terminations.py``.

Keep it dependency-light (torch only, no manaenv term/sim imports) so terms can
delegate to it without an import cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

_REGISTRY_ATTR = "_mana_runtime_state"

# A slot's reset takes the reset env-id tensor and clears that slot for those envs.
ResetFn = Callable[[torch.Tensor], None]
SnapshotFn = Callable[[], Any]


def runtime_state(env: Any) -> RuntimeStateRegistry:
    """Get-or-create the env's :class:`RuntimeStateRegistry` (lazy, so it works
    even on paths/tests that skip the build-time attach)."""
    reg = getattr(env, _REGISTRY_ATTR, None)
    if not isinstance(reg, RuntimeStateRegistry):
        reg = RuntimeStateRegistry(env)
        setattr(env, _REGISTRY_ATTR, reg)
    return reg


def _normalize_env_ids(env: Any, env_ids: Any | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
    if hasattr(env_ids, "detach"):
        return env_ids.detach().to(device=env.device, dtype=torch.int64).flatten()
    if isinstance(env_ids, int):
        return torch.tensor([int(env_ids)], dtype=torch.int64, device=env.device)
    return torch.tensor([int(e) for e in env_ids], dtype=torch.int64, device=env.device)


@dataclass
class _Slot:
    key: str
    owner: str
    kind: str  # "tensor" | "generic"
    reset: ResetFn
    snapshot: SnapshotFn | None = None
    monitor: bool = False
    tensor: torch.Tensor | None = None  # set for kind == "tensor"


class RuntimeStateRegistry:
    """Per-env runtime state: named, namespaced slots with unified reset,
    introspection, and a monitoring snapshot.

    Two slot kinds:
      * ``tensor(key, ...)`` — the registry OWNS the buffer, returns it, and the
        owner writes into it in-place each step. Reset fills it for the given
        env-ids; snapshot reads it. Used by the stage machine.
      * ``register(key, reset=..., snapshot=...)`` — a GENERIC slot holding a
        reset closure (and optional snapshot) for non-registry-owned state (e.g.
        buffers a term ``setattr``s itself, or a dict like the grasp-target pin).
        This is the shape the ``terminations.py`` seed delegates to.

    Keys are namespaced ``"<owner>.<name>"`` so ownership is legible and
    collisions impossible.
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        self._slots: dict[str, _Slot] = {}

    # ---- registration -------------------------------------------------

    def tensor(
        self,
        key: str,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        fill: Any,
        owner: str,
        monitor: bool = False,
    ) -> torch.Tensor:
        """Get-or-create a registry-owned per-env tensor slot."""
        shape = tuple(int(s) for s in shape)
        slot = self._slots.get(key)
        if (
            slot is not None
            and slot.tensor is not None
            and tuple(slot.tensor.shape) == shape
            and slot.tensor.dtype == dtype
        ):
            return slot.tensor
        buf = torch.full(shape, fill, dtype=dtype, device=self._env.device)
        env = self._env

        def _reset(ids: torch.Tensor, _buf: torch.Tensor = buf, _fill: Any = fill) -> None:
            if _buf.ndim == 0:
                _buf.fill_(_fill)
            elif _buf.shape[0] == env.num_envs:
                _buf[ids] = _fill

        self._slots[key] = _Slot(
            key=key, owner=owner, kind="tensor", reset=_reset, snapshot=(lambda _b=buf: _b), monitor=monitor, tensor=buf
        )
        return buf

    def register(
        self,
        key: str,
        *,
        reset: ResetFn,
        snapshot: SnapshotFn | None = None,
        owner: str = "",
        monitor: bool = False,
    ) -> None:
        """Register a generic slot: a ``reset(env_ids)`` closure (+ optional
        ``snapshot()``) for state the registry does not own. Idempotent by key."""
        self._slots[key] = _Slot(key=key, owner=owner, kind="generic", reset=reset, snapshot=snapshot, monitor=monitor)

    def unregister(self, key: str) -> None:
        self._slots.pop(key, None)

    def has(self, key: str) -> bool:
        return key in self._slots

    def get_tensor(self, key: str) -> torch.Tensor | None:
        """The registry-owned tensor for *key*, or ``None`` if the slot does not
        exist / is not a tensor slot (read-only peek — does NOT create)."""
        slot = self._slots.get(key)
        return slot.tensor if slot is not None else None

    # ---- reset --------------------------------------------------------

    def reset(self, env_ids: Any | None = None, *, keys: set[str] | None = None) -> tuple[str, ...]:
        """Clear slots for *env_ids* (all envs when ``None``). When *keys* is
        given, only those slots reset — lets a single owner (e.g. the stage
        machine) reset just its own slots via the same path."""
        ids = _normalize_env_ids(self._env, env_ids)
        cleared: list[str] = []
        for slot in list(self._slots.values()):
            if keys is not None and slot.key not in keys:
                continue
            if callable(slot.reset):
                slot.reset(ids)
                cleared.append(slot.key)
        return tuple(cleared)

    # ---- introspection / monitoring ----------------------------------

    def introspect(self) -> list[dict]:
        """List every slot's metadata (key, owner, kind, shape/dtype, monitor)
        — the "what state exists, who owns it" view for management."""
        out: list[dict] = []
        for slot in self._slots.values():
            entry: dict[str, Any] = {"key": slot.key, "owner": slot.owner, "kind": slot.kind, "monitor": slot.monitor}
            if slot.tensor is not None:
                entry["shape"] = tuple(slot.tensor.shape)
                entry["dtype"] = str(slot.tensor.dtype)
            out.append(entry)
        return out

    def snapshot(self, env_id: int | None = None) -> dict:
        """JSON-able values of the monitor-flagged slots (all envs, or one
        ``env_id``). Best-effort — a raising slot is skipped."""
        out: dict[str, Any] = {}
        for slot in self._slots.values():
            if not slot.monitor or slot.snapshot is None:
                continue
            try:
                out[slot.key] = self._jsonable(slot.snapshot(), env_id)
            except Exception:
                continue
        return out

    @staticmethod
    def _jsonable(val: Any, env_id: int | None) -> Any:
        if isinstance(val, torch.Tensor):
            tensor = val.detach().cpu()
            if env_id is not None and tensor.ndim >= 1 and 0 <= env_id < tensor.shape[0]:
                tensor = tensor[env_id]
            return tensor.tolist()
        return val
