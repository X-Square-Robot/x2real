"""Chunk-refill policies — CFG-reflected plugins selected by the
``client.chunk_executor`` segment of a policy yaml (PLAN.md §10.5).

A refill policy decides, given the set of envs that need a new action
chunk this step, **how** to drive the wire pipeline (encode_request →
inference_fn → decode_response with optional wire-interceptor hooks
and chunk-postprocessor passes) and which env(s) to fill from each
returned chunk.

Two shipped strategies cover every implementation in tree:

  * :class:`PerEnvIndependent` — wallx / openpi / dp_image: each env
    queries one chunk independently. Used when the inference path is
    single-env (the server reads only env 0, or the local model is
    looped per-env).
  * :class:`GlobalSync` — gr00t: one batched forward fills every env
    that needs a refill. Used when the inference path is multi-env
    (the model batches all envs in a single forward pass and returns
    ``(num_envs, T, action_dim)``).

Reflection contract: yaml writes the explicit CFG key, e.g.
``chunk_executor: {per_env_independent: {execute_horizon: 32}}``, which
the global lazy registry reflects to :class:`PerEnvIndependentCFG`.
``ManaPolicy`` instantiates ``cfg.class_type()`` and calls
``load(cfg)``; the instance is invoked with a fixed set of
pipeline-callable keyword arguments and returns a ``{env_idx: chunk}``
mapping that ManaPolicy writes into its :class:`ChunkBuffer`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from manaenv.policy.chunk_scheduler.base import REFILL_CONTEXT_KEY


def _run_one(
    request: dict[str, Any],
    *,
    env: Any,
    inference_fn: Callable[[dict[str, Any]], dict[str, Any]],
    decode_response: Callable[..., torch.Tensor | None],
    wire_hooks: Sequence[Any],
    postprocs: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    env_idx: int | None,
) -> torch.Tensor | None:
    """Shared single-shot pipeline: hooks → inference → hooks → decode → postprocs."""
    for hook in wire_hooks:
        request = hook.on_request(request, env_idx=env_idx)
    # Scheduler context is process-local plumbing, never part of a model's wire
    # protocol.  RTC consumes it; this pop also makes every other interceptor
    # safely transparent when a scheduler provides context.
    request.pop(REFILL_CONTEXT_KEY, None)
    raw = inference_fn(request)
    for hook in wire_hooks:
        raw = hook.on_response(raw, env_idx=env_idx)
    chunk = decode_response(raw, env=env, env_idx=env_idx)
    if chunk is None:
        return None
    for fn in postprocs:
        chunk = fn(chunk)
    return chunk


@dataclass
class RefillPolicyBaseCFG:
    """Base CFG for chunk refill policy plugins."""

    execute_horizon: int = 1
    refill_threshold: int = 1
    class_type: type | None = None


class RefillPolicyBase(ABC):
    """Runtime base for client-side chunk refill strategies."""

    def __init__(self) -> None:
        self.cfg: RefillPolicyBaseCFG | None = None

    def load(self, cfg: RefillPolicyBaseCFG) -> None:
        self.cfg = cfg

    @abstractmethod
    def __call__(
        self,
        *,
        env: Any,
        observation: dict[str, Any],
        env_indices: list[int],
        encode_request: Callable[..., dict[str, Any]],
        inference_fn: Callable[[dict[str, Any]], dict[str, Any]],
        decode_response: Callable[..., torch.Tensor | None],
        wire_hooks: Sequence[Any],
        postprocs: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    ) -> dict[int, torch.Tensor | None]:
        """Return ``{env_idx: chunk}`` for the envs that requested a refill."""


class PerEnvIndependent(RefillPolicyBase):
    """Refill chunks one env at a time (single-env servers / models).

    The encoded request carries the full batched observation (encoders
    do not slice); per-env state lives in the wire interceptors. One
    inference call per env, results keyed by env_idx.
    """

    def __call__(
        self,
        *,
        env: Any,
        observation: dict[str, Any],
        env_indices: list[int],
        encode_request: Callable[..., dict[str, Any]],
        inference_fn: Callable[[dict[str, Any]], dict[str, Any]],
        decode_response: Callable[..., torch.Tensor | None],
        wire_hooks: Sequence[Any],
        postprocs: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    ) -> dict[int, torch.Tensor | None]:
        out: dict[int, torch.Tensor | None] = {}
        for env_idx in env_indices:
            request = encode_request(observation, env_idx=env_idx)
            out[env_idx] = _run_one(
                request,
                env=env,
                inference_fn=inference_fn,
                decode_response=decode_response,
                wire_hooks=wire_hooks,
                postprocs=postprocs,
                env_idx=env_idx,
            )
        return out


@dataclass
class PerEnvIndependentCFG(RefillPolicyBaseCFG):
    class_type: type = PerEnvIndependent


class GlobalSync(RefillPolicyBase):
    """Refill every env from one batched forward (multi-env models).

    The implementation's ``decode_response`` must return a tensor of
    shape ``(num_envs, T, action_dim)``; the result is split per env and
    rows for ``env_indices`` are returned. Used by gr00t whose upstream
    model batches all envs in a single forward pass.
    """

    def __call__(
        self,
        *,
        env: Any,
        observation: dict[str, Any],
        env_indices: list[int],
        encode_request: Callable[..., dict[str, Any]],
        inference_fn: Callable[[dict[str, Any]], dict[str, Any]],
        decode_response: Callable[..., torch.Tensor | None],
        wire_hooks: Sequence[Any],
        postprocs: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    ) -> dict[int, torch.Tensor | None]:
        request = encode_request(observation, env_idx=None)
        chunk_batch = _run_one(
            request,
            env=env,
            inference_fn=inference_fn,
            decode_response=decode_response,
            wire_hooks=wire_hooks,
            postprocs=postprocs,
            env_idx=None,
        )
        if chunk_batch is None:
            return {env_idx: None for env_idx in env_indices}
        if chunk_batch.ndim != 3:
            raise ValueError(
                "global_sync: decode_response must return (num_envs, T, action_dim) "
                f"for batched refills; got shape={tuple(chunk_batch.shape)}"
            )
        return {env_idx: chunk_batch[env_idx] for env_idx in env_indices}


@dataclass
class GlobalSyncCFG(RefillPolicyBaseCFG):
    class_type: type = GlobalSync
