"""Asynchronous chunk scheduler."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from manaenv.policy.chunk_scheduler.base import REFILL_CONTEXT_KEY, ChunkSchedulerBase, ChunkSchedulerBaseCFG
from manaenv.utils.debug_print import manaprint
from manaenv.utils.tensor_utils import normalize_env_ids


@dataclass
class _PendingRefill:
    future: Future
    overlap_by_env: dict[int, int]
    generation_by_env: dict[int, int]


class AsyncChunkScheduler(ChunkSchedulerBase):
    """Prefetch the next chunk in the background before the buffer is empty."""

    def __init__(self) -> None:
        super().__init__()
        self._executor: ThreadPoolExecutor | None = None
        self._pending: _PendingRefill | None = None
        self._env_generations: dict[int, int] = {}

    def get_actions(
        self,
        *,
        env,
        observation,
        num_envs,
        action_dim,
        device,
        buffer,
        client,
        run_refill,
        fill_chunks,
    ):
        env_indices = [i for i in range(num_envs) if buffer.needs_refill(i, client.refill_threshold)]
        if env_indices:
            consumed = self._consume_pending(fill_chunks=fill_chunks)
            missing = [i for i in env_indices if buffer.needs_refill(i, client.refill_threshold)]
            if missing:
                if consumed:
                    manaprint("WARNING: async scheduler missing envs " f"{missing}; falling back to sync refill")
                fill_chunks(run_refill(env, observation, missing))

        self._maybe_start_refill(
            env=env,
            observation=observation,
            num_envs=num_envs,
            buffer=buffer,
            client=client,
            run_refill=run_refill,
        )
        return buffer.pop_step(num_envs, action_dim, device)

    def reset(self, env_ids=None) -> None:
        ids = normalize_env_ids(env_ids)
        if ids is None:
            pending = self._pending
            self._pending = None
            known_ids = set(self._env_generations)
            if pending is not None:
                known_ids.update(pending.generation_by_env)
                pending.future.cancel()
            for env_idx in known_ids:
                self._env_generations[env_idx] = self._env_generations.get(env_idx, 0) + 1
            return

        for env_idx in ids:
            self._env_generations[env_idx] = self._env_generations.get(env_idx, 0) + 1
        pending = self._pending
        if pending is not None and set(pending.generation_by_env).issubset(ids):
            self._pending = None
            pending.future.cancel()

    def cleanup(self) -> None:
        self.reset()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mana_policy_async_scheduler",
            )
        return self._executor

    def _maybe_start_refill(
        self,
        *,
        env,
        observation,
        num_envs,
        buffer,
        client,
        run_refill,
    ) -> None:
        if self._pending is not None:
            return
        overlap_trigger = self._overlap_rows(client)
        if overlap_trigger <= 0:
            return

        env_indices: list[int] = []
        overlap_by_env: dict[int, int] = {}
        cfg = self._typed_cfg()
        refill_threshold = max(1, int(client.refill_threshold))
        # Cropped refills replace the buffer wholesale.  Waiting until the old
        # buffer is empty is therefore part of the alignment contract: with a
        # larger threshold, unconsumed rows would be dropped while the new
        # chunk is still cropped by the full overlap captured below.
        if cfg.crop_overlap and refill_threshold != 1:
            raise ValueError(
                "async_chunk_scheduler with crop_overlap=True requires " f"refill_threshold=1, got {refill_threshold}"
            )
        for env_idx in range(num_envs):
            chunk = buffer.frames.get(env_idx)
            if chunk is None:
                continue
            remaining = int(chunk.shape[0])
            if 0 < remaining <= overlap_trigger:
                env_indices.append(env_idx)
                overlap_by_env[env_idx] = remaining if cfg.crop_overlap else 0
        if not env_indices:
            return

        refill_observation = dict(observation)
        refill_observation[REFILL_CONTEXT_KEY] = {
            env_idx: {
                "execute_horizon_rows": max(1, int(client.execute_horizon)),
                "advanced_rows": max(0, int(client.execute_horizon) - overlap_by_env[env_idx]),
                "overlap_rows": overlap_by_env[env_idx],
                "crop_overlap": bool(cfg.crop_overlap),
            }
            for env_idx in env_indices
        }
        future = self._get_executor().submit(
            run_refill,
            env,
            refill_observation,
            list(env_indices),
        )
        self._pending = _PendingRefill(
            future=future,
            overlap_by_env=overlap_by_env,
            generation_by_env={env_idx: self._env_generations.get(env_idx, 0) for env_idx in env_indices},
        )

    def _overlap_rows(self, client) -> int:
        horizon = max(1, int(client.execute_horizon))
        ratio = float(self._typed_cfg().inference_trigger_ratio)
        ratio = min(max(ratio, 0.0), 1.0)
        trigger_step = min(max(int(round(horizon * ratio)), 1), horizon)
        return max(horizon - trigger_step, 0)

    def _consume_pending(self, *, fill_chunks) -> bool:
        pending = self._pending
        if pending is None:
            return False
        self._pending = None

        try:
            chunks = pending.future.result()
        except Exception as exc:
            manaprint("WARNING: async scheduler failed; " f"falling back to sync refill: {exc}")
            return True

        current_chunks = {
            env_idx: chunk
            for env_idx, chunk in chunks.items()
            if pending.generation_by_env.get(int(env_idx)) == self._env_generations.get(int(env_idx), 0)
        }
        if current_chunks:
            fill_chunks(self._crop_overlaps(current_chunks, pending.overlap_by_env))
        return True

    def _typed_cfg(self) -> AsyncChunkSchedulerCFG:
        if self.cfg is None:
            return AsyncChunkSchedulerCFG()
        if not isinstance(self.cfg, AsyncChunkSchedulerCFG):
            raise TypeError(f"AsyncChunkScheduler requires AsyncChunkSchedulerCFG, got {type(self.cfg).__name__}")
        return self.cfg

    @staticmethod
    def _crop_overlaps(
        chunks: dict[int, torch.Tensor | None],
        overlap_by_env: dict[int, int],
    ):
        out: dict[int, torch.Tensor | None] = {}
        for env_idx, chunk in chunks.items():
            if chunk is None:
                out[env_idx] = None
                continue
            overlap = max(0, int(overlap_by_env.get(int(env_idx), 0)))
            out[env_idx] = _crop_overlap(chunk, overlap)
        return out


@dataclass
class AsyncChunkSchedulerCFG(ChunkSchedulerBaseCFG):
    """CFG for asynchronous background refill scheduling."""

    inference_trigger_ratio: float = 0.5
    crop_overlap: bool = True
    class_type: type = AsyncChunkScheduler


def _crop_overlap(chunk: torch.Tensor, overlap: int) -> torch.Tensor:
    if overlap <= 0:
        return chunk
    if chunk.ndim == 2:
        start = min(overlap, int(chunk.shape[0]))
        return chunk[start:]
    if chunk.ndim == 3:
        start = min(overlap, int(chunk.shape[1]))
        return chunk[:, start:]
    return chunk
