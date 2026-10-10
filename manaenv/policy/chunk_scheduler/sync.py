"""Synchronous chunk scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from manaenv.policy.chunk_scheduler.base import ChunkSchedulerBase, ChunkSchedulerBaseCFG


class SyncChunkScheduler(ChunkSchedulerBase):
    """Refill missing chunks before popping the next action row."""

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
            fill_chunks(run_refill(env, observation, env_indices))
        return buffer.pop_step(num_envs, action_dim, device)


@dataclass
class SyncChunkSchedulerCFG(ChunkSchedulerBaseCFG):
    """CFG for the synchronous chunk scheduler."""

    class_type: type = SyncChunkScheduler
