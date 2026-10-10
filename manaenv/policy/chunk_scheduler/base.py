"""Chunk refill schedulers for ManaPolicy.

Schedulers decide when a policy chunk should be refilled and how the next
chunk is connected to the existing buffer. The refill implementation itself
stays in ``client.chunk_executor``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

# Private key used only inside ManaPolicy's in-process chunk pipeline.  Async
# schedulers attach timing/overlap facts to an observation; ManaPolicy moves the
# per-env value onto the encoded request so a wire interceptor can consume it.
# The key is removed before the request reaches a model server.
REFILL_CONTEXT_KEY = "_mana_refill_context"


@dataclass
class ChunkSchedulerBaseCFG:
    """Base CFG for chunk scheduler plugins."""

    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    class_type: type | None = None


class ChunkSchedulerBase(ABC):
    """Runtime base for client-side chunk scheduling."""

    def __init__(self) -> None:
        self.cfg: ChunkSchedulerBaseCFG | None = None

    def load(self, cfg: ChunkSchedulerBaseCFG) -> None:
        self.cfg = cfg

    @abstractmethod
    def get_actions(
        self,
        *,
        env: Any,
        observation: dict[str, Any],
        num_envs: int,
        action_dim: int,
        device: Any,
        buffer: Any,
        client: Any,
        run_refill: Callable[
            [Any, dict[str, Any], list[int]],
            dict[int, torch.Tensor | None],
        ],
        fill_chunks: Callable[[dict[int, torch.Tensor | None]], None],
    ) -> torch.Tensor:
        """Return the next action batch after applying scheduler logic."""

    def reset(self, env_ids: Any = None) -> None:
        """Optional per-env reset hook."""

    def cleanup(self) -> None:
        """Optional resource release hook."""
