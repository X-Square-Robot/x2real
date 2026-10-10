from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from .model import RandomizationContext

T = TypeVar("T")


def _seed_from_parts(parts: Iterable[object]) -> int:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "big")


@dataclass
class RandomStream:
    _random: random.Random

    @classmethod
    def from_context(cls, context: RandomizationContext, *parts: object) -> RandomStream:
        seed = _seed_from_parts(
            (
                context.namespace,
                context.seed,
                context.scene_id,
                context.episode_index,
                context.variant_index,
                *parts,
            )
        )
        return cls(random.Random(seed))

    def spawn(self, *parts: object) -> RandomStream:
        seed = _seed_from_parts((self._random.random(), *parts))
        return RandomStream(random.Random(seed))

    def uniform(self, low: float, high: float) -> float:
        return self._random.uniform(low, high)

    def randint(self, low: int, high: int) -> int:
        return self._random.randint(low, high)

    def choice(self, values: Sequence[T]) -> T:
        if not values:
            raise ValueError("cannot choose from an empty sequence")
        return self._random.choice(values)


def context_for_reset(
    *,
    base_seed: int,
    reset_index: int,
    env_id: str | int | None = None,
    task_id: str | None = None,
    namespace: str = "asset_randomizer",
) -> RandomizationContext:
    """Build a reproducible context that changes with each environment reset."""

    if reset_index < 0:
        raise ValueError("reset_index must be non-negative")
    effective_namespace = f"{namespace}:{task_id}" if task_id else namespace
    return RandomizationContext(
        seed=int(base_seed),
        scene_id=None if env_id is None else str(env_id),
        episode_index=int(reset_index),
        namespace=effective_namespace,
    )
