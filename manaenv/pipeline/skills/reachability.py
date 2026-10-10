"""Small batched reachability selector used by evaluation skills.

The public distribution keeps the skill-facing contract but does not ship the
training, teleoperation, or navigation planner stack.  An evaluator supplied by
an optional robot backend may implement ``batch_reachable``; this helper only
coordinates those queries across candidate stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class ReachableSelection:
    """Selected arm, candidate index, and accumulated evaluator cost."""

    arm: str
    index: int
    path_cost: float


def _best_goal(
    *,
    evaluator: Any,
    observation: dict[str, Any],
    arm: str,
    positions: torch.Tensor,
    quaternions: torch.Tensor,
) -> tuple[int, float] | None:
    query = getattr(evaluator, "batch_reachable", None)
    if query is None or int(positions.shape[0]) == 0:
        return None
    result = query(
        env=observation["gym_env"],
        robot=observation.get("robot"),
        env_idx=int(observation["env_idx"]),
        arm_name=arm,
        target_positions=positions,
        target_quats=quaternions,
        env_origin=observation.get("env_origin"),
        object_name=observation.get("object_name"),
    )
    best_index = getattr(result, "best_index", None)
    costs = getattr(result, "path_cost", None)
    if best_index is None or costs is None:
        return None
    index = int(best_index)
    if index < 0 or index >= int(positions.shape[0]):
        return None
    return index, float(torch.as_tensor(costs)[index].item())


def select_reachable(
    *,
    evaluator: Any,
    observation: dict[str, Any],
    arm_order: Sequence[str],
    stages: Sequence[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int | None = None,
    max_rounds: int = 1,
) -> ReachableSelection | None:
    """Return the first candidate whose complete pose sequence is reachable."""

    if evaluator is None or not stages:
        return None
    primary_pos, primary_quat = stages[0]
    count = int(primary_pos.shape[0])
    if count == 0:
        return None
    size = count if batch_size is None else max(int(batch_size), 1)
    for arm in arm_order:
        for round_index in range(max(int(max_rounds), 1)):
            start = round_index * size
            if start >= count:
                break
            end = min(start + size, count)
            hit = _best_goal(
                evaluator=evaluator,
                observation=observation,
                arm=arm,
                positions=primary_pos[start:end],
                quaternions=primary_quat[start:end],
            )
            if hit is None:
                continue
            local_index, total_cost = hit
            index = start + local_index
            for positions, quaternions in stages[1:]:
                stage_hit = _best_goal(
                    evaluator=evaluator,
                    observation=observation,
                    arm=arm,
                    positions=positions[index : index + 1],
                    quaternions=quaternions[index : index + 1],
                )
                if stage_hit is None:
                    break
                total_cost += stage_hit[1]
            else:
                return ReachableSelection(arm=arm, index=index, path_cost=total_cost)
    return None


__all__ = ["ReachableSelection", "select_reachable"]
