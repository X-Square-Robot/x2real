from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.artifacts.base import BaseArtifact


@dataclass
class EvalArtifact(BaseArtifact):
    """Artifact produced by evaluation/benchmark."""

    artifact_type: str = "eval"
    policy_ref: str = ""
    env_refs: list[str] = field(default_factory=list)
    num_episodes: int = 0
    success_rate: float = 0.0
    avg_score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    failure_cases: list[dict[str, Any]] = field(default_factory=list)
    debug_record_paths: list[str] = field(default_factory=list)


def failure_cases_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One failure case per failed episode of a rollout *result*.

    Reads the per-episode rows the evaluation runtime already builds
    (``result["evaluation"]["episodes"]``) and reshapes each failure into the
    key names the failure analyzers aggregate on: ``error_type`` (the WHY) and
    ``env_ref`` (the WHERE), keeping ``stage`` (the WHICH) plus the episode
    identity so a pattern can be traced back to the run that produced it.
    A rollout that died before evaluating (``result["error"]``) has no episodes
    and so no cases — that is an infrastructure fault, not a policy failure.
    """
    episodes = ((result.get("evaluation") or {}).get("episodes")) or []
    env_ref = result.get("env_name", "")
    cases: list[dict[str, Any]] = []
    for episode in episodes:
        if not isinstance(episode, dict) or episode.get("success", False):
            continue
        cases.append(
            {
                "error_type": episode.get("failed_reason") or "unknown",
                "env_ref": env_ref,
                "stage": episode.get("failed_stage"),
                "episode_index": episode.get("episode_index"),
                "env_id": episode.get("env_id"),
                "score": episode.get("score"),
            }
        )
    return cases
