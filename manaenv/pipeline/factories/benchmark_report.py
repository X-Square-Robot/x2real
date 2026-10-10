"""Pure-logic benchmark report builder.

Aggregation, rendering and serialisation live here.
``BenchmarkFactory`` is the sole caller — other modules must not import
this file directly.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from manaenv.pipeline.data_types import output_path
from manaenv.utils.debug_print import manaprint


def stage_breakdown(per_stage_pass_rate: dict[str, float], num_episodes: int) -> list[dict[str, Any]]:
    """Attribute episodes to the stage their gated frontier stalled at.

    Relies on the DAG-success gating invariant: per-stage pass rates are monotone
    non-increasing along the chain (reach >= grasp >= retract), so the fraction of
    episodes whose frontier stopped exactly at stage ``i`` is ``rate[i] - rate[i+1]``
    (the last stage's whole rate is "completed"). Pure arithmetic on the already
    aggregated per-stage rates — no per-episode records needed.
    """
    stages = list(per_stage_pass_rate.items())
    n = max(int(num_episodes), 0)
    out: list[dict[str, Any]] = []
    for i, (label, rate) in enumerate(stages):
        rate_f = float(rate)
        is_last = i + 1 >= len(stages)
        # Passing the final stage is completion, not a stall: there is no next
        # stage to fall short of, so its whole pass rate belongs to
        # ``completed_fraction`` (reported separately) — not ``stall_fraction``.
        # Only interior stages attribute a stall of ``rate[i] - rate[i + 1]``.
        stall_fraction = 0.0 if is_last else max(rate_f - float(stages[i + 1][1]), 0.0)
        out.append(
            {
                "stage": label,
                "pass_rate": round(rate_f, 4),
                "stall_fraction": round(stall_fraction, 4),
                "stall_count": int(round(stall_fraction * n)),
            }
        )
    return out


def _aggregate_per_stage(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-stage pass rates across orders that share the same stage chain.

    Episode-weighted across orders; orders with a different stage label set are
    skipped (a benchmark mixing distinct staged tasks). Empty for non-staged runs.
    """
    if not evaluations:
        return {}
    base_labels = list(evaluations[0]["per_stage_pass_rate"].keys())
    if not base_labels:
        return {}
    total_ep = 0
    weighted = {label: 0.0 for label in base_labels}
    for ev in evaluations:
        ps = ev.get("per_stage_pass_rate") or {}
        if list(ps.keys()) != base_labels:
            continue
        n = int(ev.get("num_episodes", 0) or 0)
        total_ep += n
        for label in base_labels:
            weighted[label] += float(ps.get(label, 0.0)) * n
    if total_ep > 0:
        merged_rates = {label: weighted[label] / total_ep for label in base_labels}
    else:
        merged_rates = {label: float(evaluations[0]["per_stage_pass_rate"].get(label, 0.0)) for label in base_labels}
        total_ep = int(evaluations[0].get("num_episodes", 0) or 0)
    rate0 = float(merged_rates[base_labels[0]])
    rate_last = float(merged_rates[base_labels[-1]])
    return {
        "stages": stage_breakdown(merged_rates, total_ep),
        "num_episodes": total_ep,
        "not_reached_fraction": round(max(1.0 - rate0, 0.0), 4),
        "completed_fraction": round(rate_last, 4),
    }


def failure_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Count the failed episodes by WHERE they died and WHY.

    Every episode is Success or Failure(stage, reason), so this is a direct tally
    of the per-episode records — not the rate arithmetic ``stage_breakdown`` does.
    The two agree when the gating invariant holds; ``per_stage`` remains the
    funnel view (how far runs got), this is the attribution view (what killed them).

    ``infra_errors`` is kept apart on purpose: an order that never built its env
    produced no episodes, and folding a crashed run into the policy's failure
    counts would read as a policy that failed every episode.
    """
    by_stage: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    pairs: dict[tuple, int] = {}
    total = 0
    failed = 0

    for r in results:
        evaluation = r.get("evaluation") if isinstance(r.get("evaluation"), dict) else {}
        for episode in evaluation.get("episodes") or []:
            if not isinstance(episode, dict):
                continue
            total += 1
            if episode.get("success", False):
                continue
            failed += 1
            stage = str(episode.get("failed_stage") or "unattributed")
            reason = str(episode.get("failed_reason") or "unknown")
            by_stage[stage] = by_stage.get(stage, 0) + 1
            by_reason[reason] = by_reason.get(reason, 0) + 1
            pairs[(stage, reason)] = pairs.get((stage, reason), 0) + 1

    ranked = [
        {
            "stage": stage,
            "reason": reason,
            "count": count,
            "fraction": round(count / failed, 4) if failed else 0.0,
        }
        for (stage, reason), count in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    infra_errors = [{"env_name": r.get("env_name", ""), "error": str(r["error"])} for r in results if r.get("error")]
    return {
        "total_episodes": total,
        "failed_episodes": failed,
        "by_stage": dict(sorted(by_stage.items(), key=lambda kv: -kv[1])),
        "by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
        "by_stage_reason": ranked,
        "infra_errors": infra_errors,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics from raw per-order results."""
    total = len(results)
    failed = sum(1 for r in results if r.get("error"))
    completed = total - failed

    rates: list[float] = []
    weights: list[float] = []
    missing_sr = 0
    total_episodes = 0
    avg_scores: list[float] = []
    score_weights: list[float] = []
    missing_evaluation = 0
    max_score = 0.0
    max_possible_score = 0.0

    for r in results:
        if r.get("error"):
            continue
        metrics = r.get("metrics", {})
        evaluation = r.get("evaluation", {}) if isinstance(r.get("evaluation"), dict) else {}
        n_ep = metrics.get("num_episodes", r.get("num_episodes", evaluation.get("num_episodes", 0)))
        if isinstance(n_ep, (int, float)):
            total_episodes += int(n_ep)

        sr = metrics.get("success_rate")
        if sr is None:
            sr = evaluation.get("success_rate")
        if sr is not None:
            sr_f = float(sr)
            rates.append(sr_f)
            ep_weight = max(int(n_ep), 1) if isinstance(n_ep, (int, float)) else 1
            weights.append(ep_weight)
        else:
            missing_sr += 1

        evaluation = r.get("evaluation")
        if isinstance(evaluation, dict) and evaluation.get("avg_score") is not None:
            ep_weight = max(int(evaluation.get("num_episodes", n_ep or 0)), 1)
            avg_scores.append(float(evaluation.get("avg_score", 0.0)))
            score_weights.append(ep_weight)
            max_score = max(max_score, float(evaluation.get("max_score", 0.0)))
            max_possible_score = max(max_possible_score, float(evaluation.get("max_possible_score", 0.0)))
        else:
            missing_evaluation += 1

    staged_evaluations = [
        r["evaluation"]
        for r in results
        if isinstance(r.get("evaluation"), dict) and r["evaluation"].get("per_stage_pass_rate")
    ]
    per_stage_block = _aggregate_per_stage(staged_evaluations)

    avg_sr = sum(rates) / len(rates) if rates else 0.0
    w_total = sum(weights)
    weighted_sr = sum(r * w for r, w in zip(rates, weights)) / w_total if w_total > 0 else 0.0
    avg_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
    score_weight_total = sum(score_weights)
    weighted_avg_score = (
        sum(score * weight for score, weight in zip(avg_scores, score_weights)) / score_weight_total
        if score_weight_total > 0
        else 0.0
    )

    return {
        "total_orders": total,
        "completed_orders": completed,
        "failed_orders": failed,
        "missing_success_rate_orders": missing_sr,
        "missing_evaluation_orders": missing_evaluation,
        "total_episodes": total_episodes,
        "avg_success_rate": round(avg_sr, 4),
        "weighted_success_rate": round(weighted_sr, 4),
        "avg_score": round(avg_score, 4),
        "weighted_avg_score": round(weighted_avg_score, 4),
        "max_score": round(max_score, 4),
        "max_possible_score": round(max_possible_score, 4),
        "per_stage": per_stage_block,
        "failure_report": failure_report(results),
    }


def build_report(
    results: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a full benchmark report (summary + results + metadata)."""
    return {
        "summary": summarize(results),
        "results": results,
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(metadata or {}),
        },
    }


def render_summary(report: dict[str, Any]) -> list[str]:
    """Produce human-readable summary lines for CLI output."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("  Benchmark Summary")
    lines.append("=" * 64)

    summary = report.get("summary", {})
    lines.append(
        f"  Orders: {summary.get('total_orders', 0)} total, "
        f"{summary.get('completed_orders', 0)} completed, "
        f"{summary.get('failed_orders', 0)} failed"
    )
    lines.append(f"  Total episodes: {summary.get('total_episodes', 0)}")
    lines.append(f"  Avg success rate:      {summary.get('avg_success_rate', 0):.4f}")
    lines.append(f"  Weighted success rate:  {summary.get('weighted_success_rate', 0):.4f}")
    lines.append(f"  Avg score:             {summary.get('avg_score', 0):.4f}")
    lines.append(f"  Weighted avg score:    {summary.get('weighted_avg_score', 0):.4f}")
    if summary.get("max_possible_score", 0) > 0:
        lines.append(
            f"  Score range observed:  {summary.get('max_score', 0):.4f}/{summary.get('max_possible_score', 0):.4f}"
        )

    per_stage = summary.get("per_stage") if isinstance(summary.get("per_stage"), dict) else None
    if per_stage and per_stage.get("stages"):
        lines.append("-" * 64)
        lines.append("  Per-stage pass rate / stall attribution (到位率/抓取率/回收率):")
        for st in per_stage["stages"]:
            lines.append(
                f"    {st['stage']:<10} pass={st['pass_rate']:.4f}  "
                f"stalled_here={st['stall_fraction']:.4f} (n={st['stall_count']})"
            )
        lines.append(
            f"    not_reached={per_stage.get('not_reached_fraction', 0):.4f}  "
            f"completed={per_stage.get('completed_fraction', 0):.4f}"
        )

    if summary.get("missing_success_rate_orders", 0) > 0:
        lines.append(f"  (!) {summary['missing_success_rate_orders']} order(s) missing success_rate metric")
    if summary.get("missing_evaluation_orders", 0) > 0:
        lines.append(f"  (!) {summary['missing_evaluation_orders']} order(s) missing evaluation summary")

    lines.append("-" * 64)

    for r in report.get("results", []):
        env = r.get("env_name", "?")
        err = r.get("error")
        if err:
            lines.append(f"  {env}: ERROR - {err}")
            continue
        metrics = r.get("metrics", {})
        evaluation = r.get("evaluation", {}) if isinstance(r.get("evaluation"), dict) else {}
        sr = metrics.get("success_rate")
        if sr is None:
            sr = evaluation.get("success_rate", "N/A")
        n_ep = metrics.get("num_episodes", r.get("num_episodes", evaluation.get("num_episodes", "N/A")))
        avg_score = evaluation.get("avg_score", "N/A")
        lines.append(f"  {env}: success_rate={sr}, avg_score={avg_score}, episodes={n_ep}")
        per_stage = evaluation.get("per_stage_pass_rate") if isinstance(evaluation, dict) else None
        if isinstance(per_stage, dict) and per_stage:
            lines.append("    stages: " + ", ".join(f"{name}={rate:.4f}" for name, rate in per_stage.items()))

    lines.append("=" * 64)
    return lines


def save_report(
    report: dict[str, Any],
    report_path: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Write benchmark report to disk as JSON.

    Returns the written file path, or empty string if nothing to write.
    """
    if not report.get("results"):
        manaprint("INFO: No benchmark results to write")
        return ""

    if report_path is None:
        base = output_dir or os.path.expanduser(output_path("benchmark"))
        report_path = os.path.join(base, "benchmark_report.json")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    manaprint(f"INFO: Benchmark report saved to {report_path}")
    return report_path
