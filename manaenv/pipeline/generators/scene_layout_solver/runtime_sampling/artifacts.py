from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..common.reproducibility import (
    REPRODUCIBILITY_VERSION,
    RUNTIME_PARTITION_VERSION,
    RUNTIME_SAMPLER_VERSION,
    canonical_json,
    stable_fingerprint,
)
from .schema import RuntimeSamplingPlan

RUNTIME_PLAN_SCHEMA = "scene_layout_solver.runtime_sampling_plan.v1"


def runtime_plan_to_payload(
    source: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _runtime_plan_payload(source)
    payload["schema"] = RUNTIME_PLAN_SCHEMA
    artifacts = dict(payload.get("artifacts") or {}) if isinstance(payload.get("artifacts"), Mapping) else {}
    reproducibility = (
        dict(artifacts.get("reproducibility") or {}) if isinstance(artifacts.get("reproducibility"), Mapping) else {}
    )
    reproducibility.setdefault("reproducibility_version", REPRODUCIBILITY_VERSION)
    reproducibility.setdefault("runtime_sampler_version", RUNTIME_SAMPLER_VERSION)
    reproducibility.setdefault("runtime_partition_version", RUNTIME_PARTITION_VERSION)
    if metadata:
        artifacts["runtime_metadata"] = dict(metadata)
    artifacts["reproducibility"] = reproducibility
    payload["artifacts"] = artifacts
    reproducibility["runtime_plan_fingerprint"] = runtime_plan_fingerprint(payload)
    return payload


def runtime_plan_fingerprint(source: Any) -> str:
    payload = _runtime_plan_payload(source)
    return stable_fingerprint(_runtime_plan_fingerprint_payload(payload))


def _runtime_plan_payload(source: Any) -> dict[str, Any]:
    if isinstance(source, RuntimeSamplingPlan):
        return source.to_json()
    if hasattr(source, "to_json") and not isinstance(source, Mapping):
        payload = source.to_json()
        if isinstance(payload, Mapping):
            return dict(payload)
    if isinstance(source, Mapping):
        raw = source.get("runtime_plan")
        if isinstance(raw, Mapping):
            return dict(raw)
        return dict(source)
    raise RuntimeError("runtime_plan_source_invalid: source does not contain a runtime sampling plan")


def _runtime_plan_fingerprint_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(canonical_json(source))
    artifacts = dict(payload.get("artifacts") or {}) if isinstance(payload.get("artifacts"), Mapping) else {}
    reproducibility = (
        dict(artifacts.get("reproducibility") or {}) if isinstance(artifacts.get("reproducibility"), Mapping) else {}
    )
    reproducibility.pop("runtime_plan_fingerprint", None)
    reproducibility.pop("artifact_fingerprint", None)
    artifacts.pop("runtime_metadata", None)
    if reproducibility:
        artifacts["reproducibility"] = reproducibility
    else:
        artifacts.pop("reproducibility", None)
    artifacts.pop("runtime_plan_fingerprint", None)
    if artifacts:
        payload["artifacts"] = artifacts
    else:
        payload.pop("artifacts", None)
    return payload
