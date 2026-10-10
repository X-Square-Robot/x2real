from __future__ import annotations

import hashlib
import json
import random
from typing import Any

REPRODUCIBILITY_VERSION = "scene_layout_solver.repro.v2.master_seed"
STATIC_SOLVER_VERSION = "mini_domain_kernel.v3.repro1"
RUNTIME_SAMPLER_VERSION = "runtime_plan_sampler.v2.repro1"
RUNTIME_PARTITION_VERSION = "runtime_partition.v3.repro1"


def canonical_json(payload: Any) -> str:
    """Return a deterministic JSON representation for seed/fingerprint inputs."""

    return json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_fingerprint(payload: Any, *, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[: max(1, int(length))]


def stable_seed(base_seed: int, *parts: Any, payload: Any | None = None) -> int:
    data = {
        "base_seed": int(base_seed),
        "parts": list(parts),
        "payload": payload,
        "version": REPRODUCIBILITY_VERSION,
    }
    # random.Random accepts arbitrary-width ints; use 64 bits to keep logs compact.
    return int(hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:16], 16)


def stable_rng(base_seed: int, *parts: Any, payload: Any | None = None) -> random.Random:
    return random.Random(stable_seed(base_seed, *parts, payload=payload))


def pose_signature(poses: dict[str, Any]) -> tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...]:
    out: list[tuple[str, tuple[float, ...], tuple[float, ...]]] = []
    for name, pose in sorted(poses.items()):
        position = tuple(round(float(value), 8) for value in getattr(pose, "position", ()))
        quaternion = tuple(round(float(value), 8) for value in getattr(pose, "quaternion", ()))
        out.append((str(name), position, quaternion))
    return tuple(out)


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set):
        return [_canonicalize(item) for item in sorted(value, key=repr)]
    if hasattr(value, "to_json"):
        try:
            return _canonicalize(value.to_json())
        except Exception:
            pass
    return repr(value)
