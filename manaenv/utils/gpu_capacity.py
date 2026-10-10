"""GPU capacity estimation and parallel-env resolution.

Two distinct concepts share the ``num_envs`` name in the codebase:

* **Assembly sample count** — how many unique (scene, robot, task)
  combinations to draw.  Resolved by ``AssemblyFactory._resolve_sample_count``.
* **Isaac vectorised-env parallelism** — how many copies of the *same*
  assembled environment run in parallel inside one ``gym.make`` call.
  Resolved by :func:`resolve_num_parallel_envs` in this module.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from manaenv.utils.debug_print import manaprint

_CACHE_DIR = Path("~/.manaenv/capacity_cache").expanduser()
_CACHE_TTL_DAYS = 30


# ------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------


def resolve_num_parallel_envs(order: Any, gpu_id: int = 0) -> int:
    """Resolve the Isaac vectorised-env parallelism for a typed runtime order.

    Reads the strictly typed ``num_parallel_envs`` field from a
    evaluation order CFG.
    """
    return max(int(getattr(order, "num_parallel_envs", 1)), 1)


def estimate_num_envs(
    env_obj: Any,
    gpu_id: int = 0,
    safety_margin: float = 0.85,
    warmup_steps: int = 10,
) -> int:
    """Profile VRAM by building 1 env and extrapolate max safe count.

    Must be called **after** ``AppLauncher`` has been started.
    """
    import torch

    if not torch.cuda.is_available():
        manaprint("WARNING: [gpu_capacity] No CUDA device — returning num_envs=1")
        return 1

    props = torch.cuda.get_device_properties(gpu_id)
    total = props.total_mem
    usable = int(total * safety_margin)

    torch.cuda.reset_peak_memory_stats(gpu_id)
    baseline = torch.cuda.memory_allocated(gpu_id)

    original_num_envs = getattr(env_obj, "num_envs", 1)
    env_obj.num_envs = 1

    try:
        gym_env = env_obj.build_gym_environment(mode="evaluate")
        if gym_env is None:
            manaprint("WARNING: [gpu_capacity] build failed — returning num_envs=1")
            env_obj.num_envs = original_num_envs
            return 1

        gym_env.reset()
        for _ in range(warmup_steps):
            zero_act = torch.zeros(1, gym_env.action_space.shape[-1], device=gym_env.device)
            gym_env.step(zero_act)

        peak = torch.cuda.max_memory_allocated(gpu_id)
        per_env = max(peak - baseline, 1)

        gym_env.close()
        torch.cuda.empty_cache()
    except Exception as exc:
        manaprint(f"WARNING: [gpu_capacity] profiling failed ({exc}) — returning num_envs=1")
        env_obj.num_envs = original_num_envs
        return 1

    env_obj.num_envs = original_num_envs

    remaining = usable - baseline
    n = max(1, int(remaining / per_env))

    per_env_mb = per_env / (1024 * 1024)
    manaprint(
        f"INFO: [gpu_capacity] GPU {gpu_id} ({props.name}): "
        f"total={total / (1024**3):.1f}GB, per_env≈{per_env_mb:.0f}MB, "
        f"estimated max num_envs={n}"
    )
    return n


# ------------------------------------------------------------------
#  Capacity cache (file-backed)
# ------------------------------------------------------------------


class CapacityCache:
    """File-backed cache mapping (env_name, gpu_model, vram) -> optimal num_envs."""

    def __init__(self, cache_dir: str | Path | None = None):
        self._dir = Path(cache_dir or _CACHE_DIR).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "capacity.json"
        self._data: dict[str, dict] = self._load()

    def get(self, env_name: str, gpu_id: int = 0) -> int | None:
        key = self._make_key(env_name, gpu_id)
        entry = self._data.get(key)
        if entry is None:
            return None
        age_days = (time.time() - entry.get("timestamp", 0)) / 86400
        if age_days > _CACHE_TTL_DAYS:
            return None
        return entry.get("num_envs")

    def put(self, env_name: str, num_envs: int, vram_per_env_mb: float = 0.0, gpu_id: int = 0) -> None:
        key = self._make_key(env_name, gpu_id)
        self._data[key] = {
            "num_envs": num_envs,
            "vram_per_env_mb": vram_per_env_mb,
            "timestamp": time.time(),
        }
        self._save()

    # -- internals -------------------------------------------------

    @staticmethod
    def _make_key(env_name: str, gpu_id: int = 0) -> str:
        try:
            import torch

            props = torch.cuda.get_device_properties(gpu_id)
            gpu_tag = f"{props.name}:{props.total_mem // (1024**3)}GB"
        except Exception:
            gpu_tag = "unknown"
        raw = f"{env_name}:{gpu_tag}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _load(self) -> dict:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text())
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        try:
            self._file.write_text(json.dumps(self._data, indent=2, default=str))
        except Exception as exc:
            manaprint(f"WARNING: [CapacityCache] save failed: {exc}")
