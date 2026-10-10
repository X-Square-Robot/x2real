"""Process-wide master seed and deterministic derivation.

Contract (see unified seed design):

* Set ``pipeline.reproducibility.master_seed`` (or CLI ``--seed``) → every
  subsystem derives from that single entropy budget via ``stable_seed``.
* Omit master seed → OS entropy everywhere; never silently fall back to ``0``.
* Sub-YAML ``seed:`` values remain optional per-stream overrides.
* Publish before reflecting ``scene_orders``: schema-vNext materialization runs
  during YAML reflection and must already see the process-wide master seed.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from manaenv.pipeline.generators.scene_layout_solver.common.reproducibility import stable_rng, stable_seed

_MASTER_SEED: int | None = None
_CLI_OVERRIDE: int | None = None

# Namespace constants for the derivation table.
NS_SCENE_MATERIALIZE = "scene_materialize"
NS_SCENE_LAYOUT = "scene_layout"
NS_SCENE_VARIANT = "scene_variant"
NS_RUNTIME_SAMPLE = "runtime_sample"
NS_EVENT = "event"
NS_ISAAC_ENV = "isaac_env"
NS_ORDER = "order"


def set_cli_seed_override(seed: int | None) -> None:
    """Record CLI ``--seed``; applied when pipeline publishes YAML master_seed."""

    global _CLI_OVERRIDE
    _CLI_OVERRIDE = None if seed is None else int(seed)


def get_cli_seed_override() -> int | None:
    return _CLI_OVERRIDE


def set_master_seed(seed: int | None) -> None:
    """Publish the process-wide master seed (``None`` = full OS entropy)."""

    global _MASTER_SEED
    _MASTER_SEED = None if seed is None else int(seed)


def get_master_seed() -> int | None:
    return _MASTER_SEED


def clear_master_seed() -> None:
    """Reset process state (tests)."""

    global _MASTER_SEED, _CLI_OVERRIDE
    _MASTER_SEED = None
    _CLI_OVERRIDE = None


def publish_from_pipeline(*, yaml_master_seed: int | None = None) -> int | None:
    """Resolve CLI override over YAML and publish. Returns the effective master."""

    cli = get_cli_seed_override()
    effective = cli if cli is not None else (None if yaml_master_seed is None else int(yaml_master_seed))
    set_master_seed(effective)
    return effective


def peek_yaml_master_seed(precompiled_tree: Mapping[str, Any] | None) -> int | None:
    """Read ``pipeline.reproducibility.master_seed`` from an expanded YAML tree.

    Used to publish the process-wide seed before ``scene_orders`` reflection
    materializes schema-vNext backgrounds/assets.
    """

    if not isinstance(precompiled_tree, Mapping):
        return None
    pipeline = precompiled_tree.get("pipeline")
    if not isinstance(pipeline, Mapping):
        return None
    repro = pipeline.get("reproducibility")
    if not isinstance(repro, Mapping):
        return None
    raw = repro.get("master_seed")
    if raw is None or raw is False:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def os_entropy_seed() -> int:
    """Draw a non-zero-width seed from OS entropy (never a silent ``0`` default)."""

    return int(random.SystemRandom().randint(0, 2**63 - 1))


def to_isaac_seed(seed: int) -> int:
    """Fit a derived seed into signed 32-bit range for Isaac/carb ``set_int``.

    ``stable_seed`` returns 64-bit values; writing those to ``env_cfg.seed``
    raises ``TypeError: set_int(): incompatible function arguments``.
    """

    return int(seed) & 0x7FFFFFFF


def derive(namespace: str, *parts: Any) -> int | None:
    """Derive a child seed from the master, or ``None`` when master is unset."""

    master = get_master_seed()
    if master is None:
        return None
    return stable_seed(int(master), str(namespace), *parts)


def resolve_base_seed(
    namespace: str,
    *parts: Any,
    override: int | None = None,
    allow_os_entropy: bool = True,
) -> int | None:
    """Resolve a subsystem base seed.

    Priority: explicit ``override`` → master-derived → OS entropy (or ``None``
    when ``allow_os_entropy`` is false so callers can keep ``random.Random(None)``).
    """

    if override is not None:
        return int(override)
    derived = derive(namespace, *parts)
    if derived is not None:
        return int(derived)
    if allow_os_entropy:
        return os_entropy_seed()
    return None


def resolve_rng(namespace: str, *parts: Any, override: int | None = None) -> random.Random:
    """RNG for a subsystem base (build-time / one-shot)."""

    seed = resolve_base_seed(namespace, *parts, override=override, allow_os_entropy=False)
    if seed is None:
        return random.Random()  # OS entropy via Random(None) semantics
    return random.Random(int(seed))


def event_rng(
    *,
    key: str,
    env_id: int,
    episode_token: int,
    override: int | None = None,
) -> random.Random:
    """Per-env, per-episode event sampling RNG."""

    if override is not None:
        base = int(override)
    else:
        derived = derive(NS_EVENT, str(key))
        if derived is None:
            # No master and no override: fresh OS stream (envs differ automatically).
            return random.Random()
        base = int(derived)
    return stable_rng(int(base), NS_EVENT, str(key), int(env_id), int(episode_token))
