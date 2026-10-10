"""Policy CFG discovery for the global lazy registry."""

from __future__ import annotations

from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = ["POLICY_DOMAIN", "discover_policy"]

POLICY_DOMAIN: str = "policy"

# symbol -> module hosting <symbol> and <symbol>CFG.
_POLICY_SYMBOLS: dict[str, str] = {
    # Policy envelope.
    "ManaPolicy": "manaenv.policy.mana_policy",
    # client.chunk_executor refill policies.
    "PerEnvIndependent": "manaenv.policy.refill_policies",
    "GlobalSync": "manaenv.policy.refill_policies",
    # client.chunk_postprocessors.
    "TemporalInterpolate": "manaenv.policy.chunk_postprocessors",
    "HorizonClip": "manaenv.policy.chunk_postprocessors",
    # client.wire_interceptors.
    "SaveRequestImages": "manaenv.policy.wire_interceptors",
    "WallXRtc": "manaenv.policy.wire_interceptors",
    # client.chunk_scheduler plugins.
    "SyncChunkScheduler": "manaenv.policy.chunk_scheduler.sync",
    "AsyncChunkScheduler": "manaenv.policy.chunk_scheduler.async",
}


def discover_policy(symbol: str) -> ComponentSpec | None:
    """Resolve policy-domain symbols (policy envelope + client plugins)."""
    module_path = _POLICY_SYMBOLS.get(symbol)
    if module_path is None:
        return None
    return ComponentSpec(module_path=module_path, class_name=symbol, symbol=symbol)


register_provider(POLICY_DOMAIN, discover_policy)
