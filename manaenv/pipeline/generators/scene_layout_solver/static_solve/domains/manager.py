from __future__ import annotations

import json
from typing import Any

from ...common.geometry import Pose
from ...common.ids import DomainId, NodeId
from ...common.reproducibility import stable_rng
from ..ir import CompiledLayoutIR
from .base import DomainContext, DomainProvider, GroupDomainResult, NodeDomainResult


class DomainManager:
    def __init__(
        self,
        ir: CompiledLayoutIR,
        *,
        seed: int,
        max_candidates: int,
        relation_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.ir = ir
        self.seed = int(seed)
        self.max_candidates = max(1, int(max_candidates))
        self.relation_overrides = dict(relation_overrides or {})
        self.cache: dict[tuple[Any, ...], NodeDomainResult | GroupDomainResult] = {}

    def instantiate(
        self,
        domain_id: DomainId,
        assignment: dict[NodeId, Pose],
        *,
        sampler_overrides: dict[str, Any] | None = None,
        relation_overrides: dict[str, Any] | None = None,
        budget_scale: float = 1.0,
        force_refresh: bool = False,
    ) -> NodeDomainResult | GroupDomainResult:
        provider = self._provider_for(domain_id)
        effective_relation_overrides = relation_overrides if relation_overrides is not None else self.relation_overrides
        key = self._cache_key(
            domain_id, provider, assignment, sampler_overrides, effective_relation_overrides, budget_scale
        )
        if not force_refresh and key in self.cache:
            return self.cache[key]
        if provider is None:
            result: NodeDomainResult | GroupDomainResult
            if domain_id.kind == "group":
                result = GroupDomainResult(diagnostics={"reason": "provider_missing"})
            else:
                result = NodeDomainResult(diagnostics={"reason": "provider_missing"})
            self.cache[key] = result
            return result
        cache_nodes = provider.cache_assignment_nodes()
        assignment_sig = self._assignment_signature(assignment, cache_nodes)
        sampler_payload = self._stable_payload(sampler_overrides)
        relation_payload = self._stable_payload(effective_relation_overrides)
        ctx = DomainContext(
            domain_id=domain_id,
            assignment=dict(assignment),
            budget=max(1, int(round(self.max_candidates * max(0.1, float(budget_scale))))),
            rng=stable_rng(
                self.seed,
                "static_domain",
                domain_id.kind,
                domain_id.id,
                provider.__class__.__name__,
                payload={
                    "assignment": assignment_sig,
                    "sampler_overrides": sampler_payload,
                    "relation_overrides": relation_payload,
                    "budget_scale": round(float(budget_scale), 8),
                },
            ),
            sampler_overrides=dict(sampler_overrides or {}),
            relation_overrides=dict(effective_relation_overrides or {}),
            budget_scale=float(budget_scale),
        )
        result = provider.build_domain(ctx).evaluate(ctx)
        self.cache[key] = result
        return result

    def invalidate(self, domain_id: DomainId) -> None:
        for key in list(self.cache):
            if key[:2] == (domain_id.kind, domain_id.id):
                self.cache.pop(key, None)

    def clear(self) -> None:
        self.cache.clear()

    def _provider_for(self, domain_id: DomainId) -> DomainProvider | None:
        if domain_id.kind == "group":
            spec = self.ir.group_spec(domain_id.id)
        else:
            spec = self.ir.node_spec(domain_id.id)
        providers = getattr(spec, "domain_providers", None)
        return providers[0] if providers else None

    @staticmethod
    def _assignment_signature(assignment: dict[NodeId, Pose], nodes: tuple[NodeId, ...] | None = None) -> tuple:
        if nodes is not None:
            assignment = {name: assignment[name] for name in nodes if name in assignment}
        return tuple(
            (
                name,
                tuple(round(float(v), 8) for v in pose.position),
                tuple(round(float(v), 8) for v in pose.quaternion),
            )
            for name, pose in sorted(assignment.items())
        )

    @staticmethod
    def _stable_payload(payload: dict[str, Any] | None) -> str:
        try:
            return json.dumps(payload or {}, sort_keys=True, default=str)
        except TypeError:
            return repr(payload)

    def _cache_key(
        self,
        domain_id: DomainId,
        provider: DomainProvider | None,
        assignment: dict[NodeId, Pose],
        sampler_overrides: dict[str, Any] | None,
        relation_overrides: dict[str, Any] | None,
        budget_scale: float,
    ) -> tuple[Any, ...]:
        return (
            domain_id.kind,
            domain_id.id,
            provider.__class__.__name__ if provider is not None else "",
            self._assignment_signature(assignment, provider.cache_assignment_nodes() if provider is not None else ()),
            self._stable_payload(sampler_overrides),
            self._stable_payload(relation_overrides),
            round(float(budget_scale), 8),
        )
