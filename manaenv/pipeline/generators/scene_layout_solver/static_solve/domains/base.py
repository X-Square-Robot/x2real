from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ...common.diagnostics import LayoutDiagnostic
from ...common.geometry import Pose
from ...common.ids import DomainId, NodeId


@dataclass(frozen=True)
class DomainContext:
    domain_id: DomainId
    assignment: dict[NodeId, Pose]
    budget: int
    rng: random.Random
    sampler_overrides: dict[str, Any] = field(default_factory=dict)
    relation_overrides: dict[str, Any] = field(default_factory=dict)
    budget_scale: float = 1.0


@dataclass
class NodeDomainResult:
    candidates: list[Pose] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def candidate_count(self) -> int:
        return len(self.candidates)


@dataclass
class GroupDomainResult:
    member_candidates: dict[NodeId, list[Pose]] = field(default_factory=dict)
    order: list[NodeId] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    joint_cases: list[dict[NodeId, Pose]] = field(default_factory=list)

    def candidate_count(self) -> int:
        if self.joint_cases:
            return len(self.joint_cases)
        return sum(len(cands) for cands in self.member_candidates.values())


class Domain:
    def evaluate(self, ctx: DomainContext) -> NodeDomainResult | GroupDomainResult:
        raise NotImplementedError


@dataclass
class StaticNodeDomain(Domain):
    candidates: list[Pose]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, ctx: DomainContext) -> NodeDomainResult:
        return NodeDomainResult(candidates=list(self.candidates), diagnostics=dict(self.diagnostics))


@dataclass
class StaticGroupDomain(Domain):
    member_candidates: dict[NodeId, list[Pose]]
    order: list[NodeId]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    joint_cases: list[dict[NodeId, Pose]] = field(default_factory=list)

    def evaluate(self, ctx: DomainContext) -> GroupDomainResult:
        return GroupDomainResult(
            member_candidates={name: list(cands) for name, cands in self.member_candidates.items()},
            order=list(self.order),
            diagnostics=dict(self.diagnostics),
            joint_cases=[dict(case) for case in self.joint_cases],
        )


class DomainProvider:
    depends_on: tuple[DomainId, ...] = ()

    def build_domain(self, ctx: DomainContext) -> Domain:
        raise NotImplementedError

    def precheck(self, assignment: dict[NodeId, Pose] | None = None) -> list[LayoutDiagnostic]:
        return []

    def cache_assignment_nodes(self) -> tuple[NodeId, ...]:
        return ()
