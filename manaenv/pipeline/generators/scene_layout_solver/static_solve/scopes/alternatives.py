from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ...common.diagnostics import LayoutDiagnostic
from ...common.geometry import Pose
from ...common.ids import DomainId, NodeId
from ..constraints.evaluator import ConstraintEvaluator
from ..domains.base import GroupDomainResult, NodeDomainResult
from ..domains.manager import DomainManager
from .model import ScopeDomainRef, SupportSurfaceComponent


@dataclass(frozen=True)
class DomainAlternative:
    domain_id: DomainId
    poses: dict[NodeId, Pose]
    score: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DomainAlternativeResult:
    alternatives: tuple[DomainAlternative, ...]
    diagnostics: tuple[LayoutDiagnostic, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComponentAlternative:
    component_id: str
    poses: dict[NodeId, Pose]
    source_alternatives: tuple[DomainAlternative, ...]
    score: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)


class DomainAlternativeProvider:
    def __init__(
        self,
        *,
        domain_mgr: DomainManager,
        evaluator: ConstraintEvaluator,
        max_alternatives: int,
    ) -> None:
        self.domain_mgr = domain_mgr
        self.evaluator = evaluator
        self.max_alternatives = max(1, int(max_alternatives))

    def alternatives_for(
        self,
        ref: ScopeDomainRef,
        assignment: dict[NodeId, Pose],
    ) -> DomainAlternativeResult:
        result = self.domain_mgr.instantiate(ref.domain_id, assignment)
        if isinstance(result, NodeDomainResult):
            alternatives = [
                DomainAlternative(
                    domain_id=ref.domain_id,
                    poses={ref.node_ids[0]: pose},
                    score=_pose_score(pose),
                    evidence={"source": "node_candidate", "candidate_index": idx},
                )
                for idx, pose in enumerate(result.candidates[: self.max_alternatives])
            ]
            alternatives.sort(key=lambda alt: (alt.score, tuple(sorted(alt.poses))))
            return DomainAlternativeResult(
                alternatives=tuple(alternatives), evidence={"candidate_count": len(result.candidates)}
            )
        if isinstance(result, GroupDomainResult):
            alternatives = self._group_alternatives(ref, result, assignment)
            return DomainAlternativeResult(
                alternatives=tuple(alternatives),
                evidence={
                    "joint_case_count": len(result.joint_cases),
                    "member_candidate_counts": {name: len(cands) for name, cands in result.member_candidates.items()},
                },
            )
        return DomainAlternativeResult(alternatives=())

    def _group_alternatives(
        self,
        ref: ScopeDomainRef,
        result: GroupDomainResult,
        assignment: dict[NodeId, Pose],
    ) -> list[DomainAlternative]:
        alternatives: list[DomainAlternative] = []
        if result.joint_cases:
            alternatives.extend(
                DomainAlternative(
                    domain_id=ref.domain_id,
                    poses=dict(case),
                    score=sum(_pose_score(pose) for pose in case.values()),
                    evidence={"source": "group_joint_case", "candidate_index": idx},
                )
                for idx, case in enumerate(result.joint_cases[: self.max_alternatives])
            )
            alternatives.sort(key=lambda alt: (alt.score, tuple(sorted(alt.poses))))
            if len(alternatives) >= self.max_alternatives:
                return alternatives[: self.max_alternatives]
        order = result.order or list(ref.node_ids)
        candidates = {name: list(result.member_candidates.get(name, [])) for name in order}
        out: list[DomainAlternative] = []
        local = dict(assignment)

        def clear(start_nodes: set[NodeId]) -> None:
            for node_id in list(local):
                if node_id not in start_nodes:
                    local.pop(node_id, None)

        start_nodes = set(local)

        def dfs() -> None:
            if len(out) >= self.max_alternatives:
                return
            remaining = [name for name in order if name not in local]
            if not remaining:
                poses = {name: local[name] for name in order if name in local}
                out.append(
                    DomainAlternative(
                        domain_id=ref.domain_id,
                        poses=poses,
                        score=sum(_pose_score(pose) for pose in poses.values()),
                        evidence={"source": "group_member_candidates", "candidate_index": len(out)},
                    )
                )
                return
            name = min(remaining, key=lambda item: (len(candidates.get(item, [])), item))
            for pose in candidates.get(name, [])[: self.max_alternatives]:
                local[name] = pose
                if self.evaluator.forward_check(name, local, self.domain_mgr) and self.evaluator.incremental_check(
                    name, local
                ):
                    dfs()
                local.pop(name, None)
                if len(out) >= self.max_alternatives:
                    return

        dfs()
        clear(start_nodes)
        alternatives.extend(out)
        alternatives.sort(key=lambda alt: (alt.score, tuple(sorted(alt.poses))))
        return alternatives[: self.max_alternatives]


class ComponentAlternativeProvider:
    def __init__(
        self,
        *,
        domain_mgr: DomainManager,
        evaluator: ConstraintEvaluator,
        max_component_steps: int,
    ) -> None:
        self.domain_mgr = domain_mgr
        self.evaluator = evaluator
        self.max_component_steps = max(1, int(max_component_steps))
        self.domain_provider = DomainAlternativeProvider(
            domain_mgr=domain_mgr,
            evaluator=evaluator,
            max_alternatives=max(1, int(max_component_steps)),
        )

    def iter_component_alternatives(
        self,
        component: SupportSurfaceComponent,
        assignment: dict[NodeId, Pose],
    ) -> Iterator[ComponentAlternative]:
        refs = sorted(
            component.domain_refs,
            key=lambda ref: (
                1 if ref.domain_id.kind == "group" else 0,
                len(ref.node_ids),
                ref.domain_id.kind,
                ref.domain_id.id,
            ),
        )
        local = dict(assignment)
        chosen: list[DomainAlternative] = []
        attempts = 0

        def restore(start_nodes: set[NodeId]) -> None:
            for node_id in list(local):
                if node_id not in start_nodes:
                    local.pop(node_id, None)

        start_nodes = set(local)

        def dfs(index: int) -> Iterator[ComponentAlternative]:
            nonlocal attempts
            if attempts >= self.max_component_steps:
                return
            if index >= len(refs):
                poses = {node_id: local[node_id] for alt in chosen for node_id in alt.poses}
                yield ComponentAlternative(
                    component_id=component.component_id,
                    poses=poses,
                    source_alternatives=tuple(chosen),
                    score=sum(alt.score for alt in chosen),
                    evidence={"attempt_index": attempts},
                )
                return
            ref = refs[index]
            result = self.domain_provider.alternatives_for(ref, local)
            if not result.alternatives:
                return
            for alternative in result.alternatives:
                if attempts >= self.max_component_steps:
                    return
                attempts += 1
                touched = tuple(alternative.poses)
                previous = {node_id: local[node_id] for node_id in touched if node_id in local}
                for node_id, pose in alternative.poses.items():
                    local[node_id] = pose
                if self._alternative_valid(touched, local):
                    chosen.append(alternative)
                    yield from dfs(index + 1)
                    chosen.pop()
                for node_id in touched:
                    if node_id in previous:
                        local[node_id] = previous[node_id]
                    else:
                        local.pop(node_id, None)

        yield from dfs(0)
        restore(start_nodes)

    def _alternative_valid(self, touched: tuple[NodeId, ...], assignment: dict[NodeId, Pose]) -> bool:
        for node_id in touched:
            if not self.evaluator.forward_check(node_id, assignment, self.domain_mgr):
                return False
            if not self.evaluator.incremental_check(node_id, assignment):
                return False
        return True


def _pose_score(pose: Pose) -> float:
    return abs(float(pose.position[0])) + abs(float(pose.position[1])) + abs(float(pose.position[2]))
