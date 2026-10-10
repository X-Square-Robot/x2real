from __future__ import annotations

from typing import Any

from ...common.diagnostics import LayoutDiagnostic
from ...common.geometry import Pose
from ...common.ids import NodeId
from ..domains.manager import DomainManager
from ..ir import CompiledLayoutIR
from .base import HardFactor


class ConstraintEvaluator:
    def __init__(self, ir: CompiledLayoutIR) -> None:
        self.ir = ir
        self.factors_by_node: dict[NodeId, list[HardFactor]] = {}
        for node_id, spec in ir.node_specs.items():
            indexed = list(enumerate(spec.hard_factors))
            indexed.sort(key=lambda item: (int(getattr(item[1], "priority", 100)), item[0]))
            self.factors_by_node[node_id] = [factor for _, factor in indexed]

    def incremental_check(self, node_id: NodeId, assignment: dict[NodeId, Pose]) -> bool:
        for factor in self.factors_by_node.get(node_id, []):
            if not factor.check(assignment):
                return False
        return True

    def forward_check(self, node_id: NodeId, assignment: dict[NodeId, Pose], domain_mgr: DomainManager) -> bool:
        for factor in self.factors_by_node.get(node_id, []):
            result = factor.forward_check(node_id, assignment, self.ir, domain_mgr)
            if result is False:
                return False
        return True

    def check_all(self, assignment: dict[NodeId, Pose]) -> list[LayoutDiagnostic]:
        out: list[LayoutDiagnostic] = []
        for node_id in sorted(self.factors_by_node):
            for factor in self.factors_by_node[node_id]:
                if factor.check(assignment):
                    continue
                meta = factor.describe_failure(assignment)
                out.append(
                    LayoutDiagnostic(
                        code=str(meta.get("code") or "constraint_unsatisfied"),
                        message=str(meta.get("message") or "layout constraint failed"),
                        target=str(meta.get("target") or node_id),
                        meta={k: v for k, v in meta.items() if k not in {"code", "message", "target"}},
                    )
                )
        return out

    def describe_failures(self, node_id: NodeId, assignment: dict[NodeId, Pose]) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for factor in self.factors_by_node.get(node_id, []):
            if not factor.check(assignment):
                failures.append(factor.describe_failure(assignment))
        return failures
