from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from ...common.diagnostics import LayoutDiagnostic
from ...common.geometry import Pose
from ...common.ids import DomainId, NodeId
from ..constraints.evaluator import ConstraintEvaluator
from ..domains.base import GroupDomainResult, NodeDomainResult
from ..domains.manager import DomainManager
from ..domains.placement_providers import InVolumeDomainProvider, PoseSampleDomainProvider, SupportOnDomainProvider
from ..ir import CompiledLayoutIR
from ..scopes.transaction_solver import SupportSurfaceTransactionSolver


@dataclass
class KernelSolveResult:
    success: bool
    poses: dict[NodeId, Pose] = field(default_factory=dict)
    diagnostics: list[LayoutDiagnostic] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


class MiniKernelSolver:
    def __init__(
        self,
        ir: CompiledLayoutIR,
        *,
        seed: int,
        max_candidates: int = 64,
        max_search_steps: int = 50000,
        relation_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.ir = ir
        self.seed = int(seed)
        self.max_candidates = max(1, int(max_candidates))
        self.max_search_steps = max(1, int(max_search_steps))
        self.domain_mgr = DomainManager(
            ir,
            seed=seed,
            max_candidates=max_candidates,
            relation_overrides=relation_overrides,
        )
        self.evaluator = ConstraintEvaluator(ir)
        self.steps = 0
        self.trace: list[dict[str, Any]] = []
        self.last_failure: LayoutDiagnostic | None = None
        self._ordering, self._ordering_surface_problems = self._build_ordering_index(ir)

    @staticmethod
    def _build_ordering_index(
        ir: CompiledLayoutIR,
    ) -> tuple[dict[NodeId, list[tuple[str, str, NodeId, str | None]]], dict[str, Any]]:
        surface_by_edge: dict[tuple[NodeId, NodeId, str], str] = {}
        surface_problems: dict[str, Any] = {}
        for spec in ir.node_specs.values():
            for factor in spec.hard_factors:
                relation = getattr(factor, "relation", None)
                surface_id = getattr(factor, "surface_id", None)
                problem = getattr(factor, "problem", None)
                if relation is None or not surface_id:
                    continue
                direction = getattr(relation, "direction", None)
                node_1 = getattr(relation, "node_1", None)
                node_2 = getattr(relation, "node_2", None)
                if direction == "left_of":
                    edge = (node_1, node_2, "x")
                elif direction == "right_of":
                    edge = (node_2, node_1, "x")
                elif direction == "back_of":
                    edge = (node_1, node_2, "y")
                elif direction == "front_of":
                    edge = (node_2, node_1, "y")
                else:
                    continue
                surface_by_edge[(str(edge[0]), str(edge[1]), edge[2])] = str(surface_id)
                if problem is not None:
                    surface_problems[str(surface_id)] = problem

        out: dict[NodeId, list[tuple[str, str, NodeId, str | None]]] = {}
        for pre, post, axis in ir.ordering_edges:
            surface_id = surface_by_edge.get((pre, post, axis))
            out.setdefault(pre, []).append(("pre", axis, post, surface_id))
            out.setdefault(post, []).append(("post", axis, pre, surface_id))
        return out, surface_problems

    def solve(self) -> KernelSolveResult:
        try:
            return self._solve()
        finally:
            self.domain_mgr.clear()

    def _solve(self) -> KernelSolveResult:
        if self.ir.unsupported:
            return KernelSolveResult(
                success=False,
                diagnostics=list(self.ir.unsupported),
                artifacts={"ir": self.ir.to_json()},
            )
        depth_limit = max(1, sys.getrecursionlimit() - 64)
        sequence_len = len(self.ir.execution_sequence)
        if sequence_len >= depth_limit:
            return KernelSolveResult(
                success=False,
                diagnostics=[
                    LayoutDiagnostic(
                        "execution_sequence_too_deep",
                        "kernel execution sequence is too deep for recursive search",
                        target="kernel",
                        meta={
                            "execution_sequence_length": sequence_len,
                            "recursion_limit": sys.getrecursionlimit(),
                            "guard_margin": 64,
                        },
                    )
                ],
                artifacts={"ir": self.ir.to_json(), "steps": self.steps},
            )
        assignment: dict[NodeId, Pose] = {}
        ok = self._dfs(0, assignment)
        if not ok:
            diag = self.last_failure or LayoutDiagnostic("layout_solve_failed", "layout solve failed", target="kernel")
            return KernelSolveResult(
                success=False,
                poses=dict(assignment),
                diagnostics=[diag],
                trace=list(self.trace),
                artifacts={"ir": self.ir.to_json(), "steps": self.steps},
            )
        missing = sorted(name for name in self.ir.node_specs if name not in assignment)
        if missing:
            return KernelSolveResult(
                success=False,
                poses=dict(assignment),
                diagnostics=[
                    LayoutDiagnostic(
                        "unplaced_objects",
                        "layout solver left objects without solved poses",
                        target="kernel",
                        meta={"objects": missing},
                    )
                ],
                trace=list(self.trace),
                artifacts={"ir": self.ir.to_json(), "steps": self.steps},
            )
        final_diags = self.evaluator.check_all(assignment)
        if final_diags:
            return KernelSolveResult(
                success=False,
                poses=dict(assignment),
                diagnostics=final_diags[:1],
                trace=list(self.trace),
                artifacts={"ir": self.ir.to_json(), "steps": self.steps},
            )
        return KernelSolveResult(
            success=True,
            poses=dict(assignment),
            trace=list(self.trace),
            artifacts={"ir": self.ir.to_json(), "steps": self.steps},
        )

    def _dfs(self, idx: int, assignment: dict[NodeId, Pose]) -> bool:
        if self.steps >= self.max_search_steps:
            self.last_failure = LayoutDiagnostic(
                "solve_retry_budget_exhausted",
                "kernel search budget exhausted",
                target="kernel",
            )
            return False
        if idx >= len(self.ir.execution_sequence):
            return True
        domain_id = self.ir.execution_sequence[idx]
        if domain_id.kind == "support_component":
            return self._solve_support_component_at(idx, domain_id, assignment)
        if domain_id.kind == "group":
            return self._solve_group_at(idx, domain_id, assignment)
        return self._solve_node_at(idx, domain_id, assignment)

    def _solve_support_component_at(self, idx: int, domain_id: DomainId, assignment: dict[NodeId, Pose]) -> bool:
        component = self.ir.support_surface_components.get(domain_id.id)
        if component is None:
            self.last_failure = LayoutDiagnostic(
                "support_surface_component_missing",
                "support component domain is not present in IR",
                target=domain_id.id,
            )
            return False
        if all(node_id in assignment for node_id in component.node_ids):
            return self._dfs(idx + 1, assignment)
        solver = SupportSurfaceTransactionSolver(
            domain_mgr=self.domain_mgr,
            evaluator=self.evaluator,
            max_component_steps=max(
                self.max_candidates,
                min(self.max_search_steps, self.max_candidates * 32),
            ),
        )
        trace = self._trace_event(
            domain_id,
            "support_component",
            candidates={
                "domain_refs": len(component.domain_refs),
                "nodes": len(component.node_ids),
            },
        )
        attempted = 0
        checkpoint = dict(assignment)
        for alternative in solver.iter_component_alternatives(component, assignment):
            if self.steps >= self.max_search_steps:
                self.last_failure = LayoutDiagnostic(
                    "solve_retry_budget_exhausted",
                    "kernel search budget exhausted",
                    target=domain_id.id,
                )
                break
            attempted += 1
            self.steps += 1
            assignment.clear()
            assignment.update(checkpoint)
            assignment.update(alternative.poses)
            touched = tuple(alternative.poses)
            if all(
                self.evaluator.forward_check(node_id, assignment, self.domain_mgr)
                and self.evaluator.incremental_check(node_id, assignment)
                for node_id in touched
            ):
                if self._dfs(idx + 1, assignment):
                    self.trace.append(
                        {
                            **trace,
                            "success": True,
                            "committed": sorted(touched),
                            "mode": "support_component",
                            "attempted_alternatives": attempted,
                            "component_id": component.component_id,
                        }
                    )
                    return True
            for node_id in touched:
                self._record_candidate_failure(node_id, assignment)
            assignment.clear()
            assignment.update(checkpoint)
        assignment.clear()
        assignment.update(checkpoint)
        if self.last_failure is None or self.last_failure.code in {
            "collision_unsatisfied",
            "constraint_unsatisfied",
        }:
            self.last_failure = LayoutDiagnostic(
                "support_surface_component_no_viable_alternative",
                "support-surface component has no viable transaction alternative",
                target=component.component_id,
                meta={
                    "domain_refs": [ref.domain_id.to_json() for ref in component.domain_refs],
                    "attempted_alternatives": attempted,
                    "unknown_rejection_count": solver.trace.unknown_rejection_count,
                },
            )
        self.trace.append(
            {
                **trace,
                "success": False,
                "attempted_alternatives": attempted,
                "component_id": component.component_id,
                "diagnostic": self.last_failure.to_json(),
            }
        )
        return False

    def _solve_node_at(self, idx: int, domain_id: DomainId, assignment: dict[NodeId, Pose]) -> bool:
        node_id = domain_id.id
        if node_id in assignment:
            return self._dfs(idx + 1, assignment)
        result = self.domain_mgr.instantiate(domain_id, assignment)
        candidates = list(result.candidates) if isinstance(result, NodeDomainResult) else []
        trace = self._trace_event(domain_id, "node", candidates={node_id: len(candidates)})
        if not candidates:
            self.last_failure = LayoutDiagnostic(
                "node_runtime_domain_empty",
                "node domain produced no candidates",
                target=node_id,
                meta=dict(result.diagnostics if isinstance(result, NodeDomainResult) else {}),
            )
            self.trace.append({**trace, "success": False, "diagnostic": self.last_failure.to_json()})
            return False
        for pose in self._ordered_candidates(node_id, candidates, assignment):
            self.steps += 1
            assignment[node_id] = pose
            if self.evaluator.forward_check(node_id, assignment, self.domain_mgr) and self.evaluator.incremental_check(
                node_id, assignment
            ):
                if self._dfs(idx + 1, assignment):
                    self.trace.append({**trace, "success": True, "committed": [node_id]})
                    return True
            self._record_candidate_failure(node_id, assignment)
            assignment.pop(node_id, None)
            if self.steps >= self.max_search_steps:
                break
        assignment.pop(node_id, None)
        if self._node_packer_name(node_id) == "support_scope" and (
            self.last_failure is None or self.last_failure.code in {"collision_unsatisfied", "constraint_unsatisfied"}
        ):
            self.last_failure = LayoutDiagnostic(
                "support_scope_pack_failed",
                "support scope candidates cannot satisfy local constraints",
                target=node_id,
                meta={
                    "candidate_count": len(candidates),
                    "packer": "support_scope",
                    "unknown_rejection_count": True,
                },
            )
        elif self.last_failure is None or self.last_failure.code == "constraint_unsatisfied":
            self.last_failure = LayoutDiagnostic(
                "node_pack_failed",
                "node candidates cannot satisfy constraints",
                target=node_id,
                meta={"candidate_count": len(candidates)},
            )
        self.trace.append({**trace, "success": False, "diagnostic": self.last_failure.to_json()})
        return False

    def _solve_group_at(self, idx: int, domain_id: DomainId, assignment: dict[NodeId, Pose]) -> bool:
        group = self.ir.group_spec(domain_id.id)
        if group is None:
            return self._dfs(idx + 1, assignment)
        result = self.domain_mgr.instantiate(domain_id, assignment)
        if not isinstance(result, GroupDomainResult):
            self.last_failure = LayoutDiagnostic(
                "group_domain_unavailable",
                "group domain unavailable",
                target=domain_id.id,
            )
            return False
        order = result.order or list(group.members)
        candidates = {name: list(result.member_candidates.get(name, [])) for name in order}
        existing_nodes = set(assignment)
        trace = self._trace_event(
            domain_id,
            "group",
            candidates={name: len(cands) for name, cands in candidates.items()},
        )
        if result.joint_cases:
            trace["joint_case_count"] = len(result.joint_cases)
            if self._solve_group_joint_cases(
                domain_id.id,
                order,
                result.joint_cases,
                assignment,
                on_complete=lambda: self._dfs(idx + 1, assignment),
            ):
                committed = sorted(set(assignment) - existing_nodes)
                self.trace.append(
                    {
                        **trace,
                        "success": True,
                        "committed": committed,
                        "mode": "joint_cases",
                    }
                )
                return True
            # Continue into the legacy member-candidate path when present. The
            # joint path is preferred, but fallback remains explicit in trace.
            trace["joint_cases_failed"] = True
        if any(not candidates.get(name) for name in order):
            empty = [name for name in order if not candidates.get(name)]
            self.last_failure = LayoutDiagnostic(
                "scope_group_domain_empty",
                "group domain produced empty member candidates",
                target=domain_id.id,
                meta={"empty_nodes": empty, **dict(result.diagnostics)},
            )
            self.trace.append({**trace, "success": False, "diagnostic": self.last_failure.to_json()})
            return False

        def clear_group() -> None:
            for member in order:
                if member not in existing_nodes:
                    assignment.pop(member, None)

        if self._solve_group_members(
            domain_id.id,
            order,
            candidates,
            assignment,
            on_complete=lambda: self._dfs(idx + 1, assignment),
        ):
            committed = sorted(set(assignment) - existing_nodes)
            self.trace.append({**trace, "success": True, "committed": committed})
            return True
        clear_group()
        if self.last_failure is None:
            self.last_failure = LayoutDiagnostic(
                self._group_failure_code(domain_id.id),
                "group candidates cannot satisfy constraints",
                target=domain_id.id,
                meta={
                    "candidate_counts": {name: len(cands) for name, cands in candidates.items()},
                    "packer": self._group_packer_name(domain_id.id),
                    "unknown_rejection_count": True,
                },
            )
        self.trace.append({**trace, "success": False, "diagnostic": self.last_failure.to_json()})
        return False

    def _solve_group_joint_cases(
        self,
        group_id: str,
        order: list[NodeId],
        joint_cases: list[dict[NodeId, Pose]],
        assignment: dict[NodeId, Pose],
        *,
        on_complete: Any | None = None,
    ) -> bool:
        existing_nodes = set(assignment)

        def clear_case() -> None:
            for member in order:
                if member not in existing_nodes:
                    assignment.pop(member, None)

        for case in joint_cases:
            if self.steps >= self.max_search_steps:
                self.last_failure = LayoutDiagnostic(
                    "solve_retry_budget_exhausted",
                    "kernel search budget exhausted",
                    target=group_id,
                )
                return False
            self.steps += 1
            for member in order:
                if member in case and member not in existing_nodes:
                    assignment[member] = case[member]
            ok = True
            for member in order:
                if member in assignment and (
                    not self.evaluator.forward_check(member, assignment, self.domain_mgr)
                    or not self.evaluator.incremental_check(member, assignment)
                ):
                    self._record_candidate_failure(member, assignment)
                    ok = False
                    break
            if ok:
                if bool(on_complete()) if on_complete is not None else True:
                    return True
            clear_case()
        clear_case()
        if self.last_failure is None:
            self.last_failure = LayoutDiagnostic(
                self._group_failure_code(group_id),
                "joint group cases cannot satisfy constraints",
                target=group_id,
                meta={
                    "joint_case_count": len(joint_cases),
                    "packer": self._group_packer_name(group_id),
                },
            )
        return False

    def _solve_group_members(
        self,
        group_id: str,
        order: list[NodeId],
        candidates: dict[NodeId, list[Pose]],
        assignment: dict[NodeId, Pose],
        *,
        on_complete: Any | None = None,
    ) -> bool:
        if all(name in assignment for name in order):
            return bool(on_complete()) if on_complete is not None else True
        remaining = [name for name in order if name not in assignment]
        ranked: list[tuple[int, str, list[Pose]]] = []
        for name in remaining:
            viable: list[Pose] = []
            for pose in self._ordered_candidates(name, candidates.get(name, []), assignment):
                assignment[name] = pose
                if self.evaluator.forward_check(name, assignment, self.domain_mgr) and self.evaluator.incremental_check(
                    name, assignment
                ):
                    viable.append(pose)
                else:
                    self._record_candidate_failure(name, assignment)
                assignment.pop(name, None)
            if not viable:
                self.last_failure = LayoutDiagnostic(
                    self._group_failure_code(group_id),
                    "group member has no viable candidate",
                    target=group_id,
                    meta={
                        "empty_node": name,
                        "candidate_count": len(candidates.get(name, [])),
                        "packer": self._group_packer_name(group_id),
                        "unknown_rejection_count": True,
                    },
                )
                return False
            ranked.append((len(viable), name, viable))
        ranked.sort(key=lambda item: (item[0], item[1]))
        _, name, viable = ranked[0]
        for pose in self._ordered_candidates(name, viable, assignment):
            self.steps += 1
            assignment[name] = pose
            if self._solve_group_members(group_id, order, candidates, assignment, on_complete=on_complete):
                return True
            assignment.pop(name, None)
            if self.steps >= self.max_search_steps:
                self.last_failure = LayoutDiagnostic(
                    "solve_retry_budget_exhausted",
                    "kernel search budget exhausted",
                    target=group_id,
                )
                return False
        assignment.pop(name, None)
        return False

    def _record_candidate_failure(self, node_id: NodeId, assignment: dict[NodeId, Pose]) -> None:
        failures = self.evaluator.describe_failures(node_id, assignment)
        if not failures:
            return
        item = failures[0]
        self.last_failure = LayoutDiagnostic(
            code=str(item.get("code") or "constraint_unsatisfied"),
            message=str(item.get("message") or "constraint failed"),
            target=str(item.get("target") or node_id),
            meta={k: v for k, v in item.items() if k not in {"code", "message", "target"}},
        )

    @staticmethod
    def _group_failure_code(group_id: str) -> str:
        if group_id.startswith("scatter:"):
            return "scatter_group_capacity_insufficient"
        if group_id.startswith("region:"):
            return "sampling_budget_exhausted"
        if group_id.startswith("volume:"):
            return "volume_group_capacity_insufficient"
        return "scope_group_retry_exhausted"

    def _group_packer_name(self, group_id: str) -> str:
        spec = self.ir.group_spec(group_id)
        provider = spec.domain_providers[0] if spec and spec.domain_providers else None
        relation = getattr(provider, "relation", None)
        if group_id.startswith("scatter:"):
            return f"scatter_{str(getattr(relation, 'mode', 'group') or 'group')}"
        if group_id.startswith("region:"):
            return "region_sample"
        if group_id.startswith("volume:"):
            return "volume_group"
        return "group"

    def _node_packer_name(self, node_id: str) -> str:
        spec = self.ir.node_spec(node_id)
        provider = spec.domain_providers[0] if spec and spec.domain_providers else None
        if isinstance(provider, SupportOnDomainProvider):
            return "support_scope"
        if isinstance(provider, PoseSampleDomainProvider):
            return "pose_sample"
        if isinstance(provider, InVolumeDomainProvider):
            return "in_volume"
        return "node"

    def _ordered_candidates(
        self, node_id: NodeId, candidates: list[Pose], assignment: dict[NodeId, Pose]
    ) -> list[Pose]:
        hints = self._ordering.get(node_id, [])
        if not hints:
            return list(candidates)

        def score(pose: Pose) -> tuple[float, float, float, float]:
            total = 0.0
            hard = 0.0
            for role, axis, other, surface_id in hints:
                value = self._ordering_axis_value(pose, axis, surface_id, assignment)
                if other in assignment:
                    other_value = self._ordering_axis_value(assignment[other], axis, surface_id, assignment)
                    if role == "pre":
                        total += max(0.0, value - other_value)
                        hard += value - other_value
                    else:
                        total += max(0.0, other_value - value)
                        hard += other_value - value
                else:
                    total += value if role == "pre" else -value
            return (total, hard, pose.position[0], pose.position[1])

        return sorted(candidates, key=score)

    def _ordering_axis_value(
        self,
        pose: Pose,
        axis: str,
        surface_id: str | None,
        assignment: dict[NodeId, Pose],
    ) -> float:
        if surface_id:
            problem = self._ordering_surface_problems.get(surface_id)
            if problem is not None:
                surface = problem.resolve_surface(surface_id, assignment)
                if surface is not None:
                    local_x, local_y = surface.world_to_local_xy(pose.position)
                    return local_x if axis == "x" else local_y
        axis_idx = 0 if axis == "x" else 1
        return pose.position[axis_idx]

    @staticmethod
    def _trace_event(domain_id: DomainId, kind: str, *, candidates: dict[str, int]) -> dict[str, Any]:
        return {
            "scope_id": f"{domain_id.kind}:{domain_id.id}",
            "kind": kind,
            "domain_id": domain_id.to_json(),
            "objects": sorted(candidates),
            "meta": {
                "packer": "generic_domain_solver",
                "candidate_counts": dict(candidates),
            },
        }
