from __future__ import annotations

from typing import Any

from ...common.reproducibility import stable_seed
from ..schema import AssignmentPlan, PlacementDomain, RuntimeSamplingPlan
from .builder import build_fixed_joint_assignment, build_joint_assignment
from .matcher import fixed_scopes, overlap_components, placement_scopes, polygon_overlap_warnings
from .models import CouplingComponent
from .rewriter import rewrite_assignments


def couple_same_surface_runtime_domains(plan: RuntimeSamplingPlan, *, seed: int = 0) -> None:
    """Couple overlapping runtime placement domains on the same support surface.

    Treating overlapping reset domains as independent lets runtime sampling
    collide. This pass rewrites supported overlap components into one joint
    case_pool while leaving non-overlapping domains untouched.
    """

    fixed = fixed_scopes(plan)
    for warning in polygon_overlap_warnings(plan, fixed):
        plan.add_diagnostic(
            "runtime_joint_partition_polygon_overlap_unsupported",
            "overlapping fixed domain and polygon-filtered region cannot be coupled yet",
            severity="error",
            **warning,
        )

    components = sorted(overlap_components(placement_scopes(plan)), key=_component_sort_key)
    if not components:
        return

    replaced_assignments: set[str] = set()
    consumed_fixed: set[tuple[str, str]] = set()
    remove_domains: set[str] = set()
    add_domains: list[PlacementDomain] = []
    add_assignments: list[AssignmentPlan] = []
    diagnostics: list[dict[str, Any]] = []

    for component in components:
        result = _build_component(
            component,
            seed=stable_seed(seed, "runtime_coupling_component", payload=_component_sort_key(component)),
        )
        if result is None:
            plan.add_diagnostic(
                "runtime_coupling_component_unsupported",
                "overlapping runtime placement domains could not be compiled into a supported joint component",
                severity="warning",
                target=",".join(scope.scope_id for scope in component.scopes),
                surface=component.scopes[0].surface if component.scopes else "",
                kinds=[scope.kind for scope in component.scopes],
                objects=[obj for scope in component.scopes for obj in scope.objects],
            )
            continue
        joint_domains, joint_assignment, meta = result
        diagnostics.append(meta)
        for group in component.group_scopes:
            replaced_assignments.add(group.assignment.plan_id)
            remove_domains.update(group.assignment.domains)
        for fixed_scope in component.fixed_scopes:
            if fixed_scope.origin == "assignment":
                consumed_fixed.add((fixed_scope.assignment_id, fixed_scope.object_name))
                remove_domains.add(fixed_scope.domain.domain_id)
        add_domains.extend(joint_domains)
        add_assignments.append(joint_assignment)

    if not add_assignments:
        return

    plan.domains = [domain for domain in plan.domains if domain.domain_id not in remove_domains]
    plan.domains.extend(add_domains)
    plan.assignments = rewrite_assignments(plan.assignments, replaced_assignments, consumed_fixed)
    plan.assignments.extend(add_assignments)
    plan.artifacts.setdefault("runtime_domain_coupling", []).extend(diagnostics)


def _build_component(
    component: CouplingComponent,
    *,
    seed: int,
) -> tuple[list[PlacementDomain], AssignmentPlan, dict[str, Any]] | None:
    groups = component.group_scopes
    fixed = component.fixed_scopes
    if not groups and len(fixed) >= 2:
        return build_fixed_joint_assignment(fixed, seed=seed)
    if len(groups) == 1 and fixed:
        return build_joint_assignment(groups[0], fixed, seed=seed)
    return None


def _component_sort_key(component: CouplingComponent) -> tuple[str, ...]:
    return tuple(sorted(scope.scope_id for scope in component.scopes))
