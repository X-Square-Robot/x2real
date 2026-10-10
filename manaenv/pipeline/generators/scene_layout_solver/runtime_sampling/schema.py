from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from ..common.diagnostics import LayoutDiagnostic
from .conflicts import runtime_conflict_diagnostics
from .semantic_pools import semantic_pool_artifacts


def _as_list(values: Iterable[Any] | None) -> list[Any]:
    return list(values or [])


def _diagnostic(code: str, message: str, *, target: str = "", severity: str = "error", **meta: Any) -> LayoutDiagnostic:
    return LayoutDiagnostic(code=code, message=message, target=target, severity=severity, meta=dict(meta))


@dataclass(frozen=True)
class PlacementDomain:
    domain_id: str
    object_name: str
    frame: str
    geometry_type: str = "rectangle"
    surface: str | None = None
    x: tuple[float, float] | None = None
    y: tuple[float, float] | None = None
    z: tuple[float, float] | None = None
    yaw: tuple[float, float] | None = None
    base_quaternion: tuple[float, float, float, float] | None = None
    cells: tuple[dict[str, Any], ...] = ()
    poses: tuple[dict[str, Any], ...] = ()
    range_choices: tuple[dict[str, Any], ...] = ()
    yaw_choices: tuple[dict[str, Any], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.domain_id,
            "object": self.object_name,
            "frame": self.frame,
            "geometry_type": self.geometry_type,
            "meta": dict(self.meta),
        }
        if self.surface is not None:
            payload["surface"] = self.surface
        if self.x is not None:
            payload["x"] = [float(self.x[0]), float(self.x[1])]
        if self.y is not None:
            payload["y"] = [float(self.y[0]), float(self.y[1])]
        if self.z is not None:
            payload["z"] = [float(self.z[0]), float(self.z[1])]
        if self.yaw is not None:
            payload["yaw"] = [float(self.yaw[0]), float(self.yaw[1])]
        if self.base_quaternion is not None:
            payload["base_quaternion"] = [float(value) for value in self.base_quaternion]
        if self.cells:
            payload["cells"] = [dict(cell) for cell in self.cells]
        if self.poses:
            payload["poses"] = [dict(pose) for pose in self.poses]
        if self.range_choices:
            payload["range_choices"] = [dict(choice) for choice in self.range_choices]
        if self.yaw_choices:
            payload["yaw_choices"] = [dict(choice) for choice in self.yaw_choices]
        return payload


@dataclass(frozen=True)
class AssignmentPlan:
    plan_id: str
    mode: str
    objects: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    bind: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    cases: tuple[dict[str, Any], ...] = ()
    source_event: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.plan_id,
            "mode": self.mode,
            "objects": list(self.objects),
            "domains": list(self.domains),
            "bind": dict(self.bind),
            "constraints": dict(self.constraints),
            "cases": [dict(case) for case in self.cases],
            "source_event": self.source_event,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class ConditionalAssignmentPlan:
    plan_id: str
    parent: str
    children: tuple[str, ...]
    parent_domain: str
    child_domains: dict[str, str]
    dependencies: dict[str, tuple[str, ...]] = field(default_factory=dict)
    cases: tuple[dict[str, Any], ...] = ()
    source_event: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.plan_id,
            "parent": self.parent,
            "children": list(self.children),
            "parent_domain": self.parent_domain,
            "child_domains": dict(self.child_domains),
            "dependencies": {name: list(deps) for name, deps in self.dependencies.items()},
            "cases": [dict(case) for case in self.cases],
            "source_event": self.source_event,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class RuntimeOutputPlan:
    plan_id: str
    output_name: str
    mode: str
    values: tuple[Any, ...] = ()
    cases: tuple[dict[str, Any], ...] = ()
    source_event: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.plan_id,
            "name": self.output_name,
            "mode": self.mode,
            "values": list(self.values),
            "cases": [dict(case) for case in self.cases],
            "source_event": self.source_event,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class RuntimeEventPatch:
    event_name: str
    mode: str
    params: dict[str, Any]
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "event_name": self.event_name,
            "mode": self.mode,
            "params": dict(self.params),
            "reason": self.reason,
            "meta": dict(self.meta),
        }


@dataclass
class RuntimeSamplingPlan:
    success: bool = True
    domains: list[PlacementDomain] = field(default_factory=list)
    assignments: list[AssignmentPlan] = field(default_factory=list)
    conditional_assignments: list[ConditionalAssignmentPlan] = field(default_factory=list)
    outputs: list[RuntimeOutputPlan] = field(default_factory=list)
    event_patches: list[RuntimeEventPatch] = field(default_factory=list)
    diagnostics: list[LayoutDiagnostic] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def add_diagnostic(
        self,
        code: str,
        message: str,
        *,
        target: str = "",
        severity: str = "error",
        **meta: Any,
    ) -> None:
        if severity == "error":
            self.success = False
        self.diagnostics.append(_diagnostic(code, message, target=target, severity=severity, **meta))

    def validate(self) -> list[LayoutDiagnostic]:  # noqa: C901
        diagnostics: list[LayoutDiagnostic] = []
        domain_ids = [domain.domain_id for domain in self.domains]
        duplicate_domains = sorted({domain_id for domain_id in domain_ids if domain_ids.count(domain_id) > 1})
        for domain_id in duplicate_domains:
            diagnostics.append(_diagnostic("runtime_domain_duplicate", "domain ids must be unique", target=domain_id))

        known_domains = set(domain_ids)
        output_ids = [output.plan_id for output in self.outputs]
        duplicate_outputs = sorted({output_id for output_id in output_ids if output_ids.count(output_id) > 1})
        for output_id in duplicate_outputs:
            diagnostics.append(_diagnostic("runtime_output_duplicate", "output ids must be unique", target=output_id))

        for domain in self.domains:
            if domain.geometry_type not in {
                "rectangle",
                "box",
                "cylinder",
                "cells",
                "polygon",
                "solution_jitter",
                "pose_list",
                "surface_range_choice",
            }:
                diagnostics.append(
                    _diagnostic(
                        "runtime_domain_geometry_unsupported",
                        "unsupported runtime domain geometry",
                        target=domain.domain_id,
                        geometry_type=domain.geometry_type,
                    )
                )
            if domain.geometry_type == "rectangle" and (domain.x is None or domain.y is None):
                diagnostics.append(
                    _diagnostic(
                        "runtime_domain_rectangle_invalid",
                        "rectangle domain requires x and y ranges",
                        target=domain.domain_id,
                    )
                )
            if domain.geometry_type in {"box", "cylinder"} and (
                domain.x is None or domain.y is None or domain.z is None
            ):
                diagnostics.append(
                    _diagnostic(
                        "runtime_domain_volume_invalid",
                        "box/cylinder domain requires x, y, and z ranges",
                        target=domain.domain_id,
                    )
                )
            if domain.geometry_type == "cylinder" and float((domain.meta or {}).get("radius") or 0.0) <= 0.0:
                diagnostics.append(
                    _diagnostic(
                        "runtime_domain_cylinder_radius_missing",
                        "cylinder domain requires positive meta.radius",
                        target=domain.domain_id,
                    )
                )
            if domain.geometry_type == "pose_list" and not domain.poses:
                diagnostics.append(
                    _diagnostic(
                        "runtime_domain_pose_list_empty", "pose_list domain requires poses", target=domain.domain_id
                    )
                )
            if domain.geometry_type == "pose_list" and domain.poses:
                diagnostics.extend(
                    _weighted_items_diagnostics(
                        domain.poses,
                        target=domain.domain_id,
                        invalid_code="runtime_domain_pose_weight_invalid",
                        all_zero_code="runtime_domain_pose_weights_all_zero",
                        invalid_message="pose_list candidate weights must be finite and non-negative",
                        all_zero_message="pose_list domain requires at least one positive pose weight",
                        item_kind="pose",
                    )
                )
            if domain.geometry_type == "surface_range_choice" and not domain.range_choices:
                diagnostics.append(
                    _diagnostic(
                        "runtime_domain_surface_range_choice_empty",
                        "surface_range_choice domain requires range choices",
                        target=domain.domain_id,
                    )
                )
            if domain.geometry_type == "surface_range_choice" and domain.range_choices:
                diagnostics.extend(
                    _weighted_items_diagnostics(
                        domain.range_choices,
                        target=domain.domain_id,
                        invalid_code="runtime_domain_range_choice_weight_invalid",
                        all_zero_code="runtime_domain_range_choice_weights_all_zero",
                        invalid_message="surface_range_choice weights must be finite and non-negative",
                        all_zero_message="surface_range_choice domain requires at least one positive range weight",
                        item_kind="range_choice",
                    )
                )

        for assignment in self.assignments:
            if assignment.mode not in {"fixed", "partition", "permutation", "case_pool"}:
                diagnostics.append(
                    _diagnostic(
                        "runtime_assignment_mode_unsupported",
                        "unsupported assignment mode",
                        target=assignment.plan_id,
                        mode=assignment.mode,
                    )
                )
            for domain_id in assignment.domains:
                if domain_id not in known_domains:
                    diagnostics.append(
                        _diagnostic(
                            "runtime_assignment_domain_unknown",
                            "assignment references unknown domain",
                            target=assignment.plan_id,
                            domain=domain_id,
                        )
                    )
            for obj, domain_id in assignment.bind.items():
                if obj not in assignment.objects:
                    diagnostics.append(
                        _diagnostic(
                            "runtime_assignment_bind_object_unknown",
                            "assignment binds object outside object list",
                            target=assignment.plan_id,
                            object=obj,
                        )
                    )
                if domain_id not in known_domains:
                    diagnostics.append(
                        _diagnostic(
                            "runtime_assignment_bind_domain_unknown",
                            "assignment bind references unknown domain",
                            target=assignment.plan_id,
                            domain=domain_id,
                        )
                    )
            for case in assignment.cases:
                raw_bind = case.get("bind") if isinstance(case, Mapping) else None
                if not isinstance(raw_bind, Mapping):
                    diagnostics.append(
                        _diagnostic(
                            "runtime_assignment_case_bind_missing",
                            "assignment case requires bind mapping",
                            target=assignment.plan_id,
                        )
                    )
                    continue
                for obj, domain_id in raw_bind.items():
                    if str(obj) not in assignment.objects:
                        diagnostics.append(
                            _diagnostic(
                                "runtime_assignment_case_object_unknown",
                                "assignment case binds object outside object list",
                                target=assignment.plan_id,
                                object=str(obj),
                            )
                        )
                    if str(domain_id) not in known_domains:
                        diagnostics.append(
                            _diagnostic(
                                "runtime_assignment_case_domain_unknown",
                                "assignment case bind references unknown domain",
                                target=assignment.plan_id,
                                domain=str(domain_id),
                            )
                        )
            if assignment.mode == "case_pool" and assignment.cases:
                diagnostics.extend(
                    _weighted_items_diagnostics(
                        assignment.cases,
                        target=assignment.plan_id,
                        invalid_code="runtime_assignment_case_weight_invalid",
                        all_zero_code="runtime_assignment_case_weights_all_zero",
                        invalid_message="assignment case weights must be finite and non-negative",
                        all_zero_message="case_pool assignment requires at least one positive case weight",
                        item_kind="case",
                    )
                )

        for conditional in self.conditional_assignments:
            if conditional.parent_domain not in known_domains:
                diagnostics.append(
                    _diagnostic(
                        "runtime_conditional_parent_domain_unknown",
                        "conditional plan references unknown parent domain",
                        target=conditional.plan_id,
                        domain=conditional.parent_domain,
                    )
                )
            for child, domain_id in conditional.child_domains.items():
                if child not in conditional.children:
                    diagnostics.append(
                        _diagnostic(
                            "runtime_conditional_child_unknown",
                            "conditional child domain references unknown child",
                            target=conditional.plan_id,
                            child=child,
                        )
                    )
                if domain_id not in known_domains:
                    diagnostics.append(
                        _diagnostic(
                            "runtime_conditional_child_domain_unknown",
                            "conditional plan references unknown child domain",
                            target=conditional.plan_id,
                            domain=domain_id,
                        )
                    )
            for case in conditional.cases:
                raw_bind = case.get("bind") if isinstance(case, Mapping) else None
                if not isinstance(raw_bind, Mapping):
                    diagnostics.append(
                        _diagnostic(
                            "runtime_conditional_case_bind_missing",
                            "conditional case requires bind mapping",
                            target=conditional.plan_id,
                        )
                    )
                    continue
                for obj, domain_id in raw_bind.items():
                    if str(obj) not in conditional.children:
                        diagnostics.append(
                            _diagnostic(
                                "runtime_conditional_case_child_unknown",
                                "conditional case binds object outside children",
                                target=conditional.plan_id,
                                object=str(obj),
                            )
                        )
                    if str(domain_id) not in known_domains:
                        diagnostics.append(
                            _diagnostic(
                                "runtime_conditional_case_domain_unknown",
                                "conditional case bind references unknown domain",
                                target=conditional.plan_id,
                                domain=str(domain_id),
                            )
                        )
            if conditional.cases:
                diagnostics.extend(
                    _weighted_items_diagnostics(
                        conditional.cases,
                        target=conditional.plan_id,
                        invalid_code="runtime_conditional_case_weight_invalid",
                        all_zero_code="runtime_conditional_case_weights_all_zero",
                        invalid_message="conditional case weights must be finite and non-negative",
                        all_zero_message="conditional case_pool requires at least one positive case weight",
                        item_kind="case",
                    )
                )
            diagnostics.extend(_cycle_diagnostics(conditional))

        for output in self.outputs:
            if output.mode not in {"fixed", "case_pool", "derived"}:
                diagnostics.append(
                    _diagnostic(
                        "runtime_output_mode_unsupported",
                        "unsupported runtime output mode",
                        target=output.plan_id,
                        mode=output.mode,
                    )
                )
            if not output.output_name:
                diagnostics.append(
                    _diagnostic("runtime_output_name_missing", "runtime output requires name", target=output.plan_id)
                )
            if output.mode == "fixed" and not output.values:
                diagnostics.append(
                    _diagnostic(
                        "runtime_output_values_empty",
                        "fixed runtime output requires at least one value",
                        target=output.plan_id,
                    )
                )
            for case in output.cases:
                if not isinstance(case, Mapping) or not isinstance(case.get("values"), list):
                    diagnostics.append(
                        _diagnostic(
                            "runtime_output_case_values_missing",
                            "runtime output case requires values list",
                            target=output.plan_id,
                        )
                    )
            if output.mode == "derived":
                source = str((output.meta or {}).get("from") or "").strip()
                if not source:
                    diagnostics.append(
                        _diagnostic(
                            "runtime_output_derived_source_missing",
                            "derived runtime output requires meta.from",
                            target=output.plan_id,
                        )
                    )
            if output.mode == "case_pool":
                if not output.cases:
                    diagnostics.append(
                        _diagnostic(
                            "runtime_output_cases_empty",
                            "case_pool runtime output requires cases",
                            target=output.plan_id,
                        )
                    )
                else:
                    diagnostics.extend(
                        _weighted_items_diagnostics(
                            output.cases,
                            target=output.plan_id,
                            invalid_code="runtime_output_case_weight_invalid",
                            all_zero_code="runtime_output_case_weights_all_zero",
                            invalid_message="runtime output case weights must be finite and non-negative",
                            all_zero_message="runtime output requires at least one positive case weight",
                            item_kind="output_case",
                        )
                    )
        diagnostics.extend(_duplicate_runtime_owner_diagnostics(self.assignments, self.conditional_assignments))
        diagnostics.extend(runtime_conflict_diagnostics(self))
        return diagnostics

    def finalize(self) -> RuntimeSamplingPlan:
        pools = semantic_pool_artifacts(self)
        if pools:
            self.artifacts["semantic_pools"] = pools
        else:
            self.artifacts.pop("semantic_pools", None)
        diagnostics = self.validate()
        if diagnostics:
            self.success = False
            self.diagnostics.extend(diagnostics)
        return self

    def to_json(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "domains": [domain.to_json() for domain in self.domains],
            "assignments": [assignment.to_json() for assignment in self.assignments],
            "conditional_assignments": [assignment.to_json() for assignment in self.conditional_assignments],
            "outputs": [output.to_json() for output in self.outputs],
            "event_patches": [patch.to_json() for patch in self.event_patches],
            "diagnostics": [diag.to_json() for diag in self.diagnostics],
            "artifacts": dict(self.artifacts),
        }


def _cycle_diagnostics(plan: ConditionalAssignmentPlan) -> list[LayoutDiagnostic]:
    deps: dict[str, set[str]] = {name: set(values) for name, values in plan.dependencies.items()}
    deps.setdefault(plan.parent, set())
    for child in plan.children:
        deps.setdefault(child, set()).add(plan.parent)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dep in sorted(deps.get(node, set())):
            if visit(dep):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    if any(visit(node) for node in sorted(deps)):
        return [
            _diagnostic(
                "runtime_conditional_cycle", "conditional assignment dependencies must form a DAG", target=plan.plan_id
            )
        ]
    return []


def _weighted_items_diagnostics(
    items: Iterable[Any],
    *,
    target: str,
    invalid_code: str,
    all_zero_code: str,
    invalid_message: str,
    all_zero_message: str,
    item_kind: str,
) -> list[LayoutDiagnostic]:
    diagnostics: list[LayoutDiagnostic] = []
    positive_count = 0
    item_count = 0
    for index, item in enumerate(items):
        item_count += 1
        if not isinstance(item, Mapping):
            diagnostics.append(
                _diagnostic(
                    invalid_code,
                    invalid_message,
                    target=target,
                    item_kind=item_kind,
                    index=index,
                    reason="item_not_mapping",
                )
            )
            continue
        raw_weight = item.get("weight", 1.0)
        weight = _finite_nonnegative_weight(raw_weight)
        if weight is None:
            diagnostics.append(
                _diagnostic(
                    invalid_code,
                    invalid_message,
                    target=target,
                    item_kind=item_kind,
                    index=index,
                    candidate_id=str(item.get("id") or ""),
                    weight=raw_weight,
                )
            )
            continue
        if weight > 0.0:
            positive_count += 1
    if item_count and positive_count == 0:
        diagnostics.append(_diagnostic(all_zero_code, all_zero_message, target=target, item_kind=item_kind))
    return diagnostics


def _finite_nonnegative_weight(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return None
    try:
        weight = float(raw)
    except (TypeError, ValueError):
        return None
    if not isfinite(weight) or weight < 0.0:
        return None
    return weight


def _duplicate_runtime_owner_diagnostics(
    assignments: Iterable[AssignmentPlan],
    conditionals: Iterable[ConditionalAssignmentPlan],
) -> list[LayoutDiagnostic]:
    owners_by_object: dict[str, set[str]] = {}

    def add_owner(object_name: str, owner: str) -> None:
        name = str(object_name).strip()
        if not name:
            return
        owners_by_object.setdefault(name, set()).add(owner)

    for assignment in assignments:
        owner = f"assignment:{assignment.plan_id}"
        for object_name in _assignment_owned_objects(assignment):
            add_owner(object_name, owner)
    for conditional in conditionals:
        owner = f"conditional:{conditional.plan_id}"
        owned = set(str(child) for child in conditional.children)
        for case in conditional.cases:
            raw_bind = case.get("bind") if isinstance(case, Mapping) else None
            if isinstance(raw_bind, Mapping):
                owned.update(str(name) for name in raw_bind)
        for object_name in owned:
            add_owner(object_name, owner)

    diagnostics: list[LayoutDiagnostic] = []
    for object_name, owners in sorted(owners_by_object.items()):
        if len(owners) <= 1:
            continue
        diagnostics.append(
            _diagnostic(
                "runtime_object_duplicate_owner",
                "runtime object is controlled by multiple assignment owners",
                target=object_name,
                owners=sorted(owners),
            )
        )
    return diagnostics


def _assignment_owned_objects(assignment: AssignmentPlan) -> set[str]:
    owned = set(str(value) for value in assignment.objects)
    owned.update(str(value) for value in assignment.bind)
    for case in assignment.cases:
        raw_bind = case.get("bind") if isinstance(case, Mapping) else None
        if isinstance(raw_bind, Mapping):
            owned.update(str(value) for value in raw_bind)
    return owned


def diagnostics_to_json(diagnostics: Iterable[LayoutDiagnostic]) -> list[dict[str, Any]]:
    return [diag.to_json() for diag in diagnostics]


def normalize_role_bindings(raw: Mapping[str, Any] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, value in dict(raw or {}).items():
        if value is None:
            out[str(key)] = []
        elif isinstance(value, str):
            out[str(key)] = [value]
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
            out[str(key)] = [str(item) for item in value]
        else:
            out[str(key)] = [str(value)]
    return out
