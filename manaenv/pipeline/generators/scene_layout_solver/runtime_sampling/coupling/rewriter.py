from __future__ import annotations

from ..schema import AssignmentPlan


def rewrite_assignments(
    assignments: list[AssignmentPlan],
    replaced_assignments: set[str],
    consumed_fixed: set[tuple[str, str]],
) -> list[AssignmentPlan]:
    out: list[AssignmentPlan] = []
    for assignment in assignments:
        if assignment.plan_id in replaced_assignments:
            continue
        if assignment.mode != "fixed":
            out.append(assignment)
            continue
        bind = {
            obj: domain_id
            for obj, domain_id in assignment.bind.items()
            if (assignment.plan_id, str(obj)) not in consumed_fixed
        }
        if not bind:
            continue
        out.append(
            AssignmentPlan(
                plan_id=assignment.plan_id,
                mode=assignment.mode,
                objects=tuple(bind),
                domains=tuple(bind.values()),
                bind=bind,
                constraints=dict(assignment.constraints),
                cases=tuple(assignment.cases),
                source_event=assignment.source_event,
                meta=dict(assignment.meta),
            )
        )
    return out
