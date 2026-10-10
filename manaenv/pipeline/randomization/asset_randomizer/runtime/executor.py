from __future__ import annotations

from ..backends.base import SceneBackend
from ..model import ApplyReport, RandomizationPlan


def apply_plan(*, backend: SceneBackend, plan: RandomizationPlan) -> ApplyReport:
    if not plan.edits:
        return ApplyReport()
    return backend.apply_material_edits(plan.edits)
