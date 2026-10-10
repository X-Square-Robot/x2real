from __future__ import annotations

from .backends.base import SceneBackend
from .materials.library import RandomizationLibraryInput
from .model import ApplyReport, RandomizationContext, RandomizationPlan, RandomizationReport, RandomizationSpec
from .runtime.executor import apply_plan as _apply_plan
from .runtime.planner import compile_plan as _compile_plan


def compile_plan(
    *,
    backend: SceneBackend,
    spec: RandomizationSpec,
    context: RandomizationContext,
    library: RandomizationLibraryInput | None = None,
) -> RandomizationPlan:
    return _compile_plan(backend=backend, spec=spec, context=context, library=library)


def apply_plan(*, backend: SceneBackend, plan: RandomizationPlan) -> ApplyReport:
    return _apply_plan(backend=backend, plan=plan)


def randomize_asset_appearance(
    *,
    backend: SceneBackend,
    spec: RandomizationSpec,
    context: RandomizationContext,
    library: RandomizationLibraryInput | None = None,
) -> RandomizationReport:
    plan = compile_plan(backend=backend, spec=spec, context=context, library=library)
    report = apply_plan(backend=backend, plan=plan)
    return RandomizationReport(plan=plan, apply_report=report)
