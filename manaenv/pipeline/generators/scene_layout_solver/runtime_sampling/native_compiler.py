from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..assets.bridge import LayoutAssetRecord, build_asset_records
from ..common.diagnostics import LayoutDiagnostic, LayoutSpecError
from ..common.geometry import ManualSupportSurface
from ..common.reproducibility import RUNTIME_PARTITION_VERSION, RUNTIME_SAMPLER_VERSION, stable_fingerprint
from ..scene_spec.authoring import instance_aliases_from_layout, materialize_task_event_terms
from ..scene_spec.schema import SceneLayoutSpec
from ..static_solve.compiler import MiniLayoutCompiler
from ..static_solve.ir import CompiledLayoutIR
from ..static_solve.problem import LayoutProblem, build_layout_problem
from ..variants.planner import LayoutCase, VariantPlanner
from .authoring_compiler import RuntimeAuthoringCompilerMixin
from .compile_utils import surface_range_json
from .domain_export import RuntimeDomainExporter
from .events import NormalizedRuntimeEvent
from .graph import RuntimeCompileGraph, RuntimeCompileGraphBuilder
from .schema import RuntimeSamplingPlan, normalize_role_bindings
from .support_domains import runtime_surface_json


class TaskLayoutCompiler(RuntimeAuthoringCompilerMixin):
    """Compile ManaEnv reset randomization terms into cheap runtime sampling plans.

    This release-core compiler only understands scene-layout runtime events.
    Legacy ManaEnv reset event migration belongs in tools/manaenv_legacy or a
    bridge adapter before this compiler is called.
    """

    def __init__(
        self,
        layout_payload: Mapping[str, Any],
        assets: Mapping[str, Any] | Iterable[Any],
        *,
        task_payload: Mapping[str, Any] | None = None,
        event_terms: Mapping[str, Any] | None = None,
        role_bindings: Mapping[str, Any] | None = None,
        nums: int | None = None,
        variant_index: int = 0,
        variant_case: LayoutCase | None = None,
        seed: int = 0,
    ) -> None:
        self.layout_payload = dict(layout_payload or {})
        self.assets = assets
        self.task_payload = dict(task_payload or {})
        raw_terms = event_terms
        if raw_terms is None:
            event_section = self.task_payload.get("event") if isinstance(self.task_payload, dict) else {}
            raw_terms = (event_section or {}).get("terms") if isinstance(event_section, dict) else {}
        self.event_terms = materialize_task_event_terms(
            dict(raw_terms or {}),
            instance_aliases=instance_aliases_from_layout(self.layout_payload),
        )
        self.role_bindings = normalize_role_bindings(role_bindings)
        self.nums = nums
        self.variant_index = int(variant_index)
        self.variant_case = variant_case
        self.seed = int(seed)
        self.plan = RuntimeSamplingPlan()
        self.spec: SceneLayoutSpec | None = None
        self.surfaces: dict[str, ManualSupportSurface] = {}
        self.records: dict[str, LayoutAssetRecord] = {}
        self.problem: LayoutProblem | None = None
        self.ir: CompiledLayoutIR | None = None
        self.runtime_graph: RuntimeCompileGraph | None = None
        self.domain_exporter: RuntimeDomainExporter | None = None

    def compile(self) -> RuntimeSamplingPlan:
        try:
            self._build_problem()
            self.plan.artifacts["reproducibility"] = {
                "sampler_version": RUNTIME_SAMPLER_VERSION,
                "partition_version": RUNTIME_PARTITION_VERSION,
                "compile_seed": self.seed,
                "layout_fingerprint": stable_fingerprint(self.layout_payload),
                "event_terms_fingerprint": stable_fingerprint(self.event_terms),
            }
            self.plan.artifacts["support_surfaces"] = [
                runtime_surface_json(surface) for surface in self.surfaces.values()
            ]
            events = self._normalize_events()
            self.plan.artifacts["normalized_events"] = [
                {
                    "name": event.name,
                    "objects": list(event.objects),
                    "reference": event.reference,
                    "params": dict(event.params),
                }
                for event in events
            ]
            if self.ir is not None and self.ir.unsupported:
                for diagnostic in self.ir.unsupported:
                    self.plan.add_diagnostic(
                        diagnostic.code,
                        diagnostic.message,
                        target=diagnostic.target,
                        **diagnostic.meta,
                    )
                return self.plan.finalize()
            for event in events:
                self._compile_event(event)
        except LayoutSpecError as exc:
            self.plan.add_diagnostic(
                exc.diagnostic.code, exc.diagnostic.message, target=exc.diagnostic.target, **exc.diagnostic.meta
            )
        except Exception as exc:
            self.plan.add_diagnostic("task_layout_compile_exception", str(exc), target="task_layout_compiler")
        return self.plan.finalize()

    def _build_problem(self) -> None:
        self.spec = SceneLayoutSpec.from_dict(self.layout_payload)
        self.surfaces = {surface.surface_id: ManualSupportSurface.from_spec(surface) for surface in self.spec.surfaces}
        self.records = build_asset_records(self.assets, self.spec.objects)
        planner = VariantPlanner(self.spec, nums=self.nums)
        cases = planner.plan()
        if self.variant_case is None:
            self.variant_case = (
                cases[self.variant_index % len(cases)] if cases else LayoutCase(case_id="task_runtime_compile")
            )
        self.problem = build_layout_problem(
            self.spec,
            self.variant_case,
            self.surfaces,
            self.records,
        )
        self.ir = MiniLayoutCompiler(self.problem, self.records, case=self.variant_case).compile()
        self.runtime_graph = RuntimeCompileGraphBuilder(self.problem, self.ir).build()
        self.domain_exporter = RuntimeDomainExporter(self.problem, self.records, self.ir)
        self.plan.artifacts["variant"] = {
            "variant_index": int(self.variant_index),
            "case_id": self.variant_case.case_id,
            "object_overrides": {
                name: {
                    "surface": override.surface,
                    "slot_id": override.slot_id,
                    "surface_range": surface_range_json(override.surface_range),
                }
                for name, override in self.variant_case.object_overrides.items()
            },
        }
        self.plan.artifacts["runtime_compile_graph"] = self.runtime_graph.to_json()

    def _normalize_events(self) -> list[NormalizedRuntimeEvent]:
        ignored: list[str] = []
        for name in self.event_terms:
            if str(name).startswith("layout."):
                self.plan.add_diagnostic(
                    "layout_runtime_event_unsupported",
                    "legacy scene-layout runtime events are not supported by schema-vNext release",
                    target=str(name),
                )
            else:
                ignored.append(str(name))
        if ignored:
            self.plan.artifacts["ignored_non_layout_events"] = ignored
        return []

    def _add_export_diagnostic(self, diagnostic: LayoutDiagnostic) -> None:
        self.plan.add_diagnostic(
            diagnostic.code,
            diagnostic.message,
            target=diagnostic.target,
            severity=diagnostic.severity,
            **diagnostic.meta,
        )
