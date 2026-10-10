from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .assets.bridge import build_asset_records
from .common.diagnostics import LayoutDiagnostic, LayoutSpecError
from .common.geometry import ManualSupportSurface, Pose
from .export.apply import apply_solved_poses
from .scene_spec.schema import SceneLayoutSpec
from .solver import ScopedBacktrackingSolver
from .static_solve.feasibility import FeasibilityAnalyzer
from .static_solve.problem import build_layout_problem
from .static_solve.solve.retry import RecoveryAttempt, RecoveryConfig, recovery_attempts
from .variants.planner import LayoutCase, VariantPlanner


@dataclass
class LayoutSolveResult:
    success: bool
    case: LayoutCase | None = None
    poses: dict[str, Pose] = field(default_factory=dict)
    diagnostics: list[LayoutDiagnostic] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "case_id": self.case.case_id if self.case else "",
            "poses": {
                name: {
                    "position": list(pose.position),
                    "quaternion": list(pose.quaternion),
                }
                for name, pose in self.poses.items()
            },
            "diagnostics": [diag.to_json() for diag in self.diagnostics],
            "artifacts": dict(self.artifacts),
        }


class SurfaceScopedRunner:
    def __init__(self, spec: SceneLayoutSpec, *, nums: int | None = None) -> None:
        self.spec = spec
        self.nums = nums
        self._cases = VariantPlanner(spec, nums=nums).plan()

    @property
    def cases(self) -> list[LayoutCase]:
        return list(self._cases)

    def case_for_index(self, variant_index: int) -> LayoutCase:
        if not self._cases:
            return LayoutCase(case_id="case_000")
        idx = int(variant_index) % len(self._cases)
        return self._cases[idx]

    def solve(
        self,
        assets: Mapping[str, Any] | Iterable[Any],
        *,
        variant_index: int = 0,
    ) -> LayoutSolveResult:
        if not self.spec.enabled:
            return LayoutSolveResult(success=True, case=LayoutCase(case_id="layout_disabled"))
        try:
            case = self.case_for_index(variant_index)
            surfaces = {surface.surface_id: ManualSupportSurface.from_spec(surface) for surface in self.spec.surfaces}
            records = build_asset_records(assets, self.spec.objects)
            problem = build_layout_problem(self.spec, case, surfaces, records)
            recovery_config = RecoveryConfig.from_dict(self.spec.recovery)
            attempts_json: list[dict[str, Any]] = []
            last_result = None
            last_feasibility = None
            for attempt in recovery_attempts(recovery_config):
                relation_overrides = self._relation_overrides_for_attempt(attempt)
                feasibility = FeasibilityAnalyzer(problem, relation_overrides=relation_overrides).analyze()
                last_feasibility = feasibility
                attempt_record = {
                    "attempt": attempt.to_json(),
                    "feasibility": feasibility.to_json(),
                    "solve_success": False,
                    "diagnostics": [],
                }
                attempts_json.append(attempt_record)
                if not feasibility.success:
                    attempt_record["diagnostics"] = [issue.to_json() for issue in feasibility.issues]
                    continue
                solver = ScopedBacktrackingSolver(
                    self.spec,
                    surfaces,
                    records,
                    problem=problem,
                    seed=self._case_seed(case, variant_index=variant_index, attempt=attempt),
                    relation_overrides=relation_overrides,
                    recovery_config=recovery_config,
                )
                result = solver.solve(case)
                last_result = result
                attempt_record["solve_success"] = bool(result.success)
                attempt_record["diagnostics"] = [diag.to_json() for diag in result.diagnostics]
                if result.success:
                    artifacts = {
                        "feasibility": feasibility.to_json(),
                        "recovery": {
                            "attempts": attempts_json,
                            "selected_attempt": attempt.to_json(),
                            "used_relaxation": bool(attempt.tier > 0),
                        },
                        **dict(result.artifacts),
                    }
                    return LayoutSolveResult(
                        success=True,
                        case=case,
                        poses=result.poses,
                        diagnostics=result.diagnostics,
                        artifacts=artifacts,
                    )

            diagnostics: list[LayoutDiagnostic]
            if last_result is not None:
                diagnostics = list(last_result.diagnostics)
            elif last_feasibility is not None:
                diagnostics = [issue.to_diagnostic() for issue in last_feasibility.issues]
            else:
                diagnostics = [
                    LayoutDiagnostic("layout_solver_no_attempt", "no recovery attempt was executed", target="runner")
                ]
            return LayoutSolveResult(
                success=False,
                case=case,
                poses=dict(last_result.poses) if last_result is not None else {},
                diagnostics=diagnostics,
                artifacts={
                    "feasibility": last_feasibility.to_json() if last_feasibility is not None else {},
                    "recovery": {
                        "attempts": attempts_json,
                        "selected_attempt": None,
                        "used_relaxation": False,
                    },
                    **(dict(last_result.artifacts) if last_result is not None else {}),
                },
            )
        except LayoutSpecError as exc:
            return LayoutSolveResult(success=False, diagnostics=[exc.diagnostic])
        except Exception as exc:
            return LayoutSolveResult(
                success=False,
                diagnostics=[
                    LayoutDiagnostic(
                        code="layout_solver_exception",
                        message=str(exc),
                        target="runner",
                    )
                ],
            )

    @staticmethod
    def _relation_overrides_for_attempt(attempt: RecoveryAttempt) -> dict[str, Any]:
        if attempt.surface_range_tier == "strict":
            return {}
        return {
            "support_on": {
                "surface_range_tier": attempt.surface_range_tier,
                "surface_range_expand_m": float(attempt.surface_range_expand_m),
            }
        }

    def _case_seed(self, case: LayoutCase, *, variant_index: int, attempt: RecoveryAttempt) -> int:
        raw = case.params.get("seed") if isinstance(case.params, dict) else None
        base = int(raw) if raw is not None else self.spec.seed + int(variant_index)
        return base + int(attempt.tier) * 1009


def layout_fingerprint(layout_payload: dict[str, Any]) -> str:
    raw = json.dumps(layout_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def solve_and_apply(
    layout_payload: dict[str, Any],
    assets: Mapping[str, Any] | Iterable[Any],
    *,
    variant_index: int = 0,
    nums: int | None = None,
) -> LayoutSolveResult:
    spec = SceneLayoutSpec.from_dict(layout_payload)
    runner = SurfaceScopedRunner(spec, nums=nums)
    result = runner.solve(assets, variant_index=variant_index)
    if result.success and result.poses:
        apply_solved_poses(assets, result.poses)
    return result
