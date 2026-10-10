from __future__ import annotations

from dataclasses import dataclass, field

from .assets.bridge import LayoutAssetRecord
from .common.diagnostics import LayoutDiagnostic
from .common.geometry import ManualSupportSurface, Pose
from .common.reproducibility import STATIC_SOLVER_VERSION, pose_signature, stable_fingerprint
from .scene_spec.schema import SceneLayoutSpec
from .static_solve.compiler import MiniLayoutCompiler
from .static_solve.problem import LayoutProblem
from .static_solve.solve.kernel import MiniKernelSolver
from .static_solve.solve.retry import RecoveryConfig
from .variants.planner import LayoutCase


@dataclass
class SolverResult:
    success: bool
    poses: dict[str, Pose] = field(default_factory=dict)
    diagnostics: list[LayoutDiagnostic] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)


class ScopedBacktrackingSolver:
    """Facade over the mini domain kernel.

    The public runner-facing API is kept stable, but the solve semantics now
    come from compiled domains and hard factors rather than scope-kind dispatch.
    """

    def __init__(
        self,
        spec: SceneLayoutSpec,
        surfaces: dict[str, ManualSupportSurface],
        records: dict[str, LayoutAssetRecord],
        *,
        problem: LayoutProblem,
        seed: int = 0,
        relation_overrides: dict | None = None,
        recovery_config: RecoveryConfig | None = None,
    ) -> None:
        self.spec = spec
        self.surfaces = surfaces
        self.records = records
        self.problem = problem
        self.seed = int(seed)
        self.relation_overrides = dict(relation_overrides or {})
        self.recovery_config = recovery_config or RecoveryConfig.from_dict(spec.recovery)

    def solve(self, case: LayoutCase) -> SolverResult:
        ir = MiniLayoutCompiler(
            self.problem,
            self.records,
            case=case,
            relation_overrides=self.relation_overrides,
        ).compile()
        solver = MiniKernelSolver(
            ir,
            seed=self.seed,
            max_candidates=max(64, int(self.recovery_config.support_budget_floor)),
            max_search_steps=max(1024, int(self.recovery_config.support_budget_ceiling)),
            relation_overrides=self.relation_overrides,
        )
        result = solver.solve()
        reproducibility = {
            "version": STATIC_SOLVER_VERSION,
            "seed": self.seed,
            "case_id": case.case_id,
            "ir_fingerprint": stable_fingerprint(ir.to_json()),
            "surface_fingerprint": stable_fingerprint(_surface_payload(self.surfaces)),
            "asset_fingerprint": stable_fingerprint(_asset_payload(self.records)),
            "pose_fingerprint": stable_fingerprint(pose_signature(result.poses)) if result.success else "",
        }
        return SolverResult(
            success=result.success,
            poses=result.poses,
            diagnostics=result.diagnostics,
            artifacts={
                "kernel": result.artifacts,
                "kernel_trace": result.trace,
                "reproducibility": reproducibility,
            },
        )


def _range_payload(axis: object) -> list[float]:
    return [round(float(getattr(axis, "min")), 8), round(float(getattr(axis, "max")), 8)]


def _surface_payload(surfaces: dict[str, ManualSupportSurface]) -> dict:
    return {
        name: {
            "center": [round(float(v), 8) for v in surface.center],
            "x_axis": [round(float(v), 8) for v in surface.x_axis],
            "y_axis": [round(float(v), 8) for v in surface.y_axis],
            "normal": [round(float(v), 8) for v in surface.normal],
            "bounds": {
                "x": _range_payload(surface.bounds.x),
                "y": _range_payload(surface.bounds.y),
            },
            "shape": surface.shape,
            "margin": round(float(surface.margin), 8),
        }
        for name, surface in sorted(surfaces.items())
    }


def _asset_payload(records: dict[str, LayoutAssetRecord]) -> dict:
    return {
        name: {
            "usd_path": record.usd_path,
            "scale": [round(float(v), 8) for v in record.scale],
            "local_bounds": {
                "x": _range_payload(record.local_bounds.x),
                "y": _range_payload(record.local_bounds.y),
                "z": _range_payload(record.local_bounds.z),
            },
            "tags": sorted(record.tags),
        }
        for name, record in sorted(records.items())
    }
