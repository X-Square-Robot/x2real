from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...common.partitioning.models import PartitionCase, PartitionProblem
from ...common.partitioning.scoring import tier_counts
from ..cell_domains import placement_domain_from_cell, safe_id
from ..compile_utils import bounds2d_json
from ..meta import (
    CELL_SHAPE_RECTANGLE,
    META_CELL_SHAPE,
    META_FOOTPRINT,
    META_FOOTPRINT_BOUNDS,
    META_PARTITION_CASE,
    META_PARTITION_TIER,
    META_PARTITION_TOPOLOGY,
    META_PLAN_KIND,
    META_POLYGON_FILTER,
    PLAN_PARTITION_CASE_POOL,
)
from ..schema import PlacementDomain
from ..yaw_choices import yaw_choice_semantic_meta


def export_partition_cases(
    cases: list[PartitionCase],
    *,
    problem: PartitionProblem,
    domain_id: str,
    frame: str,
    source_event: str,
    authoring_frame: str,
    parent: str | None,
    surface: str | None,
    yaw_by_object: Mapping[str, tuple[float, float] | None],
    default_yaw: tuple[float, float] | None,
    yaw_choices_by_object: Mapping[str, tuple[dict[str, Any], ...]] | None = None,
    base_quaternion_by_object: Mapping[str, tuple[float, float, float, float]] | None = None,
) -> tuple[tuple[PlacementDomain, ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    domains: list[PlacementDomain] = []
    payload_cases: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        bind: dict[str, str] = {}
        case_domains: list[str] = []
        for name in problem.names:
            cell = case.cells[name]
            profile = problem.profiles[name]
            footprint = profile.footprint
            boundary_gap = (
                problem.min_spacing * 0.5
                if getattr(problem, "spacing_mode", "footprint_gap") == "footprint_gap"
                else 0.0
            )
            child_domain_id = f"{domain_id}:case_{case_index:03d}:{_safe_id(name)}"
            yaw_choices = tuple(dict(yaw_choices_by_object or {}).get(name, ()) or ())
            bind[name] = child_domain_id
            case_domains.append(child_domain_id)
            domains.append(
                placement_domain_from_cell(
                    domain_id=child_domain_id,
                    object_name=name,
                    frame=frame,
                    surface=surface,
                    cell=cell,
                    profile=profile,
                    yaw=dict(yaw_by_object).get(name, default_yaw),
                    yaw_choices=yaw_choices,
                    base_quaternion=dict(base_quaternion_by_object or {}).get(name),
                    boundary_gap=boundary_gap,
                    meta={
                        "source_event": source_event,
                        "authoring_frame": authoring_frame,
                        **yaw_choice_semantic_meta(name, yaw_choices),
                        META_PARTITION_CASE: case_index,
                        META_PARTITION_TIER: case.tier,
                        META_PARTITION_TOPOLOGY: case.topology,
                        META_PLAN_KIND: PLAN_PARTITION_CASE_POOL,
                        META_CELL_SHAPE: CELL_SHAPE_RECTANGLE,
                        META_FOOTPRINT: [footprint[0], footprint[1]],
                        META_FOOTPRINT_BOUNDS: bounds2d_json(profile.footprint_bounds),
                        **({"parent": parent} if parent else {}),
                        **({"surface": surface} if surface else {}),
                    },
                )
            )
        payload_cases.append(
            {
                "id": f"partition_case_{case_index:03d}",
                "bind": bind,
                "domains": case_domains,
                "quality": case.quality,
                "tier": case.tier,
                "topology": case.topology,
                "metrics": dict(case.metrics),
                "feature": {name: [xy[0], xy[1]] for name, xy in case.feature.items()},
            }
        )
    meta = {
        META_PLAN_KIND: PLAN_PARTITION_CASE_POOL,
        "candidate_tiers": tier_counts(cases),
        "selected_count": len(cases),
        "min_spacing": problem.min_spacing,
        "bounds": bounds2d_json(problem.bounds),
        META_POLYGON_FILTER: problem.accept_cell is not None,
    }
    return tuple(domains), tuple(payload_cases), meta


def _safe_id(value: str) -> str:
    return safe_id(value)
