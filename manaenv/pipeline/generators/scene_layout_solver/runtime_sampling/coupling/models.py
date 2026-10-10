from __future__ import annotations

from dataclasses import dataclass

from ...common.geometry import Bounds2D
from ..partitioning import ObjectProfile, PartitionCase, PartitionCell
from ..schema import AssignmentPlan, PlacementDomain


@dataclass(frozen=True)
class FixedScope:
    assignment_id: str
    object_name: str
    domain: PlacementDomain
    profile: ObjectProfile
    center_bounds: Bounds2D
    swept_bounds: Bounds2D
    source_event: str
    origin: str = "assignment"


@dataclass(frozen=True)
class GroupScope:
    assignment: AssignmentPlan
    surface: str
    bounds: Bounds2D
    frame: str
    domains: dict[str, PlacementDomain]
    profiles: dict[str, ObjectProfile]


@dataclass(frozen=True)
class FixedCase:
    scope: FixedScope
    cell: PartitionCell


@dataclass(frozen=True)
class JointCase:
    fixed: tuple[FixedCase, ...]
    group: PartitionCase


@dataclass(frozen=True)
class PlacementScope:
    scope_id: str
    kind: str
    assignment_id: str
    objects: tuple[str, ...]
    surface: str
    frame: str
    center_bounds: Bounds2D | None
    trigger_bounds: Bounds2D
    fixed: FixedScope | None = None
    group: GroupScope | None = None


@dataclass(frozen=True)
class CouplingComponent:
    scopes: tuple[PlacementScope, ...]

    @property
    def fixed_scopes(self) -> list[FixedScope]:
        return [scope.fixed for scope in self.scopes if scope.fixed is not None]

    @property
    def group_scopes(self) -> list[GroupScope]:
        return [scope.group for scope in self.scopes if scope.group is not None]
