from __future__ import annotations

from typing import Any

from ...assets.bridge import LayoutAssetRecord
from ...assets.collision import CollisionObject, objects_collide
from ...common.constants import GEOM_EPS
from ...common.geometry import ManualSupportSurface, Pose, Range1D, bounds_overlap, world_aabb_for_pose
from ...common.ids import NodeId
from ...common.relations import relation_axis_and_order
from ...scene_spec.schema import RelativeDirectionSpec
from ..domains.manager import DomainManager
from ..ir import CompiledLayoutIR
from ..problem import LayoutProblem
from .base import HardFactor
from .relation_geometry import footprint_axis_span, footprint_ranges_on_surface, footprint_ranges_separated


def normalize_ignore_pairs(raw: object) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if not isinstance(raw, list):
        return pairs
    for item in raw:
        if isinstance(item, str):
            parts = [part.strip() for part in item.replace(":", ",").split(",") if part.strip()]
        elif isinstance(item, (list, tuple)):
            parts = [str(part).strip() for part in item if str(part).strip()]
        else:
            continue
        if len(parts) < 2:
            continue
        pairs.add(tuple(sorted((parts[0], parts[1]))))
    return pairs


class CollisionFactor(HardFactor):
    priority = 200

    def __init__(
        self,
        records: dict[str, LayoutAssetRecord],
        *,
        node_id: str,
        mode: str = "multi_box",
        eps: float = GEOM_EPS,
        ignore_pairs: set[tuple[str, str]] | None = None,
    ) -> None:
        self.records = records
        self.node_id = str(node_id)
        self.mode = "aabb" if str(mode).lower() == "aabb" else "multi_box"
        self.eps = max(0.0, float(eps))
        self.ignore_pairs = set(ignore_pairs or set())
        self._last_pair: tuple[str, str] | None = None

    def check(self, assignment: dict[NodeId, Pose]) -> bool:
        if self.node_id not in assignment:
            return True
        candidate = CollisionObject(
            name=self.node_id,
            proxy=self.records[self.node_id].collision_proxy,
            pose=assignment[self.node_id],
        )
        for other_name, other_pose in assignment.items():
            if other_name == self.node_id or other_name not in self.records:
                continue
            pair = tuple(sorted((self.node_id, other_name)))
            if pair in self.ignore_pairs:
                continue
            other = CollisionObject(name=other_name, proxy=self.records[other_name].collision_proxy, pose=other_pose)
            if self._collides(candidate, other):
                self._last_pair = pair
                return False
        self._last_pair = None
        return True

    def _collides(self, candidate: CollisionObject, other: CollisionObject) -> bool:
        if self.mode == "aabb":
            return bounds_overlap(
                world_aabb_for_pose(self.records[candidate.name].local_bounds, candidate.pose),
                world_aabb_for_pose(self.records[other.name].local_bounds, other.pose),
                eps=self.eps,
            )
        return objects_collide(candidate, other, eps=self.eps, mode=self.mode)

    def describe_failure(self, assignment: dict[NodeId, Pose]) -> dict[str, Any]:
        pair = self._last_pair or (self.node_id, "")
        return {
            "code": "collision_unsatisfied",
            "message": "objects overlap",
            "target": ",".join(name for name in pair if name),
            "factor": "collision",
            "mode": self.mode,
            "eps": self.eps,
        }


class RelativeDirectionFactor(HardFactor):
    priority = 80

    def __init__(
        self,
        relation: RelativeDirectionSpec,
        records: dict[str, LayoutAssetRecord],
        *,
        problem: LayoutProblem | None = None,
        surface_id: str | None = None,
    ) -> None:
        self.relation = relation
        self.records = records
        self.problem = problem
        self.surface_id = surface_id

    def check(self, assignment: dict[NodeId, Pose]) -> bool:
        a = assignment.get(self.relation.node_1)
        b = assignment.get(self.relation.node_2)
        if a is None or b is None:
            return True
        parsed = relation_axis_and_order(self.relation)
        if parsed is None:
            return True
        axis, before_name, after_name = parsed
        before = assignment[before_name]
        after = assignment[after_name]
        margin = max(0.0, float(self.relation.margin))
        surface = self._surface(assignment)
        if surface is not None:
            before_range = footprint_axis_span(
                record=self.records[before_name],
                pose=before,
                surface=surface,
                axis=axis,
            )
            after_range = footprint_axis_span(
                record=self.records[after_name],
                pose=after,
                surface=surface,
                axis=axis,
            )
            return before_range.max + margin <= after_range.min + GEOM_EPS

        axis_idx = 0 if axis == "x" else 1
        before_span = _axis_span(self.records[before_name], axis, before, surface=None)
        after_span = _axis_span(self.records[after_name], axis, after, surface=None)
        return (
            before.position[axis_idx] + before_span * 0.5 + margin
            <= after.position[axis_idx] - after_span * 0.5 + GEOM_EPS
        )

    def _surface(self, assignment: dict[NodeId, Pose]) -> ManualSupportSurface | None:
        if self.problem is None or not self.surface_id:
            return None
        return self.problem.resolve_surface(self.surface_id, assignment)

    def forward_check(
        self,
        node_id: NodeId,
        assignment: dict[NodeId, Pose],
        ir: CompiledLayoutIR,
        domain_mgr: DomainManager,
    ) -> bool | None:
        if node_id not in {self.relation.node_1, self.relation.node_2}:
            return None
        return self.check(assignment)

    def describe_failure(self, assignment: dict[NodeId, Pose]) -> dict[str, Any]:
        return {
            "code": "relative_direction_unsatisfied",
            "message": "relative direction constraint failed",
            "target": f"{self.relation.node_1}->{self.relation.node_2}",
            "factor": "relative_direction",
            "direction": self.relation.direction,
        }


class GroupSpacingFactor(HardFactor):
    priority = 90

    def __init__(
        self,
        *,
        group_id: str,
        members: list[str],
        min_spacing: float,
        spacing_mode: str = "footprint_gap",
        records: dict[str, LayoutAssetRecord] | None = None,
        problem: LayoutProblem | None = None,
        surface_id: str | None = None,
    ) -> None:
        self.group_id = str(group_id)
        self.members = list(members)
        self.min_spacing = max(0.0, float(min_spacing))
        mode = str(spacing_mode or "footprint_gap").strip().lower()
        self.spacing_mode = (
            "center_distance" if mode in {"center", "center_distance", "center-distance"} else "footprint_gap"
        )
        self.records = records or {}
        self.problem = problem
        self.surface_id = surface_id
        self._last_pair: tuple[str, str] | None = None

    def check(self, assignment: dict[NodeId, Pose]) -> bool:
        if self.min_spacing <= 0.0:
            return True
        placed = [name for name in self.members if name in assignment]
        surface = self._surface(assignment)
        if (
            self.spacing_mode == "footprint_gap"
            and surface is not None
            and all(name in self.records for name in placed)
        ):
            ranges = {
                name: footprint_ranges_on_surface(record=self.records[name], pose=assignment[name], surface=surface)
                for name in placed
            }
            for i, a in enumerate(placed):
                for b in placed[i + 1 :]:
                    if not footprint_ranges_separated(ranges[a], ranges[b], min_gap=self.min_spacing):
                        self._last_pair = (a, b)
                        return False
            self._last_pair = None
            return True

        for i, a in enumerate(placed):
            pa = assignment[a].position
            for b in placed[i + 1 :]:
                pb = assignment[b].position
                dx = pa[0] - pb[0]
                dy = pa[1] - pb[1]
                if (dx * dx + dy * dy) ** 0.5 < self.min_spacing - GEOM_EPS:
                    self._last_pair = (a, b)
                    return False
        self._last_pair = None
        return True

    def _surface(self, assignment: dict[NodeId, Pose]) -> ManualSupportSurface | None:
        if self.problem is None or not self.surface_id:
            return None
        return self.problem.resolve_surface(self.surface_id, assignment)

    def describe_failure(self, assignment: dict[NodeId, Pose]) -> dict[str, Any]:
        return {
            "code": "group_spacing_unsatisfied",
            "message": "group min_spacing failed",
            "target": self.group_id,
            "factor": "group_spacing",
            "pair": list(self._last_pair or ()),
            "min_spacing": self.min_spacing,
            "spacing_semantics": self.spacing_mode,
        }


class GroupOrderFactor(HardFactor):
    priority = 85

    def __init__(
        self,
        *,
        group_id: str,
        members: list[str],
        axis: str,
        records: dict[str, LayoutAssetRecord],
        surface: ManualSupportSurface | None = None,
        problem: LayoutProblem | None = None,
        surface_id: str | None = None,
        margin: float = 0.0,
    ) -> None:
        self.group_id = str(group_id)
        self.members = list(members)
        self.axis = str(axis)
        self.records = records
        self.surface = surface
        self.problem = problem
        self.surface_id = surface_id
        self.margin = max(0.0, float(margin))
        self._last_pair: tuple[str, str] | None = None

    def check(self, assignment: dict[NodeId, Pose]) -> bool:
        surface = self._surface(assignment)
        placed = [name for name in self.members if name in assignment]
        for left, right in zip(placed, placed[1:]):
            left_range = self._axis_range(left, assignment[left], surface)
            right_range = self._axis_range(right, assignment[right], surface)
            if left_range.max + self.margin > right_range.min + GEOM_EPS:
                self._last_pair = (left, right)
                return False
        self._last_pair = None
        return True

    def _surface(self, assignment: dict[NodeId, Pose]) -> ManualSupportSurface | None:
        if self.problem is not None and self.surface_id:
            return self.problem.resolve_surface(self.surface_id, assignment)
        return self.surface

    def _axis_range(self, name: str, pose: Pose, surface: ManualSupportSurface | None) -> Range1D:
        if surface is None:
            axis_idx = 0 if self.axis == "x" else 1
            span = _axis_span(self.records[name], self.axis, pose, surface=None)
            center = float(pose.position[axis_idx])
            return Range1D(center - span * 0.5, center + span * 0.5)
        return footprint_axis_span(record=self.records[name], pose=pose, surface=surface, axis=self.axis)

    def describe_failure(self, assignment: dict[NodeId, Pose]) -> dict[str, Any]:
        return {
            "code": "group_order_unsatisfied",
            "message": "group order_axis failed",
            "target": self.group_id,
            "factor": "group_order",
            "axis": self.axis,
            "pair": list(self._last_pair or ()),
            "margin": self.margin,
        }


def _axis_span(
    record: LayoutAssetRecord,
    axis: str,
    pose: Pose,
    *,
    surface: ManualSupportSurface | None,
) -> float:
    if surface is None:
        bounds = world_aabb_for_pose(record.local_bounds, pose)
        return bounds.x.length if axis == "x" else bounds.y.length
    return footprint_axis_span(record=record, pose=pose, surface=surface, axis=axis).length
