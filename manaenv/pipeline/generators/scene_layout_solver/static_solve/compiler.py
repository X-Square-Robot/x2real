from __future__ import annotations

from typing import Any

from ..assets.bridge import LayoutAssetRecord
from ..common.constants import GEOM_EPS
from ..common.diagnostics import LayoutDiagnostic
from ..common.ids import DomainId, GroupId, NodeId
from ..common.relations import relation_axis_and_order
from ..common.volumes import volume_parent
from ..scene_spec.schema import RelativeDirectionSpec
from ..variants.planner import LayoutCase
from .constraints.factors import (
    CollisionFactor,
    GroupOrderFactor,
    GroupSpacingFactor,
    RelativeDirectionFactor,
    normalize_ignore_pairs,
)
from .domains.placement_providers import (
    InVolumeDomainProvider,
    PoseSampleDomainProvider,
    SupportOnDomainProvider,
    TabletopGroupDomainProvider,
    VolumeGroupDomainProvider,
)
from .ir import CompiledLayoutIR, GroupSpec, NodeSpec
from .problem import LayoutProblem, ProblemObject
from .scheduler import schedule_execution_sequence
from .scopes.planner import SupportSurfaceScopePlanner


class MiniLayoutCompiler:
    def __init__(
        self,
        problem: LayoutProblem,
        records: dict[str, LayoutAssetRecord],
        *,
        case: LayoutCase,
        relation_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.problem = problem
        self.records = records
        self.case = case
        self.relation_overrides = dict(relation_overrides or {})
        self.unsupported: list[LayoutDiagnostic] = []
        self.grouped_objects: set[str] = set()
        self.object_domain_ids: dict[str, DomainId] = {}
        self.group_specs: dict[GroupId, GroupSpec] = {}
        self.node_specs: dict[NodeId, NodeSpec] = {}
        self.ordering_edges: list[tuple[NodeId, NodeId, str]] = []

    def compile(self) -> CompiledLayoutIR:
        for name, obj in sorted(self.problem.objects.items()):
            self.node_specs[name] = NodeSpec(
                node_id=name,
                asset=obj.asset,
                support=obj.support,
                local_bounds=obj.asset.local_bounds,
            )
        self._build_group_specs()
        self._build_node_providers()
        self._attach_factors()
        support_scope_plan = SupportSurfaceScopePlanner(self.problem).plan(
            node_specs=self.node_specs,
            group_specs=self.group_specs,
            grouped_objects=self.grouped_objects,
            object_domain_ids=self.object_domain_ids,
            group_surface=self._group_surface,
            local_surface_parent=self.problem.local_surface_parent,
        )
        sequence = schedule_execution_sequence(
            node_specs=self.node_specs,
            group_specs=self.group_specs,
            grouped_objects=self.grouped_objects,
            object_domain_ids=self.object_domain_ids,
            ordering_edges=self.ordering_edges,
            group_surface=self._group_surface,
            local_surface_parent=self.problem.local_surface_parent,
            node_dependency_nodes=self._node_dependency_nodes,
            group_dependency_nodes=self._group_dependency_nodes,
            support_scope_plan=support_scope_plan,
        )
        support_surface_components = {
            component.component_id: component
            for scope in support_scope_plan.scopes
            for component in scope.components
            if component.component_id in support_scope_plan.transactional_component_set
        }
        return CompiledLayoutIR(
            node_specs=self.node_specs,
            group_specs=self.group_specs,
            execution_sequence=sequence,
            ordering_edges=list(self.ordering_edges),
            support_surface_scopes={scope.scope_id: scope for scope in support_scope_plan.scopes},
            support_surface_components=support_surface_components,
            unsupported=self.unsupported,
            evidence={
                "case_id": self.case.case_id,
                "grouped_objects": sorted(self.grouped_objects),
                "relation_overrides": dict(self.relation_overrides),
                "support_surface_scope_plan": dict(support_scope_plan.evidence),
            },
        )

    def _build_group_specs(self) -> None:
        for region in self.problem.spec.regions:
            members = [name for name in region.objects if name in self.node_specs]
            group_id = f"region:{region.region_id}"
            self.grouped_objects.update(members)
            group_domain = DomainId.group(group_id)
            for name in members:
                self.object_domain_ids[name] = group_domain
            self.group_specs[group_id] = GroupSpec(
                group_id=group_id,
                members=members,
                domain_providers=[
                    TabletopGroupDomainProvider(
                        self.problem,
                        group_id=group_id,
                        members=members,
                        surface=region.surface,
                        relation=region,
                        relation_kind="region_sample",
                    )
                ],
            )
        for group in self.problem.spec.scatter_groups:
            members = [name for name in group.objects if name in self.node_specs]
            group_id = f"scatter:{group.group_id}"
            self.grouped_objects.update(members)
            group_domain = DomainId.group(group_id)
            for name in members:
                self.object_domain_ids[name] = group_domain
            self.group_specs[group_id] = GroupSpec(
                group_id=group_id,
                members=members,
                domain_providers=[
                    TabletopGroupDomainProvider(
                        self.problem,
                        group_id=group_id,
                        members=members,
                        surface=group.surface,
                        relation=group,
                        relation_kind="scatter_group",
                    )
                ],
            )
        for group in self.problem.spec.volume_groups:
            members = [name for name in group.objects if name in self.node_specs]
            group_id = f"volume:{group.group_id}"
            self.grouped_objects.update(members)
            group_domain = DomainId.group(group_id)
            for name in members:
                self.object_domain_ids[name] = group_domain
            self.group_specs[group_id] = GroupSpec(
                group_id=group_id,
                members=members,
                domain_providers=[
                    VolumeGroupDomainProvider(
                        self.problem,
                        group_id=group_id,
                        members=members,
                        relation=group,
                    )
                ],
            )

    def _build_node_providers(self) -> None:
        for name, node in sorted(self.node_specs.items()):
            if name in self.grouped_objects:
                continue
            pose_sample = self.problem.relation_index.pose_samples_by_object.get(name)
            if pose_sample is not None:
                node.domain_providers.append(PoseSampleDomainProvider(self.problem, pose_sample))
                self.object_domain_ids[name] = DomainId.node(name)
                continue
            in_volume = self.problem.relation_index.in_volumes_by_object.get(name)
            if in_volume is not None:
                node.domain_providers.append(InVolumeDomainProvider(self.problem, in_volume))
                self.object_domain_ids[name] = DomainId.node(name)
                continue
            if node.support:
                node.domain_providers.append(SupportOnDomainProvider(self.problem, name))
                self.object_domain_ids[name] = DomainId.node(name)

    def _attach_factors(self) -> None:
        collision_cfg = self.problem.spec.collision
        collision_enabled = bool(collision_cfg.get("enabled", True)) if isinstance(collision_cfg, dict) else True
        if collision_enabled:
            collision_mode = str(
                collision_cfg.get("mode", "multi_box") if isinstance(collision_cfg, dict) else "multi_box"
            )
            collision_eps = (
                collision_cfg.get("eps", collision_cfg.get("contact_tolerance", GEOM_EPS))
                if isinstance(collision_cfg, dict)
                else GEOM_EPS
            )
            ignore_pairs = (
                normalize_ignore_pairs(collision_cfg.get("ignore_pairs", []))
                if isinstance(collision_cfg, dict)
                else set()
            )
            ignore_pairs |= self._implicit_ignore_pairs()
            for node_id, node in self.node_specs.items():
                node.hard_factors.append(
                    CollisionFactor(
                        self.records,
                        node_id=node_id,
                        mode=collision_mode,
                        eps=float(collision_eps or 0.0),
                        ignore_pairs=ignore_pairs,
                    )
                )
        for group in self.group_specs.values():
            min_spacing = self._group_min_spacing(group.group_id)
            if min_spacing <= 0.0:
                min_spacing = 0.0
            group_surface = self._group_surface(group.group_id)
            spacing = GroupSpacingFactor(
                group_id=group.group_id,
                members=group.members,
                min_spacing=min_spacing,
                spacing_mode=self._group_spacing_mode(group.group_id),
                records=self.records,
                problem=self.problem,
                surface_id=group_surface,
            )
            if min_spacing > 0.0:
                for member in group.members:
                    self.node_specs[member].hard_factors.append(spacing)
            order = self._group_order_axis(group.group_id)
            if order in {"x", "y"}:
                surface = self.problem.surfaces.get(group_surface or "")
                factor = GroupOrderFactor(
                    group_id=group.group_id,
                    members=group.members,
                    axis=order,
                    records=self.records,
                    surface=surface,
                    problem=self.problem,
                    surface_id=group_surface,
                    margin=min_spacing,
                )
                for member in group.members:
                    self.node_specs[member].hard_factors.append(factor)
        for rel in self.problem.spec.relative_directions:
            self._attach_relative_factor(rel)

    def _group_min_spacing(self, group_id: str) -> float:
        if group_id.startswith("region:"):
            rid = group_id.split(":", 1)[1]
            for region in self.problem.spec.regions:
                if region.region_id == rid:
                    return max(0.0, float(region.min_spacing))
        if group_id.startswith("scatter:"):
            gid = group_id.split(":", 1)[1]
            for group in self.problem.spec.scatter_groups:
                if group.group_id == gid:
                    return max(0.0, float(group.min_spacing))
        if group_id.startswith("volume:"):
            gid = group_id.split(":", 1)[1]
            for group in self.problem.spec.volume_groups:
                if group.group_id == gid:
                    return max(0.0, float(group.min_spacing))
        return 0.0

    def _group_order_axis(self, group_id: str) -> str | None:
        if group_id.startswith("region:"):
            rid = group_id.split(":", 1)[1]
            for region in self.problem.spec.regions:
                if region.region_id == rid:
                    return region.order_axis
        return None

    def _group_spacing_mode(self, group_id: str) -> str:
        if group_id.startswith("region:"):
            rid = group_id.split(":", 1)[1]
            for region in self.problem.spec.regions:
                if region.region_id == rid:
                    return region.spacing_mode
        if group_id.startswith("scatter:"):
            gid = group_id.split(":", 1)[1]
            for group in self.problem.spec.scatter_groups:
                if group.group_id == gid:
                    return group.spacing_mode
        return "footprint_gap"

    def _group_surface(self, group_id: str) -> str | None:
        if group_id.startswith("region:"):
            rid = group_id.split(":", 1)[1]
            for region in self.problem.spec.regions:
                if region.region_id == rid:
                    return region.surface
        if group_id.startswith("scatter:"):
            gid = group_id.split(":", 1)[1]
            for group in self.problem.spec.scatter_groups:
                if group.group_id == gid:
                    return group.surface
        return None

    def _node_dependency_nodes(self, node_id: str) -> list[str]:
        rel = self.problem.relation_index.in_volumes_by_object.get(node_id)
        if rel is None:
            return []
        volume = self.problem.spec.volumes_by_id.get(rel.volume)
        parent = volume_parent(volume) if volume is not None else None
        return [parent] if parent else []

    def _group_dependency_nodes(self, group_id: str) -> list[str]:
        if not group_id.startswith("volume:"):
            return []
        gid = group_id.split(":", 1)[1]
        group = next((item for item in self.problem.spec.volume_groups if item.group_id == gid), None)
        volume = self.problem.spec.volumes_by_id.get(group.volume) if group is not None else None
        parent = volume_parent(volume) if volume is not None else None
        return [parent] if parent else []

    def _implicit_ignore_pairs(self) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for obj in self.problem.objects.values():
            parent = self.problem.local_surface_parent(obj.support) if obj.support else None
            if parent:
                pairs.add(tuple(sorted((obj.name, parent))))
        for rel in self.problem.spec.in_volumes:
            volume = self.problem.spec.volumes_by_id.get(rel.volume)
            parent = volume_parent(volume) if volume is not None else None
            if parent:
                pairs.add(tuple(sorted((rel.object_name, parent))))
        for group in self.problem.spec.volume_groups:
            volume = self.problem.spec.volumes_by_id.get(group.volume)
            parent = volume_parent(volume) if volume is not None else None
            members = [name for name in group.objects if name in self.node_specs]
            if parent and group.ignore_parent_collision:
                for name in members:
                    pairs.add(tuple(sorted((name, parent))))
            if group.ignore_intra_group_collision:
                for idx, left in enumerate(members):
                    for right in members[idx + 1 :]:
                        pairs.add(tuple(sorted((left, right))))
        return pairs

    def _attach_relative_factor(self, rel: RelativeDirectionSpec) -> None:
        left = self.problem.objects.get(rel.node_1)
        right = self.problem.objects.get(rel.node_2)
        if left is None or right is None:
            return
        left_surface = self._placement_surface(rel.node_1, left)
        right_surface = self._placement_surface(rel.node_2, right)
        if not left_surface or not right_surface:
            self.unsupported.append(
                LayoutDiagnostic(
                    "relative_direction_non_layout_scope",
                    "relative_direction currently requires both objects to have tabletop placement surfaces",
                    target=f"{rel.node_1}->{rel.node_2}",
                    meta={"left_surface": left_surface, "right_surface": right_surface},
                )
            )
            return
        if left_surface != right_surface:
            self.unsupported.append(
                LayoutDiagnostic(
                    "relative_direction_cross_surface_unsupported",
                    "cross-surface relative_direction is unsupported",
                    target=f"{rel.node_1}->{rel.node_2}",
                    meta={"left_surface": left_surface, "right_surface": right_surface},
                )
            )
            return
        factor = RelativeDirectionFactor(rel, self.records, problem=self.problem, surface_id=left_surface)
        self.node_specs[rel.node_1].hard_factors.append(factor)
        self.node_specs[rel.node_2].hard_factors.append(factor)
        parsed = relation_axis_and_order(rel)
        if parsed is not None:
            axis, pre, post = parsed
            self.ordering_edges.append((pre, post, axis))

    def _placement_surface(self, node_id: str, obj: ProblemObject) -> str | None:
        for region in self.problem.spec.regions:
            if node_id in region.objects:
                return region.surface
        for group in self.problem.spec.scatter_groups:
            if node_id in group.objects:
                return group.surface
        return obj.support
