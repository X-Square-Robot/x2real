from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from ..common.constants import RUNTIME_PARTITION_SEED_STRIDE
from ..common.geometry import Range1D
from ..common.reproducibility import stable_seed
from ..runtime_sampling import RuntimeSamplingPlan
from ..runtime_sampling.cell_domains import safe_id
from ..runtime_sampling.coupling import couple_same_surface_runtime_domains
from ..runtime_sampling.events import NormalizedRuntimeEvent
from ..runtime_sampling.native_compiler import TaskLayoutCompiler
from ..runtime_sampling.schema import AssignmentPlan, ConditionalAssignmentPlan, PlacementDomain
from ..scene_spec.schema import SurfaceRangeSpec
from ..variants.planner import LayoutCase, ObjectPlacementOverride
from .layout_plan import build_canonical_layout_plan
from .materialization import SceneMaterializationSample
from .parser import SelectorSpec, VNextSceneSpec, VNextTaskSpec
from .runtime_compile import RuntimePlacementPlanEntry, compile_runtime_sample

SCHEMA_VNEXT_RUNTIME_EVENT = "schema_vnext.runtime_sample"


def compile_schema_vnext_runtime_sampling_plan(
    task_payload: Mapping[str, Any],
    scene_payload: Mapping[str, Any],
    assets: Mapping[str, Any] | Iterable[Any],
    *,
    materialization: Mapping[str, Any] | SceneMaterializationSample | None = None,
    seed: int | None = None,
) -> RuntimeSamplingPlan | None:
    """Compile schema-vNext placement runtime into the full RuntimeSamplingPlan backend."""

    scene = VNextSceneSpec.from_dict(dict(scene_payload or {}))
    task = VNextTaskSpec.from_dict(dict(task_payload or {}), scene=scene)
    vnext_plan = compile_runtime_sample(task, scene)
    if not vnext_plan.placements:
        return None

    materialized = _materialization_sample(materialization)
    _require_scene_materialization_before_runtime_compile(scene, materialized)
    layout_plan = build_canonical_layout_plan(scene)
    layout_payload = _runtime_backend_layout_payload(scene, layout_plan=layout_plan, materialization=materialized)
    runtime_id = vnext_plan.runtime_sample_id or "runtime"
    compile_seed = int(seed if seed is not None else vnext_plan.seed if vnext_plan.seed is not None else 0)

    aggregate = RuntimeSamplingPlan()
    non_slot_entries: list[RuntimePlacementPlanEntry] = []
    non_slot_domains: list[dict[str, Any]] = []
    selector_annotations: dict[str, dict[str, dict[str, Any]]] = {}

    for entry in vnext_plan.placements:
        if entry.target_type == "group":
            group = layout_plan.groups.get(entry.target_id)
            if group is not None and group.group_type == "slot":
                slot_plan = _compile_slot_group_entry(
                    entry,
                    layout_plan=layout_plan,
                    layout_payload=layout_payload,
                    assets=assets,
                    runtime_id=runtime_id,
                    seed=compile_seed,
                )
                _merge_runtime_plan(aggregate, slot_plan)
                continue
            group_domain = _group_domain_payload(entry, scene=scene, layout_plan=layout_plan)
            non_slot_entries.append(entry)
            non_slot_domains.append(group_domain)
            continue
        domain_payload = _node_domain_payload(entry, scene=scene, layout_plan=layout_plan)
        non_slot_entries.append(entry)
        non_slot_domains.append(domain_payload)
        selectors = _domain_selector_payloads(entry)
        if selectors:
            selector_annotations[entry.entry_id] = selectors

    if non_slot_domains:
        node_layout = _layout_without_variants(layout_payload)
        for entry in non_slot_entries:
            _apply_entry_domain_overrides(node_layout, entry, scene=scene, layout_plan=layout_plan)
        node_plan = _compile_schema_vnext_domains(
            node_layout,
            assets,
            plan_id=runtime_id,
            seed=stable_seed(compile_seed, "schema_vnext_non_slot_runtime", runtime_id),
            domains=non_slot_domains,
        )
        _rename_group_case_pool_assignments(node_plan, non_slot_entries, runtime_id=runtime_id)
        _annotate_case_pool_selectors(node_plan, non_slot_entries)
        _annotate_domain_selectors(node_plan, selector_annotations)
        _merge_runtime_plan(aggregate, node_plan)

    aggregate.artifacts.setdefault("schema_vnext_runtime_sampling", {})
    aggregate.artifacts["schema_vnext_runtime_sampling"].update(
        {
            "runtime_sample_id": runtime_id,
            "backend": "RuntimeSamplingPlan",
            "placement_count": len(vnext_plan.placements),
            "source": "schema_vnext_runtime_sample",
        }
    )
    aggregate.finalize()
    return aggregate


def _require_scene_materialization_before_runtime_compile(
    scene: VNextSceneSpec,
    materialization: SceneMaterializationSample | None,
) -> None:
    required_variants = tuple(variant.variant_id for variant in scene.variants.materialization)
    if not required_variants:
        return
    if materialization is None:
        raise RuntimeError(
            "schema_vnext_runtime_materialization_required: scene materialization variants must be "
            "sampled before RuntimeSamplingPlan compile"
        )
    missing = sorted(
        variant_id for variant_id in required_variants if variant_id not in materialization.variant_choices
    )
    if missing:
        raise RuntimeError(
            "schema_vnext_runtime_materialization_incomplete: materialization sample is missing "
            f"variant choices {missing}"
        )


def _runtime_backend_layout_payload(
    scene: VNextSceneSpec,
    *,
    layout_plan: Any,
    materialization: SceneMaterializationSample | None,
) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "enabled": True,
        "seed": _scene_seed(scene),
        "objects": [],
    }
    background_poses = _runtime_backend_background_poses(scene, materialization)
    if background_poses:
        layout["background_poses"] = background_poses
    surfaces = _runtime_backend_surfaces(layout_plan)
    if surfaces:
        layout["surfaces"] = surfaces
    surface_catalog_refs = _runtime_backend_surface_catalog_refs(layout_plan)
    if surface_catalog_refs:
        layout["surface_catalog_refs"] = surface_catalog_refs
        if scene.background.surface_catalog:
            layout["surface_catalog_path"] = scene.background.surface_catalog
    volumes = _runtime_backend_volumes(layout_plan, scene=scene)
    if volumes:
        layout["volumes"] = volumes
    relative_directions = _runtime_backend_relative_directions(layout_plan)
    if relative_directions:
        layout["relative_directions"] = relative_directions

    grouped_nodes = {node for group in layout_plan.groups.values() for node in group.nodes}
    object_names: set[str] = set()
    in_volumes = []

    def append_object(payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        name = str(payload.get("name") or "").strip()
        if not name or name in object_names:
            return
        object_names.add(name)
        layout["objects"].append(payload)

    for node_id, node in layout_plan.nodes.items():
        if node_id in grouped_nodes:
            continue
        append_object(
            _runtime_backend_node_object(
                node,
                scene=scene,
                layout_plan=layout_plan,
                materialization=materialization,
            )
        )
        node_volume = _runtime_backend_node_volume_domain(node, layout_plan=layout_plan)
        if node_volume:
            in_volumes.append(node_volume)

    regions = []
    scatter_groups = []
    volume_groups = []
    for group in layout_plan.groups.values():
        if group.group_type == "slot":
            for payload in _runtime_backend_slot_member_objects(
                group,
                scene=scene,
                layout_plan=layout_plan,
                materialization=materialization,
            ):
                append_object(payload)
            continue
        for payload in _runtime_backend_group_member_objects(
            group,
            scene=scene,
            layout_plan=layout_plan,
            materialization=materialization,
        ):
            append_object(payload)
        group_payload = _runtime_backend_group_domain(group, scene=scene, layout_plan=layout_plan)
        if not group_payload:
            continue
        if group.group_type == "region":
            regions.append(group_payload)
        elif group.group_type == "scatter":
            scatter_groups.append(group_payload)
        elif group.group_type == "volume":
            if _volume_group_outputs_as_in_volume(group, scene=scene):
                in_volumes.append(group_payload)
            else:
                volume_groups.append(group_payload)
    if regions:
        layout["regions"] = regions
    if scatter_groups:
        layout["scatter_groups"] = scatter_groups
    if in_volumes:
        layout["in_volumes"] = in_volumes
    if volume_groups:
        layout["volume_groups"] = volume_groups

    pose_samples = [
        _runtime_backend_pose_sample(node, object_id=scene.layout.nodes[node_id].object_id)
        for node_id, node in layout_plan.nodes.items()
        if node.placement_type == "pose"
    ]
    pose_samples = [item for item in pose_samples if item]
    if pose_samples:
        layout["pose_samples"] = pose_samples
    if not layout.get("objects") and not pose_samples and not volumes:
        layout["enabled"] = False
    return layout


def _runtime_backend_background_poses(
    scene: VNextSceneSpec,
    materialization: SceneMaterializationSample | None,
) -> dict[str, dict[str, Any]]:
    raw_background = scene.source.get("background") if isinstance(scene.source, Mapping) else {}
    asset_id = materialization.background_asset if materialization is not None else ""
    if not asset_id:
        asset = (
            raw_background.get("asset")
            if isinstance(raw_background, Mapping) and isinstance(raw_background.get("asset"), Mapping)
            else {}
        )
        asset_id = str(asset.get("id") or "").strip()
    if not asset_id:
        return {}
    if asset_id in scene.background.pose_by_asset:
        position, orientation = scene.background.pose_by_asset[asset_id]
    else:
        pose = (
            raw_background.get("pose")
            if isinstance(raw_background, Mapping) and isinstance(raw_background.get("pose"), Mapping)
            else {}
        )
        position = list(pose.get("position", scene.background.pose_position))
        orientation = list(pose.get("orientation", scene.background.pose_orientation))
    return {
        str(asset_id): {
            "position": list(position),
            "orientation": list(orientation),
        }
    }


def _runtime_backend_surfaces(layout_plan: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for surface_id, surface in layout_plan.surfaces.items():
        if surface.alias_type != "manual":
            continue
        payload = deepcopy(dict(surface.geometry))
        payload.setdefault("id", surface_id)
        out.append(payload)
    return out


def _runtime_backend_surface_catalog_refs(layout_plan: Any) -> list[str]:
    refs: list[str] = []
    for surface in layout_plan.surfaces.values():
        if surface.alias_type != "ref":
            continue
        ref = str(surface.ref or "").strip()
        if ref == "background.tabletop":
            refs.append("tabletop")
    return refs


def _runtime_backend_volumes(layout_plan: Any, *, scene: VNextSceneSpec) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for volume_id, volume in layout_plan.volumes.items():
        if volume.alias_type == "manual":
            payload = deepcopy(dict(volume.geometry))
            payload.setdefault("id", volume_id)
            out.append(payload)
            continue
        if volume.alias_type != "ref":
            continue
        payload = _object_local_volume_payload(str(volume.ref or ""), scene=scene)
        if payload:
            out.append(payload)
    return out


def _runtime_backend_relative_directions(layout_plan: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    nodes = layout_plan.nodes if hasattr(layout_plan, "nodes") else {}
    for relation in getattr(layout_plan, "relations", ()) or ():
        if getattr(relation, "relation_type", "") != "relative_direction":
            continue
        node_1 = nodes.get(str(relation.a))
        node_2 = nodes.get(str(relation.b))
        if node_1 is None or node_2 is None:
            continue
        payload = {
            "node_1": str(node_1.object_id),
            "node_2": str(node_2.object_id),
            "direction": str(relation.direction),
        }
        params = getattr(relation, "params", {}) or {}
        if isinstance(params, Mapping):
            for key, value in params.items():
                payload[str(key)] = deepcopy(value)
        out.append(payload)
    return out


def _runtime_backend_node_object(
    node: Any,
    *,
    scene: VNextSceneSpec,
    layout_plan: Any,
    materialization: SceneMaterializationSample | None = None,
) -> dict[str, Any] | None:
    object_id = node.object_id
    if node.placement_type == "surface":
        payload = {"name": object_id, "support": _runtime_backend_surface_id(layout_plan, node.surface or "")}
        if node.base_orientation is not None:
            payload["orientation"] = list(node.base_orientation)
        if node.ranges:
            payload["surface_ranges"] = [_range_choice_payload(choice) for choice in node.ranges]
        if node.yaws:
            _assign_yaw_payload(payload, node.yaws)
        _apply_layout_object_extras(payload, scene=scene, object_id=object_id, materialization=materialization)
        return payload
    if node.placement_type == "pose":
        payload = {"name": object_id}
        _apply_layout_object_extras(payload, scene=scene, object_id=object_id, materialization=materialization)
        return payload
    if node.placement_type == "volume":
        payload = {"name": object_id}
        if node.yaws:
            _assign_yaw_payload(payload, node.yaws)
        _apply_layout_object_extras(payload, scene=scene, object_id=object_id, materialization=materialization)
        return payload
    return None


def _runtime_backend_node_volume_domain(node: Any, *, layout_plan: Any) -> dict[str, Any] | None:
    if node.placement_type != "volume":
        return None
    return {
        "object": node.object_id,
        "volume": _runtime_backend_volume_id(layout_plan, node.volume or ""),
        "mode": str((node.params or {}).get("mode") or "fit").strip().lower(),
    }


def _runtime_backend_pose_sample(node: Any, *, object_id: str) -> dict[str, Any] | None:
    if node.placement_type != "pose" or not node.poses:
        return None
    return {
        "object": object_id,
        "poses": [
            {
                "id": choice.choice_id,
                "position": list(choice.position),
                "quaternion": list(choice.orientation),
                "weight": choice.weight,
            }
            for choice in node.poses
        ],
    }


def _runtime_backend_group_domain(group: Any, *, scene: VNextSceneSpec, layout_plan: Any) -> dict[str, Any] | None:
    if group.group_type not in {"region", "scatter", "volume"}:
        return None
    if _volume_group_outputs_as_in_volume(group, scene=scene):
        node_id = group.nodes[0]
        return {
            "object": scene.layout.nodes[node_id].object_id,
            "volume": _runtime_backend_volume_id(layout_plan, group.volume or ""),
            "mode": _volume_group_mode(scene, group.group_id) or "fit",
        }
    payload = {
        "id": group.group_id,
        "objects": [scene.layout.nodes[node_id].object_id for node_id in group.nodes],
    }
    if group.surface:
        payload["surface"] = _runtime_backend_surface_id(layout_plan, group.surface)
    if group.volume:
        payload["volume"] = _runtime_backend_volume_id(layout_plan, group.volume)
    if group.ranges:
        payload["surface_ranges"] = [_range_choice_payload(choice) for choice in group.ranges]
    if group.yaws:
        _assign_yaw_payload(payload, group.yaws)
    raw_group = _raw_group_payload(scene, group.group_id)
    for key in (
        "method",
        "sample_method",
        "mode",
        "grid_step",
        "grid_snap",
        "min_spacing",
        "spacing_mode",
        "jitter",
        "align_axis",
        "order_axis",
        "max_attempts",
        "ignore_parent_collision",
        "ignore_intra_group_collision",
    ):
        if key in raw_group:
            payload[key] = deepcopy(raw_group[key])
    _normalize_group_sampling_method(payload, group.group_type)
    return payload


def _runtime_backend_group_member_objects(
    group: Any,
    *,
    scene: VNextSceneSpec,
    layout_plan: Any,
    materialization: SceneMaterializationSample | None = None,
) -> list[dict[str, Any]]:
    if group.group_type not in {"region", "scatter", "volume"}:
        return []
    support = _runtime_backend_surface_id(layout_plan, group.surface or "") if group.surface else ""
    out: list[dict[str, Any]] = []
    for node_id in group.nodes:
        object_id = scene.layout.nodes[node_id].object_id
        payload = {"name": object_id}
        if support:
            payload["support"] = support
        if group.base_orientation is not None:
            payload["orientation"] = list(group.base_orientation)
        if group.yaws:
            _assign_yaw_payload(payload, group.yaws)
        _apply_layout_object_extras(payload, scene=scene, object_id=object_id, materialization=materialization)
        out.append(payload)
    return out


def _runtime_backend_slot_member_objects(
    group: Any,
    *,
    scene: VNextSceneSpec,
    layout_plan: Any,
    materialization: SceneMaterializationSample | None = None,
) -> list[dict[str, Any]]:
    default_surface = ""
    if group.slots:
        first_slot = next(iter(group.slots.values()))
        default_surface = _runtime_backend_surface_id(layout_plan, first_slot.surface)
    out: list[dict[str, Any]] = []
    for node_id in group.nodes:
        object_id = scene.layout.nodes[node_id].object_id
        payload = {"name": object_id}
        if default_surface:
            payload["support"] = default_surface
        _apply_layout_object_extras(payload, scene=scene, object_id=object_id, materialization=materialization)
        out.append(payload)
    return out


def _compile_slot_group_entry(
    entry: RuntimePlacementPlanEntry,
    *,
    layout_plan: Any,
    layout_payload: Mapping[str, Any],
    assets: Mapping[str, Any] | Iterable[Any],
    runtime_id: str,
    seed: int,
) -> RuntimeSamplingPlan:
    group = layout_plan.groups[entry.target_id]
    plan_seed = stable_seed(seed, "schema_vnext_slot_runtime", entry.entry_id)
    compiler = _prepared_schema_vnext_compiler(layout_payload, assets, seed=plan_seed)
    event = NormalizedRuntimeEvent(name=SCHEMA_VNEXT_RUNTIME_EVENT, params={"plan_id": runtime_id, "seed": plan_seed})
    assignment_id = f"{runtime_id}:{safe_id(entry.entry_id)}"
    object_names = [str(entry.domain.node_objects[node_id]) for node_id in group.nodes]
    domains: list[PlacementDomain] = []
    cases: list[dict[str, Any]] = []

    for case_index, assignment_case_id in enumerate(entry.domain.assignments):
        assignment_case = entry.domain.assignment_cases[assignment_case_id]
        bind: dict[str, str] = {}
        slot_ids: dict[str, str] = {}
        for node_id, slot_id in assignment_case.bindings.items():
            object_name = str(entry.domain.node_objects[str(node_id)])
            slot = group.slots.get(str(slot_id))
            if slot is None:
                raise RuntimeError(
                    f"schema_vnext_slot_assignment_slot_unknown: {entry.entry_id}:{assignment_case_id}:{slot_id}"
                )
            layout_case = LayoutCase(
                case_id=str(assignment_case.assignment_id),
                params={
                    **dict(assignment_case.params),
                    "source_case_id": str(assignment_case.assignment_id),
                    "weight": _assignment_case_weight(assignment_case),
                },
            )
            slot_yaw = _slot_fixed_yaw(slot)
            override = ObjectPlacementOverride(
                surface=_runtime_backend_surface_id(layout_plan, slot.surface),
                surface_range=_surface_range_spec_for_slot(slot),
                slot_id=slot.slot_id,
                yaw=slot_yaw,
            )
            domain = compiler._variant_choice_domain(
                object_name,
                override,
                domain_id=":".join(
                    safe_id(part)
                    for part in (
                        runtime_id,
                        entry.entry_id,
                        assignment_case.assignment_id,
                        object_name,
                    )
                ),
                target=f"{SCHEMA_VNEXT_RUNTIME_EVENT}.slot_assignment.{entry.entry_id}",
                event=event,
                case=layout_case,
            )
            if domain is None:
                _raise_if_plan_failed(compiler.plan)
                raise RuntimeError("schema_vnext_runtime_sampling_plan_compile_failed: slot domain did not compile")
            if slot_yaw is None:
                domain = _domain_with_slot_yaws(domain, slot)
            domains.append(domain)
            bind[object_name] = domain.domain_id
            slot_ids[object_name] = slot.slot_id
        cases.append(
            {
                "id": str(assignment_case.assignment_id),
                "source_case_id": str(assignment_case.assignment_id),
                "weight": _assignment_case_weight(assignment_case),
                "bind": bind,
                "slot_ids": slot_ids,
                "slot_order": list(group.slots),
                "object_order": list(object_names),
                "params": dict(assignment_case.params),
                "case_index": case_index,
            }
        )

    if not cases:
        raise RuntimeError(f"schema_vnext_slot_assignment_cases_empty: {entry.entry_id}")
    selector = entry.selectors.get("assignments")
    assignment_meta = {
        "authoring_event": SCHEMA_VNEXT_RUNTIME_EVENT,
        "source": "schema_vnext.layout.groups.slot.assignments",
        "variant_id": entry.entry_id,
        "slot_group": group.group_id,
        "case_count": len(cases),
        "object_count": len(object_names),
        "weight_scope": "slot_assignment_case",
        "weight_total": sum(float(case.get("weight", 1.0)) for case in cases),
    }
    if selector is not None:
        assignment_meta["selector"] = _selector_payload(selector)
    compiler.plan.domains.extend(domains)
    compiler.plan.assignments.append(
        AssignmentPlan(
            plan_id=assignment_id,
            mode="case_pool",
            objects=tuple(object_names),
            domains=tuple(domain.domain_id for domain in domains),
            bind=dict(cases[0].get("bind") or {}),
            cases=tuple(cases),
            source_event=event.name,
            meta=assignment_meta,
        )
    )
    compiler.plan.artifacts.setdefault("schema_vnext_slot_assignment_pools", []).append(
        {
            "id": entry.entry_id,
            "assignment": assignment_id,
            "slot_group": group.group_id,
            "objects": list(object_names),
            "cases": [
                {
                    "id": str(case["id"]),
                    "source_case_id": str(case["source_case_id"]),
                    "weight": float(case.get("weight", 1.0)),
                    "slot_ids": dict(case.get("slot_ids") or {}),
                }
                for case in cases
            ],
        }
    )
    compiler.plan.finalize()
    _raise_if_plan_failed(compiler.plan)
    return compiler.plan


def _compile_schema_vnext_domains(
    layout_payload: Mapping[str, Any],
    assets: Mapping[str, Any] | Iterable[Any],
    *,
    plan_id: str,
    seed: int,
    domains: list[Mapping[str, Any]],
) -> RuntimeSamplingPlan:
    compiler = _prepared_schema_vnext_compiler(layout_payload, assets, seed=seed)
    event = NormalizedRuntimeEvent(name=SCHEMA_VNEXT_RUNTIME_EVENT, params={"plan_id": plan_id, "seed": seed})

    compiled_domains: list[PlacementDomain] = []
    fixed_by_object: dict[str, PlacementDomain] = {}
    parent_domain_by_object: dict[str, PlacementDomain] = {}
    child_domains_by_parent: dict[str, dict[str, str]] = {}
    conditional_cases_by_parent: dict[str, list[dict[str, Any]]] = {}
    conditional_meta_by_parent: dict[str, dict[str, Any]] = {}

    for index, raw_domain in enumerate(domains):
        compiled = compiler._compile_sampling_domain(dict(raw_domain), event=event, index=index)
        if compiled is None:
            _raise_if_plan_failed(compiler.plan)
            raise RuntimeError("schema_vnext_runtime_sampling_plan_compile_failed: domain did not compile")
        domain_items = list(compiled.domains)
        parent = compiled.parent
        compiled_domains.extend(domain_items)
        if parent:
            child_domains = child_domains_by_parent.setdefault(parent, {})
            conditional_meta_by_parent.setdefault(parent, {}).update(dict(compiled.meta or {}))
            if compiled.cases:
                conditional_cases_by_parent.setdefault(parent, []).extend(dict(case) for case in compiled.cases)
                first_bind = dict(compiled.cases[0].get("bind") or {})
                for child, domain_id in first_bind.items():
                    child_domains[str(child)] = str(domain_id)
            else:
                for domain in domain_items:
                    child_domains[domain.object_name] = domain.domain_id
        else:
            if not (compiled.meta or {}).get("assignment_emitted"):
                for domain in domain_items:
                    fixed_by_object[domain.object_name] = domain
                    parent_domain_by_object[domain.object_name] = domain
            parent_domains = (compiled.meta or {}).get("parent_domains_by_object")
            if isinstance(parent_domains, Mapping):
                by_domain_id = {domain.domain_id: domain for domain in domain_items}
                for name, domain_id in parent_domains.items():
                    domain = by_domain_id.get(str(domain_id))
                    if domain is not None:
                        parent_domain_by_object[str(name)] = domain

    compiler.plan.domains.extend(compiled_domains)
    conditional_parents = set(child_domains_by_parent)
    fixed_bind = {name: domain.domain_id for name, domain in fixed_by_object.items() if name not in conditional_parents}
    if fixed_bind:
        compiler.plan.assignments.append(
            AssignmentPlan(
                plan_id=plan_id if not child_domains_by_parent else f"{plan_id}:fixed",
                mode="fixed",
                objects=tuple(fixed_bind),
                domains=tuple(fixed_bind.values()),
                bind=fixed_bind,
                source_event=event.name,
                meta={"authoring_event": SCHEMA_VNEXT_RUNTIME_EVENT},
            )
        )

    for parent, child_domains in child_domains_by_parent.items():
        parent_domain = parent_domain_by_object.get(parent)
        if parent_domain is None:
            compiler.plan.add_diagnostic(
                "schema_vnext_runtime_parent_domain_missing",
                "object-relative runtime sampling requires a parent domain in the same sampling plan",
                target=event.name,
                parent=parent,
                children=sorted(child_domains),
            )
            _raise_if_plan_failed(compiler.plan)
        compiler.plan.conditional_assignments.append(
            ConditionalAssignmentPlan(
                plan_id=plan_id if len(child_domains_by_parent) == 1 and not fixed_bind else f"{plan_id}:{parent}",
                parent=parent,
                children=tuple(child_domains),
                parent_domain=parent_domain.domain_id,
                child_domains=child_domains,
                dependencies={child: (parent,) for child in child_domains},
                source_event=event.name,
                cases=tuple(conditional_cases_by_parent.get(parent, [])),
                meta={
                    "authoring_event": SCHEMA_VNEXT_RUNTIME_EVENT,
                    **conditional_meta_by_parent.get(parent, {}),
                },
            )
        )

    couple_same_surface_runtime_domains(
        compiler.plan,
        seed=seed + len(plan_id) * RUNTIME_PARTITION_SEED_STRIDE,
    )
    _rewrite_plan_authoring_event(compiler.plan)
    compiler.plan.finalize()
    _raise_if_plan_failed(compiler.plan)
    return compiler.plan


def _prepared_schema_vnext_compiler(
    layout_payload: Mapping[str, Any],
    assets: Mapping[str, Any] | Iterable[Any],
    *,
    seed: int,
) -> TaskLayoutCompiler:
    compiler = TaskLayoutCompiler(layout_payload, assets, event_terms={}, seed=seed)
    compiler.compile()
    _raise_if_plan_failed(compiler.plan)
    compiler.plan.event_patches.clear()
    compiler.plan.artifacts["schema_vnext_compile_backend"] = {
        "source_event": SCHEMA_VNEXT_RUNTIME_EVENT,
        "uses_authoring_event_compatibility": False,
        "uses_scene_materialization_bridge": False,
        "source_layout": "schema_vnext_canonical_layout_plan",
    }
    return compiler


def _raise_if_plan_failed(plan: RuntimeSamplingPlan) -> None:
    if not plan.success:
        details = [diagnostic.to_json() for diagnostic in plan.diagnostics]
        raise RuntimeError(f"schema_vnext_runtime_sampling_plan_compile_failed: {details}")


def _rewrite_plan_authoring_event(plan: RuntimeSamplingPlan) -> None:
    for index, assignment in enumerate(plan.assignments):
        meta = dict(assignment.meta)
        if meta.get("authoring_event"):
            meta["authoring_event"] = SCHEMA_VNEXT_RUNTIME_EVENT
        plan.assignments[index] = replace(assignment, source_event=SCHEMA_VNEXT_RUNTIME_EVENT, meta=meta)
    for index, conditional in enumerate(plan.conditional_assignments):
        meta = dict(conditional.meta)
        if meta.get("authoring_event"):
            meta["authoring_event"] = SCHEMA_VNEXT_RUNTIME_EVENT
        plan.conditional_assignments[index] = replace(
            conditional,
            source_event=SCHEMA_VNEXT_RUNTIME_EVENT,
            meta=meta,
        )


def _node_domain_payload(
    entry: RuntimePlacementPlanEntry, *, scene: VNextSceneSpec, layout_plan: Any
) -> dict[str, Any]:
    node = scene.layout.nodes.get(entry.target_id)
    if node is None:
        raise RuntimeError(f"schema_vnext_runtime_target_unknown: {entry.entry_id}:{entry.target_id}")
    if entry.domain.domain_kind == "pose":
        return {
            "id": entry.entry_id,
            "node": node.object_id,
            "frame": {"type": "pose-sample"},
            "sample": {"type": "inherit"},
        }
    if entry.domain.domain_kind == "volume":
        layout_node = layout_plan.nodes.get(entry.target_id)
        if layout_node is None:
            raise RuntimeError(f"schema_vnext_runtime_target_unknown: {entry.entry_id}:{entry.target_id}")
        parent, volume_name = _runtime_backend_volume_parent_and_name(layout_plan, layout_node.volume or "")
        payload = {
            "id": entry.entry_id,
            "node": node.object_id,
            "frame": {"type": "object-volume", "node": parent, "volume": volume_name},
            "sample": _sample_payload(entry),
        }
        payload["sample"]["mode"] = _entry_node_volume_mode(entry, layout_node) or "fit"
        return payload
    if entry.domain.domain_kind != "surface":
        raise RuntimeError(f"schema_vnext_runtime_node_domain_unsupported: {entry.entry_id}:{entry.domain.domain_kind}")
    layout_node = layout_plan.nodes.get(entry.target_id)
    if layout_node is None:
        raise RuntimeError(f"schema_vnext_runtime_target_unknown: {entry.entry_id}:{entry.target_id}")
    local_surface = _object_local_surface_ref(layout_plan, layout_node.surface or "")
    frame = {"type": "scene-surface"}
    if local_surface is not None:
        parent, surface_name = local_surface
        frame = {"type": "node-surface", "node": parent, "surface": surface_name}
    return {
        "id": entry.entry_id,
        "node": node.object_id,
        "frame": frame,
        "sample": _sample_payload(entry),
    }


def _group_domain_payload(
    entry: RuntimePlacementPlanEntry,
    *,
    scene: VNextSceneSpec,
    layout_plan: Any,
) -> dict[str, Any]:
    group = layout_plan.groups.get(entry.target_id)
    if group is None:
        raise RuntimeError(f"schema_vnext_runtime_group_unknown: {entry.entry_id}:{entry.target_id}")
    payload: dict[str, Any] = {
        "id": entry.entry_id,
        "frame": {"type": "scene-surface"},
        "sample": _sample_payload(entry),
    }
    if group.group_type == "region":
        payload["region"] = group.group_id
    elif group.group_type == "scatter":
        payload["scatter_group"] = group.group_id
    elif group.group_type == "volume":
        if _volume_group_outputs_as_in_volume(group, scene=scene):
            if len(group.nodes) != 1:
                raise RuntimeError(f"schema_vnext_runtime_volume_fit_group_invalid: {entry.entry_id}")
            object_id = scene.layout.nodes[group.nodes[0]].object_id
            parent, volume_name = _runtime_backend_volume_parent_and_name(layout_plan, group.volume or "")
            payload["node"] = object_id
            payload["frame"] = {"type": "object-volume", "node": parent, "volume": volume_name}
            payload["sample"]["mode"] = _entry_volume_mode(entry, scene=scene, group_id=group.group_id) or "fit"
        else:
            payload["volume_group"] = group.group_id
            payload["frame"] = {"type": "object-volume"}
    else:
        raise RuntimeError(f"schema_vnext_runtime_group_domain_unsupported: {entry.entry_id}:{group.group_type}")
    parent_surface = _object_local_surface_ref(layout_plan, group.surface or "")
    if parent_surface is not None:
        parent, surface_name = parent_surface
        payload["frame"] = {"type": "node-surface", "node": parent, "surface": surface_name}
    return payload


def _sample_payload(entry: RuntimePlacementPlanEntry) -> dict[str, Any]:
    if entry.sample_type == "inherit":
        return {"type": "inherit"}
    if entry.sample_type == "override":
        return {"type": "inherit"}
    if entry.sample_type == "jitter":
        sample = {
            "type": "explicit",
            "geometry": "rectangle",
            "relative_to": "current",
        }
        if "dx" in entry.jitter:
            sample["dx"] = list(entry.jitter["dx"])
        if "dy" in entry.jitter:
            sample["dy"] = list(entry.jitter["dy"])
        if "yaw_delta" in entry.jitter:
            sample["yaw_delta"] = list(entry.jitter["yaw_delta"])
        return sample
    raise RuntimeError(f"schema_vnext_runtime_sample_type_unsupported: {entry.entry_id}:{entry.sample_type}")


def _apply_entry_domain_overrides(
    layout_payload: dict[str, Any],
    entry: RuntimePlacementPlanEntry,
    *,
    scene: VNextSceneSpec,
    layout_plan: Any,
) -> None:
    if entry.sample_type != "override":
        return
    if entry.target_type == "node":
        node = scene.layout.nodes.get(entry.target_id)
        if node is None:
            return
        if entry.domain.domain_kind == "surface":
            obj = _layout_object(layout_payload, node.object_id)
            if entry.domain.ranges:
                obj["surface_ranges"] = [_range_choice_payload(choice) for choice in entry.domain.ranges]
            if entry.domain.yaws:
                _assign_yaw_payload(obj, entry.domain.yaws)
        elif entry.domain.domain_kind == "pose" and entry.domain.poses:
            _replace_pose_sample(layout_payload, node.object_id, entry.domain.poses)
        elif entry.domain.domain_kind == "volume":
            in_volume = _layout_in_volume(layout_payload, node.object_id)
            mode = str(entry.domain.params.get("mode") or "").strip().lower()
            if mode:
                in_volume["mode"] = mode
            if entry.domain.yaws:
                obj = _layout_object(layout_payload, node.object_id)
                _assign_yaw_payload(obj, entry.domain.yaws)
        return
    if entry.target_type != "group":
        return
    group = layout_plan.groups.get(entry.target_id)
    if group is None or group.group_type not in {"region", "scatter", "volume"}:
        return
    if group.group_type == "region":
        group_payload = _layout_group(layout_payload, "regions", group.group_id)
    elif group.group_type == "scatter":
        group_payload = _layout_group(layout_payload, "scatter_groups", group.group_id)
    elif _volume_group_outputs_as_in_volume(group, scene=scene):
        object_id = scene.layout.nodes[group.nodes[0]].object_id
        group_payload = _layout_in_volume(layout_payload, object_id)
    else:
        group_payload = _layout_group(layout_payload, "volume_groups", group.group_id)
    if entry.domain.ranges and group.group_type in {"region", "scatter"}:
        group_payload["surface_ranges"] = [_range_choice_payload(choice) for choice in entry.domain.ranges]
    if entry.domain.yaws:
        _assign_yaw_payload(group_payload, entry.domain.yaws)
    for key, value in dict(entry.domain.params).items():
        group_payload[key] = deepcopy(value)
    _normalize_group_sampling_method(group_payload, group.group_type, replace_existing=True)


def _layout_object(layout_payload: dict[str, Any], object_id: str) -> dict[str, Any]:
    for item in layout_payload.get("objects", []) or []:
        if isinstance(item, dict) and str(item.get("name") or "") == str(object_id):
            return item
    item = {"name": str(object_id)}
    layout_payload.setdefault("objects", []).append(item)
    return item


def _layout_group(layout_payload: dict[str, Any], section_name: str, group_id: str) -> dict[str, Any]:
    for item in layout_payload.get(section_name, []) or []:
        if isinstance(item, dict) and str(item.get("id") or "") == str(group_id):
            return item
    item = {"id": str(group_id)}
    layout_payload.setdefault(section_name, []).append(item)
    return item


def _layout_in_volume(layout_payload: dict[str, Any], object_id: str) -> dict[str, Any]:
    for item in layout_payload.get("in_volumes", []) or []:
        if isinstance(item, dict) and str(item.get("object") or "") == str(object_id):
            return item
    item = {"object": str(object_id)}
    layout_payload.setdefault("in_volumes", []).append(item)
    return item


def _replace_pose_sample(layout_payload: dict[str, Any], object_id: str, poses: Any) -> None:
    pose_payload = {
        "object": str(object_id),
        "poses": [
            {
                "id": choice.choice_id,
                "position": list(choice.position),
                "quaternion": list(choice.orientation),
                "weight": float(choice.weight),
            }
            for choice in poses
        ],
    }
    pose_samples = layout_payload.setdefault("pose_samples", [])
    for index, item in enumerate(list(pose_samples)):
        if isinstance(item, dict) and str(item.get("object") or "") == str(object_id):
            pose_samples[index] = pose_payload
            return
    pose_samples.append(pose_payload)


def _range_choice_payload(choice: Any) -> dict[str, Any]:
    return {
        "id": choice.choice_id,
        "weight": float(choice.weight),
        "surface_range": {"x": list(choice.x), "y": list(choice.y)},
    }


def _surface_range_spec_for_slot(slot: Any) -> SurfaceRangeSpec:
    ranges = list(slot.ranges or [])
    if len(ranges) != 1:
        raise RuntimeError(
            f"schema_vnext_slot_runtime_range_count_unsupported: slot runtime assignment requires exactly one range ({slot.slot_id})"
        )
    choice = ranges[0]
    return SurfaceRangeSpec(x=Range1D.from_pair(choice.x), y=Range1D.from_pair(choice.y))


def _assignment_case_weight(assignment_case: Any) -> float:
    params = assignment_case.params if isinstance(assignment_case.params, Mapping) else {}
    try:
        return max(0.0, float(params.get("weight", 1.0)))
    except (TypeError, ValueError):
        return 0.0


def _domain_with_slot_yaws(domain: PlacementDomain, slot: Any) -> PlacementDomain:
    yaws = list(slot.yaws or [])
    if not yaws:
        return domain
    meta = dict(domain.meta)
    meta["slot_yaw_source"] = slot.slot_id
    if len(yaws) == 1:
        return replace(domain, yaw=tuple(float(value) for value in yaws[0].yaw), yaw_choices=(), meta=meta)
    yaw_choices = []
    for choice in yaws:
        yaw = tuple(choice.yaw)
        if len(yaw) != 2 or abs(float(yaw[0]) - float(yaw[1])) > 1e-9:
            raise RuntimeError(
                f"schema_vnext_slot_runtime_yaw_choice_range_unsupported: multiple slot yaws must be fixed values ({choice.choice_id})"
            )
        yaw_choices.append({"id": choice.choice_id, "weight": float(choice.weight), "yaw": float(yaw[0])})
    return replace(domain, yaw=None, yaw_choices=tuple(yaw_choices), meta=meta)


def _slot_fixed_yaw(slot: Any) -> tuple[float, float] | None:
    yaws = list(getattr(slot, "yaws", ()) or ())
    if len(yaws) != 1:
        return None
    yaw = tuple(yaws[0].yaw)
    if len(yaw) != 2 or abs(float(yaw[0]) - float(yaw[1])) > 1e-9:
        return None
    value = float(yaw[0])
    return (value, value)


def _assign_yaw_payload(payload: dict[str, Any], yaws: Any) -> None:
    values = list(yaws or [])
    payload.pop("yaw", None)
    payload.pop("yaw_samples", None)
    if not values:
        return
    if len(values) == 1:
        payload["yaw"] = list(values[0].yaw)
        return
    samples = []
    for choice in values:
        yaw = tuple(choice.yaw)
        if len(yaw) != 2 or abs(float(yaw[0]) - float(yaw[1])) > 1e-9:
            raise RuntimeError(
                f"schema_vnext_runtime_yaw_selector_range_unsupported: multiple yaw choices must be fixed values ({choice.choice_id})"
            )
        samples.append({"id": choice.choice_id, "weight": float(choice.weight), "yaw": float(yaw[0])})
    payload["yaw_samples"] = samples


def _layout_without_variants(layout_payload: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(layout_payload or {}))
    out.pop("variants", None)
    return out


def _runtime_backend_surface_id(layout_plan: Any, surface_id: str) -> str:
    raw = str(surface_id or "").strip()
    if not raw:
        return ""
    surface = layout_plan.surfaces.get(raw) if hasattr(layout_plan, "surfaces") else None
    if surface is None or surface.alias_type != "ref":
        return raw
    ref = str(surface.ref or "").strip()
    if ref == "background.tabletop":
        return "tabletop"
    return ref or raw


def _object_local_surface_ref(layout_plan: Any, surface_id: str) -> tuple[str, str] | None:
    raw = str(surface_id or "").strip()
    if not raw:
        return None
    surface = layout_plan.surfaces.get(raw) if hasattr(layout_plan, "surfaces") else None
    if surface is None or surface.alias_type != "ref":
        return None
    ref = str(surface.ref or "").strip()
    if ref.startswith("background.") or "." not in ref:
        return None
    parent, surface_name = ref.split(".", 1)
    parent = parent.strip()
    surface_name = surface_name.strip()
    if not parent or not surface_name:
        return None
    return parent, surface_name


def _runtime_backend_volume_id(layout_plan: Any, volume_id: str) -> str:
    raw = str(volume_id or "").strip()
    if not raw:
        return ""
    volume = layout_plan.volumes.get(raw) if hasattr(layout_plan, "volumes") else None
    if volume is None or volume.alias_type != "ref":
        return raw
    return str(volume.ref or "").strip() or raw


def _runtime_backend_volume_parent_and_name(layout_plan: Any, volume_id: str) -> tuple[str, str]:
    resolved = _runtime_backend_volume_id(layout_plan, volume_id)
    if "." not in resolved:
        raise RuntimeError(f"schema_vnext_object_volume_ref_invalid: {volume_id}")
    parent, volume_name = resolved.split(".", 1)
    return parent, volume_name


def _object_local_volume_payload(ref: str, *, scene: VNextSceneSpec) -> dict[str, Any] | None:
    if "." not in ref:
        return None
    object_id, volume_name = ref.split(".", 1)
    raw_object = scene.object_spec_for_id(object_id)
    if raw_object is None:
        return None
    anchors = raw_object.anchors if isinstance(raw_object.anchors, Mapping) else {}
    local_volumes = anchors.get("local_volumes") if isinstance(anchors.get("local_volumes"), Mapping) else {}
    raw_volume = local_volumes.get(volume_name) if isinstance(local_volumes, Mapping) else None
    if not isinstance(raw_volume, Mapping):
        return None
    payload = deepcopy(dict(raw_volume))
    payload["id"] = volume_name
    payload["frame"] = {"type": "object", "node": object_id}
    return payload


def _volume_group_outputs_as_in_volume(group: Any, *, scene: VNextSceneSpec) -> bool:
    return (
        group.group_type == "volume"
        and len(group.nodes) == 1
        and _volume_group_mode(scene, group.group_id) in {"fit", "center"}
    )


def _volume_group_mode(scene: VNextSceneSpec, group_id: str) -> str:
    raw_group = _raw_group_payload(scene, group_id)
    return str(raw_group.get("mode") or raw_group.get("method") or "").strip().lower()


def _entry_volume_mode(entry: RuntimePlacementPlanEntry, *, scene: VNextSceneSpec, group_id: str) -> str:
    params = dict(entry.domain.params or {})
    return str(params.get("mode") or params.get("method") or _volume_group_mode(scene, group_id)).strip().lower()


def _entry_node_volume_mode(entry: RuntimePlacementPlanEntry, node: Any) -> str:
    params = dict(entry.domain.params or {})
    node_params = dict(getattr(node, "params", {}) or {})
    return str(params.get("mode") or node_params.get("mode") or "").strip().lower()


def _raw_group_payload(scene: VNextSceneSpec, group_id: str) -> Mapping[str, Any]:
    layout = scene.source.get("layout") if isinstance(scene.source, Mapping) else None
    groups = layout.get("groups") if isinstance(layout, Mapping) else None
    raw = groups.get(group_id) if isinstance(groups, Mapping) else None
    return raw if isinstance(raw, Mapping) else {}


def _apply_layout_object_extras(
    payload: dict[str, Any],
    *,
    scene: VNextSceneSpec,
    object_id: str,
    materialization: SceneMaterializationSample | None = None,
) -> None:
    raw_object = scene.object_spec_for_id(object_id)
    if raw_object is None:
        return
    # Match spawn materialization: pool asset_scales (possibly anisotropic) first,
    # then optional isotropic object/node scale ranges.
    norm_scale = None
    if materialization is not None and materialization.object_asset_scales:
        norm_scale = materialization.object_asset_scales.get(object_id)
    rand_scale = _materialized_object_scale(materialization, object_id)
    if norm_scale is not None:
        payload["scale"] = [axis * rand_scale for axis in norm_scale] if rand_scale is not None else list(norm_scale)
    elif rand_scale is not None:
        payload["scale"] = [rand_scale, rand_scale, rand_scale]
    # Object-level base orientation (XYZW) fills in when placement/group did not set one.
    if "orientation" not in payload and getattr(raw_object, "orientation", None) is not None:
        payload["orientation"] = list(raw_object.orientation)
    anchors = raw_object.anchors if isinstance(raw_object.anchors, Mapping) else {}
    local_surfaces = anchors.get("local_surfaces") if isinstance(anchors.get("local_surfaces"), Mapping) else None
    if local_surfaces:
        payload["support_surfaces"] = deepcopy(dict(local_surfaces))


def _materialized_object_scale(materialization: SceneMaterializationSample | None, object_id: str) -> float | None:
    if materialization is None:
        return None
    value = materialization.object_scales.get(object_id)
    if value is None:
        return None
    return float(value)


def _normalize_group_sampling_method(
    payload: dict[str, Any], group_type: str, *, replace_existing: bool = False
) -> None:
    method = payload.pop("method", None)
    if method is None:
        return
    if group_type == "region":
        if replace_existing or "sample_method" not in payload:
            payload["sample_method"] = method
        return
    if group_type in {"scatter", "volume"}:
        if replace_existing or "mode" not in payload:
            payload["mode"] = method


def _scene_seed(scene: VNextSceneSpec) -> int:
    if scene.layout.seed is not None:
        return int(scene.layout.seed)
    raw = scene.definition.get("seed") if isinstance(scene.definition, Mapping) else None
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _annotate_domain_selectors(
    plan: RuntimeSamplingPlan,
    selectors_by_domain: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    if not selectors_by_domain:
        return
    for index, domain in enumerate(plan.domains):
        selectors = selectors_by_domain.get(domain.domain_id)
        if not selectors:
            continue
        meta = dict(domain.meta)
        existing = dict(meta.get("selectors") or {}) if isinstance(meta.get("selectors"), Mapping) else {}
        existing.update({str(key): dict(value) for key, value in selectors.items()})
        meta["selectors"] = existing
        plan.domains[index] = _domain_with_meta(domain, meta)


def _annotate_case_pool_selector(plan: RuntimeSamplingPlan, entry: RuntimePlacementPlanEntry) -> None:
    selector = entry.selectors.get("ranges")
    if selector is None:
        return
    payload = _selector_payload(selector)
    payload["candidate_field"] = "surface_range_choice"
    for index, assignment in enumerate(plan.assignments):
        if assignment.mode == "case_pool":
            meta = dict(assignment.meta)
            meta["selector"] = payload
            plan.assignments[index] = replace(assignment, meta=meta)
    for index, conditional in enumerate(plan.conditional_assignments):
        if conditional.cases:
            meta = dict(conditional.meta)
            meta["selector"] = payload
            plan.conditional_assignments[index] = replace(conditional, meta=meta)


def _rename_group_case_pool_assignments(
    plan: RuntimeSamplingPlan,
    entries: Iterable[RuntimePlacementPlanEntry],
    *,
    runtime_id: str,
) -> None:
    entry_by_group = {str(entry.target_id): entry for entry in entries if entry.target_type == "group"}
    for index, assignment in enumerate(plan.assignments):
        if assignment.mode != "case_pool":
            continue
        group_id = _case_pool_group_id(assignment.meta)
        entry = entry_by_group.get(group_id)
        if entry is None:
            continue
        plan.assignments[index] = replace(assignment, plan_id=f"{runtime_id}:{safe_id(entry.entry_id)}")


def _annotate_case_pool_selectors(plan: RuntimeSamplingPlan, entries: Iterable[RuntimePlacementPlanEntry]) -> None:
    entry_by_group = {str(entry.target_id): entry for entry in entries if entry.target_type == "group"}
    for index, assignment in enumerate(plan.assignments):
        if assignment.mode != "case_pool":
            continue
        entry = entry_by_group.get(_case_pool_group_id(assignment.meta))
        if entry is None:
            continue
        selector = entry.selectors.get("ranges")
        if selector is None:
            continue
        payload = _selector_payload(selector)
        payload["candidate_field"] = "surface_range_choice"
        meta = dict(assignment.meta)
        meta["selector"] = payload
        plan.assignments[index] = replace(assignment, meta=meta)
    for index, conditional in enumerate(plan.conditional_assignments):
        entry = entry_by_group.get(_case_pool_group_id(conditional.meta))
        if entry is None:
            continue
        selector = entry.selectors.get("ranges")
        if selector is None:
            continue
        payload = _selector_payload(selector)
        payload["candidate_field"] = "surface_range_choice"
        meta = dict(conditional.meta)
        meta["selector"] = payload
        plan.conditional_assignments[index] = replace(conditional, meta=meta)


def _case_pool_group_id(meta: Mapping[str, Any]) -> str:
    for key in ("region", "scatter_group", "volume_group"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    return ""


def _domain_selector_payloads(entry: RuntimePlacementPlanEntry) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in ("ranges", "poses", "yaws"):
        selector = entry.selectors.get(name)
        if selector is not None:
            out[name] = _selector_payload(selector)
    return out


def _selector_payload(selector: SelectorSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": selector.selector_type}
    if selector.selector_type == "weighted":
        payload["weights"] = {str(key): float(value) for key, value in selector.weights.items()}
        payload["default_weight"] = float(selector.default_weight)
    elif selector.selector_type == "schedule":
        payload["mode"] = selector.mode
        payload["order"] = list(selector.order)
        payload["blocks"] = [{"id": block.candidate_id, "count": int(block.count)} for block in selector.blocks]
        payload["counts"] = {str(key): int(value) for key, value in selector.counts.items()}
        if selector.order_policy is not None:
            payload["order_policy"] = selector.order_policy
        payload["on_exhausted"] = selector.on_exhausted
    return payload


def _domain_with_meta(domain: PlacementDomain, meta: Mapping[str, Any]) -> PlacementDomain:
    return PlacementDomain(
        domain_id=domain.domain_id,
        object_name=domain.object_name,
        frame=domain.frame,
        geometry_type=domain.geometry_type,
        surface=domain.surface,
        x=domain.x,
        y=domain.y,
        z=domain.z,
        yaw=domain.yaw,
        base_quaternion=domain.base_quaternion,
        cells=domain.cells,
        poses=domain.poses,
        range_choices=domain.range_choices,
        yaw_choices=domain.yaw_choices,
        meta=dict(meta),
    )


def _merge_runtime_plan(target: RuntimeSamplingPlan, source: RuntimeSamplingPlan) -> None:
    if not source.success:
        target.success = False
    target.domains.extend(source.domains)
    target.assignments.extend(source.assignments)
    target.conditional_assignments.extend(source.conditional_assignments)
    target.outputs.extend(source.outputs)
    target.event_patches.extend(source.event_patches)
    target.diagnostics.extend(source.diagnostics)
    _merge_artifacts(target.artifacts, source.artifacts)


def _merge_artifacts(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key == "support_surfaces" and isinstance(value, list):
            existing = {str(item.get("id")) for item in target.get(key, []) if isinstance(item, Mapping)}
            merged = list(target.get(key, []) or [])
            for item in value:
                if isinstance(item, Mapping) and str(item.get("id")) not in existing:
                    merged.append(dict(item))
                    existing.add(str(item.get("id")))
            target[key] = merged
        elif key not in target:
            target[key] = deepcopy(value)
        else:
            target.setdefault("merged_subplan_artifacts", {})[key] = deepcopy(value)


def _materialization_sample(
    payload: Mapping[str, Any] | SceneMaterializationSample | None,
) -> SceneMaterializationSample | None:
    if isinstance(payload, SceneMaterializationSample):
        return payload
    if not isinstance(payload, Mapping) or not payload:
        return None
    objects = payload.get("objects") if isinstance(payload.get("objects"), Mapping) else {}
    selected = (
        payload.get("selected_candidate_refs") if isinstance(payload.get("selected_candidate_refs"), Mapping) else {}
    )
    selected_assignments = (
        payload.get("selected_assignments") if isinstance(payload.get("selected_assignments"), Mapping) else {}
    )
    object_scales = payload.get("object_scales") if isinstance(payload.get("object_scales"), Mapping) else {}
    object_spawn_masses = (
        payload.get("object_spawn_masses") if isinstance(payload.get("object_spawn_masses"), Mapping) else {}
    )
    object_asset_scales = (
        payload.get("object_asset_scales") if isinstance(payload.get("object_asset_scales"), Mapping) else {}
    )
    variants = payload.get("variant_choices") if isinstance(payload.get("variant_choices"), Mapping) else {}
    return SceneMaterializationSample(
        variant_choices={str(key): str(value) for key, value in variants.items()},
        background_asset=str(payload.get("background_asset") or ""),
        object_assets={str(key): str(value) for key, value in objects.items()},
        selected_candidate_refs={str(key): str(value) for key, value in selected.items()},
        object_scales={str(key): float(value) for key, value in object_scales.items()},
        object_spawn_masses={str(key): float(value) for key, value in object_spawn_masses.items()},
        object_asset_scales={str(key): _scale_tuple(value) for key, value in object_asset_scales.items()},
        background_asset_scale=(
            _scale_tuple(payload.get("background_asset_scale"))
            if payload.get("background_asset_scale") is not None
            else None
        ),
        selected_assignments={str(key): str(value) for key, value in selected_assignments.items()},
    )


def _scale_tuple(value: Any) -> tuple[float, float, float]:
    raw = list(value) if isinstance(value, (list, tuple)) else []
    if len(raw) != 3:
        return (1.0, 1.0, 1.0)
    return (float(raw[0]), float(raw[1]), float(raw[2]))
