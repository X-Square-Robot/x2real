from __future__ import annotations

import importlib
import random
from collections.abc import Mapping
from copy import deepcopy
from itertools import product
from typing import Any

from .layout_plan import GroupDomainPlan, NodeDomainPlan, build_canonical_layout_plan
from .materialization import AssetTagResolver, SceneMaterializationSample, sample_scene_materialization
from .parser import VNextSceneSpec, compose_scale_factor, scene_runtime_topology_mode, scene_uses_active_only_topology


def is_vnext_scene_config(config: Any) -> bool:
    if not isinstance(config, Mapping):
        return False
    background = config.get("background")
    layout = config.get("layout")
    variants = config.get("variants", {})
    return (
        isinstance(background, Mapping)
        and isinstance(background.get("asset"), Mapping)
        and isinstance(layout, Mapping)
        and not bool(layout.get("enabled"))
        and isinstance(variants, Mapping)
    )


def _preload_background_asset_module(module_path: str) -> None:
    """Import task-local ``@register_asset`` modules before object asset_id resolve.

    SceneOrder generates ``objects_component`` before ``background_component``, so
    relying solely on background ``_preload_asset_module`` is too late for object
    tags. Materialize-time import matches the standard register-on-import contract.
    """

    path = str(module_path or "").strip()
    if not path:
        return
    try:
        module = importlib.import_module(path)
    except Exception as exc:
        raise RuntimeError(f"failed to preload background.asset_module={path!r}: {exc}") from exc
    register_assets = getattr(module, "register_assets", None)
    if callable(register_assets):
        register_assets()


def materialize_manaenv_vnext_scene_config(
    scene_config: Mapping[str, Any] | None,
    *,
    rng: random.Random | None = None,
    asset_tag_resolver: AssetTagResolver | None = None,
) -> dict[str, Any]:
    """Materialize schema-vNext scene choices for ManaEnv scene construction.

    The public scene schema stays in metadata. The returned payload contains
    ManaEnv background/object component configs plus the internal scene-layout
    solver layout needed for static solve and runtime-plan compilation.
    """

    source = deepcopy(dict(scene_config or {}))
    scene = VNextSceneSpec.from_dict(source)
    _preload_background_asset_module(scene.background.asset_module or "")
    # Seed resolution: YAML override → pipeline.master_seed derive → OS entropy.
    # Never silently defaults to 0.
    rng = rng if rng is not None else random.Random(_resolve_scene_materialize_seed(scene))
    materialization = sample_scene_materialization(scene, rng=rng, asset_tag_resolver=asset_tag_resolver)
    active_only = scene_uses_active_only_topology(scene)
    runtime_asset_cache = _runtime_asset_cache_payload(
        scene,
        materialization=materialization,
        rng=rng,
        active_only=active_only,
    )
    solver_layout = build_manaenv_scene_solver_layout(scene, materialization=materialization)

    definition = deepcopy(dict(source.get("definition") or {}))
    metadata = deepcopy(dict(definition.get("metadata") or {}))
    metadata["schema_vnext_scene"] = source
    metadata["schema_vnext_materialization"] = _materialization_payload(
        materialization,
        runtime_asset_cache=runtime_asset_cache,
    )
    metadata["schema_vnext_runtime_assets"] = runtime_asset_cache
    metadata["schema_vnext_runtime_topology"] = scene_runtime_topology_mode(scene)
    metadata["layout"] = solver_layout
    definition["metadata"] = metadata

    out: dict[str, Any] = {
        "definition": definition,
        "layout": solver_layout,
    }
    if isinstance(source.get("render_settings"), Mapping):
        out["render_settings"] = deepcopy(dict(source["render_settings"]))
    out["background"] = _materialize_background(scene, materialization)
    out["objects"] = _materialized_objects(scene, materialization, runtime_asset_cache=runtime_asset_cache)
    return out


def build_manaenv_scene_solver_layout(
    scene: VNextSceneSpec,
    *,
    materialization: SceneMaterializationSample | None = None,
) -> dict[str, Any]:
    plan = build_canonical_layout_plan(scene)
    layout: dict[str, Any] = {
        "enabled": True,
        "seed": _scene_seed(scene),
        "objects": [],
    }
    background_poses = _background_poses(scene, materialization)
    if background_poses:
        layout["background_poses"] = background_poses
    surfaces = _solver_surfaces(plan)
    if surfaces:
        layout["surfaces"] = surfaces
    surface_catalog_refs = _surface_catalog_refs(plan)
    if surface_catalog_refs:
        layout["surface_catalog_refs"] = surface_catalog_refs
        if scene.background.surface_catalog:
            layout["surface_catalog_path"] = scene.background.surface_catalog
    volumes = _solver_volumes(plan, scene=scene)
    if volumes:
        layout["volumes"] = volumes

    grouped_nodes = {node for group in plan.groups.values() for node in group.nodes}
    layout_object_names: set[str] = set()
    in_volumes = []

    def append_layout_object(payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        name = str(payload.get("name") or "").strip()
        if not name or name in layout_object_names:
            return
        layout_object_names.add(name)
        layout["objects"].append(payload)

    for node_id, node in plan.nodes.items():
        if node_id in grouped_nodes:
            continue
        append_layout_object(_node_solver_domain(node, scene=scene, plan=plan, materialization=materialization))
        node_volume = _node_solver_volume_domain(node, plan=plan)
        if node_volume:
            in_volumes.append(node_volume)

    regions = []
    scatter_groups = []
    volume_groups = []
    slot_groups = []
    for group_id, group in plan.groups.items():
        if group.group_type == "slot":
            for payload in _slot_group_solver_objects(group, scene=scene, plan=plan, materialization=materialization):
                append_layout_object(payload)
            slot_groups.append(group)
            continue
        for payload in _group_member_solver_objects(group, scene=scene, plan=plan, materialization=materialization):
            append_layout_object(payload)
        group_payload = _group_solver_domain(group, scene=scene, plan=plan)
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
    if slot_groups:
        selected_assignments = dict(materialization.selected_assignments) if materialization is not None else {}
        layout["variants"] = _slot_groups_assignment_payload(
            slot_groups,
            scene=scene,
            plan=plan,
            selected_assignments=selected_assignments,
        )

    pose_samples = [
        _pose_sample_from_node(node, object_id=scene.layout.nodes[node_id].object_id)
        for node_id, node in plan.nodes.items()
        if node.placement_type == "pose"
    ]
    pose_samples = [item for item in pose_samples if item]
    if pose_samples:
        layout["pose_samples"] = pose_samples
    relative_directions = _solver_relative_directions(plan)
    if relative_directions:
        layout["relative_directions"] = relative_directions

    if not layout.get("objects") and not pose_samples:
        layout["enabled"] = False
    return layout


def _materialize_background(
    scene: VNextSceneSpec,
    materialization: SceneMaterializationSample,
) -> dict[str, Any]:
    position, orientation = _resolve_background_pose(scene, materialization.background_asset)
    out: dict[str, Any] = {
        "gen_method": 0,
        "tags": materialization.background_asset,
        "name": scene.background.background_id or "background",
        "initial_position": list(position),
        "initial_orientation": list(orientation),
    }
    # Forward ManaEnv AssetBasedComponent preload onto the component payload.
    module = str(scene.background.asset_module or "").strip()
    if module:
        out["asset_module"] = module
    return out


def _scale_spec_to_list(scale_spec: Any) -> list[float] | None:
    """Convert an authored ``ScaleSpec`` into a concrete vec3 scale list."""
    if scale_spec is None:
        return None
    axis = getattr(scale_spec, "axis_scale", None)
    if axis is not None:
        return [float(value) for value in axis]
    scale_range = getattr(scale_spec, "scale_range", None)
    if scale_range is not None:
        factor = float(scale_range[0])
        return [factor, factor, factor]
    return None


def _materialize_object(object_id: str, raw_object: Any, materialization: SceneMaterializationSample) -> dict[str, Any]:
    payload = {
        "gen_method": 0,
        "tags": materialization.object_assets[object_id],
        "name": object_id,
        "semantic_tags": list(raw_object.semantic_tags),
    }
    norm_scale = materialization.object_asset_scales.get(object_id) if materialization.object_asset_scales else None
    rand_scale = _materialized_object_scale(materialization, object_id)
    if norm_scale is not None:
        payload["scale"] = [axis * rand_scale for axis in norm_scale] if rand_scale is not None else list(norm_scale)
    elif rand_scale is not None:
        payload["scale"] = [rand_scale, rand_scale, rand_scale]
    else:
        fallback_scale = _scale_spec_to_list(raw_object.scale)
        if fallback_scale is not None:
            payload["scale"] = fallback_scale
    spawn_mass = _materialized_object_spawn_mass(materialization, object_id)
    if spawn_mass is not None:
        payload["spawn_mass"] = float(spawn_mass)
    _apply_spawn_friction_overrides(payload, raw_object)
    if getattr(raw_object, "kinematic", False):
        payload["kinematic"] = True
    collision_approximation = getattr(raw_object, "collision_approximation", None)
    if isinstance(collision_approximation, str) and collision_approximation.strip():
        payload["collision_approximation"] = collision_approximation.strip()
    if getattr(raw_object, "strip_child_z_rotation", False):
        payload["strip_child_z_rotation"] = True
    return payload


def _materialized_objects(
    scene: VNextSceneSpec,
    materialization: SceneMaterializationSample,
    *,
    runtime_asset_cache: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cache_objects = dict((runtime_asset_cache or {}).get("objects") or {})
    for object_id, raw_object in scene.objects.items():
        spawned_via_cache = False
        # Cache metadata is keyed by concrete ids (materialized_ids). For
        # non-instance pools that is just object_id; for instances it is
        # content_1..N, each with an independent cache window.
        for concrete_id in raw_object.materialized_ids:
            cache_entry = cache_objects.get(concrete_id)
            if not isinstance(cache_entry, Mapping) or not isinstance(cache_entry.get("slots"), list):
                continue
            spawned_via_cache = True
            for slot in cache_entry["slots"]:
                if not isinstance(slot, Mapping):
                    continue
                slot_name = str(slot.get("name") or "").strip()
                asset_id = str(slot.get("asset_id") or "").strip()
                if not slot_name or not asset_id:
                    continue
                out.append(
                    _materialize_object_with_asset(
                        slot_name,
                        raw_object,
                        asset_id,
                        materialization=materialization,
                        initial_position=slot.get("initial_position"),
                        scale=slot.get("scale"),
                    )
                )
        if spawned_via_cache:
            continue
        for concrete_id in raw_object.materialized_ids:
            out.append(_materialize_object(concrete_id, raw_object, materialization))
    return out


def _runtime_asset_cache_payload(
    scene: VNextSceneSpec,
    *,
    materialization: SceneMaterializationSample,
    rng: random.Random,
    cache_size: int = 3,
    active_only: bool = False,
) -> dict[str, Any]:
    objects: dict[str, Any] = {}
    for object_id, raw_object in scene.objects.items():
        if raw_object.asset.selector_type != "pool":
            continue
        # Instance expansion remaps object_assets onto concrete ids (content_1..N)
        # and drops the template key. Each concrete id gets its own cache window
        # so spawn names match runtime_compile slot names (id, id__slot_1, ...).
        object_cache_size = 1 if active_only else int(raw_object.asset.cache_size or cache_size)
        for concrete_id in raw_object.materialized_ids:
            materialized_asset = materialization.object_assets[concrete_id]
            slots = _runtime_asset_cache_slots(
                concrete_id,
                raw_object,
                materialized_asset=materialized_asset,
                rng=rng,
                cache_size=object_cache_size,
            )
            objects[concrete_id] = {
                "source_object_id": object_id,
                "cache_size": object_cache_size,
                "isolation": {"z": _ISOLATION_Z},
                "slots": slots,
            }
    return {"objects": objects}


def _runtime_asset_cache_slots(
    object_id: str,
    raw_object: Any,
    *,
    materialized_asset: str,
    rng: random.Random,
    cache_size: int,
) -> list[dict[str, Any]]:
    pool: list[tuple[str, str, tuple[float, float, float] | None, Mapping[str, float], float | None]] = []
    for candidate in raw_object.asset.candidates:
        asset_scale_factors = getattr(candidate, "asset_scale_factors", {}) or {}
        for asset_id in candidate.assets:
            pool.append(
                (
                    candidate.candidate_id,
                    asset_id,
                    candidate.asset_scales.get(asset_id),
                    dict(candidate.scale_normalization),
                    asset_scale_factors.get(asset_id),
                )
            )
    by_asset = {
        asset_id: (candidate_id, scale, policy, factor) for candidate_id, asset_id, scale, policy, factor in pool
    }
    materialized_candidate, materialized_scale, materialized_policy, materialized_factor = by_asset.get(
        materialized_asset, ("", None, {}, None)
    )
    # Default ``shuffle``: slot 0 = materialized pick; remaining slots are a
    # shuffled sample of the pool (random semantics for general tasks).
    # Opt-in ``authored`` (nut_on_screw): candidate[i] → slot i so task materials
    # can target screw_a↔nut, screw_b↔nut__slot_1, … without reshuffling.
    slot_order = str(getattr(raw_object.asset, "cache_slot_order", "shuffle") or "shuffle").strip().lower()
    chosen: list[tuple[str, str, tuple[float, float, float] | None, Mapping[str, float], float | None]]
    if slot_order == "authored":
        chosen = []
        for entry in pool:
            if len(chosen) >= int(cache_size):
                break
            chosen.append(entry)
        while len(chosen) < int(cache_size):
            if pool:
                chosen.append(pool[rng.randrange(len(pool))])
            elif materialized_asset:
                chosen.append(("", materialized_asset, None, {}, None))
            else:
                break
    else:
        chosen = [
            (materialized_candidate, materialized_asset, materialized_scale, materialized_policy, materialized_factor)
        ]
        remaining = [
            (candidate_id, asset_id, scale, policy, factor)
            for candidate_id, asset_id, scale, policy, factor in pool
            if asset_id != materialized_asset
        ]
        rng.shuffle(remaining)
        while len(chosen) < int(cache_size):
            if remaining:
                chosen.append(remaining.pop(0))
            elif pool:
                chosen.append(pool[rng.randrange(len(pool))])
            else:
                break
    slots = []
    for index, (candidate_id, asset_id, scale, scale_policy, scale_factor) in enumerate(chosen):
        slot_name = str(object_id) if index == 0 else f"{object_id}__slot_{index}"
        scale = scale or _lazy_normalized_asset_scale(asset_id, scale_policy, scale_factor)
        slot = {
            "index": index,
            "name": slot_name,
            "candidate_id": candidate_id,
            "asset_id": asset_id,
            "initial_position": _isolation_position(index),
        }
        if scale is not None:
            slot["scale"] = list(scale)
        slots.append(slot)
    if slots:
        slots[0]["initial_position"] = None
    return slots


def _lazy_normalized_asset_scale(
    asset_id: str,
    scale_policy: Mapping[str, float],
    scale_factor: float | None = None,
) -> tuple[float, float, float] | None:
    if not scale_policy and not scale_factor:
        return None
    try:
        from isaaclab_arena.assets.asset_registry import AssetRegistry

        from ..assets.geometry_resolver import ObjectGeometryResolver
    except Exception:
        return None
    try:
        asset_class = AssetRegistry().get_asset_by_name(asset_id)
        geometry = ObjectGeometryResolver().resolve(asset_class)
    except Exception:
        return None
    if geometry is None:
        return None
    dims = [float(value) for value in geometry.local_bounds.size]
    if not dims:
        return None
    # Per-asset concrete multiplier first, candidate scale_normalization second.
    return compose_scale_factor(dims, scale_factor, scale_policy)


def _materialize_object_with_asset(
    object_id: str,
    raw_object: Any,
    asset_id: str,
    *,
    materialization: SceneMaterializationSample | None = None,
    initial_position: Any = None,
    scale: Any = None,
) -> dict[str, Any]:
    payload = {
        "gen_method": 0,
        "tags": asset_id,
        "name": object_id,
        "semantic_tags": list(raw_object.semantic_tags) if initial_position is None else [],
    }
    if scale is not None:
        payload["scale"] = list(scale)
    else:
        fallback_scale = _scale_spec_to_list(raw_object.scale)
        if fallback_scale is not None:
            payload["scale"] = fallback_scale
    if initial_position is not None:
        payload["initial_position"] = list(initial_position)
    # Cache slots share the logical object's sampled mass (keyed by source object id).
    spawn_mass = _materialized_object_spawn_mass(materialization, getattr(raw_object, "object_id", ""))
    if spawn_mass is not None:
        payload["spawn_mass"] = float(spawn_mass)
    _apply_spawn_friction_overrides(payload, raw_object)
    if getattr(raw_object, "kinematic", False):
        payload["kinematic"] = True
    collision_approximation = getattr(raw_object, "collision_approximation", None)
    if isinstance(collision_approximation, str) and collision_approximation.strip():
        payload["collision_approximation"] = collision_approximation.strip()
    if getattr(raw_object, "strip_child_z_rotation", False):
        payload["strip_child_z_rotation"] = True
    return payload


# Sentinel depth for inactive runtime-cache slots; mirrors the legacy away_z=-10
# convention used by hand-authored event terms. Runtime teleport uses the same
# value via manaenv_bridge.runtime_events._INACTIVE_ISOLATION_Z.
_ISOLATION_Z = -10.0


def _isolation_position(index: int) -> list[float]:
    # Park inactive cache slots BELOW the workspace, matching the legacy away_z
    # convention. These invisible spares keep gravity + collision enabled, so
    # parking them above (z>0) let PhysX drop them through the tabletop onto the
    # active object and scatter the scene. Below the floor they fall away harmlessly.
    return [0.15 * float(index), 0.0, _ISOLATION_Z - 0.1 * float(index)]


def _solver_surfaces(plan: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for surface_id, surface in plan.surfaces.items():
        if surface.alias_type != "manual":
            continue
        payload = deepcopy(dict(surface.geometry))
        payload.setdefault("id", surface_id)
        out.append(payload)
    return out


def _surface_catalog_refs(plan: Any) -> list[str]:
    refs: list[str] = []
    for surface_id, surface in plan.surfaces.items():
        if surface.alias_type != "ref":
            continue
        ref = str(surface.ref or "").strip()
        if ref == "background.tabletop":
            refs.append("tabletop")
    return refs


def _solver_volumes(plan: Any, *, scene: VNextSceneSpec) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for volume_id, volume in plan.volumes.items():
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


def _node_solver_domain(
    node: NodeDomainPlan,
    *,
    scene: VNextSceneSpec,
    plan: Any,
    materialization: SceneMaterializationSample | None = None,
) -> dict[str, Any] | None:
    object_id = node.object_id
    if node.placement_type == "surface":
        payload = {"name": object_id, "support": _solver_surface_id(plan, node.surface or "")}
        if node.base_orientation is not None:
            payload["orientation"] = list(node.base_orientation)
        if node.ranges:
            payload["surface_ranges"] = [_range_choice_payload(choice) for choice in node.ranges]
        if node.yaws:
            _assign_solver_yaws(payload, node.yaws, target=object_id)
        _apply_layout_object_extras(payload, scene=scene, object_id=object_id, materialization=materialization)
        return payload
    if node.placement_type == "pose":
        payload = {"name": object_id}
        _apply_layout_object_extras(payload, scene=scene, object_id=object_id, materialization=materialization)
        return payload
    if node.placement_type == "volume":
        payload = {"name": object_id}
        if node.yaws:
            _assign_solver_yaws(payload, node.yaws, target=object_id)
        _apply_layout_object_extras(payload, scene=scene, object_id=object_id, materialization=materialization)
        return payload
    return None


def _node_solver_volume_domain(node: NodeDomainPlan, *, plan: Any) -> dict[str, Any] | None:
    if node.placement_type != "volume":
        return None
    mode = str((node.params or {}).get("mode") or "fit").strip().lower()
    return {
        "object": node.object_id,
        "volume": _solver_volume_id(plan, node.volume or ""),
        "mode": mode,
    }


def _pose_sample_from_node(node: NodeDomainPlan, *, object_id: str) -> dict[str, Any] | None:
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


def _group_solver_domain(group: GroupDomainPlan, *, scene: VNextSceneSpec, plan: Any) -> dict[str, Any] | None:
    if group.group_type not in {"region", "scatter", "volume"}:
        return None
    if _volume_group_outputs_as_in_volume(group, scene=scene):
        node_id = group.nodes[0]
        return {
            "object": scene.layout.nodes[node_id].object_id,
            "volume": _solver_volume_id(plan, group.volume or ""),
            "mode": _volume_group_mode(scene, group.group_id) or "fit",
        }
    payload = {
        "id": group.group_id,
        "objects": [scene.layout.nodes[node_id].object_id for node_id in group.nodes],
    }
    if group.surface:
        payload["surface"] = _solver_surface_id(plan, group.surface)
    if group.volume:
        payload["volume"] = _solver_volume_id(plan, group.volume)
    if group.ranges:
        payload["surface_ranges"] = [_range_choice_payload(choice) for choice in group.ranges]
    if group.yaws:
        _assign_solver_yaws(payload, group.yaws, target=group.group_id)
    raw_group = _raw_group_payload(scene, group.group_id)
    # Keep parity with runtime_sampling_plan group export: order/align axes are
    # required for left-to-right pen packing and must not be dropped here.
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
    return payload


def _group_member_solver_objects(
    group: GroupDomainPlan,
    *,
    scene: VNextSceneSpec,
    plan: Any,
    materialization: SceneMaterializationSample | None = None,
) -> list[dict[str, Any]]:
    if group.group_type not in {"region", "scatter", "volume"}:
        return []
    support = _solver_surface_id(plan, group.surface or "") if group.surface else ""
    out: list[dict[str, Any]] = []
    for node_id in group.nodes:
        object_id = scene.layout.nodes[node_id].object_id
        payload = {"name": object_id}
        if support:
            payload["support"] = support
        if group.base_orientation is not None:
            payload["orientation"] = list(group.base_orientation)
        if group.yaws:
            _assign_solver_yaws(payload, group.yaws, target=object_id)
        _apply_layout_object_extras(payload, scene=scene, object_id=object_id, materialization=materialization)
        out.append(payload)
    return out


def _slot_group_solver_objects(
    group: GroupDomainPlan,
    *,
    scene: VNextSceneSpec,
    plan: Any,
    materialization: SceneMaterializationSample | None = None,
) -> list[dict[str, Any]]:
    default_surface = ""
    if group.slots:
        first_slot = next(iter(group.slots.values()))
        default_surface = _solver_surface_id(plan, first_slot.surface)
    out: list[dict[str, Any]] = []
    for node_id in group.nodes:
        object_id = scene.layout.nodes[node_id].object_id
        payload = {"name": object_id}
        if default_surface:
            payload["support"] = default_surface
        _apply_layout_object_extras(payload, scene=scene, object_id=object_id, materialization=materialization)
        out.append(payload)
    return out


def _slot_groups_assignment_payload(
    groups: list[GroupDomainPlan],
    *,
    scene: VNextSceneSpec,
    plan: Any,
    selected_assignments: Mapping[str, str],
) -> dict[str, Any]:
    if len(groups) == 1:
        return _slot_group_assignment_payload(
            groups[0],
            scene=scene,
            plan=plan,
            selected_assignment=selected_assignments.get(groups[0].group_id),
        )
    slots: list[dict[str, Any]] = []
    seen_slots: set[str] = set()
    for group in groups:
        for slot_payload in _slot_payloads(group, plan=plan):
            raw_slot_id = str(slot_payload["id"])
            slot_id = _solver_slot_id(group, raw_slot_id)
            if slot_id in seen_slots:
                raise ValueError(f"schema_vnext_static_slot_id_duplicate: {slot_id}")
            seen_slots.add(slot_id)
            slot_payload = dict(slot_payload)
            slot_payload["id"] = slot_id
            slots.append(slot_payload)
    grouped_options = [
        _slot_group_case_options(group, scene=scene, selected_assignment=selected_assignments.get(group.group_id))
        for group in groups
    ]
    cases: list[dict[str, Any]] = []
    for combo in product(*grouped_options):
        case_ids = [str(option["assignment_id"]) for option in combo]
        objects: dict[str, str] = {}
        group_params: dict[str, Any] = {}
        for group, option in zip(groups, combo, strict=True):
            objects.update(
                {object_id: _solver_slot_id(group, slot_id) for object_id, slot_id in dict(option["objects"]).items()}
            )
            group_params[group.group_id] = {
                "assignment_id": option["assignment_id"],
                "params": dict(option.get("params") or {}),
            }
        cases.append(
            {
                "id": "__".join(f"{group.group_id}:{case_id}" for group, case_id in zip(groups, case_ids, strict=True)),
                "objects": objects,
                "params": {"slot_groups": group_params},
            }
        )
    return {
        "type": "fixed_cases",
        "slots": slots,
        "cases": cases,
    }


def _slot_group_assignment_payload(
    group: GroupDomainPlan,
    *,
    scene: VNextSceneSpec,
    plan: Any,
    selected_assignment: str | None = None,
) -> dict[str, Any]:
    mode = _slot_assignment_mode(scene, group.group_id)
    slots = _slot_payloads(group, plan=plan)
    if not selected_assignment and mode == "permutation" and len(group.nodes) == len(slots):
        return {
            "type": "permutation",
            "objects": [scene.layout.nodes[node_id].object_id for node_id in group.nodes],
            "slots": slots,
        }
    cases = [
        {
            "id": option["assignment_id"],
            "objects": option["objects"],
            "params": dict(option.get("params") or {}),
        }
        for option in _slot_group_case_options(group, scene=scene, selected_assignment=selected_assignment)
    ]
    return {
        "type": "fixed_cases",
        "slots": slots,
        "cases": cases,
    }


def _slot_payloads(group: GroupDomainPlan, *, plan: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for slot_id, slot in group.slots.items():
        if not slot.ranges:
            continue
        payload = {
            "id": slot_id,
            "surface": _solver_surface_id(plan, slot.surface),
            "surface_range": _fixed_range_payload(slot.ranges[0]),
        }
        slot_yaw = _slot_static_yaw(slot)
        if slot_yaw is not None:
            payload["yaw"] = list(slot_yaw)
        out.append(payload)
    return out


def _solver_slot_id(group: GroupDomainPlan, slot_id: str) -> str:
    return f"{group.group_id}.{slot_id}"


def _slot_group_case_options(
    group: GroupDomainPlan,
    *,
    scene: VNextSceneSpec,
    selected_assignment: str | None = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for assignment_id in group.assignment_ids:
        if selected_assignment and assignment_id != selected_assignment:
            continue
        case = group.assignment_cases[assignment_id]
        cases.append(
            {
                "assignment_id": case.assignment_id,
                "objects": {
                    scene.layout.nodes[node_id].object_id: slot_id for node_id, slot_id in case.bindings.items()
                },
                "params": dict(case.params),
            }
        )
    return cases


def _solver_relative_directions(plan: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    nodes = plan.nodes if hasattr(plan, "nodes") else {}
    for relation in getattr(plan, "relations", ()) or ():
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


def _range_choice_payload(choice: Any) -> dict[str, Any]:
    return {
        "id": choice.choice_id,
        "weight": choice.weight,
        "surface_range": _fixed_range_payload(choice),
    }


def _fixed_range_payload(choice: Any) -> dict[str, Any]:
    return {
        "x": list(choice.x),
        "y": list(choice.y),
    }


def _slot_fixed_yaw(slot: Any) -> tuple[float, float] | None:
    yaws = list(getattr(slot, "yaws", ()) or ())
    if len(yaws) != 1:
        return None
    yaw = tuple(yaws[0].yaw)
    if len(yaw) != 2 or abs(float(yaw[0]) - float(yaw[1])) > 1e-9:
        return None
    value = float(yaw[0])
    return (value, value)


def _slot_static_yaw(slot: Any) -> tuple[float, float] | None:
    yaws = list(getattr(slot, "yaws", ()) or ())
    if not yaws:
        return None
    if len(yaws) == 1:
        yaw = tuple(yaws[0].yaw)
        if len(yaw) != 2:
            return None
        return (float(yaw[0]), float(yaw[1]))
    raise RuntimeError(
        f"schema_vnext_static_slot_yaw_choices_unsupported: static slot yaw accepts a single value or range ({slot.slot_id})"
    )


def _yaw_choice_payload(choice: Any) -> dict[str, Any]:
    yaw = tuple(choice.yaw)
    if len(yaw) != 2 or abs(float(yaw[0]) - float(yaw[1])) > 1e-9:
        raise ValueError(
            f"schema_vnext_solver_yaw_sample_range_unsupported: weighted yaw samples must be fixed yaw values ({choice.choice_id})"
        )
    return {
        "id": choice.choice_id,
        "weight": choice.weight,
        "yaw": float(yaw[0]),
    }


def _materialization_payload(
    materialization: SceneMaterializationSample,
    *,
    runtime_asset_cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "variant_choices": dict(materialization.variant_choices),
        "background_asset": materialization.background_asset,
        "objects": dict(materialization.object_assets),
        "object_asset_scales": {key: list(value) for key, value in materialization.object_asset_scales.items()},
        "background_asset_scale": (
            list(materialization.background_asset_scale) if materialization.background_asset_scale is not None else None
        ),
        "selected_candidate_refs": dict(materialization.selected_candidate_refs),
        "object_scales": dict(materialization.object_scales),
        "object_spawn_masses": dict(materialization.object_spawn_masses),
        "selected_assignments": dict(materialization.selected_assignments),
        "runtime_asset_cache": dict(runtime_asset_cache or {}),
    }


def _background_poses(
    scene: VNextSceneSpec, materialization: SceneMaterializationSample | None
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
    position, orientation = _resolve_background_pose(scene, asset_id)
    return {
        str(asset_id): {
            "position": list(position),
            "orientation": list(orientation),
        }
    }


def _resolve_background_pose(
    scene: VNextSceneSpec, asset_id: str
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Prefer ``pose_by_asset[asset_id]`` when present; else fall back to ``background.pose``."""
    key = str(asset_id or "").strip()
    if key and key in scene.background.pose_by_asset:
        return scene.background.pose_by_asset[key]
    return scene.background.pose_position, scene.background.pose_orientation


def _raw_group_payload(scene: VNextSceneSpec, group_id: str) -> Mapping[str, Any]:
    layout = scene.source.get("layout") if isinstance(scene.source, Mapping) else None
    groups = layout.get("groups") if isinstance(layout, Mapping) else None
    raw = groups.get(group_id) if isinstance(groups, Mapping) else None
    return raw if isinstance(raw, Mapping) else {}


def _slot_assignment_mode(scene: VNextSceneSpec, group_id: str) -> str:
    raw_group = _raw_group_payload(scene, group_id)
    assignments = raw_group.get("assignments") if isinstance(raw_group.get("assignments"), Mapping) else {}
    return str(assignments.get("mode") or "").strip()


def _solver_surface_id(plan: Any, surface_id: str) -> str:
    raw = str(surface_id or "").strip()
    if not raw:
        return ""
    surface = plan.surfaces.get(raw) if hasattr(plan, "surfaces") else None
    if surface is None or surface.alias_type != "ref":
        return raw
    ref = str(surface.ref or "").strip()
    if ref == "background.tabletop":
        return "tabletop"
    return ref or raw


def _solver_volume_id(plan: Any, volume_id: str) -> str:
    raw = str(volume_id or "").strip()
    if not raw:
        return ""
    volume = plan.volumes.get(raw) if hasattr(plan, "volumes") else None
    if volume is None or volume.alias_type != "ref":
        return raw
    return str(volume.ref or "").strip() or raw


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


def _volume_group_outputs_as_in_volume(group: GroupDomainPlan, *, scene: VNextSceneSpec) -> bool:
    return (
        group.group_type == "volume"
        and len(group.nodes) == 1
        and _volume_group_mode(scene, group.group_id) in {"fit", "center"}
    )


def _volume_group_mode(scene: VNextSceneSpec, group_id: str) -> str:
    raw_group = _raw_group_payload(scene, group_id)
    return str(raw_group.get("mode") or "").strip().lower()


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


def _materialized_object_spawn_mass(materialization: SceneMaterializationSample | None, object_id: str) -> float | None:
    if materialization is None or not object_id:
        return None
    value = materialization.object_spawn_masses.get(object_id)
    if value is None:
        return None
    return float(value)


def _apply_spawn_friction_overrides(payload: dict[str, Any], raw_object: Any) -> None:
    """Copy authored rigid-body friction overrides onto the Scene DSL payload."""
    static_friction = getattr(raw_object, "spawn_static_friction", None)
    if static_friction is not None:
        payload["spawn_static_friction"] = float(static_friction)
    dynamic_friction = getattr(raw_object, "spawn_dynamic_friction", None)
    if dynamic_friction is not None:
        payload["spawn_dynamic_friction"] = float(dynamic_friction)


def _assign_solver_yaws(payload: dict[str, Any], choices: Any, *, target: str) -> None:
    values = list(choices or [])
    if not values:
        return
    if len(values) == 1:
        payload["yaw"] = list(values[0].yaw)
        return
    payload["yaw_samples"] = [_yaw_choice_payload(choice) for choice in values]


def _explicit_scene_seed(scene: VNextSceneSpec) -> int | None:
    """Return the author-provided scene seed override, or ``None`` when unset."""

    if scene.layout.seed is not None:
        return int(scene.layout.seed)
    raw = scene.definition.get("seed") if isinstance(scene.definition, Mapping) else None
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _resolve_scene_materialize_seed(scene: VNextSceneSpec) -> int | None:
    """YAML override → master derive → ``None`` (OS entropy via ``Random(None)``)."""

    from manaenv.pipeline.runtime import master_seed as master_seed_runtime

    scene_type = str(scene.definition.get("scene_type") or "").strip() if isinstance(scene.definition, Mapping) else ""
    return master_seed_runtime.resolve_base_seed(
        master_seed_runtime.NS_SCENE_MATERIALIZE,
        scene_type or "scene",
        override=_explicit_scene_seed(scene),
        allow_os_entropy=False,
    )


def _scene_seed(scene: VNextSceneSpec) -> int:
    """Layout-solver seed: override → master derive → OS entropy (never silent ``0``)."""

    from manaenv.pipeline.runtime import master_seed as master_seed_runtime

    scene_type = str(scene.definition.get("scene_type") or "").strip() if isinstance(scene.definition, Mapping) else ""
    return int(
        master_seed_runtime.resolve_base_seed(
            master_seed_runtime.NS_SCENE_LAYOUT,
            scene_type or "scene",
            override=_explicit_scene_seed(scene),
            allow_os_entropy=True,
        )
    )
