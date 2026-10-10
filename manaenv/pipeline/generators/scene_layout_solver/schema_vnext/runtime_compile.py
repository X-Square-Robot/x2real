from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from ..common.diagnostics import error
from .layout_plan import (
    CanonicalLayoutPlan,
    GroupDomainPlan,
    NodeDomainPlan,
    PoseChoicePlan,
    RangeChoicePlan,
    YawChoicePlan,
    _pose_choices,
    _range_choices,
    _yaw_choices,
    build_canonical_layout_plan,
)
from .lighting_manifest import load_lighting_photometric_profile, load_lighting_texture_manifest
from .material_manifest import load_material_template_manifest
from .materialization import SceneMaterializationSample
from .parser import SelectorSpec, VNextSceneSpec, VNextTaskSpec, _list, _mapping, _non_empty_str, _scalar_or_pair
from .selector_engine import SelectorRuntimeState, select_candidate


@dataclass(frozen=True)
class RuntimePlacementDomainPlan:
    domain_kind: str
    ranges: tuple[RangeChoicePlan, ...] = ()
    yaws: tuple[YawChoicePlan, ...] = ()
    poses: tuple[PoseChoicePlan, ...] = ()
    assignments: tuple[str, ...] = ()
    assignment_cases: Mapping[str, Any] = field(default_factory=dict)
    slot_order: tuple[str, ...] = ()
    node_objects: Mapping[str, str] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimePlacementPlanEntry:
    entry_id: str
    target_type: str
    target_id: str
    sample_type: str
    domain: RuntimePlacementDomainPlan
    selectors: Mapping[str, SelectorSpec] = field(default_factory=dict)
    jitter: Mapping[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeLightingPlanEntry:
    channel: str
    kind: str
    selector: SelectorSpec | None = None
    candidates: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    value_range: tuple[float, float] | None = None
    source_channel: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeMaterialPlanEntry:
    entry_id: str
    target_type: str
    policy_type: str
    candidates: tuple[str, ...]
    selector: SelectorSpec
    objects: tuple[str, ...] = ()
    prim_paths: tuple[str, ...] = ()
    surface_material: str | None = None
    material_slots: str | tuple[str, ...] = "auto"
    candidate_templates: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    candidate_payloads: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    apply: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeAssetCandidatePlan:
    candidate_id: str
    assets: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeAssetPlanEntry:
    entry_id: str
    target_type: str
    target_id: str
    object_id: str
    selector: SelectorSpec
    candidates: tuple[RuntimeAssetCandidatePlan, ...]
    cache_size: int = 3
    slots: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeSampleCompilePlan:
    """Compiled vNext runtime_sample authoring directives.

    Placement entries in this object are not a reset-time pose backend. They
    must be compiled into the full ``RuntimeSamplingPlan`` before ManaEnv reset.
    """

    runtime_sample_id: str
    seed: int | None
    placements: tuple[RuntimePlacementPlanEntry, ...] = ()
    assets: tuple[RuntimeAssetPlanEntry, ...] = ()
    lighting: tuple[RuntimeLightingPlanEntry, ...] = ()
    materials: tuple[RuntimeMaterialPlanEntry, ...] = ()


@dataclass(frozen=True)
class RuntimePlacementSelection:
    entry_id: str
    target_type: str
    target_id: str
    domain_kind: str
    choices: Mapping[str, str]
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeSampleSelectionResult:
    runtime_sample_id: str
    env_id: int
    placements: Mapping[str, RuntimePlacementSelection]
    assets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    lighting: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    materials: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeEpisodeContext:
    runtime_sample_id: str
    episode_index: int
    env_id: int
    seed: int | None
    compile_context: Mapping[str, Any]
    selections: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_sample_id": self.runtime_sample_id,
            "episode_index": self.episode_index,
            "env_id": self.env_id,
            "seed": self.seed,
            "compile_context": dict(self.compile_context),
            "selections": dict(self.selections),
        }


def compile_runtime_sample(task: VNextTaskSpec, scene: VNextSceneSpec) -> RuntimeSampleCompilePlan:
    layout_plan = build_canonical_layout_plan(scene)
    runtime_payload = _mapping(
        (_mapping(task.source.get("layout"), target="task.layout")).get("runtime_sample"),
        target="task.layout.runtime_sample",
    )
    _validate_global_scene_variant_selection(runtime_payload, scene=scene)
    targets = _mapping(runtime_payload.get("targets", {}), target="task.layout.runtime_sample.targets")
    placement_payloads = targets.get("placements", [])
    global_payload = _mapping(runtime_payload.get("global", {}), target="task.layout.runtime_sample.global")
    entries = tuple(
        _compile_runtime_placement(raw_placement, index=index, task=task, layout_plan=layout_plan)
        for index, raw_placement in enumerate(placement_payloads)
    )
    return RuntimeSampleCompilePlan(
        runtime_sample_id=task.runtime_sample.runtime_id,
        seed=task.runtime_sample.seed,
        assets=_compile_assets(task, scene=scene),
        placements=entries,
        lighting=_compile_lighting(global_payload.get("lighting", {}), scene=scene),
        materials=_compile_materials(targets.get("materials", []), scene=scene),
    )


def sample_runtime_placement_choices(
    plan: RuntimeSampleCompilePlan,
    *,
    rng: random.Random,
    state: SelectorRuntimeState | None = None,
    env_id: int = 0,
) -> RuntimeSampleSelectionResult:
    runtime_state = state if state is not None else SelectorRuntimeState()
    placements: dict[str, RuntimePlacementSelection] = {}
    for entry in plan.placements:
        choices: dict[str, str] = {}
        details: dict[str, Any] = {}
        for selector_name, selector in entry.selectors.items():
            candidate_ids = _candidate_ids(entry.domain, selector_name)
            result = select_candidate(
                candidate_ids,
                selector,
                selector_id=f"{entry.entry_id}.{selector_name}",
                runtime_sample_id=plan.runtime_sample_id,
                env_id=env_id,
                rng=rng,
                state=runtime_state,
            )
            choices[selector_name] = result.selected_id
            if selector_name == "assignments":
                details["assignment"] = _assignment_detail(entry, result.selected_id)
        placements[entry.entry_id] = RuntimePlacementSelection(
            entry_id=entry.entry_id,
            target_type=entry.target_type,
            target_id=entry.target_id,
            domain_kind=entry.domain.domain_kind,
            choices=choices,
            details=details,
        )
    assets = _sample_assets(plan, rng=rng, state=runtime_state, env_id=env_id)
    lighting = _sample_lighting(plan, rng=rng, state=runtime_state, env_id=env_id)
    materials = _sample_materials(plan, rng=rng, state=runtime_state, env_id=env_id)
    return RuntimeSampleSelectionResult(
        runtime_sample_id=plan.runtime_sample_id,
        env_id=env_id,
        placements=placements,
        assets=assets,
        lighting=lighting,
        materials=materials,
    )


def build_runtime_episode_context(
    plan: RuntimeSampleCompilePlan,
    selection: RuntimeSampleSelectionResult,
    *,
    episode_index: int,
    materialization: SceneMaterializationSample | None = None,
    seed: int | None = None,
) -> RuntimeEpisodeContext:
    return RuntimeEpisodeContext(
        runtime_sample_id=plan.runtime_sample_id,
        episode_index=int(episode_index),
        env_id=selection.env_id,
        seed=seed if seed is not None else plan.seed,
        compile_context=_compile_context(materialization),
        selections={
            "assets": dict(selection.assets),
            "placements": _placement_context(plan, selection),
            "lighting": dict(selection.lighting),
            "materials": dict(selection.materials),
        },
    )


def _compile_runtime_placement(
    raw_placement: Any,
    *,
    index: int,
    task: VNextTaskSpec,
    layout_plan: CanonicalLayoutPlan,
) -> RuntimePlacementPlanEntry:
    placement = _mapping(raw_placement, target=f"task.layout.runtime_sample.targets.placements[{index}]")
    entry_id = str(placement.get("id") or "").strip()
    if not entry_id:
        raise error("schema_vnext_id_required", "runtime placement requires id", target=f"placements[{index}]")
    target = _mapping(placement.get("target"), target=f"placements.{entry_id}.target")
    target_type = str(target.get("type") or "").strip()
    target_id = str(target.get("id") or "").strip()
    sample = _mapping(placement.get("sample"), target=f"placements.{entry_id}.sample")
    sample_type = str(sample.get("type") or "").strip()
    inherited = _inherited_domain(
        target_type=target_type, target_id=target_id, entry_id=entry_id, layout_plan=layout_plan
    )
    domain = _effective_domain(inherited, sample, sample_type=sample_type, target_type=target_type, entry_id=entry_id)
    explicit_selectors = dict(task.runtime_sample.placements[index].selectors)
    selectors = _runtime_selectors(domain, explicit_selectors=explicit_selectors, target=entry_id)
    jitter = _jitter(sample, domain_kind=domain.domain_kind, sample_type=sample_type, target=entry_id)
    return RuntimePlacementPlanEntry(
        entry_id=entry_id,
        target_type=target_type,
        target_id=target_id,
        sample_type=sample_type,
        domain=domain,
        selectors=selectors,
        jitter=jitter,
    )


def _compile_context(materialization: SceneMaterializationSample | None) -> Mapping[str, Any]:
    if materialization is None:
        return {"materialized": {"background": {}, "objects": {}}}
    objects: dict[str, Mapping[str, Any]] = {}
    for object_id, asset_id in materialization.object_assets.items():
        target_key = f"object:{object_id}"
        objects[object_id] = {
            "candidate_ref_id": materialization.selected_candidate_refs.get(target_key),
            "asset_id": asset_id,
            "scale": materialization.object_scales.get(object_id),
        }
        scale = materialization.object_asset_scales.get(object_id)
        if scale is not None:
            objects[object_id]["scale"] = list(scale)
    return {
        "materialized": {
            "background": {
                "candidate_ref_id": materialization.selected_candidate_refs.get("background"),
                "asset_id": materialization.background_asset,
            },
            "objects": objects,
        },
        "scene_variants": {
            variant_id: {"choice_id": choice_id} for variant_id, choice_id in materialization.variant_choices.items()
        },
    }


def _placement_context(
    plan: RuntimeSampleCompilePlan,
    selection: RuntimeSampleSelectionResult,
) -> Mapping[str, Mapping[str, Any]]:
    entries = {entry.entry_id: entry for entry in plan.placements}
    out: dict[str, Mapping[str, Any]] = {}
    for entry_id, selected in selection.placements.items():
        entry = entries[entry_id]
        out[entry_id] = {
            "target": {"type": entry.target_type, "id": entry.target_id},
            "sample_type": entry.sample_type,
            "domain_kind": selected.domain_kind,
            "range_id": selected.choices.get("ranges"),
            "yaw_id": selected.choices.get("yaws"),
            "pose_id": selected.choices.get("poses"),
            "case_id": selected.choices.get("assignments"),
            "choices": dict(selected.choices),
            "params": dict((selected.details.get("assignment") or {}).get("params", {})),
            **({"assignment": selected.details["assignment"]} if "assignment" in selected.details else {}),
        }
    return out


def _inherited_domain(
    *,
    target_type: str,
    target_id: str,
    entry_id: str,
    layout_plan: CanonicalLayoutPlan,
) -> RuntimePlacementDomainPlan:
    if target_type == "node":
        owner = layout_plan.placement_owners.get(target_id)
        if owner != f"node:{target_id}":
            raise error(
                "schema_vnext_runtime_target_not_owner",
                "target node is not a node-level placement owner",
                target=entry_id,
            )
        node = layout_plan.nodes[target_id]
        return _node_domain(node)
    if target_type == "group":
        group = layout_plan.groups.get(target_id)
        if group is None:
            raise error(
                "schema_vnext_unknown_reference", "unknown group runtime target", target=entry_id, ref=target_id
            )
        return _group_domain(group, layout_plan=layout_plan)
    raise error(
        "schema_vnext_runtime_target_type_unsupported",
        "runtime placement target only supports node or group",
        target=entry_id,
        type=target_type,
    )


def _node_domain(node: NodeDomainPlan) -> RuntimePlacementDomainPlan:
    if node.placement_type == "surface":
        return RuntimePlacementDomainPlan(domain_kind="surface", ranges=node.ranges, yaws=node.yaws)
    if node.placement_type == "pose":
        return RuntimePlacementDomainPlan(domain_kind="pose", poses=node.poses)
    if node.placement_type == "volume":
        return RuntimePlacementDomainPlan(domain_kind="volume", yaws=node.yaws, params=dict(node.params))
    raise error(
        "schema_vnext_runtime_target_not_owner", "target node has no runtime placement domain", target=node.node_id
    )


def _group_domain(group: GroupDomainPlan, *, layout_plan: CanonicalLayoutPlan) -> RuntimePlacementDomainPlan:
    if group.group_type in {"region", "scatter"}:
        return RuntimePlacementDomainPlan(
            domain_kind=group.group_type,
            ranges=group.ranges,
            yaws=group.yaws,
            params=dict(group.params),
        )
    if group.group_type == "volume":
        return RuntimePlacementDomainPlan(domain_kind="volume", yaws=group.yaws, params=dict(group.params))
    if group.group_type == "slot":
        return RuntimePlacementDomainPlan(
            domain_kind="slot",
            assignments=group.assignment_ids,
            assignment_cases=group.assignment_cases,
            slot_order=tuple(group.slots),
            node_objects={node_id: layout_plan.nodes[node_id].object_id for node_id in group.nodes},
            params=dict(group.params),
        )
    raise error(
        "schema_vnext_group_type_unsupported",
        "unsupported runtime group type",
        target=group.group_id,
        type=group.group_type,
    )


def _effective_domain(
    inherited: RuntimePlacementDomainPlan,
    sample: Mapping[str, Any],
    *,
    sample_type: str,
    target_type: str,
    entry_id: str,
) -> RuntimePlacementDomainPlan:
    if sample_type in {"inherit", "jitter"}:
        return inherited
    if sample_type != "override":
        raise error(
            "schema_vnext_sample_type_unsupported", "unsupported sample type", target=entry_id, type=sample_type
        )
    if inherited.domain_kind == "slot":
        raise error(
            "schema_vnext_slot_runtime_sample_type_invalid",
            "slot group runtime sampling only supports inherit",
            target=entry_id,
        )
    overrides = _mapping(sample.get("overrides", {}), target=f"placements.{entry_id}.sample.overrides")
    allowed = _override_allowlist(inherited.domain_kind, target_type=target_type)
    forbidden = sorted(set(overrides) - allowed)
    if forbidden:
        raise error(
            "schema_vnext_runtime_override_field_forbidden",
            "override contains unsupported fields",
            target=entry_id,
            fields=forbidden,
        )
    ranges = inherited.ranges
    yaws = inherited.yaws
    poses = inherited.poses
    if "ranges" in overrides:
        ranges = _range_choices(overrides["ranges"], target=f"placements.{entry_id}.sample.overrides.ranges")
    if "yaws" in overrides:
        yaws = _yaw_choices(overrides["yaws"], target=f"placements.{entry_id}.sample.overrides.yaws")
    if "poses" in overrides:
        poses = _pose_choices(overrides["poses"], target=f"placements.{entry_id}.sample.overrides.poses")
    params = {
        **dict(inherited.params),
        **{key: value for key, value in overrides.items() if key not in {"ranges", "yaws", "poses"}},
    }
    return RuntimePlacementDomainPlan(
        domain_kind=inherited.domain_kind,
        ranges=ranges,
        yaws=yaws,
        poses=poses,
        assignments=inherited.assignments,
        assignment_cases=inherited.assignment_cases,
        slot_order=inherited.slot_order,
        node_objects=inherited.node_objects,
        params=params,
    )


def _override_allowlist(domain_kind: str, *, target_type: str) -> set[str]:
    if domain_kind == "surface" and target_type == "node":
        return {"ranges", "yaws"}
    if domain_kind == "pose" and target_type == "node":
        return {"poses"}
    if domain_kind == "region":
        return {
            "ranges",
            "method",
            "sample_method",
            "min_spacing",
            "spacing_mode",
            "yaws",
            "grid_step",
            "grid_snap",
            "align_axis",
            "order_axis",
            "jitter",
            "max_attempts",
        }
    if domain_kind == "scatter":
        return {
            "ranges",
            "method",
            "sample_method",
            "min_spacing",
            "spacing_mode",
            "yaws",
            "grid_step",
            "grid_snap",
            "jitter",
            "max_attempts",
        }
    if domain_kind == "volume" and target_type == "node":
        return {"mode", "yaws"}
    if domain_kind == "volume":
        return {"method", "mode", "yaws", "min_spacing", "max_attempts"}
    return set()


def _runtime_selectors(
    domain: RuntimePlacementDomainPlan,
    *,
    explicit_selectors: Mapping[str, SelectorSpec],
    target: str,
) -> dict[str, SelectorSpec]:
    selectors: dict[str, SelectorSpec] = {}
    for name, candidates in {
        "ranges": tuple(choice.choice_id for choice in domain.ranges),
        "yaws": tuple(choice.choice_id for choice in domain.yaws),
        "poses": tuple(choice.choice_id for choice in domain.poses),
        "assignments": domain.assignments,
    }.items():
        if not candidates:
            continue
        selector = explicit_selectors.get(name) or _authored_weight_selector(domain, name)
        _validate_selector_candidates(selector, set(candidates), target=f"{target}.{name}")
        selectors[name] = selector
    unknown_selectors = sorted(set(explicit_selectors) - set(selectors))
    if unknown_selectors:
        raise error(
            "schema_vnext_runtime_selector_field_unsupported",
            "selector has no candidate domain",
            target=target,
            selectors=unknown_selectors,
        )
    return selectors


def _authored_weight_selector(domain: RuntimePlacementDomainPlan, name: str) -> SelectorSpec:
    if name == "ranges":
        return SelectorSpec(
            selector_type="weighted", weights={choice.choice_id: choice.weight for choice in domain.ranges}
        )
    if name == "yaws":
        return SelectorSpec(
            selector_type="weighted", weights={choice.choice_id: choice.weight for choice in domain.yaws}
        )
    if name == "poses":
        return SelectorSpec(
            selector_type="weighted", weights={choice.choice_id: choice.weight for choice in domain.poses}
        )
    if name == "assignments":
        return SelectorSpec(selector_type="uniform")
    raise error("schema_vnext_runtime_selector_field_unsupported", "unsupported selector field", target=name)


def _validate_selector_candidates(selector: SelectorSpec, candidate_ids: set[str], *, target: str) -> None:
    if selector.selector_type == "weighted":
        unknown = sorted(set(selector.weights) - candidate_ids)
        if unknown:
            raise error(
                "schema_vnext_selector_unknown_candidate",
                "selector weights reference unknown candidates",
                target=target,
                ids=unknown,
            )
        if not any(selector.weights.get(candidate_id, selector.default_weight) > 0.0 for candidate_id in candidate_ids):
            raise error(
                "schema_vnext_selector_weights_all_zero",
                "selector requires at least one positive effective weight",
                target=target,
            )
    elif selector.selector_type == "schedule":
        if selector.mode == "cycle":
            unknown = sorted(set(selector.order) - candidate_ids)
        elif selector.mode == "blocks":
            unknown = sorted({block.candidate_id for block in selector.blocks} - candidate_ids)
        elif selector.mode == "counts":
            unknown = sorted(set(selector.counts) - candidate_ids)
        else:
            unknown = []
        if unknown:
            raise error(
                "schema_vnext_selector_unknown_candidate",
                "schedule references unknown candidates",
                target=target,
                ids=unknown,
            )


def _jitter(
    sample: Mapping[str, Any],
    *,
    domain_kind: str,
    sample_type: str,
    target: str,
) -> Mapping[str, tuple[float, float]]:
    if sample_type != "jitter":
        return {}
    forbidden = sorted({"base", "overrides"}.intersection(sample))
    if forbidden:
        raise error(
            "schema_vnext_runtime_jitter_field_forbidden",
            "jitter does not support base or overrides",
            target=target,
            fields=forbidden,
        )
    if domain_kind not in {"surface", "region", "scatter"}:
        raise error(
            "schema_vnext_runtime_jitter_domain_unsupported",
            "jitter is only supported for surface/region/scatter domains",
            target=target,
            domain_kind=domain_kind,
        )
    out: dict[str, tuple[float, float]] = {}
    for name in ("dx", "dy", "yaw_delta"):
        if name in sample:
            out[name] = _scalar_or_pair(sample[name], target=f"placements.{target}.sample.{name}")
    return out


def _candidate_ids(domain: RuntimePlacementDomainPlan, selector_name: str) -> tuple[str, ...]:
    if selector_name == "ranges":
        return tuple(choice.choice_id for choice in domain.ranges)
    if selector_name == "yaws":
        return tuple(choice.choice_id for choice in domain.yaws)
    if selector_name == "poses":
        return tuple(choice.choice_id for choice in domain.poses)
    if selector_name == "assignments":
        return domain.assignments
    raise error("schema_vnext_runtime_selector_field_unsupported", "unsupported selector field", target=selector_name)


def _compile_assets(task: VNextTaskSpec, *, scene: VNextSceneSpec) -> tuple[RuntimeAssetPlanEntry, ...]:
    entries: list[RuntimeAssetPlanEntry] = []
    occupied_objects: dict[str, str] = {}
    for spec in task.runtime_sample.assets:
        if spec.target_type != "node":
            raise error(
                "schema_vnext_runtime_asset_target_type_unsupported",
                "runtime asset target only supports node",
                target=spec.entry_id,
                type=spec.target_type,
            )
        node = scene.layout.nodes.get(spec.target_id)
        if node is None:
            raise error(
                "schema_vnext_unknown_reference",
                "unknown runtime asset node target",
                target=spec.entry_id,
                ref=spec.target_id,
            )
        # Instance nodes (content_1) resolve to the parent pool object (content).
        obj = scene.object_spec_for_id(node.object_id)
        if obj is None:
            raise error(
                "schema_vnext_runtime_asset_target_unknown_object",
                "runtime asset target references unknown object",
                target=spec.entry_id,
                object=node.object_id,
            )
        pool_object_id = obj.object_id
        if obj.asset.selector_type != "pool":
            raise error(
                "schema_vnext_runtime_asset_target_not_pool",
                "runtime asset target object must use asset.type: pool",
                target=spec.entry_id,
                object=pool_object_id,
            )
        if pool_object_id in occupied_objects:
            raise error(
                "schema_vnext_runtime_asset_target_conflict",
                "multiple runtime asset entries target the same object",
                target=spec.entry_id,
                object=pool_object_id,
                entries=[occupied_objects[pool_object_id], spec.entry_id],
            )
        candidates = tuple(
            RuntimeAssetCandidatePlan(candidate_id=candidate.candidate_id, assets=tuple(candidate.assets))
            for candidate in obj.asset.candidates
        )
        _validate_selector_candidates(
            spec.selector, {candidate.candidate_id for candidate in candidates}, target=spec.entry_id
        )
        occupied_objects[pool_object_id] = spec.entry_id
        # One authored asset entry against any instance expands to all concrete
        # ids (content_1..N) so runtime swap targets real scene object names.
        # Couple their selectors with shared_id so one episode picks one concrete
        # asset (e.g. beans_a) for every instance rather than mixing per slot.
        concrete_ids = obj.materialized_ids
        selector = spec.selector
        if len(concrete_ids) > 1 and not str(getattr(selector, "shared_id", "") or "").strip():
            selector = replace(selector, shared_id=f"instance_pool:{pool_object_id}")
        for concrete_id in concrete_ids:
            entry_id = (
                spec.entry_id
                if len(concrete_ids) == 1
                else (f"{concrete_id}_asset" if spec.entry_id.endswith("_asset") else f"{spec.entry_id}:{concrete_id}")
            )
            entries.append(
                RuntimeAssetPlanEntry(
                    entry_id=entry_id,
                    target_type=spec.target_type,
                    target_id=concrete_id if obj.instance_ids else spec.target_id,
                    object_id=concrete_id,
                    selector=selector,
                    candidates=candidates,
                    cache_size=int(spec.cache_size),
                    slots=_cache_slot_names(concrete_id, int(spec.cache_size)),
                )
            )
    return tuple(entries)


def _cache_slot_names(object_id: str, cache_size: int) -> tuple[str, ...]:
    return tuple(str(object_id) if index == 0 else f"{object_id}__slot_{index}" for index in range(int(cache_size)))


def _assignment_detail(entry: RuntimePlacementPlanEntry, assignment_id: str) -> Mapping[str, Any]:
    case = entry.domain.assignment_cases.get(assignment_id)
    if case is None:
        raise error(
            "schema_vnext_unknown_reference",
            "unknown slot assignment selection",
            target=entry.entry_id,
            ref=assignment_id,
        )
    slot_order = [slot_id for slot_id in entry.domain.slot_order if slot_id in set(case.bindings.values())]
    nodes_by_slot = {slot_id: node_id for node_id, slot_id in case.bindings.items()}
    nodes_by_slot_order = [nodes_by_slot[slot_id] for slot_id in slot_order]
    return {
        "id": assignment_id,
        "bindings": dict(case.bindings),
        "params": dict(case.params),
        "slot_order": slot_order,
        "nodes_by_slot_order": nodes_by_slot_order,
        "objects_by_slot_order": [entry.domain.node_objects[node_id] for node_id in nodes_by_slot_order],
    }


def _validate_global_scene_variant_selection(runtime_payload: Mapping[str, Any], *, scene: VNextSceneSpec) -> None:
    global_payload = _mapping(runtime_payload.get("global", {}), target="task.layout.runtime_sample.global")
    selection = global_payload.get("scene_variant_selection")
    if selection in (None, {}):
        return
    payload = _mapping(selection, target="task.layout.runtime_sample.global.scene_variant_selection")
    source = str(payload.get("source") or "").strip()
    if not source:
        raise error(
            "schema_vnext_scene_variant_selection_source_missing",
            "scene_variant_selection requires source",
            target="scene_variant_selection",
        )
    materialization = {variant.variant_id for variant in scene.variants.materialization}
    assignments = {variant.variant_id for variant in scene.variants.assignments}
    if source in materialization:
        raise error(
            "schema_vnext_runtime_materialization_variant_unsupported",
            "asset/background materialization variants cannot be selected at runtime",
            target="scene_variant_selection",
            source=source,
        )
    if source in assignments:
        raise error(
            "schema_vnext_runtime_assignment_variant_unsupported",
            "slot assignment runtime selection must target the slot group placement entry",
            target="scene_variant_selection",
            source=source,
        )
    raise error(
        "schema_vnext_unknown_reference",
        "unknown scene variant selection source",
        target="scene_variant_selection",
        ref=source,
    )


def _compile_lighting(value: Any, *, scene: VNextSceneSpec) -> tuple[RuntimeLightingPlanEntry, ...]:
    payload = _mapping(value or {}, target="task.layout.runtime_sample.global.lighting")
    if not payload:
        return ()
    dome = _mapping(payload.get("dome", {}), target="task.layout.runtime_sample.global.lighting.dome")
    tabletop_rect = _mapping(
        payload.get("tabletop_rect", {}), target="task.layout.runtime_sample.global.lighting.tabletop_rect"
    )
    scene_lighting = _mapping(scene.lighting or {}, target="scene.lighting")
    scene_dome = _mapping(scene_lighting.get("dome", {}), target="scene.lighting.dome")
    scene_tabletop_rect = _mapping(scene_lighting.get("tabletop_rect", {}), target="scene.lighting.tabletop_rect")
    entries: list[RuntimeLightingPlanEntry] = []
    texture_pool: Mapping[str, Mapping[str, Any]] | None = None
    profile: Mapping[str, Any] | None = None
    if "texture" in dome:
        scene_texture = _mapping(scene_dome.get("texture"), target="scene.lighting.dome.texture")
        texture_pool = _texture_pool(scene_texture)
        texture = _mapping(dome["texture"], target="task.layout.runtime_sample.global.lighting.dome.texture")
        selector = SelectorSpec.from_dict(
            texture.get("selector", {"type": "uniform"}),
            target="task.layout.runtime_sample.global.lighting.dome.texture.selector",
            candidate_ids=set(texture_pool),
        )
        entries.append(
            RuntimeLightingPlanEntry(channel="dome.texture", kind="texture", selector=selector, candidates=texture_pool)
        )
    for channel in ("intensity", "rotation", "color_temperature"):
        if channel not in dome:
            continue
        scalar = _mapping(dome[channel], target=f"task.layout.runtime_sample.global.lighting.dome.{channel}")
        if "source" in scalar:
            if channel != "intensity":
                raise error(
                    "schema_vnext_lighting_recommendation_channel_unsupported",
                    "photometric recommendations are only supported for dome intensity",
                    target=f"task.layout.runtime_sample.global.lighting.dome.{channel}",
                )
            profile = profile or _load_scene_photometric_profile(scene_lighting)
            if texture_pool is None:
                raise error(
                    "schema_vnext_lighting_recommendation_texture_required",
                    "recommended dome intensity requires runtime dome texture selection",
                    target="task.layout.runtime_sample.global.lighting.dome.intensity",
                )
            entries.append(_compile_dome_intensity_recommendation(scalar, texture_pool=texture_pool, profile=profile))
            continue
        value_range = _lighting_scalar_range(
            scalar,
            scene_scalar=_mapping(scene_dome.get(channel, {}), target=f"scene.lighting.dome.{channel}"),
            target=f"task.layout.runtime_sample.global.lighting.dome.{channel}",
        )
        entries.append(RuntimeLightingPlanEntry(channel=f"dome.{channel}", kind="scalar", value_range=value_range))
    if "enabled" in tabletop_rect:
        entries.append(
            _compile_tabletop_rect_enabled(
                tabletop_rect["enabled"],
                target="task.layout.runtime_sample.global.lighting.tabletop_rect.enabled",
            )
        )
    if "intensity" in tabletop_rect:
        scalar = _mapping(
            tabletop_rect["intensity"],
            target="task.layout.runtime_sample.global.lighting.tabletop_rect.intensity",
        )
        if "source" in scalar:
            profile = profile or _load_scene_photometric_profile(scene_lighting)
            entries.append(_compile_tabletop_rect_recommendation(scalar, profile=profile))
        else:
            value_range = _lighting_scalar_range(
                scalar,
                scene_scalar=_mapping(
                    scene_tabletop_rect.get("intensity", {}), target="scene.lighting.tabletop_rect.intensity"
                ),
                target="task.layout.runtime_sample.global.lighting.tabletop_rect.intensity",
            )
            entries.append(
                RuntimeLightingPlanEntry(channel="tabletop_rect.intensity", kind="scalar", value_range=value_range)
            )
    return tuple(entries)


def _compile_tabletop_rect_enabled(value: Any, *, target: str) -> RuntimeLightingPlanEntry:
    """Compile a constant or Bernoulli on/off switch for the desk RectLight."""

    if isinstance(value, bool):
        return RuntimeLightingPlanEntry(
            channel="tabletop_rect.enabled",
            kind="boolean",
            params={"value": bool(value)},
        )
    payload = _mapping(value, target=target)
    unsupported = sorted(set(payload) - {"probability"})
    if unsupported:
        raise error(
            "schema_vnext_lighting_enabled_field_unsupported",
            "tabletop_rect.enabled only supports probability",
            target=target,
            fields=unsupported,
        )
    if "probability" not in payload:
        raise error(
            "schema_vnext_lighting_enabled_probability_missing",
            "tabletop_rect.enabled requires probability in [0, 1]",
            target=f"{target}.probability",
        )
    probability = _finite_lighting_number(payload["probability"], target=f"{target}.probability")
    if probability < 0.0 or probability > 1.0:
        raise error(
            "schema_vnext_lighting_enabled_probability_invalid",
            "tabletop_rect.enabled probability must be in [0, 1]",
            target=f"{target}.probability",
        )
    return RuntimeLightingPlanEntry(
        channel="tabletop_rect.enabled",
        kind="bernoulli",
        params={"probability": float(probability)},
    )


def _load_scene_photometric_profile(scene_lighting: Mapping[str, Any]) -> Mapping[str, Any]:
    profile = _mapping(scene_lighting.get("photometric_profile", {}), target="scene.lighting.photometric_profile")
    manifest_uri = str(profile.get("manifest") or "").strip()
    if not manifest_uri:
        raise error(
            "schema_vnext_lighting_profile_manifest_missing",
            "recommended lighting requires scene.lighting.photometric_profile.manifest",
            target="scene.lighting.photometric_profile.manifest",
        )
    return load_lighting_photometric_profile(manifest_uri)


def _compile_dome_intensity_recommendation(
    value: Mapping[str, Any],
    *,
    texture_pool: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, Any],
) -> RuntimeLightingPlanEntry:
    target = "task.layout.runtime_sample.global.lighting.dome.intensity"
    _validate_recommendation_fields(value, allowed={"source", "mode", "ev_offset"}, target=target)
    contract = _mapping(profile.get("lighting_contract"), target="scene.lighting.photometric_profile")
    recommendation = _mapping(contract.get("recommendation"), target="scene.lighting.photometric_profile")
    mode = str(value.get("mode") or recommendation.get("default_mode") or "").strip()
    if mode not in {"range", "nominal"}:
        raise error(
            "schema_vnext_lighting_recommendation_mode_invalid",
            "recommended dome intensity mode must be range or nominal",
            target=f"{target}.mode",
        )
    ev_offset = _finite_lighting_number(
        value.get("ev_offset", recommendation.get("default_ev_offset", 0.0)), target=f"{target}.ev_offset"
    )
    multiplier = _lighting_ev_multiplier(ev_offset, target=f"{target}.ev_offset")
    profile_candidates = _matching_photometric_candidates(texture_pool, profile=profile)
    if any(
        not math.isfinite(float(candidate["dome_intensity"]["range"][1]) * multiplier)
        for candidate in profile_candidates.values()
    ):
        raise error(
            "schema_vnext_lighting_ev_offset_invalid",
            "lighting ev_offset produces an invalid candidate intensity",
            target=f"{target}.ev_offset",
        )
    dome_contract = _mapping(contract.get("dome"), target="scene.lighting.photometric_profile")
    return RuntimeLightingPlanEntry(
        channel="dome.intensity",
        kind="candidate_recommendation",
        candidates=profile_candidates,
        source_channel="dome.texture",
        params={
            "profile_id": str(profile.get("profile_id") or ""),
            "mode": mode,
            "ev_offset": ev_offset,
            "dome_exposure": float(dome_contract["exposure"]),
        },
    )


def _compile_tabletop_rect_recommendation(
    value: Mapping[str, Any], *, profile: Mapping[str, Any]
) -> RuntimeLightingPlanEntry:
    target = "task.layout.runtime_sample.global.lighting.tabletop_rect.intensity"
    _validate_recommendation_fields(value, allowed={"source"}, target=target)
    contract = _mapping(profile.get("lighting_contract"), target="scene.lighting.photometric_profile")
    rect = _mapping(contract.get("tabletop_rect"), target="scene.lighting.photometric_profile")
    return RuntimeLightingPlanEntry(
        channel="tabletop_rect.intensity",
        kind="profile_scalar",
        params={
            "profile_id": str(profile.get("profile_id") or ""),
            "source": "recommendation",
            "role": str(rect.get("role") or ""),
            "value": float(rect["intensity"]),
        },
    )


def _validate_recommendation_fields(value: Mapping[str, Any], *, allowed: set[str], target: str) -> None:
    source = str(value.get("source") or "").strip()
    if source != "recommendation":
        raise error(
            "schema_vnext_lighting_recommendation_source_invalid",
            "lighting recommendation source must be recommendation",
            target=f"{target}.source",
        )
    forbidden = sorted(set(value) - allowed)
    if forbidden:
        raise error(
            "schema_vnext_lighting_recommendation_field_unsupported",
            "recommended lighting contains unsupported fields",
            target=target,
            fields=forbidden,
        )


def _matching_photometric_candidates(
    texture_pool: Mapping[str, Mapping[str, Any]], *, profile: Mapping[str, Any]
) -> Mapping[str, Mapping[str, Any]]:
    profile_candidates = _mapping(profile.get("candidates"), target="scene.lighting.photometric_profile.candidates")
    out: dict[str, Mapping[str, Any]] = {}
    for candidate_id, texture in texture_pool.items():
        candidate = profile_candidates.get(candidate_id)
        if not isinstance(candidate, Mapping):
            raise error(
                "schema_vnext_lighting_profile_candidate_missing",
                "photometric profile must cover every runtime texture candidate",
                target="scene.lighting.photometric_profile.candidates",
                ref=candidate_id,
            )
        texture_uri = str(texture.get("uri") or "").strip()
        profile_uri = str(candidate.get("uri") or "").strip()
        if texture_uri != profile_uri:
            raise error(
                "schema_vnext_lighting_profile_candidate_uri_mismatch",
                "photometric profile candidate URI must match the runtime texture pool",
                target="scene.lighting.photometric_profile.candidates",
                ref=candidate_id,
            )
        out[candidate_id] = dict(candidate)
    return out


def _finite_lighting_number(value: Any, *, target: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise error(
            "schema_vnext_lighting_number_invalid",
            "lighting value must be a finite number",
            target=target,
        ) from exc
    if not math.isfinite(parsed):
        raise error(
            "schema_vnext_lighting_number_invalid",
            "lighting value must be a finite number",
            target=target,
        )
    return parsed


def _lighting_ev_multiplier(ev_offset: float, *, target: str) -> float:
    try:
        multiplier = math.pow(2.0, ev_offset)
    except OverflowError as exc:
        raise error(
            "schema_vnext_lighting_ev_offset_invalid",
            "lighting ev_offset produces an invalid multiplier",
            target=target,
        ) from exc
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise error(
            "schema_vnext_lighting_ev_offset_invalid",
            "lighting ev_offset produces an invalid multiplier",
            target=target,
        )
    return multiplier


def _texture_pool(scene_texture: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    has_pool = "pool" in scene_texture
    manifest_uri = str(scene_texture.get("manifest") or "").strip()
    if has_pool and manifest_uri:
        raise error(
            "schema_vnext_lighting_texture_source_conflict",
            "lighting texture supports pool or manifest, not both",
            target="scene.lighting.dome.texture",
        )
    if manifest_uri:
        pool: dict[str, Mapping[str, Any]] = load_lighting_texture_manifest(
            manifest_uri,
            target="scene.lighting.dome.texture.manifest",
        )
    else:
        pool_items = _list(scene_texture.get("pool"), target="scene.lighting.dome.texture.pool")
        pool = {}
        for index, raw_item in enumerate(pool_items):
            item = _mapping(raw_item, target=f"scene.lighting.dome.texture.pool[{index}]")
            candidate_id = _non_empty_str(item.get("id"), target=f"scene.lighting.dome.texture.pool[{index}].id")
            if candidate_id in pool:
                raise error(
                    "schema_vnext_duplicate_id",
                    "lighting texture candidate ids must be unique",
                    target="scene.lighting.dome.texture.pool",
                )
            if not str(item.get("uri") or "").strip():
                raise error(
                    "schema_vnext_lighting_texture_uri_missing",
                    "lighting texture candidate requires uri",
                    target=candidate_id,
                )
            pool[candidate_id] = dict(item)
    if not pool:
        raise error(
            "schema_vnext_lighting_texture_pool_empty",
            "scene lighting texture pool is required",
            target="scene.lighting.dome.texture.pool",
        )
    default = str(scene_texture.get("default") or "").strip()
    if default and default not in pool:
        raise error(
            "schema_vnext_lighting_default_unknown",
            "lighting texture default must reference pool",
            target="scene.lighting.dome.texture.default",
            ref=default,
        )
    return pool


def _lighting_scalar_range(
    value: Mapping[str, Any], *, scene_scalar: Mapping[str, Any], target: str
) -> tuple[float, float]:
    if "range" in value and "value" in value:
        raise error(
            "schema_vnext_lighting_scalar_conflict", "lighting scalar supports range or value, not both", target=target
        )
    if "range" in value:
        parsed = _scalar_or_pair(value["range"], target=f"{target}.range")
    elif "value" in value:
        parsed = _scalar_or_pair(value["value"], target=f"{target}.value")
    else:
        raise error("schema_vnext_lighting_scalar_missing", "lighting scalar requires range or value", target=target)
    if "range" in scene_scalar:
        allowed = _scalar_or_pair(scene_scalar["range"], target=f"{target}.scene_range")
        if parsed[0] < allowed[0] or parsed[1] > allowed[1]:
            raise error(
                "schema_vnext_lighting_scalar_out_of_range",
                "task lighting scalar cannot expand scene range",
                target=target,
            )
    return parsed


def _compile_materials(value: Any, *, scene: VNextSceneSpec) -> tuple[RuntimeMaterialPlanEntry, ...]:
    entries: list[RuntimeMaterialPlanEntry] = []
    occupied: set[tuple[str, str]] = set()
    occupied_surfaces: set[tuple[str, str]] = set()
    valid_objects = _runtime_object_ids(scene)
    for index, raw_item in enumerate(_list(value or [], target="task.layout.runtime_sample.targets.materials")):
        item = _mapping(raw_item, target=f"task.layout.runtime_sample.targets.materials[{index}]")
        entry_id = _non_empty_str(item.get("id"), target=f"task.layout.runtime_sample.targets.materials[{index}].id")
        target = _mapping(item.get("target"), target=f"materials.{entry_id}.target")
        if str(target.get("type") or "").strip() == "object_material":
            entry = _compile_scene_material_variant(
                entry_id=entry_id,
                item=item,
                target=target,
                scene=scene,
                occupied=occupied,
            )
            entries.append(entry)
            continue
        forbidden_target = sorted(set(target) - {"objects", "surface_material", "material_slots"})
        if forbidden_target:
            raise error(
                "schema_vnext_material_target_field_forbidden",
                "unsupported material target fields",
                target=entry_id,
                fields=forbidden_target,
            )
        has_objects = "objects" in target
        has_surface_material = "surface_material" in target
        if has_objects == has_surface_material:
            raise error(
                "schema_vnext_material_target_invalid",
                "material target requires exactly one of objects or surface_material",
                target=entry_id,
            )
        material_slots = _material_slots(
            target.get("material_slots", "auto"), target=f"materials.{entry_id}.target.material_slots"
        )
        policy = _mapping(item.get("policy"), target=f"materials.{entry_id}.policy")
        policy_type = _non_empty_str(policy.get("type"), target=f"materials.{entry_id}.policy.type")
        if policy_type not in {"material_template", "material_local_adjustment"}:
            raise error(
                "schema_vnext_material_policy_type_unsupported",
                "unsupported material policy type",
                target=entry_id,
                type=policy_type,
            )
        if "library" in policy or "channel_matrix" in policy:
            raise error(
                "schema_vnext_material_policy_field_forbidden",
                "material policy does not expose library/channel_matrix",
                target=entry_id,
            )
        requested_candidates, inline_candidate_payloads = _policy_candidates_with_payloads(
            policy, target=f"materials.{entry_id}.policy.candidates"
        )
        objects: tuple[str, ...] = ()
        prim_paths: tuple[str, ...] = ()
        surface_material: str | None = None
        candidate_payloads: Mapping[str, Mapping[str, Any]] = {}
        if has_surface_material:
            if policy_type != "material_template":
                raise error(
                    "schema_vnext_surface_material_policy_type_unsupported",
                    "surface material targets only support material_template policy",
                    target=entry_id,
                    type=policy_type,
                )
            surface_material = _non_empty_str(
                target.get("surface_material"),
                target=f"materials.{entry_id}.target.surface_material",
            )
            surface_id = _background_surface_ref_id(
                surface_material,
                target=f"materials.{entry_id}.target.surface_material",
            )
            pool = scene.background.surface_material_pool.get(surface_id)
            if pool is None:
                raise error(
                    "schema_vnext_surface_material_pool_missing",
                    "scene background does not define a material pool for the requested surface",
                    target=entry_id,
                    surface=surface_material,
                )
            manifest_templates = load_material_template_manifest(
                pool.manifest,
                target=f"background.surface_material_pool.{surface_id}.manifest",
            )
            pool_candidates = pool.candidates or tuple(manifest_templates)
            unknown_pool_candidates = sorted(set(pool_candidates) - set(manifest_templates))
            if unknown_pool_candidates:
                raise error(
                    "schema_vnext_surface_material_pool_unknown_candidate",
                    "surface material pool references candidates not present in the manifest",
                    target=f"background.surface_material_pool.{surface_id}.candidates",
                    candidates=unknown_pool_candidates,
                )
            candidates = requested_candidates or pool_candidates
            unknown_requested = sorted(set(candidates) - set(pool_candidates))
            if unknown_requested:
                raise error(
                    "schema_vnext_surface_material_candidate_not_allowed",
                    "task material candidates must be a subset of the scene surface material pool",
                    target=entry_id,
                    candidates=unknown_requested,
                )
            for slot in _material_slot_keys(material_slots):
                key = (surface_material, slot)
                if key in occupied_surfaces:
                    raise error(
                        "schema_vnext_material_target_conflict",
                        "multiple material entries target the same surface material and slot",
                        target=entry_id,
                        surface=surface_material,
                        slot=slot,
                    )
                occupied_surfaces.add(key)
            prim_paths = _background_surface_material_prim_paths(scene, surface_id=surface_id)
            candidate_payloads = {candidate: manifest_templates[candidate] for candidate in candidates}
            apply_options = {
                "clone_shared_materials": True,
                **dict(item.get("apply") or {}),
            }
        else:
            objects = tuple(
                _non_empty_str(obj, target=f"materials.{entry_id}.target.objects[]")
                for obj in _list(target.get("objects"), target=f"materials.{entry_id}.target.objects")
            )
            unknown = sorted(set(objects) - valid_objects)
            if unknown:
                raise error(
                    "schema_vnext_material_target_unknown_object",
                    "material target references unknown objects",
                    target=entry_id,
                    objects=unknown,
                )
            for obj in objects:
                for slot in _material_slot_keys(material_slots):
                    key = (obj, slot)
                    if key in occupied:
                        raise error(
                            "schema_vnext_material_target_conflict",
                            "multiple material entries target the same object and slot",
                            target=entry_id,
                            object=obj,
                            slot=slot,
                        )
                    occupied.add(key)
            candidates = requested_candidates
            candidate_payloads = inline_candidate_payloads
            apply_options = dict(item.get("apply") or {})
        if not candidates:
            raise error(
                "schema_vnext_material_candidates_empty", "material policy requires candidates", target=entry_id
            )
        if has_surface_material and inline_candidate_payloads:
            raise error(
                "schema_vnext_surface_material_inline_candidates_forbidden",
                "surface material candidates must reference the scene material pool, not inline channel payloads",
                target=entry_id,
            )
        selector = SelectorSpec.from_dict(
            item.get("selector", {"type": "uniform"}),
            target=f"materials.{entry_id}.selector",
            candidate_ids=set(candidates),
        )
        entries.append(
            RuntimeMaterialPlanEntry(
                entry_id=entry_id,
                target_type="surface_material" if has_surface_material else "objects",
                policy_type=policy_type,
                candidates=candidates,
                selector=selector,
                objects=objects,
                prim_paths=prim_paths,
                surface_material=surface_material,
                material_slots=material_slots,
                candidate_payloads=candidate_payloads,
                apply=apply_options,
            )
        )
    return tuple(entries)


def _compile_scene_material_variant(
    *,
    entry_id: str,
    item: Mapping[str, Any],
    target: Mapping[str, Any],
    scene: VNextSceneSpec,
    occupied: set[tuple[str, str]],
) -> RuntimeMaterialPlanEntry:
    forbidden_target = sorted(set(target) - {"type", "object", "material"})
    if forbidden_target:
        raise error(
            "schema_vnext_material_target_field_forbidden",
            "unsupported object_material target fields",
            target=entry_id,
            fields=forbidden_target,
        )
    object_id = _non_empty_str(target.get("object"), target=f"materials.{entry_id}.target.object")
    material_id = _non_empty_str(target.get("material"), target=f"materials.{entry_id}.target.material")
    obj = scene.objects.get(object_id)
    if obj is None:
        raise error(
            "schema_vnext_material_target_unknown_object",
            "material target references unknown object",
            target=entry_id,
            objects=[object_id],
        )
    variant = obj.material_variants.get(material_id)
    if variant is None:
        raise error(
            "schema_vnext_material_variant_unknown",
            "material target references unknown scene material variant",
            target=entry_id,
            object=object_id,
            material=material_id,
        )
    material_slots = (variant.path,)
    for materialized_id in obj.materialized_ids:
        key = (materialized_id, variant.path)
        if key in occupied:
            raise error(
                "schema_vnext_material_target_conflict",
                "multiple material entries target the same object and slot",
                target=entry_id,
                object=materialized_id,
                slot=variant.path,
            )
        occupied.add(key)
    candidates, candidate_templates = _scene_material_candidate_templates(
        object_id=object_id,
        obj=obj,
        material_id=material_id,
        variant=variant,
        target=entry_id,
    )
    selector_payload = item.get("selector") or variant.selector or {"type": "uniform"}
    selector = SelectorSpec.from_dict(
        selector_payload,
        target=f"materials.{entry_id}.selector",
        candidate_ids=set(candidates),
    )
    return RuntimeMaterialPlanEntry(
        entry_id=entry_id,
        target_type="objects",
        objects=obj.materialized_ids,
        material_slots=material_slots,
        policy_type="material_template",
        candidates=candidates,
        selector=selector,
        candidate_templates=candidate_templates,
        apply=dict(item.get("apply") or {}),
    )


def _scene_material_candidate_templates(
    *,
    object_id: str,
    obj: Any,
    material_id: str,
    variant: Any,
    target: str,
) -> tuple[tuple[str, ...], Mapping[str, Mapping[str, Any]]]:
    candidates: list[str] = []
    templates: dict[str, Mapping[str, Any]] = {}
    seen: set[str] = set()
    object_asset_id = obj.asset.asset_id if obj.asset.selector_type == "asset_id" else None
    for index, texture_ref in enumerate(variant.candidates):
        candidate_id = _texture_candidate_id(texture_ref, index=index)
        while candidate_id in seen:
            candidate_id = f"{candidate_id}_{index}"
        seen.add(candidate_id)
        candidates.append(candidate_id)
        templates[candidate_id] = {
            "id": candidate_id,
            "object": object_id,
            "object_asset_id": object_asset_id,
            "material": material_id,
            "path": variant.path,
            "channel": variant.channel,
            "texture": texture_ref,
        }
    if not candidates:
        raise error("schema_vnext_material_candidates_empty", "material variant requires candidates", target=target)
    return tuple(candidates), templates


def _texture_candidate_id(texture_ref: str, *, index: int) -> str:
    raw = str(texture_ref).strip().rsplit("/", 1)[-1].rsplit(".", 1)[0]
    candidate = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_")
    return candidate or f"candidate_{index}"


def _runtime_object_ids(scene: VNextSceneSpec) -> set[str]:
    ids: set[str] = set()
    for obj in scene.objects.values():
        ids.update(obj.materialized_ids)
        # Pool cache bodies are concrete scene assets (object / object__slot_i)
        # and are valid material targets for shape↔color pairing (nut_on_screw).
        if obj.asset.selector_type == "pool":
            cache_size = int(obj.asset.cache_size or 0)
            if cache_size > 0:
                ids.update(_cache_slot_names(obj.object_id, cache_size))
    return ids


def _material_slots(value: Any, *, target: str) -> str | tuple[str, ...]:
    if isinstance(value, str) and value in {"auto", "all"}:
        return value
    slots = tuple(_non_empty_str(item, target=f"{target}[]") for item in _list(value, target=target))
    if not slots:
        raise error(
            "schema_vnext_material_slots_empty", "material_slots must be auto, all, or a non-empty list", target=target
        )
    return slots


def _material_slot_keys(value: str | tuple[str, ...]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else value


def _policy_candidates(policy: Mapping[str, Any], *, target: str) -> tuple[str, ...]:
    candidates, _payloads = _policy_candidates_with_payloads(policy, target=target)
    return candidates


def _policy_candidates_with_payloads(
    policy: Mapping[str, Any], *, target: str
) -> tuple[tuple[str, ...], Mapping[str, Mapping[str, Any]]]:
    """Parse material policy candidates.

    Accepts either bare ids (``red_plastic``) or inline templates::

        - id: red
          channels:
            base_color: {source: constant, value: [1.0, 0.0, 0.0]}
    """
    if "candidates" not in policy:
        return (), {}
    ids: list[str] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(_list(policy.get("candidates"), target=target)):
        item_target = f"{target}[{index}]"
        if isinstance(raw, str) or not isinstance(raw, Mapping):
            ids.append(_non_empty_str(raw, target=item_target))
            continue
        candidate_id = _non_empty_str(raw.get("id"), target=f"{item_target}.id")
        channels = raw.get("channels")
        if not isinstance(channels, Mapping) or not channels:
            raise error(
                "schema_vnext_material_candidate_channels_invalid",
                "inline material candidate requires a non-empty channels mapping",
                target=item_target,
            )
        ids.append(candidate_id)
        payloads[candidate_id] = {
            "id": candidate_id,
            "tags": ["schema_vnext", candidate_id],
            "channels": {
                str(key): dict(value) if isinstance(value, Mapping) else value for key, value in channels.items()
            },
        }
    duplicates = sorted({candidate for candidate in ids if ids.count(candidate) > 1})
    if duplicates:
        raise error("schema_vnext_duplicate_id", "ids must be unique", target=target, ids=duplicates)
    return tuple(ids), payloads


def _background_surface_ref_id(value: str, *, target: str) -> str:
    prefix = "background."
    if not value.startswith(prefix) or value == prefix:
        raise error(
            "schema_vnext_surface_material_ref_unsupported",
            "surface material targets must use background.<surface_id>",
            target=target,
            ref=value,
        )
    return value[len(prefix) :]


def _background_surface_material_prim_paths(scene: VNextSceneSpec, *, surface_id: str) -> tuple[str, ...]:
    if surface_id != "tabletop":
        raise error(
            "schema_vnext_surface_material_runtime_target_unsupported",
            "only background.tabletop has a runtime material target contract",
            target=f"background.{surface_id}",
        )
    background_id = scene.background.background_id or "background"
    # Spawned backgrounds use asset_anchor_path(): ``{ENV_REGEX_NS}/_<id>``.
    return (
        f"/World/{{ENV_REGEX_NS}}/_{background_id}/SchemaVNextInsertedTable/Aligned/wooden_table_02/tabletop_shell",
    )


def _sample_lighting(
    plan: RuntimeSampleCompilePlan,
    *,
    rng: random.Random,
    state: SelectorRuntimeState,
    env_id: int,
) -> Mapping[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for entry in plan.lighting:
        if entry.kind == "texture":
            result = select_candidate(
                tuple(entry.candidates),
                entry.selector or SelectorSpec(selector_type="uniform"),
                selector_id=f"lighting.{entry.channel}",
                runtime_sample_id=plan.runtime_sample_id,
                env_id=env_id,
                rng=rng,
                state=state,
            )
            out[entry.channel] = {"candidate_id": result.selected_id, **dict(entry.candidates[result.selected_id])}
        elif entry.kind == "scalar":
            assert entry.value_range is not None
            low, high = entry.value_range
            value = low if low == high else rng.uniform(low, high)
            out[entry.channel] = {"value": value}
        elif entry.kind == "candidate_recommendation":
            source = out.get(str(entry.source_channel or ""))
            candidate_id = str((source or {}).get("candidate_id") or "")
            if candidate_id not in entry.candidates:
                raise RuntimeError(
                    f"schema_vnext_lighting_recommendation_source_missing: {entry.channel}:{candidate_id}"
                )
            sampled = sample_recommended_dome_intensity(
                entry.candidates[candidate_id],
                mode=str(entry.params.get("mode") or ""),
                ev_offset=float(entry.params.get("ev_offset", 0.0)),
                rng=rng,
            )
            out[entry.channel] = {
                **sampled,
                "profile_id": str(entry.params.get("profile_id") or ""),
                "dome_exposure": float(entry.params["dome_exposure"]),
            }
        elif entry.kind == "profile_scalar":
            out[entry.channel] = dict(entry.params)
        elif entry.kind == "boolean":
            out[entry.channel] = {"value": bool(entry.params.get("value"))}
        elif entry.kind == "bernoulli":
            probability = float(entry.params.get("probability", 0.5))
            out[entry.channel] = {
                "value": bool(rng.random() < probability),
                "probability": probability,
            }
        else:
            raise RuntimeError(f"schema_vnext_lighting_entry_kind_unsupported: {entry.channel}:{entry.kind}")
    return out


def sample_recommended_dome_intensity(
    candidate: Mapping[str, Any], *, mode: str, ev_offset: float, rng: random.Random
) -> dict[str, Any]:
    candidate_id = str(candidate.get("id") or "").strip()
    intensity = candidate.get("dome_intensity")
    if not candidate_id or not isinstance(intensity, Mapping):
        raise RuntimeError("schema_vnext_lighting_recommendation_candidate_invalid")
    nominal = float(intensity["nominal"])
    value_range = intensity["range"]
    low, high = float(value_range[0]), float(value_range[1])
    distribution = str(intensity.get("distribution") or "")
    if mode == "nominal":
        base_value = nominal
    elif mode == "range" and distribution == "log_uniform":
        base_value = low if low == high else math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        raise RuntimeError(f"schema_vnext_lighting_recommendation_mode_invalid: {candidate_id}:{mode}:{distribution}")
    multiplier = _lighting_ev_multiplier(float(ev_offset), target="runtime.lighting.dome.intensity.ev_offset")
    return {
        "value": base_value * multiplier,
        "source": "recommendation",
        "candidate_id": candidate_id,
        "mode": mode,
        "base_nominal": nominal,
        "base_range": [low, high],
        "distribution": distribution,
        "ev_offset": float(ev_offset),
    }


def _sample_assets(
    plan: RuntimeSampleCompilePlan,
    *,
    rng: random.Random,
    state: SelectorRuntimeState,
    env_id: int,
) -> Mapping[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    shared_asset_selections: dict[str, str] = {}
    for entry in plan.assets:
        candidates_by_id = {candidate.candidate_id: candidate for candidate in entry.candidates}
        result = select_candidate(
            tuple(candidates_by_id),
            entry.selector,
            selector_id=f"assets.{entry.entry_id}",
            runtime_sample_id=plan.runtime_sample_id,
            env_id=env_id,
            rng=rng,
            state=state,
        )
        candidate = candidates_by_id[result.selected_id]
        # shared_id couples several asset entries to the same concrete asset_id
        # (e.g. all front plates use the identical plate model per episode).
        shared_id = str(getattr(entry.selector, "shared_id", "") or "").strip()
        shared_asset_id = shared_asset_selections.get(shared_id) if shared_id else None
        if shared_asset_id is not None and shared_asset_id in candidate.assets:
            concrete_asset_id = shared_asset_id
        else:
            concrete_asset_id = candidate.assets[rng.randrange(len(candidate.assets))]
            if shared_id and shared_asset_id is None:
                shared_asset_selections[shared_id] = concrete_asset_id
        out[entry.entry_id] = {
            "candidate_id": result.selected_id,
            "asset_id": concrete_asset_id,
            "target": {"type": entry.target_type, "id": entry.target_id, "object_id": entry.object_id},
            "assets": list(candidate.assets),
            "cache_size": int(entry.cache_size),
            "slots": list(entry.slots),
        }
    return out


def _sample_materials(
    plan: RuntimeSampleCompilePlan,
    *,
    rng: random.Random,
    state: SelectorRuntimeState,
    env_id: int,
) -> Mapping[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    shared_material_selections: dict[str, str] = {}
    for entry in plan.materials:
        shared_id = str(getattr(entry.selector, "shared_id", "") or "").strip()
        if shared_id and shared_id in shared_material_selections:
            selected_id = shared_material_selections[shared_id]
        else:
            result = select_candidate(
                entry.candidates,
                entry.selector,
                selector_id=f"materials.{shared_id or entry.entry_id}",
                runtime_sample_id=plan.runtime_sample_id,
                env_id=env_id,
                rng=rng,
                state=state,
            )
            selected_id = result.selected_id
            if shared_id:
                shared_material_selections[shared_id] = selected_id
        if selected_id not in entry.candidates:
            raise RuntimeError(f"schema_vnext_material_shared_candidate_mismatch: {entry.entry_id}:{selected_id}")
        out[entry.entry_id] = {
            "candidate_id": selected_id,
            "policy_type": entry.policy_type,
            "target": {
                "type": entry.target_type,
                "objects": list(entry.objects),
                "prim_paths": list(entry.prim_paths),
                "surface_material": entry.surface_material,
                "material_slots": entry.material_slots,
            },
            "candidate": dict(entry.candidate_templates.get(selected_id) or {}),
            "material_template": dict(entry.candidate_payloads.get(selected_id) or {}),
            "apply": dict(entry.apply),
        }
    return out
