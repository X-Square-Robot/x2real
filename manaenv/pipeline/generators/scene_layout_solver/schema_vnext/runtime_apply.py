from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from ..common.diagnostics import error
from ..common.geometry import QuatXYZW, Vec3
from ..runtime_sampling.sampler import RuntimePlanSampler
from .materialization import SceneMaterializationSample
from .parser import SelectorSpec, VNextSceneSpec, VNextTaskSpec
from .runtime_compile import (
    RuntimeSampleCompilePlan,
    RuntimeSampleSelectionResult,
    compile_runtime_sample,
    sample_recommended_dome_intensity,
    sample_runtime_placement_choices,
)
from .runtime_sampling_plan import compile_schema_vnext_runtime_sampling_plan
from .selector_engine import SelectorRuntimeState, select_candidate


@dataclass(frozen=True)
class VNextRuntimeBasePose:
    position: Vec3
    quaternion: QuatXYZW


@dataclass(frozen=True)
class VNextRuntimePoseSample:
    position: Vec3
    quaternion: QuatXYZW


@dataclass(frozen=True)
class VNextRuntimeApplyResult:
    plan: Mapping[str, Any]
    selection: RuntimeSampleSelectionResult
    poses: Mapping[str, VNextRuntimePoseSample]
    context: Mapping[str, Any]


@dataclass(frozen=True)
class VNextRuntimeCompiledPlan:
    effect_plan: RuntimeSampleCompilePlan
    runtime_plan: Mapping[str, Any] | None
    materialization: Mapping[str, Any] | SceneMaterializationSample | None = None


def compile_schema_vnext_runtime_apply_plan(
    task_payload: Mapping[str, Any],
    scene_payload: Mapping[str, Any],
    *,
    materialization: Mapping[str, Any] | SceneMaterializationSample | None = None,
    assets: Mapping[str, Any] | Iterable[Any] | None = None,
    seed: int | None = None,
) -> VNextRuntimeCompiledPlan:
    """Compile schema-vNext reset-time directives into reusable runtime plans."""

    scene = VNextSceneSpec.from_dict(dict(scene_payload or {}))
    task = VNextTaskSpec.from_dict(dict(task_payload or {}), scene=scene)
    effect_plan = compile_runtime_sample(task, scene)
    runtime_plan_payload: Mapping[str, Any] | None = None
    if effect_plan.placements:
        if assets is None:
            raise RuntimeError("schema_vnext_runtime_assets_required: placement runtime sampling requires scene assets")
        runtime_plan = compile_schema_vnext_runtime_sampling_plan(
            task_payload,
            scene_payload,
            assets,
            materialization=materialization,
            seed=int(seed if seed is not None else effect_plan.seed if effect_plan.seed is not None else 0),
        )
        if runtime_plan is None:
            raise RuntimeError("schema_vnext_runtime_sampling_plan_missing")
        runtime_plan_payload = runtime_plan.to_json()
    return VNextRuntimeCompiledPlan(
        effect_plan=effect_plan,
        runtime_plan=runtime_plan_payload,
        materialization=materialization,
    )


def sample_schema_vnext_runtime_from_plan(
    compiled_plan: VNextRuntimeCompiledPlan,
    *,
    seed: int,
    episode_index: int,
    env_id: int = 0,
    selector_state: SelectorRuntimeState | None = None,
    base_poses: Mapping[str, VNextRuntimeBasePose] | None = None,
) -> VNextRuntimeApplyResult:
    """Cheap-sample one episode from a precompiled schema-vNext runtime plan."""

    effect_plan = compiled_plan.effect_plan
    effects_only = replace(effect_plan, placements=())
    selection = sample_runtime_placement_choices(
        effects_only,
        rng=random.Random(int(seed)),
        state=selector_state,
        env_id=int(env_id),
    )
    if not effect_plan.placements:
        context = _context_base(effect_plan.runtime_sample_id, episode_index=episode_index, env_id=env_id, seed=seed)
        context["compile_context"] = _materialization_compile_context(compiled_plan.materialization)
        context["selections"]["assets"] = dict(selection.assets)
        context["selections"]["lighting"] = dict(selection.lighting)
        context["selections"]["materials"] = dict(selection.materials)
        return VNextRuntimeApplyResult(plan={}, selection=selection, poses={}, context=context)
    if not compiled_plan.runtime_plan:
        raise RuntimeError("schema_vnext_runtime_sampling_plan_missing")

    plan_payload = dict(compiled_plan.runtime_plan)
    result = RuntimePlanSampler(plan_payload, plan_id=effect_plan.runtime_sample_id).sample_result(
        seed=int(seed),
        base_quats={name: pose.quaternion for name, pose in dict(base_poses or {}).items()},
        base_poses=base_poses,
        episode_index=int(episode_index),
        env_id=int(env_id),
    )
    context = dict(result.context)
    context["runtime_sample_id"] = effect_plan.runtime_sample_id
    context["episode_index"] = int(episode_index)
    context["env_id"] = int(env_id)
    context["seed"] = int(seed)
    context.setdefault("selections", {})
    context.setdefault("domain_samples", {})
    context.setdefault("outputs", {})
    context["compile_context"] = _materialization_compile_context(compiled_plan.materialization)
    context.setdefault("selections", {})["assets"] = dict(selection.assets)
    context.setdefault("selections", {})["lighting"] = dict(selection.lighting)
    context.setdefault("selections", {})["materials"] = dict(selection.materials)
    poses = {
        name: VNextRuntimePoseSample(position=pose.position, quaternion=pose.quaternion)
        for name, pose in result.poses.items()
    }
    context["applied"] = {
        "placements": {
            name: {
                "position": [float(value) for value in pose.position],
                "quaternion": [float(value) for value in pose.quaternion],
            }
            for name, pose in sorted(poses.items())
        }
    }
    return VNextRuntimeApplyResult(plan=plan_payload, selection=selection, poses=poses, context=context)


def runtime_effects_plan_to_payload(
    plan: RuntimeSampleCompilePlan,
    *,
    runtime_asset_cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize reset-time material and lighting selectors for ManaEnv event params."""

    return {
        "runtime_sample_id": plan.runtime_sample_id,
        "seed": plan.seed,
        "assets": [_asset_entry_to_payload(entry, runtime_asset_cache=runtime_asset_cache) for entry in plan.assets],
        "lighting": [
            {
                "channel": entry.channel,
                "kind": entry.kind,
                "selector": _selector_to_payload(entry.selector) if entry.selector is not None else None,
                "candidates": {str(key): dict(value) for key, value in entry.candidates.items()},
                "value_range": list(entry.value_range) if entry.value_range is not None else None,
                "source_channel": entry.source_channel,
                "params": dict(entry.params),
            }
            for entry in plan.lighting
        ],
        "materials": [
            {
                "id": entry.entry_id,
                "target_type": entry.target_type,
                "objects": list(entry.objects),
                "prim_paths": list(entry.prim_paths),
                "surface_material": entry.surface_material,
                "material_slots": (
                    list(entry.material_slots) if isinstance(entry.material_slots, tuple) else entry.material_slots
                ),
                "policy_type": entry.policy_type,
                "candidates": list(entry.candidates),
                "candidate_templates": {
                    str(key): dict(value) for key, value in getattr(entry, "candidate_templates", {}).items()
                },
                "candidate_payloads": {str(key): dict(value) for key, value in entry.candidate_payloads.items()},
                "selector": _selector_to_payload(entry.selector),
                "apply": dict(entry.apply),
            }
            for entry in plan.materials
        ],
    }


def _select_runtime_asset_slot(
    *,
    entry_id: str,
    selector: SelectorSpec,
    selected_candidate_id: str,
    concrete_assets: tuple[str, ...],
    slots: list[dict[str, Any]],
    rng: random.Random,
    shared_asset_selections: dict[str, str],
    distinct_asset_selections: dict[str, set[str]],
) -> tuple[Mapping[str, Any], str]:
    """Select one concrete cached slot, honoring shared/distinct group contracts."""
    shared_id = str(selector.shared_id or "").strip()
    distinct_id = str(selector.distinct_id or "").strip()
    shared_asset_id = shared_asset_selections.get(shared_id) if shared_id else None
    used_distinct_assets = distinct_asset_selections.setdefault(distinct_id, set()) if distinct_id else set()

    if shared_asset_id is not None:
        matching = [slot for slot in slots if str(slot.get("asset_id") or "") == shared_asset_id]
        if matching:
            return matching[rng.randrange(len(matching))], shared_asset_id

    candidate_slots = [slot for slot in slots if str(slot.get("candidate_id") or "") == selected_candidate_id]
    selectable_slots = candidate_slots or slots
    if distinct_id:
        selectable_slots = [
            slot for slot in selectable_slots if str(slot.get("asset_id") or "") not in used_distinct_assets
        ]
        if not selectable_slots:
            raise error(
                "schema_vnext_selector_distinct_exhausted",
                "distinct selector group has no unused concrete asset",
                target=f"assets.{entry_id}",
                distinct_id=distinct_id,
                used_asset_ids=sorted(used_distinct_assets),
            )

    selected_slot: Mapping[str, Any] = (
        selectable_slots[rng.randrange(len(selectable_slots))] if selectable_slots else {}
    )
    asset_id = str(selected_slot.get("asset_id") or "")
    if not asset_id:
        selectable_assets = [asset for asset in concrete_assets if asset not in used_distinct_assets]
        if not selectable_assets:
            raise error(
                "schema_vnext_selector_distinct_exhausted",
                "distinct selector group has no unused concrete asset",
                target=f"assets.{entry_id}",
                distinct_id=distinct_id,
                used_asset_ids=sorted(used_distinct_assets),
            )
        asset_id = selectable_assets[rng.randrange(len(selectable_assets))]

    if shared_id and shared_asset_id is None:
        shared_asset_selections[shared_id] = asset_id
    if distinct_id:
        used_distinct_assets.add(asset_id)
    return selected_slot, asset_id


def sample_runtime_effects_payload(
    payload: Mapping[str, Any],
    *,
    seed: int,
    env_id: int,
    state: SelectorRuntimeState,
) -> RuntimeSampleSelectionResult:
    """Cheap-sample material and lighting effects from a precompiled payload."""

    runtime_sample_id = str(payload.get("runtime_sample_id") or "runtime")
    rng = random.Random(int(seed))
    assets: dict[str, Mapping[str, Any]] = {}
    shared_asset_selections: dict[str, str] = {}
    distinct_asset_selections: dict[str, set[str]] = {}
    for raw_entry in payload.get("assets", []) or []:
        if not isinstance(raw_entry, Mapping):
            continue
        entry_id = str(raw_entry.get("id") or "").strip()
        if not entry_id:
            continue
        candidates: dict[str, tuple[str, ...]] = {}
        for raw_candidate in raw_entry.get("candidates", []) or []:
            if not isinstance(raw_candidate, Mapping):
                continue
            candidate_id = str(raw_candidate.get("id") or "").strip()
            concrete_assets = tuple(str(item) for item in raw_candidate.get("assets", []) or [] if str(item))
            if candidate_id and concrete_assets:
                candidates[candidate_id] = concrete_assets
        selector = _selector_from_payload(
            raw_entry.get("selector") or {"type": "uniform"},
            candidate_ids=tuple(candidates),
            target=f"assets.{entry_id}",
        )
        result = select_candidate(
            tuple(candidates),
            selector,
            selector_id=f"assets.{entry_id}",
            runtime_sample_id=runtime_sample_id,
            env_id=env_id,
            rng=rng,
            state=state,
        )
        concrete_assets = candidates[result.selected_id]
        slots = [
            dict(slot)
            for slot in raw_entry.get("slots", []) or []
            if isinstance(slot, Mapping) and str(slot.get("name") or "").strip()
        ]
        if not slots:
            slots = [
                {"name": str(slot_name), "asset_id": "", "candidate_id": ""}
                for slot_name in raw_entry.get("slots", []) or []
                if str(slot_name)
            ]
        # shared_id couples entries to one concrete asset; distinct_id deals
        # entries from the same group without replacement.
        selected_slot, asset_id = _select_runtime_asset_slot(
            entry_id=entry_id,
            selector=selector,
            selected_candidate_id=result.selected_id,
            concrete_assets=concrete_assets,
            slots=slots,
            rng=rng,
            shared_asset_selections=shared_asset_selections,
            distinct_asset_selections=distinct_asset_selections,
        )
        slot_name = str(selected_slot.get("name") or "")
        inactive_slots = [str(slot.get("name") or "") for slot in slots if str(slot.get("name") or "") != slot_name]
        target_payload = raw_entry.get("target") if isinstance(raw_entry.get("target"), Mapping) else {}
        assets[entry_id] = {
            "candidate_id": result.selected_id,
            "asset_id": asset_id,
            "slot": slot_name,
            "inactive_slots": inactive_slots,
            "target": dict(target_payload),
            "assets": list(concrete_assets),
        }
    lighting: dict[str, Mapping[str, Any]] = {}
    for raw_entry in payload.get("lighting", []) or []:
        if not isinstance(raw_entry, Mapping):
            continue
        channel = str(raw_entry.get("channel") or "").strip()
        kind = str(raw_entry.get("kind") or "").strip()
        if not channel:
            continue
        if kind == "texture":
            candidates = {
                str(key): dict(value)
                for key, value in dict(raw_entry.get("candidates") or {}).items()
                if isinstance(value, Mapping)
            }
            selector = _selector_from_payload(
                raw_entry.get("selector") or {"type": "uniform"},
                candidate_ids=tuple(candidates),
                target=f"lighting.{channel}",
            )
            result = select_candidate(
                tuple(candidates),
                selector,
                selector_id=f"lighting.{channel}",
                runtime_sample_id=runtime_sample_id,
                env_id=env_id,
                rng=rng,
                state=state,
            )
            selected_payload = candidates[result.selected_id]
            lighting[channel] = {
                "candidate_id": result.selected_id,
                **{str(key): value for key, value in selected_payload.items()},
            }
        elif kind == "scalar":
            value_range = raw_entry.get("value_range")
            if not isinstance(value_range, list) or len(value_range) != 2:
                raise RuntimeError(f"schema_vnext_runtime_effect_scalar_range_invalid: {channel}")
            low, high = float(value_range[0]), float(value_range[1])
            lighting[channel] = {"value": low if low == high else rng.uniform(low, high)}
        elif kind == "candidate_recommendation":
            source_channel = str(raw_entry.get("source_channel") or "").strip()
            candidate_id = str((lighting.get(source_channel) or {}).get("candidate_id") or "")
            candidates = {
                str(key): dict(value)
                for key, value in dict(raw_entry.get("candidates") or {}).items()
                if isinstance(value, Mapping)
            }
            if candidate_id not in candidates:
                raise RuntimeError(f"schema_vnext_lighting_recommendation_source_missing: {channel}:{candidate_id}")
            params = dict(raw_entry.get("params") or {})
            sampled = sample_recommended_dome_intensity(
                candidates[candidate_id],
                mode=str(params.get("mode") or ""),
                ev_offset=float(params.get("ev_offset", 0.0)),
                rng=rng,
            )
            lighting[channel] = {
                **sampled,
                "profile_id": str(params.get("profile_id") or ""),
                "dome_exposure": float(params["dome_exposure"]),
            }
        elif kind == "profile_scalar":
            lighting[channel] = dict(raw_entry.get("params") or {})

    materials: dict[str, Mapping[str, Any]] = {}
    shared_material_selections: dict[str, str] = {}
    for raw_entry in payload.get("materials", []) or []:
        if not isinstance(raw_entry, Mapping):
            continue
        entry_id = str(raw_entry.get("id") or "").strip()
        candidates = tuple(str(item) for item in raw_entry.get("candidates", []) or [])
        candidate_templates = {
            str(key): dict(value)
            for key, value in dict(raw_entry.get("candidate_templates") or {}).items()
            if isinstance(value, Mapping)
        }
        if not entry_id:
            continue
        selector = _selector_from_payload(
            raw_entry.get("selector") or {"type": "uniform"},
            candidate_ids=candidates,
            target=f"materials.{entry_id}",
        )
        shared_id = str(getattr(selector, "shared_id", "") or "").strip()
        if shared_id and shared_id in shared_material_selections:
            selected_id = shared_material_selections[shared_id]
        else:
            result = select_candidate(
                candidates,
                selector,
                selector_id=f"materials.{shared_id or entry_id}",
                runtime_sample_id=runtime_sample_id,
                env_id=env_id,
                rng=rng,
                state=state,
            )
            selected_id = result.selected_id
            if shared_id:
                shared_material_selections[shared_id] = selected_id
        if selected_id not in candidates:
            raise RuntimeError(f"schema_vnext_material_shared_candidate_mismatch: {entry_id}:{selected_id}")
        material_slots = raw_entry.get("material_slots", "auto")
        candidate_payloads = {
            str(key): dict(value)
            for key, value in dict(raw_entry.get("candidate_payloads") or {}).items()
            if isinstance(value, Mapping)
        }
        materials[entry_id] = {
            "candidate_id": selected_id,
            "policy_type": str(raw_entry.get("policy_type") or ""),
            "target": {
                "type": str(raw_entry.get("target_type") or "objects"),
                "objects": [str(item) for item in raw_entry.get("objects", []) or []],
                "prim_paths": [str(item) for item in raw_entry.get("prim_paths", []) or []],
                "surface_material": (
                    str(raw_entry.get("surface_material")) if raw_entry.get("surface_material") is not None else None
                ),
                "material_slots": tuple(material_slots) if isinstance(material_slots, list) else material_slots,
            },
            "candidate": dict(candidate_templates.get(selected_id) or {}),
            "material_template": dict(candidate_payloads.get(selected_id) or {}),
            "apply": dict(raw_entry.get("apply") or {}),
        }
    return RuntimeSampleSelectionResult(
        runtime_sample_id=runtime_sample_id,
        env_id=int(env_id),
        placements={},
        assets=assets,
        lighting=lighting,
        materials=materials,
    )


def _asset_entry_to_payload(entry: Any, *, runtime_asset_cache: Mapping[str, Any] | None) -> dict[str, Any]:
    cache_objects = {}
    if isinstance(runtime_asset_cache, Mapping):
        cache_objects = dict(runtime_asset_cache.get("objects") or {})
    cache_entry = cache_objects.get(entry.object_id) if isinstance(cache_objects, Mapping) else None
    slots = []
    if isinstance(cache_entry, Mapping):
        raw_slots = cache_entry.get("slots")
        if isinstance(raw_slots, list):
            slots = [dict(slot) for slot in raw_slots if isinstance(slot, Mapping)]
    if not slots:
        slots = [{"name": slot_name, "asset_id": "", "candidate_id": ""} for slot_name in entry.slots]
    return {
        "id": entry.entry_id,
        "target": {"type": entry.target_type, "id": entry.target_id, "object_id": entry.object_id},
        "cache_size": int(entry.cache_size),
        "slots": slots,
        "candidates": [
            {"id": candidate.candidate_id, "assets": list(candidate.assets)} for candidate in entry.candidates
        ],
        "selector": _selector_to_payload(entry.selector),
    }


def sample_schema_vnext_runtime(
    task_payload: Mapping[str, Any],
    scene_payload: Mapping[str, Any],
    *,
    materialization: Mapping[str, Any] | SceneMaterializationSample | None = None,
    assets: Mapping[str, Any] | Iterable[Any] | None = None,
    seed: int,
    episode_index: int,
    env_id: int = 0,
    selector_state: SelectorRuntimeState | None = None,
    base_poses: Mapping[str, VNextRuntimeBasePose] | None = None,
) -> VNextRuntimeApplyResult:
    """Compile once for this call, then cheap-sample one schema-vNext runtime episode.

    Batch/runtime callers should use ``compile_schema_vnext_runtime_apply_plan``
    and ``sample_schema_vnext_runtime_from_plan`` so reset-time sampling never
    recompiles the RuntimeSamplingPlan per episode.
    """

    compiled_plan = compile_schema_vnext_runtime_apply_plan(
        task_payload,
        scene_payload,
        materialization=materialization,
        assets=assets,
        seed=seed,
    )
    return sample_schema_vnext_runtime_from_plan(
        compiled_plan,
        seed=seed,
        episode_index=episode_index,
        env_id=env_id,
        selector_state=selector_state,
        base_poses=base_poses,
    )


def _context_base(runtime_sample_id: str, *, episode_index: int, env_id: int, seed: int) -> dict[str, Any]:
    return {
        "runtime_sample_id": str(runtime_sample_id),
        "episode_index": int(episode_index),
        "env_id": int(env_id),
        "seed": int(seed),
        "selections": {},
        "domain_samples": {},
        "outputs": {},
    }


def _materialization_compile_context(
    materialization: Mapping[str, Any] | SceneMaterializationSample | None,
) -> dict[str, Any]:
    if materialization is None:
        return {"materialized": {"background": {}, "objects": {}}}
    if isinstance(materialization, SceneMaterializationSample):
        objects = {
            object_id: {
                "candidate_ref_id": materialization.selected_candidate_refs.get(f"object:{object_id}"),
                "asset_id": asset_id,
                "scale": materialization.object_scales.get(object_id),
            }
            for object_id, asset_id in sorted(materialization.object_assets.items())
        }
        return {
            "materialized": {
                "background": {
                    "candidate_ref_id": materialization.selected_candidate_refs.get("background"),
                    "asset_id": materialization.background_asset,
                },
                "objects": objects,
            }
        }
    return dict(materialization)


def _selector_to_payload(selector: SelectorSpec | None) -> dict[str, Any] | None:
    if selector is None:
        return None
    payload: dict[str, Any] = {"type": selector.selector_type}
    if selector.shared_id:
        payload["shared_id"] = selector.shared_id
    if selector.distinct_id:
        payload["distinct_id"] = selector.distinct_id
    if selector.selector_type == "weighted":
        payload["weights"] = {str(key): float(value) for key, value in selector.weights.items()}
        payload["default_weight"] = float(selector.default_weight)
    elif selector.selector_type == "schedule":
        payload["mode"] = selector.mode
        if selector.mode == "cycle":
            payload["order"] = list(selector.order)
        elif selector.mode == "blocks":
            payload["blocks"] = [{"id": block.candidate_id, "count": int(block.count)} for block in selector.blocks]
        elif selector.mode == "counts":
            payload["counts"] = {str(key): int(value) for key, value in selector.counts.items()}
            if selector.order:
                payload["order"] = list(selector.order)
            if selector.order_policy is not None:
                payload["order_policy"] = selector.order_policy
        payload["on_exhausted"] = selector.on_exhausted
    return payload


def _selector_from_payload(payload: Any, *, candidate_ids: tuple[str, ...], target: str) -> SelectorSpec:
    return SelectorSpec.from_dict(payload, target=target, candidate_ids=set(candidate_ids))
