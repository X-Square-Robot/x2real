from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from manaenv.pipeline.runtime import master_seed as master_seed_runtime

from ..common.reproducibility import stable_seed
from ..runtime_sampling.artifacts import runtime_plan_to_payload
from ..schema_vnext import (
    VNextSceneSpec,
    VNextTaskSpec,
    compile_runtime_sample,
    runtime_effects_plan_to_payload,
)
from ..schema_vnext.runtime_sampling_plan import compile_schema_vnext_runtime_sampling_plan
from .asset_pose_bridge import initialize_schema_vnext_asset_pose_frames
from .runtime_events import apply_schema_vnext_runtime_sample
from .runtime_instruction import _definition_runtime_instruction
from .task_runtime_consumer import attach_task_runtime_consumer_event_for_task


def _resolve_runtime_sample_seed(runtime: Mapping[str, Any], *, task_payload: Mapping[str, Any]) -> int:
    """Resolve runtime_sample base seed: YAML override → master derive → OS entropy.

    Never defaults to ``0``.
    """

    override = runtime.get("seed") if isinstance(runtime, Mapping) else None
    override_int: int | None
    if override is None or override is False:
        override_int = None
    else:
        try:
            override_int = int(override)
        except (TypeError, ValueError):
            override_int = None
    definition = task_payload.get("definition") if isinstance(task_payload, Mapping) else None
    task_id = ""
    if isinstance(definition, Mapping):
        task_id = str(definition.get("task_type") or definition.get("id") or "").strip()
    plan_id = str(runtime.get("id") or "runtime").strip() or "runtime"
    return int(
        master_seed_runtime.resolve_base_seed(
            master_seed_runtime.NS_RUNTIME_SAMPLE,
            task_id or plan_id,
            plan_id,
            override=override_int,
            allow_os_entropy=True,
        )
    )


def compile_layout_runtime_events_for_task(builder: Any, scene: Any) -> None:
    """Inject schema-vNext runtime events after scene/task binding.

    The vNext path is intentionally not a compatibility shim for legacy task
    authoring events. Finalize injects the asset pose-frame startup bridge, the
    native ``layout.runtime_sample`` reset event, the generic task-template
    runtime consumer, and role projections needed by those consumers.
    """

    runtime_cfgs = getattr(builder, "runtime_cfgs", None)
    if not isinstance(runtime_cfgs, dict):
        return
    scene_payload = _schema_vnext_scene_payload(scene)
    if scene_payload and _owns_schema_vnext_asset_pose_frame_event(builder):
        _append_schema_vnext_asset_pose_frame_event(
            runtime_cfgs,
            asset_names=_schema_vnext_pose_asset_names(_schema_vnext_materialization_payload(scene)),
        )
    _inject_schema_vnext_runtime_sample(builder, scene, runtime_cfgs)
    _compile_case_bank_layout_event_params(runtime_cfgs)
    _inject_task_runtime_consumer(builder, runtime_cfgs)
    _project_task_roles_for_runtime_consumers(builder, runtime_cfgs)


def _inject_task_runtime_consumer(builder: Any, runtime_cfgs: dict[str, Any]) -> None:
    if attach_task_runtime_consumer_event_for_task(builder):
        runtime_cfgs["_task_runtime_consumer"] = {
            "injected_reset_event": True,
            "boundary": "task_template_consumer",
        }


def _inject_schema_vnext_runtime_sample(builder: Any, scene: Any, runtime_cfgs: dict[str, Any]) -> None:
    task_payload = _schema_vnext_task_payload(builder)
    if not task_payload:
        return
    scene_payload = _schema_vnext_scene_payload(scene)
    if not scene_payload:
        raise RuntimeError("task.layout.runtime_sample requires scene metadata from schema-vNext scene materialization")
    injected_native, native_summary = _append_schema_vnext_runtime_sample_event(
        runtime_cfgs,
        task_payload=task_payload,
        scene_payload=scene_payload,
        materialization=_schema_vnext_materialization_payload(scene),
        assets=getattr(scene, "assets", None),
        scene_type=getattr(scene, "scene_type", ""),
        runtime_templates=_schema_vnext_runtime_templates(builder, task_payload),
    )
    runtime = _runtime_sample_payload(task_payload)
    runtime_cfgs["_schema_vnext_runtime_sample"] = {
        "runtime_sample_id": str(runtime.get("id") or "runtime"),
        "injected_native_event": bool(injected_native),
        "runtime_boundary": "schema_vnext_native_reset_event",
        **native_summary,
    }


def _schema_vnext_task_payload(builder: Any) -> dict[str, Any]:
    definition = getattr(builder, "definition", None)
    if not isinstance(definition, Mapping):
        cfg = getattr(builder, "cfg", None)
        definition = getattr(cfg, "definition", None)
    if not isinstance(definition, Mapping):
        return {}
    layout = definition.get("layout")
    if not isinstance(layout, Mapping) or not isinstance(layout.get("runtime_sample"), Mapping):
        return {}
    payload = {
        "definition": dict(definition),
        "layout": dict(layout),
    }
    roles = getattr(builder, "roles", None)
    if isinstance(roles, Mapping):
        payload["roles"] = dict(roles)
    return payload


def _schema_vnext_scene_payload(scene: Any) -> dict[str, Any]:
    metadata = getattr(scene, "metadata", {}) or {}
    raw = metadata.get("schema_vnext_scene") if isinstance(metadata, Mapping) else None
    return dict(raw or {}) if isinstance(raw, Mapping) else {}


def _schema_vnext_materialization_payload(scene: Any) -> dict[str, Any]:
    metadata = getattr(scene, "metadata", {}) or {}
    raw = metadata.get("schema_vnext_materialization") if isinstance(metadata, Mapping) else None
    return dict(raw or {}) if isinstance(raw, Mapping) else {}


def _owns_schema_vnext_asset_pose_frame_event(builder: Any) -> bool:
    """Return whether this task level should contribute the startup bridge.

    Recursive graph children are finalized independently and their identical
    event fields are deduplicated when the graph assembles. The composite level
    must not append a second copy as an extra field, because extras intentionally
    bypass child deduplication.
    """

    return not bool(getattr(builder, "_composite_children", None))


def _append_schema_vnext_asset_pose_frame_event(
    runtime_cfgs: dict[str, Any],
    *,
    asset_names: list[str],
) -> bool:
    if not asset_names:
        return False

    from isaaclab.managers import EventTermCfg
    from isaaclab_arena.utils.configclass import make_configclass

    field_name = "schema_vnext__initialize_asset_pose_frames"
    event_terms = runtime_cfgs.get("event_terms")
    if event_terms is not None and hasattr(event_terms, field_name):
        existing = getattr(event_terms, field_name)
        params = getattr(existing, "params", None)
        if (
            getattr(existing, "func", None) is initialize_schema_vnext_asset_pose_frames
            and isinstance(params, Mapping)
            and list(params.get("asset_names") or []) == asset_names
        ):
            return False
        raise RuntimeError(f"schema_vnext_asset_pose_frame_conflict: event_terms.{field_name} already exists")

    term_cfg = EventTermCfg(
        func=initialize_schema_vnext_asset_pose_frames,
        mode="startup",
        params={"asset_names": asset_names},
    )
    if event_terms is None:
        EventsCfg = make_configclass("SchemaVNextAssetPoseFrameCfg", [(field_name, type(term_cfg), term_cfg)])
        runtime_cfgs["event_terms"] = EventsCfg()
        return True

    term_fields = _event_term_config_fields(event_terms)
    term_fields.insert(0, (field_name, type(term_cfg), term_cfg))
    EventsCfg = make_configclass("SchemaVNextAssetPoseFrameCfg", term_fields)
    runtime_cfgs["event_terms"] = EventsCfg()
    return True


def _schema_vnext_pose_asset_names(materialization: Mapping[str, Any]) -> list[str]:
    objects = materialization.get("objects")
    names = {str(name).strip() for name in (objects if isinstance(objects, Mapping) else {}) if str(name).strip()}
    runtime_cache = materialization.get("runtime_asset_cache")
    cache_objects = runtime_cache.get("objects") if isinstance(runtime_cache, Mapping) else {}
    if isinstance(cache_objects, Mapping):
        for entry in cache_objects.values():
            if not isinstance(entry, Mapping):
                continue
            for slot in entry.get("slots", []) or []:
                if not isinstance(slot, Mapping):
                    continue
                name = str(slot.get("name") or "").strip()
                if name:
                    names.add(name)
    return sorted(names)


def _append_schema_vnext_runtime_sample_event(
    runtime_cfgs: dict[str, Any],
    *,
    task_payload: Mapping[str, Any],
    scene_payload: Mapping[str, Any],
    materialization: Mapping[str, Any],
    assets: Any,
    scene_type: str,
    runtime_templates: Mapping[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    runtime = _runtime_sample_payload(task_payload)
    if not _has_schema_vnext_runtime_sample(runtime):
        return False, {}
    if assets is None:
        raise RuntimeError("schema_vnext.runtime_sample requires scene.assets during TaskGenerator.finalize")

    from isaaclab.managers import EventTermCfg
    from isaaclab_arena.utils.configclass import make_configclass

    event_terms = runtime_cfgs.get("event_terms")
    field_name = "schema_vnext__runtime_sample"
    if event_terms is not None and hasattr(event_terms, field_name):
        raise RuntimeError(
            "schema_vnext_runtime_sample_conflict: event_terms.schema_vnext__runtime_sample already exists"
        )
    plan_id = str(runtime.get("id") or "runtime").strip() or "runtime"
    seed = _resolve_runtime_sample_seed(runtime, task_payload=task_payload)
    scene_spec = VNextSceneSpec.from_dict(dict(scene_payload or {}))
    task_spec = VNextTaskSpec.from_dict(dict(task_payload or {}), scene=scene_spec)
    compiled_runtime = compile_runtime_sample(task_spec, scene_spec)
    runtime_effects_plan = runtime_effects_plan_to_payload(
        compiled_runtime,
        runtime_asset_cache=_schema_vnext_runtime_asset_cache_payload(
            scene_payload=scene_payload, materialization=materialization
        ),
    )
    _prune_runtime_asset_targets_from_legacy_reset_events(runtime_cfgs, runtime_effects_plan)
    summary: dict[str, Any] = {
        "asset_backend": "windowed_prespawn" if compiled_runtime.assets else "none",
        "asset_target_count": len(compiled_runtime.assets),
        "placement_backend": "none",
        "domain_count": 0,
        "assignment_count": 0,
        "conditional_count": 0,
        "case_count": 0,
    }
    placement_plan = compile_schema_vnext_runtime_sampling_plan(
        task_payload,
        scene_payload,
        assets,
        materialization=materialization,
        seed=stable_seed(seed, "schema_vnext_runtime_sampling_plan", plan_id),
    )
    runtime_plan_payload: dict[str, Any] | None = None
    if placement_plan is not None:
        runtime_plan_payload = runtime_plan_to_payload(
            placement_plan,
            metadata={
                "source": "schema_vnext_task_finalize",
                "event_field": field_name,
                "scene_type": scene_type,
                "runtime_sample_id": plan_id,
            },
        )
        case_count = sum(len(assignment.cases) for assignment in placement_plan.assignments) + sum(
            len(assignment.cases) for assignment in placement_plan.conditional_assignments
        )
        summary = {
            "asset_backend": "windowed_prespawn" if compiled_runtime.assets else "none",
            "asset_target_count": len(compiled_runtime.assets),
            "placement_backend": "RuntimeSamplingPlan",
            "domain_count": len(placement_plan.domains),
            "assignment_count": len(placement_plan.assignments),
            "conditional_count": len(placement_plan.conditional_assignments),
            "case_count": case_count,
        }
    from .distractor_free_space import DEFAULT_DISTRACTOR_ATTEMPTS, DEFAULT_DISTRACTOR_CLEARANCE_M

    event_params: dict[str, Any] = {
        "task_payload": dict(task_payload),
        "scene_payload": dict(scene_payload),
        "materialization": dict(materialization or {}),
        "runtime_plan": runtime_plan_payload,
        "runtime_slot_bounds": _runtime_asset_slot_bounds_payload(
            assets,
            runtime_effects_plan,
        ),
        "plan_id": plan_id,
        "seed": seed,
        "task_id": (
            (task_payload.get("definition") or {}).get("task_type")
            if isinstance(task_payload.get("definition"), Mapping)
            else None
        ),
        "runtime_options": {
            "report_limit": 128,
            "parked_slot_collision": task_spec.runtime_sample.parked_slot_collision,
        },
        "runtime_templates": dict(runtime_templates or {}),
        "runtime_effects_plan": runtime_effects_plan,
        # Reset-time free-space pack for semantic_tags:[distractor] objects.
        "distractor_clearance_m": DEFAULT_DISTRACTOR_CLEARANCE_M,
        "distractor_attempts": DEFAULT_DISTRACTOR_ATTEMPTS,
    }
    runtime_instruction = _definition_runtime_instruction(task_payload)
    if runtime_instruction is not None:
        event_params["runtime_instruction"] = runtime_instruction
    term_cfg = EventTermCfg(
        func=apply_schema_vnext_runtime_sample,
        mode="reset",
        params=event_params,
    )
    if event_terms is None:
        EventsCfg = make_configclass("SchemaVNextRuntimeSampleCfg", [(field_name, type(term_cfg), term_cfg)])
        runtime_cfgs["event_terms"] = EventsCfg()
        return True, summary
    term_fields = _event_term_config_fields(event_terms)
    # Full-scene reset terms overwrite every rigid-body pose, so they must run
    # before schema-vNext materializes the episode. Remaining task-authored reset
    # consumers must stay after it so they read the current active-slot mapping.
    insert_index = 0
    for index, (existing_name, _existing_type, existing_cfg) in enumerate(term_fields):
        existing_func = getattr(existing_cfg, "func", None)
        resets_full_scene = (
            str(existing_name).endswith("reset_scene_to_default")
            or getattr(existing_func, "__name__", "") == "reset_scene_to_default"
        )
        if resets_full_scene:
            insert_index = index + 1
    term_fields.insert(insert_index, (field_name, type(term_cfg), term_cfg))
    EventsCfg = make_configclass("SchemaVNextRuntimeSampleCfg", term_fields)
    runtime_cfgs["event_terms"] = EventsCfg()
    return True, summary


def _runtime_asset_slot_bounds_payload(
    assets: Any,
    runtime_effects_plan: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Export scaled local bounds for every runtime-selectable cache slot."""

    if not isinstance(runtime_effects_plan, Mapping):
        return {}
    from ..assets.collection import asset_map
    from ..assets.geometry_resolver import ObjectGeometryResolver

    by_name = asset_map(assets)
    resolver = ObjectGeometryResolver()
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for entry in runtime_effects_plan.get("assets", []) or []:
        if not isinstance(entry, Mapping):
            continue
        target = entry.get("target") if isinstance(entry.get("target"), Mapping) else {}
        object_id = str(target.get("object_id") or target.get("id") or "").strip()
        if not object_id:
            continue
        object_out = out.setdefault(object_id, {})
        for slot in entry.get("slots", []) or []:
            if isinstance(slot, Mapping):
                slot_name = str(slot.get("name") or "").strip()
            else:
                slot_name = str(slot or "").strip()
            if not slot_name:
                continue
            asset = by_name.get(slot_name)
            geometry = resolver.resolve(asset) if asset is not None else None
            if geometry is None:
                raise RuntimeError(f"schema_vnext_runtime_slot_geometry_missing:{slot_name}")
            slot_scale = slot.get("scale") if isinstance(slot, Mapping) else None
            scale = _runtime_asset_scale(asset, slot_scale=slot_scale)
            bounds = geometry.local_bounds.scaled(scale)
            object_out[slot_name] = {
                "x": [float(bounds.x.min), float(bounds.x.max)],
                "y": [float(bounds.y.min), float(bounds.y.max)],
                "z": [float(bounds.z.min), float(bounds.z.max)],
            }
    return out


def _runtime_asset_scale(asset: Any, *, slot_scale: Any = None) -> tuple[float, float, float]:
    raw = slot_scale if slot_scale is not None else getattr(asset, "scale", None)
    if isinstance(raw, (int, float)):
        value = float(raw)
        return (value, value, value)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    return (1.0, 1.0, 1.0)


def _runtime_sample_payload(task_payload: Mapping[str, Any]) -> dict[str, Any]:
    layout = task_payload.get("layout") if isinstance(task_payload.get("layout"), Mapping) else {}
    runtime = layout.get("runtime_sample") if isinstance(layout.get("runtime_sample"), Mapping) else {}
    return {str(key): value for key, value in runtime.items()}


def _prune_runtime_asset_targets_from_legacy_reset_events(
    runtime_cfgs: dict[str, Any],
    runtime_effects_plan: Mapping[str, Any],
) -> None:
    controlled = _runtime_asset_controlled_names(runtime_effects_plan)
    if not controlled:
        return
    event_terms = runtime_cfgs.get("event_terms")
    if event_terms is None:
        return
    for term in vars(event_terms).values():
        params = getattr(term, "params", None)
        if not isinstance(params, dict) or "asset_cfg" not in params:
            continue
        params["asset_cfg"] = _remove_controlled_asset_cfgs(params["asset_cfg"], controlled)


def _runtime_asset_controlled_names(runtime_effects_plan: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for entry in runtime_effects_plan.get("assets", []) or []:
        if not isinstance(entry, Mapping):
            continue
        target = entry.get("target") if isinstance(entry.get("target"), Mapping) else {}
        object_id = str(target.get("object_id") or target.get("id") or "").strip()
        if object_id:
            names.add(object_id)
        for slot in entry.get("slots", []) or []:
            if isinstance(slot, Mapping):
                name = str(slot.get("name") or "").strip()
            else:
                name = str(slot or "").strip()
            if name:
                names.add(name)
    return names


def _remove_controlled_asset_cfgs(value: Any, controlled: set[str]) -> Any:
    if isinstance(value, list):
        return [item for item in value if _asset_cfg_name(item) not in controlled]
    return [] if _asset_cfg_name(value) in controlled else value


def _asset_cfg_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _schema_vnext_runtime_asset_cache_payload(
    *,
    scene_payload: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    definition = scene_payload.get("definition") if isinstance(scene_payload.get("definition"), Mapping) else {}
    metadata = definition.get("metadata") if isinstance(definition.get("metadata"), Mapping) else {}
    cache = (
        metadata.get("schema_vnext_runtime_assets")
        if isinstance(metadata.get("schema_vnext_runtime_assets"), Mapping)
        else None
    )
    if isinstance(cache, Mapping):
        return dict(cache)
    cache = (
        materialization.get("runtime_asset_cache")
        if isinstance(materialization.get("runtime_asset_cache"), Mapping)
        else None
    )
    return dict(cache or {})


def _schema_vnext_runtime_templates(builder: Any, task_payload: Mapping[str, Any]) -> dict[str, Any]:
    explicit = _explicit_runtime_templates(builder, task_payload)
    if explicit:
        return explicit

    for method_name in ("make_runtime_templates", "get_runtime_templates"):
        method = getattr(builder, method_name, None)
        if not callable(method):
            continue
        value = _call_runtime_templates_hook(method, task_payload=task_payload)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise RuntimeError(f"schema_vnext_runtime_templates_invalid: {method_name} must return a mapping or None")
        return dict(value)

    # Graph YAML ``definition.runtime_instruction`` (e.g. {direction} ↔ place_beside).
    from manaenv.pipeline.runtime.runtime_instruction import runtime_instruction_templates

    authored = runtime_instruction_templates(task_payload)
    return dict(authored) if authored else {}


def _explicit_runtime_templates(builder: Any, task_payload: Mapping[str, Any]) -> dict[str, Any]:
    for value in (
        task_payload.get("runtime_templates"),
        getattr(builder, "runtime_templates", None),
        getattr(getattr(builder, "cfg", None), "runtime_templates", None),
    ):
        if isinstance(value, Mapping) and value:
            return dict(value)
    return {}


def _call_runtime_templates_hook(method: Any, *, task_payload: Mapping[str, Any]) -> Any:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method()
    params = signature.parameters
    if "task_payload" in params or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return method(task_payload=task_payload)
    return method()


def _has_schema_vnext_runtime_sample(runtime: Mapping[str, Any]) -> bool:
    targets = runtime.get("targets") if isinstance(runtime.get("targets"), Mapping) else {}
    placements = targets.get("placements") if isinstance(targets, Mapping) else []
    if isinstance(placements, list) and placements:
        return True
    # Asset-only plans (e.g. shared_id cup pools with poses owned by a custom
    # reset event) must still inject schema_vnext__runtime_sample.
    assets = targets.get("assets") if isinstance(targets, Mapping) else []
    if isinstance(assets, list) and assets:
        return True
    global_payload = runtime.get("global") if isinstance(runtime.get("global"), Mapping) else {}
    if (
        isinstance(global_payload, Mapping)
        and isinstance(global_payload.get("lighting"), Mapping)
        and global_payload["lighting"]
    ):
        return True
    materials = targets.get("materials") if isinstance(targets, Mapping) else []
    return bool(isinstance(materials, list) and materials)


def _event_term_config_fields(event_terms: Any) -> list[tuple[str, type, Any]]:
    fields = getattr(event_terms, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        names = [str(name) for name in fields]
    else:
        names = [str(name) for name in vars(event_terms)]
    out: list[tuple[str, type, Any]] = []
    seen: set[str] = set()
    for name in names:
        if name in seen or not hasattr(event_terms, name):
            continue
        value = getattr(event_terms, name)
        out.append((name, type(value), value))
        seen.add(name)
    return out


def _project_task_roles_for_runtime_consumers(builder: Any, runtime_cfgs: dict[str, Any]) -> None:
    """Project scene-bound role objects to asset names after layout finalization.

    Mimic-gen finalization may bind ``task.roles`` to concrete Asset objects.
    ManaEnv runtime skills still consume role values as scene object names
    (for example ``PickSkill.cfg.object_name`` resolves ``pickable`` through
    ``task.roles``).  By the time layout runtime sampling has been compiled,
    role-dependent term materialization has already run, so the safe public
    representation for downstream runtime consumers is the asset name string.
    """

    roles = getattr(builder, "roles", None)
    if not isinstance(roles, dict) or not roles:
        return
    projected = {str(key): _project_role_value(value, role_key=str(key)) for key, value in roles.items()}
    if projected == roles:
        return
    setattr(builder, "roles", projected)
    runtime_cfgs["_scene_layout_runtime_role_projection"] = {
        "projected": True,
        "role_count": len(projected),
    }


def _project_role_value(value: Any, *, role_key: str | None = None) -> Any:
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_project_role_value(item, role_key=role_key) for item in value]
    if isinstance(value, Mapping):
        return {key: _project_role_value(item, role_key=str(key)) for key, item in value.items()}
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    private_name = getattr(value, "_name", None)
    if isinstance(private_name, str) and private_name:
        return private_name
    # Identity roles (peg_mid → PegDb asset) often lack .name; keep the role key
    # so place/pick skills still receive a scene entity string.
    if role_key:
        return role_key
    return value


def _compile_case_bank_layout_event_params(runtime_cfgs: dict[str, Any]) -> None:
    """Compile graph-level ``apply_case_bank_layout`` params to ``compiled_bank``.

    Fail-closed at finalize: missing bank path, non-mapping YAML, non-id split,
    invalid ranges, missing ``.zone``, or requested ``by_region`` success keys.
    Also reorders the applicator after ``schema_vnext__runtime_sample`` so the
    frozen reset chain holds regardless of YAML field order.
    """

    event_terms = runtime_cfgs.get("event_terms")
    if event_terms is None:
        return
    term_fields = _event_term_config_fields(event_terms)
    applicator_indices: list[int] = []
    for index, (name, _typ, cfg) in enumerate(term_fields):
        func = getattr(cfg, "func", None)
        func_name = getattr(func, "__name__", "") if func is not None else ""
        if name == "apply_case_bank_layout" or func_name == "apply_case_bank_layout":
            applicator_indices.append(index)
    if not applicator_indices:
        return
    if len(applicator_indices) > 1:
        raise RuntimeError("case_bank_layout_compile: multiple apply_case_bank_layout event terms")

    index = applicator_indices[0]
    name, typ, cfg = term_fields[index]
    params = getattr(cfg, "params", None)
    if not isinstance(params, dict):
        raise RuntimeError(f"case_bank_layout_compile: event_terms.{name} params must be a mapping")
    bank_path = params.get("bank_path")
    if not bank_path:
        raise RuntimeError(f"case_bank_layout_compile: event_terms.{name} requires bank_path")
    required = params.get("required_zone_keys")
    required_keys: Sequence[str] | None
    if required is None:
        required_keys = None
    elif isinstance(required, (list, tuple)):
        required_keys = [str(item) for item in required]
    else:
        raise RuntimeError("case_bank_layout_compile: required_zone_keys must be a list when set")

    object_asset = params.get("object_asset")
    object_name = None if object_asset is None else str(object_asset)
    compiled = compile_case_bank_layout(
        bank_path=str(bank_path),
        required_zone_keys=required_keys,
        object_asset=object_name,
    )
    params["compiled_bank"] = compiled
    runtime_cfgs["_case_bank_layout"] = {
        "bank_path": str(bank_path),
        "compiled": True,
        "zone_keys": sorted(compiled.get("operation_zone_sets") or {}),
    }
    _reorder_apply_case_bank_layout_event(runtime_cfgs, field_name=name)


def compile_case_bank_layout(
    *,
    bank_path: str | None = None,
    bank_document: Mapping[str, Any] | None = None,
    required_zone_keys: Sequence[str] | None = None,
    object_asset: str | None = None,
) -> dict[str, Any]:
    """Pure-data compile of a P3 ``operation_zones`` / ``case_bank`` document.

    Prefer ``bank_document`` in unit tests; production finalize passes ``bank_path``.
    """

    if bank_document is None:
        if not bank_path:
            raise RuntimeError("case_bank_layout_compile: bank_path or bank_document is required")
        resolved = _resolve_case_bank_path(bank_path)
        if not resolved.is_file():
            raise RuntimeError(f"case_bank_layout_compile: bank file not found: {bank_path} ({resolved})")
        try:
            raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - surface YAML errors as compile failures
            raise RuntimeError(f"case_bank_layout_compile: failed to parse {bank_path}: {exc}") from exc
    else:
        raw = bank_document
        resolved = Path(str(bank_path)) if bank_path else Path("<inline>")

    if not isinstance(raw, Mapping):
        raise RuntimeError(f"case_bank_layout_compile: bank root must be a mapping ({resolved})")

    return compile_case_bank_document(
        dict(raw),
        source=str(resolved),
        required_zone_keys=required_zone_keys,
        object_asset=object_asset,
    )


def compile_case_bank_document(
    raw: Mapping[str, Any],
    *,
    source: str = "<inline>",
    required_zone_keys: Sequence[str] | None = None,
    object_asset: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize a case/zone bank into ``compiled_bank`` payload."""

    task = str(raw.get("task") or "").strip()
    if not task:
        raise RuntimeError(f"case_bank_layout_compile: missing task ({source})")

    split = raw.get("split")
    case_axes = raw.get("case_axes") if isinstance(raw.get("case_axes"), Mapping) else {}
    if split is None and isinstance(case_axes, Mapping):
        split = case_axes.get("split")
    if str(split or "").strip() != "id":
        raise RuntimeError(f"case_bank_layout_compile: split must be 'id' (got {split!r}) ({source})")

    roles = dict(raw["roles"]) if isinstance(raw.get("roles"), Mapping) else {}
    if object_asset:
        pickable = roles.get("pickable")
        if pickable is not None and str(pickable) != str(object_asset):
            raise RuntimeError(
                f"case_bank_layout_compile: object_asset {object_asset!r} != roles.pickable {pickable!r} ({source})"
            )

    zone_sets_raw = raw.get("operation_zone_sets")
    if not isinstance(zone_sets_raw, Mapping) or not zone_sets_raw:
        raise RuntimeError(f"case_bank_layout_compile: operation_zone_sets required ({source})")

    compiled_zones: dict[str, dict[str, Any]] = {}
    planner_only: list[str] = []
    for key, entry in zone_sets_raw.items():
        zone_key = str(key)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"case_bank_layout_compile: operation_zone_sets.{zone_key} must be a mapping ({source})")
        if "zone" in entry:
            zone = entry["zone"]
            if not isinstance(zone, Mapping):
                raise RuntimeError(
                    f"case_bank_layout_compile: operation_zone_sets.{zone_key}.zone must be a mapping ({source})"
                )
            compiled_zones[zone_key] = {"zone": _compile_zone_bounds(zone, label=f"{zone_key}.zone", source=source)}
        elif "by_region" in entry:
            planner_only.append(zone_key)
        else:
            raise RuntimeError(
                f"case_bank_layout_compile: operation_zone_sets.{zone_key} needs .zone or by_region ({source})"
            )

    required = [str(k) for k in (required_zone_keys or compiled_zones.keys())]
    for zone_key in required:
        if zone_key in compiled_zones:
            continue
        if zone_key in planner_only or (
            isinstance(zone_sets_raw.get(zone_key), Mapping) and "by_region" in zone_sets_raw[zone_key]
        ):
            raise RuntimeError(
                f"case_bank_layout_compile: success zone_key {zone_key!r} resolves only to by_region; "
                f"use a key with operation_zone_sets.<key>.zone ({source})"
            )
        raise RuntimeError(f"case_bank_layout_compile: required success zone_key {zone_key!r} missing .zone ({source})")

    robot_start = _compile_robot_start(raw, source=source)
    object_placement = _compile_object_placement(raw, source=source)
    place_target_placement = _compile_place_target_placement(raw, source=source)

    out: dict[str, Any] = {
        "task": task,
        "split": "id",
        "roles": roles,
        "operation_zone_sets": {key: compiled_zones[key] for key in required if key in compiled_zones},
        "planner_only_zone_keys": list(planner_only),
        "source": source,
    }
    if case_axes:
        out["case_axes"] = {
            "robot_start_ids": [str(x) for x in (case_axes.get("robot_start_ids") or [])],
            "object_position_ids": [str(x) for x in (case_axes.get("object_position_ids") or [])],
            "split": "id",
        }
        if not out["case_axes"]["robot_start_ids"] or not out["case_axes"]["object_position_ids"]:
            raise RuntimeError(f"case_bank_layout_compile: case_axes ids incomplete ({source})")
    if robot_start is not None:
        out["robot_start"] = robot_start
    if "robot_start_anchors" in raw:
        out["robot_start_anchors"] = _compile_robot_start_anchors(raw["robot_start_anchors"], source=source)
    if object_placement is not None:
        out["object_placement"] = object_placement
    if place_target_placement is not None:
        out["place_target_placement"] = place_target_placement
    if "object_anchors" in raw:
        support = raw.get("support_frames") if isinstance(raw.get("support_frames"), Mapping) else {}
        pick_support = support.get("pick_support") if isinstance(support, Mapping) else None
        if not isinstance(pick_support, Mapping):
            raise RuntimeError(
                f"case_bank_layout_compile: object_anchors require support_frames.pick_support ({source})"
            )
        out["support_frames"] = {
            "pick_support": _compile_support_frame(pick_support, source=source, label="support_frames.pick_support")
        }
        out["object_anchors"] = _compile_object_anchors(raw["object_anchors"], source=source)

    if "robot_start_anchors" in out and "robot_start" in out:
        raise RuntimeError(f"case_bank_layout_compile: robot_start and robot_start_anchors both set ({source})")
    if "robot_start_anchors" not in out and "robot_start" not in out:
        raise RuntimeError(f"case_bank_layout_compile: missing robot_start / robot_start_anchors ({source})")
    if "case_axes" in out and "robot_start_anchors" not in out:
        raise RuntimeError(f"case_bank_layout_compile: case_axes require robot_start_anchors ({source})")
    if "case_axes" in out and "object_anchors" not in out:
        raise RuntimeError(f"case_bank_layout_compile: case_axes require object_anchors ({source})")
    return out


def _resolve_case_bank_path(path_text: str) -> Path:
    path = Path(str(path_text)).expanduser()
    if path.is_absolute():
        return path
    package_root = Path(__file__).resolve().parents[4]  # manaenv/
    for candidate in (package_root / "defination" / path, package_root / path, Path.cwd() / path):
        if candidate.is_file() or candidate == package_root / "defination" / path:
            return candidate
    return package_root / "defination" / path


def _compile_zone_bounds(zone: Mapping[str, Any], *, label: str, source: str) -> dict[str, Any]:
    frame = str(zone.get("frame") or "world").strip() or "world"
    if frame != "world":
        raise RuntimeError(f"case_bank_layout_compile: {label} frame must be 'world' (got {frame!r}) ({source})")
    shape = str(zone.get("shape") or "aabb").strip().lower()
    if shape == "circle":
        center_role = str(zone.get("center_role") or "").strip()
        if not center_role:
            raise RuntimeError(f"case_bank_layout_compile: {label}.center_role required for circle ({source})")
        try:
            radius = float(zone.get("radius_m"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"case_bank_layout_compile: {label}.radius_m invalid ({source})") from exc
        if radius < 0:
            raise RuntimeError(f"case_bank_layout_compile: {label}.radius_m must be >= 0 ({source})")
        return {"frame": "world", "shape": "circle", "center_role": center_role, "radius_m": radius}
    if shape != "aabb":
        raise RuntimeError(f"case_bank_layout_compile: {label}.shape must be 'aabb' or 'circle' ({source})")
    x_range = _compile_range2(zone.get("x_range"), label=f"{label}.x_range", source=source)
    y_range = _compile_range2(zone.get("y_range"), label=f"{label}.y_range", source=source)
    yaw = _compile_range2(zone.get("yaw_deg_range"), label=f"{label}.yaw_deg_range", source=source)
    return {
        "frame": "world",
        "shape": "aabb",
        "x_range": x_range,
        "y_range": y_range,
        "yaw_deg_range": yaw,
    }


def _compile_range2(value: Any, *, label: str, source: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeError(f"case_bank_layout_compile: {label} must be [lo, hi] ({source})")
    try:
        lo = float(value[0])
        hi = float(value[1])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"case_bank_layout_compile: {label} values must be numeric ({source})") from exc
    if not (lo <= hi):
        raise RuntimeError(f"case_bank_layout_compile: {label} requires lo <= hi ({source})")
    return [lo, hi]


def _compile_robot_start(raw: Mapping[str, Any], *, source: str) -> dict[str, Any] | None:
    robot_start = raw.get("robot_start")
    if robot_start is None:
        return None
    if not isinstance(robot_start, Mapping):
        raise RuntimeError(f"case_bank_layout_compile: robot_start must be a mapping ({source})")
    disk = robot_start.get("disk")
    rectangle = robot_start.get("rectangle")
    if (disk is None) == (rectangle is None):
        raise RuntimeError(f"case_bank_layout_compile: robot_start needs exactly one of disk / rectangle ({source})")
    if isinstance(rectangle, Mapping):
        out = {
            "mode": "rectangle",
            "x_range": _compile_range2(rectangle.get("x_range"), label="robot_start.rectangle.x_range", source=source),
            "y_range": _compile_range2(rectangle.get("y_range"), label="robot_start.rectangle.y_range", source=source),
        }
        yaw_policy = rectangle.get("yaw_policy")
        if yaw_policy is not None:
            out["yaw_policy"] = str(yaw_policy)
        return out
    if not isinstance(disk, Mapping):
        raise RuntimeError(f"case_bank_layout_compile: robot_start.disk must be a mapping ({source})")
    center = disk.get("center")
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise RuntimeError(f"case_bank_layout_compile: robot_start.disk.center must be [x, y] ({source})")
    try:
        radius = float(disk.get("radius_m"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"case_bank_layout_compile: robot_start.disk.radius_m invalid ({source})") from exc
    if radius < 0:
        raise RuntimeError(f"case_bank_layout_compile: robot_start.disk.radius_m must be >= 0 ({source})")
    out: dict[str, Any] = {
        "mode": "disk",
        "center": [float(center[0]), float(center[1])],
        "radius_m": radius,
    }
    yaw_policy = disk.get("yaw_policy")
    if yaw_policy is not None:
        out["yaw_policy"] = str(yaw_policy)
    return out


def _compile_place_target_placement(raw: Mapping[str, Any], *, source: str) -> dict[str, Any] | None:
    placement = raw.get("place_target_placement")
    if placement is None:
        return None
    if not isinstance(placement, Mapping):
        raise RuntimeError(f"case_bank_layout_compile: place_target_placement must be a mapping ({source})")
    world_xy = placement.get("world_xy")
    if not isinstance(world_xy, Mapping):
        raise RuntimeError(f"case_bank_layout_compile: place_target_placement.world_xy required ({source})")
    try:
        z = float(placement.get("z", 0.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"case_bank_layout_compile: place_target_placement.z invalid ({source})") from exc
    out: dict[str, Any] = {
        "x_range": _compile_range2(
            world_xy.get("x_range"), label="place_target_placement.world_xy.x_range", source=source
        ),
        "y_range": _compile_range2(
            world_xy.get("y_range"), label="place_target_placement.world_xy.y_range", source=source
        ),
        "z": z,
        "keep_orientation": bool(placement.get("keep_orientation", False)),
    }
    orientation = placement.get("orientation_wxyz")
    if orientation is not None:
        if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
            raise RuntimeError(
                f"case_bank_layout_compile: place_target_placement.orientation_wxyz must have 4 values ({source})"
            )
        try:
            out["orientation_wxyz"] = [float(value) for value in orientation]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"case_bank_layout_compile: place_target_placement.orientation_wxyz must be numeric ({source})"
            ) from exc
    return out


def _compile_robot_start_anchors(raw: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not raw:
        raise RuntimeError(f"case_bank_layout_compile: robot_start_anchors must be a non-empty mapping ({source})")
    out: dict[str, Any] = {}
    for key, entry in raw.items():
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"case_bank_layout_compile: robot_start_anchors.{key} must be a mapping ({source})")
        position = entry.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            raise RuntimeError(f"case_bank_layout_compile: robot_start_anchors.{key}.position invalid ({source})")
        try:
            yaw_deg = float(entry.get("yaw_deg"))
            radius_m = float(entry.get("radius_m", 0.0))
            yaw_jitter = float(entry.get("yaw_jitter_deg", 0.0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"case_bank_layout_compile: robot_start_anchors.{key} numeric fields invalid ({source})"
            ) from exc
        if radius_m < 0 or yaw_jitter < 0:
            raise RuntimeError(f"case_bank_layout_compile: robot_start_anchors.{key} ranges must be >= 0 ({source})")
        out[str(key)] = {
            "position": [float(position[0]), float(position[1]), float(position[2]) if len(position) > 2 else 0.0],
            "yaw_deg": yaw_deg,
            "radius_m": radius_m,
            "yaw_jitter_deg": yaw_jitter,
        }
    return out


def _compile_object_placement(raw: Mapping[str, Any], *, source: str) -> dict[str, Any] | None:
    placement = raw.get("object_placement")
    if placement is None:
        return None
    if not isinstance(placement, Mapping):
        raise RuntimeError(f"case_bank_layout_compile: object_placement must be a mapping ({source})")
    table_local = placement.get("table_local_xy")
    if not isinstance(table_local, Mapping):
        raise RuntimeError(f"case_bank_layout_compile: object_placement.table_local_xy required ({source})")
    x_range = _compile_range2(
        table_local.get("x_range"), label="object_placement.table_local_xy.x_range", source=source
    )
    y_range = _compile_range2(
        table_local.get("y_range"), label="object_placement.table_local_xy.y_range", source=source
    )
    load_pose = placement.get("load_pose_world")
    if not isinstance(load_pose, (list, tuple)) or len(load_pose) < 3:
        raise RuntimeError(f"case_bank_layout_compile: object_placement.load_pose_world must be [x,y,z] ({source})")
    support_raw = placement.get("support_frame")
    if not isinstance(support_raw, Mapping):
        raise RuntimeError(
            f"case_bank_layout_compile: object_placement.support_frame is required for table_local_xy "
            f"(declare the table surface frame; no Python default) ({source})"
        )
    support = _compile_support_frame(support_raw, source=source, label="object_placement.support_frame")
    return {
        "mode": "table_local_xy",
        "x_range": x_range,
        "y_range": y_range,
        "load_pose_world": [float(load_pose[0]), float(load_pose[1]), float(load_pose[2])],
        "support_frame": support,
    }


def _compile_support_frame(raw: Mapping[str, Any], *, source: str, label: str = "support_frame") -> dict[str, Any]:
    center = raw.get("center")
    x_axis = raw.get("x_axis")
    y_axis = raw.get("y_axis")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        raise RuntimeError(f"case_bank_layout_compile: {label}.center invalid ({source})")
    if not isinstance(x_axis, (list, tuple)) or len(x_axis) != 3:
        raise RuntimeError(f"case_bank_layout_compile: {label}.x_axis invalid ({source})")
    if not isinstance(y_axis, (list, tuple)) or len(y_axis) != 3:
        raise RuntimeError(f"case_bank_layout_compile: {label}.y_axis invalid ({source})")
    try:
        object_root_z = float(raw.get("object_root_z", center[2] if len(center) > 2 else 0.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"case_bank_layout_compile: {label}.object_root_z invalid ({source})") from exc
    return {
        "center": [float(center[0]), float(center[1]), float(center[2]) if len(center) > 2 else 0.0],
        "x_axis": [float(x_axis[0]), float(x_axis[1]), float(x_axis[2])],
        "y_axis": [float(y_axis[0]), float(y_axis[1]), float(y_axis[2])],
        "object_root_z": object_root_z,
    }


def _compile_object_anchors(raw: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not raw:
        raise RuntimeError(f"case_bank_layout_compile: object_anchors must be a non-empty mapping ({source})")
    out: dict[str, Any] = {}
    for key, entry in raw.items():
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"case_bank_layout_compile: object_anchors.{key} must be a mapping ({source})")
        local_xy = entry.get("local_xy")
        if not isinstance(local_xy, (list, tuple)) or len(local_xy) != 2:
            raise RuntimeError(f"case_bank_layout_compile: object_anchors.{key}.local_xy invalid ({source})")
        try:
            yaw_deg = float(entry.get("yaw_deg", 0.0))
            radius_m = float(entry.get("radius_m", 0.0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"case_bank_layout_compile: object_anchors.{key} numeric fields invalid ({source})"
            ) from exc
        if radius_m < 0:
            raise RuntimeError(f"case_bank_layout_compile: object_anchors.{key}.radius_m must be >= 0 ({source})")
        out[str(key)] = {
            "local_xy": [float(local_xy[0]), float(local_xy[1])],
            "yaw_deg": yaw_deg,
            "radius_m": radius_m,
            "approach_region": str(entry.get("approach_region") or "") or None,
            "object_region": str(entry.get("object_region") or "") or None,
        }
    return out


def _reorder_apply_case_bank_layout_event(runtime_cfgs: dict[str, Any], *, field_name: str) -> None:
    """Place ``apply_case_bank_layout`` after runtime_sample (or after scene reset)."""

    event_terms = runtime_cfgs.get("event_terms")
    if event_terms is None:
        return
    term_fields = _event_term_config_fields(event_terms)
    applicator = None
    remaining: list[tuple[str, type, Any]] = []
    for name, typ, cfg in term_fields:
        if name == field_name:
            applicator = (name, typ, cfg)
            continue
        remaining.append((name, typ, cfg))
    if applicator is None:
        return

    insert_index = 0
    for index, (existing_name, _typ, existing_cfg) in enumerate(remaining):
        existing_func = getattr(existing_cfg, "func", None)
        is_runtime_sample = (
            existing_name == "schema_vnext__runtime_sample"
            or getattr(existing_func, "__name__", "") == "apply_schema_vnext_runtime_sample"
        )
        is_scene_reset = (
            str(existing_name).endswith("reset_scene_to_default")
            or getattr(existing_func, "__name__", "") == "reset_scene_to_default"
        )
        if is_runtime_sample or is_scene_reset:
            insert_index = index + 1
    remaining.insert(insert_index, applicator)

    from isaaclab_arena.utils.configclass import make_configclass

    EventsCfg = make_configclass("CaseBankLayoutOrderedEventsCfg", remaining)
    runtime_cfgs["event_terms"] = EventsCfg()
