from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from manaenv.utils.asset_utils import inactive_pool_slot_position
from manaenv.utils.debug_print import manaprint
from manaenv.utils.tensor_utils import as_torch, select_env_item, to_numpy_keep_shape
from manaenv.utils.usd_pose_sync import set_existing_pose_xform_ops
from manaenv.utils.usd_pose_sync import (
    should_author_pose_owned_visibility_to_usd as _should_author_pose_owned_visibility_to_usd,
)
from manaenv.utils.usd_pose_sync import should_sync_runtime_pose_to_usd as _should_sync_runtime_pose_to_usd

from ..common.geometry import (
    Bounds2D,
    Bounds3D,
    dot,
    projected_footprint_bounds,
    quat_normalize,
    support_contact_offset,
    vec_add,
    vec_mul,
    vec_sub,
)
from ..common.reproducibility import stable_seed
from ..runtime_sampling.sampler import Quat, RuntimePlanSampler, RuntimePoseSample
from ..runtime_sampling.template import resolve_runtime_template_fields
from ..schema_vnext import SelectorRuntimeState, sample_runtime_effects_payload
from .asset_pose_bridge import (
    actor_pose_to_canonical_pose,
    asset_supports_canonical_pose_bridge,
    asset_supports_sim_root_pose,
    canonical_pose_to_actor_pose,
)
from .distractor_free_space import (
    DEFAULT_DISTRACTOR_ATTEMPTS,
    DEFAULT_DISTRACTOR_CLEARANCE_M,
    reposition_distractors_in_free_space,
    surface_local_rect_from_world_zone,
)

LOGGER = logging.getLogger(__name__)

# Inactive runtime-cache slots stay in the PhysX scene (no disable_simulation:
# island recomposition stalled resets) and are teleported once per reset.
# Gravity and collision are both cleared on parked spares so they stop falling
# and leave the broadphase pair-test set. They still park BELOW the workspace
# (legacy away_z). Parking above (z>0) used to drop invisible spares through
# the tabletop onto the active object between resets.
_INACTIVE_ISOLATION_Z = -10.0
_GRAVITY_CTRL_LOGGED = False
_COLLISION_CTRL_LOGGED = False

# One-shot backend-neutral mass dump for spawn_mass debugging (cleared when process exits).
_RUNTIME_SPAWN_MASS_DUMPED = False

# Hang-bar / fixed-pose cloth reset still clears package ``/Aligned`` children to
# identity. Schema-vNext rigid objects use the canonical↔actor pose bridge instead;
# do not reintroduce a YAML ``ignore_aligned`` opt-in here.
_ALIGNED_IGNORED_PRIMS: set[str] = set()


def apply_schema_vnext_runtime_sample(
    env: Any,
    env_ids: Any,
    *,
    task_payload: Mapping[str, Any],
    scene_payload: Mapping[str, Any],
    materialization: Mapping[str, Any] | None = None,
    plan_id: str | None = None,
    seed: int,
    task_id: str | None = None,
    runtime_options: Mapping[str, Any] | None = None,
    runtime_plan: Mapping[str, Any] | None = None,
    runtime_slot_bounds: Mapping[str, Any] | None = None,
    runtime_effects_plan: Mapping[str, Any] | None = None,
    runtime_templates: Mapping[str, Any] | None = None,
    runtime_instruction: Mapping[str, Any] | None = None,
    distractor_clearance_m: float = DEFAULT_DISTRACTOR_CLEARANCE_M,
    distractor_attempts: int = DEFAULT_DISTRACTOR_ATTEMPTS,
) -> None:
    """Native schema-vNext reset event for placement, material, lighting, and context."""

    if env_ids is None:
        return

    env_id_list = (
        [int(v) for v in env_ids.detach().cpu().tolist()] if hasattr(env_ids, "detach") else [int(v) for v in env_ids]
    )
    plan_key = str(plan_id or _runtime_sample_id(task_payload) or "runtime")
    state = _schema_vnext_selector_state(env, plan_key)
    episode_indices = _schema_vnext_effect_episode_indices(env, plan_key, env_id_list)
    placement_sampler = _schema_vnext_runtime_plan_sampler(runtime_plan, task_payload=task_payload, plan_key=plan_key)
    if runtime_effects_plan is None and _schema_vnext_task_has_effects(task_payload):
        raise RuntimeError("schema_vnext_runtime_effects_plan_missing")
    object_ids = placement_sampler.objects if placement_sampler is not None else set()
    base_pose_object_ids = (
        set(object_ids) | set(getattr(placement_sampler, "parent_objects", set()))
        if placement_sampler is not None
        else set()
    )
    runtime_slot_names = _runtime_asset_slot_names(runtime_effects_plan)
    runtime_slot_parking_indices = {name: index for index, name in enumerate(sorted(runtime_slot_names))}
    scene_assets: dict[str, Any] = {}
    missing: list[str] = []
    for name in sorted(set(base_pose_object_ids) | runtime_slot_names):
        asset = _resolve_asset(env, name, None)
        if asset is None:
            missing.append(name)
        else:
            scene_assets[name] = asset
    if missing:
        raise RuntimeError(f"schema-vNext runtime sample references missing scene assets: {missing}")

    _debug_dump_runtime_spawn_masses_once(env, scene_assets, materialization)

    pose_batches: dict[str, dict[str, list[Any]]] = {name: {"env_ids": [], "rows": []} for name in scene_assets}
    contexts_by_env: dict[int, dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    # Per (slot, env) gravity intent: active slots keep gravity, inactive spares
    # get gravity disabled so they rest at their parked (z=-10) pose instead of
    # free-falling for the whole episode. Unlike disabling simulation, toggling
    # the gravity flag does not recompose the PhysX island, so resets stay cheap.
    slot_gravity_toggles: dict[str, dict[int, bool]] = {}
    for local_index, (env_id, episode_token) in enumerate(zip(env_id_list, episode_indices, strict=True)):
        sample_seed = stable_seed(
            int(seed),
            "schema_vnext_runtime_sample",
            plan_key,
            int(env_id),
            int(episode_token),
        )
        effects = _sample_schema_vnext_effects(
            runtime_effects_plan,
            plan_key=plan_key,
            seed=int(sample_seed),
            env_id=int(env_id),
            state=state,
        )
        applied_assets = _apply_schema_vnext_assets_for_env(
            env,
            env_id=int(env_id),
            assets=effects.assets,
            plan_id=plan_key,
        )
        _collect_slot_gravity_toggles(slot_gravity_toggles, applied_assets, env_id=int(env_id))
        base_poses = {
            name: _base_pose(env, scene_assets[name], local_index, env_ids)
            for name in sorted(base_pose_object_ids)
            if name in scene_assets
        }
        base_quats = {name: pose[1] for name, pose in base_poses.items()}
        sampled_poses = {}
        if placement_sampler is not None:
            result = placement_sampler.sample_result(
                seed=int(sample_seed),
                base_quats=base_quats,
                base_poses=base_poses,
                episode_index=int(episode_token),
                env_id=int(env_id),
            )
            sampled_poses = result.poses
            context = dict(result.context)
        else:
            context = {"plan_id": plan_key, "selections": {}, "domain_samples": {}, "outputs": {}}
        sampled_poses = _correct_active_slot_support_poses(
            sampled_poses,
            applied_assets=applied_assets,
            runtime_slot_bounds=runtime_slot_bounds,
            runtime_plan=runtime_plan,
            context=context,
        )
        # nut_on_screw: matching screw must share the nut's left/right half-row.
        side_rng = random.Random(stable_seed(int(sample_seed), "nut_screw_same_side", plan_key, int(env_id)))
        sampled_poses = _ensure_nut_on_screw_matching_same_side(
            sampled_poses,
            applied_assets=applied_assets,
            env_id=int(env_id),
            rng=side_rng,
        )
        # place_beside runs before clutter so its success zone can be kept clear.
        dir_seed = stable_seed(int(sample_seed), "place_beside_direction", plan_key, int(env_id), int(episode_token))
        _attach_place_beside_direction(
            context,
            task_payload=task_payload,
            rng=random.Random(int(dir_seed)),
            placements=_placements_payload(sampled_poses),
        )
        # Distractors: reject-sample into free tabletop after task AABBs.
        free_space_rng = random.Random(stable_seed(int(sample_seed), "distractor_free_space", plan_key, int(env_id)))
        free_space = reposition_distractors_in_free_space(
            sampled_poses,
            scene_payload=scene_payload,
            runtime_plan=runtime_plan,
            runtime_slot_bounds=runtime_slot_bounds,
            applied_assets=applied_assets,
            context=context,
            rng=free_space_rng,
            clearance_m=float(distractor_clearance_m),
            attempts=int(distractor_attempts),
            keepout_rects_by_surface=_place_beside_keepout_rects(
                context,
                task_payload=task_payload,
                sampled_poses=sampled_poses,
                runtime_plan=runtime_plan,
            ),
            support_surface_lookup=lambda plan, object_id, ctx: _runtime_support_surface(
                plan, object_id=object_id, context=ctx
            ),
        )
        sampled_poses = free_space.poses
        _attach_schema_vnext_runtime_context_base(
            context,
            runtime_sample_id=plan_key,
            episode_index=int(episode_token),
            env_id=int(env_id),
            seed=int(sample_seed),
            materialization=materialization,
        )
        context.setdefault("selections", {})["assets"] = dict(effects.assets)
        context.setdefault("selections", {})["lighting"] = dict(effects.lighting)
        context.setdefault("selections", {})["materials"] = dict(effects.materials)
        raw_applied = context.get("applied")
        applied = {key: value for key, value in raw_applied.items()} if isinstance(raw_applied, Mapping) else {}
        applied["assets"] = applied_assets
        applied["lighting"] = _apply_schema_vnext_lighting(effects.lighting, env_id=int(env_id))
        applied["materials"] = _apply_schema_vnext_materials_for_env(
            env,
            env_id=int(env_id),
            materials=effects.materials,
            base_seed=int(sample_seed),
            task_id=task_id,
            runtime_options=runtime_options,
        )
        applied["placements"] = _placements_payload(sampled_poses)
        applied["distractor_free_space"] = dict(free_space.diagnostics)
        context["applied"] = applied
        from manaenv.pipeline.runtime.runtime_instruction import (
            attach_runtime_instruction_tags,
            write_runtime_instruction_from_context,
        )

        # Object kind tags ({fruit} → peach) must exist before template resolve.
        attach_runtime_instruction_tags(context, task_payload=task_payload)
        _attach_runtime_template_values(context, runtime_templates)
        write_runtime_instruction_from_context(env, env_id=int(env_id), context=context)
        contexts_by_env[int(env_id)] = context
        reports.append({"env_id": int(env_id), "episode_index": int(episode_token), **applied})
        for name, pose in sampled_poses.items():
            slot_plan = _asset_slot_plan_for_object(applied_assets, name)
            active_name = slot_plan.get("active_slot") or name
            if active_name not in pose_batches:
                raise RuntimeError(f"schema-vNext sampled pose for undeclared object: {active_name}")
            _append_pose_batch(pose_batches, active_name, int(env_id), _pose_row(pose))
            for index, inactive_name in enumerate(slot_plan.get("inactive_slots") or []):
                if inactive_name not in pose_batches:
                    continue
                _append_pose_batch(
                    pose_batches,
                    inactive_name,
                    int(env_id),
                    _inactive_pose_row(
                        index=runtime_slot_parking_indices.get(inactive_name, index),
                        quaternion=pose.quaternion,
                    ),
                )

    from .runtime_instruction import apply_runtime_instruction_for_envs

    apply_runtime_instruction_for_envs(
        env,
        env_id_list,
        task_payload=task_payload,
        runtime_instruction=runtime_instruction,
        plan_id=plan_key,
        contexts_by_env=contexts_by_env,
    )
    _register_runtime_context(env, plan_key, contexts_by_env)
    _append_schema_vnext_effect_report(env, reports)
    if not pose_batches:
        return

    import torch

    from manaenv.terms.events import _write_root_pose_with_zero_velocity

    for name, batch in pose_batches.items():
        rows = batch["rows"]
        batch_env_ids = batch["env_ids"]
        if not rows:
            continue
        asset = scene_assets[name]
        local_pose = torch.tensor(rows, dtype=torch.float32, device=env.device)
        batch_env_ids_tensor = torch.tensor(batch_env_ids, dtype=torch.long, device=env.device)
        sim_pose = None
        if asset_supports_canonical_pose_bridge(asset):
            sim_pose = canonical_pose_to_actor_pose(env, asset, local_pose, batch_env_ids_tensor)
        elif asset_supports_sim_root_pose(asset):
            # Articulations retain the pre-bridge direct-root behavior until a
            # real articulated-package integration contract is validated.
            sim_pose = local_pose.clone()
            sim_pose[:, :3] += env.scene.env_origins[batch_env_ids_tensor]
        _sync_pose_to_usd(asset, batch_env_ids_tensor, local_pose.clone())
        if sim_pose is None:
            continue
        # Teleporting only the root leaves articulated child links (e.g. glasses
        # temples) at their stale poses; the helper re-writes joint state and runs
        # forward kinematics so children follow the new root on reset.
        _write_root_pose_with_zero_velocity(asset, sim_pose, batch_env_ids_tensor)

    # Apply the configured collision policy, then gravity, AFTER pose writes.
    # The default drops parked spares out of pair testing. Tasks that keep
    # authored collision avoid a live PhysX collider reactivation boundary.
    runtime_option_values = dict(runtime_options or {})
    apply_runtime_slot_activation(
        env,
        scene_assets,
        slot_gravity_toggles,
        parked_slot_collision=str(runtime_option_values.get("parked_slot_collision", "disable")),
    )


apply_schema_vnext_runtime_sample._schema_vnext_runtime_sample_event = True


def _ignore_aligned_xforms(asset: Any, env_ids: Any) -> None:
    """Force ``/Aligned`` children to identity for fixed-pose hang-bar resets.

    Schema-vNext rigid placement uses the canonical↔actor pose bridge instead.
    Must re-apply every call: ``reset_scene_to_default`` / USD restore can bring
    ``/Aligned`` back. The prim-path set only dedupes log spam, not the clear.
    """
    try:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom
    except ImportError:
        return

    try:
        stage = omni.usd.get_context().get_stage()
    except Exception:
        return
    if stage is None:
        return

    for i, env_idx in enumerate(env_ids):
        prim_path = _asset_prim_path_for_env(asset, env_idx=int(env_idx), local_index=i)
        if not prim_path:
            continue
        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            continue
        cleared = 0
        for prim in Usd.PrimRange(root):
            if prim.GetName() != "Aligned":
                continue
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
            xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
            xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))
            cleared += 1
        if cleared and prim_path not in _ALIGNED_IGNORED_PRIMS:
            manaprint(f"FIX: ignore /Aligned -> identity ({cleared} prims) under {prim_path}")
            _ALIGNED_IGNORED_PRIMS.add(prim_path)


def _debug_dump_runtime_spawn_masses_once(
    env: Any,
    scene_assets: Mapping[str, Any],
    materialization: Mapping[str, Any] | None,
) -> None:
    """Print backend-neutral masses once so spawn_mass overrides can be verified.

    Always dumps runtime mass for every scene asset. Appends
    ``intended_spawn_mass`` only when ``object_spawn_masses`` declares one.
    """
    global _RUNTIME_SPAWN_MASS_DUMPED
    if _RUNTIME_SPAWN_MASS_DUMPED:
        return

    intended = {}
    if isinstance(materialization, Mapping):
        raw = materialization.get("object_spawn_masses")
        if isinstance(raw, Mapping):
            intended = {str(key): float(value) for key, value in raw.items()}

    manaprint("DEBUG: runtime mass dump (env0, once per process)")
    complete = True
    for name in sorted(scene_assets):
        asset = scene_assets[name]
        try:
            masses = _runtime_asset_masses(asset)
            env0_values = to_numpy_keep_shape(select_env_item(masses, 0))
            if env0_values is None:
                raise RuntimeError("runtime mass source returned no env0 values")
            env0 = env0_values.reshape(-1).tolist()
            logical = name.split("__slot_", 1)[0]
            want = intended.get(name, intended.get(logical))
            if want is None:
                manaprint(f"DEBUG:   {name}: runtime={env0}")
            else:
                manaprint(f"DEBUG:   {name}: runtime={env0} intended_spawn_mass={want:.6g} kg")
        except Exception as exc:
            complete = False
            manaprint(f"ERROR:   {name}: mass read failed: {exc}")
    _RUNTIME_SPAWN_MASS_DUMPED = complete


def _runtime_asset_masses(asset: Any) -> Any:
    """Read current rigid-body masses through the backend-neutral asset API."""
    data = getattr(asset, "data", None)
    masses = getattr(data, "body_mass", None)
    if masses is not None:
        return masses

    view = _asset_runtime_view(asset, "get_masses")
    if view is not None:
        return view.get_masses()
    raise RuntimeError("asset exposes neither data.body_mass nor a view with get_masses")


def _asset_runtime_view(asset: Any, *required_methods: str) -> Any | None:
    """Resolve a backend view that provides every requested capability."""
    for attr_name in ("root_view", "root_physx_view"):
        view = getattr(asset, attr_name, None)
        if view is not None and all(callable(getattr(view, method, None)) for method in required_methods):
            return view
    return None


def _runtime_sample_id(task_payload: Mapping[str, Any]) -> str:
    layout = task_payload.get("layout") if isinstance(task_payload.get("layout"), Mapping) else {}
    runtime = layout.get("runtime_sample") if isinstance(layout.get("runtime_sample"), Mapping) else {}
    return str(runtime.get("id") or "").strip()


def _schema_vnext_runtime_plan_sampler(
    runtime_plan: Mapping[str, Any] | None,
    *,
    task_payload: Mapping[str, Any],
    plan_key: str,
) -> RuntimePlanSampler | None:
    if isinstance(runtime_plan, Mapping) and runtime_plan:
        return RuntimePlanSampler(runtime_plan, plan_id=plan_key)
    if _schema_vnext_task_has_placements(task_payload):
        raise RuntimeError("schema_vnext_runtime_sampling_plan_missing")
    return None


def _schema_vnext_task_has_placements(task_payload: Mapping[str, Any]) -> bool:
    layout = task_payload.get("layout") if isinstance(task_payload.get("layout"), Mapping) else {}
    runtime = layout.get("runtime_sample") if isinstance(layout.get("runtime_sample"), Mapping) else {}
    targets = runtime.get("targets") if isinstance(runtime.get("targets"), Mapping) else {}
    placements = targets.get("placements") if isinstance(targets, Mapping) else []
    return bool(isinstance(placements, list) and placements)


def _sample_schema_vnext_effects(
    runtime_effects_plan: Mapping[str, Any] | None,
    *,
    plan_key: str,
    seed: int,
    env_id: int,
    state: SelectorRuntimeState,
) -> Any:
    if runtime_effects_plan is None:
        return type(
            "SchemaVNextEmptyEffects",
            (),
            {
                "runtime_sample_id": str(plan_key),
                "env_id": int(env_id),
                "placements": {},
                "assets": {},
                "lighting": {},
                "materials": {},
            },
        )()
    return sample_runtime_effects_payload(
        runtime_effects_plan,
        seed=int(seed),
        env_id=int(env_id),
        state=state,
    )


def _schema_vnext_task_has_effects(task_payload: Mapping[str, Any]) -> bool:
    layout = task_payload.get("layout") if isinstance(task_payload.get("layout"), Mapping) else {}
    runtime = layout.get("runtime_sample") if isinstance(layout.get("runtime_sample"), Mapping) else {}
    global_payload = runtime.get("global") if isinstance(runtime.get("global"), Mapping) else {}
    lighting = global_payload.get("lighting") if isinstance(global_payload, Mapping) else None
    targets = runtime.get("targets") if isinstance(runtime.get("targets"), Mapping) else {}
    assets = targets.get("assets") if isinstance(targets, Mapping) else None
    materials = targets.get("materials") if isinstance(targets, Mapping) else None
    return bool(lighting or assets or materials)


def _attach_schema_vnext_runtime_context_base(
    context: dict[str, Any],
    *,
    runtime_sample_id: str,
    episode_index: int,
    env_id: int,
    seed: int,
    materialization: Mapping[str, Any] | None,
) -> None:
    context["runtime_sample_id"] = str(runtime_sample_id)
    context["episode_index"] = int(episode_index)
    context["env_id"] = int(env_id)
    context["seed"] = int(seed)
    context.setdefault("selections", {})
    context.setdefault("domain_samples", {})
    context.setdefault("outputs", {})
    context["compile_context"] = _schema_vnext_compile_context(materialization)


def _schema_vnext_compile_context(materialization: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(materialization, Mapping):
        return {"materialized": {"background": {}, "objects": {}}}
    objects = materialization.get("objects") if isinstance(materialization.get("objects"), Mapping) else {}
    selected = (
        materialization.get("selected_candidate_refs")
        if isinstance(materialization.get("selected_candidate_refs"), Mapping)
        else {}
    )
    return {
        "materialized": {
            "background": {
                "candidate_ref_id": selected.get("background"),
                "asset_id": materialization.get("background_asset"),
            },
            "objects": {
                str(object_id): {
                    "candidate_ref_id": selected.get(f"object:{object_id}"),
                    "asset_id": asset_id,
                }
                for object_id, asset_id in objects.items()
            },
        },
        "scene_variants": {
            str(variant_id): {"choice_id": choice_id}
            for variant_id, choice_id in (
                materialization.get("variant_choices").items()
                if isinstance(materialization.get("variant_choices"), Mapping)
                else ()
            )
        },
    }


def _schema_vnext_selector_state(env: Any, plan_id: str) -> SelectorRuntimeState:
    state = getattr(env, "_schema_vnext_runtime_selector_state", None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, "_schema_vnext_runtime_selector_state", state)
    value = state.get(plan_id)
    if not isinstance(value, SelectorRuntimeState):
        value = SelectorRuntimeState()
        state[plan_id] = value
    return value


def _schema_vnext_effect_episode_indices(env: Any, plan_id: str, env_id_list: list[int]) -> list[int]:
    state = getattr(env, "_schema_vnext_runtime_effect_reset_counters", None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, "_schema_vnext_runtime_effect_reset_counters", state)
    counters = state.setdefault(str(plan_id or "runtime"), {})
    if not isinstance(counters, dict):
        counters = {}
        state[str(plan_id or "runtime")] = counters
    out: list[int] = []
    for env_id in env_id_list:
        key = int(env_id)
        value = int(counters.get(key, 0))
        out.append(value)
        counters[key] = value + 1
    return out


def _apply_schema_vnext_assets_for_env(
    env: Any,
    *,
    env_id: int,
    assets: Mapping[str, Mapping[str, Any]],
    plan_id: str,
) -> dict[str, Any]:
    if not assets:
        return {"applied": False, "reason": "empty"}
    reports: dict[str, Any] = {}
    by_object: dict[str, Any] = {}
    for entry_id, selection in sorted(assets.items()):
        target = selection.get("target") if isinstance(selection.get("target"), Mapping) else {}
        object_id = str(target.get("object_id") or target.get("id") or "").strip()
        active_slot = str(selection.get("slot") or object_id).strip()
        inactive_slots = [str(slot) for slot in selection.get("inactive_slots", []) or [] if str(slot)]
        asset_id = str(selection.get("asset_id") or "").strip()
        if not object_id or not active_slot:
            raise RuntimeError(f"schema_vnext_runtime_asset_selection_invalid:{entry_id}")
        if not asset_id:
            LOGGER.warning(
                "schema_vnext runtime asset selection produced empty asset_id for %s (env=%s); "
                "overlay/logs will show it as missing",
                object_id,
                env_id,
            )
        _record_active_asset_slot(
            env,
            plan_id=plan_id,
            env_id=int(env_id),
            object_id=object_id,
            active_slot=active_slot,
            asset_id=asset_id,
        )
        _set_slot_visibility(env, active_slot, env_id=int(env_id), visible=True)
        for inactive_slot in inactive_slots:
            _set_slot_visibility(env, inactive_slot, env_id=int(env_id), visible=False)
        report = {
            "object_id": object_id,
            "active_slot": active_slot,
            "inactive_slots": inactive_slots,
            "asset_id": asset_id,
            "candidate_id": selection.get("candidate_id"),
        }
        reports[str(entry_id)] = report
        by_object[object_id] = report
    # nut_on_screw: bind logical destination → screw_{letter} matching the
    # active nut pool candidate (a→screw_a … d→screw_d) for place/success.
    nut_rep = by_object.get("nut")
    if isinstance(nut_rep, Mapping):
        cand = str(nut_rep.get("candidate_id") or "").strip().lower()
        if cand in {"a", "b", "c", "d"}:
            screw_name = f"screw_{cand}"
            _record_active_asset_slot(
                env,
                plan_id=plan_id,
                env_id=int(env_id),
                object_id="destination",
                active_slot=screw_name,
                asset_id=screw_name,
            )

            manaprint(
                f"[vnext] nut_on_screw destination bind env={int(env_id)} " f"nut_cand={cand!r} -> {screw_name!r}"
            )
    return {"applied": True, "entries": reports, "by_object": by_object}


_NUT_ON_SCREW_LETTERS = ("a", "b", "c", "d")
_NUT_ON_SCREW_NAMES = tuple(f"screw_{letter}" for letter in _NUT_ON_SCREW_LETTERS)


def _ensure_nut_on_screw_matching_same_side(
    sampled_poses: dict[str, Any],
    *,
    applied_assets: Mapping[str, Any],
    env_id: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Swap matching screw onto the nut's left/right half-row when needed.

    Mid-band slots: left half y<0 (slot_0/1), right half y>0 (slot_2/3). Nut
    hand regions share the same sign. Letter→screw bind is unchanged; only
    poses are exchanged so the matched screw sits on the same side as the nut.
    """
    if not sampled_poses or "nut" not in sampled_poses:
        return sampled_poses
    by_object = applied_assets.get("by_object") if isinstance(applied_assets, Mapping) else None
    if not isinstance(by_object, Mapping):
        return sampled_poses
    nut_rep = by_object.get("nut")
    if not isinstance(nut_rep, Mapping):
        return sampled_poses
    cand = str(nut_rep.get("candidate_id") or "").strip().lower()
    if cand not in _NUT_ON_SCREW_LETTERS:
        return sampled_poses
    match_name = f"screw_{cand}"
    nut_pose = sampled_poses.get("nut")
    match_pose = sampled_poses.get(match_name)
    if nut_pose is None or match_pose is None:
        return sampled_poses
    if any(name not in sampled_poses for name in _NUT_ON_SCREW_NAMES):
        return sampled_poses

    nut_y = float(nut_pose.position[1])
    match_y = float(match_pose.position[1])
    # Same-sign y ⇒ already on the correct half-row.
    if nut_y == 0.0 or match_y == 0.0:
        return sampled_poses
    if (nut_y > 0.0) == (match_y > 0.0):
        return sampled_poses

    want_positive = nut_y > 0.0
    donors = [
        name
        for name in _NUT_ON_SCREW_NAMES
        if name != match_name
        and float(sampled_poses[name].position[1]) != 0.0
        and (float(sampled_poses[name].position[1]) > 0.0) == want_positive
    ]
    if not donors:
        return sampled_poses
    donor_name = str(rng.choice(donors))
    donor_pose = sampled_poses[donor_name]
    updated = dict(sampled_poses)
    updated[match_name] = donor_pose
    updated[donor_name] = match_pose

    manaprint(
        f"[vnext] nut_on_screw same_side swap env={int(env_id)} " f"{match_name!r}<->{donor_name!r} nut_y={nut_y:.3f}"
    )
    return updated


def _runtime_asset_slot_names(runtime_effects_plan: Mapping[str, Any] | None) -> set[str]:
    names: set[str] = set()
    if not isinstance(runtime_effects_plan, Mapping):
        return names
    for entry in runtime_effects_plan.get("assets", []) or []:
        if not isinstance(entry, Mapping):
            continue
        for slot in entry.get("slots", []) or []:
            if isinstance(slot, Mapping):
                name = str(slot.get("name") or "").strip()
            else:
                name = str(slot or "").strip()
            if name:
                names.add(name)
    return names


def _record_active_asset_slot(
    env: Any,
    *,
    plan_id: str,
    env_id: int,
    object_id: str,
    active_slot: str,
    asset_id: str,
) -> None:
    from manaenv.utils.asset_utils import record_active_asset_slot

    record_active_asset_slot(
        env,
        plan_id=plan_id,
        env_id=env_id,
        object_id=object_id,
        active_slot=active_slot,
        asset_id=asset_id,
    )


def _asset_slot_plan_for_object(applied_assets: Mapping[str, Any], object_id: str) -> Mapping[str, Any]:
    by_object = applied_assets.get("by_object") if isinstance(applied_assets.get("by_object"), Mapping) else {}
    return dict(by_object.get(str(object_id), {}) or {})


def _correct_active_slot_support_poses(
    sampled_poses: Mapping[str, Any],
    *,
    applied_assets: Mapping[str, Any],
    runtime_slot_bounds: Mapping[str, Any] | None,
    runtime_plan: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Move a selected cache slot along its support normal to preserve contact.

    Runtime placement is compiled from the logical object's geometry (slot 0),
    while the reset selector may activate a cache slot with a different local
    bbox.  Applying the same origin to that slot can leave it intersecting the
    tabletop; PhysX then ejects it and the next reset observes a fallen object.
    The correction is deliberately a pure pose adjustment: it preserves the
    sampled quaternion and keeps the active slot's full footprint on the
    effective support surface.
    """

    if not sampled_poses or not isinstance(runtime_slot_bounds, Mapping):
        return dict(sampled_poses)
    by_object = applied_assets.get("by_object") if isinstance(applied_assets, Mapping) else None
    if not isinstance(by_object, Mapping):
        return dict(sampled_poses)
    corrected = dict(sampled_poses)
    for object_id, pose in sampled_poses.items():
        report = by_object.get(str(object_id))
        if not isinstance(report, Mapping):
            continue
        active_slot = str(report.get("active_slot") or object_id).strip()
        if not active_slot:
            continue
        object_bounds = runtime_slot_bounds.get(str(object_id))
        if not isinstance(object_bounds, Mapping):
            continue
        reference_payload = object_bounds.get(str(object_id))
        active_payload = object_bounds.get(active_slot)
        if not isinstance(reference_payload, Mapping) or not isinstance(active_payload, Mapping):
            LOGGER.warning(
                "schema-vNext active slot support correction skipped for %s: missing bounds for %s/%s",
                object_id,
                object_id,
                active_slot,
            )
            continue
        surface = _runtime_support_surface(
            runtime_plan,
            object_id=str(object_id),
            context=context,
        )
        if surface is None:
            continue
        try:
            reference_bounds = Bounds3D.from_dict(dict(reference_payload))
            active_bounds = Bounds3D.from_dict(dict(active_payload))
            normal = _runtime_surface_vector(surface, "normal", default=(0.0, 0.0, 1.0))
            reference_offset = support_contact_offset(reference_bounds, pose.quaternion, normal)
            active_offset = support_contact_offset(active_bounds, pose.quaternion, normal)
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("schema-vNext active slot support correction skipped for %s: invalid bounds", object_id)
            continue
        delta = float(active_offset - reference_offset)
        position = tuple(float(value) for value in pose.position)
        if abs(delta) > 1e-9:
            position = tuple(position[index] + float(normal[index]) * delta for index in range(3))
        try:
            position = _clamp_pose_to_runtime_surface(
                position,
                quaternion=pose.quaternion,
                local_bounds=active_bounds,
                surface=surface,
            )
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("schema-vNext active slot surface clamp skipped for %s: invalid surface", object_id)
            continue
        if position == tuple(float(value) for value in pose.position):
            continue
        corrected[object_id] = replace(
            pose,
            position=position,
        )
    return corrected


def _runtime_support_surface(
    runtime_plan: Mapping[str, Any] | None,
    *,
    object_id: str,
    context: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(runtime_plan, Mapping):
        return None
    domains = [item for item in runtime_plan.get("domains", []) or [] if isinstance(item, Mapping)]
    domain_id = ""
    domain_samples = context.get("domain_samples") if isinstance(context, Mapping) else None
    sample = domain_samples.get(object_id) if isinstance(domain_samples, Mapping) else None
    if isinstance(sample, Mapping):
        domain_id = str(sample.get("domain_id") or "").strip()
    candidates = [domain for domain in domains if not domain_id or str(domain.get("id") or "") == domain_id]
    if domain_id and not candidates:
        return None
    if not domain_id:
        candidates = [domain for domain in domains if str(domain.get("object") or "") == object_id]
    surface_ids = {
        str(domain.get("frame") or "").split(":", 1)[1].strip()
        for domain in candidates
        if str(domain.get("frame") or "").startswith("surface:")
    }
    surface_ids.discard("")
    if len(surface_ids) != 1:
        return None
    surface_id = next(iter(surface_ids))
    raw_surfaces = runtime_plan.get("support_surfaces")
    if not isinstance(raw_surfaces, list):
        artifacts = runtime_plan.get("artifacts")
        raw_surfaces = artifacts.get("support_surfaces") if isinstance(artifacts, Mapping) else None
    matches = []
    for surface in raw_surfaces or []:
        if not isinstance(surface, Mapping) or str(surface.get("id") or "") != surface_id:
            continue
        matches.append(surface)
    return matches[0] if len(matches) == 1 else None


def _runtime_support_normal(
    runtime_plan: Mapping[str, Any] | None,
    *,
    object_id: str,
    context: Mapping[str, Any] | None,
) -> tuple[float, float, float] | None:
    """Return the unique support normal for a sampled surface domain."""

    surface = _runtime_support_surface(runtime_plan, object_id=object_id, context=context)
    if surface is None:
        return None
    try:
        return _runtime_surface_vector(surface, "normal", default=(0.0, 0.0, 1.0))
    except (TypeError, ValueError):
        return None


def _runtime_surface_vector(
    surface: Mapping[str, Any],
    key: str,
    *,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    raw = surface.get(key, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"runtime_surface_{key}_must_be_vec3")
    values = tuple(float(value) for value in raw)
    length = sum(value * value for value in values) ** 0.5
    if length <= 1e-9:
        raise ValueError(f"runtime_surface_{key}_zero_length")
    return tuple(value / length for value in values)


def _clamp_pose_to_runtime_surface(
    position: tuple[float, float, float],
    *,
    quaternion: Any,
    local_bounds: Bounds3D,
    surface: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Clamp an active slot's rotated footprint inside the effective surface bounds."""

    center_raw = surface.get("center", (0.0, 0.0, 0.0))
    if not isinstance(center_raw, (list, tuple)) or len(center_raw) != 3:
        raise ValueError("runtime_surface_center_must_be_vec3")
    center = tuple(float(value) for value in center_raw)
    x_axis = _runtime_surface_vector(surface, "x_axis", default=(1.0, 0.0, 0.0))
    y_axis = _runtime_surface_vector(surface, "y_axis", default=(0.0, 1.0, 0.0))
    raw_bounds = surface.get("bounds")
    if not isinstance(raw_bounds, Mapping):
        return position
    effective = Bounds2D.from_dict(dict(raw_bounds)).shrink(float(surface.get("margin", 0.0) or 0.0))
    if effective is None:
        return position
    footprint = projected_footprint_bounds(local_bounds, quaternion, x_axis, y_axis)
    local = vec_sub(position, center)
    local_x = dot(local, x_axis)
    local_y = dot(local, y_axis)
    origin_x = _clamp_surface_origin(
        local_x,
        effective.x.min - footprint.x.min,
        effective.x.max - footprint.x.max,
    )
    origin_y = _clamp_surface_origin(
        local_y,
        effective.y.min - footprint.y.min,
        effective.y.max - footprint.y.max,
    )
    return vec_add(
        position,
        vec_add(vec_mul(x_axis, origin_x - local_x), vec_mul(y_axis, origin_y - local_y)),
    )


def _clamp_surface_origin(value: float, minimum: float, maximum: float) -> float:
    if minimum > maximum:
        raise ValueError("runtime_surface_cannot_fit_active_slot")
    return min(max(float(value), float(minimum)), float(maximum))


def _append_pose_batch(pose_batches: dict[str, dict[str, list[Any]]], name: str, env_id: int, row: list[float]) -> None:
    batch = pose_batches.setdefault(str(name), {"env_ids": [], "rows": []})
    batch["env_ids"].append(int(env_id))
    batch["rows"].append(row)


def _pose_row(pose: Any) -> list[float]:
    return [
        float(pose.position[0]),
        float(pose.position[1]),
        float(pose.position[2]),
        float(pose.quaternion[0]),
        float(pose.quaternion[1]),
        float(pose.quaternion[2]),
        float(pose.quaternion[3]),
    ]


def _inactive_pose_row(*, index: int, quaternion: Any) -> list[float]:
    position = inactive_pool_slot_position(index, isolation_z=_INACTIVE_ISOLATION_Z)
    return [
        position[0],
        position[1],
        position[2],
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
    ]


def _collect_slot_gravity_toggles(
    slot_gravity_toggles: dict[str, dict[int, bool]],
    applied_assets: Mapping[str, Any],
    *,
    env_id: int,
) -> None:
    """Record per (slot, env) gravity and collision intent from the asset slot plan.

    Active slots keep gravity and collision (True); inactive spares get both
    disabled (False) so they rest at their parked pose and leave PhysX pair testing.
    """
    by_object = applied_assets.get("by_object") if isinstance(applied_assets, Mapping) else None
    if not isinstance(by_object, Mapping):
        return
    for report in by_object.values():
        if not isinstance(report, Mapping):
            continue
        active_slot = str(report.get("active_slot") or "").strip()
        if active_slot:
            slot_gravity_toggles.setdefault(active_slot, {})[int(env_id)] = True
        for inactive_slot in report.get("inactive_slots") or []:
            name = str(inactive_slot or "").strip()
            if name:
                slot_gravity_toggles.setdefault(name, {})[int(env_id)] = False


def _apply_slot_gravity_toggles(
    env: Any,
    scene_assets: Mapping[str, Any],
    slot_gravity_toggles: Mapping[str, Mapping[int, bool]],
) -> None:
    """Enable/disable per-body gravity through any capable backend view.

    GPU-safe and cheap: ``set_disable_gravities`` only flips a per-body flag and
    keeps the body in the simulation island (no recomposition), so parked spares
    stop falling without the reset stalls that disabling simulation caused.
    """
    global _GRAVITY_CTRL_LOGGED
    if not slot_gravity_toggles:
        return
    disabled_count = 0
    enabled_count = 0
    view_missing = 0
    for name, env_states in slot_gravity_toggles.items():
        asset = scene_assets.get(name) if isinstance(scene_assets, Mapping) else None
        if asset is None:
            asset = _resolve_asset(env, name, None)
        if asset is None:
            continue
        view = _asset_runtime_view(asset, "set_disable_gravities", "get_disable_gravities")
        if view is None:
            view_missing += 1
            continue
        enable_ids = [int(e) for e, keep_gravity in env_states.items() if keep_gravity]
        disable_ids = [int(e) for e, keep_gravity in env_states.items() if not keep_gravity]
        if _set_view_gravity(view, enable_ids, gravity_enabled=True):
            enabled_count += len(enable_ids)
        if _set_view_gravity(view, disable_ids, gravity_enabled=False):
            disabled_count += len(disable_ids)
    if not _GRAVITY_CTRL_LOGGED:
        _GRAVITY_CTRL_LOGGED = True
        LOGGER.info(
            "schema-vNext parked-spare gravity control active: gravity disabled on %d body-slots, "
            "re-enabled on %d body-slots (%d slots had no compatible gravity view).",
            disabled_count,
            enabled_count,
            view_missing,
        )


def _set_view_gravity(view: Any, env_ids: list[int], *, gravity_enabled: bool) -> bool:
    """Set the gravity flag for ``env_ids`` on one capable rigid-body view."""
    if not env_ids:
        return False
    try:
        import torch
        import warp as wp

        count = int(getattr(view, "count", 0))
        valid = [idx for idx in env_ids if 0 <= idx < count]
        if not valid:
            return False
        native_data = view.get_disable_gravities()
        if isinstance(native_data, wp.array):
            indices = wp.array(valid, dtype=wp.int32, device=native_data.device)
            value_shape = (len(valid), *tuple(native_data.shape[1:]))
            disabled = wp.full(
                value_shape,
                value=0 if gravity_enabled else 1,
                dtype=native_data.dtype,
                device=native_data.device,
            )
            wp.copy(native_data[indices], disabled)
            view.set_disable_gravities(native_data, indices)
            return True

        # Test doubles and backend-neutral ProxyArray implementations expose a
        # Torch view instead of a native Warp buffer.
        data = as_torch(native_data)
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"unsupported gravity buffer type: {type(data).__name__}")
        data = data.reshape(count).clone()
        flag = 0 if gravity_enabled else 1
        indices = torch.tensor(valid, device=data.device, dtype=torch.long)
        data[indices] = flag
        view.set_disable_gravities(data, indices)
        return True
    except Exception:
        LOGGER.debug("schema-vNext failed to set slot gravity flag", exc_info=True)
        return False


def _asset_collision_view(asset: Any) -> Any | None:
    """Resolve a backend view that can toggle per-env collision."""
    for attr_name in ("root_view", "root_physx_view"):
        view = getattr(asset, attr_name, None)
        if view is None:
            continue
        has_tensor = callable(getattr(view, "get_collision_enabled", None)) and callable(
            getattr(view, "set_collision_enabled", None)
        )
        has_methods = callable(getattr(view, "enable_collision", None)) and callable(
            getattr(view, "disable_collision", None)
        )
        if has_tensor or has_methods:
            return view
    return None


def _apply_slot_collision_toggles(
    env: Any,
    scene_assets: Mapping[str, Any],
    slot_collision_toggles: Mapping[str, Mapping[int, bool]],
) -> None:
    """Enable collision on active slots and disable it on parked spares.

    Parked dynamic CCD bodies otherwise stay in the PhysX broadphase and dominate
    ``PxSimulationFilterCallback::pairFound``. Collision disable removes them
    from pair testing without ``disable_simulation`` island recomposition.
    """
    global _COLLISION_CTRL_LOGGED
    if not slot_collision_toggles:
        return
    disabled_count = 0
    enabled_count = 0
    missing = 0
    for name, env_states in slot_collision_toggles.items():
        asset = scene_assets.get(name) if isinstance(scene_assets, Mapping) else None
        if asset is None:
            asset = _resolve_asset(env, name, None)
        if asset is None:
            missing += 1
            continue
        enable_ids = [int(e) for e, keep_collision in env_states.items() if keep_collision]
        disable_ids = [int(e) for e, keep_collision in env_states.items() if not keep_collision]
        view = _asset_collision_view(asset)
        enabled_ok = _set_view_collision(view, enable_ids, collision_enabled=True) if view is not None else False
        disabled_ok = _set_view_collision(view, disable_ids, collision_enabled=False) if view is not None else False
        if enable_ids and not enabled_ok:
            enabled_ok = _write_usd_slot_collision(asset, enable_ids, collision_enabled=True)
        if disable_ids and not disabled_ok:
            disabled_ok = _write_usd_slot_collision(asset, disable_ids, collision_enabled=False)
        if enabled_ok:
            enabled_count += len(enable_ids)
        if disabled_ok:
            disabled_count += len(disable_ids)
        if (enable_ids and not enabled_ok) or (disable_ids and not disabled_ok):
            missing += 1
    if not _COLLISION_CTRL_LOGGED:
        _COLLISION_CTRL_LOGGED = True
        LOGGER.info(
            "schema-vNext parked-spare collision control active: collision disabled on %d body-slots, "
            "re-enabled on %d body-slots (%d slots had no compatible collision view or USD collider).",
            disabled_count,
            enabled_count,
            missing,
        )


def apply_runtime_slot_activation(
    env: Any,
    scene_assets: Mapping[str, Any],
    slot_activation: Mapping[str, Mapping[int, bool]],
    *,
    parked_slot_collision: str = "disable",
) -> None:
    """Apply one active/parked slot map to collision and gravity controls.

    ``True`` marks an active workspace slot and ``False`` a parked spare. Pose
    writers must call this after zeroing root velocities. Under the default
    policy, collision is restored before gravity. ``parked_slot_collision='keep'``
    leaves authored collision enabled for every slot while still disabling
    gravity on spares.
    """
    if not slot_activation:
        return
    collision_policy = str(parked_slot_collision or "").strip().lower()
    if collision_policy not in {"disable", "keep"}:
        raise ValueError(f"unsupported parked_slot_collision policy: {parked_slot_collision!r}")
    if collision_policy == "disable":
        _apply_slot_collision_toggles(env, scene_assets, slot_activation)
    _apply_slot_gravity_toggles(env, scene_assets, slot_activation)


def _set_view_collision(view: Any, env_ids: list[int], *, collision_enabled: bool) -> bool:
    """Set collision for ``env_ids`` on a capable rigid/geometry view."""
    if not env_ids:
        return False
    try:
        import torch

        count = int(getattr(view, "count", 0) or 0)
        if count > 0:
            valid = [idx for idx in env_ids if 0 <= idx < count]
        else:
            valid = [int(idx) for idx in env_ids]
        if not valid:
            return False
        if (
            count > 0
            and callable(getattr(view, "get_collision_enabled", None))
            and callable(getattr(view, "set_collision_enabled", None))
        ):
            data = as_torch(view.get_collision_enabled())
            if not isinstance(data, torch.Tensor):
                raise TypeError(f"unsupported collision buffer type: {type(data).__name__}")
            data = data.reshape(count).clone()
            flag = 1 if collision_enabled else 0
            indices = torch.tensor(valid, device=data.device, dtype=torch.long)
            data[indices] = flag
            view.set_collision_enabled(data, indices)
            return True
        method_name = "enable_collision" if collision_enabled else "disable_collision"
        method = getattr(view, method_name, None)
        if not callable(method):
            return False
        method(valid)
        return True
    except Exception:
        LOGGER.debug("schema-vNext failed to set slot collision flag", exc_info=True)
        return False


def _runtime_usd_stage() -> Any:
    import omni.usd

    return omni.usd.get_context().get_stage()


def _iter_collision_prims(root_prim: Any, usd_physics: Any, usd: Any) -> list[Any]:
    if root_prim is None or not callable(getattr(root_prim, "IsValid", None)) or not root_prim.IsValid():
        return []
    prims = []
    for prim in usd.PrimRange(root_prim):
        if prim.HasAPI(usd_physics.CollisionAPI):
            prims.append(prim)
    return prims


def _write_collision_enabled_attr(prim: Any, *, enabled: bool, usd_physics: Any) -> bool:
    api = usd_physics.CollisionAPI(prim)
    attr = api.GetCollisionEnabledAttr() if api else None
    if attr is None and api is not None and callable(getattr(api, "CreateCollisionEnabledAttr", None)):
        attr = api.CreateCollisionEnabledAttr(bool(enabled))
    if attr is None:
        return False
    attr.Set(bool(enabled))
    return True


def _write_usd_slot_collision(asset: Any, env_ids: list[int], *, collision_enabled: bool) -> bool:
    """Toggle ``physics:collisionEnabled`` on CollisionAPI prims under each env slot."""
    if not env_ids:
        return False
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        return False
    try:
        stage = _runtime_usd_stage()
        if stage is None:
            return False
        wrote = False
        for env_id in env_ids:
            prim_path = _asset_prim_path_for_env(asset, env_idx=int(env_id), local_index=0)
            if not prim_path:
                continue
            root = stage.GetPrimAtPath(prim_path)
            for prim in _iter_collision_prims(root, UsdPhysics, Usd):
                if _write_collision_enabled_attr(prim, enabled=collision_enabled, usd_physics=UsdPhysics):
                    wrote = True
        return wrote
    except Exception:
        LOGGER.debug("schema-vNext failed to set USD slot collision", exc_info=True)
        return False


def _set_slot_visibility(env: Any, object_name: str, *, env_id: int, visible: bool) -> None:
    """Apply pool visibility only to USD-owned assets.

    PhysX/Fabric pool slots stay authored as visible. Their active/inactive
    state is represented by the simulation-owned root pose: inactive slots are
    parked below the workspace with gravity and collision disabled. Changing USD
    visibility for those roots creates a second render-state authority that can
    hide the Fabric-owned active slot in GPU async rendering.
    """
    asset = _resolve_asset(env, object_name, None)
    if asset is None:
        return
    if not _should_author_pose_owned_visibility_to_usd(asset):
        return
    try:
        import omni.usd
        from pxr import UsdGeom
    except ImportError:
        return
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim_path = _asset_prim_path_for_env(asset, env_idx=int(env_id), local_index=0)
        if not prim_path:
            return
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible() if visible else imageable.MakeInvisible()
    except Exception:
        LOGGER.debug("schema-vNext failed to set slot visibility for %s", object_name, exc_info=True)


def _apply_schema_vnext_lighting(
    lighting: Mapping[str, Mapping[str, Any]],
    *,
    env_id: int | None = None,
) -> dict[str, Any]:
    if not lighting:
        return {"applied": False, "reason": "empty"}
    report = {"applied": False, "channels": {key: dict(value) for key, value in lighting.items()}}
    try:
        import omni.usd
        from pxr import UsdLux
    except Exception as exc:
        raise RuntimeError(f"schema_vnext_lighting_apply_failed: usd_unavailable:{type(exc).__name__}") from exc
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("schema_vnext_lighting_apply_failed: stage_unavailable")
        if any(channel.startswith("dome.") for channel in lighting):
            from manaenv.pipeline.generators.scene.utils.lighting_utils import dome_lights_disabled

            if dome_lights_disabled():
                # Preview mode: all DomeLights were silenced post-spawn; do not
                # recreate /World/DomeLight and undo the inspection setup.
                report["dome_light"] = {"applied": False, "reason": "MANAENV_DISABLE_DOME_LIGHT"}
                manaprint("INFO: MANAENV_DISABLE_DOME_LIGHT — skipped dome lighting apply")
            else:
                light, created = _resolve_or_create_dome_light(stage, UsdLux=UsdLux)
                rotation_op = None
                if "dome.intensity" in lighting:
                    intensity = lighting["dome.intensity"]
                    light.CreateIntensityAttr(float(intensity["value"]))
                    if "dome_exposure" in intensity:
                        light.CreateExposureAttr(float(intensity["dome_exposure"]))
                if "dome.texture" in lighting and lighting["dome.texture"].get("uri"):
                    light.CreateTextureFileAttr(str(lighting["dome.texture"]["uri"]))
                    if created:
                        light.CreateTextureFormatAttr("latlong")
                if "dome.color_temperature" in lighting:
                    light.CreateEnableColorTemperatureAttr(True)
                    light.CreateColorTemperatureAttr(float(lighting["dome.color_temperature"]["value"]))
                if "dome.rotation" in lighting:
                    rotation_op = _set_dome_light_rotation(light, float(lighting["dome.rotation"]["value"]))
                role_attr = light.GetPrim().GetAttribute("lighting:role")
                report["dome_light"] = {
                    "prim_path": str(light.GetPath()),
                    "created": bool(created),
                    "lighting_role": str(role_attr.Get() or "") if role_attr.IsValid() else "",
                    "rotation_op": rotation_op,
                }
                report["applied"] = True
        if "tabletop_rect.enabled" in lighting or "tabletop_rect.intensity" in lighting:
            report["tabletop_rect_light"] = _apply_tabletop_rect_lighting(stage, lighting, UsdLux=UsdLux, env_id=env_id)
            if report["tabletop_rect_light"].get("applied"):
                report["applied"] = True
        return report
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("schema_vnext_lighting_apply_failed:"):
            raise
        raise RuntimeError(f"schema_vnext_lighting_apply_failed: apply_failed:{type(exc).__name__}:{exc}") from exc


def _scope_prims_to_env(prims: list[Any], *, env_id: int | None) -> list[Any]:
    """Keep only the prims cloned under ``/World/envs/env_<id>`` when there are any.

    Lighting is sampled once per environment, so an environment's draw must not
    reach its neighbours' lights: with a stage-wide apply the last environment of
    the reset loop would decide the state of all of them. Stages without per-env
    clones (single env, or lights authored above the clone root) fall back to the
    full list.
    """
    if env_id is None:
        return prims
    env_token = f"/envs/env_{int(env_id)}/"
    scoped = [prim for prim in prims if env_token in str(prim.GetPath())]
    return scoped or prims


def _apply_tabletop_rect_lighting(
    stage: Any,
    lighting: Mapping[str, Mapping[str, Any]],
    *,
    UsdLux: Any,
    env_id: int | None = None,
) -> dict[str, Any]:
    """Toggle/set this environment's ``tabletop_local_key`` RectLights for a reset.

    ``tabletop_rect.enabled`` (Bernoulli/boolean sample) turns desk lights on or
    off. When on, ``tabletop_rect.intensity`` is applied if present. Inactive
    prims are included so a prior off sample (or post-spawn silence) can be
    turned back on. ``MANAENV_ENABLE_TABLETOP_RECT_LIGHT=1`` forces on.
    """

    from manaenv.pipeline.generators.scene.utils.lighting_utils import tabletop_rect_lights_enabled

    enabled_payload = lighting.get("tabletop_rect.enabled")
    intensity_payload = lighting.get("tabletop_rect.intensity")
    if tabletop_rect_lights_enabled():
        enabled = True
        enabled_source = "MANAENV_ENABLE_TABLETOP_RECT_LIGHT"
    elif enabled_payload is not None:
        enabled = bool(enabled_payload.get("value"))
        enabled_source = "sample"
    elif intensity_payload is not None:
        # Intensity-only authoring implies the light should be on.
        enabled = True
        enabled_source = "intensity_implies_on"
    else:
        return {"applied": False, "reason": "empty"}

    rect_prims = _scope_prims_to_env(
        _find_tabletop_rect_lights(stage, UsdLux=UsdLux, include_inactive=True),
        env_id=env_id,
    )
    if enabled and not rect_prims:
        raise RuntimeError("schema_vnext_lighting_apply_failed: tabletop_rect_light_missing")

    intensity = None
    if enabled and intensity_payload is not None and intensity_payload.get("value") is not None:
        intensity = float(intensity_payload["value"])

    paths: list[str] = []
    for prim in rect_prims:
        light = UsdLux.RectLight(prim)
        if enabled:
            if intensity is not None:
                light.CreateIntensityAttr(intensity)
            prim.SetActive(True)
        else:
            light.CreateIntensityAttr(0.0)
            prim.SetActive(False)
        paths.append(str(prim.GetPath()))

    return {
        "applied": True,
        "enabled": enabled,
        "enabled_source": enabled_source,
        "intensity": intensity,
        "prim_paths": paths,
        "count": len(paths),
    }


def _resolve_or_create_dome_light(stage: Any, *, UsdLux: Any = None) -> tuple[Any, bool]:
    if UsdLux is None:
        from pxr import UsdLux as _UsdLux

        UsdLux = _UsdLux
    dome_prims = [
        prim
        for prim in stage.Traverse()
        if prim.IsActive() and prim.IsDefined() and not prim.IsInPrototype() and prim.IsA(UsdLux.DomeLight)
    ]
    if len(dome_prims) > 1:
        paths = ",".join(sorted(str(prim.GetPath()) for prim in dome_prims))
        raise RuntimeError(f"schema_vnext_lighting_apply_failed: dome_light_ambiguous:{paths}")
    if dome_prims:
        return UsdLux.DomeLight(dome_prims[0]), False

    from pxr import Sdf

    return UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")), True


def _find_tabletop_rect_lights(stage: Any, *, UsdLux: Any = None, include_inactive: bool = False) -> list[Any]:
    """Return prims tagged ``lighting:role=tabletop_local_key``.

    Inactive prims are optional so reset can re-enable lights silenced after
    spawn or by a previous off sample. Traversal normally skips inactive prims,
    so inactive discovery walks via ``GetPseudoRoot`` + ``Usd.PrimRange`` with
    ``Usd.TraverseInstanceProxies`` is unnecessary; use stage pseudo-root range
    that includes inactive when requested.
    """

    if UsdLux is None:
        from pxr import UsdLux as _UsdLux

        UsdLux = _UsdLux
    matches: list[Any] = []
    if include_inactive:
        from pxr import Usd

        root = stage.GetPseudoRoot()
        prim_iter = Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
    else:
        prim_iter = stage.Traverse()
    for prim in prim_iter:
        if prim.IsInPrototype() or not prim.IsDefined():
            continue
        if not include_inactive and not prim.IsActive():
            continue
        if not prim.IsA(UsdLux.RectLight):
            continue
        role_attr = prim.GetAttribute("lighting:role")
        if role_attr.IsValid() and str(role_attr.Get() or "") == "tabletop_local_key":
            matches.append(prim)
    return matches


def _resolve_tabletop_rect_light(stage: Any, *, UsdLux: Any = None) -> Any:
    """Resolve the unique *active* tabletop RectLight (legacy intensity-only path)."""

    if UsdLux is None:
        from pxr import UsdLux as _UsdLux

        UsdLux = _UsdLux
    matches = _find_tabletop_rect_lights(stage, UsdLux=UsdLux, include_inactive=False)
    if not matches:
        raise RuntimeError("schema_vnext_lighting_apply_failed: tabletop_rect_light_missing")
    if len(matches) > 1:
        paths = ",".join(sorted(str(prim.GetPath()) for prim in matches))
        raise RuntimeError(f"schema_vnext_lighting_apply_failed: tabletop_rect_light_ambiguous:{paths}")
    return UsdLux.RectLight(matches[0])


def _set_dome_light_rotation(light: Any, rotation: float) -> str:
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(light.GetPrim())
    op_name = "xformOp:rotateZ:schema_vnext_runtime"
    matches = [op for op in xform.GetOrderedXformOps() or () if op.GetOpName() == op_name]
    if len(matches) > 1:
        raise RuntimeError("schema_vnext_lighting_apply_failed: duplicate_runtime_rotation_op")
    if matches:
        op = matches[0]
    else:
        op = xform.AddRotateZOp(opSuffix="schema_vnext_runtime")
    op.Set(float(rotation) * 180.0 / 3.141592653589793)
    return op.GetOpName()


def _apply_schema_vnext_materials_for_env(
    env: Any,
    *,
    env_id: int,
    materials: Mapping[str, Mapping[str, Any]],
    base_seed: int,
    task_id: str | None,
    runtime_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not materials:
        return {"applied": False, "reason": "empty"}
    config = _material_randomization_config_from_vnext_selection(materials)
    try:
        from manaenv.pipeline.randomization.asset_randomizer.adapters.manaenv_events import randomize_materials_on_reset
    except Exception as exc:
        raise RuntimeError(
            f"schema_vnext_material_apply_failed: asset_randomizer_unavailable:{type(exc).__name__}"
        ) from exc
    options = dict(runtime_options or {})
    try:
        randomize_materials_on_reset(
            env,
            [int(env_id)],
            config=config,
            base_seed=int(base_seed),
            task_id=task_id,
            cache_root=options.get("cache_root"),
            report_limit=int(options.get("report_limit", 64)),
            reset_renderer_accumulation_after_apply=bool(options.get("reset_renderer_accumulation_after_apply", True)),
            reset_rtx_sensors_after_apply=bool(options.get("reset_rtx_sensors_after_apply", True)),
        )
        return {"applied": True, "config": config}
    except Exception as exc:
        LOGGER.warning("schema-vNext material runtime apply failed: %s", exc, exc_info=True)
        raise RuntimeError(f"schema_vnext_material_apply_failed: apply_failed:{type(exc).__name__}:{exc}") from exc


def _material_randomization_config_from_vnext_selection(materials: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    templates: dict[str, dict[str, Any]] = {}
    adjustments: dict[str, dict[str, Any]] = {}
    profiles: list[dict[str, Any]] = []
    for entry_id, selection in sorted(materials.items()):
        candidate_id = str(selection.get("candidate_id") or "").strip()
        policy_type = str(selection.get("policy_type") or "").strip()
        target = selection.get("target") if isinstance(selection.get("target"), Mapping) else {}
        targets: dict[str, Any] = {
            "material_slots": target.get("material_slots", "auto"),
        }
        objects = list(target.get("objects") or [])
        prim_paths = list(target.get("prim_paths") or [])
        if objects:
            targets["asset_names"] = objects
        if prim_paths:
            targets["prim_paths"] = prim_paths
        apply_options = dict(selection.get("apply") or {})
        profile = {
            "id": str(entry_id),
            "type": policy_type,
            "targets": targets,
            "apply": apply_options,
        }
        if policy_type == "material_template":
            template_payload = selection.get("material_template")
            candidate = selection.get("candidate") if isinstance(selection.get("candidate"), Mapping) else {}
            if isinstance(template_payload, Mapping) and template_payload:
                templates[candidate_id] = _material_template_with_apply_overrides(
                    dict(template_payload),
                    apply_options=apply_options,
                )
            elif candidate:
                templates[candidate_id] = _material_template_candidate_from_payload(candidate_id, candidate)
            else:
                templates[candidate_id] = _material_template_candidate(candidate_id)
            profile["library"] = "schema_vnext_runtime_templates"
            profile["material_id"] = candidate_id
        elif policy_type == "material_local_adjustment":
            adjustments[candidate_id] = _material_adjustment_candidate(candidate_id)
            profile["library"] = "schema_vnext_runtime_adjustments"
            profile["adjustment_id"] = candidate_id
        else:
            continue
        profiles.append(profile)
    return {
        "schema_version": 1,
        "enabled": True,
        "seed_scope": "reset",
        "material_libraries": {
            "templates": {
                "id": "schema_vnext_runtime_templates",
                "materials": list(templates.values()),
            },
            "local_adjustments": {
                "id": "schema_vnext_runtime_adjustments",
                "adjustments": list(adjustments.values()),
            },
        },
        "material_randomization_profiles": profiles,
    }


def _material_template_candidate_from_payload(candidate_id: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    texture_ref = str(candidate.get("texture") or "").strip()
    if not texture_ref:
        raise RuntimeError(f"schema_vnext_material_template_texture_missing: {candidate_id}")
    channel = str(candidate.get("channel") or "base_color").strip() or "base_color"
    uri = _resolve_schema_vnext_texture_uri(
        texture_ref,
        object_asset_id=str(candidate.get("object_asset_id") or "").strip() or None,
    )
    return {
        "id": str(candidate_id),
        "tags": ["schema_vnext", str(candidate_id)],
        "channels": {
            channel: {"source": "texture", "uri": uri},
        },
    }


def _resolve_schema_vnext_texture_uri(texture_ref: str, *, object_asset_id: str | None = None) -> str:
    ref = str(texture_ref or "").strip()
    if not ref:
        raise RuntimeError("schema_vnext_texture_ref_empty")
    if _is_direct_texture_ref(ref):
        return ref
    if object_asset_id:
        try:
            from manaenv.assets.local_discovery import resolve_usd_path

            usd_path = resolve_usd_path(object_asset_id)
            return _derive_sibling_texture_uri(usd_path, ref)
        except Exception:
            LOGGER.debug("schema-vNext texture context resolve failed for %s", ref, exc_info=True)
    try:
        from manaenv.assets.local_discovery import resolve_texture_paths

        matches = resolve_texture_paths(gen_method=0, tags=ref, category="object")
        if matches:
            return matches[0]
    except Exception:
        LOGGER.debug("schema-vNext texture registry resolve failed for %s", ref, exc_info=True)
    raise RuntimeError(f"schema_vnext_texture_ref_unresolved: {ref}")


def _is_direct_texture_ref(texture_ref: str) -> bool:
    ref = str(texture_ref or "").strip()
    return "://" in ref or ref.startswith("/") or ref.startswith("~")


def _derive_sibling_texture_uri(usd_path: str, texture_name: str) -> str:
    base = str(usd_path).rsplit("/", 1)[0]
    suffix = str(texture_name).rsplit("/", 1)[-1]
    if "." not in suffix.rsplit("/", 1)[-1]:
        suffix = f"{suffix}.jpg"
    return f"{base}/textures/{suffix}"


def _material_template_with_apply_overrides(
    template: dict[str, Any],
    *,
    apply_options: Mapping[str, Any],
) -> dict[str, Any]:
    compatibility = dict(template.get("compatibility") or {})
    if "uv_tile_size_m" in apply_options:
        try:
            tile = float(apply_options["uv_tile_size_m"])
        except (TypeError, ValueError):
            tile = 0.0
        if tile > 0.0:
            compatibility["uv_tile_size_m"] = tile
            if "texture_scale" not in apply_options:
                compatibility["texture_scale"] = 1.0 / tile
    if "texture_scale" in apply_options:
        compatibility["texture_scale"] = apply_options["texture_scale"]
    template["compatibility"] = compatibility
    return template


def _material_template_candidate(candidate_id: str) -> dict[str, Any]:
    palette = {
        "red_plastic": ([1.0, 0.04, 0.02], 0.32, 0.0),
        "blue_matte": ([0.03, 0.18, 1.0], 0.86, 0.0),
        "green_semigloss": ([0.02, 0.82, 0.18], 0.42, 0.0),
        "dark_metal": ([0.05, 0.055, 0.06], 0.18, 1.0),
    }
    if str(candidate_id) not in palette:
        raise RuntimeError(f"schema_vnext_material_template_unknown: {candidate_id}")
    color, roughness, metallic = palette[str(candidate_id)]
    return {
        "id": str(candidate_id),
        "tags": ["schema_vnext", str(candidate_id)],
        "channels": {
            "base_color": {"source": "constant", "value": color},
            "roughness": {"source": "constant", "value": roughness},
            "metallic": {"source": "constant", "value": metallic},
        },
    }


def _material_adjustment_candidate(candidate_id: str) -> dict[str, Any]:
    if str(candidate_id) == "slightly_darker":
        scale_range = [0.65, 0.75]
    elif str(candidate_id) == "slightly_brighter":
        scale_range = [1.20, 1.35]
    else:
        raise RuntimeError(f"schema_vnext_material_adjustment_unknown: {candidate_id}")
    return {
        "id": str(candidate_id),
        "tags": ["schema_vnext", str(candidate_id)],
        "channels": {
            "base_color": {
                "constant": {
                    "op": "jitter_relative",
                    "scale_range": scale_range,
                    "default": [0.55, 0.55, 0.55],
                }
            }
        },
    }


def _append_schema_vnext_effect_report(env: Any, reports: list[dict[str, Any]]) -> None:
    if not reports:
        return
    state = getattr(env, "_schema_vnext_runtime_effect_reports", None)
    if not isinstance(state, list):
        state = []
        setattr(env, "_schema_vnext_runtime_effect_reports", state)
    state.extend(reports)
    del state[:-128]


def _placements_payload(sampled_poses: Mapping[str, RuntimePoseSample]) -> dict[str, dict[str, list[float]]]:
    return {
        object_id: {
            "position": [float(value) for value in pose.position],
            "quaternion": [float(value) for value in pose.quaternion],
        }
        for object_id, pose in sorted(sampled_poses.items())
    }


def _attach_place_beside_direction(
    context: dict[str, Any],
    *,
    task_payload: Mapping[str, Any],
    rng: random.Random,
    placements: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    from manaenv.pipeline.runtime.beside_placement import attach_place_beside_to_context, definition_beside_placement

    spec = definition_beside_placement(task_payload)
    if not spec:
        return
    if placements is None:
        applied = context.get("applied") if isinstance(context.get("applied"), Mapping) else {}
        placements = applied.get("placements") if isinstance(applied, Mapping) else {}
    if not isinstance(placements, Mapping):
        return
    attach_place_beside_to_context(context, placements=placements, spec=spec, rng=rng)


def _place_beside_keepout_rects(
    context: Mapping[str, Any],
    *,
    task_payload: Mapping[str, Any],
    sampled_poses: Mapping[str, RuntimePoseSample],
    runtime_plan: Mapping[str, Any] | None,
) -> dict[str, list[Any]]:
    """Surface-local keep-outs around the beside reference block.

    Clears all four cardinal place bands (left/right/up/down) so clutter cannot
    sit in any beside success / approach zone. Uses the task's ``gap`` and
    ``zone_half_width``; keyed by support surface id.
    """

    from manaenv.pipeline.runtime.beside_placement import (
        ALL_DIRECTIONS,
        definition_beside_placement,
        direction_zone_ranges,
    )

    spec = definition_beside_placement(task_payload)
    if not spec:
        return {}
    reference = str(spec.get("reference") or "block").strip()
    pose = sampled_poses.get(reference)
    surface = _runtime_support_surface(runtime_plan, object_id=reference, context=context)
    if pose is None or not isinstance(surface, Mapping):
        return {}
    gap = float(spec.get("gap", 0.20))
    half_width = float(spec.get("zone_half_width", 0.07))
    world_center = tuple(float(value) for value in pose.position)
    rects: list[Any] = []
    for direction in ALL_DIRECTIONS:
        zone = direction_zone_ranges(direction, gap=gap, half_width=half_width)
        try:
            rects.append(
                surface_local_rect_from_world_zone(
                    surface,
                    world_center=world_center,
                    x_range=(float(zone["x_range"][0]), float(zone["x_range"][1])),
                    y_range=(float(zone["y_range"][0]), float(zone["y_range"][1])),
                )
            )
        except (IndexError, TypeError, ValueError):
            continue
    surface_id = str(surface.get("id") or "")
    return {surface_id: rects} if surface_id and rects else {}


def _attach_runtime_template_values(context: dict[str, Any], runtime_templates: Mapping[str, Any] | None) -> None:
    if not isinstance(runtime_templates, Mapping) or not runtime_templates:
        return
    if isinstance(runtime_templates.get("fields"), Mapping):
        context["template_fields"] = resolve_runtime_template_fields(runtime_templates, context)
        return
    resolved: dict[str, Any] = {}
    for name, spec in runtime_templates.items():
        if isinstance(spec, Mapping):
            resolved[str(name)] = resolve_runtime_template_fields(spec, context)
    if resolved:
        context["template_fields"] = resolved


def _register_runtime_context(
    env: Any,
    plan_id: str | None,
    contexts_by_env: Mapping[int, Mapping[str, Any]],
) -> None:
    if not contexts_by_env:
        return
    plan_key = str(plan_id or "runtime")
    env_ids = sorted(int(env_id) for env_id in contexts_by_env)
    payloads = {int(env_id): dict(context) for env_id, context in contexts_by_env.items()}

    generic_api = getattr(env, "set_runtime_context", None)
    if callable(generic_api):
        try:
            generic_api("scene_layout", plan_key, env_ids, payloads)
            return
        except TypeError:
            try:
                generic_api(namespace="scene_layout", plan_id=plan_key, env_ids=env_ids, payloads=payloads)
                return
            except TypeError:
                pass

    try:
        from manaenv.runtime_context import set_runtime_context

        set_runtime_context(env, "scene_layout", plan_key, env_ids, payloads)
        return
    except Exception as exc:
        LOGGER.debug("schema-vNext generic runtime context API unavailable: %s", exc, exc_info=True)

    try:
        from ..adapters.manaenv_runtime_context import set_runtime_context

        set_runtime_context(env, "scene_layout", plan_key, env_ids, payloads)
        return
    except Exception as exc:
        LOGGER.debug("release runtime context adapter unavailable: %s", exc, exc_info=True)

    raise RuntimeError("schema_vnext_runtime_context_api_unavailable")


def _runtime_episode_indices(env: Any, plan_id: str | None, env_id_list: list[int]) -> list[int]:
    """Return deterministic per-env episode indices and advance reset counters."""

    state = getattr(env, "_scene_layout_runtime_reset_counters", None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, "_scene_layout_runtime_reset_counters", state)

    plan_key = str(plan_id or "runtime")
    counters = state.setdefault(plan_key, {})
    if not isinstance(counters, dict):
        counters = {}
        state[plan_key] = counters

    values: list[int] = []
    for env_id in env_id_list:
        key = int(env_id)
        value = int(counters.get(key, 0))
        values.append(value)
        counters[key] = value + 1
    return values


def _resolve_asset(env: Any, object_name: str, asset_cfgs: Mapping[str, Any] | None) -> Any | None:
    cfg = dict(asset_cfgs or {}).get(object_name, object_name)
    scene_name = _scene_entity_name(cfg)
    try:
        return env.scene[scene_name]
    except Exception:
        return None


def _scene_entity_name(cfg: Any) -> str:
    try:
        from manaenv.utils.asset_utils import get_scene_entity_name

        return str(get_scene_entity_name(cfg))
    except Exception:
        pass
    if hasattr(cfg, "name"):
        return str(cfg.name)
    return str(cfg)


def _base_quat(asset: Any, local_index: int, env_ids: Any) -> Quat:
    from manaenv.utils.tensor_utils import as_torch

    try:
        raw = as_torch(asset.data.root_quat_w)[env_ids][local_index].detach().cpu().tolist()
        return _quat(raw)
    except Exception as exc:
        LOGGER.debug(
            "scene_layout_solver failed to read runtime base quaternion; using identity: %s", exc, exc_info=True
        )
        return (0.0, 0.0, 0.0, 1.0)


def _base_pose(env: Any, asset: Any, local_index: int, env_ids: Any) -> tuple[tuple[float, float, float], Quat]:
    from manaenv.utils.tensor_utils import as_torch

    if asset_supports_canonical_pose_bridge(asset):
        selected_env_ids = env_ids[local_index : local_index + 1]
        root_pose = as_torch(asset.data.root_pose_w)[selected_env_ids].detach().clone()
        root_pose = actor_pose_to_canonical_pose(env, asset, root_pose, selected_env_ids)
        row = root_pose[0].detach().cpu().tolist()
        return ((float(row[0]), float(row[1]), float(row[2])), _quat(row[3:7]))

    quat = _base_quat(asset, local_index, env_ids)
    try:
        raw_pos = as_torch(asset.data.root_pos_w)[env_ids][local_index].detach().cpu().tolist()
        pos = [float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2])]
        try:
            raw_origin = as_torch(env.scene.env_origins)[env_ids][local_index].detach().cpu().tolist()
            pos = [pos[i] - float(raw_origin[i]) for i in range(3)]
        except Exception as exc:
            LOGGER.debug(
                "scene_layout_solver failed to subtract env origin for runtime base pose: %s",
                exc,
                exc_info=True,
            )
        return ((pos[0], pos[1], pos[2]), quat)
    except Exception as exc:
        LOGGER.debug(
            "scene_layout_solver failed to read runtime base position; using origin: %s",
            exc,
            exc_info=True,
        )
        return ((0.0, 0.0, 0.0), quat)


def _quat(raw: Any) -> Quat:
    try:
        return quat_normalize(raw)
    except Exception:
        LOGGER.debug("scene_layout_solver invalid runtime base quaternion; using identity: %r", raw, exc_info=True)
        return (0.0, 0.0, 0.0, 1.0)


def _sync_pose_to_usd(asset: Any, env_ids: Any, local_pose: Any) -> None:
    if not _should_sync_runtime_pose_to_usd(asset):
        return

    try:
        import omni.usd
        from pxr import Gf, UsdGeom
    except ImportError:
        LOGGER.debug("scene_layout_solver USD modules are unavailable; skipping runtime USD pose sync", exc_info=True)
        return

    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            LOGGER.debug("scene_layout_solver USD stage is unavailable; skipping runtime USD pose sync")
            return
        for i, env_idx in enumerate(env_ids):
            idx = int(env_idx)
            prim_path = _asset_prim_path_for_env(asset, env_idx=idx, local_index=i)
            if not prim_path:
                continue
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            p = local_pose[i, :3].detach().cpu().tolist()
            q = local_pose[i, 3:7].detach().cpu().tolist()
            xformable = UsdGeom.Xformable(prim)
            if not set_existing_pose_xform_ops(
                xformable,
                position_xyz=p,
                quaternion_xyzw=q,
                gf=Gf,
                usd_geom=UsdGeom,
            ):
                LOGGER.warning(
                    "scene_layout_solver cannot safely sync runtime pose to USD without existing "
                    "xformOp:translate and xformOp:orient on %s",
                    prim_path,
                )
                continue

    except Exception as exc:
        LOGGER.warning("scene_layout_solver failed to sync runtime pose to USD: %s", exc, exc_info=True)


def _asset_prim_path_for_env(asset: Any, *, env_idx: int, local_index: int) -> str:
    if not asset_supports_canonical_pose_bridge(asset):
        # Preserve the pre-bridge path for non-physical assets and unsupported
        # articulations. Their runtime pose semantics are intentionally unchanged.
        prim_paths = getattr(asset, "prim_paths", None)
        if prim_paths is not None:
            try:
                values = list(prim_paths)
                if values:
                    index = int(env_idx) if 0 <= int(env_idx) < len(values) else int(local_index)
                    value = values[min(index, len(values) - 1)]
                    if value:
                        return str(value)
            except Exception:
                pass

    # Supported rigid runtime samples are canonical poses for the outer/default
    # prim. Internal actor paths such as ``.../Aligned`` must retain their fixed
    # package transform.
    cfg = getattr(asset, "cfg", None)
    raw = str(getattr(cfg, "prim_path", "") or getattr(asset, "prim_path", "") or "")
    if not raw:
        return ""
    raw = raw.replace("{ENV_REGEX_NS}", f"/World/envs/env_{int(env_idx)}")
    raw = raw.replace("/World/envs/env_.*", f"/World/envs/env_{int(env_idx)}")
    raw = raw.replace("env_.*", f"env_{int(env_idx)}")
    if not raw.startswith("/"):
        raw = "/" + raw
    return raw
