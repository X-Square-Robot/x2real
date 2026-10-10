"""Asset inspection and lookup utilities.

Helpers for inspecting USD assets at task-configuration time, e.g.
detecting whether an object is rigid or articulated and listing its
rigid bodies for contact-sensor setup.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # keep import-time lightweight; heavy deps are imported lazily


class AssetNameProxy:
    """Lightweight proxy whose ``.name`` always returns a ``str``.

    ``_patch_runtime_cfgs`` may replace the internal ``_name`` attribute
    with a list when the semantic tag resolves to multiple assets.  The
    property transparently picks the first element so that downstream
    consumers (``ObjectVelocityRecorderCfg(object_name=...)``) always
    receive a single string.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        n = self._name
        if isinstance(n, list):
            return n[0] if n else ""
        return n


def collapse_single_item(value: Any) -> Any:
    """Recursively unwrap single-element lists."""
    if isinstance(value, list) and len(value) == 1:
        return collapse_single_item(value[0])
    return value


def get_scene_entity_name(entity_cfg: Any) -> str:
    """Return the scene entity key from a cfg-like object or a plain string."""
    return entity_cfg if isinstance(entity_cfg, str) else entity_cfg.name


def is_kinematic_asset(asset: Any) -> bool:
    """Return whether an asset is configured or USD-authored as kinematic."""
    if getattr(asset, "_mana_kinematic", False):
        return True
    if getattr(asset, "_mana_runtime_kinematic", False):
        return True
    for cfg_name in ("cfg", "object_cfg"):
        cfg = getattr(asset, cfg_name, None)
        spawn = getattr(cfg, "spawn", None)
        rigid_props = getattr(spawn, "rigid_props", None)
        if bool(getattr(rigid_props, "kinematic_enabled", None)):
            return True
    return _runtime_kinematic_enabled(asset)


def _runtime_kinematic_enabled(asset: Any, *, stage: Any | None = None) -> bool:
    """Read an already-spawned rigid asset's authored USD kinematic flag."""
    if stage is None:
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
        except Exception:
            return False
    paths: list[str] = []
    for view_name in ("root_view", "root_physx_view"):
        view = getattr(asset, view_name, None)
        raw_paths = getattr(view, "prim_paths", None) if view is not None else None
        if raw_paths is not None:
            paths.extend(str(path) for path in raw_paths)
    for path in dict.fromkeys(paths):
        try:
            attr = stage.GetPrimAtPath(path).GetAttribute("physics:kinematicEnabled")
            if attr and bool(attr.Get()):
                try:
                    asset.__dict__["_mana_runtime_kinematic"] = True
                except (AttributeError, TypeError):
                    pass
                return True
        except Exception:
            continue
    return False


def install_kinematic_velocity_write_guards(scene_or_env: Any) -> int:
    """No-op root-velocity writes on kinematic rigid bodies in a live scene.

    Assembly-time Arena event rewrites are necessary but not sufficient: IsaacLab
    / Arena helpers may still call ``write_root_velocity_to_sim*`` during reset.
    PhysX rejects those calls with ``Body must be non-kinematic``. Patch the
    write methods on each kinematic ``RigidObject`` so every caller is safe.
    """
    scene = getattr(scene_or_env, "scene", scene_or_env)
    rigid_objects = getattr(scene, "rigid_objects", None)
    if not isinstance(rigid_objects, dict):
        return 0
    guarded = 0
    for asset in rigid_objects.values():
        if not is_kinematic_asset(asset):
            continue
        if getattr(asset, "_mana_kinematic_velocity_guard", False):
            continue
        for method_name in (
            "write_root_velocity_to_sim",
            "write_root_velocity_to_sim_index",
            "write_root_velocity_to_sim_mask",
            "write_root_com_velocity_to_sim",
            "write_root_com_velocity_to_sim_index",
            "write_root_com_velocity_to_sim_mask",
        ):
            if not callable(getattr(asset, method_name, None)):
                continue
            setattr(asset, method_name, _noop_root_velocity_write)
        asset.__dict__["_mana_kinematic_velocity_guard"] = True
        guarded += 1
    return guarded


def _noop_root_velocity_write(*_args: Any, **_kwargs: Any) -> None:
    """Intentional no-op for kinematic rigid-body velocity writes."""
    return None


def scene_has_asset(scene_or_env: Any, name: str) -> bool:
    """Return whether *scene_or_env* contains an asset keyed by *name*."""
    scene = getattr(scene_or_env, "scene", scene_or_env)
    assets = getattr(scene, "assets", None)
    if isinstance(assets, dict):
        return name in assets
    try:
        scene[name]
        return True
    except Exception:
        return False


def pool_cache_slot_name(object_id: str, slot: int) -> str:
    """Return the scene entity name for pool window slot ``slot`` (0 = base id)."""
    base = str(object_id)
    index = int(slot)
    return base if index == 0 else f"{base}__slot_{index}"


def inactive_pool_slot_position(
    slot_index: int,
    *,
    isolation_z: float = -10.0,
    spacing: float = 0.5,
) -> tuple[float, float, float]:
    """Return a collision-safe parking position for an inactive runtime-pool slot.

    Inactive dynamic bodies can remain in the collision world during
    schema-vNext resets and therefore must not share a parking pose. A vertical
    lane below the workspace keeps every slot separated while preserving the
    legacy ``away_z`` convention.
    """
    index = int(slot_index)
    if index < 0:
        raise ValueError("inactive pool slot index must be non-negative")
    separation = float(spacing)
    if separation <= 0.0:
        raise ValueError("inactive pool slot spacing must be positive")
    return 0.0, 0.0, float(isolation_z) - separation * index


def _runtime_assets_payload(env_or_scene: Any) -> Mapping[str, Any] | None:
    """Best-effort lookup of ``schema_vnext_runtime_assets`` metadata.

    Live gym envs expose IsaacLab ``InteractiveScene`` on ``env.scene``; the
    schema-vNext cache lives on the ManaEnv/Arena scene hung off
    ``AssembledEnvironment`` (see ``mana_scene``).
    """
    candidates: list[Any] = [
        env_or_scene,
        getattr(env_or_scene, "scene", None),
        getattr(env_or_scene, "cfg", None),
        getattr(getattr(env_or_scene, "cfg", None), "scene", None),
    ]
    try:
        from manaenv.runtime_owners import mana_scene

        candidates.append(mana_scene(env_or_scene))
    except Exception:
        pass
    for holder in candidates:
        if holder is None:
            continue
        direct = getattr(holder, "_schema_vnext_runtime_assets", None)
        if isinstance(direct, Mapping):
            return direct
        metadata = getattr(holder, "metadata", None)
        if isinstance(metadata, Mapping):
            payload = metadata.get("schema_vnext_runtime_assets")
            if isinstance(payload, Mapping):
                return payload
            nested = metadata.get("schema_vnext_materialization")
            if isinstance(nested, Mapping):
                nested_cache = nested.get("runtime_asset_cache")
                if isinstance(nested_cache, Mapping):
                    return nested_cache
    return None


def resolve_pool_cache_size(env_or_scene: Any, object_id: str, *, max_slots: int = 64) -> int:
    """Resolve the prespawn window size for a pool object from scene metadata or entities.

    Scene ``asset.cache_size`` is authoritative. Prefer ``schema_vnext_runtime_assets``
    when available; otherwise count ``object_id`` / ``object_id__slot_N`` entities.
    """
    object_id = str(object_id).strip()
    if not object_id:
        raise ValueError("resolve_pool_cache_size requires a non-empty object_id")

    runtime_assets = _runtime_assets_payload(env_or_scene)
    if isinstance(runtime_assets, Mapping):
        objects = runtime_assets.get("objects")
        if isinstance(objects, Mapping):
            entry = objects.get(object_id)
            if isinstance(entry, Mapping):
                cache_size = entry.get("cache_size")
                if isinstance(cache_size, int) and not isinstance(cache_size, bool) and cache_size > 0:
                    return int(cache_size)
                slots = entry.get("slots")
                if isinstance(slots, list) and slots:
                    return len(slots)

    count = 0
    for index in range(int(max_slots)):
        name = pool_cache_slot_name(object_id, index)
        if scene_has_asset(env_or_scene, name):
            count += 1
            continue
        if index == 0:
            break
        break
    if count <= 0:
        raise ValueError(f"pool cache for {object_id!r} not found in scene")
    return count


def resolve_pool_slot_asset_id(env_or_scene: Any, object_id: str, slot_name: str) -> str:
    """Return the concrete asset id for a prespawned pool cache slot, or empty string."""
    object_id = str(object_id).strip()
    slot_name = str(slot_name).strip()
    if not object_id or not slot_name:
        return ""
    runtime_assets = _runtime_assets_payload(env_or_scene)
    objects = runtime_assets.get("objects") if isinstance(runtime_assets, Mapping) else None
    entry = objects.get(object_id) if isinstance(objects, Mapping) else None
    slots = entry.get("slots") if isinstance(entry, Mapping) else None
    if not isinstance(slots, list):
        return ""
    for slot in slots:
        if not isinstance(slot, Mapping):
            continue
        if str(slot.get("name") or "").strip() != slot_name:
            continue
        return str(slot.get("asset_id") or "").strip()
    return ""


def record_active_asset_slot(
    env: Any,
    *,
    plan_id: str,
    env_id: int,
    object_id: str,
    active_slot: str,
    asset_id: str,
) -> None:
    """Record the active cache slot / asset id for overlay and role resolution."""
    state = getattr(env, "_schema_vnext_active_asset_slots", None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, "_schema_vnext_active_asset_slots", state)
    plan_state = state.setdefault(str(plan_id or "runtime"), {})
    env_state = plan_state.setdefault(int(env_id), {})
    env_state[str(object_id)] = {"slot": str(active_slot), "asset_id": str(asset_id)}


def resolve_active_scene_entity_name(
    env: Any,
    logical_name: str,
    *,
    env_id: int | None = None,
    plan_id: str | None = None,
) -> str:
    """Return the active cache slot for a logical asset when runtime metadata is present."""

    state = getattr(env, "_schema_vnext_active_asset_slots", None)
    if not isinstance(state, dict):
        return str(logical_name)
    plan_items = [(str(plan_id or "runtime"), state.get(str(plan_id or "runtime")))] if plan_id else state.items()
    for _, plan_state in plan_items:
        if not isinstance(plan_state, dict):
            continue
        env_items = [(int(env_id), plan_state.get(int(env_id)))] if env_id is not None else plan_state.items()
        for _, env_state in env_items:
            if not isinstance(env_state, dict):
                continue
            entry = env_state.get(str(logical_name))
            if isinstance(entry, dict) and str(entry.get("slot") or "").strip():
                return str(entry["slot"])
            if isinstance(entry, str) and entry.strip():
                return entry
    return str(logical_name)


def resolve_active_asset_id(
    env: Any,
    logical_name: str,
    *,
    env_id: int | None = None,
    plan_id: str | None = None,
) -> str | None:
    """Return the active pool asset id bound to a logical object this episode.

    Sibling of :func:`resolve_active_scene_entity_name`: that returns the active
    cache *slot* (scene entity used for pose reads), while this returns the
    concrete pool *asset id* (e.g. ``nut_d`` / ``screw_a``) used for instruction
    labels and grasp keys. Returns ``None`` when no schema-vNext runtime asset
    selection has been recorded for the object.
    """

    state = getattr(env, "_schema_vnext_active_asset_slots", None)
    if not isinstance(state, dict):
        return None
    plan_items = [(str(plan_id or "runtime"), state.get(str(plan_id or "runtime")))] if plan_id else state.items()
    for _, plan_state in plan_items:
        if not isinstance(plan_state, dict):
            continue
        env_items = [(int(env_id), plan_state.get(int(env_id)))] if env_id is not None else plan_state.items()
        for _, env_state in env_items:
            if not isinstance(env_state, dict):
                continue
            entry = env_state.get(str(logical_name))
            if isinstance(entry, dict):
                asset_id = str(entry.get("asset_id") or "").strip()
                if asset_id:
                    return asset_id
    return None


def resolve_active_scene_entity(
    env: Any, logical_name: str, *, env_id: int | None = None, plan_id: str | None = None
) -> Any:
    """Resolve and return the active scene entity for a logical runtime-randomized object."""

    return env.scene[resolve_active_scene_entity_name(env, logical_name, env_id=env_id, plan_id=plan_id)]


@contextmanager
def _open_stage(path: str):
    from pxr import Usd

    stage = Usd.Stage.Open(path)
    try:
        yield stage
    finally:
        del stage


def _is_rigid_body(prim) -> bool:
    from pxr import UsdPhysics

    return prim.HasAPI(UsdPhysics.RigidBodyAPI)


def _is_articulation_root(prim) -> bool:
    from pxr import UsdPhysics

    return prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def _get_all_prims(stage, prim=None):
    if prim is None:
        prim = stage.GetPseudoRoot()
    result = []
    for child in prim.GetAllChildren():
        result.append(child)
        result.extend(_get_all_prims(stage, child))
    return result


def get_rigid_body_prim_names(usd_path: str) -> list[str]:
    """Return relative prim names of rigid bodies inside an articulated USD.

    For pure rigid-body assets (single RigidBodyAPI at the root) the
    returned list is empty — the caller should fall back to the top-level
    ``{ENV_REGEX_NS}/<asset>`` path.

    For articulated assets the result contains names like
    ``"glasses_a"``, ``"glasses_a_l"`` which are the relative paths from
    the default prim.
    """
    with _open_stage(usd_path) as stage:
        root = stage.GetDefaultPrim()
        if root is None or not root.IsValid():
            return []
        # Pure rigid body — no sub-paths needed.
        if _is_rigid_body(root) and not _is_articulation_root(root):
            return []
        names: list[str] = []
        for child in _get_all_prims(stage, root):
            if _is_rigid_body(child):
                rel = child.GetPath().MakeRelativePath(root.GetPath())
                names.append(str(rel))
        return names


def get_contact_filter_paths(
    asset_name: str,
    usd_path: str,
) -> list[str]:
    """Return prim-path expressions for contact filtering.

    For rigid objects returns ``["<anchor>"]``. For articulated objects
    returns one entry per rigid body, e.g. ``["<anchor>/<body>", ...]``.

    The returned strings use single ``{ENV_REGEX_NS}`` braces so that
    Isaac Lab's ``InteractiveScene`` can resolve them via ``.format()``.
    """
    from isaaclab_arena.assets.object_base import asset_anchor_path

    prefix = asset_anchor_path(asset_name)
    body_names = get_rigid_body_prim_names(usd_path)
    if not body_names:
        return [prefix]
    return [f"{prefix}/{bn}" for bn in body_names]
