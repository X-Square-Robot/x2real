"""Scene term builders — produce sensor / asset cfgs for SceneCfg.

Each function accepts declarative parameters from YAML and returns an
IsaacLab config object (e.g. ``ContactSensorCfg``) ready for
``make_configclass``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import FunctionType, MethodType, ModuleType
from typing import Any

from manaenv.utils.debug_print import manaprint


def _asset_name(asset: Any) -> str:
    name = getattr(asset, "name", asset)
    name = str(name)
    if name.startswith("@"):
        name = name[1:]
    return name


def contact_sensor(
    source_assets: str | list[str],
    suffix: str = "contact_sensor",
    contact_against: str | list[str] | None = None,
    contact_against_assets: str | list[str] | None = None,
    **kwargs: Any,
) -> dict:
    """Build one ``ContactSensorCfg`` per source asset.

    Returns a dict mapping ``{asset}_{suffix}`` → ``ContactSensorCfg``
    so the caller can flatten them into the dynamic SceneCfg.

    Parameters
    ----------
    source_assets:
        Asset name(s) that need contact sensors.
    suffix:
        Appended to each asset name to form the sensor field name.
    contact_against:
        Optional prim paths for ``filter_prim_paths_expr``.
    contact_against_assets:
        Scene asset name(s) (e.g. resolved ``@destination``); each becomes
        ``{ENV_REGEX_NS}/<name>`` and is appended to *contact_against*.
    """
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

    if isinstance(source_assets, str):
        source_assets = [source_assets]
    source_assets = [_asset_name(asset) for asset in source_assets]
    if contact_against is None:
        contact_against = []
    elif isinstance(contact_against, str):
        contact_against = [contact_against]

    if contact_against_assets is not None:
        if isinstance(contact_against_assets, str):
            contact_against_assets = [contact_against_assets]
        for asset_name in contact_against_assets:
            contact_against.append(f"{{ENV_REGEX_NS}}/{_asset_name(asset_name)}")

    sensors: dict[str, ContactSensorCfg] = {}
    for asset in source_assets:
        name = f"{asset}_{suffix}"
        sensors[name] = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{asset}",
            filter_prim_paths_expr=contact_against,
            **kwargs,
        )
    return sensors


def contact_sensor_group(entries: list[dict[str, Any]]) -> dict:
    """Build multiple contact-sensor groups in one YAML term."""

    sensors: dict[str, Any] = {}
    for entry in entries:
        sensors.update(contact_sensor(**dict(entry or {})))
    return sensors


def _project_role_value(bound: Any) -> Any:
    if isinstance(bound, list):
        return [_project_role_value(item) for item in bound]
    name = getattr(bound, "name", None)
    return name if isinstance(name, str) else bound


def _resolve_sensor_ref(value: Any, roles: dict[str, Any]) -> Any:
    """Resolve ``@role`` references to bound asset name(s), flattening lists."""
    if isinstance(value, str) and value.startswith("@"):
        return _project_role_value(roles.get(value[1:], value[1:]))
    if isinstance(value, list):
        resolved: list[Any] = []
        for item in value:
            projected = _resolve_sensor_ref(item, roles)
            if isinstance(projected, list):
                resolved.extend(projected)
            else:
                resolved.append(projected)
        return resolved
    return value


def _split_contact_against(value: Any) -> tuple[list[str], list[str]]:
    """Split a resolved ``against`` value into (raw prim paths, asset names).

    Strings starting with ``{ENV_REGEX_NS}/`` or ``/`` are passed through as raw
    prim paths (e.g. robot link paths); everything else is treated as a scene
    asset name to be wrapped into ``{ENV_REGEX_NS}/<name>`` by ``contact_sensor``.
    """
    raw_paths: list[str] = []
    asset_names: list[str] = []
    if value is None:
        return raw_paths, asset_names
    items = value if isinstance(value, list) else [value]
    for item in items:
        name = str(getattr(item, "name", item))
        if name.startswith("{ENV_REGEX_NS}/") or name.startswith("/"):
            raw_paths.append(name)
        else:
            asset_names.append(name)
    return raw_paths, asset_names


def build_contact_sensors(
    entries: list[dict[str, Any]],
    roles: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, type, Any]], dict[str, list[str]]]:
    """Build contact sensors from the declarative ``sensor_component`` schema.

    Each entry is a single-key mapping ``{sensor_name: {type, source, against}}``.
    ``source`` / ``against`` may be ``@role`` references resolved against *roles*.
    ``against`` unifies the legacy ``contact_against`` (raw prim paths) and
    ``contact_against_assets`` (asset names) — see :func:`_split_contact_against`.

    Returns ``(scene_fields, roles_update)``:

    * ``scene_fields`` — ``(field_name, type, cfg)`` tuples ready for
      ``make_configclass`` (reuses :func:`contact_sensor`, so the finalize-time
      expansions ``finalize_multi_asset_contact_sensors`` /
      ``finalize_pool_cache_contact_sensors`` /
      ``expand_articulation_contact_paths`` keep working unchanged).
    * ``roles_update`` — ``sensor_name`` -> generated field names, published via
      ``ROLES_UPDATE_KEY`` so termination can reference ``@<sensor_name>``.
    """
    roles = roles or {}
    scene_fields: list[tuple[str, type, Any]] = []
    roles_update: dict[str, list[str]] = {}

    for entry in entries or []:
        for sensor_name, spec in (entry or {}).items():
            spec = dict(spec or {})
            sensor_type = spec.get("type", "contact")
            if sensor_type != "contact":
                raise ValueError(
                    f"sensor {sensor_name!r}: unsupported type {sensor_type!r} (only 'contact' is supported)"
                )

            source = _resolve_sensor_ref(spec.get("source"), roles)
            against = _resolve_sensor_ref(spec.get("against"), roles)
            raw_paths, asset_names = _split_contact_against(against)

            built = contact_sensor(
                source_assets=source,
                suffix=sensor_name,
                contact_against=raw_paths or None,
                contact_against_assets=asset_names or None,
            )
            for field_name, cfg_obj in built.items():
                scene_fields.append((field_name, type(cfg_obj), cfg_obj))
            roles_update[sensor_name] = list(built.keys())

    return scene_fields, roles_update


def _remap_contact_param_value(item: Any, remap: dict[str, list[str]]) -> tuple[Any, bool]:
    name = getattr(item, "name", item) if item is not None else None
    if isinstance(name, str) and name in remap:
        mapped = remap[name]
        return (mapped if len(mapped) > 1 else mapped[0]), True
    return item, False


def _flatten_contact_param_items(value: Any) -> list[Any]:
    """Flatten nested sensor-name lists from role resolution.

    ``contact_sensor_cfgs: [@left, @right]`` resolves each ``@sensor`` role to a
    list of field names, producing ``[[left_0], [right_0]]``. Pool-slot remap
    must see the inner strings, not the nested lists.
    """
    if not isinstance(value, list):
        return [value]
    flat: list[Any] = []
    for item in value:
        if isinstance(item, list):
            flat.extend(_flatten_contact_param_items(item))
        else:
            flat.append(item)
    return flat


def _rebuild_scene_cfg(scene_cfg: Any, to_remove: set[str], to_add: dict[str, Any]) -> Any:
    from isaaclab_arena.utils.configclass import make_configclass

    fields: list[tuple[str, type, Any]] = []
    for key, value in vars(scene_cfg).items():
        if key.startswith("_") or key in to_remove:
            continue
        fields.append((key, type(value), value))
    for key, value in to_add.items():
        fields.append((key, type(value), value))
    return make_configclass("SceneCfg", fields)()


def _remap_role_sensor_names(roles: dict[str, Any], remap: dict[str, list[str]]) -> None:
    for role_name, role_value in list(roles.items()):
        if isinstance(role_value, str):
            if role_value in remap:
                roles[role_name] = list(remap[role_value])
            continue
        if not isinstance(role_value, list):
            continue
        new_list: list[Any] = []
        changed = False
        for item in role_value:
            if isinstance(item, str) and item in remap:
                new_list.extend(remap[item])
                changed = True
            else:
                new_list.append(item)
        if changed:
            roles[role_name] = new_list


def _patch_contact_params(params: dict[str, Any], remap: dict[str, list[str]]) -> None:
    plural_key = "contact_sensor_cfgs"
    if plural_key in params:
        # Scalar / flat / nested lists (multi-@ role resolve → [[left],[right]]).
        items = _flatten_contact_param_items(params[plural_key])
        new_list: list[Any] = []
        replaced = False
        for item in items:
            mapped_item, item_replaced = _remap_contact_param_value(item, remap)
            if item_replaced:
                if isinstance(mapped_item, list):
                    new_list.extend(mapped_item)
                else:
                    new_list.append(mapped_item)
                replaced = True
            else:
                new_list.append(item)
        # Rewrite on remap OR flatten so nested role lists become a flat name list.
        if replaced or new_list != params[plural_key]:
            params[plural_key] = new_list

    singular_key = "contact_sensor_cfg"
    if singular_key in params:
        singular_value = params[singular_key]
        items = _flatten_contact_param_items(singular_value)
        new_list: list[Any] = []
        replaced = False
        for item in items:
            mapped_item, item_replaced = _remap_contact_param_value(item, remap)
            if item_replaced:
                if isinstance(mapped_item, list):
                    new_list.extend(mapped_item)
                else:
                    new_list.append(mapped_item)
                replaced = True
            else:
                new_list.append(item)
        if replaced or (isinstance(singular_value, list) and new_list != singular_value):
            params[singular_key] = new_list if len(new_list) != 1 else new_list[0]


def _walk_and_patch_contact_params(
    obj: Any,
    remap: dict[str, list[str]],
    visited: set[int] | None = None,
) -> None:
    if visited is None:
        visited = set()
    if isinstance(obj, (type, ModuleType, FunctionType, MethodType)):
        return
    obj_id = id(obj)
    if obj_id in visited:
        return
    if isinstance(obj, dict):
        visited.add(obj_id)
        _patch_contact_params(obj, remap)
        for value in obj.values():
            _walk_and_patch_contact_params(value, remap, visited)
    elif isinstance(obj, (list, tuple)):
        visited.add(obj_id)
        for item in obj:
            _walk_and_patch_contact_params(item, remap, visited)
    elif hasattr(obj, "__dict__"):
        visited.add(obj_id)
        for attr_name in list(vars(obj).keys()):
            _walk_and_patch_contact_params(getattr(obj, attr_name), remap, visited)


def expand_articulation_contact_paths(
    runtime_cfgs: dict[str, Any],
    scene: Any,
) -> None:
    """Expand contact sensor paths when the rigid body is nested below root.

    Covers two packaging shapes where the object root prim itself carries no
    ``PhysxContactReportAPI``: articulated assets, and rigid assets wrapped in
    an extra xform (e.g. felix ``/World/Aligned``).

    When ``contact_against_assets`` resolves to such an object the generated
    filter path ``{ENV_REGEX_NS}/<name>`` is insufficient — contact sensors
    need the individual rigid-body sub-paths. This function detects those cases
    and replaces the single path with one entry per nested rigid body.

    Additionally, when a sensor's own ``prim_path`` points to such an asset
    root, the path is rewritten to ``{ENV_REGEX_NS}/<name>/.*`` so that
    IsaacLab matches the rigid-body child that carries the reporter. Assets
    whose root prim already has the ``RigidBodyAPI`` are left unchanged.
    """
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
    from isaaclab_arena.assets.object_base import ObjectType, asset_anchor_path

    from manaenv.utils.asset_utils import get_contact_filter_paths, get_rigid_body_prim_names

    scene_cfg = runtime_cfgs.get("scene_cfg")
    if scene_cfg is None:
        return

    assets = getattr(scene, "assets", {})
    if isinstance(assets, dict):
        assets = assets.values()

    asset_map: dict[str, Any] = {}
    for a in assets:
        name = getattr(a, "name", None)
        if name:
            asset_map[name] = a

    plain_prefix = "{ENV_REGEX_NS}/"
    body_names_cache: dict[str, list[str]] = {}

    def _plain_asset_name(path: str) -> str | None:
        if not path.startswith(plain_prefix):
            return None
        rest = path[len(plain_prefix) :]
        return rest if rest and "/" not in rest else None

    def _nested_body_names(obj: Any) -> list[str]:
        """Relative rigid-body sub-paths when the physics body is *not* at the
        object root — articulations, or rigid assets packaged with an extra
        wrapper xform (e.g. felix ``/World/Aligned``). Empty when the root prim
        itself carries the ``RigidBodyAPI`` (nothing to expand)."""
        if obj is None:
            return []
        usd_path = getattr(obj, "usd_path", None)
        is_articulation = getattr(obj, "object_type", None) == ObjectType.ARTICULATION
        if not usd_path:
            return ["*"] if is_articulation else []
        if usd_path not in body_names_cache:
            try:
                body_names_cache[usd_path] = get_rigid_body_prim_names(usd_path)
            except Exception:
                body_names_cache[usd_path] = ["*"] if is_articulation else []
        return body_names_cache[usd_path]

    for _attr, val in list(vars(scene_cfg).items()):
        if not isinstance(val, ContactSensorCfg):
            continue

        # --- Expand prim_path when the source body is nested below root ---
        asset_name = _plain_asset_name(val.prim_path or "")
        if asset_name is not None:
            obj = asset_map.get(asset_name)
            anchor = asset_anchor_path(asset_name)
            if _nested_body_names(obj):
                val.prim_path = f"{anchor}/.*"
            else:
                val.prim_path = anchor

        # --- Expand filter_prim_paths_expr when the target body is nested ---
        filters = val.filter_prim_paths_expr
        if not filters:
            continue

        new_filters: list[str] = []
        changed = False
        for path in filters:
            asset_name = _plain_asset_name(path)
            if asset_name is None:
                new_filters.append(path)
                continue
            obj = asset_map.get(asset_name)
            usd_path = getattr(obj, "usd_path", None) if obj is not None else None
            if usd_path and _nested_body_names(obj):
                new_filters.extend(get_contact_filter_paths(asset_name, usd_path))
            else:
                new_filters.append(asset_anchor_path(asset_name))
            changed = True

        if changed:
            val.filter_prim_paths_expr = new_filters


def _runtime_asset_cache_slot_map(scene: Any) -> dict[str, tuple[str, ...]]:
    """Return logical pool object id -> cache slot names (len > 1 only)."""
    metadata = getattr(scene, "metadata", None) or {}
    cache = metadata.get("schema_vnext_runtime_assets")
    if not isinstance(cache, dict):
        materialization = metadata.get("schema_vnext_materialization")
        if isinstance(materialization, dict):
            cache = materialization.get("runtime_asset_cache")
    if not isinstance(cache, dict):
        return {}

    out: dict[str, tuple[str, ...]] = {}
    objects = cache.get("objects") or {}
    if not isinstance(objects, dict):
        return out
    for object_id, entry in objects.items():
        if not isinstance(entry, dict):
            continue
        slots: list[str] = []
        for slot in entry.get("slots") or []:
            if isinstance(slot, dict):
                name = str(slot.get("name") or "").strip()
            else:
                name = str(slot or "").strip()
            if name:
                slots.append(name)
        if len(slots) > 1:
            out[str(object_id)] = tuple(slots)
    return out


def finalize_multi_asset_contact_sensors(
    runtime_cfgs: dict[str, Any],
    roles: dict[str, Any],
    tag_to_name: dict[str, Any],
) -> None:
    """Turn one semantic-tag ``ContactSensorCfg`` into N sensors when ``tag_to_name[tag]`` is a list.

    Task generation resolves ``@touchable`` to the literal tag string, producing
    ``prim_path`` ``{ENV_REGEX_NS}/touchable``. After scene assembly, the same tag maps to
    multiple asset names; this replaces the placeholder row in ``scene_cfg`` and rewrites
    ``contact_sensor`` role / termination ``contact_sensor_cfgs`` accordingly.
    """
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

    scene_cfg = runtime_cfgs.get("scene_cfg")
    if scene_cfg is None:
        return

    prim_prefix = "{ENV_REGEX_NS}/"
    to_remove: set[str] = set()
    to_add: dict[str, Any] = {}
    remap: dict[str, list[str]] = {}

    for attr, val in list(vars(scene_cfg).items()):
        if not isinstance(val, ContactSensorCfg):
            continue
        pp = val.prim_path or ""
        if not pp.startswith(prim_prefix):
            continue
        tag = pp[len(prim_prefix) :]
        names = tag_to_name.get(tag)
        if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
            continue

        suffix = "contact_sensor"
        if attr.startswith(f"{tag}_"):
            suffix = attr[len(tag) + 1 :]

        to_remove.add(attr)
        new_map = contact_sensor(
            source_assets=names,
            suffix=suffix,
            contact_against=list(val.filter_prim_paths_expr or []),
        )
        to_add.update(new_map)
        remap[attr] = list(new_map.keys())

    if not to_remove:
        return

    runtime_cfgs["scene_cfg"] = _rebuild_scene_cfg(scene_cfg, to_remove, to_add)
    _remap_role_sensor_names(roles, remap)
    _walk_and_patch_contact_params(runtime_cfgs, remap)


def _expand_contact_filter_for_pool_slots(
    filters: list[str] | None,
    slot_map: dict[str, tuple[str, ...]],
) -> list[str]:
    """Expand ``{ENV_REGEX_NS}/<pool_id>`` filter paths to every cache slot."""
    prim_prefix = "{ENV_REGEX_NS}/"
    expanded: list[str] = []
    for path in list(filters or []):
        if not str(path).startswith(prim_prefix):
            expanded.append(path)
            continue
        asset_name = str(path)[len(prim_prefix) :]
        # Nested rigid paths (``name/child``) are left for articulation expand.
        if "/" in asset_name:
            expanded.append(path)
            continue
        slots = slot_map.get(asset_name)
        if slots:
            expanded.extend(f"{prim_prefix}{slot}" for slot in slots)
        else:
            expanded.append(path)
    return expanded


# Task sensor roles whose ``against`` filters should track the embodiment EE.
# Left/right hints select per-arm paths on dual-arm robots; single-arm robots
# (Franka / ARX) ignore the hint and return their only finger set.
_EE_CONTACT_SENSOR_ROLE_ARM_HINTS: dict[str, str | None] = {
    "gripper_contact_sensor": None,
    "left_gripper_contact_sensor": "left",
    "right_gripper_contact_sensor": "right",
}


def bind_robot_ee_contact_sensor_filters(
    runtime_cfgs: dict[str, Any],
    roles: dict[str, Any],
    robot: Any,
) -> None:
    """Bind gripper-contact sensor filters to embodiment-owned EE paths.

    Declarative atom stages often hardcode EX001 finger prims in ``against``.
    At finalize time the concrete embodiment is available, so rewrite every
    known gripper-contact role (unscoped + left/right) with
    ``robot.get_ee_contact_prim_paths(...)`` before pool-slot expansion copies
    those filters. Without this, Franka/ARX assemblies crash PhysX contact
    init (``filter_count`` on a null backend) when EX001 links are missing.
    """
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

    scene_cfg = runtime_cfgs.get("scene_cfg")
    get_contact_paths = getattr(robot, "get_ee_contact_prim_paths", None)
    if scene_cfg is None or not callable(get_contact_paths):
        return

    rebound: list[str] = []
    for role_name, arm_hint in _EE_CONTACT_SENSOR_ROLE_ARM_HINTS.items():
        sensor_names = roles.get(role_name, [])
        if not sensor_names:
            continue
        if not isinstance(sensor_names, list):
            sensor_names = [sensor_names]

        try:
            raw_paths = get_contact_paths(arm_hint) if arm_hint is not None else get_contact_paths()
        except TypeError:
            # Embodiments that ignore arm_mode still accept a positional no-op.
            raw_paths = get_contact_paths()
        contact_paths = [str(path) for path in (raw_paths or []) if str(path)]
        if not contact_paths:
            continue

        for sensor_ref in sensor_names:
            sensor_name = str(getattr(sensor_ref, "name", sensor_ref))
            sensor_cfg = getattr(scene_cfg, sensor_name, None)
            if sensor_cfg is None:
                continue
            # Duck-type across Isaac Lab reload boundaries.
            if not isinstance(sensor_cfg, ContactSensorCfg) and type(sensor_cfg).__name__ != "ContactSensorCfg":
                continue
            if not hasattr(sensor_cfg, "filter_prim_paths_expr"):
                continue
            if list(sensor_cfg.filter_prim_paths_expr or []) == contact_paths:
                continue
            sensor_cfg.filter_prim_paths_expr = list(contact_paths)
            rebound.append(sensor_name)

    if rebound:
        manaprint(f"INFO: [robot_contact] rebound {rebound} to embodiment EE contact paths")


def bind_ee_contact_sensor_object_filters(
    runtime_cfgs: dict[str, Any],
    roles: dict[str, Any],
    robot: Any,
    scene: Any,
) -> None:
    """Filter embodiment EE contact sensors against a task object (pool-aware).

    EX001 H-jaw USD nests duplicate ``*_gripper_*_link`` prim names, so mounting
    a ContactSensor on the object with gripper-link filters fails PhysX init
    (``expected 1, found N``). The inverse is safe: keep the embodiment's
    ``left/right_gripper_contact`` sensors and filter them to the object's
    unique pool-slot prim paths. Opt in with role ``ee_contact_against``.
    """
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

    get_sensor_names = getattr(robot, "get_ee_contact_sensor_names", None)
    against = roles.get("ee_contact_against")
    if against is None or not callable(get_sensor_names):
        return

    object_names: list[str] = []
    against_items = against if isinstance(against, list) else [against]
    for item in against_items:
        name = str(getattr(item, "name", item) or "").strip()
        if name.startswith("@"):
            bound = roles.get(name[1:])
            if bound is None:
                continue
            if isinstance(bound, list):
                object_names.extend(str(getattr(x, "name", x)) for x in bound)
            else:
                object_names.append(str(getattr(bound, "name", bound)))
        elif name:
            object_names.append(name)
    object_names = [n for n in object_names if n]
    if not object_names:
        return

    slot_map = _runtime_asset_cache_slot_map(scene)
    filter_paths: list[str] = []
    seen: set[str] = set()
    for object_name in object_names:
        slots = slot_map.get(object_name) or (object_name,)
        for slot in slots:
            path = f"{{ENV_REGEX_NS}}/{slot}"
            if path in seen:
                continue
            seen.add(path)
            filter_paths.append(path)
    if not filter_paths:
        return

    # Embodiment EE sensors live on ``robot.scene_config`` (merged into the env
    # scene at assembly). Task ``scene_cfg`` is optional — nut_on_screw stages
    # often have none, so do not require it before patching the robot.
    robot_scene = getattr(robot, "scene_config", None)
    task_scene = runtime_cfgs.get("scene_cfg") if isinstance(runtime_cfgs, Mapping) else None
    targets = [robot_scene, task_scene]

    rebound: list[str] = []
    for sensor_name in get_sensor_names() or []:
        name = str(sensor_name)
        for owner in targets:
            if owner is None:
                continue
            sensor_cfg = getattr(owner, name, None)
            # Duck-type: configclass instances may not share the imported class
            # identity across Isaac Lab reload boundaries.
            if sensor_cfg is None or not hasattr(sensor_cfg, "filter_prim_paths_expr"):
                continue
            if not isinstance(sensor_cfg, ContactSensorCfg) and type(sensor_cfg).__name__ != "ContactSensorCfg":
                continue
            sensor_cfg.filter_prim_paths_expr = list(filter_paths)
            if name not in rebound:
                rebound.append(name)

    if rebound:

        manaprint(f"INFO: [robot_contact] bound EE sensors {rebound} against object filters: {filter_paths}")
    else:

        manaprint(
            "WARNING: [robot_contact] ee_contact_against set but no EE contact sensors were patched "
            f"(sensors={list(get_sensor_names() or [])}, objects={object_names})"
        )


def finalize_pool_cache_contact_sensors(
    runtime_cfgs: dict[str, Any],
    roles: dict[str, Any],
    scene: Any,
) -> bool:
    """Expand contact sensors onto every schema-vNext runtime pool cache slot.

    ContactSensorCfg ``prim_path`` / filters are fixed at PhysX view init, so a
    sensor mounted only on the logical pool id (slot 0) misses the active spare
    when that episode activates ``id__slot_k``. This mirrors
    :func:`finalize_multi_asset_contact_sensors`: one sensor per source slot,
    destination pool filters expanded to all destination slots, and
    ``contact_sensor_cfgs`` / sensor roles remapped to the generated list.

    Returns True when ``scene_cfg`` / roles / termination params were rewritten.
    """
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

    scene_cfg = runtime_cfgs.get("scene_cfg")
    if scene_cfg is None:
        return False

    slot_map = _runtime_asset_cache_slot_map(scene)
    if not slot_map:
        return False

    prim_prefix = "{ENV_REGEX_NS}/"
    to_remove: set[str] = set()
    to_add: dict[str, Any] = {}
    remap: dict[str, list[str]] = {}
    filter_only_changed = False

    for attr, val in list(vars(scene_cfg).items()):
        if not isinstance(val, ContactSensorCfg):
            continue
        pp = val.prim_path or ""
        # Only plain object roots (``{ENV_REGEX_NS}/name``); nested /.* paths are
        # produced later by expand_articulation_contact_paths.
        if not pp.startswith(prim_prefix) or "/" in pp[len(prim_prefix) :]:
            continue
        source_name = pp[len(prim_prefix) :]
        source_slots = slot_map.get(source_name)
        filters = list(val.filter_prim_paths_expr or [])
        expanded_filters = _expand_contact_filter_for_pool_slots(filters, slot_map)
        if source_slots is None and expanded_filters == filters:
            continue

        suffix = "contact_sensor"
        if attr.startswith(f"{source_name}_"):
            suffix = attr[len(source_name) + 1 :]

        if source_slots is None:
            # Source is not a pool; only destination filter paths need expanding.
            val.filter_prim_paths_expr = expanded_filters
            filter_only_changed = True
            continue

        to_remove.add(attr)
        new_map = contact_sensor(
            source_assets=list(source_slots),
            suffix=suffix,
            contact_against=expanded_filters,
        )
        to_add.update(new_map)
        remap[attr] = list(new_map.keys())

    if not to_remove:
        return filter_only_changed

    runtime_cfgs["scene_cfg"] = _rebuild_scene_cfg(scene_cfg, to_remove, to_add)
    _remap_role_sensor_names(roles, remap)
    _walk_and_patch_contact_params(runtime_cfgs, remap)
    return True
