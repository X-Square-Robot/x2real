from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

from ..common.diagnostics import error
from .parser import VNextSceneSpec, _list, _mapping, _non_empty_str, _scalar_or_pair, _weight


@dataclass(frozen=True)
class RangeChoicePlan:
    choice_id: str
    x: tuple[float, float]
    y: tuple[float, float]
    weight: float = 1.0


@dataclass(frozen=True)
class YawChoicePlan:
    choice_id: str
    yaw: tuple[float, float]
    weight: float = 1.0


@dataclass(frozen=True)
class PoseChoicePlan:
    choice_id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    weight: float = 1.0


@dataclass(frozen=True)
class SurfaceAliasPlan:
    surface_id: str
    alias_type: str
    ref: str | None = None
    geometry: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VolumeAliasPlan:
    volume_id: str
    alias_type: str
    ref: str | None = None
    geometry: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeDomainPlan:
    node_id: str
    object_id: str
    placement_type: str | None = None
    surface: str | None = None
    volume: str | None = None
    base_orientation: tuple[float, float, float, float] | None = None
    ranges: tuple[RangeChoicePlan, ...] = ()
    yaws: tuple[YawChoicePlan, ...] = ()
    poses: tuple[PoseChoicePlan, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SlotDomainPlan:
    slot_id: str
    surface: str
    ranges: tuple[RangeChoicePlan, ...]
    yaws: tuple[YawChoicePlan, ...]


@dataclass(frozen=True)
class SlotAssignmentCasePlan:
    assignment_id: str
    bindings: Mapping[str, str]
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupDomainPlan:
    group_id: str
    group_type: str
    nodes: tuple[str, ...]
    surface: str | None = None
    volume: str | None = None
    base_orientation: tuple[float, float, float, float] | None = None
    ranges: tuple[RangeChoicePlan, ...] = ()
    yaws: tuple[YawChoicePlan, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    slots: Mapping[str, SlotDomainPlan] = field(default_factory=dict)
    assignment_ids: tuple[str, ...] = ()
    assignment_cases: Mapping[str, SlotAssignmentCasePlan] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationConstraintPlan:
    relation_type: str
    a: str
    b: str
    direction: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalLayoutPlan:
    seed: int | None
    surfaces: Mapping[str, SurfaceAliasPlan]
    volumes: Mapping[str, VolumeAliasPlan]
    nodes: Mapping[str, NodeDomainPlan]
    groups: Mapping[str, GroupDomainPlan]
    placement_owners: Mapping[str, str]
    relations: tuple[RelationConstraintPlan, ...] = ()


def build_canonical_layout_plan(scene: VNextSceneSpec) -> CanonicalLayoutPlan:
    layout = _mapping(scene.source.get("layout"), target="layout")
    surfaces = _surface_alias_plans(layout.get("surfaces"), scene=scene)
    volumes = _volume_alias_plans(layout.get("volumes", {}))
    nodes = _node_domain_plans(layout.get("nodes"), surfaces=surfaces, volumes=volumes)
    groups = _group_domain_plans(layout.get("groups", {}), surfaces=surfaces, volumes=volumes)
    owners = _placement_owners(nodes, groups)
    relations = _relation_plans(layout.get("relations", []))
    return CanonicalLayoutPlan(
        seed=scene.layout.seed,
        surfaces=surfaces,
        volumes=volumes,
        nodes=nodes,
        groups=groups,
        placement_owners=owners,
        relations=relations,
    )


def _surface_alias_plans(value: Any, *, scene: VNextSceneSpec) -> dict[str, SurfaceAliasPlan]:
    payload = _mapping(value, target="layout.surfaces")
    out: dict[str, SurfaceAliasPlan] = {}
    for surface_id, raw_surface in payload.items():
        sid = _non_empty_str(surface_id, target="layout.surfaces")
        data = _mapping(raw_surface, target=f"layout.surfaces.{sid}")
        surface_type = _non_empty_str(data.get("type"), target=f"layout.surfaces.{sid}.type")
        if surface_type == "ref":
            ref = str(data["ref"])
            override = _background_surface_override(ref, scene=scene)
            if override is not None:
                geometry = dict(override)
                geometry["id"] = sid
                out[sid] = SurfaceAliasPlan(surface_id=sid, alias_type="manual", geometry=geometry)
            else:
                out[sid] = SurfaceAliasPlan(surface_id=sid, alias_type="ref", ref=ref)
        elif surface_type == "manual":
            out[sid] = SurfaceAliasPlan(surface_id=sid, alias_type="manual", geometry=dict(data))
        else:
            raise error(
                "schema_vnext_surface_type_unsupported", "unsupported surface alias type", target=sid, type=surface_type
            )
    return out


def _background_surface_override(ref: str, *, scene: VNextSceneSpec) -> Mapping[str, Any] | None:
    if not ref.startswith("background."):
        return None
    surface_id = ref.split(".", 1)[1]
    return scene.background.surface_overrides.get(surface_id)


def _volume_alias_plans(value: Any) -> dict[str, VolumeAliasPlan]:
    payload = _mapping(value or {}, target="layout.volumes")
    out: dict[str, VolumeAliasPlan] = {}
    for volume_id, raw_volume in payload.items():
        vid = _non_empty_str(volume_id, target="layout.volumes")
        data = _mapping(raw_volume, target=f"layout.volumes.{vid}")
        volume_type = _non_empty_str(data.get("type"), target=f"layout.volumes.{vid}.type")
        if volume_type == "ref":
            out[vid] = VolumeAliasPlan(volume_id=vid, alias_type="ref", ref=str(data["ref"]))
        elif volume_type == "manual":
            out[vid] = VolumeAliasPlan(volume_id=vid, alias_type="manual", geometry=dict(data))
        else:
            raise error(
                "schema_vnext_volume_type_unsupported", "unsupported volume alias type", target=vid, type=volume_type
            )
    return out


def _node_domain_plans(
    value: Any,
    *,
    surfaces: Mapping[str, SurfaceAliasPlan],
    volumes: Mapping[str, VolumeAliasPlan],
) -> dict[str, NodeDomainPlan]:
    payload = _mapping(value, target="layout.nodes")
    out: dict[str, NodeDomainPlan] = {}
    for node_id, raw_node in payload.items():
        nid = _non_empty_str(node_id, target="layout.nodes")
        data = _mapping(raw_node, target=f"layout.nodes.{nid}")
        object_id = _non_empty_str(data.get("object", nid), target=f"layout.nodes.{nid}.object")
        placement = data.get("placement")
        if placement is None:
            out[nid] = NodeDomainPlan(node_id=nid, object_id=object_id)
            continue
        out[nid] = _node_placement_plan(nid, object_id, placement, surfaces=surfaces, volumes=volumes)
    return out


def _node_placement_plan(
    node_id: str,
    object_id: str,
    placement: Any,
    *,
    surfaces: Mapping[str, SurfaceAliasPlan],
    volumes: Mapping[str, VolumeAliasPlan],
) -> NodeDomainPlan:
    data = _mapping(placement, target=f"layout.nodes.{node_id}.placement")
    placement_type = _non_empty_str(data.get("type"), target=f"layout.nodes.{node_id}.placement.type")
    if placement_type == "surface":
        surface = _non_empty_str(data.get("surface"), target=f"layout.nodes.{node_id}.placement.surface")
        if surface not in surfaces:
            raise error("schema_vnext_unknown_reference", "unknown surface reference", target=node_id, ref=surface)
        return NodeDomainPlan(
            node_id=node_id,
            object_id=object_id,
            placement_type="surface",
            surface=surface,
            base_orientation=(
                _vec(data.get("orientation"), length=4, target=f"layout.nodes.{node_id}.placement.orientation")
                if "orientation" in data
                else None
            ),
            ranges=_range_choices(data.get("ranges"), target=f"layout.nodes.{node_id}.placement.ranges"),
            yaws=_yaw_choices(data.get("yaws", 0.0), target=f"layout.nodes.{node_id}.placement.yaws"),
        )
    if placement_type == "pose":
        return NodeDomainPlan(
            node_id=node_id,
            object_id=object_id,
            placement_type="pose",
            poses=_pose_choices(data.get("poses"), target=f"layout.nodes.{node_id}.placement.poses"),
        )
    if placement_type == "volume":
        volume = _non_empty_str(data.get("volume"), target=f"layout.nodes.{node_id}.placement.volume")
        if volume not in volumes:
            raise error("schema_vnext_unknown_reference", "unknown volume reference", target=node_id, ref=volume)
        params: dict[str, Any] = {}
        if "mode" in data:
            params["mode"] = str(data.get("mode") or "").strip().lower()
        return NodeDomainPlan(
            node_id=node_id,
            object_id=object_id,
            placement_type="volume",
            volume=volume,
            yaws=(
                _yaw_choices(data.get("yaws"), target=f"layout.nodes.{node_id}.placement.yaws")
                if "yaws" in data
                else ()
            ),
            params=params,
        )
    raise error(
        "schema_vnext_placement_type_unsupported",
        "unsupported node placement type",
        target=node_id,
        type=placement_type,
    )


def _group_domain_plans(
    value: Any,
    *,
    surfaces: Mapping[str, SurfaceAliasPlan],
    volumes: Mapping[str, VolumeAliasPlan],
) -> dict[str, GroupDomainPlan]:
    payload = _mapping(value or {}, target="layout.groups")
    out: dict[str, GroupDomainPlan] = {}
    for group_id, raw_group in payload.items():
        gid = _non_empty_str(group_id, target="layout.groups")
        data = _mapping(raw_group, target=f"layout.groups.{gid}")
        group_type = _non_empty_str(data.get("type"), target=f"layout.groups.{gid}.type")
        if group_type == "slot":
            out[gid] = _slot_group_plan(gid, data, surfaces=surfaces)
            continue
        nodes = tuple(
            _non_empty_str(item, target=f"layout.groups.{gid}.nodes[]")
            for item in _list(data.get("nodes"), target=f"layout.groups.{gid}.nodes")
        )
        if group_type in {"region", "scatter"}:
            surface = _non_empty_str(data.get("surface"), target=f"layout.groups.{gid}.surface")
            if surface not in surfaces:
                raise error("schema_vnext_unknown_reference", "unknown surface reference", target=gid, ref=surface)
            out[gid] = GroupDomainPlan(
                group_id=gid,
                group_type=group_type,
                nodes=nodes,
                surface=surface,
                base_orientation=(
                    _vec(data.get("orientation"), length=4, target=f"layout.groups.{gid}.orientation")
                    if "orientation" in data
                    else None
                ),
                ranges=(
                    _range_choices(data.get("ranges"), target=f"layout.groups.{gid}.ranges") if "ranges" in data else ()
                ),
                yaws=_yaw_choices(data.get("yaws"), target=f"layout.groups.{gid}.yaws") if "yaws" in data else (),
                params=_group_runtime_params(data),
            )
        elif group_type == "volume":
            volume = _non_empty_str(data.get("volume"), target=f"layout.groups.{gid}.volume")
            if volume not in volumes:
                raise error("schema_vnext_unknown_reference", "unknown volume reference", target=gid, ref=volume)
            out[gid] = GroupDomainPlan(
                group_id=gid,
                group_type=group_type,
                nodes=nodes,
                volume=volume,
                base_orientation=(
                    _vec(data.get("orientation"), length=4, target=f"layout.groups.{gid}.orientation")
                    if "orientation" in data
                    else None
                ),
                yaws=_yaw_choices(data.get("yaws"), target=f"layout.groups.{gid}.yaws") if "yaws" in data else (),
                params=_group_runtime_params(data),
            )
        else:
            raise error("schema_vnext_group_type_unsupported", "unsupported group type", target=gid, type=group_type)
    return out


def _slot_group_plan(
    group_id: str, data: Mapping[str, Any], *, surfaces: Mapping[str, SurfaceAliasPlan]
) -> GroupDomainPlan:
    nodes = tuple(
        _non_empty_str(item, target=f"layout.groups.{group_id}.nodes[]")
        for item in _list(data.get("nodes"), target=f"layout.groups.{group_id}.nodes")
    )
    slots_payload = _mapping(data.get("slots"), target=f"layout.groups.{group_id}.slots")
    slots: dict[str, SlotDomainPlan] = {}
    for slot_id, raw_slot in slots_payload.items():
        sid = _non_empty_str(slot_id, target=f"layout.groups.{group_id}.slots")
        slot = _mapping(raw_slot, target=f"layout.groups.{group_id}.slots.{sid}")
        surface = _non_empty_str(slot.get("surface"), target=f"layout.groups.{group_id}.slots.{sid}.surface")
        if surface not in surfaces:
            raise error(
                "schema_vnext_unknown_reference", "unknown surface reference", target=f"{group_id}.{sid}", ref=surface
            )
        slots[sid] = SlotDomainPlan(
            slot_id=sid,
            surface=surface,
            ranges=_range_choices(slot.get("ranges"), target=f"layout.groups.{group_id}.slots.{sid}.ranges"),
            yaws=_yaw_choices(slot.get("yaws"), target=f"layout.groups.{group_id}.slots.{sid}.yaws"),
        )
    assignments = _mapping(data.get("assignments"), target=f"layout.groups.{group_id}.assignments")
    mode = _non_empty_str(assignments.get("mode"), target=f"layout.groups.{group_id}.assignments.mode")
    if mode == "explicit":
        assignment_cases = _explicit_assignment_cases(group_id, assignments)
    elif mode == "permutation":
        assignment_cases = _permutation_assignment_cases(group_id, assignments)
    else:
        raise error(
            "schema_vnext_slot_assignment_mode_unsupported",
            "unsupported slot assignments mode",
            target=group_id,
            mode=mode,
        )
    return GroupDomainPlan(
        group_id=group_id,
        group_type="slot",
        nodes=nodes,
        params=_group_runtime_params(data),
        slots=slots,
        assignment_ids=tuple(assignment_cases),
        assignment_cases=assignment_cases,
    )


def _group_runtime_params(data: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {
        "type",
        "nodes",
        "surface",
        "volume",
        "orientation",
        "ranges",
        "yaws",
        "slots",
        "assignments",
        "default_assignment",
    }
    return {str(key): value for key, value in data.items() if str(key) not in reserved}


def _explicit_assignment_cases(group_id: str, assignments: Mapping[str, Any]) -> dict[str, SlotAssignmentCasePlan]:
    cases: dict[str, SlotAssignmentCasePlan] = {}
    for item in _list(assignments.get("items"), target=f"layout.groups.{group_id}.assignments.items"):
        raw_item = _mapping(item, target=f"layout.groups.{group_id}.assignments.items[]")
        assignment_id = _non_empty_str(raw_item.get("id"), target=f"layout.groups.{group_id}.assignments.items[].id")
        bindings = _mapping(
            raw_item.get("bindings"), target=f"layout.groups.{group_id}.assignments.{assignment_id}.bindings"
        )
        params = _mapping(
            raw_item.get("params", {}), target=f"layout.groups.{group_id}.assignments.{assignment_id}.params"
        )
        cases[assignment_id] = SlotAssignmentCasePlan(
            assignment_id=assignment_id,
            bindings={str(node_id): str(slot_id) for node_id, slot_id in bindings.items()},
            params=dict(params),
        )
    return cases


def _permutation_assignment_cases(group_id: str, assignments: Mapping[str, Any]) -> dict[str, SlotAssignmentCasePlan]:
    nodes = tuple(
        _non_empty_str(item, target=f"layout.groups.{group_id}.assignments.nodes[]")
        for item in _list(assignments.get("nodes"), target=f"layout.groups.{group_id}.assignments.nodes")
    )
    slots = tuple(
        _non_empty_str(item, target=f"layout.groups.{group_id}.assignments.slots[]")
        for item in _list(assignments.get("slots"), target=f"layout.groups.{group_id}.assignments.slots")
    )
    cases: dict[str, SlotAssignmentCasePlan] = {}
    for index, permuted_slots in enumerate(permutations(slots)):
        assignment_id = f"perm_{index}"
        cases[assignment_id] = SlotAssignmentCasePlan(
            assignment_id=assignment_id,
            bindings=dict(zip(nodes, permuted_slots, strict=True)),
            params={},
        )
    return cases


def _placement_owners(nodes: Mapping[str, NodeDomainPlan], groups: Mapping[str, GroupDomainPlan]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for node_id, node in nodes.items():
        if node.placement_type is not None:
            owners[node_id] = f"node:{node_id}"
    for group in groups.values():
        for node_id in group.nodes:
            if node_id in owners:
                raise error(
                    "schema_vnext_node_group_placement_conflict", "node has multiple placement owners", target=node_id
                )
            owners[node_id] = f"group:{group.group_id}"
    return owners


def _relation_plans(value: Any) -> tuple[RelationConstraintPlan, ...]:
    relations: list[RelationConstraintPlan] = []
    for index, raw_relation in enumerate(_list(value or [], target="layout.relations")):
        relation = _mapping(raw_relation, target=f"layout.relations[{index}]")
        relation_type = _non_empty_str(relation.get("type"), target=f"layout.relations[{index}].type")
        if relation_type != "relative_direction":
            raise error(
                "schema_vnext_relation_type_unsupported",
                "unsupported relation type",
                target=f"layout.relations[{index}]",
                type=relation_type,
            )
        relations.append(
            RelationConstraintPlan(
                relation_type=relation_type,
                a=_non_empty_str(relation.get("a"), target=f"layout.relations[{index}].a"),
                b=_non_empty_str(relation.get("b"), target=f"layout.relations[{index}].b"),
                direction=_non_empty_str(relation.get("direction"), target=f"layout.relations[{index}].direction"),
                params={key: value for key, value in relation.items() if key not in {"type", "a", "b", "direction"}},
            )
        )
    return tuple(relations)


def _range_choices(value: Any, *, target: str) -> tuple[RangeChoicePlan, ...]:
    if isinstance(value, Mapping):
        x = _scalar_or_pair(value.get("x"), target=f"{target}.x")
        y = _scalar_or_pair(value.get("y"), target=f"{target}.y")
        return (RangeChoicePlan(choice_id="default", x=x, y=y),)
    choices: list[RangeChoicePlan] = []
    for index, raw_item in enumerate(_list(value, target=target)):
        item = _mapping(raw_item, target=f"{target}[{index}]")
        choice_id = _non_empty_str(item.get("id"), target=f"{target}[{index}].id")
        choices.append(
            RangeChoicePlan(
                choice_id=choice_id,
                x=_scalar_or_pair(item.get("x"), target=f"{target}.{choice_id}.x"),
                y=_scalar_or_pair(item.get("y"), target=f"{target}.{choice_id}.y"),
                weight=_weight(item.get("weight"), target=f"{target}.{choice_id}.weight"),
            )
        )
    return tuple(choices)


def _yaw_choices(value: Any, *, target: str) -> tuple[YawChoicePlan, ...]:
    if isinstance(value, (int, float)):
        return (YawChoicePlan(choice_id="default", yaw=_scalar_or_pair(value, target=target)),)
    if isinstance(value, list) and value and all(not isinstance(item, Mapping) for item in value):
        return (YawChoicePlan(choice_id="default", yaw=_scalar_or_pair(value, target=target)),)
    choices: list[YawChoicePlan] = []
    for index, raw_item in enumerate(_list(value, target=target)):
        item = _mapping(raw_item, target=f"{target}[{index}]")
        choice_id = _non_empty_str(item.get("id"), target=f"{target}[{index}].id")
        raw_yaw = item["value"] if "value" in item else item["range"]
        choices.append(
            YawChoicePlan(
                choice_id=choice_id,
                yaw=_scalar_or_pair(raw_yaw, target=f"{target}.{choice_id}"),
                weight=_weight(item.get("weight"), target=f"{target}.{choice_id}.weight"),
            )
        )
    return tuple(choices)


def _pose_choices(value: Any, *, target: str) -> tuple[PoseChoicePlan, ...]:
    choices: list[PoseChoicePlan] = []
    for index, raw_item in enumerate(_list(value, target=target)):
        item = _mapping(raw_item, target=f"{target}[{index}]")
        choice_id = _non_empty_str(item.get("id"), target=f"{target}[{index}].id")
        choices.append(
            PoseChoicePlan(
                choice_id=choice_id,
                position=_vec(item.get("position"), length=3, target=f"{target}.{choice_id}.position"),
                orientation=_vec(item.get("orientation"), length=4, target=f"{target}.{choice_id}.orientation"),
                weight=_weight(item.get("weight"), target=f"{target}.{choice_id}.weight"),
            )
        )
    return tuple(choices)


def _vec(value: Any, *, length: int, target: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise error("schema_vnext_vector_invalid", f"expected {length} numeric values", target=target)
    return tuple(float(item) for item in value)
