from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..common.constants import POLYGON_CELL_SAMPLE_TRIES
from ..common.geometry import QuatXYZW as Quat
from ..common.geometry import Vec3, quat_mul, quat_normalize, quat_rotate, vec_add, vec_mul, yaw_quat
from ..common.reproducibility import stable_fingerprint, stable_rng
from ..common.surface_polygon import Polygon2D
from ..common.weighted import weighted_choice


class RuntimePlanSampleError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimePoseSample:
    position: Vec3
    quaternion: Quat


@dataclass(frozen=True)
class RuntimeSampleResult:
    poses: dict[str, RuntimePoseSample]
    context: dict[str, Any]


class RuntimePlanSampler:
    """Pure Python sampler for compiled RuntimeSamplingPlan payloads."""

    def __init__(self, runtime_plan: Mapping[str, Any], *, plan_id: str | None = None) -> None:
        self.plan = dict(runtime_plan or {})
        self.plan_id = str(plan_id or "").strip()
        self.domains = {
            str(item.get("id")): dict(item)
            for item in self.plan.get("domains", []) or []
            if isinstance(item, Mapping) and item.get("id")
        }
        raw_surfaces = self.plan.get("support_surfaces")
        if not isinstance(raw_surfaces, list):
            raw_surfaces = (self.plan.get("artifacts", {}) or {}).get("support_surfaces", [])
        self.surfaces = {
            str(item.get("id")): dict(item)
            for item in raw_surfaces or []
            if isinstance(item, Mapping) and item.get("id")
        }

    @property
    def objects(self) -> set[str]:
        return _objects_for_plan(self.plan, plan_id=self.plan_id)

    @property
    def parent_objects(self) -> set[str]:
        return _object_frame_parents_for_plan(self.plan, plan_id=self.plan_id)

    def sample(
        self,
        *,
        seed: int,
        base_quats: Mapping[str, Quat] | None = None,
        base_poses: Mapping[str, Any] | None = None,
        episode_index: int | None = None,
        env_id: int = 0,
    ) -> dict[str, RuntimePoseSample]:
        return self.sample_result(
            seed=seed,
            base_quats=base_quats,
            base_poses=base_poses,
            episode_index=episode_index,
            env_id=env_id,
        ).poses

    def sample_result(
        self,
        *,
        seed: int,
        base_quats: Mapping[str, Quat] | None = None,
        base_poses: Mapping[str, Any] | None = None,
        episode_index: int | None = None,
        env_id: int = 0,
    ) -> RuntimeSampleResult:
        base_seed = int(seed)
        bases = dict(base_quats or {})
        parent_bases = _normalize_base_poses(base_poses)
        sampled: dict[str, tuple[Vec3, Quat]] = {}
        context: dict[str, Any] = {
            "plan_id": self.plan_id,
            "selections": {},
            "domain_samples": {},
            "outputs": {},
        }

        for assignment in self.plan.get("assignments", []) or []:
            if not isinstance(assignment, Mapping) or not _plan_id_matches(
                str(assignment.get("id") or ""), self.plan_id
            ):
                continue
            assignment_id = str(assignment.get("id") or "")
            assignment_rng = stable_rng(base_seed, "runtime_assignment", self.plan_id, assignment_id)
            bind, selected_case = _assignment_bind_for_sample(
                assignment,
                assignment_rng,
                episode_index=episode_index,
                env_id=env_id,
                plan_id=self.plan_id,
            )
            if selected_case is not None:
                _record_assignment_selection(context, assignment, selected_case)
            for object_name, domain_id in sorted(bind.items()):
                name = str(object_name)
                pose, domain_sample = self._sample_domain_pose(
                    str(domain_id),
                    rng=stable_rng(base_seed, "runtime_domain", self.plan_id, assignment_id, name, str(domain_id)),
                    parent_pose=None,
                    base_poses=parent_bases,
                    base_quat=bases.get(name, _identity_quat()),
                    selector_id=f"{assignment_id}:{name}",
                    episode_index=episode_index,
                    env_id=env_id,
                )
                sampled[name] = pose
                _record_domain_sample(context, name, str(domain_id), domain_sample)

        for conditional in self.plan.get("conditional_assignments", []) or []:
            if not isinstance(conditional, Mapping) or not _plan_id_matches(
                str(conditional.get("id") or ""), self.plan_id
            ):
                continue
            conditional_id = str(conditional.get("id") or "")
            parent = str(conditional.get("parent") or "")
            if not parent:
                raise RuntimePlanSampleError(f"conditional assignment {conditional.get('id')} missing parent")
            parent_pose = sampled.get(parent)
            if parent_pose is None:
                parent_pose, domain_sample = self._sample_domain_pose(
                    str(conditional.get("parent_domain") or ""),
                    rng=stable_rng(
                        base_seed,
                        "runtime_conditional_parent",
                        self.plan_id,
                        conditional_id,
                        parent,
                        str(conditional.get("parent_domain") or ""),
                    ),
                    parent_pose=None,
                    base_poses=parent_bases,
                    base_quat=bases.get(parent, _identity_quat()),
                    selector_id=f"{conditional_id}:{parent}",
                    episode_index=episode_index,
                    env_id=env_id,
                )
                sampled[parent] = parent_pose
                _record_domain_sample(context, parent, str(conditional.get("parent_domain") or ""), domain_sample)
            child_map, selected_case = _child_domain_map(
                conditional,
                rng=stable_rng(base_seed, "runtime_conditional_case", self.plan_id, conditional_id),
                episode_index=episode_index,
                env_id=env_id,
                plan_id=self.plan_id,
            )
            if selected_case is not None:
                _record_assignment_selection(context, conditional, selected_case)
            for child, domain_id in sorted(child_map.items()):
                name = str(child)
                pose, domain_sample = self._sample_domain_pose(
                    str(domain_id),
                    rng=stable_rng(
                        base_seed, "runtime_conditional_child", self.plan_id, conditional_id, name, str(domain_id)
                    ),
                    parent_pose=parent_pose,
                    base_poses=parent_bases,
                    base_quat=bases.get(name, _identity_quat()),
                    selector_id=f"{conditional_id}:{name}",
                    episode_index=episode_index,
                    env_id=env_id,
                )
                sampled[name] = pose
                _record_domain_sample(context, name, str(domain_id), domain_sample)

        for output in self.plan.get("outputs", []) or []:
            if not isinstance(output, Mapping) or not _plan_id_matches(str(output.get("id") or ""), self.plan_id):
                continue
            output_name = str(output.get("name") or output.get("id") or "")
            output_rng = stable_rng(base_seed, "runtime_output", self.plan_id, str(output.get("id") or output_name))
            context["outputs"][output_name] = _output_value_for_sample(output, output_rng, context)

        poses = {name: RuntimePoseSample(position=pose[0], quaternion=pose[1]) for name, pose in sampled.items()}
        return RuntimeSampleResult(poses=poses, context=context)

    def _sample_domain_pose(  # noqa: C901
        self,
        domain_id: str,
        *,
        rng: random.Random,
        parent_pose: tuple[Vec3, Quat] | None,
        base_poses: Mapping[str, tuple[Vec3, Quat]],
        base_quat: Quat,
        selector_id: str,
        episode_index: int | None,
        env_id: int,
    ) -> tuple[tuple[Vec3, Quat], dict[str, Any] | None]:
        domain = self.domains.get(str(domain_id))
        if not domain:
            raise RuntimePlanSampleError(f"runtime domain not found: {domain_id}")
        geometry_type = str(domain.get("geometry_type") or "rectangle")
        raw_base_quat = domain.get("base_quaternion")
        if isinstance(raw_base_quat, list) and len(raw_base_quat) == 4:
            base_quat = _quat(raw_base_quat)
        yaw, yaw_sample = _sample_yaw_for_domain(
            domain,
            rng,
            plan_id=self.plan_id,
            selector_id=f"{selector_id}:yaws",
            episode_index=episode_index,
            env_id=env_id,
        )
        quat = quat_mul(yaw_quat(yaw), base_quat) if abs(yaw) > 1e-12 else base_quat
        domain_sample: dict[str, Any] | None = None
        if geometry_type == "pose_list":
            poses = [item for item in domain.get("poses", []) or [] if isinstance(item, Mapping)]
            if not poses:
                raise RuntimePlanSampleError(f"pose_list domain has no poses: {domain_id}")
            selected = _select_mapping_candidate(
                poses,
                rng,
                selector=_domain_choice_selector(domain, "poses"),
                plan_id=self.plan_id,
                selector_id=f"{selector_id}:poses",
                env_id=env_id,
                episode_index=episode_index,
            )
            if selected is None:
                raise RuntimePlanSampleError(f"pose_list domain has no selectable poses: {domain_id}")
            item = dict(selected)
            position = item.get("position")
            raw_quat = item.get("quaternion")
            if not isinstance(position, list) or len(position) != 3:
                raise RuntimePlanSampleError(f"pose_list candidate missing position: {domain_id}")
            if isinstance(raw_quat, list) and len(raw_quat) == 4:
                quat = _quat(raw_quat)
            domain_sample = {
                "kind": "pose_choice",
                "pose_id": str(item.get("id") or ""),
                "weight": _mapping_weight(item),
                "meta": dict(item.get("meta") or {}) if isinstance(item.get("meta"), Mapping) else {},
            }
            _attach_yaw_sample(domain_sample, yaw_sample)
            return ((float(position[0]), float(position[1]), float(position[2])), quat), domain_sample
        if geometry_type == "surface_range_choice":
            choices = [item for item in domain.get("range_choices", []) or [] if isinstance(item, Mapping)]
            selected = _select_mapping_candidate(
                choices,
                rng,
                selector=_domain_choice_selector(domain, "ranges"),
                plan_id=self.plan_id,
                selector_id=f"{selector_id}:ranges",
                env_id=env_id,
                episode_index=episode_index,
            )
            if selected is None:
                raise RuntimePlanSampleError(f"surface_range_choice domain has no selectable ranges: {domain_id}")
            choice = dict(selected)
            if str(choice.get("geometry_type") or "rectangle") == "cells":
                cells = [item for item in choice.get("cells", []) or [] if isinstance(item, Mapping)]
                if not cells:
                    raise RuntimePlanSampleError(f"surface_range_choice has empty cells: {domain_id}")
                cell = dict(rng.choice(cells))
                xy = _sample_cell_xy(cell, rng)
                if xy is None:
                    raise RuntimePlanSampleError(f"failed to sample surface_range_choice cells: {domain_id}")
                x, y = xy
                z = _sample_range(cell.get("z"), rng, default=_range_center(choice.get("z", domain.get("z"))))
            else:
                cell = None
                x = _sample_range(choice.get("x"), rng)
                y = _sample_range(choice.get("y"), rng)
                z = _sample_range(choice.get("z", domain.get("z")), rng)
            if x is None or y is None:
                raise RuntimePlanSampleError(f"surface_range_choice missing x/y range: {domain_id}")
            domain_sample = {
                "kind": "surface_range_choice",
                "range_id": str(choice.get("id") or choice.get("choice_id") or ""),
                "weight": _mapping_weight(choice),
                "source": str(choice.get("source") or ""),
                "surface": str(domain.get("surface") or ""),
                "meta": dict(choice.get("meta") or {}) if isinstance(choice.get("meta"), Mapping) else {},
            }
            if isinstance(cell, Mapping):
                domain_sample["cell_id"] = str(cell.get("id") or cell.get("cell_id") or "")
        if geometry_type == "surface_range_choice":
            pass
        elif geometry_type not in {"rectangle", "box", "cylinder", "polygon", "cells"}:
            raise RuntimePlanSampleError(f"unsupported runtime domain geometry: {geometry_type}")
        elif geometry_type == "cells":
            cells = [item for item in domain.get("cells", []) or [] if isinstance(item, Mapping)]
            if not cells:
                raise RuntimePlanSampleError(f"cells domain has no cells: {domain_id}")
            cell = dict(rng.choice(cells))
            xy = _sample_cell_xy(cell, rng)
            if xy is None:
                raise RuntimePlanSampleError(f"failed to sample cell domain: {domain_id}")
            x, y = xy
            z = _sample_range(cell.get("z"), rng, default=_range_center(domain.get("z")))
            domain_sample = {
                "kind": "cell_choice",
                "cell_id": str(cell.get("id") or cell.get("cell_id") or ""),
                "weight": _mapping_weight(cell),
                "surface": str(domain.get("surface") or ""),
            }
        elif geometry_type == "cylinder":
            xy = _sample_cylinder_xy(domain, rng)
            if xy is None:
                raise RuntimePlanSampleError(f"failed to sample cylinder domain: {domain_id}")
            x, y = xy
            z = _sample_range(domain.get("z"), rng)
        else:
            x = _sample_range(domain.get("x"), rng)
            y = _sample_range(domain.get("y"), rng)
            z = _sample_range(domain.get("z"), rng)
        if x is None or y is None:
            raise RuntimePlanSampleError(f"runtime domain missing x/y range: {domain_id}")
        if yaw_sample:
            if domain_sample is None:
                domain_sample = {"kind": "yaw_choice"}
            _attach_yaw_sample(domain_sample, yaw_sample)
        frame = str(domain.get("frame") or "")
        if frame.startswith("surface:"):
            surface_id = frame.split(":", 1)[1]
            surface = self.surfaces.get(surface_id)
            if surface is None:
                raise RuntimePlanSampleError(f"runtime support surface not found: {surface_id}")
            return (_surface_local_to_world(surface, x, y, z), quat), domain_sample
        if frame.startswith("object:"):
            parent_name = frame.split(":", 1)[1]
            if parent_pose is None:
                parent_pose = base_poses.get(parent_name)
            if parent_pose is None:
                raise RuntimePlanSampleError(f"object-frame domain requires parent pose: {domain_id}")
            parent_pos, parent_quat = parent_pose
            offset = quat_rotate(parent_quat, (float(x), float(y), float(z or 0.0)))
            return (
                (parent_pos[0] + offset[0], parent_pos[1] + offset[1], parent_pos[2] + offset[2]),
                quat_mul(parent_quat, quat),
            ), domain_sample
        if frame == "world" and z is not None:
            return ((float(x), float(y), float(z)), quat), domain_sample
        raise RuntimePlanSampleError(f"unsupported runtime domain frame: {frame}")


def _objects_for_plan(plan: Mapping[str, Any], *, plan_id: str) -> set[str]:
    out: set[str] = set()
    for assignment in plan.get("assignments", []) or []:
        if not isinstance(assignment, Mapping) or not _plan_id_matches(str(assignment.get("id") or ""), plan_id):
            continue
        out.update(str(obj) for obj in assignment.get("objects", []) or [])
        for case in assignment.get("cases", []) or []:
            if isinstance(case, Mapping):
                out.update(str(obj) for obj in dict(case.get("bind") or {}))
    for conditional in plan.get("conditional_assignments", []) or []:
        if not isinstance(conditional, Mapping) or not _plan_id_matches(str(conditional.get("id") or ""), plan_id):
            continue
        if conditional.get("parent"):
            out.add(str(conditional["parent"]))
        out.update(str(obj) for obj in conditional.get("children", []) or [])
        for case in conditional.get("cases", []) or []:
            if isinstance(case, Mapping):
                out.update(str(obj) for obj in dict(case.get("bind") or {}))
    return out


def _object_frame_parents_for_plan(plan: Mapping[str, Any], *, plan_id: str) -> set[str]:
    domain_ids = _domain_ids_for_plan(plan, plan_id=plan_id)
    out: set[str] = set()
    for domain in plan.get("domains", []) or []:
        if not isinstance(domain, Mapping):
            continue
        domain_id = str(domain.get("id") or "")
        if domain_ids and domain_id not in domain_ids:
            continue
        frame = str(domain.get("frame") or "")
        if frame.startswith("object:"):
            parent = frame.split(":", 1)[1].strip()
            if parent:
                out.add(parent)
    return out


def _domain_ids_for_plan(plan: Mapping[str, Any], *, plan_id: str) -> set[str]:
    out: set[str] = set()
    for assignment in plan.get("assignments", []) or []:
        if not isinstance(assignment, Mapping) or not _plan_id_matches(str(assignment.get("id") or ""), plan_id):
            continue
        out.update(str(domain_id) for domain_id in assignment.get("domains", []) or [])
        for case in assignment.get("cases", []) or []:
            if isinstance(case, Mapping):
                out.update(str(domain_id) for domain_id in dict(case.get("bind") or {}).values())
    for conditional in plan.get("conditional_assignments", []) or []:
        if not isinstance(conditional, Mapping) or not _plan_id_matches(str(conditional.get("id") or ""), plan_id):
            continue
        if conditional.get("parent_domain"):
            out.add(str(conditional["parent_domain"]))
        out.update(str(domain_id) for domain_id in dict(conditional.get("child_domains") or {}).values())
        for case in conditional.get("cases", []) or []:
            if isinstance(case, Mapping):
                out.update(str(domain_id) for domain_id in dict(case.get("bind") or {}).values())
    if not out:
        out.update(str(domain.get("id")) for domain in plan.get("domains", []) or [] if isinstance(domain, Mapping))
    return out


def _normalize_base_poses(base_poses: Mapping[str, Any] | None) -> dict[str, tuple[Vec3, Quat]]:
    out: dict[str, tuple[Vec3, Quat]] = {}
    for name, raw in dict(base_poses or {}).items():
        pose = _normalize_base_pose(raw)
        if pose is not None:
            out[str(name)] = pose
    return out


def _normalize_base_pose(raw: Any) -> tuple[Vec3, Quat] | None:
    if raw is None:
        return None
    position = getattr(raw, "position", None)
    quaternion = getattr(raw, "quaternion", None)
    if position is not None and quaternion is not None:
        return (_vec3(position), _quat(quaternion))
    if isinstance(raw, Mapping):
        position = raw.get("position")
        quaternion = raw.get("quaternion")
        if position is not None and quaternion is not None:
            return (_vec3(position), _quat(quaternion))
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        position, quaternion = raw
        return (_vec3(position), _quat(quaternion))
    if isinstance(raw, (list, tuple)) and len(raw) == 7:
        return (_vec3(raw[:3]), _quat(raw[3:7]))
    return None


def _plan_id_matches(candidate: str, requested: str) -> bool:
    if not requested:
        return True
    return candidate == requested or candidate.startswith(f"{requested}:")


def _assignment_bind_for_sample(
    assignment: Mapping[str, Any],
    rng: random.Random,
    *,
    episode_index: int | None,
    env_id: int,
    plan_id: str,
) -> tuple[dict[str, str], Mapping[str, Any] | None]:
    if str(assignment.get("mode") or "") == "case_pool":
        case = _select_case(
            assignment.get("cases"),
            rng,
            selector=_case_selector(assignment),
            plan_id=plan_id,
            selector_id=str(assignment.get("id") or ""),
            env_id=env_id,
            episode_index=episode_index,
        )
        if case is not None:
            return {str(child): str(domain_id) for child, domain_id in dict(case.get("bind") or {}).items()}, case
    return {str(child): str(domain_id) for child, domain_id in dict(assignment.get("bind") or {}).items()}, None


def _child_domain_map(
    conditional: Mapping[str, Any],
    *,
    rng: random.Random | None = None,
    episode_index: int | None = None,
    env_id: int = 0,
    plan_id: str = "",
) -> tuple[dict[str, str], Mapping[str, Any] | None]:
    cases = conditional.get("cases")
    if rng is not None:
        case = _select_case(
            cases,
            rng,
            selector=_case_selector(conditional),
            plan_id=plan_id,
            selector_id=str(conditional.get("id") or ""),
            env_id=env_id,
            episode_index=episode_index,
        )
        if case is not None:
            return {str(child): str(domain_id) for child, domain_id in dict(case.get("bind") or {}).items()}, case
    out = {str(child): str(domain_id) for child, domain_id in dict(conditional.get("child_domains") or {}).items()}
    shared = str(conditional.get("child_domain") or "").strip()
    if shared:
        for child in conditional.get("children", []) or []:
            out.setdefault(str(child), shared)
    return out, None


def _record_assignment_selection(
    context: dict[str, Any],
    assignment: Mapping[str, Any],
    case: Mapping[str, Any],
) -> None:
    assignment_id = str(assignment.get("id") or "")
    meta = assignment.get("meta") if isinstance(assignment.get("meta"), Mapping) else {}
    selection_key = str(meta.get("variant_id") or _plan_suffix(assignment_id) or assignment_id)
    slot_ids = {str(obj): str(slot) for obj, slot in dict(case.get("slot_ids") or {}).items()}
    object_names = [str(name) for name in assignment.get("objects", []) or []]
    objects_by_slot_order = _objects_by_slot_order(slot_ids, case.get("slot_order"), object_names)
    entry = {
        "assignment_id": assignment_id,
        "case_id": str(case.get("id") or ""),
        "source_case_id": str(case.get("source_case_id") or case.get("id") or ""),
        "weight": _mapping_weight(case),
        "bind": {str(obj): str(domain_id) for obj, domain_id in dict(case.get("bind") or {}).items()},
        "slots_by_object": slot_ids,
        "objects": object_names,
        "object_order": [str(name) for name in case.get("object_order", object_names) or []],
        "objects_by_slot_order": objects_by_slot_order,
        "display_names_by_slot_order": [_display_name(name) for name in objects_by_slot_order],
        "case_index": case.get("case_index"),
        "params": dict(case.get("params") or {}) if isinstance(case.get("params"), Mapping) else {},
    }
    selector_result = case.get("_selector_result") if isinstance(case.get("_selector_result"), Mapping) else None
    if selector_result:
        entry["selector"] = dict(selector_result)
    context.setdefault("selections", {})[selection_key] = entry
    context.setdefault("assignments", {})[assignment_id] = entry


def _record_domain_sample(
    context: dict[str, Any],
    object_name: str,
    domain_id: str,
    sample: Mapping[str, Any] | None,
) -> None:
    if not isinstance(sample, Mapping) or not sample:
        return
    entry = {
        "object": str(object_name),
        "domain_id": str(domain_id),
        **dict(sample),
    }
    context.setdefault("domain_samples", {})[str(object_name)] = entry
    context.setdefault("domains", {})[str(domain_id)] = entry


def _objects_by_slot_order(slot_ids: Mapping[str, str], raw_slot_order: Any, object_names: list[str]) -> list[str]:
    by_slot = {str(slot): str(obj) for obj, slot in slot_ids.items()}
    out: list[str] = []
    if isinstance(raw_slot_order, (list, tuple)):
        for slot in raw_slot_order:
            obj = by_slot.get(str(slot))
            if obj is not None and obj not in out:
                out.append(obj)
    if not out and slot_ids:
        for obj, _slot in sorted(slot_ids.items(), key=lambda item: (item[1], item[0])):
            if obj not in out:
                out.append(obj)
    for obj in object_names:
        if obj not in out:
            out.append(obj)
    return out


def _output_value_for_sample(
    output: Mapping[str, Any], rng: random.Random, context: Mapping[str, Any]
) -> dict[str, Any]:
    if str(output.get("mode") or "") == "case_pool":
        case = _select_case(output.get("cases"), rng)
        if case is None:
            return _output_entry(str(output.get("name") or ""), [], case_id="")
        values = list(case.get("values") or [])
        entry = _output_entry(
            str(output.get("name") or ""),
            values,
            case_id=str(case.get("id") or ""),
            weight=_mapping_weight(case),
            params=dict(case.get("params") or {}) if isinstance(case.get("params"), Mapping) else {},
        )
        case_items = case.get("items")
        if isinstance(case_items, list):
            entry["items"] = [dict(item) for item in case_items if isinstance(item, Mapping)]
            entry["object_names"] = [str(item.get("object_name")) for item in entry["items"] if item.get("object_name")]
            entry["display_names"] = [str(item.get("display_name")) for item in entry["items"]]
        return entry
    if str(output.get("mode") or "") == "derived":
        meta = output.get("meta") if isinstance(output.get("meta"), Mapping) else {}
        source_path = str(meta.get("from") or "")
        resolved = _resolve_context_path(context, source_path)
        return _derived_output_entry(str(output.get("name") or ""), resolved, source_path=source_path)
    return _output_entry(str(output.get("name") or ""), list(output.get("values") or []), case_id="fixed")


def _derived_output_entry(output_name: str, resolved: Any, *, source_path: str) -> dict[str, Any]:
    if isinstance(resolved, Mapping):
        if isinstance(resolved.get("values"), list):
            entry = _output_entry(output_name, list(resolved.get("values") or []), case_id="derived")
        elif isinstance(resolved.get("object_names"), list):
            entry = _output_entry(output_name, list(resolved.get("object_names") or []), case_id="derived")
        elif isinstance(resolved.get("objects_by_slot_order"), list):
            entry = _output_entry(output_name, list(resolved.get("objects_by_slot_order") or []), case_id="derived")
        else:
            entry = _output_entry(output_name, [dict(resolved)], case_id="derived")
        for key in ("items", "object_names", "display_names"):
            if isinstance(resolved.get(key), list):
                entry[key] = [dict(item) if isinstance(item, Mapping) else item for item in resolved.get(key) or []]
        entry["source"] = source_path
        return entry
    values = list(resolved) if isinstance(resolved, (list, tuple)) else [resolved]
    entry = _output_entry(output_name, values, case_id="derived")
    entry["source"] = source_path
    return entry


def _output_entry(
    output_name: str,
    values: list[Any],
    *,
    case_id: str,
    weight: float = 1.0,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    items = [_value_descriptor(value) for value in values]
    return {
        "name": output_name,
        "case_id": case_id,
        "weight": float(weight),
        "values": list(values),
        "items": items,
        "object_names": [str(item["object_name"]) for item in items if item.get("object_name")],
        "display_names": [str(item["display_name"]) for item in items],
        "params": dict(params or {}),
    }


def _value_descriptor(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw_value = value.get("value")
        if raw_value is None:
            raw_value = value.get("object")
        if raw_value is None:
            raw_value = value.get("name")
        if raw_value is None:
            raw_value = value.get("id")
        object_name = str(value.get("object") or raw_value or "").strip()
        display_name = str(value.get("display_name") or value.get("label") or _display_name(raw_value))
        return {
            "value": raw_value,
            "object_name": object_name,
            "display_name": display_name,
            "meta": dict(value.get("meta") or {}) if isinstance(value.get("meta"), Mapping) else {},
        }
    return {
        "value": value,
        "object_name": str(value) if isinstance(value, str) else "",
        "display_name": _display_name(value),
        "meta": {},
    }


def _resolve_context_path(context: Mapping[str, Any], path: str) -> Any:
    current: Any = context
    for raw_part in [part for part in str(path).split(".") if part]:
        if isinstance(current, Mapping):
            if raw_part not in current:
                raise RuntimePlanSampleError(f"runtime output source path not found: {path}")
            current = current[raw_part]
        elif isinstance(current, list):
            try:
                current = current[int(raw_part)]
            except (ValueError, IndexError) as exc:
                raise RuntimePlanSampleError(f"runtime output source list index invalid: {path}") from exc
        else:
            raise RuntimePlanSampleError(f"runtime output source path cannot descend into scalar: {path}")
    return current


def _display_name(value: Any) -> str:
    text = str(value)
    if text.startswith("button_"):
        text = text[len("button_") :]
    return text.replace("_", " ")


def _plan_suffix(plan_id: str) -> str:
    if ":" not in plan_id:
        return plan_id
    return plan_id.rsplit(":", 1)[1]


def _select_case(
    cases: Any,
    rng: random.Random,
    *,
    selector: Mapping[str, Any] | None = None,
    plan_id: str = "",
    selector_id: str = "",
    env_id: int = 0,
    episode_index: int | None = None,
) -> Mapping[str, Any] | None:
    items = sorted(
        [case for case in cases or [] if isinstance(case, Mapping)],
        key=lambda case: (str(case.get("id") or ""), stable_fingerprint(case)),
    )
    if not items:
        return None
    if isinstance(selector, Mapping) and selector.get("candidate_field"):
        return _select_grouped_case(
            items,
            rng,
            selector=selector,
            plan_id=plan_id,
            selector_id=selector_id,
            env_id=env_id,
            episode_index=episode_index,
        )
    selector_type = str((selector or {}).get("type") or "").strip().lower() if isinstance(selector, Mapping) else ""
    if selector_type == "schedule":
        return _select_scheduled_case(
            items,
            selector,
            plan_id=plan_id,
            selector_id=selector_id,
            env_id=env_id,
            episode_index=episode_index,
        )
    if selector_type in {"uniform", "weighted"}:
        return _select_mapping_candidate(
            items,
            rng,
            selector=selector,
            plan_id=plan_id,
            selector_id=selector_id,
            env_id=env_id,
            episode_index=episode_index,
        )
    selected = weighted_choice(items, rng=rng, weight_fn=_mapping_weight)
    return dict(selected) if selected is not None else None


def _select_grouped_case(
    cases: list[Mapping[str, Any]],
    rng: random.Random,
    *,
    selector: Mapping[str, Any],
    plan_id: str,
    selector_id: str,
    env_id: int,
    episode_index: int | None,
) -> Mapping[str, Any] | None:
    field = str(selector.get("candidate_field") or "").strip()
    if not field:
        return None
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        candidate_id = str(case.get(field) or "").strip()
        if candidate_id:
            groups.setdefault(candidate_id, []).append(case)
    if not groups:
        return None
    group_items = [
        {"id": candidate_id, "weight": _group_selector_weight(selector, candidate_id)}
        for candidate_id in sorted(groups)
    ]
    if str(selector.get("type") or "").strip().lower() == "uniform":
        group_items = [{"id": candidate_id, "weight": 1.0} for candidate_id in sorted(groups)]
    selected_group = _select_mapping_candidate(
        group_items,
        rng,
        selector=selector,
        plan_id=plan_id,
        selector_id=f"{selector_id}:{field}",
        env_id=env_id,
        episode_index=episode_index,
    )
    if selected_group is None:
        return None
    candidate_id = str(selected_group.get("id") or "")
    selected_case = weighted_choice(groups.get(candidate_id, ()), rng=rng, weight_fn=_mapping_weight)
    if selected_case is None:
        return None
    out = dict(selected_case)
    out["_selector_result"] = {
        "type": str(selector.get("type") or "weighted"),
        "candidate_field": field,
        "selected_id": candidate_id,
        "case_id": str(out.get("id") or ""),
        "group_case_count": len(groups.get(candidate_id, ())),
        "plan_id": str(plan_id),
        "selector_id": str(selector_id),
        "env_id": int(env_id),
    }
    if isinstance(selected_group.get("_selector_result"), Mapping):
        out["_selector_result"]["candidate_selector"] = dict(selected_group["_selector_result"])
    return out


def _group_selector_weight(selector: Mapping[str, Any], candidate_id: str) -> float:
    if str(selector.get("type") or "").strip().lower() == "weighted":
        weights = selector.get("weights") if isinstance(selector.get("weights"), Mapping) else {}
        raw = weights.get(candidate_id, selector.get("default_weight", 0.0))
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 0.0
    return 1.0


def _case_selector(assignment: Mapping[str, Any]) -> dict[str, Any]:
    meta = assignment.get("meta") if isinstance(assignment.get("meta"), Mapping) else {}
    selector = meta.get("selector") if isinstance(meta.get("selector"), Mapping) else None
    return dict(selector or {})


def _select_scheduled_case(
    cases: list[Mapping[str, Any]],
    selector: Mapping[str, Any],
    *,
    plan_id: str,
    selector_id: str,
    env_id: int,
    episode_index: int | None,
) -> Mapping[str, Any] | None:
    by_id = {str(case.get("id") or ""): case for case in cases if str(case.get("id") or "")}
    sequence = _schedule_sequence(
        selector, case_ids=set(by_id), plan_id=plan_id, selector_id=selector_id, env_id=env_id
    )
    if not sequence:
        return None
    index = int(episode_index or 0)
    if index < len(sequence):
        selected_id = sequence[index]
    else:
        on_exhausted = _schedule_on_exhausted(selector)
        if on_exhausted == "repeat":
            selected_id = sequence[index % len(sequence)]
        elif on_exhausted == "hold_last":
            selected_id = sequence[-1]
        else:
            raise RuntimePlanSampleError(
                f"schema_vnext_selector_schedule_exhausted: {selector_id} exhausted at episode {index}"
            )
    selected = by_id.get(str(selected_id))
    if selected is None:
        raise RuntimePlanSampleError(f"schedule selector references unknown case: {selector_id}.{selected_id}")
    out = dict(selected)
    out["_selector_result"] = {
        "type": "schedule",
        "mode": str(selector.get("mode") or ""),
        "index": index,
        "selected_id": str(selected_id),
        "sequence_length": len(sequence),
        "plan_id": str(plan_id),
        "selector_id": str(selector_id),
        "env_id": int(env_id),
    }
    return out


def _schedule_sequence(
    selector: Mapping[str, Any],
    *,
    case_ids: set[str],
    plan_id: str,
    selector_id: str,
    env_id: int,
) -> tuple[str, ...]:
    mode = str(selector.get("mode") or "").strip().lower()
    if mode == "cycle":
        sequence = tuple(str(item) for item in selector.get("order", []) or [])
    elif mode == "blocks":
        out: list[str] = []
        for raw in selector.get("blocks", []) or []:
            if not isinstance(raw, Mapping):
                continue
            candidate_id = str(raw.get("id") or "").strip()
            try:
                count = int(raw.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            if candidate_id and count > 0:
                out.extend([candidate_id] * count)
        sequence = tuple(out)
    elif mode == "counts":
        raw_counts = selector.get("counts") if isinstance(selector.get("counts"), Mapping) else {}
        counts = {str(key): int(value) for key, value in raw_counts.items() if int(value) > 0}
        order = tuple(str(item) for item in selector.get("order", []) or []) or tuple(sorted(counts))
        sequence_list = [candidate_id for candidate_id in order for _ in range(counts.get(candidate_id, 0))]
        if str(selector.get("order_policy") or "shuffle_once").strip() == "shuffle_once":
            rng = stable_rng(
                0,
                "runtime_schedule_sequence",
                str(plan_id),
                str(selector_id),
                int(env_id),
                payload=dict(selector),
            )
            rng.shuffle(sequence_list)
        sequence = tuple(sequence_list)
    else:
        raise RuntimePlanSampleError(f"unsupported schedule selector mode: {mode}")
    unknown = sorted(set(sequence) - set(case_ids))
    if unknown:
        raise RuntimePlanSampleError(f"schedule selector references unknown cases: {selector_id}: {unknown}")
    if not sequence:
        raise RuntimePlanSampleError(f"schedule selector has empty sequence: {selector_id}")
    return sequence


def _schedule_on_exhausted(selector: Mapping[str, Any]) -> str:
    mode = str(selector.get("mode") or "").strip().lower()
    default = "repeat" if mode == "cycle" else "stop"
    value = str(selector.get("on_exhausted") or default).strip()
    return value if value in {"repeat", "hold_last", "stop"} else default


def _select_mapping_candidate(
    items: list[Mapping[str, Any]],
    rng: random.Random,
    *,
    selector: Mapping[str, Any] | None,
    plan_id: str,
    selector_id: str,
    env_id: int,
    episode_index: int | None,
) -> Mapping[str, Any] | None:
    candidates = sorted(items, key=lambda item: (_candidate_id(item), stable_fingerprint(item)))
    if not candidates:
        return None
    selector_type = str((selector or {}).get("type") or "").strip().lower() if isinstance(selector, Mapping) else ""
    if selector_type == "schedule":
        return _select_scheduled_case(
            candidates,
            selector or {},
            plan_id=plan_id,
            selector_id=selector_id,
            env_id=env_id,
            episode_index=episode_index,
        )
    if selector_type == "uniform":
        selected = rng.choice(candidates)
        out = dict(selected)
        out["_selector_result"] = {
            "type": "uniform",
            "selected_id": _candidate_id(selected),
            "candidate_count": len(candidates),
            "plan_id": str(plan_id),
            "selector_id": str(selector_id),
            "env_id": int(env_id),
        }
        return out
    if selector_type == "weighted":
        selected = weighted_choice(candidates, rng=rng, weight_fn=lambda item: _selector_weight(selector or {}, item))
        if selected is None:
            return None
        out = dict(selected)
        out["_selector_result"] = {
            "type": "weighted",
            "selected_id": _candidate_id(selected),
            "weights": {_candidate_id(item): _selector_weight(selector or {}, item) for item in candidates},
            "plan_id": str(plan_id),
            "selector_id": str(selector_id),
            "env_id": int(env_id),
        }
        return out
    selected = weighted_choice(candidates, rng=rng, weight_fn=_mapping_weight)
    return dict(selected) if selected is not None else None


def _selector_weight(selector: Mapping[str, Any], item: Mapping[str, Any]) -> float:
    weights = selector.get("weights") if isinstance(selector.get("weights"), Mapping) else {}
    raw_default = selector.get("default_weight", 0.0)
    raw = weights.get(_candidate_id(item), raw_default)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _candidate_id(item: Mapping[str, Any]) -> str:
    for key in ("id", "choice_id", "pose_id", "yaw_id", "range_id"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return stable_fingerprint(item)


def _domain_choice_selector(domain: Mapping[str, Any], choice_kind: str) -> dict[str, Any]:
    raw_meta = domain.get("meta")
    meta = raw_meta if isinstance(raw_meta, Mapping) else {}
    selectors = meta.get("selectors") if isinstance(meta.get("selectors"), Mapping) else {}
    selector = selectors.get(choice_kind)
    return dict(selector or {}) if isinstance(selector, Mapping) else {}


def _apply_domain_yaw_delta(
    domain: Mapping[str, Any],
    yaw: float,
    rng: random.Random,
) -> tuple[float, dict[str, Any] | None]:
    raw_meta = domain.get("meta")
    meta = raw_meta if isinstance(raw_meta, Mapping) else {}
    if "yaw_delta" not in meta:
        return yaw, None
    delta = _sample_range(meta.get("yaw_delta"), rng, default=0.0) or 0.0
    return yaw + float(delta), {"kind": "yaw_delta", "yaw_delta": float(delta), "yaw": yaw + float(delta)}


def _mapping_weight(item: Mapping[str, Any]) -> float:
    raw = item.get("weight", 1.0)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _sample_yaw_for_domain(
    domain: Mapping[str, Any],
    rng: random.Random,
    *,
    plan_id: str,
    selector_id: str,
    episode_index: int | None,
    env_id: int,
) -> tuple[float, dict[str, Any] | None]:
    choices = [item for item in domain.get("yaw_choices", []) or [] if isinstance(item, Mapping)]
    selected = (
        _select_mapping_candidate(
            choices,
            rng,
            selector=_domain_choice_selector(domain, "yaws"),
            plan_id=plan_id,
            selector_id=selector_id,
            env_id=env_id,
            episode_index=episode_index,
        )
        if choices
        else None
    )
    if selected is not None:
        choice = dict(selected)
        yaw = float(choice.get("yaw", 0.0) or 0.0)
        yaw, delta_sample = _apply_domain_yaw_delta(domain, yaw, rng)
        sample = {
            "yaw_id": str(choice.get("id") or choice.get("yaw_id") or ""),
            "yaw": yaw,
            "yaw_weight": _mapping_weight(choice),
            "meta": dict(choice.get("meta") or {}) if isinstance(choice.get("meta"), Mapping) else {},
        }
        if delta_sample:
            sample.update(delta_sample)
        return yaw, sample
    yaw = _sample_range(domain.get("yaw"), rng, default=0.0) or 0.0
    yaw, delta_sample = _apply_domain_yaw_delta(domain, yaw, rng)
    return yaw, delta_sample


def _attach_yaw_sample(domain_sample: dict[str, Any], yaw_sample: Mapping[str, Any] | None) -> None:
    if not yaw_sample:
        return
    if yaw_sample.get("yaw_id"):
        domain_sample["yaw_id"] = str(yaw_sample.get("yaw_id"))
    if isinstance(yaw_sample.get("yaw"), (int, float)):
        domain_sample["yaw"] = float(yaw_sample["yaw"])
    if isinstance(yaw_sample.get("yaw_weight"), (int, float)):
        domain_sample["yaw_weight"] = float(yaw_sample["yaw_weight"])
    if "weight" not in domain_sample and "yaw_weight" in domain_sample:
        domain_sample["weight"] = domain_sample["yaw_weight"]


def _sample_cell_xy(cell: Mapping[str, Any], rng: random.Random) -> tuple[float, float] | None:
    shape = str(cell.get("shape") or "rectangle").strip().lower()
    if shape != "polygon":
        x = _sample_range(cell.get("x"), rng, default=_range_center(cell.get("x")))
        y = _sample_range(cell.get("y"), rng, default=_range_center(cell.get("y")))
        return (x, y) if x is not None and y is not None else None
    bbox = cell.get("bbox") if isinstance(cell.get("bbox"), Mapping) else cell
    outer = cell.get("outer")
    if not isinstance(outer, list):
        return None
    for _ in range(POLYGON_CELL_SAMPLE_TRIES):
        x = _sample_range(bbox.get("x"), rng, default=_range_center(bbox.get("x")))  # type: ignore[union-attr]
        y = _sample_range(bbox.get("y"), rng, default=_range_center(bbox.get("y")))  # type: ignore[union-attr]
        if x is not None and y is not None and _polygon_contains(cell, x, y):
            return (x, y)
    return None


def _sample_cylinder_xy(domain: Mapping[str, Any], rng: random.Random) -> tuple[float, float] | None:
    cx = _range_center(domain.get("x"))
    cy = _range_center(domain.get("y"))
    if cx is None or cy is None:
        return None
    meta = domain.get("meta") if isinstance(domain.get("meta"), Mapping) else {}
    radius = float(meta.get("radius") or 0.0)
    if radius <= 0.0:
        x = domain.get("x")
        y = domain.get("y")
        if isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(y, (list, tuple)) and len(y) == 2:
            radius = min(abs(float(x[1]) - float(x[0])), abs(float(y[1]) - float(y[0]))) * 0.5
    if radius <= 0.0:
        return (cx, cy)
    r = radius * math.sqrt(rng.random())
    theta = 2.0 * math.pi * rng.random()
    return (cx + math.cos(theta) * r, cy + math.sin(theta) * r)


def _polygon_contains(cell: Mapping[str, Any], x: float, y: float) -> bool:
    outer = cell.get("outer")
    holes = cell.get("holes", [])
    try:
        polygon = Polygon2D.from_payload(outer, holes)
    except Exception:
        return False
    return polygon.contains_point(float(x), float(y))


def _surface_local_to_world(surface: Mapping[str, Any], x: float, y: float, z: float | None) -> Vec3:
    center = _vec3(surface.get("center"), default=(0.0, 0.0, 0.0))
    x_axis = _vec3(surface.get("x_axis"), default=(1.0, 0.0, 0.0))
    y_axis = _vec3(surface.get("y_axis"), default=(0.0, 1.0, 0.0))
    normal = _vec3(surface.get("normal"), default=(0.0, 0.0, 1.0))
    z_world = center[2] if z is None else float(z)
    z_offset = z_world - center[2]
    return vec_add(
        vec_add(vec_add(center, vec_mul(x_axis, float(x))), vec_mul(y_axis, float(y))),
        vec_mul(normal, z_offset),
    )


def _sample_range(raw: Any, rng: random.Random, *, default: float | None = None) -> float | None:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        return float(raw[0])
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo = float(raw[0])
        hi = float(raw[1])
        if abs(hi - lo) <= 1e-12:
            return lo
        return rng.uniform(lo, hi)
    return default


def _range_center(raw: Any) -> float | None:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        return float(raw[0])
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (float(raw[0]) + float(raw[1])) * 0.5
    return None


def _vec3(raw: Any, *, default: Vec3 | None = None) -> Vec3:
    if raw is None:
        if default is None:
            raise ValueError("vec3_missing")
        return default
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _quat(raw: Any) -> Quat:
    try:
        return quat_normalize(raw)
    except Exception as exc:
        raise RuntimePlanSampleError(f"invalid runtime quaternion: {raw!r}") from exc


def _identity_quat() -> Quat:
    return (0.0, 0.0, 0.0, 1.0)
