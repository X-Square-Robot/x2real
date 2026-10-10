"""Reset-time free-space rejection sampling for distractor clutter.

Modeled on ``replay_insert_objects.placement`` (without arm swept-volume): after
task objects are sampled, re-place every ``distractor``-tagged object uniformly
over the usable support surface minus task AABBs (plus a conservative clearance)
and minus any task keep-out zone. Distractor placement is intentionally
independent of the scene's authored task bands: the authored node domain is only
a compile-time feasibility hint and is ignored here. A reset never hard-fails;
infeasible distractors keep their original pose and emit a diagnostic.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..common.constants import GEOM_EPS
from ..common.geometry import (
    Bounds2D,
    Bounds3D,
    Range1D,
    Vec3,
    dot,
    projected_footprint_bounds,
    vec_add,
    vec_mul,
    vec_sub,
)
from ..runtime_sampling.sampler import RuntimePoseSample

LOGGER = logging.getLogger(__name__)

DEFAULT_DISTRACTOR_CLEARANCE_M = 0.05
DEFAULT_DISTRACTOR_ATTEMPTS = 96
_CLEARANCE_BACKOFF = (1.0, 0.5, 0.25, 0.0)


@dataclass(frozen=True)
class DistractorFreeSpaceResult:
    poses: dict[str, RuntimePoseSample]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _SurfaceObject:
    object_id: str
    pose: RuntimePoseSample
    surface_id: str
    surface: Mapping[str, Any]
    local_bounds: Bounds3D
    footprint: Bounds2D
    world_rect: Bounds2D
    area: float
    x_axis: Vec3
    y_axis: Vec3
    center: Vec3


def reposition_distractors_in_free_space(
    sampled_poses: Mapping[str, RuntimePoseSample],
    *,
    scene_payload: Mapping[str, Any],
    runtime_plan: Mapping[str, Any] | None,
    runtime_slot_bounds: Mapping[str, Any] | None,
    applied_assets: Mapping[str, Any],
    context: Mapping[str, Any] | None,
    rng: random.Random,
    clearance_m: float = DEFAULT_DISTRACTOR_CLEARANCE_M,
    attempts: int = DEFAULT_DISTRACTOR_ATTEMPTS,
    keepout_rects_by_surface: Mapping[str, Sequence[Bounds2D]] | None = None,
    support_surface_lookup: Any | None = None,
) -> DistractorFreeSpaceResult:
    """Reposition distractors into free tabletop space after task objects.

    Non-distractor poses are never modified. Distractors keep their sampled yaw
    and Z; only the surface-local (x, y) origin is re-drawn, uniformly over the
    whole usable surface.

    ``keepout_rects_by_surface`` adds task-owned exclusion rects in surface-local
    coordinates (for example the beside-placement success zone).

    ``support_surface_lookup`` is ``(runtime_plan, object_id, context) -> surface``
    and defaults to the runtime_events helper when available.
    """

    poses = {str(name): pose for name, pose in dict(sampled_poses or {}).items()}
    distractor_ids = _distractor_object_ids(scene_payload)
    distractor_ids &= set(poses)
    diagnostics: dict[str, Any] = {
        "status": "skipped",
        "clearance_m": float(clearance_m),
        "attempts": int(attempts),
        "distractor_ids": sorted(distractor_ids),
        "moved": [],
        "kept_original": [],
        "warnings": [],
    }
    if not distractor_ids:
        return DistractorFreeSpaceResult(poses=poses, diagnostics=diagnostics)

    lookup = support_surface_lookup or _default_support_surface_lookup()
    surface_objects = _surface_objects(
        poses,
        runtime_plan=runtime_plan,
        runtime_slot_bounds=runtime_slot_bounds,
        applied_assets=applied_assets,
        context=context,
        support_surface_lookup=lookup,
    )
    by_surface: dict[str, list[_SurfaceObject]] = {}
    for item in surface_objects.values():
        by_surface.setdefault(item.surface_id, []).append(item)

    moved: list[str] = []
    kept: list[str] = []
    warnings: list[str] = []
    for surface_id, members in sorted(by_surface.items()):
        distractors = [item for item in members if item.object_id in distractor_ids]
        if not distractors:
            continue
        obstacles = [item.world_rect for item in members if item.object_id not in distractor_ids]
        obstacles.extend((keepout_rects_by_surface or {}).get(surface_id, ()))
        usable = _usable_surface_bounds(members[0].surface)
        if usable is None:
            for item in distractors:
                kept.append(item.object_id)
                warnings.append(f"usable_surface_empty:{surface_id}:{item.object_id}")
            continue
        distractors.sort(key=lambda item: (-item.area, item.object_id))
        for item in distractors:
            new_pose = _place_distractor(
                item,
                usable=usable,
                obstacles=obstacles,
                rng=rng,
                clearance_m=float(clearance_m),
                attempts=max(1, int(attempts)),
            )
            if new_pose is None:
                kept.append(item.object_id)
                warnings.append(f"placement_exhausted:{surface_id}:{item.object_id}")
                obstacles.append(item.world_rect)
                continue
            poses[item.object_id] = new_pose
            obstacles.append(_world_rect_at(item, new_pose))
            if _position_changed(item.pose.position, new_pose.position):
                moved.append(item.object_id)
            else:
                kept.append(item.object_id)

    diagnostics.update(
        {
            "status": "applied",
            "moved": moved,
            "kept_original": kept,
            "warnings": warnings,
        }
    )
    for warning in warnings:
        LOGGER.warning("schema-vNext distractor free-space: %s", warning)
    return DistractorFreeSpaceResult(poses=poses, diagnostics=diagnostics)


def surface_local_rect_from_world_zone(
    surface: Mapping[str, Any],
    *,
    world_center: Sequence[float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> Bounds2D:
    """Project a world-axis-aligned zone AABB into surface-local (x, y).

    Used for task keep-out zones (``place_beside``) whose bands are authored as
    world-frame offsets from a reference object. Rotated surfaces get the
    conservative AABB of the projected corners.
    """

    center = _surface_center(surface)
    x_axis = _surface_vector(surface, "x_axis", default=(1.0, 0.0, 0.0))
    y_axis = _surface_vector(surface, "y_axis", default=(0.0, 1.0, 0.0))
    xs: list[float] = []
    ys: list[float] = []
    for dx in (float(x_range[0]), float(x_range[1])):
        for dy in (float(y_range[0]), float(y_range[1])):
            corner = (float(world_center[0]) + dx, float(world_center[1]) + dy, float(center[2]))
            local = vec_sub(corner, center)
            xs.append(float(dot(local, x_axis)))
            ys.append(float(dot(local, y_axis)))
    return Bounds2D(Range1D(min(xs), max(xs)), Range1D(min(ys), max(ys)))


def _default_support_surface_lookup():
    from .runtime_events import _runtime_support_surface

    def _lookup(runtime_plan, object_id, context):
        return _runtime_support_surface(runtime_plan, object_id=object_id, context=context)

    return _lookup


def _distractor_object_ids(scene_payload: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for raw in scene_payload.get("objects") or []:
        if not isinstance(raw, Mapping):
            continue
        tags = raw.get("semantic_tags") or []
        if not isinstance(tags, (list, tuple)):
            continue
        if "distractor" in {str(tag).strip() for tag in tags}:
            object_id = str(raw.get("id") or "").strip()
            if object_id:
                out.add(object_id)
    return out


def _surface_objects(
    poses: Mapping[str, RuntimePoseSample],
    *,
    runtime_plan: Mapping[str, Any] | None,
    runtime_slot_bounds: Mapping[str, Any] | None,
    applied_assets: Mapping[str, Any],
    context: Mapping[str, Any] | None,
    support_surface_lookup,
) -> dict[str, _SurfaceObject]:
    out: dict[str, _SurfaceObject] = {}
    for object_id, pose in poses.items():
        surface = support_surface_lookup(runtime_plan, object_id, context)
        if not isinstance(surface, Mapping):
            continue
        local_bounds = _local_bounds_for_object(
            object_id,
            runtime_slot_bounds=runtime_slot_bounds,
            applied_assets=applied_assets,
            runtime_plan=runtime_plan,
            context=context,
        )
        if local_bounds is None:
            continue
        try:
            x_axis = _surface_vector(surface, "x_axis", default=(1.0, 0.0, 0.0))
            y_axis = _surface_vector(surface, "y_axis", default=(0.0, 1.0, 0.0))
            center = _surface_center(surface)
        except (TypeError, ValueError):
            continue
        footprint = projected_footprint_bounds(local_bounds, pose.quaternion, x_axis, y_axis)
        world_rect = _pose_world_rect(pose, center=center, footprint=footprint, x_axis=x_axis, y_axis=y_axis)
        out[object_id] = _SurfaceObject(
            object_id=object_id,
            pose=pose,
            surface_id=str(surface.get("id") or ""),
            surface=surface,
            local_bounds=local_bounds,
            footprint=footprint,
            world_rect=world_rect,
            area=float(footprint.x.length * footprint.y.length),
            x_axis=x_axis,
            y_axis=y_axis,
            center=center,
        )
    return out


def _local_bounds_for_object(
    object_id: str,
    *,
    runtime_slot_bounds: Mapping[str, Any] | None,
    applied_assets: Mapping[str, Any],
    runtime_plan: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
) -> Bounds3D | None:
    by_object = applied_assets.get("by_object") if isinstance(applied_assets, Mapping) else None
    active_slot = object_id
    if isinstance(by_object, Mapping):
        report = by_object.get(object_id)
        if isinstance(report, Mapping):
            active_slot = str(report.get("active_slot") or object_id).strip() or object_id
    if isinstance(runtime_slot_bounds, Mapping):
        object_bounds = runtime_slot_bounds.get(object_id)
        if isinstance(object_bounds, Mapping):
            for key in (active_slot, object_id):
                payload = object_bounds.get(key)
                if isinstance(payload, Mapping):
                    try:
                        return Bounds3D.from_dict(dict(payload))
                    except (KeyError, TypeError, ValueError):
                        pass
    return _footprint_bounds3d_from_domain(object_id, runtime_plan=runtime_plan, context=context)


def _footprint_bounds3d_from_domain(
    object_id: str,
    *,
    runtime_plan: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
) -> Bounds3D | None:
    domain = _domain_for_object(object_id, runtime_plan=runtime_plan, context=context)
    if domain is None:
        return None
    meta = domain.get("meta") if isinstance(domain.get("meta"), Mapping) else {}
    footprint_bounds = meta.get("footprint_bounds")
    if isinstance(footprint_bounds, Mapping):
        try:
            bounds2 = Bounds2D.from_dict(dict(footprint_bounds))
            return Bounds3D(bounds2.x, bounds2.y, Range1D(0.0, 0.01))
        except (KeyError, TypeError, ValueError):
            pass
    footprint = meta.get("footprint")
    if isinstance(footprint, (list, tuple)) and len(footprint) == 2:
        sx = max(GEOM_EPS, float(footprint[0]))
        sy = max(GEOM_EPS, float(footprint[1]))
        return Bounds3D(
            Range1D(-0.5 * sx, 0.5 * sx),
            Range1D(-0.5 * sy, 0.5 * sy),
            Range1D(0.0, 0.01),
        )
    return None


def _domain_for_object(
    object_id: str,
    *,
    runtime_plan: Mapping[str, Any] | None,
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
    if domain_id:
        for domain in domains:
            if str(domain.get("id") or "") == domain_id:
                return domain
        return None
    matches = [domain for domain in domains if str(domain.get("object") or "") == object_id]
    return matches[0] if len(matches) == 1 else None


def _usable_surface_bounds(surface: Mapping[str, Any]) -> Bounds2D | None:
    raw_bounds = surface.get("bounds")
    if not isinstance(raw_bounds, Mapping):
        return None
    try:
        bounds = Bounds2D.from_dict(dict(raw_bounds))
    except (KeyError, TypeError, ValueError):
        return None
    return bounds.shrink(float(surface.get("margin", 0.0) or 0.0))


def _place_distractor(
    item: _SurfaceObject,
    *,
    usable: Bounds2D,
    obstacles: list[Bounds2D],
    rng: random.Random,
    clearance_m: float,
    attempts: int,
) -> RuntimePoseSample | None:
    """Uniform rejection sampling over the whole usable surface.

    The authored domain is deliberately not consulted, and the sampled pose is
    always re-drawn: distractor layout must be independent of the task bands,
    otherwise clutter reproduces the authored strip. The original pose is kept
    only when every clearance level is exhausted (caller emits a diagnostic).
    """

    for scale in _CLEARANCE_BACKOFF:
        gap = max(0.0, float(clearance_m) * float(scale))
        origin_domain = _origin_domain(usable, item.footprint, pad=gap)
        if origin_domain is None:
            continue
        for _attempt in range(attempts):
            ox = rng.uniform(origin_domain.x.min, origin_domain.x.max)
            oy = rng.uniform(origin_domain.y.min, origin_domain.y.max)
            candidate = Bounds2D(
                Range1D(ox + item.footprint.x.min, ox + item.footprint.x.max),
                Range1D(oy + item.footprint.y.min, oy + item.footprint.y.max),
            )
            if not _rect_inside(candidate, usable, pad=gap):
                continue
            if any(not _rects_clear(candidate, obstacle, gap=gap) for obstacle in obstacles):
                continue
            world_xy = vec_add(item.center, vec_add(vec_mul(item.x_axis, ox), vec_mul(item.y_axis, oy)))
            position = (float(world_xy[0]), float(world_xy[1]), float(item.pose.position[2]))
            return replace(item.pose, position=position)
    return None


def _origin_domain(allowed: Bounds2D, footprint: Bounds2D, *, pad: float) -> Bounds2D | None:
    margin = max(0.0, float(pad))
    x = Range1D(
        allowed.x.min + margin - footprint.x.min,
        allowed.x.max - margin - footprint.x.max,
    )
    y = Range1D(
        allowed.y.min + margin - footprint.y.min,
        allowed.y.max - margin - footprint.y.max,
    )
    if x.min > x.max + GEOM_EPS or y.min > y.max + GEOM_EPS:
        return None
    return Bounds2D(x, y)


def _pose_world_rect(
    pose: RuntimePoseSample,
    *,
    center: Vec3,
    footprint: Bounds2D,
    x_axis: Vec3,
    y_axis: Vec3,
) -> Bounds2D:
    local = vec_sub(tuple(float(v) for v in pose.position), center)
    ox = float(dot(local, x_axis))
    oy = float(dot(local, y_axis))
    return Bounds2D(
        Range1D(ox + footprint.x.min, ox + footprint.x.max),
        Range1D(oy + footprint.y.min, oy + footprint.y.max),
    )


def _world_rect_at(item: _SurfaceObject, pose: RuntimePoseSample) -> Bounds2D:
    return _pose_world_rect(
        pose,
        center=item.center,
        footprint=item.footprint,
        x_axis=item.x_axis,
        y_axis=item.y_axis,
    )


def _surface_center(surface: Mapping[str, Any]) -> Vec3:
    raw = surface.get("center", (0.0, 0.0, 0.0))
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("runtime_surface_center_must_be_vec3")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _surface_vector(surface: Mapping[str, Any], key: str, *, default: Vec3) -> Vec3:
    raw = surface.get(key, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"runtime_surface_{key}_must_be_vec3")
    values = tuple(float(value) for value in raw)
    length = sum(value * value for value in values) ** 0.5
    if length <= 1e-9:
        raise ValueError(f"runtime_surface_{key}_zero_length")
    return tuple(value / length for value in values)  # type: ignore[return-value]


def _rect_inside(rect: Bounds2D, container: Bounds2D, *, pad: float) -> bool:
    margin = max(0.0, float(pad))
    return (
        rect.x.min + GEOM_EPS >= container.x.min + margin
        and rect.x.max - GEOM_EPS <= container.x.max - margin
        and rect.y.min + GEOM_EPS >= container.y.min + margin
        and rect.y.max - GEOM_EPS <= container.y.max - margin
    )


def _rects_clear(left: Bounds2D, right: Bounds2D, *, gap: float) -> bool:
    """True when footprints are separated by ``gap`` on at least one axis."""

    return _range_gap(left.x, right.x) >= float(gap) - GEOM_EPS or _range_gap(left.y, right.y) >= float(gap) - GEOM_EPS


def _range_gap(left: Range1D, right: Range1D) -> float:
    return max(right.min - left.max, left.min - right.max)


def _position_changed(left: tuple[float, float, float], right: tuple[float, float, float]) -> bool:
    return any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(left, right, strict=True))
