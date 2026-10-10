from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from ...common.diagnostics import LayoutDiagnostic
from ...common.geometry import Bounds2D, Range1D
from ...scene_spec.schema import SurfaceRangeSpec


@dataclass(frozen=True)
class LocalPlacement:
    name: str
    x: float
    y: float
    yaw: float | None = None


RangeForObject = Callable[[str, SurfaceRangeSpec | None], Bounds2D | None]
FitBoundsForObject = Callable[[Bounds2D, str], Bounds2D | None]
FootprintSize = Callable[[str], tuple[float, float]]
PlacementPredicate = Callable[[LocalPlacement], bool]
YawRangeInput = tuple[float, float] | dict[str, tuple[float, float]] | None


@dataclass(frozen=True)
class GroupObjectDomain:
    name: str
    bounds: Bounds2D
    footprint: tuple[float, float]


@dataclass(frozen=True)
class GroupCandidateSet:
    candidates_by_name: dict[str, list[LocalPlacement]]
    meta: dict = field(default_factory=dict)


class GroupDomainBuilder:
    """Build per-object center domains for tabletop group relations.

    The builder intentionally delegates support semantics to SupportOnProvider
    through range_for_object. Region/scatter wrappers only add their group
    range; object-level support ranges and variant overrides remain owned by
    the support provider.
    """

    def __init__(
        self,
        *,
        range_for_object: RangeForObject,
        fit_bounds_for_object: FitBoundsForObject,
        footprint_size: FootprintSize,
    ) -> None:
        self.range_for_object = range_for_object
        self.fit_bounds_for_object = fit_bounds_for_object
        self.footprint_size = footprint_size

    def build(
        self,
        *,
        names: list[str],
        group_surface_range: SurfaceRangeSpec | None,
        target: str,
        relation: str,
    ) -> dict[str, GroupObjectDomain] | LayoutDiagnostic:
        domains: dict[str, GroupObjectDomain] = {}
        for name in names:
            bounds = self.range_for_object(name, group_surface_range)
            if bounds is None:
                return LayoutDiagnostic(
                    "surface_range_empty", "object has empty group/support surface range", target=name
                )
            fit_bounds = self.fit_bounds_for_object(bounds, name)
            if fit_bounds is None:
                return LayoutDiagnostic(
                    "object_larger_than_surface",
                    "object footprint does not fit in group/support domain",
                    target=name,
                    meta={"scope": relation, "group": target},
                )
            domains[name] = GroupObjectDomain(name=name, bounds=fit_bounds, footprint=self.footprint_size(name))
        return domains


def random_candidates(
    *,
    domains: dict[str, GroupObjectDomain],
    rng: random.Random,
    attempts: int,
    valid_placement: PlacementPredicate,
    scope_id: str,
    packer: str,
    yaw_range: YawRangeInput = None,
) -> GroupCandidateSet | LayoutDiagnostic:
    candidates_by_name: dict[str, list[LocalPlacement]] = {}
    rejection_counts: dict[str, int] = {}
    for domain in domains.values():
        candidates: list[LocalPlacement] = []
        for _ in range(max(1, int(attempts))):
            x = rng.uniform(domain.bounds.x.min, domain.bounds.x.max)
            y = rng.uniform(domain.bounds.y.min, domain.bounds.y.max)
            item = _local_placement(domain.name, x, y, rng=rng, yaw_range=yaw_range)
            if valid_placement(item):
                candidates.append(item)
            else:
                _inc(rejection_counts, "invalid_placement")
        if not candidates:
            return LayoutDiagnostic(
                f"{packer}_no_candidate",
                "group domain produced no valid random candidate",
                target=domain.name,
                meta={"scope_id": scope_id, "packer": packer, "rejection_counts": rejection_counts},
            )
        candidates_by_name[domain.name] = candidates
    return GroupCandidateSet(
        candidates_by_name=candidates_by_name,
        meta={
            "scope_id": scope_id,
            "packer": packer,
            "candidate_generator": "random",
            "candidate_counts": {name: len(cands) for name, cands in candidates_by_name.items()},
            "rejection_counts": rejection_counts,
        },
    )


def stratified_candidates(
    *,
    domains: dict[str, GroupObjectDomain],
    rng: random.Random,
    valid_placement: PlacementPredicate,
    scope_id: str,
    packer: str,
    jitter_ratio: float = 0.5,
    max_candidates_per_object: int = 512,
    target_min_candidates: int = 32,
    yaw_range: YawRangeInput = None,
) -> GroupCandidateSet | LayoutDiagnostic:
    candidates_by_name: dict[str, list[LocalPlacement]] = {}
    rejection_counts: dict[str, int] = {}
    raw_counts: dict[str, int] = {}
    for domain in domains.values():
        target = max(1, min(int(max_candidates_per_object), max(int(target_min_candidates), 1)))
        nx = max(1, int(math.ceil(math.sqrt(target))))
        ny = max(1, int(math.ceil(target / nx)))
        cell_w = domain.bounds.x.length / nx if nx > 0 else 0.0
        cell_h = domain.bounds.y.length / ny if ny > 0 else 0.0
        jitter = max(0.0, float(jitter_ratio))
        candidates: list[LocalPlacement] = []
        raw_count = 0
        for ix in range(nx):
            for iy in range(ny):
                raw_count += 1
                if domain.bounds.x.length <= 1e-9:
                    x = domain.bounds.x.center
                else:
                    lo = domain.bounds.x.min + ix * cell_w
                    hi = domain.bounds.x.min + (ix + 1) * cell_w
                    center = (lo + hi) * 0.5
                    half = (hi - lo) * 0.5 * min(1.0, jitter)
                    x = rng.uniform(center - half, center + half) if half > 0.0 else center
                if domain.bounds.y.length <= 1e-9:
                    y = domain.bounds.y.center
                else:
                    lo = domain.bounds.y.min + iy * cell_h
                    hi = domain.bounds.y.min + (iy + 1) * cell_h
                    center = (lo + hi) * 0.5
                    half = (hi - lo) * 0.5 * min(1.0, jitter)
                    y = rng.uniform(center - half, center + half) if half > 0.0 else center
                item = _local_placement(domain.name, float(x), float(y), rng=rng, yaw_range=yaw_range)
                if valid_placement(item):
                    candidates.append(item)
                else:
                    _inc(rejection_counts, "invalid_placement")
                if len(candidates) >= max_candidates_per_object:
                    break
            if len(candidates) >= max_candidates_per_object:
                break
        raw_counts[domain.name] = raw_count
        if not candidates:
            return LayoutDiagnostic(
                f"{packer}_no_candidate",
                "group domain produced no valid stratified candidate",
                target=domain.name,
                meta={"scope_id": scope_id, "packer": packer, "rejection_counts": rejection_counts},
            )
        rng.shuffle(candidates)
        candidates_by_name[domain.name] = candidates
    return GroupCandidateSet(
        candidates_by_name=candidates_by_name,
        meta={
            "scope_id": scope_id,
            "packer": packer,
            "candidate_generator": "stratified",
            "raw_candidate_counts": raw_counts,
            "candidate_counts": {name: len(cands) for name, cands in candidates_by_name.items()},
            "rejection_counts": rejection_counts,
        },
    )


def poisson_disk_candidates(
    *,
    domains: dict[str, GroupObjectDomain],
    rng: random.Random,
    attempts: int,
    valid_placement: PlacementPredicate,
    scope_id: str,
    packer: str,
    min_spacing: float = 0.0,
    max_candidates_per_object: int = 512,
    yaw_range: YawRangeInput = None,
) -> GroupCandidateSet | LayoutDiagnostic:
    candidates_by_name: dict[str, list[LocalPlacement]] = {}
    rejection_counts: dict[str, int] = {}
    raw_counts: dict[str, int] = {}
    for domain in domains.values():
        radius = _poisson_radius(domain, min_spacing=min_spacing)
        candidates: list[LocalPlacement] = []
        raw_count = 0
        for _ in range(max(1, int(attempts))):
            raw_count += 1
            x = rng.uniform(domain.bounds.x.min, domain.bounds.x.max)
            y = rng.uniform(domain.bounds.y.min, domain.bounds.y.max)
            item = _local_placement(domain.name, x, y, rng=rng, yaw_range=yaw_range)
            if not valid_placement(item):
                _inc(rejection_counts, "invalid_placement")
                continue
            if any(_dist2(item, other) < radius * radius - 1e-12 for other in candidates):
                _inc(rejection_counts, "poisson_radius")
                continue
            candidates.append(item)
            if len(candidates) >= max_candidates_per_object:
                break
        raw_counts[domain.name] = raw_count
        if not candidates:
            return LayoutDiagnostic(
                f"{packer}_no_candidate",
                "group domain produced no valid poisson-disk candidate",
                target=domain.name,
                meta={
                    "scope_id": scope_id,
                    "packer": packer,
                    "radius": radius,
                    "rejection_counts": rejection_counts,
                },
            )
        rng.shuffle(candidates)
        candidates_by_name[domain.name] = candidates
    return GroupCandidateSet(
        candidates_by_name=candidates_by_name,
        meta={
            "scope_id": scope_id,
            "packer": packer,
            "candidate_generator": "poisson_disk",
            "raw_candidate_counts": raw_counts,
            "candidate_counts": {name: len(cands) for name, cands in candidates_by_name.items()},
            "rejection_counts": rejection_counts,
            "poisson_radius_by_object": {
                name: _poisson_radius(domain, min_spacing=min_spacing) for name, domain in domains.items()
            },
        },
    )


def grid_candidates(
    *,
    domains: dict[str, GroupObjectDomain],
    rng: random.Random,
    valid_placement: PlacementPredicate,
    scope_id: str,
    packer: str,
    grid_step: float | None = None,
    grid_snap: object | None = None,
    align_axis: str | None = None,
    jitter_ratio: float = 0.0,
    max_candidates_per_object: int = 512,
    target_min_candidates: int = 16,
    yaw_range: YawRangeInput = None,
) -> GroupCandidateSet | LayoutDiagnostic:
    step = _grid_step(grid_step=grid_step, grid_snap=grid_snap, domains=domains)
    origin_x, origin_y = _grid_origin(grid_snap)
    snap_mode = _grid_mode(grid_snap)
    fixed_value = _common_aligned_value(
        domains,
        axis=align_axis,
        step=step,
        origin_x=origin_x,
        origin_y=origin_y,
        snap_mode=snap_mode,
    )
    if isinstance(fixed_value, LayoutDiagnostic):
        return fixed_value

    candidates_by_name: dict[str, list[LocalPlacement]] = {}
    rejection_counts: dict[str, int] = {}
    raw_counts: dict[str, int] = {}
    refill_counts: dict[str, int] = {}
    for domain in domains.values():
        candidates, raw_count = _grid_for_domain(
            domain,
            step=step,
            origin_x=origin_x,
            origin_y=origin_y,
            snap_mode=snap_mode,
            align_axis=align_axis,
            fixed_value=fixed_value,
            jitter_ratio=jitter_ratio,
            rng=rng,
            valid_placement=valid_placement,
            max_candidates=max_candidates_per_object,
            yaw_range=yaw_range,
        )
        raw_counts[domain.name] = raw_count
        if len(candidates) < target_min_candidates and grid_snap is None:
            refill, refill_raw = _grid_for_domain(
                domain,
                step=max(0.005, step * 0.5),
                origin_x=origin_x,
                origin_y=origin_y,
                snap_mode=snap_mode,
                align_axis=align_axis,
                fixed_value=fixed_value,
                jitter_ratio=jitter_ratio,
                rng=rng,
                valid_placement=valid_placement,
                max_candidates=max_candidates_per_object,
                yaw_range=yaw_range,
                existing={(round(item.x, 9), round(item.y, 9)) for item in candidates},
            )
            candidates.extend(refill)
            raw_counts[domain.name] += refill_raw
            refill_counts[domain.name] = len(refill)
        if not candidates:
            _inc(rejection_counts, "empty_candidate_domain")
            return LayoutDiagnostic(
                f"{packer}_no_candidate",
                "group domain produced no valid grid candidate",
                target=domain.name,
                meta={
                    "scope_id": scope_id,
                    "packer": packer,
                    "step": step,
                    "raw_candidate_counts": raw_counts,
                    "rejection_counts": rejection_counts,
                },
            )
        rng.shuffle(candidates)
        candidates_by_name[domain.name] = candidates[:max_candidates_per_object]

    return GroupCandidateSet(
        candidates_by_name=candidates_by_name,
        meta={
            "scope_id": scope_id,
            "packer": packer,
            "candidate_generator": "grid",
            "step": step,
            "raw_candidate_counts": raw_counts,
            "candidate_counts": {name: len(cands) for name, cands in candidates_by_name.items()},
            "refill_counts": refill_counts,
            "rejection_counts": rejection_counts,
        },
    )


def scatter_grid_step(
    *,
    domains: dict[str, GroupObjectDomain],
    count: int,
    min_spacing: float,
) -> float:
    if not domains:
        return 0.05
    min_x = min(domain.bounds.x.min for domain in domains.values())
    max_x = max(domain.bounds.x.max for domain in domains.values())
    min_y = min(domain.bounds.y.min for domain in domains.values())
    max_y = max(domain.bounds.y.max for domain in domains.values())
    area = max(1e-6, (max_x - min_x) * (max_y - min_y))
    cell = math.sqrt(area / max(1, int(count)))
    footprint = max(max(domain.footprint) for domain in domains.values())
    return max(0.01, min(0.08, max(cell, footprint + max(0.0, float(min_spacing)))))


def _grid_step(
    *,
    grid_step: float | None,
    grid_snap: object | None,
    domains: dict[str, GroupObjectDomain],
) -> float:
    snap_step = _snap_value(grid_snap, "step")
    if snap_step is not None and snap_step > 0.0:
        return float(snap_step)
    if grid_step is not None and float(grid_step) > 0.0:
        return float(grid_step)
    if not domains:
        return 0.05
    spans = [
        span for domain in domains.values() for span in (domain.bounds.x.length, domain.bounds.y.length) if span > 1e-9
    ]
    footprints = [span for domain in domains.values() for span in domain.footprint if span > 1e-9]
    if not spans or not footprints:
        return 0.01
    min_span = min(spans)
    min_footprint = min(footprints)
    return max(0.01, min(0.05, min(min_span / 8.0, min_footprint / 3.0)))


def _grid_origin(grid_snap: object | None) -> tuple[float, float]:
    origin = _snap_value(grid_snap, "origin")
    if isinstance(origin, (list, tuple)) and len(origin) >= 2:
        return (float(origin[0]), float(origin[1]))
    return (0.0, 0.0)


def _grid_for_domain(
    domain: GroupObjectDomain,
    *,
    step: float,
    origin_x: float,
    origin_y: float,
    snap_mode: str,
    align_axis: str | None,
    fixed_value: float | None,
    jitter_ratio: float,
    rng: random.Random,
    valid_placement: PlacementPredicate,
    max_candidates: int,
    yaw_range: YawRangeInput,
    existing: set[tuple[float, float]] | None = None,
) -> tuple[list[LocalPlacement], int]:
    existing = existing or set()
    xs = (
        [fixed_value]
        if align_axis == "x" and fixed_value is not None
        else _axis_grid(domain.bounds.x, step, origin_x, mode=snap_mode)
    )
    ys = (
        [fixed_value]
        if align_axis == "y" and fixed_value is not None
        else _axis_grid(domain.bounds.y, step, origin_y, mode=snap_mode)
    )
    jitter = max(0.0, float(jitter_ratio)) * step * 0.5
    raw_count = 0
    candidates: list[LocalPlacement] = []
    for x in xs:
        for y in ys:
            raw_count += 1
            xx = float(x)
            yy = float(y)
            if jitter > 0.0:
                if align_axis != "x":
                    xx += rng.uniform(-jitter, jitter)
                if align_axis != "y":
                    yy += rng.uniform(-jitter, jitter)
            if not domain.bounds.contains(xx, yy):
                continue
            key = (round(xx, 9), round(yy, 9))
            if key in existing:
                continue
            item = _local_placement(domain.name, xx, yy, rng=rng, yaw_range=yaw_range)
            if not valid_placement(item):
                continue
            candidates.append(item)
            if len(candidates) >= max_candidates:
                return candidates, raw_count
    return candidates, raw_count


def _local_placement(
    name: str,
    x: float,
    y: float,
    *,
    rng: random.Random,
    yaw_range: YawRangeInput,
) -> LocalPlacement:
    yaw = None
    effective = _yaw_range_for_name(yaw_range, name)
    if effective is not None:
        lo, hi = float(effective[0]), float(effective[1])
        yaw = lo if abs(hi - lo) <= 1e-12 else rng.uniform(lo, hi)
    return LocalPlacement(name=name, x=float(x), y=float(y), yaw=yaw)


def _yaw_range_for_name(yaw_range: YawRangeInput, name: str) -> tuple[float, float] | None:
    if yaw_range is None:
        return None
    if isinstance(yaw_range, dict):
        return yaw_range.get(name)
    return yaw_range


def _axis_grid(axis: Range1D, step: float, origin: float, *, mode: str = "center") -> list[float]:
    if axis.length <= 1e-9:
        return [axis.center]
    step = max(1e-6, float(step))
    offset = 0.5 if mode == "cell_center" else 0.0
    start = math.ceil((axis.min - origin) / step - offset)
    end = math.floor((axis.max - origin) / step - offset)
    values = [origin + (idx + offset) * step for idx in range(start, end + 1)]
    if not values:
        return [axis.center]
    if all(abs(value - axis.center) > 1e-9 for value in values):
        values.append(axis.center)
    return sorted(float(value) for value in values)


def _common_aligned_value(
    domains: dict[str, GroupObjectDomain],
    *,
    axis: str | None,
    step: float,
    origin_x: float,
    origin_y: float,
    snap_mode: str,
) -> float | None | LayoutDiagnostic:
    if axis not in {"x", "y"}:
        return None
    common: Range1D | None = None
    for domain in domains.values():
        item = domain.bounds.x if axis == "x" else domain.bounds.y
        common = item if common is None else common.intersect(item)
        if common is None:
            return LayoutDiagnostic(
                "region_align_axis_no_common_range",
                "aligned region objects do not share a common fixed-axis support range",
                target=axis,
                meta={"axis": axis},
            )
    assert common is not None
    origin = origin_x if axis == "x" else origin_y
    return _snap_axis_center(common, step, origin, mode=snap_mode)


def _snap_axis_center(axis: Range1D, step: float, origin: float, *, mode: str) -> float:
    if step <= 0.0:
        return axis.center
    offset = 0.5 if mode == "cell_center" else 0.0
    raw_idx = (axis.center - origin) / step - offset
    if mode == "floor":
        idx = math.floor(raw_idx)
    elif mode == "ceil":
        idx = math.ceil(raw_idx)
    else:
        idx = round(raw_idx)
    snapped = origin + (idx + offset) * step
    return min(max(snapped, axis.min), axis.max)


def _grid_mode(grid_snap: object | None) -> str:
    raw = _snap_value(grid_snap, "mode")
    if raw is None:
        raw = _snap_value(grid_snap, "mod")
    mode = str(raw or "center").strip().lower()
    if mode in {"nearest", "round"}:
        return "center"
    if mode in {"center", "cell_center", "floor", "ceil", "line", "grid_line"}:
        return mode
    return "center"


def _snap_value(grid_snap: object | None, key: str):
    if grid_snap is None:
        return None
    if isinstance(grid_snap, dict):
        return grid_snap.get(key)
    return getattr(grid_snap, key, None)


def _inc(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key, 0)) + 1


def _poisson_radius(domain: GroupObjectDomain, *, min_spacing: float) -> float:
    footprint_radius = max(0.0, min(domain.footprint) * 0.5)
    return max(0.0, float(min_spacing), footprint_radius)


def _dist2(a: LocalPlacement, b: LocalPlacement) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy
