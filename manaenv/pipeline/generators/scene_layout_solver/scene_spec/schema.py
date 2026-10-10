from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from ..common.diagnostics import error
from ..common.geometry import Bounds2D, Bounds3D, QuatXYZW, Range1D, Vec3, quat_normalize
from ..common.surface_catalog import expand_surface_catalog_refs
from ..common.surface_polygon import Polygon2D
from ..common.volumes import bounds_from_center_size
from ..common.weighted import WeightError, parse_nonnegative_weight
from .authoring import materialize_layout_payload


def _list_str(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return [str(values)]


def _range_pair(values: object, *, target: str) -> tuple[float, float]:
    try:
        raw = Range1D.from_pair(values)
    except Exception as exc:  # noqa: BLE001
        raise error("range_pair_invalid", "expected a scalar or [min, max] range", target=target) from exc
    return (float(raw.min), float(raw.max))


def _vec3(values: object, *, target: str) -> Vec3:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise error("invalid_vec3", "expected a 3-value list", target=target)
    return (float(values[0]), float(values[1]), float(values[2]))


def _quat(values: object | None) -> QuatXYZW | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise error("invalid_quaternion", "expected XYZW quaternion")
    return quat_normalize(values)


@dataclass(frozen=True)
class SurfaceRangeSpec:
    x: Range1D | None = None
    y: Range1D | None = None
    policy: str = "strict"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SurfaceRangeSpec | None:
        if data is None:
            return None
        if not isinstance(data, dict):
            raise error("surface_range_invalid", "surface_range must be a dict")
        policy = str(data.get("policy") or "strict").strip().lower()
        if policy not in {"strict", "expandable", "surface"}:
            raise error("surface_range_policy_invalid", "unsupported surface_range policy", policy=policy)
        return cls(
            x=Range1D.from_pair(data["x"]) if "x" in data else None,
            y=Range1D.from_pair(data["y"]) if "y" in data else None,
            policy=policy,
        )

    def to_bounds(self, fallback: Bounds2D) -> Bounds2D:
        return Bounds2D(self.x or fallback.x, self.y or fallback.y)

    def relaxed(self, *, tier: str, expand_m: float) -> SurfaceRangeSpec | None:
        normalized = str(tier or "strict").strip().lower()
        if normalized in {"", "strict"} or self.policy == "strict":
            return self
        if normalized == "surface" and self.policy == "surface":
            return None
        if normalized in {"expand", "surface"} and self.policy in {"expandable", "surface"}:
            pad = max(0.0, float(expand_m))
            if pad <= 1e-12:
                return self
            return SurfaceRangeSpec(
                x=self.x.expanded(pad) if self.x is not None else None,
                y=self.y.expanded(pad) if self.y is not None else None,
                policy=self.policy,
            )
        return self


@dataclass(frozen=True)
class SurfaceRangeChoiceSpec:
    range_id: str
    surface_range: SurfaceRangeSpec
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, index: int, target: str) -> SurfaceRangeChoiceSpec:
        if not isinstance(data, dict):
            raise error("surface_range_choice_invalid", "surface_ranges entries must be mappings", target=target)
        range_id = str(data.get("id") or data.get("range_id") or f"range_{index:03d}").strip()
        if not range_id:
            raise error("surface_range_choice_id_missing", "surface_range choice requires id", target=target)
        raw_range = data.get("surface_range")
        if raw_range is not None:
            if not isinstance(raw_range, dict):
                raise error(
                    "surface_range_choice_invalid",
                    "surface_range choice surface_range must be a mapping",
                    target=f"{target}:{range_id}",
                )
            range_payload = dict(raw_range)
        else:
            range_payload = {key: data[key] for key in ("x", "y", "policy") if key in data}
        surface_range = SurfaceRangeSpec.from_dict(range_payload)
        if surface_range is None:
            raise error("surface_range_choice_invalid", "surface_range choice must define a range", target=target)
        try:
            weight = parse_nonnegative_weight(data.get("weight"), default=1.0)
        except WeightError as exc:
            raise error(
                "surface_range_choice_weight_invalid",
                "surface_range choice weight must be a finite non-negative number",
                target=f"{target}:{range_id}",
                weight=data.get("weight"),
            ) from exc
        return cls(range_id=range_id, surface_range=surface_range, weight=weight)


def _surface_range_choices(data: dict[str, Any], *, target: str) -> tuple[SurfaceRangeChoiceSpec, ...]:
    if data.get("surface_ranges") is None:
        return ()
    if data.get("surface_range") is not None:
        raise error(
            "surface_range_choice_conflict",
            "define either surface_range or surface_ranges, not both",
            target=target,
        )
    raw = data.get("surface_ranges")
    if not isinstance(raw, list):
        raise error("surface_ranges_invalid", "surface_ranges must be a list", target=target)
    choices = tuple(
        SurfaceRangeChoiceSpec.from_dict(item, index=index, target=target)
        for index, item in enumerate(raw)
        if isinstance(item, dict)
    )
    if len(choices) != len(raw):
        raise error("surface_range_choice_invalid", "surface_ranges entries must be mappings", target=target)
    if not choices:
        raise error("surface_ranges_empty", "surface_ranges must contain at least one choice", target=target)
    ids = [choice.range_id for choice in choices]
    duplicates = sorted({range_id for range_id in ids if ids.count(range_id) > 1})
    if duplicates:
        raise error(
            "surface_range_choice_duplicate",
            "surface_range choice ids must be unique",
            target=target,
            ids=duplicates,
        )
    if not any(choice.weight > 0.0 for choice in choices):
        raise error(
            "surface_range_choice_all_zero",
            "surface_ranges must contain at least one choice with weight > 0",
            target=target,
        )
    return choices


@dataclass(frozen=True)
class YawChoiceSpec:
    yaw_id: str
    yaw: float
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, index: int, target: str) -> YawChoiceSpec:
        if not isinstance(data, dict):
            raise error("yaw_sample_invalid", "yaw_samples entries must be mappings", target=target)
        yaw_id = str(data.get("id") or data.get("yaw_id") or f"yaw_{index:03d}").strip()
        if not yaw_id:
            raise error("yaw_sample_id_missing", "yaw_sample choice requires id", target=target)
        if data.get("yaw") is None:
            raise error("yaw_sample_yaw_missing", "yaw_sample choice requires yaw", target=f"{target}:{yaw_id}")
        try:
            yaw = float(data.get("yaw"))
        except (TypeError, ValueError) as exc:
            raise error(
                "yaw_sample_yaw_invalid",
                "yaw_sample yaw must be a finite number",
                target=f"{target}:{yaw_id}",
                yaw=data.get("yaw"),
            ) from exc
        if not isfinite(yaw):
            raise error(
                "yaw_sample_yaw_invalid",
                "yaw_sample yaw must be a finite number",
                target=f"{target}:{yaw_id}",
                yaw=data.get("yaw"),
            )
        try:
            weight = parse_nonnegative_weight(data.get("weight"), default=1.0)
        except WeightError as exc:
            raise error(
                "yaw_sample_weight_invalid",
                "yaw_sample weight must be a finite non-negative number",
                target=f"{target}:{yaw_id}",
                weight=data.get("weight"),
            ) from exc
        return cls(yaw_id=yaw_id, yaw=yaw, weight=weight)


def _yaw_choices(data: dict[str, Any], *, target: str) -> tuple[YawChoiceSpec, ...]:
    raw = data.get("yaw_samples", data.get("yaw_choices"))
    if raw is None:
        return ()
    if data.get("yaw") is not None:
        raise error("yaw_sample_conflict", "define either yaw or yaw_samples, not both", target=target)
    if not isinstance(raw, list):
        raise error("yaw_samples_invalid", "yaw_samples must be a list", target=target)
    choices = tuple(
        YawChoiceSpec.from_dict(item, index=index, target=target)
        for index, item in enumerate(raw)
        if isinstance(item, dict)
    )
    if len(choices) != len(raw):
        raise error("yaw_sample_invalid", "yaw_samples entries must be mappings", target=target)
    if not choices:
        raise error("yaw_samples_empty", "yaw_samples must contain at least one choice", target=target)
    ids = [choice.yaw_id for choice in choices]
    duplicates = sorted({yaw_id for yaw_id in ids if ids.count(yaw_id) > 1})
    if duplicates:
        raise error("yaw_sample_duplicate", "yaw_sample choice ids must be unique", target=target, ids=duplicates)
    if not any(choice.weight > 0.0 for choice in choices):
        raise error(
            "yaw_sample_all_zero",
            "yaw_samples must contain at least one choice with weight > 0",
            target=target,
        )
    return choices


@dataclass(frozen=True)
class ManualSupportSurfaceSpec:
    surface_id: str
    frame: str
    center: Vec3
    x_axis: Vec3
    y_axis: Vec3
    normal: Vec3 | None
    bounds: Bounds2D
    shape: str = "rectangle"
    polygon: Polygon2D | None = None
    margin: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManualSupportSurfaceSpec:
        surface_id = str(data.get("id") or "").strip()
        if not surface_id:
            raise error("surface_id_missing", "support surface id is required")
        frame = str(data.get("frame") or "world").strip().lower()
        if frame != "world":
            raise error("surface_frame_unsupported", "V1 only supports world-frame manual surfaces", target=surface_id)
        shape = str(data.get("shape") or "rectangle").strip().lower()
        polygon = None
        if shape == "rectangle":
            if "bounds" in data:
                bounds = Bounds2D.from_dict(data["bounds"])
            elif "size" in data:
                bounds = Bounds2D.from_size(data["size"])
            else:
                raise error("surface_bounds_missing", "manual surface requires bounds or size", target=surface_id)
        elif shape == "polygon":
            try:
                polygon = Polygon2D.from_payload(data.get("outer"), data.get("holes"))
            except ValueError as exc:
                raise error("surface_polygon_invalid", str(exc), target=surface_id) from exc
            min_x, max_x, min_y, max_y = polygon.bounds
            bounds = Bounds2D(Range1D(min_x, max_x), Range1D(min_y, max_y))
        else:
            raise error(
                "surface_shape_unsupported", "unsupported support surface shape", target=surface_id, shape=shape
            )
        return cls(
            surface_id=surface_id,
            frame=frame,
            center=_vec3(data.get("center"), target=f"surface:{surface_id}"),
            x_axis=_vec3(data.get("x_axis"), target=f"surface:{surface_id}:x_axis"),
            y_axis=_vec3(data.get("y_axis"), target=f"surface:{surface_id}:y_axis"),
            normal=(
                _vec3(data["normal"], target=f"surface:{surface_id}:normal") if data.get("normal") is not None else None
            ),
            bounds=bounds,
            shape=shape,
            polygon=polygon,
            margin=float(data.get("margin", 0.0) or 0.0),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(frozen=True)
class LocalSupportSurfaceSpec:
    surface_id: str
    parent: str
    center: Vec3
    x_axis: Vec3
    y_axis: Vec3
    normal: Vec3 | None
    bounds: Bounds2D
    shape: str = "rectangle"
    polygon: Polygon2D | None = None
    margin: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, parent: str, name: str, data: dict[str, Any]) -> LocalSupportSurfaceSpec:
        surface_name = str(data.get("id") or name).strip()
        if not surface_name:
            raise error("local_surface_id_missing", "local support surface requires id", target=parent)
        surface_id = surface_name if "." in surface_name else f"{parent}.{surface_name}"
        shape = str(data.get("shape") or "rectangle").strip().lower()
        polygon = None
        if shape == "rectangle":
            if "bounds" in data:
                bounds = Bounds2D.from_dict(data["bounds"])
            elif "size" in data:
                bounds = Bounds2D.from_size(data["size"])
            else:
                raise error(
                    "local_surface_bounds_missing", "local support surface requires bounds or size", target=surface_id
                )
        elif shape == "polygon":
            try:
                polygon = Polygon2D.from_payload(data.get("outer"), data.get("holes"))
            except ValueError as exc:
                raise error("local_surface_polygon_invalid", str(exc), target=surface_id) from exc
            min_x, max_x, min_y, max_y = polygon.bounds
            bounds = Bounds2D(Range1D(min_x, max_x), Range1D(min_y, max_y))
        else:
            raise error(
                "local_surface_shape_unsupported",
                "unsupported local support surface shape",
                target=surface_id,
                shape=shape,
            )
        return cls(
            surface_id=surface_id,
            parent=str(parent),
            center=_vec3(data.get("center"), target=f"local_surface:{surface_id}"),
            x_axis=_vec3(data.get("x_axis", [1.0, 0.0, 0.0]), target=f"local_surface:{surface_id}:x_axis"),
            y_axis=_vec3(data.get("y_axis", [0.0, 1.0, 0.0]), target=f"local_surface:{surface_id}:y_axis"),
            normal=(
                _vec3(data["normal"], target=f"local_surface:{surface_id}:normal")
                if data.get("normal") is not None
                else None
            ),
            bounds=bounds,
            shape=shape,
            polygon=polygon,
            margin=float(data.get("margin", 0.0) or 0.0),
            provenance=dict(data.get("provenance") or {}),
        )


def _local_support_surfaces(parent: str, raw: object) -> tuple[LocalSupportSurfaceSpec, ...]:
    if raw is None:
        return ()
    surfaces: list[LocalSupportSurfaceSpec] = []
    if isinstance(raw, dict):
        for name, payload in raw.items():
            if not isinstance(payload, dict):
                raise error(
                    "local_surface_invalid", "local support surface must be a mapping", target=f"{parent}.{name}"
                )
            surfaces.append(LocalSupportSurfaceSpec.from_dict(parent, str(name), payload))
        return tuple(surfaces)
    if isinstance(raw, list):
        for index, payload in enumerate(raw):
            if not isinstance(payload, dict):
                raise error(
                    "local_surface_invalid", "local support surface must be a mapping", target=f"{parent}:{index}"
                )
            surfaces.append(LocalSupportSurfaceSpec.from_dict(parent, str(index), payload))
        return tuple(surfaces)
    raise error("local_surfaces_invalid", "support_surfaces must be a mapping or list", target=parent)


@dataclass(frozen=True)
class LayoutObjectSpec:
    name: str
    bind: str = "asset_name"
    support: str | None = None
    surface_range: SurfaceRangeSpec | None = None
    surface_ranges: tuple[SurfaceRangeChoiceSpec, ...] = ()
    geometry: dict[str, Any] = field(default_factory=dict)
    scale: Any = None
    orientation: QuatXYZW | None = None
    yaw: tuple[float, float] | None = None
    yaw_samples: tuple[YawChoiceSpec, ...] = ()
    tags: list[str] = field(default_factory=list)
    support_surfaces: tuple[LocalSupportSurfaceSpec, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayoutObjectSpec:
        name = str(data.get("name") or "").strip()
        if not name:
            raise error("layout_object_name_missing", "layout object requires name")
        return cls(
            name=name,
            bind=str(data.get("bind") or "asset_name"),
            support=str(data["support"]).strip() if data.get("support") else None,
            surface_range=SurfaceRangeSpec.from_dict(data.get("surface_range")),
            surface_ranges=_surface_range_choices(data, target=f"layout_object:{name}"),
            geometry=dict(data.get("geometry") or {}),
            scale=data.get("scale"),
            orientation=_quat(data.get("orientation")),
            yaw=(
                _range_pair(data.get("yaw"), target=f"layout_object:{name}:yaw")
                if data.get("yaw") is not None
                else None
            ),
            yaw_samples=_yaw_choices(data, target=f"layout_object:{name}"),
            tags=_list_str(data.get("tags")),
            support_surfaces=_local_support_surfaces(name, data.get("support_surfaces", data.get("local_surfaces"))),
        )


@dataclass(frozen=True)
class SlotSpec:
    slot_id: str
    surface: str
    surface_range: SurfaceRangeSpec | None = None
    yaw: tuple[float, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlotSpec:
        slot_id = str(data.get("id") or "").strip()
        if not slot_id:
            raise error("slot_id_missing", "slot requires id")
        surface = str(data.get("surface") or "").strip()
        if not surface:
            raise error("slot_surface_missing", "slot requires surface", target=slot_id)
        return cls(
            slot_id=slot_id,
            surface=surface,
            surface_range=SurfaceRangeSpec.from_dict(data.get("surface_range")),
            yaw=(_range_pair(data.get("yaw"), target=f"slot:{slot_id}:yaw") if data.get("yaw") is not None else None),
        )


@dataclass(frozen=True)
class RegionSampleSpec:
    region_id: str
    surface: str
    objects: list[str]
    surface_range: SurfaceRangeSpec | None = None
    surface_ranges: tuple[SurfaceRangeChoiceSpec, ...] = ()
    sample_method: str = "random"
    grid_step: float | None = None
    grid_snap: dict[str, Any] | None = None
    min_spacing: float = 0.0
    spacing_mode: str = "footprint_gap"
    align_axis: str | None = None
    order_axis: str | None = None
    jitter: float = 0.0
    yaw: tuple[float, float] | None = None
    max_attempts: int = 200

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, index: int) -> RegionSampleSpec:
        region_id = str(data.get("id") or f"region_{index}")
        surface = str(data.get("surface") or "").strip()
        if not surface:
            raise error("region_surface_missing", "region_sample requires surface", target=region_id)
        objects = _list_str(data.get("objects"))
        if not objects:
            raise error("region_objects_missing", "region_sample requires objects", target=region_id)
        align_axis = str(data.get("align_axis")).lower() if data.get("align_axis") else None
        if align_axis is not None and align_axis not in {"x", "y"}:
            raise error("region_align_axis_invalid", "align_axis must be x or y", target=region_id)
        order_axis = str(data.get("order_axis")).lower() if data.get("order_axis") else None
        if order_axis is not None and order_axis not in {"x", "y"}:
            raise error("region_order_axis_invalid", "order_axis must be x or y", target=region_id)
        sample_method = str(data.get("sample_method") or "random").strip().lower()
        if sample_method not in {"random", "grid", "jittered_grid", "stratified", "poisson", "poisson_disk"}:
            raise error(
                "region_sample_method_unsupported",
                "unsupported region_sample method",
                target=region_id,
                method=sample_method,
            )
        grid_step = float(data["grid_step"]) if data.get("grid_step") is not None else None
        grid_snap = dict(data["grid_snap"]) if isinstance(data.get("grid_snap"), dict) else None
        if grid_snap is not None:
            raw_mode = grid_snap.get("mode", grid_snap.get("mod", "center"))
            mode = str(raw_mode or "center").strip().lower()
            supported_modes = {"center", "cell_center", "nearest", "round", "floor", "ceil", "line", "grid_line"}
            if mode not in supported_modes:
                raise error(
                    "region_grid_snap_mode_unsupported", "unsupported grid_snap mode", target=region_id, mode=mode
                )
        jitter = data.get("jitter", data.get("jitter_ratio", 0.0))
        spacing_mode = str(data.get("spacing_mode") or data.get("min_spacing_mode") or "footprint_gap").strip().lower()
        if spacing_mode in {"center", "center_distance", "center-distance"}:
            spacing_mode = "center_distance"
        elif spacing_mode in {"footprint", "footprint_gap", "footprint-gap"}:
            spacing_mode = "footprint_gap"
        else:
            raise error(
                "region_spacing_mode_unsupported",
                "unsupported region_sample spacing_mode",
                target=region_id,
                spacing_mode=spacing_mode,
            )
        return cls(
            region_id=region_id,
            surface=surface,
            objects=objects,
            surface_range=SurfaceRangeSpec.from_dict(data.get("surface_range")),
            surface_ranges=_surface_range_choices(data, target=f"region_sample:{region_id}"),
            sample_method=sample_method,
            grid_step=grid_step,
            grid_snap=grid_snap,
            min_spacing=float(data.get("min_spacing", 0.0) or 0.0),
            spacing_mode=spacing_mode,
            align_axis=align_axis,
            order_axis=order_axis,
            jitter=float(jitter or 0.0),
            yaw=(
                _range_pair(data.get("yaw"), target=f"region_sample:{region_id}:yaw")
                if data.get("yaw") is not None
                else None
            ),
            max_attempts=max(1, int(data.get("max_attempts", 200) or 200)),
        )


@dataclass(frozen=True)
class ScatterGroupSpec:
    group_id: str
    surface: str
    objects: list[str]
    surface_range: SurfaceRangeSpec | None = None
    surface_ranges: tuple[SurfaceRangeChoiceSpec, ...] = ()
    mode: str = "jittered_grid"
    min_spacing: float = 0.0
    spacing_mode: str = "footprint_gap"
    jitter: float = 0.25
    yaw: tuple[float, float] | None = None
    seed: int | None = None
    max_attempts: int = 500

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, index: int) -> ScatterGroupSpec:
        group_id = str(data.get("id") or data.get("group_id") or f"scatter_{index}")
        surface = str(data.get("surface") or "").strip()
        if not surface:
            raise error("scatter_group_surface_missing", "scatter_group requires surface", target=group_id)
        objects = _list_str(data.get("objects"))
        if not objects:
            raise error(
                "scatter_group_objects_missing", "scatter_group requires explicit objects in V1", target=group_id
            )
        mode = str(data.get("mode") or "jittered_grid").strip().lower()
        if mode not in {"grid", "jittered_grid", "poisson_disk"}:
            raise error("scatter_group_mode_unsupported", "unsupported scatter_group mode", target=group_id, mode=mode)
        spacing_mode = str(data.get("spacing_mode") or data.get("min_spacing_mode") or "footprint_gap").strip().lower()
        if spacing_mode in {"center", "center_distance", "center-distance"}:
            spacing_mode = "center_distance"
        elif spacing_mode in {"footprint", "footprint_gap", "footprint-gap"}:
            spacing_mode = "footprint_gap"
        else:
            raise error(
                "scatter_group_spacing_mode_unsupported",
                "unsupported scatter_group spacing_mode",
                target=group_id,
                spacing_mode=spacing_mode,
            )
        return cls(
            group_id=group_id,
            surface=surface,
            objects=objects,
            surface_range=SurfaceRangeSpec.from_dict(data.get("surface_range")),
            surface_ranges=_surface_range_choices(data, target=f"scatter_group:{group_id}"),
            mode=mode,
            min_spacing=float(data.get("min_spacing", 0.0) or 0.0),
            spacing_mode=spacing_mode,
            jitter=float(data.get("jitter", 0.25) or 0.0),
            yaw=(
                _range_pair(data.get("yaw"), target=f"scatter_group:{group_id}:yaw")
                if data.get("yaw") is not None
                else None
            ),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            max_attempts=max(1, int(data.get("max_attempts", 500) or 500)),
        )


@dataclass(frozen=True)
class RelativeDirectionSpec:
    node_1: str
    node_2: str
    direction: str
    margin: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelativeDirectionSpec:
        node_1 = str(data.get("node_1") or data.get("a") or "").strip()
        node_2 = str(data.get("node_2") or data.get("b") or "").strip()
        direction = str(data.get("direction") or "").strip().lower()
        if not node_1 or not node_2:
            raise error("relative_direction_node_missing", "relative_direction requires node_1 and node_2")
        if direction not in {"left_of", "right_of", "front_of", "back_of", "left", "right", "front", "back"}:
            raise error("relative_direction_invalid", "unsupported relative direction", direction=direction)
        aliases = {"left": "left_of", "right": "right_of", "front": "front_of", "back": "back_of"}
        return cls(
            node_1=node_1,
            node_2=node_2,
            direction=aliases.get(direction, direction),
            margin=float(data.get("margin", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class PoseCandidateSpec:
    pose_id: str
    position: Vec3
    quaternion: QuatXYZW
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, index: int) -> PoseCandidateSpec:
        pose_id = str(data.get("id") or data.get("pose_id") or f"pose_{index:03d}")
        try:
            weight = parse_nonnegative_weight(data.get("weight"), default=1.0)
        except WeightError as exc:
            raise error(
                "pose_candidate_weight_invalid",
                "pose candidate weight must be a finite non-negative number",
                target=f"pose_candidate:{pose_id}",
                weight=data.get("weight"),
            ) from exc
        return cls(
            pose_id=pose_id,
            position=_vec3(data.get("position"), target=f"pose_candidate:{pose_id}"),
            quaternion=_quat(data.get("quaternion")) or (0.0, 0.0, 0.0, 1.0),
            weight=weight,
        )


@dataclass(frozen=True)
class PoseSampleSpec:
    object_name: str
    poses: list[PoseCandidateSpec]
    shuffle: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, index: int) -> PoseSampleSpec:
        object_name = str(data.get("object") or data.get("name") or data.get("node") or "").strip()
        if not object_name:
            raise error("pose_sample_object_missing", "pose_sample requires object", target=f"pose_sample_{index}")
        poses = [
            PoseCandidateSpec.from_dict(item, index=i)
            for i, item in enumerate(data.get("poses", data.get("candidates", [])) or [])
            if isinstance(item, dict)
        ]
        if not poses:
            raise error("pose_sample_candidates_missing", "pose_sample requires candidate poses", target=object_name)
        if not any(pose.weight > 0.0 for pose in poses):
            raise error(
                "pose_sample_candidates_all_zero",
                "pose_sample requires at least one candidate with weight > 0",
                target=object_name,
            )
        return cls(object_name=object_name, poses=poses, shuffle=bool(data.get("shuffle", False)))


@dataclass(frozen=True)
class VolumeSpec:
    volume_id: str
    frame: str
    bounds: Bounds3D
    shape: str = "box"
    center: Vec3 = (0.0, 0.0, 0.0)
    radius: float | None = None
    parent: str | None = None
    margin: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VolumeSpec:
        raw_id = str(data.get("id") or "").strip()
        if not raw_id:
            raise error("volume_id_missing", "volume requires id")
        frame, parent = _volume_frame(data.get("frame"), target=raw_id)
        volume_id = raw_id if parent is None or "." in raw_id else f"{parent}.{raw_id}"
        shape = str(data.get("shape") or "box").strip().lower()
        if shape not in {"box", "cylinder"}:
            raise error("volume_shape_unsupported", "unsupported volume shape", target=volume_id, shape=shape)
        center = _vec3(data.get("center", [0.0, 0.0, 0.0]), target=f"volume:{volume_id}:center")
        if shape == "box":
            bounds = _box_volume_bounds(data, volume_id=volume_id, center=center)
            radius = None
        else:
            radius = _volume_radius(data, target=volume_id)
            z = _volume_z_range(data, target=volume_id, center=center)
            bounds = Bounds3D(
                Range1D(center[0] - radius, center[0] + radius),
                Range1D(center[1] - radius, center[1] + radius),
                z,
            )
        return cls(
            volume_id=volume_id,
            frame=frame,
            bounds=bounds,
            shape=shape,
            center=center,
            radius=radius,
            parent=parent,
            margin=float(data.get("margin", 0.0) or 0.0),
            provenance=dict(data.get("provenance") or {}),
        )


def _volume_frame(raw: object, *, target: str) -> tuple[str, str | None]:
    if raw is None:
        return ("world", None)
    if isinstance(raw, str):
        frame = raw.strip().lower()
        if frame != "world":
            raise error("volume_frame_unsupported", "volume frame must be world or object", target=target, frame=frame)
        return ("world", None)
    if isinstance(raw, dict):
        frame = str(raw.get("type") or raw.get("frame") or "").strip().lower()
        if frame in {"object", "node", "object-local", "node-local"}:
            parent = str(raw.get("node") or raw.get("parent") or raw.get("object") or "").strip()
            if not parent:
                raise error(
                    "volume_parent_missing", "object-local volume requires frame.node or frame.parent", target=target
                )
            return ("object", parent)
        if frame == "world":
            return ("world", None)
    raise error("volume_frame_unsupported", "volume frame must be world or object", target=target)


def _box_volume_bounds(data: dict[str, Any], *, volume_id: str, center: Vec3) -> Bounds3D:
    if "bounds" in data:
        return Bounds3D.from_dict(data["bounds"])
    if "size" in data:
        return bounds_from_center_size(center, data["size"])
    if all(axis in data for axis in ("x", "y", "z")):
        return Bounds3D(Range1D.from_pair(data["x"]), Range1D.from_pair(data["y"]), Range1D.from_pair(data["z"]))
    raise error("volume_bounds_missing", "box volume requires bounds, size, or x/y/z ranges", target=volume_id)


def _volume_radius(data: dict[str, Any], *, target: str) -> float:
    raw = data.get("radius", data.get("inner_radius"))
    if raw is None:
        diameter = data.get("diameter", data.get("inner_diameter"))
        if diameter is not None:
            raw = float(diameter) * 0.5
    if raw is None:
        raise error("volume_radius_missing", "cylinder volume requires radius or diameter", target=target)
    radius = float(raw)
    if radius <= 0.0:
        raise error("volume_radius_invalid", "cylinder volume radius must be positive", target=target, radius=radius)
    return radius


def _volume_z_range(data: dict[str, Any], *, target: str, center: Vec3) -> Range1D:
    if "z" in data:
        return Range1D.from_pair(data["z"])
    if "height" in data:
        h = float(data["height"])
        if h <= 0.0:
            raise error("volume_height_invalid", "cylinder volume height must be positive", target=target, height=h)
        return Range1D(center[2] - h * 0.5, center[2] + h * 0.5)
    if "bounds" in data:
        return Bounds3D.from_dict(data["bounds"]).z
    raise error("volume_z_missing", "cylinder volume requires z range or height", target=target)


@dataclass(frozen=True)
class InVolumeSpec:
    object_name: str
    volume: str
    mode: str = "fit"

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, index: int) -> InVolumeSpec:
        object_name = str(data.get("object") or data.get("name") or data.get("node") or "").strip()
        if not object_name:
            raise error("in_volume_object_missing", "in_volume requires object", target=f"in_volume_{index}")
        volume = str(data.get("volume") or data.get("anchor") or "").strip()
        if not volume:
            raise error("in_volume_anchor_missing", "in_volume requires volume", target=object_name)
        mode = str(data.get("mode") or "fit").strip().lower()
        if mode not in {"fit", "center"}:
            raise error("in_volume_mode_unsupported", "unsupported in_volume mode", target=object_name, mode=mode)
        return cls(object_name=object_name, volume=volume, mode=mode)


@dataclass(frozen=True)
class VolumeGroupSpec:
    group_id: str
    volume: str
    objects: list[str]
    mode: str = "bulk_random"
    min_spacing: float = 0.0
    yaw: tuple[float, float] | None = None
    ignore_parent_collision: bool = True
    ignore_intra_group_collision: bool = True
    max_attempts: int = 500

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, index: int) -> VolumeGroupSpec:
        group_id = str(data.get("id") or data.get("group_id") or f"volume_group_{index}").strip()
        volume = str(data.get("volume") or data.get("anchor") or "").strip()
        if not volume:
            raise error("volume_group_anchor_missing", "volume_group requires volume", target=group_id)
        objects = _list_str(data.get("objects"))
        if not objects:
            raise error("volume_group_objects_missing", "volume_group requires objects", target=group_id)
        mode = str(data.get("mode") or "bulk_random").strip().lower()
        if mode not in {"bulk_random", "random", "grid", "jittered_grid"}:
            raise error("volume_group_mode_unsupported", "unsupported volume_group mode", target=group_id, mode=mode)
        return cls(
            group_id=group_id,
            volume=volume,
            objects=objects,
            mode=mode,
            min_spacing=float(data.get("min_spacing", 0.0) or 0.0),
            yaw=(
                _range_pair(data.get("yaw"), target=f"volume_group:{group_id}:yaw")
                if data.get("yaw") is not None
                else None
            ),
            ignore_parent_collision=bool(data.get("ignore_parent_collision", True)),
            ignore_intra_group_collision=bool(data.get("ignore_intra_group_collision", True)),
            max_attempts=max(1, int(data.get("max_attempts", 500) or 500)),
        )


def _variant_slots_from_payload(payload: object) -> list[SlotSpec]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [SlotSpec.from_dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        generator = str(payload.get("generator") or payload.get("method") or "").strip().lower()
        if generator == "grid":
            return _grid_variant_slots(payload)
        if generator == "fixed":
            return _fixed_variant_slots(payload)
        raise error(
            "variant_slots_generator_unsupported", "unsupported variant slots generator", generator=generator or None
        )
    raise error("variant_slots_invalid", "variant slots must be a list or generator dict")


def _fixed_variant_slots(payload: dict[str, Any]) -> list[SlotSpec]:
    surface = _variant_slot_surface(payload)
    positions = payload.get("positions")
    if not isinstance(positions, list) or not positions:
        raise error("variant_slots_fixed_positions_missing", "fixed slots generator requires positions")
    count = payload.get("count")
    if count is not None and int(count) != len(positions):
        raise error("variant_slots_count_mismatch", "fixed slot count must match positions length")
    base_id = str(payload.get("id") or "fixed_slots").strip()
    out: list[SlotSpec] = []
    for idx, item in enumerate(positions):
        if not isinstance(item, dict):
            raise error(
                "variant_slot_position_invalid", "fixed slot position must be a dict", target=f"{base_id}:{idx}"
            )
        slot_id = str(item.get("id") or f"{base_id}__{idx:03d}").strip()
        x = item.get("x")
        y = item.get("y")
        if x is None or y is None:
            raise error("variant_slot_position_xy_missing", "fixed slot position requires x and y", target=slot_id)
        out.append(
            SlotSpec(
                slot_id=slot_id,
                surface=str(item.get("surface") or surface),
                surface_range=_slot_surface_range(float(x), float(y), payload.get("slot_size")),
            )
        )
    return out


def _grid_variant_slots(payload: dict[str, Any]) -> list[SlotSpec]:
    surface = _variant_slot_surface(payload)
    count = int(payload.get("count", 0) or 0)
    if count <= 0:
        raise error("variant_slots_count_invalid", "grid slots generator requires positive count")
    surface_range = SurfaceRangeSpec.from_dict(payload.get("surface_range"))
    if surface_range is None or surface_range.x is None or surface_range.y is None:
        raise error("variant_slots_surface_range_missing", "grid slots generator requires x/y surface_range")
    x_fixed = surface_range.x.length <= 1e-9
    y_fixed = surface_range.y.length <= 1e-9
    if x_fixed and y_fixed and count != 1:
        raise error("variant_slots_grid_fixed_count_invalid", "fixed grid slot point requires count=1")
    if not x_fixed and not y_fixed:
        raise error(
            "variant_slots_2d_grid_unsupported",
            "grid slots generator currently supports one fixed axis and one ranged axis",
        )

    padding = max(0.0, float(payload.get("padding", 0.0) or 0.0))
    include_endpoints = _truthy(payload.get("include_endpoints", True))
    base_id = str(payload.get("id") or "grid_slots").strip()
    if x_fixed and y_fixed:
        centers = [(surface_range.x.center, surface_range.y.center)]
    elif x_fixed:
        values = _slot_axis_values(surface_range.y, count=count, padding=padding, include_endpoints=include_endpoints)
        centers = [(surface_range.x.center, value) for value in values]
    else:
        values = _slot_axis_values(surface_range.x, count=count, padding=padding, include_endpoints=include_endpoints)
        centers = [(value, surface_range.y.center) for value in values]
    return [
        SlotSpec(
            slot_id=f"{base_id}__{idx:03d}",
            surface=surface,
            surface_range=_slot_surface_range(x, y, payload.get("slot_size")),
        )
        for idx, (x, y) in enumerate(centers)
    ]


def _variant_slot_surface(payload: dict[str, Any]) -> str:
    surface = str(payload.get("surface") or "").strip()
    if not surface:
        raise error("variant_slot_surface_missing", "generated variant slots require surface")
    return surface


def _slot_axis_values(axis: Range1D, *, count: int, padding: float, include_endpoints: bool) -> list[float]:
    lo = axis.min + padding
    hi = axis.max - padding
    if lo > hi:
        raise error("variant_slots_padding_exhausted", "slot padding leaves empty range")
    if count == 1:
        return [(lo + hi) * 0.5]
    if include_endpoints:
        step = (hi - lo) / float(count - 1)
        return [lo + step * idx for idx in range(count)]
    step = (hi - lo) / float(count + 1)
    return [lo + step * (idx + 1) for idx in range(count)]


def _slot_surface_range(x: float, y: float, slot_size: object | None) -> SurfaceRangeSpec:
    if slot_size is None:
        return SurfaceRangeSpec(x=Range1D(float(x), float(x)), y=Range1D(float(y), float(y)))
    if not isinstance(slot_size, (list, tuple)) or len(slot_size) != 2:
        raise error("variant_slot_size_invalid", "slot_size must be [sx, sy]")
    sx = float(slot_size[0])
    sy = float(slot_size[1])
    if sx < 0.0 or sy < 0.0:
        raise error("variant_slot_size_invalid", "slot_size must be non-negative")
    return SurfaceRangeSpec(
        x=Range1D(float(x) - sx * 0.5, float(x) + sx * 0.5),
        y=Range1D(float(y) - sy * 0.5, float(y) + sy * 0.5),
    )


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


@dataclass(frozen=True)
class VariantSpec:
    type: str = "fixed_cases"
    objects: list[str] = field(default_factory=list)
    slots: list[SlotSpec] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)
    nums: int | None = None
    seed: int | None = None
    axes: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> VariantSpec:
        if not data:
            return cls()
        vtype = str(data.get("type") or "fixed_cases").strip().lower()
        slots = _variant_slots_from_payload(data.get("slots"))
        return cls(
            type=vtype,
            objects=_list_str(data.get("objects")),
            slots=slots,
            cases=[dict(item) for item in data.get("cases", []) if isinstance(item, dict)],
            nums=int(data["nums"]) if data.get("nums") is not None else None,
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            axes=dict(data.get("axes") or {}),
        )


@dataclass(frozen=True)
class SceneLayoutSpec:
    enabled: bool
    seed: int
    surfaces: list[ManualSupportSurfaceSpec]
    objects: list[LayoutObjectSpec]
    regions: list[RegionSampleSpec] = field(default_factory=list)
    scatter_groups: list[ScatterGroupSpec] = field(default_factory=list)
    relative_directions: list[RelativeDirectionSpec] = field(default_factory=list)
    pose_samples: list[PoseSampleSpec] = field(default_factory=list)
    volumes: list[VolumeSpec] = field(default_factory=list)
    in_volumes: list[InVolumeSpec] = field(default_factory=list)
    volume_groups: list[VolumeGroupSpec] = field(default_factory=list)
    variants: VariantSpec = field(default_factory=VariantSpec)
    recovery: dict[str, Any] = field(default_factory=dict)
    collision: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SceneLayoutSpec:
        payload = expand_surface_catalog_refs(materialize_layout_payload(dict(data or {})))
        surfaces = [
            ManualSupportSurfaceSpec.from_dict(item) for item in payload.get("surfaces", []) if isinstance(item, dict)
        ]
        objects = [LayoutObjectSpec.from_dict(item) for item in payload.get("objects", []) if isinstance(item, dict)]
        regions = [
            RegionSampleSpec.from_dict(item, index=i)
            for i, item in enumerate(payload.get("regions", payload.get("region_sample", [])))
            if isinstance(item, dict)
        ]
        scatter_groups = [
            ScatterGroupSpec.from_dict(item, index=i)
            for i, item in enumerate(payload.get("scatter_groups", payload.get("scatter_group", [])))
            if isinstance(item, dict)
        ]
        rel_dirs = [
            RelativeDirectionSpec.from_dict(item)
            for item in payload.get("relative_directions", payload.get("relative_direction", []))
            if isinstance(item, dict)
        ]
        pose_samples = [
            PoseSampleSpec.from_dict(item, index=i)
            for i, item in enumerate(payload.get("pose_samples", payload.get("pose_sample", [])))
            if isinstance(item, dict)
        ]
        volumes = [
            VolumeSpec.from_dict(item)
            for item in payload.get("volumes", payload.get("volume", []))
            if isinstance(item, dict)
        ]
        in_volumes = [
            InVolumeSpec.from_dict(item, index=i)
            for i, item in enumerate(payload.get("in_volumes", payload.get("in_volume", [])))
            if isinstance(item, dict)
        ]
        volume_groups = [
            VolumeGroupSpec.from_dict(item, index=i)
            for i, item in enumerate(payload.get("volume_groups", payload.get("volume_group", [])))
            if isinstance(item, dict)
        ]
        spec = cls(
            enabled=bool(payload.get("enabled", False)),
            seed=int(payload.get("seed", 0) or 0),
            surfaces=surfaces,
            objects=objects,
            regions=regions,
            scatter_groups=scatter_groups,
            relative_directions=rel_dirs,
            pose_samples=pose_samples,
            volumes=volumes,
            in_volumes=in_volumes,
            volume_groups=volume_groups,
            variants=VariantSpec.from_dict(payload.get("variants")),
            recovery=dict(payload.get("recovery") or {}),
            collision=dict(payload.get("collision") or {}),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        local_surfaces, all_surface_ids = self._validate_surface_ids()
        object_set = self._validate_object_ids()
        self._validate_local_surface_parents(local_surfaces, object_set)
        placement_owners: dict[str, str] = {}

        self._validate_object_supports(all_surface_ids)
        self._validate_regions(all_surface_ids, object_set, placement_owners)
        self._validate_scatter_groups(all_surface_ids, object_set, placement_owners)
        self._validate_relative_directions(object_set)
        self._validate_pose_samples(object_set, placement_owners)
        self._validate_volumes(object_set, placement_owners)
        self._validate_variants(all_surface_ids, object_set)
        self._validate_collision_config()

    def _validate_surface_ids(self) -> tuple[list[LocalSupportSurfaceSpec], set[str]]:
        surface_names = [s.surface_id for s in self.surfaces]
        if len(surface_names) != len(set(surface_names)):
            raise error("surface_duplicate", "support surface ids must be unique")
        local_surfaces = [surface for obj in self.objects for surface in obj.support_surfaces]
        local_surface_ids = [surface.surface_id for surface in local_surfaces]
        if len(local_surface_ids) != len(set(local_surface_ids)):
            raise error("local_surface_duplicate", "local support surface ids must be unique")
        surface_ids = set(surface_names)
        all_surface_ids = surface_ids | set(local_surface_ids)
        if surface_ids & set(local_surface_ids):
            raise error("surface_duplicate", "world and local support surface ids must not collide")
        if self.enabled and not surface_ids and not self.pose_samples and not self.volumes:
            raise error("layout_surface_missing", "layout enabled but no surfaces, pose samples, or volumes declared")
        return local_surfaces, all_surface_ids

    def _validate_object_ids(self) -> set[str]:
        object_names = [obj.name for obj in self.objects]
        if len(object_names) != len(set(object_names)):
            raise error("layout_object_duplicate", "layout object names must be unique")
        return set(object_names)

    def _validate_local_surface_parents(
        self,
        local_surfaces: list[LocalSupportSurfaceSpec],
        object_set: set[str],
    ) -> None:
        for surface in local_surfaces:
            if surface.parent not in object_set:
                raise error(
                    "local_surface_parent_unknown",
                    "local support surface parent is not a layout object",
                    target=surface.surface_id,
                    parent=surface.parent,
                )

    @staticmethod
    def _claim_placement_owner(
        name: str,
        owner: str,
        object_set: set[str],
        placement_owners: dict[str, str],
    ) -> None:
        if name not in object_set:
            return
        previous = placement_owners.get(name)
        if previous is not None and previous != owner:
            raise error(
                "duplicate_placement_owner",
                "layout object has multiple placement owners",
                target=name,
                previous_owner=previous,
                owner=owner,
            )
        placement_owners[name] = owner

    def _validate_object_supports(self, all_surface_ids: set[str]) -> None:
        for obj in self.objects:
            if obj.support and obj.support not in all_surface_ids:
                raise error(
                    "object_support_surface_missing",
                    "object references unknown support surface",
                    target=obj.name,
                    surface=obj.support,
                )

    def _validate_regions(
        self,
        all_surface_ids: set[str],
        object_set: set[str],
        placement_owners: dict[str, str],
    ) -> None:
        for region in self.regions:
            if region.surface not in all_surface_ids:
                raise error(
                    "region_surface_unknown",
                    "region references unknown surface",
                    target=region.region_id,
                    surface=region.surface,
                )
            for name in region.objects:
                if name not in object_set:
                    raise error(
                        "region_object_unknown",
                        "region references unknown object",
                        target=region.region_id,
                        object=name,
                    )
                self._claim_placement_owner(name, f"region_sample:{region.region_id}", object_set, placement_owners)

    def _validate_scatter_groups(
        self,
        all_surface_ids: set[str],
        object_set: set[str],
        placement_owners: dict[str, str],
    ) -> None:
        for group in self.scatter_groups:
            if group.surface not in all_surface_ids:
                raise error(
                    "scatter_group_surface_unknown",
                    "scatter_group references unknown surface",
                    target=group.group_id,
                    surface=group.surface,
                )
            for name in group.objects:
                if name not in object_set:
                    raise error(
                        "scatter_group_object_unknown",
                        "scatter_group references unknown object",
                        target=group.group_id,
                        object=name,
                    )
                self._claim_placement_owner(name, f"scatter_group:{group.group_id}", object_set, placement_owners)

    def _validate_relative_directions(self, object_set: set[str]) -> None:
        for rel in self.relative_directions:
            if rel.node_1 not in object_set or rel.node_2 not in object_set:
                raise error(
                    "relative_direction_object_unknown",
                    "relative_direction references unknown object",
                    target=f"{rel.node_1},{rel.node_2}",
                )

    def _validate_pose_samples(self, object_set: set[str], placement_owners: dict[str, str]) -> None:
        for sample in self.pose_samples:
            if sample.object_name not in object_set:
                raise error(
                    "pose_sample_object_unknown", "pose_sample references unknown object", target=sample.object_name
                )
            self._claim_placement_owner(sample.object_name, "pose_sample", object_set, placement_owners)

    def _validate_volumes(self, object_set: set[str], placement_owners: dict[str, str]) -> set[str]:
        volume_ids = {volume.volume_id for volume in self.volumes}
        if len(volume_ids) != len(self.volumes):
            raise error("volume_duplicate", "volume ids must be unique")
        for volume in self.volumes:
            if volume.parent and volume.parent not in object_set:
                raise error(
                    "volume_parent_unknown",
                    "object-local volume parent is not a layout object",
                    target=volume.volume_id,
                    parent=volume.parent,
                )
        self._validate_in_volumes(volume_ids, object_set, placement_owners)
        self._validate_volume_groups(volume_ids, object_set, placement_owners)
        return volume_ids

    def _validate_in_volumes(
        self,
        volume_ids: set[str],
        object_set: set[str],
        placement_owners: dict[str, str],
    ) -> None:
        for rel in self.in_volumes:
            if rel.object_name not in object_set:
                raise error("in_volume_object_unknown", "in_volume references unknown object", target=rel.object_name)
            if rel.volume not in volume_ids:
                raise error(
                    "in_volume_anchor_unknown",
                    "in_volume references unknown volume",
                    target=rel.object_name,
                    volume=rel.volume,
                )
            self._claim_placement_owner(rel.object_name, f"in_volume:{rel.volume}", object_set, placement_owners)

    def _validate_volume_groups(
        self,
        volume_ids: set[str],
        object_set: set[str],
        placement_owners: dict[str, str],
    ) -> None:
        for group in self.volume_groups:
            if group.volume not in volume_ids:
                raise error(
                    "volume_group_anchor_unknown",
                    "volume_group references unknown volume",
                    target=group.group_id,
                    volume=group.volume,
                )
            for name in group.objects:
                if name not in object_set:
                    raise error(
                        "volume_group_object_unknown",
                        "volume_group references unknown object",
                        target=group.group_id,
                        object=name,
                    )
                self._claim_placement_owner(name, f"volume_group:{group.group_id}", object_set, placement_owners)

    def _validate_variants(self, all_surface_ids: set[str], object_set: set[str]) -> None:
        for slot in self.variants.slots:
            if slot.surface not in all_surface_ids:
                raise error(
                    "variant_slot_surface_unknown",
                    "variant slot references unknown surface",
                    target=slot.slot_id,
                    surface=slot.surface,
                )
        for name in self.variants.objects:
            if name not in object_set:
                raise error("variant_object_unknown", "variant references unknown object", target=name)
        vtype = str(self.variants.type or "").strip().lower()
        if vtype in {"", "fixed", "fixed_cases"}:
            self._validate_fixed_variant_cases(object_set)

    def _validate_fixed_variant_cases(self, object_set: set[str]) -> None:
        if not self.variants.cases:
            return
        slot_ids = {slot.slot_id for slot in self.variants.slots}
        explicit_weights = any("weight" in case for case in self.variants.cases)
        case_ids: set[str] = set()
        total_weight = 0.0
        for index, raw_case in enumerate(self.variants.cases):
            case_id = str(raw_case.get("id") or f"case_{index:03d}")
            if explicit_weights:
                if case_id in case_ids:
                    raise error("variant_case_duplicate", "weighted fixed case ids must be unique", target=case_id)
                case_ids.add(case_id)
            try:
                weight = parse_nonnegative_weight(raw_case.get("weight"), default=1.0)
            except WeightError as exc:
                raise error("variant_case_weight_invalid", str(exc), target=case_id) from exc
            total_weight += weight
            raw_bindings = raw_case.get("bindings") or raw_case.get("objects") or {}
            if not isinstance(raw_bindings, dict):
                raise error("fixed_case_bindings_invalid", "fixed case bindings must be a dict", target=case_id)
            for obj, raw_binding in raw_bindings.items():
                object_name = str(obj)
                if object_name not in object_set:
                    raise error(
                        "fixed_case_object_unknown",
                        "fixed case references unknown object",
                        target=case_id,
                        object=object_name,
                    )
                if isinstance(raw_binding, str):
                    slot_id = raw_binding
                elif isinstance(raw_binding, dict):
                    slot_id = str(raw_binding.get("slot") or "")
                else:
                    raise error(
                        "fixed_case_binding_invalid",
                        "fixed case object binding must be a slot id or mapping",
                        target=case_id,
                        object=object_name,
                    )
                if slot_id and slot_id not in slot_ids:
                    raise error(
                        "fixed_case_slot_unknown",
                        "fixed case references unknown slot",
                        target=case_id,
                        object=object_name,
                        slot=slot_id,
                    )
        if explicit_weights and total_weight <= 0.0:
            raise error("variant_case_weight_all_zero", "weighted fixed cases require at least one positive weight")

    def _validate_collision_config(self) -> None:
        if self.collision:
            mode = str(self.collision.get("mode", "multi_box") or "multi_box").strip().lower()
            if mode not in {"aabb", "multi_box"}:
                raise error("collision_mode_unsupported", "unsupported collision mode", mode=mode)
            if self.collision.get("eps", self.collision.get("contact_tolerance", 0.0)) is not None:
                eps = float(self.collision.get("eps", self.collision.get("contact_tolerance", 0.0)) or 0.0)
                if eps < 0.0:
                    raise error(
                        "collision_eps_invalid", "collision eps/contact_tolerance must be non-negative", eps=eps
                    )
            ignore_pairs = self.collision.get("ignore_pairs", [])
            if ignore_pairs and not isinstance(ignore_pairs, list):
                raise error("collision_ignore_pairs_invalid", "collision ignore_pairs must be a list")

    @property
    def objects_by_name(self) -> dict[str, LayoutObjectSpec]:
        return {obj.name: obj for obj in self.objects}

    @property
    def surfaces_by_id(self) -> dict[str, ManualSupportSurfaceSpec]:
        return {surface.surface_id: surface for surface in self.surfaces}

    @property
    def local_surfaces_by_id(self) -> dict[str, LocalSupportSurfaceSpec]:
        return {surface.surface_id: surface for obj in self.objects for surface in obj.support_surfaces}

    @property
    def pose_samples_by_object(self) -> dict[str, PoseSampleSpec]:
        return {sample.object_name: sample for sample in self.pose_samples}

    @property
    def volumes_by_id(self) -> dict[str, VolumeSpec]:
        return {volume.volume_id: volume for volume in self.volumes}

    @property
    def in_volumes_by_object(self) -> dict[str, InVolumeSpec]:
        return {rel.object_name: rel for rel in self.in_volumes}

    @property
    def volume_groups_by_id(self) -> dict[str, VolumeGroupSpec]:
        return {group.group_id: group for group in self.volume_groups}


def bounds_from_geometry(data: dict[str, Any]) -> Bounds3D | None:
    if not data:
        return None
    if "local_bounds" in data:
        raw = data["local_bounds"]
        if isinstance(raw, dict):
            return Bounds3D.from_dict(raw)
        if isinstance(raw, (list, tuple)):
            return Bounds3D.from_list(raw)
    if "size" in data:
        raise error(
            "geometry_size_unsupported",
            "object geometry.size is not supported; provide real asset geometry.local_bounds",
        )
    return None
