"""Runtime utilities for offline-label-backed grasp selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from manaenv.utils.math_utils import quat_wxyz_normalize, quat_wxyz_to_rotmat, quat_xyzw_to_wxyz

GRASP_NPY_DTYPE = np.float64
GRASP_NPY_COLS = 17
MANIFEST_VERSION = 1
SHAPE_METADATA_KEY = "shape_metadata"
DEFAULT_PRINCIPAL_AXIS: tuple[float, float, float] = (1.0, 0.0, 0.0)

# Frame conversion aligned with InternData `vis_grasp.py`:
# R_actual = R_view @ R1.T
R1 = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=GRASP_NPY_DTYPE,
)

_FAMILY_ROUNDISH = "roundish"
_FAMILY_ELONGATED = "elongated"
_FAMILY_CYLINDRICAL = "cylindrical"
_FAMILY_FLAT_THIN = "flat_thin"
_FAMILY_GENERIC = "generic"


_ROUNDISH_LINEARITY_MAX = 1.5
_ROUNDISH_PLANARITY_MAX = 1.5
_FLAT_PLANARITY_MIN = 1.5
_FLAT_LINEARITY_MAX = 4.0
_CYLINDRICAL_LINEARITY_MIN = 1.5
_CYLINDRICAL_PLANARITY_MAX = 1.5
_ELONGATED_LINEARITY_MIN = 3.5
_UPRIGHT_MIN_DOT = 0.85
_LYING_MAX_DOT = 0.3

_THIN_OBJECT_HEIGHT_M = 0.04

AUTO_FILTER_DEFAULT_PROFILE = "auto"
MANUAL_FILTER_PROFILE = "manual"


class ShapeMetadata:
    """Per-asset intrinsic shape metadata used by runtime grasp filtering."""

    __slots__ = (
        "aabb_extent_m",
        "bbox_diag_m",
        "elongation_ratio",
        "shape_family",
        "principal_axis_local",
    )

    def __init__(
        self,
        *,
        aabb_extent_m: tuple[float, float, float],
        bbox_diag_m: float,
        elongation_ratio: float,
        shape_family: str,
        principal_axis_local: tuple[float, float, float],
    ) -> None:
        self.aabb_extent_m = (
            float(aabb_extent_m[0]),
            float(aabb_extent_m[1]),
            float(aabb_extent_m[2]),
        )
        self.bbox_diag_m = float(bbox_diag_m)
        self.elongation_ratio = float(elongation_ratio)
        self.shape_family = str(shape_family)
        self.principal_axis_local = (
            float(principal_axis_local[0]),
            float(principal_axis_local[1]),
            float(principal_axis_local[2]),
        )

    def __repr__(self) -> str:
        return (
            "ShapeMetadata("
            f"aabb_extent_m={self.aabb_extent_m}, "
            f"bbox_diag_m={self.bbox_diag_m:.4f}, "
            f"elongation_ratio={self.elongation_ratio:.3f}, "
            f"shape_family={self.shape_family!r}, "
            f"principal_axis_local={self.principal_axis_local})"
        )


def validate_grasp_rows(rows: np.ndarray) -> np.ndarray:
    """Validate grasp rows and return a float64 array shaped as (N, 17)."""
    array = np.asarray(rows, dtype=GRASP_NPY_DTYPE)
    if array.ndim == 1:
        if array.size != GRASP_NPY_COLS:
            raise ValueError(f"Expected a 17-column grasp row, got shape {array.shape}")
        array = array.reshape(1, GRASP_NPY_COLS)
    if array.ndim != 2 or array.shape[1] != GRASP_NPY_COLS:
        raise ValueError(f"Expected grasp rows with shape (N, 17), got {array.shape}")
    return array


def load_grasp_npy(path: Path) -> np.ndarray:
    """Load a grasp label array and validate the `(N, 17)` schema."""
    array = np.load(Path(path), allow_pickle=False)
    return validate_grasp_rows(array)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a grasp-label manifest, creating the base structure if absent."""
    path = Path(manifest_path)
    if not path.exists():
        return {
            "version": MANIFEST_VERSION,
            "sampler_default": {},
            "assets": {},
            "failures": [],
        }
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Manifest root must be a mapping: {path}")
    manifest = {
        "version": int(data.get("version", MANIFEST_VERSION)),
        "sampler_default": data.get("sampler_default") or {},
        "assets": data.get("assets") or {},
        "failures": data.get("failures") or [],
    }
    if not isinstance(manifest["assets"], dict):
        raise ValueError(f"Manifest 'assets' must be a mapping: {path}")
    if not isinstance(manifest["failures"], list):
        raise ValueError(f"Manifest 'failures' must be a list: {path}")
    return manifest


def get_asset_shape_metadata(manifest: Mapping[str, Any], asset_name: str) -> ShapeMetadata | None:
    """Read intrinsic shape metadata for one asset from a loaded manifest."""
    assets = manifest.get("assets") if isinstance(manifest, Mapping) else None
    if not isinstance(assets, Mapping):
        return None
    asset_payload = assets.get(asset_name)
    if not isinstance(asset_payload, Mapping):
        return None
    return metadata_from_mapping(asset_payload.get(SHAPE_METADATA_KEY))


def shape_metadata_from_extents(extent_m: tuple[float, float, float] | list[float] | np.ndarray) -> ShapeMetadata:
    """Derive shape metadata from object-local AABB extents."""
    extents = np.asarray(extent_m, dtype=np.float64).reshape(3)
    if np.any(extents < 0):
        raise ValueError(f"AABB extents must be non-negative, got {extents.tolist()}")

    safe_extents = np.where(extents <= 1.0e-6, 1.0e-6, extents)
    diag = float(np.linalg.norm(extents))
    sorted_extents = np.sort(safe_extents)
    c = float(sorted_extents[0])
    b = float(sorted_extents[1])
    a = float(sorted_extents[2])
    linearity = a / b
    planarity = b / c
    elongation = a / c

    return ShapeMetadata(
        aabb_extent_m=(float(extents[0]), float(extents[1]), float(extents[2])),
        bbox_diag_m=diag,
        elongation_ratio=elongation,
        shape_family=_classify_family(linearity, planarity),
        principal_axis_local=_principal_axis_from_extents(safe_extents),
    )


def metadata_to_mapping(metadata: ShapeMetadata) -> dict[str, Any]:
    """Render shape metadata as a YAML/JSON-friendly mapping."""
    return {
        "aabb_extent_m": list(metadata.aabb_extent_m),
        "bbox_diag_m": float(metadata.bbox_diag_m),
        "elongation_ratio": float(metadata.elongation_ratio),
        "shape_family": metadata.shape_family,
        "principal_axis_local": list(metadata.principal_axis_local),
    }


def metadata_from_mapping(payload: Mapping[str, Any] | None) -> ShapeMetadata | None:
    """Restore metadata from a manifest payload, reclassifying legacy entries."""
    if not isinstance(payload, Mapping):
        return None
    extents_raw = payload.get("aabb_extent_m")
    if extents_raw is None:
        return None
    try:
        extents = np.asarray(extents_raw, dtype=np.float64).reshape(3)
    except Exception:
        return None
    axis_raw = payload.get("principal_axis_local")
    try:
        axis_arr = np.asarray(axis_raw, dtype=np.float64).reshape(3)
        axis_norm = float(np.linalg.norm(axis_arr))
        if axis_norm < 1.0e-9:
            axis_arr = np.asarray(DEFAULT_PRINCIPAL_AXIS, dtype=np.float64)
        else:
            axis_arr = axis_arr / axis_norm
    except Exception:
        axis_arr = np.asarray(DEFAULT_PRINCIPAL_AXIS, dtype=np.float64)

    rederived = shape_metadata_from_extents(extents)
    return ShapeMetadata(
        aabb_extent_m=rederived.aabb_extent_m,
        bbox_diag_m=float(payload.get("bbox_diag_m") or rederived.bbox_diag_m),
        elongation_ratio=rederived.elongation_ratio,
        shape_family=rederived.shape_family,
        principal_axis_local=(float(axis_arr[0]), float(axis_arr[1]), float(axis_arr[2])),
    )


def classify_pose_state(
    metadata: ShapeMetadata,
    object_quat_xyzw: Any,
    *,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> str:
    """Classify object pose as ``upright``, ``lying`` or ``tilted``."""
    if metadata.shape_family == _FAMILY_ROUNDISH:
        return "upright"
    try:
        quat = _quat_to_numpy(object_quat_xyzw)
        quat = quat_wxyz_normalize(quat_xyzw_to_wxyz(quat)).astype(np.float64)
    except Exception:
        return "tilted"

    rot = quat_wxyz_to_rotmat(quat).astype(np.float64)
    axis_local = np.asarray(metadata.principal_axis_local, dtype=np.float64).reshape(3)
    axis_world = rot @ axis_local
    up = np.asarray(world_up, dtype=np.float64).reshape(3)
    up_norm = float(np.linalg.norm(up))
    axis_norm = float(np.linalg.norm(axis_world))
    if up_norm < 1.0e-9 or axis_norm < 1.0e-9:
        return "tilted"
    cos_abs = float(abs(float(np.dot(axis_world, up / up_norm)) / axis_norm))
    if cos_abs >= _UPRIGHT_MIN_DOT:
        return "upright"
    if cos_abs <= _LYING_MAX_DOT:
        return "lying"
    return "tilted"


def _axis_direction_mask(rot: torch.Tensor, *, axis_name: str, filter_spec: list[Any]) -> torch.Tensor:
    """Return the keep-mask for the local-axis direction DSL."""
    filter_conditions = {
        "x": {
            "forward": (0, 0, 1),
            "backward": (0, 0, -1),
            "upward": (2, 0, 1),
            "downward": (2, 0, -1),
        },
        "y": {
            "forward": (0, 1, 1),
            "backward": (0, 1, -1),
            "upward": (2, 1, 1),
            "downward": (2, 1, -1),
        },
        "z": {
            "forward": (0, 2, 1),
            "backward": (0, 2, -1),
            "upward": (2, 2, 1),
            "downward": (2, 2, -1),
        },
    }
    if axis_name not in filter_conditions:
        raise ValueError(f"Unsupported filter axis '{axis_name}'.")
    if len(filter_spec) not in (2, 3):
        raise ValueError(f"filter_{axis_name}_dir expects [direction, angle] or [direction, min_angle, max_angle].")

    direction = str(filter_spec[0]).strip().lower()
    if direction not in filter_conditions[axis_name]:
        allowed = ", ".join(sorted(filter_conditions[axis_name]))
        raise ValueError(f"Unsupported filter_{axis_name}_dir direction '{direction}'. Allowed: {allowed}.")

    row, col, sign = filter_conditions[axis_name][direction]
    cos_dot = sign * rot[:, row, col]
    device = rot.device
    dtype = rot.dtype
    if len(filter_spec) == 2:
        angle_deg = float(filter_spec[1])
        cos_val = torch.cos(torch.deg2rad(torch.as_tensor(angle_deg, dtype=dtype, device=device)))
        return cos_dot >= cos_val

    min_angle = float(filter_spec[1])
    max_angle = float(filter_spec[2])
    if min_angle > max_angle:
        raise ValueError(f"filter_{axis_name}_dir min_angle ({min_angle}) must not exceed max_angle ({max_angle}).")
    cos_max = float(torch.cos(torch.deg2rad(torch.as_tensor(max_angle, dtype=dtype, device=device))).item())
    cos_min = float(torch.cos(torch.deg2rad(torch.as_tensor(min_angle, dtype=dtype, device=device))).item())
    return (cos_dot >= cos_max) & (cos_dot <= cos_min)


@dataclass(frozen=True)
class GraspFilterProfile:
    """Plain holder for one filter preset."""

    name: str
    filter_x_dir: list[Any] | None = None
    filter_y_dir: list[Any] | None = None
    filter_z_dir: list[Any] | None = None
    direction_to_obj: str | None = None
    fallback: tuple[str, ...] = field(default_factory=tuple)


_PROFILES: dict[str, GraspFilterProfile] = {
    "top_down_long": GraspFilterProfile(
        name="top_down_long",
        filter_x_dir=["downward", 35.0],
        fallback=("top_down_long_relaxed", "generic_downward", "open"),
    ),
    "top_down_long_relaxed": GraspFilterProfile(
        name="top_down_long_relaxed",
        filter_x_dir=["downward", 55.0],
        fallback=("generic_downward", "open"),
    ),
    "side_cylinder": GraspFilterProfile(
        name="side_cylinder",
        filter_x_dir=["downward", 55.0, 95.0],
        fallback=("side_cylinder_relaxed", "generic_downward", "open"),
    ),
    "side_cylinder_relaxed": GraspFilterProfile(
        name="side_cylinder_relaxed",
        filter_x_dir=["downward", 30.0, 95.0],
        fallback=("generic_downward", "open"),
    ),
    "relaxed_round": GraspFilterProfile(
        name="relaxed_round",
        filter_x_dir=["downward", 0.0, 45],
        fallback=("generic_downward", "open"),
    ),
    "flat_top_down": GraspFilterProfile(
        name="flat_top_down",
        filter_x_dir=["downward", 50.0],
        fallback=("flat_top_down_relaxed", "generic_downward", "open"),
    ),
    "flat_top_down_relaxed": GraspFilterProfile(
        name="flat_top_down_relaxed",
        filter_x_dir=["downward", 75.0],
        fallback=("generic_downward", "open"),
    ),
    "flat_top_down_horizontal": GraspFilterProfile(
        name="flat_top_down_horizontal",
        filter_x_dir=["downward", 50.0],
        filter_y_dir=["upward", 85.0, 95.0],
        fallback=(
            "flat_top_down_horizontal_relaxed",
            "flat_top_down",
            "flat_top_down_relaxed",
            "generic_downward",
            "open",
        ),
    ),
    "flat_top_down_horizontal_relaxed": GraspFilterProfile(
        name="flat_top_down_horizontal_relaxed",
        filter_x_dir=["downward", 75.0],
        filter_y_dir=["upward", 80.0, 100.0],
        fallback=("flat_top_down_relaxed", "generic_downward", "open"),
    ),
    "generic_downward": GraspFilterProfile(
        name="generic_downward",
        filter_x_dir=["downward", 100.0],
        fallback=("open",),
    ),
    "open": GraspFilterProfile(name="open"),
}

_FAMILY_POSE_TABLE: dict[tuple[str, str], str] = {
    ("elongated", "lying"): "top_down_long",
    ("elongated", "tilted"): "top_down_long_relaxed",
    ("elongated", "upright"): "side_cylinder",
    ("cylindrical", "upright"): "side_cylinder",
    ("cylindrical", "lying"): "top_down_long",
    ("cylindrical", "tilted"): "top_down_long_relaxed",
    ("flat_thin", "lying"): "flat_top_down",
    ("flat_thin", "tilted"): "flat_top_down_relaxed",
    ("flat_thin", "upright"): "flat_top_down_relaxed",
    ("flat_thin", "any"): "flat_top_down",
    ("roundish", "any"): "relaxed_round",
    ("generic", "any"): "generic_downward",
}


def manual_overrides_present(cfg_like: Any) -> bool:
    """Return True when any of the override fields are explicitly set."""
    return any(
        getattr(cfg_like, name, None) is not None
        for name in ("filter_x_dir", "filter_y_dir", "filter_z_dir", "direction_to_obj")
    )


def _is_thin_lying_object(metadata: ShapeMetadata | None) -> bool:
    """True when the smallest AABB extent is at or below ``_THIN_OBJECT_HEIGHT_M``.

    Used to special-case top-down picks on thin lying objects (cards, erasers,
    sheet-like things) where the gripper closing axis must remain near horizontal
    to avoid colliding with the table.
    """
    if metadata is None:
        return False
    extents = metadata.aabb_extent_m
    if not extents:
        return False
    return min(float(e) for e in extents) <= _THIN_OBJECT_HEIGHT_M


def world_z_extent_m(
    metadata: ShapeMetadata | None,
    object_quat_xyzw: Any,
) -> float | None:
    if metadata is None:
        return None
    extents = metadata.aabb_extent_m
    if not extents or len(extents) != 3:
        return None
    try:
        quat = _quat_to_numpy(object_quat_xyzw)
        quat = quat_wxyz_normalize(quat_xyzw_to_wxyz(quat)).astype(np.float64)
    except Exception:
        return None
    rot = quat_wxyz_to_rotmat(quat).astype(np.float64)
    ex, ey, ez = (float(extents[0]), float(extents[1]), float(extents[2]))
    return float(abs(rot[2, 0]) * ex + abs(rot[2, 1]) * ey + abs(rot[2, 2]) * ez)


def should_force_horizontal_close(
    metadata: ShapeMetadata | None,
    object_quat_xyzw: Any,
    *,
    height_threshold_m: float = _THIN_OBJECT_HEIGHT_M,
) -> bool:
    height = world_z_extent_m(metadata, object_quat_xyzw)
    if height is None:
        return False
    return height <= height_threshold_m


_HORIZONTAL_CLOSE_TOLERANCES_DEG: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)


def horizontal_close_filter_y_dir(step_index: int) -> list[Any]:
    idx = max(int(step_index), 0)
    idx = min(idx, len(_HORIZONTAL_CLOSE_TOLERANCES_DEG) - 1)
    tol = float(_HORIZONTAL_CLOSE_TOLERANCES_DEG[idx])
    return ["upward", 90.0 - tol, 90.0 + tol]


def profile_for(metadata: ShapeMetadata | None, pose_state: str) -> GraspFilterProfile:
    """Resolve a base profile for ``(shape_family, pose_state)``."""
    if metadata is None:
        return _PROFILES["generic_downward"]
    family = (metadata.shape_family or "generic").strip().lower() or "generic"
    state = (pose_state or "").strip().lower() or "tilted"
    name = _FAMILY_POSE_TABLE.get((family, state))
    if name is None:
        name = _FAMILY_POSE_TABLE.get((family, "any"))
    if name is None:
        name = "generic_downward"
    if family == _FAMILY_FLAT_THIN and _is_thin_lying_object(metadata) and name == "flat_top_down":
        name = "flat_top_down_horizontal"
    return _PROFILES[name]


def profile_fallback_chain(profile: GraspFilterProfile) -> tuple[GraspFilterProfile, ...]:
    """Return the deterministic profile order: head + named fallbacks."""
    chain: list[GraspFilterProfile] = [profile]
    seen: set[str] = {profile.name}
    for name in profile.fallback:
        if name in seen:
            continue
        next_profile = _PROFILES.get(name)
        if next_profile is None:
            continue
        chain.append(next_profile)
        seen.add(name)
    return tuple(chain)


def resolve_filter_profile(
    cfg_like: Any,
    metadata: ShapeMetadata | None,
    object_quat_xyzw: Any,
) -> tuple[GraspFilterProfile, ...]:
    """Pick a filter profile chain for the current pick."""
    if manual_overrides_present(cfg_like):
        return tuple()
    profile_name = str(getattr(cfg_like, "filter_profile", "") or "").strip().lower()
    if profile_name == MANUAL_FILTER_PROFILE:
        return tuple()
    if profile_name and profile_name != AUTO_FILTER_DEFAULT_PROFILE:
        named = _PROFILES.get(profile_name)
        if named is None:
            allowed = sorted([AUTO_FILTER_DEFAULT_PROFILE, MANUAL_FILTER_PROFILE, *_PROFILES.keys()])
            raise ValueError(f"Unknown grasp filter_profile '{profile_name}'. Allowed: " + ", ".join(allowed))
        return profile_fallback_chain(named)
    pose_state = classify_pose_state(metadata, object_quat_xyzw) if metadata is not None else "tilted"
    return profile_fallback_chain(profile_for(metadata, pose_state))


def merge_resolved_profile(cfg_like: Any, profile: GraspFilterProfile) -> dict[str, Any]:
    """Return a dict of filter fields to apply for one resolution attempt."""
    return {
        "filter_x_dir": getattr(cfg_like, "filter_x_dir", None) or profile.filter_x_dir,
        "filter_y_dir": getattr(cfg_like, "filter_y_dir", None) or profile.filter_y_dir,
        "filter_z_dir": getattr(cfg_like, "filter_z_dir", None) or profile.filter_z_dir,
        "direction_to_obj": getattr(cfg_like, "direction_to_obj", None) or profile.direction_to_obj,
    }


def _classify_family(linearity: float, planarity: float) -> str:
    if linearity <= _ROUNDISH_LINEARITY_MAX and planarity <= _ROUNDISH_PLANARITY_MAX:
        return _FAMILY_ROUNDISH
    if planarity >= _FLAT_PLANARITY_MIN and linearity <= _FLAT_LINEARITY_MAX:
        return _FAMILY_FLAT_THIN
    if linearity >= _CYLINDRICAL_LINEARITY_MIN and planarity <= _CYLINDRICAL_PLANARITY_MAX:
        if linearity >= _ELONGATED_LINEARITY_MIN:
            return _FAMILY_ELONGATED
        return _FAMILY_CYLINDRICAL
    return _FAMILY_GENERIC


def _quat_to_numpy(value: Any) -> np.ndarray:
    from manaenv.utils.tensor_utils import to_numpy_keep_shape

    return np.asarray(to_numpy_keep_shape(value), dtype=np.float64).reshape(4)


def _principal_axis_from_extents(extents: np.ndarray) -> tuple[float, float, float]:
    longest_idx = int(np.argmax(extents))
    axis = [0.0, 0.0, 0.0]
    axis[longest_idx] = 1.0
    return (float(axis[0]), float(axis[1]), float(axis[2]))


__all__ = [
    "AUTO_FILTER_DEFAULT_PROFILE",
    "DEFAULT_PRINCIPAL_AXIS",
    "GRASP_NPY_COLS",
    "GRASP_NPY_DTYPE",
    "GraspFilterProfile",
    "MANIFEST_VERSION",
    "MANUAL_FILTER_PROFILE",
    "R1",
    "SHAPE_METADATA_KEY",
    "ShapeMetadata",
    "_PROFILES",
    "_axis_direction_mask",
    "classify_pose_state",
    "get_asset_shape_metadata",
    "horizontal_close_filter_y_dir",
    "load_grasp_npy",
    "load_manifest",
    "manual_overrides_present",
    "merge_resolved_profile",
    "metadata_from_mapping",
    "metadata_to_mapping",
    "profile_fallback_chain",
    "profile_for",
    "resolve_filter_profile",
    "shape_metadata_from_extents",
    "should_force_horizontal_close",
    "validate_grasp_rows",
    "world_z_extent_m",
]
