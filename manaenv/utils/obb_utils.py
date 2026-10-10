"""Oriented bounding-box face helpers for contact/push affordance (online, no labels)."""

from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from manaenv.utils.scene_object_utils import get_rigid_object_env_pose

_FACE_NAMES = ("pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z")
_SURFACE_ALIASES = {
    "front": "pos_x",
    "back": "neg_x",
    "left": "pos_y",
    "right": "neg_y",
    "top": "pos_z",
    "bottom": "neg_z",
}


@dataclass(frozen=True)
class ObbFace:
    """One OBB face in env-local coordinates."""

    name: str
    center_env: torch.Tensor
    normal_env: torch.Tensor
    center_local: torch.Tensor
    normal_local: torch.Tensor


def _resolve_env_prim_path(prim_path: str, env_idx: int) -> str:
    concrete_env = f"env_{int(env_idx)}"
    resolved = prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/{concrete_env}")
    return re.sub(r"env_\.\*", concrete_env, resolved, count=1)


def _isaaclab_math() -> Any:

    return importlib.import_module("isaaclab.utils.math")


def _rotate_vectors(quat_xyzw: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    math_utils = _isaaclab_math()
    n = int(vectors.shape[0])
    q = quat_xyzw.reshape(1, 4).expand(n, 4).to(device=vectors.device, dtype=vectors.dtype)
    return math_utils.quat_apply(q, vectors)


def _normalize(vec: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    norm = torch.linalg.norm(vec)
    if float(norm) <= eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def get_object_obb_local(
    env: Any,
    object_name: str,
    env_idx: int,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(bbox_min_local, bbox_max_local)`` in the object's root frame.

    Uses USD ``ComputeUntransformedBound`` which excludes the prim's own
    ``xformOp:scale``. Raises ``RuntimeError`` when USD is unavailable, the
    prim cannot be resolved, or the bound is degenerate -- callers must
    provide an environment where the object's USD prim is reachable.
    """

    dev = device or torch.device("cpu")
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception as exc:
        raise RuntimeError(
            f"get_object_obb_local('{object_name}'): omni.usd / pxr unavailable; "
            "OBB-based skills require an Isaac Sim USD context."
        ) from exc

    scene = getattr(env, "scene", None)
    rigid_objects = getattr(scene, "rigid_objects", None) if scene is not None else None
    obj = rigid_objects[object_name] if rigid_objects is not None and object_name in rigid_objects else None
    prim_path = str(getattr(getattr(obj, "cfg", None), "prim_path", "") or "")
    if not prim_path:
        raise RuntimeError(
            f"get_object_obb_local('{object_name}'): rigid object has no prim_path "
            "(check scene.rigid_objects registration)."
        )
    prim_path = _resolve_env_prim_path(prim_path, env_idx)

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"get_object_obb_local('{object_name}'): USD prim '{prim_path}' is invalid.")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound = cache.ComputeUntransformedBound(prim)
    box = bound.GetBox()
    bbox_min = torch.tensor(list(box.GetMin()), dtype=torch.float32, device=dev)
    bbox_max = torch.tensor(list(box.GetMax()), dtype=torch.float32, device=dev)
    if float((bbox_max - bbox_min).max()) <= 1.0e-6:
        raise RuntimeError(
            f"get_object_obb_local('{object_name}'): USD prim '{prim_path}' "
            "returned a degenerate bound (zero extent)."
        )
    return bbox_min, bbox_max


def build_obb_faces_env(
    *,
    bbox_min_local: torch.Tensor,
    bbox_max_local: torch.Tensor,
    object_pos_env: torch.Tensor,
    object_quat_xyzw: torch.Tensor,
    obj_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[ObbFace]:
    """Build six OBB faces transformed into env-local coordinates.

    ``bbox_min_local`` / ``bbox_max_local`` are expected to come from
    ``UsdGeom.BBoxCache.ComputeUntransformedBound`` which excludes the prim's
    own ``xformOp:scale``. Pass the spawn ``obj_scale`` so the post-scale
    geometry is reflected in the returned face centers.
    """

    device = object_pos_env.device
    dtype = torch.float32
    min_l = bbox_min_local.to(device=device, dtype=dtype).reshape(3)
    max_l = bbox_max_local.to(device=device, dtype=dtype).reshape(3)
    scale_t = torch.as_tensor(obj_scale, dtype=dtype, device=device).reshape(3)
    min_l = min_l * scale_t
    max_l = max_l * scale_t
    mid = 0.5 * (min_l + max_l)

    specs: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        (
            "pos_x",
            torch.tensor([max_l[0], mid[1], mid[2]], device=device, dtype=dtype),
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
        ),
        (
            "neg_x",
            torch.tensor([min_l[0], mid[1], mid[2]], device=device, dtype=dtype),
            torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=dtype),
        ),
        (
            "pos_y",
            torch.tensor([mid[0], max_l[1], mid[2]], device=device, dtype=dtype),
            torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
        ),
        (
            "neg_y",
            torch.tensor([mid[0], min_l[1], mid[2]], device=device, dtype=dtype),
            torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype),
        ),
        (
            "pos_z",
            torch.tensor([mid[0], mid[1], max_l[2]], device=device, dtype=dtype),
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
        ),
        (
            "neg_z",
            torch.tensor([mid[0], mid[1], min_l[2]], device=device, dtype=dtype),
            torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype),
        ),
    ]

    centers_local = torch.stack([spec[1] for spec in specs], dim=0)
    normals_local = torch.stack([spec[2] for spec in specs], dim=0)
    centers_env = object_pos_env.reshape(1, 3) + _rotate_vectors(object_quat_xyzw, centers_local)
    normals_env = _rotate_vectors(object_quat_xyzw, normals_local)
    normals_env = normals_env / torch.linalg.norm(normals_env, dim=-1, keepdim=True).clamp(min=1.0e-8)

    faces: list[ObbFace] = []
    for idx, (name, center_local, normal_local) in enumerate(specs):
        faces.append(
            ObbFace(
                name=name,
                center_env=centers_env[idx],
                normal_env=normals_env[idx],
                center_local=center_local,
                normal_local=normal_local,
            )
        )
    return faces


def get_obb_face_centers_env(
    env: Any,
    object_name: str,
    env_idx: int,
    *,
    env_origin: torch.Tensor | None = None,
    obj_scale: tuple[float, float, float] | None = None,
) -> list[ObbFace]:
    """Return six OBB faces for one rigid object in env-local coordinates."""

    from manaenv.utils.scene_object_utils import get_rigid_object_scale

    pose = get_rigid_object_env_pose(env, object_name, env_idx, env_origin=env_origin)
    if pose is None:
        raise RuntimeError(f"get_obb_face_centers_env: missing pose for '{object_name}'.")
    pos_env, quat = pose
    device = pos_env.device
    bbox_min, bbox_max = get_object_obb_local(env, object_name, env_idx, device=device)
    resolved_scale = obj_scale if obj_scale is not None else get_rigid_object_scale(env, object_name)
    return build_obb_faces_env(
        bbox_min_local=bbox_min,
        bbox_max_local=bbox_max,
        object_pos_env=pos_env,
        object_quat_xyzw=quat,
        obj_scale=resolved_scale,
    )


def pick_face_by_direction(faces: Sequence[ObbFace], direction: torch.Tensor) -> ObbFace:
    """Pick the face whose outward normal best aligns with ``direction``."""

    if not faces:
        raise ValueError("pick_face_by_direction requires at least one face.")
    dir_n = _normalize(torch.as_tensor(direction, dtype=torch.float32).reshape(3))
    best = faces[0]
    best_score = -1.0e9
    for face in faces:
        normal = face.normal_env.to(device=dir_n.device, dtype=dir_n.dtype)
        score = float(torch.dot(normal, dir_n))
        if score > best_score:
            best_score = score
            best = face
    return best


def pick_face_by_name(faces: Sequence[ObbFace], name: str) -> ObbFace:
    """Resolve a face by canonical surface name."""

    key = str(name or "").strip().lower()
    if key == "auto":
        key = "top"
    resolved = _SURFACE_ALIASES.get(key, key)
    for face in faces:
        if face.name == resolved:
            return face
    raise KeyError(f"OBB face '{name}' (resolved '{resolved}') not found among {[f.name for f in faces]}.")


def local_point_to_env(
    local_point: Sequence[float],
    *,
    object_pos_env: torch.Tensor,
    object_quat_xyzw: torch.Tensor,
    obj_scale: tuple[float, float, float],
) -> torch.Tensor:
    """Map one mesh-local point into env coordinates (with spawn scale)."""

    device = object_pos_env.device
    scale = torch.as_tensor(obj_scale, dtype=torch.float32, device=device).reshape(3)
    local = torch.as_tensor(local_point, dtype=torch.float32, device=device).reshape(3) * scale
    rotated = _rotate_vectors(object_quat_xyzw, local.reshape(1, 3)).reshape(3)
    return object_pos_env.reshape(3) + rotated


def local_axis_to_env(
    local_axis: Sequence[float],
    *,
    object_quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Map one mesh-local axis direction into env coordinates (no translation)."""

    device = object_quat_xyzw.device
    local = torch.as_tensor(local_axis, dtype=torch.float32, device=device).reshape(3)
    rotated = _rotate_vectors(object_quat_xyzw, local.reshape(1, 3)).reshape(3)
    return _normalize(rotated)


def _as_vec3(value: torch.Tensor) -> torch.Tensor:
    flat = value.reshape(-1)
    if flat.numel() < 3:
        raise ValueError(f"Expected a 3-vector; got shape {tuple(value.shape)}.")
    return flat[:3].reshape(3)


def _resolve_scene_asset_prim_path(env: Any, object_name: str, env_idx: int) -> str:
    """Resolve a scene entity USD path from rigid objects or articulations."""

    scene = getattr(env, "scene", None)
    if scene is None:
        return ""
    for attr in ("rigid_objects", "articulations"):
        registry = getattr(scene, attr, None)
        if registry is None or object_name not in registry:
            continue
        obj = registry[object_name]
        prim_paths = getattr(obj, "prim_paths", None)
        if prim_paths is not None:
            try:
                values = [str(v) for v in prim_paths if v]
                if values:
                    idx = min(max(int(env_idx), 0), len(values) - 1)
                    return values[idx]
            except Exception:
                pass
        cfg = getattr(obj, "cfg", None)
        raw = str(getattr(cfg, "prim_path", "") or getattr(obj, "prim_path", "") or "")
        if raw:
            return _resolve_env_prim_path(raw, env_idx)
    return ""


def get_asset_child_prim_point_env(
    env: Any,
    object_name: str,
    env_idx: int,
    child_prim: str,
    *,
    env_origin: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Return env-frame position of a child USD prim under a scene asset root.

    *child_prim* is the suffix under the resolved asset root, e.g. ``slot_a`` for
    ``<asset_root>/slot_a``. Returns ``None`` when USD is unavailable or the
    child prim is missing.
    """

    dev = device or torch.device("cpu")
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception:
        return None

    prim_path = _resolve_scene_asset_prim_path(env, object_name, env_idx)
    child = str(child_prim or "").strip().lstrip("/")
    if not prim_path or not child:
        return None
    child_path = f"{prim_path.rstrip('/')}/{child}"

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    prim = stage.GetPrimAtPath(child_path)
    if prim is None or not prim.IsValid():
        return None

    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = xf.ExtractTranslation()
    pos = torch.tensor([float(t[0]), float(t[1]), float(t[2])], dtype=torch.float32, device=dev)
    if env_origin is not None:
        origin = _as_vec3(torch.as_tensor(env_origin, dtype=torch.float32, device=dev))
    else:
        from manaenv.utils.scene_object_utils import get_env_origin

        origin = get_env_origin(env, env_idx)
        if origin is None:
            return pos
        origin = _as_vec3(origin.to(device=dev, dtype=torch.float32))
    return pos - origin


def get_object_bbox_root_frame(
    env: Any,
    object_name: str,
    env_idx: int,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(bbox_min, bbox_max)`` in the object's physics root frame, in meters.

    Computes the USD world bound at the default timecode and maps corners through
    the inverse of the prim's spawn world transform so the result stays valid after
    rigid motion.
    """
    dev = device or torch.device("cpu")
    try:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom
    except Exception as exc:
        raise RuntimeError(
            f"get_object_bbox_root_frame('{object_name}'): omni.usd / pxr unavailable; "
            "requires an Isaac Sim USD context."
        ) from exc

    prim_path = _resolve_scene_asset_prim_path(env, object_name, env_idx)
    if not prim_path:
        raise RuntimeError(
            f"get_object_bbox_root_frame('{object_name}'): scene asset has no prim_path "
            "(check scene.rigid_objects / scene.articulations registration)."
        )

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"get_object_bbox_root_frame('{object_name}'): USD prim '{prim_path}' is invalid.")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    w_min = aligned.GetMin()
    w_max = aligned.GetMax()

    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    xf_inv = xf.GetInverse()

    local_pts = []
    for ix in (w_min[0], w_max[0]):
        for iy in (w_min[1], w_max[1]):
            for iz in (w_min[2], w_max[2]):
                local_pts.append(xf_inv.Transform(Gf.Vec3d(ix, iy, iz)))
    pts = torch.tensor([[p[0], p[1], p[2]] for p in local_pts], dtype=torch.float32, device=dev)
    bbox_min = pts.min(dim=0).values
    bbox_max = pts.max(dim=0).values
    if float((bbox_max - bbox_min).max()) <= 1.0e-6:
        raise RuntimeError(f"get_object_bbox_root_frame('{object_name}'): degenerate bound at '{prim_path}'.")
    return bbox_min, bbox_max


def get_object_footprint_aabb_env(
    env: Any,
    object_name: str,
    env_idx: int,
    *,
    env_origin: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Env-frame footprint AABB: resting root-frame bound at the live position.

    Translates the root-frame bound by live root position without re-applying the
    live root quaternion (spawn orientation is already baked into the root-frame
    bound).
    """
    dev = device or torch.device("cpu")
    try:
        bbox_min, bbox_max = get_object_bbox_root_frame(env, object_name, env_idx, device=dev)
    except RuntimeError:
        return None
    pose = get_rigid_object_env_pose(env, object_name, env_idx, env_origin=env_origin)
    if pose is None:
        return None
    pos, _ = pose
    pos = _as_vec3(pos.to(device=dev, dtype=torch.float32))
    return pos + bbox_min.to(device=dev), pos + bbox_max.to(device=dev)


__all__ = [
    "ObbFace",
    "build_obb_faces_env",
    "get_object_bbox_root_frame",
    "get_object_footprint_aabb_env",
    "get_object_obb_local",
    "get_obb_face_centers_env",
    "local_axis_to_env",
    "local_point_to_env",
    "pick_face_by_direction",
    "pick_face_by_name",
]
