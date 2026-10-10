"""Manifest/npy-backed grasp payload cache and decoding for PickSkill."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from manaenv.assets.resolver import is_asset_reference, resolve_asset_path
from manaenv.utils.grasp_runtime import R1, ShapeMetadata, get_asset_shape_metadata, load_grasp_npy, load_manifest
from manaenv.utils.math_utils import quat_wxyz_to_xyzw
from manaenv.utils.motion_utils import world_to_base_frame
from manaenv.utils.tensor_utils import as_torch

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = REPO_ROOT / "manaenv"


@dataclass
class PickGraspData:
    """Cached grasp payload tensors and manifest-derived metadata."""

    rows: torch.Tensor | None = None
    widths: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    sampler_default: dict[str, Any] | None = None
    shape_metadata: ShapeMetadata | None = None
    shape_metadata_loaded: bool = False
    effective_approach_axis: tuple[float, float, float] | None = None
    # Manifest articulation metadata (optional; rigid assets omit these).
    reference_frame: str | None = None
    articulation: bool = False
    joint_type: str | None = None
    joint_axis_local: tuple[float, float, float] | None = None
    operated_body_names: tuple[str, ...] | None = None
    # Object-local grasp positions from the most recent ``decode_grasps`` call.
    # Cached (not returned) so region-based skills (e.g. ArticulationGripSkill) can filter
    # by an object-local band without changing the shared decode return arity.
    last_grasp_pos_obj: torch.Tensor | None = None
    # Label contact pose (before ``sampler_to_link6_backoff``) for debug viz/logging.
    last_label_pos_obj: torch.Tensor | None = None
    last_label_pos_env: torch.Tensor | None = None
    last_label_quat_env: torch.Tensor | None = None


def isaaclab_math_utils() -> Any:
    return importlib.import_module("isaaclab.utils.math")


def resolve_data_path(path_value: str, *, preferred_root: Path | None = None) -> Path:
    if is_asset_reference(path_value):
        return resolve_asset_path(path_value)
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    search_roots: list[Path] = []
    if preferred_root is not None:
        search_roots.append(preferred_root)
    search_roots.extend([REPO_ROOT, PACKAGE_ROOT])
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    base_root = search_roots[0] if search_roots else REPO_ROOT
    return (base_root / candidate).resolve()


def normalize_axis(axis: tuple[float, float, float] | list[float], *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(axis, dtype=torch.float32, device=device).reshape(-1)
    if tensor.numel() != 3:
        raise ValueError("Axis must have exactly 3 elements.")
    return tensor / torch.linalg.norm(tensor).clamp(min=1.0e-8)


def load_manifest_payload(*, cfg: Any, data: PickGraspData) -> None:
    manifest_path = resolve_data_path(cfg.grasp_label_manifest)
    manifest = load_manifest(manifest_path)
    asset_payload = (manifest.get("assets") or {}).get(cfg.grasp_label_asset)
    if not isinstance(asset_payload, dict):
        raise KeyError(f"Asset '{cfg.grasp_label_asset}' was not found in grasp manifest '{manifest_path}'.")
    npy_rel = str(asset_payload.get("npy_path", "")).strip()
    if not npy_rel:
        raise ValueError(f"Manifest entry for '{cfg.grasp_label_asset}' is missing npy_path.")
    npy_path = resolve_data_path(npy_rel, preferred_root=manifest_path.parent)
    rows = load_grasp_npy(npy_path)
    data.rows = torch.as_tensor(rows, dtype=torch.float32)
    sampler_default = manifest.get("sampler_default")
    data.sampler_default = dict(sampler_default) if isinstance(sampler_default, dict) else {}
    data.shape_metadata = get_asset_shape_metadata(manifest, cfg.grasp_label_asset)
    data.shape_metadata_loaded = True
    ref = str(asset_payload.get("reference_frame", "") or "").strip().lower()
    data.reference_frame = ref or None
    data.articulation = bool(asset_payload.get("articulation", False))
    jt = str(asset_payload.get("joint_type", "") or "").strip().lower()
    data.joint_type = jt or None
    axis_raw = asset_payload.get("joint_axis_local")
    if isinstance(axis_raw, (list, tuple)) and len(axis_raw) == 3:
        data.joint_axis_local = (float(axis_raw[0]), float(axis_raw[1]), float(axis_raw[2]))
    bodies = asset_payload.get("operated_body_names")
    if isinstance(bodies, (list, tuple)):
        data.operated_body_names = tuple(str(b) for b in bodies if str(b).strip())


def robot_base_world(observation: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
    env_idx = int(observation.get("env_idx", 0))
    robot = observation.get("robot")
    data = getattr(robot, "data", None) if robot is not None else None
    root_pos_w = as_torch(getattr(data, "root_pos_w", None)) if data is not None else None
    root_quat_w = as_torch(getattr(data, "root_quat_w", None)) if data is not None else None
    if isinstance(root_pos_w, torch.Tensor) and isinstance(root_quat_w, torch.Tensor):
        if (
            root_pos_w.ndim >= 2
            and root_quat_w.ndim >= 2
            and env_idx < root_pos_w.shape[0]
            and env_idx < root_quat_w.shape[0]
        ):
            return (
                root_pos_w[env_idx, :3].detach().to(dtype=torch.float32),
                root_quat_w[env_idx, :4].detach().to(dtype=torch.float32),
            )
    gym_env = observation.get("gym_env")
    cfg = getattr(gym_env, "cfg", None) if gym_env is not None else None
    arena_env = getattr(cfg, "isaaclab_arena_env", None) if cfg is not None else None
    embodiment = getattr(arena_env, "embodiment", None) if arena_env is not None else None
    initial_pose = getattr(embodiment, "initial_pose", None) if embodiment is not None else None
    if initial_pose is None:
        return None
    return (
        torch.as_tensor(initial_pose.position_xyz, dtype=torch.float32),
        torch.as_tensor(initial_pose.rotation_xyzw, dtype=torch.float32),
    )


def poses_env_to_base(
    *,
    pos_env: torch.Tensor,
    quat_world: torch.Tensor,
    observation: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = pos_env.device
    dtype = torch.float32
    base_world = robot_base_world(observation)
    if base_world is None:
        return pos_env.to(dtype=dtype), quat_world.to(dtype=dtype)
    base_pos_w, base_quat_w = base_world
    base_pos_w = base_pos_w.to(device=device, dtype=dtype)
    base_quat_w = base_quat_w.to(device=device, dtype=dtype)
    env_origin = observation.get("env_origin")
    env_origin_t = (
        env_origin.to(device=device, dtype=dtype)
        if isinstance(env_origin, torch.Tensor)
        else torch.zeros(3, dtype=dtype, device=device)
    )
    pos_world = pos_env.to(device=device, dtype=dtype) + env_origin_t.reshape(1, 3)
    quat_world_t = quat_world.to(device=device, dtype=dtype)
    pos_base_list = []
    quat_base_list = []
    for idx in range(pos_world.shape[0]):
        pos_base_i, quat_base_i = world_to_base_frame(pos_world[idx], quat_world_t[idx], base_pos_w, base_quat_w)
        pos_base_list.append(pos_base_i)
        quat_base_list.append(quat_base_i)
    return torch.stack(pos_base_list, dim=0), torch.stack(quat_base_list, dim=0)


def decode_grasps(
    *,
    cfg: Any,
    data: PickGraspData,
    observation: dict[str, Any],
    scale_points_fn: Any,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    device = observation["current_ee_pos"].device
    if data.rows is None:
        load_manifest_payload(cfg=cfg, data=data)
    assert data.rows is not None
    rows = data.rows.to(device=device)
    if data.scores is None:
        data.scores = rows[:, 0].clone()
    if data.widths is None:
        data.widths = rows[:, 1].clone()
    scores = data.scores.to(device=device)
    widths = data.widths.to(device=device)

    pos_obj = rows[:, 13:16]
    rot_view = rows[:, 4:13].reshape(-1, 3, 3)
    r1_t = torch.as_tensor(R1.T, dtype=torch.float32, device=device).reshape(1, 3, 3)
    rot_local = torch.matmul(rot_view, r1_t.expand(rot_view.shape[0], -1, -1))
    obj_scale = observation.get("object_scale", (1.0, 1.0, 1.0))
    pos_obj_scaled = scale_points_fn(pos_obj, obj_scale)
    data.last_label_pos_obj = pos_obj_scaled

    obj_pos_env, obj_quat = observation["object_pose_env"]
    obj_pos_env = torch.as_tensor(obj_pos_env, dtype=torch.float32, device=device).reshape(3)
    obj_quat = torch.as_tensor(obj_quat, dtype=torch.float32, device=device).reshape(4)
    math_utils = isaaclab_math_utils()
    obj_rot = math_utils.matrix_from_quat(obj_quat.unsqueeze(0)).squeeze(0).to(dtype=torch.float32)
    label_pos_env = obj_pos_env.unsqueeze(0) + torch.einsum("ij,nj->ni", obj_rot, pos_obj_scaled)
    data.last_label_pos_env = label_pos_env

    rot_env_canonical = torch.einsum("ij,njk->nik", obj_rot, rot_local)
    label_quat_env = math_utils.quat_from_matrix(rot_env_canonical).to(dtype=torch.float32)
    data.last_label_quat_env = label_quat_env

    rot_env_goal = rot_env_canonical
    ee_rot = cfg.ee_frame_rotation_wxyz
    r_offset = None
    if ee_rot is not None:
        r_offset = math_utils.matrix_from_quat(
            torch.as_tensor(quat_wxyz_to_xyzw(ee_rot), dtype=torch.float32, device=device).reshape(1, 4)
        ).to(dtype=torch.float32)
        rot_env_goal = torch.einsum("nij,jk->nik", rot_env_canonical, r_offset.squeeze(0))
    grasp_quat_env = math_utils.quat_from_matrix(rot_env_goal).to(dtype=torch.float32)

    # TCP backoff + pre_grasp/retreat offsets follow the *EE-link* approach axis in
    # world (touch/place convention: ``link6 = contact - backoff * approach_world``).
    # Canonical sampler +x matches EE +x on EX001 (identity remap); Franka
    # panda_hand approaches local +z, so use EE +z after ``ee_frame_rotation``.
    ee_approach_local = torch.tensor(
        [0.0, 0.0, 1.0] if ee_rot is not None else [1.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    approach_world = torch.einsum("nij,j->ni", rot_env_goal, ee_approach_local)
    approach_world = approach_world / torch.linalg.norm(approach_world, dim=-1, keepdim=True).clamp(min=1.0e-8)
    backoff = float(cfg.sampler_to_link6_backoff)
    grasp_pos_env = label_pos_env - backoff * approach_world
    grasp_pos_obj = torch.einsum("ij,nj->ni", obj_rot.transpose(0, 1), grasp_pos_env - obj_pos_env.unsqueeze(0))
    data.last_grasp_pos_obj = grasp_pos_obj

    if data.effective_approach_axis is None:
        if cfg.approach_axis_local is not None:
            axis = tuple(float(c) for c in cfg.approach_axis_local)
        else:
            sampler_default = data.sampler_default or {}
            raw = sampler_default.get("gripper_approach_direction")
            axis = tuple(float(c) for c in raw) if raw else (1.0, 0.0, 0.0)
        if len(axis) != 3:
            raise ValueError(f"approach_axis_local must have 3 elements, got: {axis!r}")
        data.effective_approach_axis = axis
    grasp_pos_base, grasp_quat_base = poses_env_to_base(
        pos_env=grasp_pos_env, quat_world=grasp_quat_env, observation=observation
    )
    # Canonical-frame orientation in base frame for the geometric filter. The
    # env->base transform is a left-multiply that commutes with the body-frame
    # remap, so ``rot_base_canonical = rot_base_goal @ R_offset^T`` (no second
    # env->base pass needed).
    if r_offset is not None:
        rot_base_goal = math_utils.matrix_from_quat(grasp_quat_base).to(dtype=torch.float32)
        rot_base_canonical = torch.einsum("nij,kj->nik", rot_base_goal, r_offset.squeeze(0))
        grasp_quat_base_canonical = math_utils.quat_from_matrix(rot_base_canonical).to(dtype=torch.float32)
    else:
        grasp_quat_base_canonical = grasp_quat_base
    return (
        scores,
        widths,
        grasp_pos_env,
        grasp_quat_env,
        approach_world,
        grasp_pos_base,
        grasp_quat_base,
        grasp_quat_base_canonical,
    )
