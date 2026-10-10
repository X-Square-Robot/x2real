"""Motion planning utilities — waypoint interpolation and trajectory builders.

Stateless functions that any planner / skill can import.
Kept separate from planner classes so new tasks can compose sub-skills
(grasp, place, pour, insert …) without inheriting a heavy base.

TODO(curobo): Add a CuroboPlanner-based alternative for each builder
  that produces collision-aware joint-space trajectories.
  See isaaclab_mimic/motion_planners/curobo/curobo_planner.py
"""

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from manaenv.utils.tensor_utils import as_quat_tensor, as_vec3_tensor, squeeze_if_single

__all__ = [
    "InterpolationSettings",
    "interpolate_waypoints",
    "build_grasp_waypoints",
    "build_place_waypoints",
    "base_to_world_frame",
    "world_to_base_frame",
    "sample_arc_poses",
    "as_quat_tensor",
    "as_vec3_tensor",
    "squeeze_if_single",
    "TcpOffsetVec",
    "parse_tcp_offset_vector",
    "combine_tcp_offset_vectors",
    "effective_tcp_offset_from_cfg",
    "tcp_offset_local_tensor",
    "tcp_offset_world_from_rot",
    "ee_pos_from_tcp_target",
    "format_tcp_offset_vector",
]


_as_quat = as_quat_tensor
_as_vec3 = as_vec3_tensor
_squeeze1 = squeeze_if_single


def _isaaclab_math_utils() -> Any:
    return importlib.import_module("isaaclab.utils.math")


# ---------------------------------------------------------------------------
#  Coordinate frame transforms
# ---------------------------------------------------------------------------


def world_to_base_frame(
    pos_world: torch.Tensor,
    quat_world: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform an XYZW pose from world frame to robot base frame."""
    math_utils = _isaaclab_math_utils()
    bq = _as_quat(base_quat)
    base_quat_inv = _squeeze1(math_utils.quat_inv(bq))
    delta = pos_world - base_pos
    v_t = _as_vec3(delta, device=base_quat_inv.device)
    pos_base = _squeeze1(math_utils.quat_apply(base_quat_inv.unsqueeze(0), v_t))
    qw_t = _as_quat(quat_world, device=base_quat_inv.device)
    quat_base = _squeeze1(math_utils.quat_mul(base_quat_inv.unsqueeze(0), qw_t))
    return pos_base, quat_base


def base_to_world_frame(
    pos_base: torch.Tensor,
    quat_base: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform pose from robot base frame to world frame.

    Inverse of :func:`world_to_base_frame`. Used to convert poses produced by
    motion planners (cuRobo FK output is in the robot's URDF base_link frame)
    back into the simulation's world frame.
    """
    math_utils = _isaaclab_math_utils()
    bq = _as_quat(base_quat)
    bp = _as_vec3(base_pos, device=bq.device)
    pb = _as_vec3(pos_base, device=bq.device)
    qb = _as_quat(quat_base, device=bq.device)
    pos_world = bp + _squeeze1(math_utils.quat_apply(bq.unsqueeze(0), pb))
    quat_world = _squeeze1(math_utils.quat_mul(bq.unsqueeze(0), qb.unsqueeze(0)))
    return pos_world, quat_world


def sample_arc_poses(
    pivot: torch.Tensor,
    axis: torch.Tensor,
    start_pos: torch.Tensor,
    start_quat: torch.Tensor,
    sweep_angle: float,
    num_samples: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Sample ``num_samples`` poses rotating about ``axis`` through ``pivot``.

    Used by the trajectory engine to densify revolute (door/knob) slides. The
    grasp orientation co-rotates with the link (same axis-angle as position).
    """
    math_utils = _isaaclab_math_utils()
    pivot_t = _as_vec3(pivot)
    axis_t = _as_vec3(axis)
    axis_t = axis_t / torch.linalg.norm(axis_t, dim=-1, keepdim=True).clamp(min=1.0e-8)
    start_pos_t = _as_vec3(start_pos, device=pivot_t.device)
    start_quat_t = _as_quat(start_quat, device=pivot_t.device)
    n = max(int(num_samples), 2)
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    delta = start_pos_t - pivot_t
    for i in range(n):
        t = float(i) / float(n - 1)
        angle = float(sweep_angle) * t
        angle_t = torch.tensor([angle], device=pivot_t.device, dtype=axis_t.dtype)
        rot_q = math_utils.quat_from_angle_axis(angle_t, axis_t)
        pos_i = pivot_t + math_utils.quat_apply(rot_q, delta)
        quat_i = math_utils.quat_mul(rot_q, start_quat_t)
        out.append((_squeeze1(pos_i), _squeeze1(quat_i)))
    return out


# ---------------------------------------------------------------------------
#  Interpolation
# ---------------------------------------------------------------------------


@dataclass
class InterpolationSettings:
    """Knobs for waypoint → dense action conversion."""

    action_type: str = "ee_rel"
    action_dim: int = 7
    ik_max_position_delta: float = 0.5
    dense_phases: tuple[str, ...] = (
        "approach",
        "descend",
        "lift",
        "move_to_place",
        "descend_to_place",
        "retreat",
        "return_home",
    )
    min_steps_dense: int = 3


def interpolate_waypoints(
    waypoints: list[dict],
    settings: InterpolationSettings,
    device: torch.device | None = None,
) -> tuple[list[dict], list[tuple[torch.Tensor, dict[str, Any]]]]:
    """Convert sparse waypoints into (normalised_waypoints, action_buffer).

    Returns:
        waypoints_out: list of normalised waypoint dicts (pos/quat as tensors).
        action_buffer: list of (action_tensor, info_dict) ready for step().
    """
    if device is None:
        device = torch.device("cpu")

    waypoints_out: list[dict] = []
    for wp in waypoints:
        pos = wp.get("pos")
        quat = wp.get("quat")
        if pos is None or quat is None:
            continue
        waypoints_out.append(
            {
                **wp,
                "pos": torch.as_tensor(pos, dtype=torch.float32, device=device),
                "quat": torch.as_tensor(quat, dtype=torch.float32, device=device),
                "gripper": float(wp.get("gripper", 0.0)),
                "phase": str(wp.get("phase", "")),
            }
        )

    if not waypoints_out:
        return [], []

    math_utils = _isaaclab_math_utils()
    prev_pos = waypoints_out[0]["pos"]
    prev_quat = waypoints_out[0]["quat"]
    action_buffer: list[tuple[torch.Tensor, dict[str, Any]]] = []

    for i in range(len(waypoints_out) - 1):
        ws = waypoints_out[i]
        we = waypoints_out[i + 1]
        distance = torch.linalg.norm(we["pos"] - ws["pos"]).item()
        phase = we.get("phase", "")

        if distance < 1e-5:
            num_steps = 1
        else:
            num_steps = max(1, int(distance / settings.ik_max_position_delta))
            if phase in settings.dense_phases:
                num_steps = max(num_steps, settings.min_steps_dense)

        for j in range(num_steps):
            t = (j + 1) / num_steps
            pos = ws["pos"] * (1 - t) + we["pos"] * t
            q1 = _as_quat(ws["quat"]).squeeze(0)
            q2 = _as_quat(we["quat"], device=q1.device).squeeze(0)
            quat = math_utils.quat_slerp(q1, q2, t)
            gripper = ws["gripper"] * (1 - t) + we["gripper"] * t

            pos_delta = pos - prev_pos
            prev_q = _as_quat(prev_quat, device=quat.device)
            q_diff = _squeeze1(math_utils.quat_mul(quat.unsqueeze(0), math_utils.quat_inv(prev_q)))
            orient_delta = _squeeze1(math_utils.axis_angle_from_quat(q_diff.unsqueeze(0)))
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat.unsqueeze(0))
            target_euler = torch.stack([roll, pitch, yaw], dim=-1).squeeze(0)

            DIM = settings.action_dim
            action = torch.zeros(DIM, dtype=torch.float32, device=device)

            if settings.action_type == "ee_rel":
                action[0:3] = pos_delta
                action[3:6] = orient_delta
                if DIM > 6:
                    action[6] = gripper
            elif settings.action_type == "ee_abs":
                action[0:3] = pos
                action[3:6] = target_euler
                if DIM > 6:
                    action[6] = gripper
            elif settings.action_type == "joint_pos":
                if DIM > 0:
                    action[-1] = gripper
            else:
                action[0:3] = pos_delta
                action[3:6] = orient_delta
                if DIM > 6:
                    action[6] = gripper

            info = {
                "phase": phase,
                "target_pos": pos.detach().cpu().tolist(),
                "target_quat": quat.detach().cpu().tolist(),
                "target_euler": target_euler.detach().cpu().tolist(),
                "prev_pos": prev_pos.detach().cpu().tolist(),
                "prev_quat": prev_quat.detach().cpu().tolist(),
                "pos_delta": pos_delta.detach().cpu().tolist(),
                "orient_delta": orient_delta.detach().cpu().tolist(),
                "gripper": gripper,
                "step": j,
                "total_steps": num_steps,
                "segment": i,
                "t": t,
            }
            action_buffer.append((action, info))
            prev_pos = pos
            prev_quat = quat

    return waypoints_out, action_buffer


# ---------------------------------------------------------------------------
#  Waypoint builders (sub-skill primitives)
# ---------------------------------------------------------------------------


def build_grasp_waypoints(
    grasp_pos: torch.Tensor,
    grasp_quat: torch.Tensor,
    *,
    start_pos: torch.Tensor | None = None,
    start_quat: torch.Tensor | None = None,
    approach_distance: float = 0.05,
    approach_axis_local: torch.Tensor | list[float] | tuple[float, float, float] | None = None,
    lift_height: float = 0.20,
    device: torch.device | None = None,
) -> list[dict]:
    """Build approach → descend → close → lift waypoints for a grasp skill."""
    dev = device or grasp_pos.device
    g_pos = grasp_pos.clone().to(dev)
    g_quat = grasp_quat.clone().to(dev)

    if approach_axis_local is None:
        pre = g_pos + torch.tensor([0, 0, approach_distance], dtype=torch.float32, device=dev)
    else:
        axis_local = torch.as_tensor(approach_axis_local, dtype=torch.float32, device=dev).reshape(-1)
        if axis_local.numel() != 3:
            raise ValueError("approach_axis_local must have exactly 3 elements")
        axis_local = axis_local / torch.linalg.norm(axis_local).clamp(min=1.0e-8)
        math_utils = _isaaclab_math_utils()
        axis_world = _squeeze1(math_utils.quat_apply(_as_quat(g_quat), _as_vec3(axis_local, device=dev)))
        # Offset backwards so the next segment advances forward along the tool axis.
        pre = g_pos - axis_world * float(approach_distance)
    lift = g_pos + torch.tensor([0, 0, lift_height], dtype=torch.float32, device=dev)

    if start_pos is None:
        start_pos = pre.clone()
    else:
        start_pos = start_pos.clone().to(dev)
    if start_quat is None:
        start_quat = g_quat.clone()
    else:
        start_quat = start_quat.clone().to(dev)

    return [
        {"pos": start_pos.clone(), "quat": start_quat.clone(), "gripper": 1.0, "phase": "start"},
        {"pos": pre.clone(), "quat": g_quat.clone(), "gripper": 1.0, "phase": "approach"},
        {"pos": g_pos.clone(), "quat": g_quat.clone(), "gripper": 1.0, "phase": "descend"},
        {"pos": g_pos.clone(), "quat": g_quat.clone(), "gripper": -1.0, "phase": "close"},
        {"pos": lift.clone(), "quat": g_quat.clone(), "gripper": -1.0, "phase": "lift"},
    ]


def build_place_waypoints(
    place_pos: torch.Tensor,
    place_quat: torch.Tensor,
    start_pos: torch.Tensor,
    start_quat: torch.Tensor,
    *,
    approach_distance: float = 0.10,
    retreat_height: float = 0.15,
) -> list[dict]:
    """Build move_to_place → descend → open → retreat waypoints for a place skill."""
    dev = place_pos.device
    pre = place_pos + torch.tensor([0, 0, approach_distance], dtype=torch.float32, device=dev)
    retreat = place_pos + torch.tensor([0, 0, retreat_height], dtype=torch.float32, device=dev)

    return [
        {"pos": start_pos.clone(), "quat": start_quat.clone(), "gripper": -1.0, "phase": "move_to_place_start"},
        {"pos": pre.clone(), "quat": place_quat.clone(), "gripper": -1.0, "phase": "move_to_place"},
        {"pos": place_pos.clone(), "quat": place_quat.clone(), "gripper": -1.0, "phase": "descend_to_place"},
        {"pos": place_pos.clone(), "quat": place_quat.clone(), "gripper": 1.0, "phase": "open_gripper"},
        {"pos": retreat.clone(), "quat": place_quat.clone(), "gripper": 1.0, "phase": "retreat"},
    ]


# ---------------------------------------------------------------------------
# Gripper tcp_offset (EE frame -> fingertip TCP)
# ---------------------------------------------------------------------------

TcpOffsetVec = tuple[float, float, float]


def parse_tcp_offset_vector(value: Any, *, default_axis: int = 0) -> TcpOffsetVec:
    """Parse ``tcp_offset`` / ``tcp_offset_delta`` as an EE-frame 3-vector."""
    if value is None:
        return (0.0, 0.0, 0.0)
    if isinstance(value, (int, float)):
        out = [0.0, 0.0, 0.0]
        if default_axis not in (0, 1, 2):
            raise ValueError(f"default_axis must be 0, 1, or 2, got {default_axis!r}")
        out[default_axis] = float(value)
        return (out[0], out[1], out[2])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq = tuple(value)
        if len(seq) == 1:
            return parse_tcp_offset_vector(seq[0], default_axis=default_axis)
        if len(seq) != 3:
            raise ValueError(f"tcp_offset vector must have 3 elements, got {value!r}")
        return (float(seq[0]), float(seq[1]), float(seq[2]))
    raise TypeError(f"tcp_offset must be a scalar or length-3 sequence, got {type(value)!r}: {value!r}")


def combine_tcp_offset_vectors(base: Any, delta: Any) -> TcpOffsetVec:
    b = parse_tcp_offset_vector(base)
    d = parse_tcp_offset_vector(delta)
    return (b[0] + d[0], b[1] + d[1], b[2] + d[2])


def effective_tcp_offset_from_cfg(cfg: Any) -> TcpOffsetVec:
    base = getattr(cfg, "tcp_offset", (0.0, 0.0, 0.0))
    delta = getattr(cfg, "tcp_offset_delta", (0.0, 0.0, 0.0))
    return combine_tcp_offset_vectors(base, delta)


def tcp_offset_local_tensor(
    offset: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    vec = parse_tcp_offset_vector(offset)
    return torch.tensor(vec, device=device, dtype=dtype).reshape(3)


def tcp_offset_world_from_rot(rot_matrix: torch.Tensor, offset: Any) -> torch.Tensor:
    """World-frame TCP displacement from EE: ``R @ tcp_offset``."""
    local = tcp_offset_local_tensor(offset, device=rot_matrix.device, dtype=rot_matrix.dtype)
    if rot_matrix.ndim == 3:
        return torch.einsum("nij,j->ni", rot_matrix, local)
    return rot_matrix @ local


def ee_pos_from_tcp_target(
    tcp_target: torch.Tensor,
    rot_matrix: torch.Tensor,
    tcp_offset: Any,
) -> torch.Tensor:
    """EE position so the TCP lands at *tcp_target*: ``tcp_target - R @ tcp_offset``."""
    world_off = tcp_offset_world_from_rot(rot_matrix, tcp_offset)
    if tcp_target.ndim == 1:
        return tcp_target - world_off.reshape(3)
    return tcp_target - world_off


def format_tcp_offset_vector(value: Any) -> str:
    x, y, z = parse_tcp_offset_vector(value)
    return f"[{x:.4f}, {y:.4f}, {z:.4f}]"
