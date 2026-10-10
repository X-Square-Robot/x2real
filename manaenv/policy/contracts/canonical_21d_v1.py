"""Mobile whole-body dual-layer action contract (P8B-revise).

Two layers
==========
1. ``mobile_semantic_20d_v1`` — **only** training label / proxy wire
   (``[T, 20]``): EE pos + Euler XYZ + grippers + base
   ``[vx, vy, yaw_rate]`` + lift + head.
2. ``manaenv_ex001_wholebody_env_21d_v1`` — **only** simulator final
   env-action (``[T, 21]``): EE pos + quat XYZW + grippers + wheel rad/s
   + lift + head. Historical name ``canonical_21d_v1`` refers to this
   **env layout only** and must never be used as the proxy wire schema.

Adapter ownership (ManaEnv runtime)
=========================================================
``semantic20_to_env21`` is the **unique** conversion site:

* init-EE-relative pose → root-frame absolute IK target
* Euler XYZ → quaternion XYZW
* ``[vx, vy, yaw_rate]`` → ``[left_wheel, right_wheel]`` rad/s
* ``abs(vy) <= 1e-3`` else fail-closed
* geometry ``r=0.078``, ``track=0.48`` (auto-collection / bench default).
  ``r=0.084``, ``track=0.458`` are experimental sim-effective values.

Forbidden
=========
* Dataset converter emitting final wheel 21D as train labels
* Proxy emitting wheel-action ``[T, 21]``
* Adapter applying Euler/odom conversion twice
* Using ``wheel_command`` as a label source

``WholeBodyEEAbsoluteIKTransform``
(``manaenv/policy/action_transforms.py``) is the current WallX named-response
policy transform (euler→quat + odom→wheel into env 21D). Its kinematics match
the adapter's odom→wheel duty, but WallX named keys ≠ this vector wire. Next
phase may reuse/extract helpers; this module does not modify it.

Isaac-free mapping / schema only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Layer 1 — training / proxy wire
# ---------------------------------------------------------------------------

WIRE_SCHEMA = "mobile_semantic_20d_v1"
WIRE_ACTION_DIM = 20
WIRE_ACTION_CHUNK_SHAPE = ("T", 20)
BASE_REPRESENTATION = "vx_vy_yaw_rate"
EE_ROTATION_REPRESENTATION = "euler_xyz"
# Match the existing public EX001 desktop embodiment token.  The action values
# are root-frame offsets/orientations relative to this reset-captured pose.
EE_REFERENCE = "init_ee_pose"
EULER_ORDER = "euler_xyz"

# ---------------------------------------------------------------------------
# Layer 2 — ManaEnv final env-action (simulator only)
# ---------------------------------------------------------------------------

ENV_ACTION_LAYOUT = "manaenv_ex001_wholebody_env_21d_v1"
ENV_ACTION_DIM = 21
# Historical alias: env layout only — never a proxy wire_schema value.
CANONICAL_21D_V1 = "canonical_21d_v1"
QUAT_ORDER = "xyzw"
ENV_BASE_REPRESENTATION = "wheel_radps_lr"
ENV_EE_ROTATION_REPRESENTATION = "quat_xyzw"

ACTION_MODE = "ik_eef_abs_wholebody"
RUNTIME_ADAPTER = "mobile_semantic20_to_env21_v1"
ADAPTER_OWNER = "manaenv.policy.contracts.canonical_21d_v1.semantic20_to_env21"
EULER_TO_QUAT_OWNER = "manaenv.utils.math_utils.euler_xyz_to_quat_xyzw"
ODOM_TO_WHEEL_OWNER = "manaenv.policy.contracts.canonical_21d_v1.odom_to_wheel_vel"

# Current policy transform layer (WallX named path). Not the pi0.5 vector wire.
WHOLEBODY_TRANSFORM = "manaenv.policy.action_transforms.WholeBodyEEAbsoluteIKTransform"

GRIPPER_VARIANT = "G"
GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 1.89
LIFT_RANGE_M = (0.0, 0.78)
BASE_VY_ABS_MAX = 1e-3
# Align inference bench with mobile_manip auto-collection
# ``compute_wheel_velocity_targets`` defaults (nominal hardware).
# 0.084 / 0.458 are experimental sim-effective parameters; do not use them
# on the bench unless collection is re-run with that geometry.
DIFF_DRIVE_WHEEL_RADIUS_M = 0.078
DIFF_DRIVE_TRACK_WIDTH_M = 0.48

CAMERA_ORDER_SIM = ("left_wrist_cam", "head_cam", "right_wrist_cam")
CAMERA_FILES = ("leftImg.mp4", "faceImg.mp4", "rightImg.mp4")
CAMERA_VIEWS_OPENPI = ("left_wrist_view", "face_view", "right_wrist_view")

TASK_ROOT_INSTRUCTIONS: dict[str, str] = {
    "move_bottle_to_bin": "Move the bottle to the bin",
    "move_fruit_to_basket": "Move the fruit to the basket",
    "put_bottle_on_woodshelf": "Place the bottle on the shelf",
}

REQUIRED_SIDECAR_FILES = (
    "evaluation_log.json",
    "instruction.json",
    "faceImg.mp4",
    "leftImg.mp4",
    "rightImg.mp4",
)
OPTIONAL_SIDECAR_FILES = ("mobile_manip_metadata.json",)

FORBIDDEN_ACTION_LABEL_KEYS = frozenset({"wheel_command"})
FORBIDDEN_WALLX_RESPONSE_KEYS = frozenset({"follow1_pos", "follow2_pos", "velocity_decomposed_odom"})
# Rejected as proxy wire (P8B-fix mistake).
FORBIDDEN_PROXY_WIRE_SCHEMAS = frozenset(
    {
        "canonical_21d_action_chunk_v1",
        "canonical_21d_v1",
        CANONICAL_21D_V1,
        ENV_ACTION_LAYOUT,
    }
)

SEMANTIC_20D_LAYOUT: tuple[dict[str, Any], ...] = (
    {"index": (0, 3), "name": "left_ee_pos", "unit": "m"},
    {"index": (3, 6), "name": "left_ee_euler_xyz", "unit": "rad", "order": EULER_ORDER},
    {"index": (6, 7), "name": "left_gripper", "unit": "joint_rad_like"},
    {"index": (7, 10), "name": "right_ee_pos", "unit": "m"},
    {"index": (10, 13), "name": "right_ee_euler_xyz", "unit": "rad", "order": EULER_ORDER},
    {"index": (13, 14), "name": "right_gripper", "unit": "joint_rad_like"},
    {
        "index": (14, 17),
        "name": "base_vx_vy_yaw",
        "unit": "mps_mps_radps",
        "base_representation": BASE_REPRESENTATION,
    },
    {"index": (17, 18), "name": "lift", "unit": "m"},
    {"index": (18, 20), "name": "head_yaw_pitch", "unit": "rad"},
)

ENV_21D_LAYOUT: tuple[dict[str, Any], ...] = (
    {"index": (0, 3), "name": "left_ee_pos", "unit": "m"},
    {"index": (3, 7), "name": "left_ee_quat_xyzw", "unit": "quat_xyzw"},
    {"index": (7, 8), "name": "left_gripper", "unit": "joint_rad_like"},
    {"index": (8, 11), "name": "right_ee_pos", "unit": "m"},
    {"index": (11, 15), "name": "right_ee_quat_xyzw", "unit": "quat_xyzw"},
    {"index": (15, 16), "name": "right_gripper", "unit": "joint_rad_like"},
    {
        "index": (16, 18),
        "name": "wheel_vel_lr",
        "unit": "rad/s",
        "base_representation": ENV_BASE_REPRESENTATION,
        "note": "Adapter output only; never proxy/train wire",
    },
    {"index": (18, 19), "name": "lift", "unit": "m"},
    {"index": (19, 21), "name": "head_yaw_pitch", "unit": "rad"},
)

MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "task_id",
    "wire_schema",
    "action_dim",
    "action_chunk_shape",
    "action_mode",
    "base_representation",
    "ee_rotation_representation",
    "ee_reference",
    "euler_order",
    "env_action_layout",
    "env_action_dim",
    "camera_order",
    "state_schema",
    "normalization_owner",
    "gripper_variant",
    "dataset_source",
    "dataset_version",
    "root_instruction",
    "runtime_adapter",
)

# Backward-compat names used by older imports/tests in this branch.
ACTION_LAYOUT_VERSION = WIRE_SCHEMA  # train/proxy identity
ACTION_DIM = WIRE_ACTION_DIM


class Canonical21DError(ValueError):
    """Fail-closed contract violation for mobile semantic20 / env21 protocol."""


def semantic_field_names_in_order() -> list[str]:
    return [str(spec["name"]) for spec in SEMANTIC_20D_LAYOUT]


def env_field_names_in_order() -> list[str]:
    return [str(spec["name"]) for spec in ENV_21D_LAYOUT]


def odom_to_wheel_vel(
    odom_vx_vy_yaw: Sequence[float] | np.ndarray,
    *,
    wheel_radius: float = DIFF_DRIVE_WHEEL_RADIUS_M,
    track_width: float = DIFF_DRIVE_TRACK_WIDTH_M,
    vy_abs_max: float = BASE_VY_ABS_MAX,
) -> np.ndarray:
    """Adapter helper: ``[vx, vy, yaw_rate]`` → ``[left, right]`` wheel rad/s.

    Owned by :func:`semantic20_to_env21`. Must not be used to build training
    labels or proxy outputs. Formula matches
    ``WholeBodyEEAbsoluteIKTransform._odom_to_wheel_vel`` kinematics.
    """
    arr = np.asarray(odom_vx_vy_yaw, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise Canonical21DError(f"base command must be length 3, got {arr.shape}")
    vy = float(arr[1])
    if abs(vy) > float(vy_abs_max):
        raise Canonical21DError(f"base.vy={vy!r} exceeds |vy|<={vy_abs_max} fail-closed gate")
    vx = float(arr[0])
    yaw = float(arr[2])
    left = (vx - 0.5 * yaw * track_width) / wheel_radius
    right = (vx + 0.5 * yaw * track_width) / wheel_radius
    return np.asarray([left, right], dtype=np.float32)


def pack_semantic_20d_row(frame: Mapping[str, Any]) -> np.ndarray:
    """Dataset converter: pack 5.1 named fields into ``mobile_semantic_20d_v1``.

    Keeps Euler XYZ and base ``[vx,vy,yaw_rate]`` — does **not** emit env 21D
    wheel actions.
    """
    if not isinstance(frame, Mapping):
        raise Canonical21DError("frame must be a mapping")
    if "base_velocity_command" not in frame:
        if "wheel_command" in frame:
            raise Canonical21DError("wheel_command is forbidden as a training label; require base_velocity_command")
        raise Canonical21DError("missing base_velocity_command")

    row = np.zeros((WIRE_ACTION_DIM,), dtype=np.float32)

    def _vec(key: str, n: int) -> np.ndarray:
        if key not in frame:
            raise Canonical21DError(f"missing required field {key!r}")
        val = np.asarray(frame[key], dtype=np.float64).reshape(-1)
        if val.shape[0] != n:
            raise Canonical21DError(f"{key} must have length {n}, got {val.shape}")
        return val.astype(np.float32)

    row[0:3] = _vec("master_left_position", 3)
    row[3:6] = _vec("master_left_rotation", 3)
    row[6] = float(_vec("master_left_gripper", 1)[0])
    row[7:10] = _vec("master_right_position", 3)
    row[10:13] = _vec("master_right_rotation", 3)
    row[13] = float(_vec("master_right_gripper", 1)[0])
    base = _vec("base_velocity_command", 3)
    # Admit-time vy gate for converter (same threshold as adapter).
    if abs(float(base[1])) > BASE_VY_ABS_MAX:
        raise Canonical21DError(f"base_velocity_command.vy={float(base[1])!r} exceeds |vy|<={BASE_VY_ABS_MAX}")
    row[14:17] = base
    row[17] = float(_vec("lifting_mechanism_command", 1)[0])
    row[18:20] = _vec("head_rotation_command", 2)
    return row


def _init_ee_reference(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
    width: int,
) -> np.ndarray:
    """Validate one reset-captured home EE reference in the root frame."""
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (width,):
        raise Canonical21DError(f"{name} must have shape ({width},), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise Canonical21DError(f"{name} must be finite")
    return arr


def euler_xyz_to_quat_xyzw(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert Euler XYZ through the runtime math implementation on demand."""
    # OD imports this module for wire-schema constants in a Torch-free host
    # interpreter. Keep the existing helper surface without importing the
    # simulator-side math module until conversion is actually requested.
    from manaenv.utils.math_utils import euler_xyz_to_quat_xyzw as convert

    return convert(euler_xyz)


def quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply XYZW quaternions through the runtime implementation on demand."""
    from manaenv.utils.math_utils import quat_multiply_xyzw as multiply

    return multiply(q1, q2)


def semantic20_to_env21(
    semantic_row: Sequence[float] | np.ndarray,
    *,
    left_init_ee_pos: Sequence[float] | np.ndarray,
    left_init_ee_quat_xyzw: Sequence[float] | np.ndarray,
    right_init_ee_pos: Sequence[float] | np.ndarray,
    right_init_ee_quat_xyzw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Unique adapter: ``mobile_semantic_20d_v1`` → ``manaenv_ex001_wholebody_env_21d_v1``.

    The semantic wire's EE translation and Euler XYZ rotation are relative to
    the reset-captured home EE pose in the robot root frame. Therefore the
    caller must supply both arms' home references; omitting them is a protocol
    error, not an invitation to treat a relative command as an absolute IK
    target. Rotation composition matches the existing WallX path exactly:
    ``q_target_root = q_relative * q_home_root``.

    Applies init-EE composition, Euler→XYZW and odom→wheel exactly once.
    """
    arr = np.asarray(semantic_row, dtype=np.float32).reshape(-1)
    if arr.shape[0] != WIRE_ACTION_DIM:
        raise Canonical21DError(f"semantic row must have length {WIRE_ACTION_DIM}, got {arr.shape}")
    left_home_pos = _init_ee_reference(left_init_ee_pos, name="left_init_ee_pos", width=3)
    left_home_quat = _init_ee_reference(left_init_ee_quat_xyzw, name="left_init_ee_quat_xyzw", width=4)
    right_home_pos = _init_ee_reference(right_init_ee_pos, name="right_init_ee_pos", width=3)
    right_home_quat = _init_ee_reference(right_init_ee_quat_xyzw, name="right_init_ee_quat_xyzw", width=4)
    out = np.zeros((ENV_ACTION_DIM,), dtype=np.float32)
    out[0:3] = left_home_pos + arr[0:3]
    out[3:7] = quat_multiply_xyzw(euler_xyz_to_quat_xyzw(arr[3:6]), left_home_quat)
    out[7] = arr[6]
    out[8:11] = right_home_pos + arr[7:10]
    out[11:15] = quat_multiply_xyzw(euler_xyz_to_quat_xyzw(arr[10:13]), right_home_quat)
    out[15] = arr[13]
    out[16:18] = odom_to_wheel_vel(arr[14:17])
    out[18] = arr[17]
    out[19:21] = arr[18:20]
    return out


def reject_wallx_named_response(payload: Mapping[str, Any]) -> None:
    """Fail-closed if a proxy payload uses WallX named response keys."""
    if not isinstance(payload, Mapping):
        raise Canonical21DError("proxy payload must be a mapping")
    present = sorted(FORBIDDEN_WALLX_RESPONSE_KEYS.intersection(payload.keys()))
    if present:
        raise Canonical21DError(
            "proxy wire forbids WallX named response keys "
            f"{present}; require action_chunk [T,{WIRE_ACTION_DIM}] under "
            f"wire_schema={WIRE_SCHEMA!r}"
        )


def validate_proxy_action_chunk(action_chunk: Any) -> np.ndarray:
    """Validate proxy wire as ``[T, 20]`` semantic commands (not env 21D wheels)."""
    arr = np.asarray(action_chunk, dtype=np.float32)
    if arr.ndim != 2:
        raise Canonical21DError(f"proxy action_chunk must be 2D [T,{WIRE_ACTION_DIM}], got shape {arr.shape!r}")
    if arr.shape[-1] == ENV_ACTION_DIM:
        raise Canonical21DError(
            f"proxy wire must be semantic {WIRE_ACTION_DIM}D "
            f"(wire_schema={WIRE_SCHEMA!r}); refusing env-action/wheel "
            f"[T,{ENV_ACTION_DIM}] (env_action_layout={ENV_ACTION_LAYOUT!r})"
        )
    if arr.shape[-1] != WIRE_ACTION_DIM:
        raise Canonical21DError(f"proxy action_chunk must have shape [T,{WIRE_ACTION_DIM}], got {arr.shape!r}")
    if arr.shape[0] < 1:
        raise Canonical21DError("action_chunk T must be >= 1")
    return arr


def reject_proxy_env21_wheel_wire(
    *,
    wire_schema: Any = None,
    action_dim: Any = None,
    base_representation: Any = None,
) -> None:
    """Explicitly reject the revoked P8B-fix 'proxy emits [T,21] wheels' wire."""
    if wire_schema in FORBIDDEN_PROXY_WIRE_SCHEMAS:
        raise Canonical21DError(
            f"wire_schema={wire_schema!r} is not a valid proxy wire; " f"use {WIRE_SCHEMA!r} (semantic 20D)"
        )
    if action_dim is not None:
        try:
            dim = int(action_dim)
        except (TypeError, ValueError) as exc:
            raise Canonical21DError(f"action_dim invalid: {action_dim!r}") from exc
        if dim == ENV_ACTION_DIM:
            raise Canonical21DError(f"proxy action_dim must be {WIRE_ACTION_DIM}, not env {ENV_ACTION_DIM}")
    if base_representation == ENV_BASE_REPRESENTATION:
        raise Canonical21DError(
            f"proxy base_representation must be {BASE_REPRESENTATION!r}, " f"not {ENV_BASE_REPRESENTATION!r}"
        )


def admit_episode(
    evaluation_log: Mapping[str, Any],
    *,
    require_max_score: float | None = 10.0,
) -> None:
    """Admit a training episode: ``success is True`` and ``score == 10``."""
    if not isinstance(evaluation_log, Mapping):
        raise Canonical21DError("evaluation_log must be a mapping")
    success = evaluation_log.get("success")
    if success is not True:
        raise Canonical21DError(f"episode rejected: success must be True, got {success!r}")
    score = evaluation_log.get("score")
    try:
        score_f = float(score)
    except (TypeError, ValueError) as exc:
        raise Canonical21DError(f"episode rejected: score not numeric: {score!r}") from exc
    if abs(score_f - 10.0) > 1e-6:
        raise Canonical21DError(f"episode rejected: score must be 10, got {score_f}")
    if require_max_score is not None:
        max_score = evaluation_log.get("max_possible_score")
        if max_score is not None and abs(float(max_score) - float(require_max_score)) > 1e-6:
            raise Canonical21DError(
                f"episode rejected: max_possible_score must be {require_max_score}, got {max_score!r}"
            )


def extract_root_instruction(instruction_sidecar: Mapping[str, Any]) -> str:
    """Return the single episode root instruction string."""
    if not isinstance(instruction_sidecar, Mapping) or not instruction_sidecar:
        raise Canonical21DError("instruction.json empty or invalid")
    if "instruction" in instruction_sidecar and isinstance(instruction_sidecar["instruction"], str):
        text = instruction_sidecar["instruction"].strip()
        if not text:
            raise Canonical21DError("root instruction empty")
        return text
    texts: list[str] = []
    for value in instruction_sidecar.values():
        if isinstance(value, Mapping) and isinstance(value.get("instruction"), str):
            texts.append(value["instruction"].strip())
    texts = [t for t in texts if t]
    if len(texts) != 1:
        raise Canonical21DError(f"expected exactly one root instruction, got {len(texts)}")
    return texts[0]


def stage_metadata_from_intervals(stage_intervals: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize stage intervals for curriculum/diagnostics (not policy wire)."""
    if not stage_intervals:
        return {"stages": [], "intervals": {}}
    intervals: dict[str, dict[str, int]] = {}
    for name, spec in stage_intervals.items():
        if not isinstance(spec, Mapping):
            raise Canonical21DError(f"stage_intervals[{name!r}] must be a mapping")
        try:
            intervals[str(name)] = {
                "start": int(spec["start"]),
                "end": int(spec["end"]),
                "first_hit_frame": int(spec["first_hit_frame"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise Canonical21DError(f"invalid stage interval for {name!r}: {spec!r}") from exc
    return {"stages": list(intervals.keys()), "intervals": intervals}


def _normalize_chunk_shape(value: Any, *, expected_dim: int) -> tuple[Any, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise Canonical21DError(f"action_chunk_shape must be [T,{expected_dim}], got {value!r}")
    t_dim, d_dim = value[0], value[1]
    try:
        d_int = int(d_dim)
    except (TypeError, ValueError) as exc:
        raise Canonical21DError(f"action_chunk_shape[-1] invalid: {d_dim!r}") from exc
    if d_int != expected_dim:
        raise Canonical21DError(f"action_chunk_shape dim mismatch: expected [T,{expected_dim}], got {value!r}")
    if t_dim not in ("T", "t", None) and not (isinstance(t_dim, int) and t_dim >= 1):
        raise Canonical21DError(f"action_chunk_shape[0] must be 'T' or positive int, got {t_dim!r}")
    return (t_dim, d_int)


def validate_checkpoint_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail-closed: train/proxy wire 20D + env-action 21D fields must agree."""
    if not isinstance(manifest, Mapping):
        raise Canonical21DError("manifest must be a mapping")
    missing = [k for k in MANIFEST_REQUIRED_FIELDS if k not in manifest]
    if missing:
        raise Canonical21DError(f"manifest missing required fields: {missing}")

    reject_proxy_env21_wheel_wire(
        wire_schema=manifest.get("wire_schema"),
        action_dim=manifest.get("action_dim"),
        base_representation=manifest.get("base_representation"),
    )

    if manifest.get("wire_schema") != WIRE_SCHEMA:
        raise Canonical21DError(f"wire_schema mismatch: expected {WIRE_SCHEMA!r}, got {manifest.get('wire_schema')!r}")
    try:
        dim = int(manifest["action_dim"])
    except (TypeError, ValueError) as exc:
        raise Canonical21DError(f"action_dim invalid: {manifest.get('action_dim')!r}") from exc
    if dim != WIRE_ACTION_DIM:
        raise Canonical21DError(f"action_dim mismatch: expected {WIRE_ACTION_DIM}, got {dim}")

    _normalize_chunk_shape(manifest.get("action_chunk_shape"), expected_dim=WIRE_ACTION_DIM)

    if manifest.get("base_representation") != BASE_REPRESENTATION:
        raise Canonical21DError(
            f"base_representation mismatch: expected {BASE_REPRESENTATION!r}, "
            f"got {manifest.get('base_representation')!r}"
        )
    if manifest.get("ee_rotation_representation") != EE_ROTATION_REPRESENTATION:
        raise Canonical21DError(
            f"ee_rotation_representation mismatch: expected {EE_ROTATION_REPRESENTATION!r}, "
            f"got {manifest.get('ee_rotation_representation')!r}"
        )
    if manifest.get("ee_reference") != EE_REFERENCE:
        raise Canonical21DError(
            f"ee_reference mismatch: expected {EE_REFERENCE!r}, got {manifest.get('ee_reference')!r}"
        )
    if manifest.get("euler_order") != EULER_ORDER:
        raise Canonical21DError(f"euler_order mismatch: expected {EULER_ORDER!r}, got {manifest.get('euler_order')!r}")
    if manifest.get("env_action_layout") != ENV_ACTION_LAYOUT:
        raise Canonical21DError(
            f"env_action_layout mismatch: expected {ENV_ACTION_LAYOUT!r}, " f"got {manifest.get('env_action_layout')!r}"
        )
    try:
        env_dim = int(manifest["env_action_dim"])
    except (TypeError, ValueError) as exc:
        raise Canonical21DError(f"env_action_dim invalid: {manifest.get('env_action_dim')!r}") from exc
    if env_dim != ENV_ACTION_DIM:
        raise Canonical21DError(f"env_action_dim mismatch: expected {ENV_ACTION_DIM}, got {env_dim}")

    if manifest.get("runtime_adapter") != RUNTIME_ADAPTER:
        raise Canonical21DError(
            f"runtime_adapter mismatch: expected {RUNTIME_ADAPTER!r}, " f"got {manifest.get('runtime_adapter')!r}"
        )
    if manifest.get("action_mode") != ACTION_MODE:
        raise Canonical21DError(f"action_mode mismatch: expected {ACTION_MODE!r}, got {manifest.get('action_mode')!r}")
    if manifest.get("gripper_variant") != GRIPPER_VARIANT:
        raise Canonical21DError(
            f"gripper_variant mismatch: expected {GRIPPER_VARIANT!r}, " f"got {manifest.get('gripper_variant')!r}"
        )

    task_id = str(manifest.get("task_id") or "")
    if task_id not in TASK_ROOT_INSTRUCTIONS:
        raise Canonical21DError(f"task_id must be one of {sorted(TASK_ROOT_INSTRUCTIONS)}, got {task_id!r}")
    expected_instr = TASK_ROOT_INSTRUCTIONS[task_id]
    if str(manifest.get("root_instruction") or "").strip() != expected_instr:
        raise Canonical21DError(
            f"root_instruction mismatch for {task_id}: expected {expected_instr!r}, "
            f"got {manifest.get('root_instruction')!r}"
        )

    for banned in ("endpoint", "endpoint_url", "server_address", "policy_space_endpoint"):
        if banned in manifest and manifest[banned]:
            raise Canonical21DError(f"manifest must not embed serving endpoint field {banned!r}")


def example_manifest(task_id: str, *, dataset_source: str, dataset_version: str) -> dict[str, Any]:
    """Build a minimal valid train/proxy checkpoint manifest."""
    if task_id not in TASK_ROOT_INSTRUCTIONS:
        raise Canonical21DError(f"unknown task_id {task_id!r}")
    return {
        "task_id": task_id,
        "wire_schema": WIRE_SCHEMA,
        "action_dim": WIRE_ACTION_DIM,
        "action_chunk_shape": list(WIRE_ACTION_CHUNK_SHAPE),
        "action_mode": ACTION_MODE,
        "base_representation": BASE_REPRESENTATION,
        "ee_rotation_representation": EE_ROTATION_REPRESENTATION,
        "ee_reference": EE_REFERENCE,
        "euler_order": EULER_ORDER,
        "env_action_layout": ENV_ACTION_LAYOUT,
        "env_action_dim": ENV_ACTION_DIM,
        "camera_order": list(CAMERA_VIEWS_OPENPI),
        "state_schema": {
            "follow1_pos": 7,
            "follow2_pos": 7,
            "velocity_decomposed_odom": 3,
            "lift": 1,
            "head_pos": 2,
        },
        "normalization_owner": "openpi_norm_stats",
        "gripper_variant": GRIPPER_VARIANT,
        "dataset_source": dataset_source,
        "dataset_version": dataset_version,
        "root_instruction": TASK_ROOT_INSTRUCTIONS[task_id],
        "runtime_adapter": RUNTIME_ADAPTER,
        "adapter_owner": ADAPTER_OWNER,
        "euler_to_quat_owner": EULER_TO_QUAT_OWNER,
        "odom_to_wheel_owner": ODOM_TO_WHEEL_OWNER,
        "wholebody_transform_reference": WHOLEBODY_TRANSFORM,
        "canonical_21d_v1_means": "env_action_layout_only_not_proxy_wire",
        "diff_drive": {
            "wheel_radius": DIFF_DRIVE_WHEEL_RADIUS_M,
            "wheel_track_width": DIFF_DRIVE_TRACK_WIDTH_M,
        },
        "base_vy_abs_max": BASE_VY_ABS_MAX,
        "forbidden_label_keys": sorted(FORBIDDEN_ACTION_LABEL_KEYS),
        "forbidden_wallx_response_keys": sorted(FORBIDDEN_WALLX_RESPONSE_KEYS),
    }


# Deprecated names kept only to fail loudly if old P8B-fix call sites linger.
def pack_canonical_21d_row(*_args: Any, **_kwargs: Any) -> np.ndarray:
    raise Canonical21DError(
        "pack_canonical_21d_row revoked: dataset converter must emit "
        f"{WIRE_SCHEMA} via pack_semantic_20d_row; env 21D is adapter-only"
    )


def validate_action_chunk(*_args: Any, **_kwargs: Any) -> np.ndarray:
    raise Canonical21DError(
        "validate_action_chunk revoked for env-21 passthrough; use " "validate_proxy_action_chunk for semantic [T,20]"
    )


__all__ = [
    "ACTION_DIM",
    "ACTION_LAYOUT_VERSION",
    "ACTION_MODE",
    "ADAPTER_OWNER",
    "BASE_REPRESENTATION",
    "BASE_VY_ABS_MAX",
    "CAMERA_FILES",
    "CAMERA_ORDER_SIM",
    "CAMERA_VIEWS_OPENPI",
    "CANONICAL_21D_V1",
    "Canonical21DError",
    "DIFF_DRIVE_TRACK_WIDTH_M",
    "DIFF_DRIVE_WHEEL_RADIUS_M",
    "EE_REFERENCE",
    "EE_ROTATION_REPRESENTATION",
    "ENV_21D_LAYOUT",
    "ENV_ACTION_DIM",
    "ENV_ACTION_LAYOUT",
    "ENV_BASE_REPRESENTATION",
    "ENV_EE_ROTATION_REPRESENTATION",
    "EULER_ORDER",
    "EULER_TO_QUAT_OWNER",
    "FORBIDDEN_ACTION_LABEL_KEYS",
    "FORBIDDEN_PROXY_WIRE_SCHEMAS",
    "FORBIDDEN_WALLX_RESPONSE_KEYS",
    "GRIPPER_CLOSED",
    "GRIPPER_OPEN",
    "GRIPPER_VARIANT",
    "LIFT_RANGE_M",
    "MANIFEST_REQUIRED_FIELDS",
    "ODOM_TO_WHEEL_OWNER",
    "OPTIONAL_SIDECAR_FILES",
    "QUAT_ORDER",
    "REQUIRED_SIDECAR_FILES",
    "RUNTIME_ADAPTER",
    "SEMANTIC_20D_LAYOUT",
    "TASK_ROOT_INSTRUCTIONS",
    "WHOLEBODY_TRANSFORM",
    "WIRE_ACTION_CHUNK_SHAPE",
    "WIRE_ACTION_DIM",
    "WIRE_SCHEMA",
    "admit_episode",
    "env_field_names_in_order",
    "example_manifest",
    "extract_root_instruction",
    "odom_to_wheel_vel",
    "pack_canonical_21d_row",
    "pack_semantic_20d_row",
    "reject_proxy_env21_wheel_wire",
    "reject_wallx_named_response",
    "semantic20_to_env21",
    "semantic_field_names_in_order",
    "stage_metadata_from_intervals",
    "validate_action_chunk",
    "validate_checkpoint_manifest",
    "validate_proxy_action_chunk",
]
