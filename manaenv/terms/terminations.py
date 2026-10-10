"""Termination functions for ManaEnv.

All functions follow the IsaacLab ``TerminationTermCfg`` signature::

    func(env: ManagerBasedRLEnv, **params) -> torch.Tensor

CfgBuilders import what they need by name.  Adding a new termination
is as simple as defining a new function here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from manaenv.pipeline.runtime.log_store import read_log
from manaenv.terms import checker
from manaenv.terms.grasp_target_selection import EPISODE_GRASP_TARGET_ATTR
from manaenv.utils.asset_utils import (
    collapse_single_item,
    pool_cache_slot_name,
    resolve_active_scene_entity_name,
    resolve_pool_cache_size,
    scene_has_asset,
)
from manaenv.utils.debug_print import manaprint
from manaenv.utils.math_utils import euler_xyz_from_quat_xyzw_tensor
from manaenv.utils.tensor_utils import as_torch, to_numpy_keep_shape

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def _resolve_name(cfg) -> str:
    """Return the scene key from a SceneEntityCfg or a plain string."""
    return cfg if isinstance(cfg, str) else cfg.name


def _false_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _resolve_step_dt_s(env: ManagerBasedRLEnv, runtime: Any | None = None) -> float:
    """Resolve one control/evaluation step duration for time-window terms.

    Prefer an injected evaluation/score runtime when it already resolved a
    positive ``step_dt_s``. Otherwise fall back through env/cfg/scene so
    DAG-bound stage predicates (``_bind_child_predicate`` only passes YAML
    params) still work without runtime injection.
    """
    step_dt = getattr(runtime, "step_dt_s", 0.0) if runtime is not None else 0.0
    if isinstance(step_dt, (int, float)) and step_dt > 0:
        return float(step_dt)

    for attr in ("step_dt", "physics_dt"):
        value = getattr(env, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)

    cfg = getattr(env, "cfg", None)
    sim = getattr(cfg, "sim", None) if cfg is not None else None
    sim_dt = getattr(sim, "dt", None) if sim is not None else None
    decimation = getattr(cfg, "decimation", None) if cfg is not None else None
    if isinstance(sim_dt, (int, float)) and sim_dt > 0:
        if isinstance(decimation, (int, float)) and decimation > 0:
            return float(sim_dt) * float(decimation)
        return float(sim_dt)

    scene = getattr(env, "scene", None)
    sim_fps = getattr(scene, "sim_fps", None)
    simulation_decimation = getattr(scene, "simulation_decimation", None)
    if sim_fps and simulation_decimation:
        return float(simulation_decimation) / float(sim_fps)
    render_decremental = getattr(scene, "render_decremental", None)
    if sim_fps and render_decremental:
        return float(render_decremental) / float(sim_fps)
    return 0.0


def _stable_term_counter_id(*parts: str) -> str:
    """Short deterministic id for per-term counters on ``env`` (valid attribute fragment)."""
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def _as_batched_tensor(values: list[float], like: torch.Tensor) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=like.dtype, device=like.device)
    return tensor.unsqueeze(0).expand(like.shape[0], -1)


def _get_root_pose_env(
    env: ManagerBasedRLEnv,
    cfg: SceneEntityCfg | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    asset = env.scene[_resolve_name(cfg)]
    if hasattr(asset, "data") and hasattr(asset.data, "root_pos_w") and hasattr(asset.data, "root_quat_w"):
        return as_torch(asset.data.root_pos_w) - env.scene.env_origins, as_torch(asset.data.root_quat_w)

    if hasattr(asset, "get_world_poses"):
        pos_w, quat_w = asset.get_world_poses()
        pos_w = torch.as_tensor(as_torch(pos_w), dtype=torch.float32, device=env.device)
        quat_w = torch.as_tensor(as_torch(quat_w), dtype=torch.float32, device=env.device)
        return pos_w - env.scene.env_origins, quat_w

    raise AttributeError(f"Scene asset '{_resolve_name(cfg)}' does not expose a root pose.")


def _get_velocity_norm(asset: Any) -> torch.Tensor:
    if hasattr(asset, "data") and hasattr(asset.data, "root_lin_vel_w"):
        return torch.norm(as_torch(asset.data.root_lin_vel_w), dim=-1)
    return torch.zeros(1, dtype=torch.float32)


def _velocity_below_threshold(
    asset: Any,
    velocity_threshold: float | None,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    if velocity_threshold is None:
        return torch.ones(num_envs, dtype=torch.bool, device=device)
    if hasattr(asset, "data") and hasattr(asset.data, "root_lin_vel_w"):
        return torch.norm(as_torch(asset.data.root_lin_vel_w), dim=-1) < velocity_threshold
    return torch.ones(num_envs, dtype=torch.bool, device=device)


def _contact_sensor_active(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg | str | None,
    force_threshold: float,
) -> torch.Tensor:
    if contact_sensor_cfg is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    sensor = env.scene[_resolve_name(contact_sensor_cfg)]
    force_matrix = as_torch(sensor.data.force_matrix_w)
    if force_matrix is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    force_norm = torch.norm(force_matrix, dim=-1)
    force_per_env = torch.amax(force_norm, dim=(1, 2))
    return force_per_env > force_threshold


def _episode_reset_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    reset_mask = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    if hasattr(env, "episode_length_buf"):
        reset_mask = reset_mask | (env.episode_length_buf == 0)
    if hasattr(env, "reset_buf"):
        reset_mask = reset_mask | (env.reset_buf > 0)
    return reset_mask


def _register_tensor_episode_state(
    env: ManagerBasedRLEnv,
    state_attr: str,
    *,
    fill_value: bool | int | float = 0,
) -> None:
    """Register a term-owned per-episode tensor buffer for reset.

    Terms lazily create per-episode buffers on the env via ``setattr``; this
    registers a generic reset slot in the unified :class:`RuntimeStateRegistry`
    keyed by the buffer's attribute name (idempotent — a term re-registers the
    same name each eval). Reset re-reads ``getattr(env, state_attr)`` so it
    tolerates the buffer being (re)created after registration.
    """
    from manaenv.pipeline.runtime.runtime_state import runtime_state

    def _reset(ids: torch.Tensor) -> None:
        value = getattr(env, state_attr, None)
        if not isinstance(value, torch.Tensor):
            return
        if value.ndim == 0:
            value.fill_(fill_value)
            return
        if value.shape[0] == env.num_envs:
            value[ids] = fill_value

    runtime_state(env).register(state_attr, reset=_reset, owner="termination")


def reset_stateful_termination_terms(env: ManagerBasedRLEnv, env_ids: Any | None = None) -> tuple[str, ...]:
    """Clear registered per-episode runtime state for reset envs.

    A reset boundary (the collection loop's auto-reset and the planner-driven
    single-env reset) calls this so stateful terms' buffers are cleared even
    when the first post-reset term evaluation happens several planner steps in.
    Delegates to the unified :class:`RuntimeStateRegistry`, so this now clears
    every registered slot (termination buffers and any other owner) in one place.
    """
    from manaenv.pipeline.runtime.runtime_state import runtime_state

    return runtime_state(env).reset(env_ids)


PLANNER_SEQUENCE_DONE_ATTR = "_mana_planner_sequence_done"


def planner_sequence_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Task-owned termination: end the episode when the scene-mutating planner has
    exhausted its scripted sequence.

    This is the "Task owns termination" boundary: a scripted planner (e.g.
    :class:`SkillTrajectoryPlanner`) sets a per-env *mechanical* flag
    (``env._mana_planner_sequence_done[env_idx] = True``) once it has nothing left
    to do — it does NOT decide success and does NOT reset the env itself. The Task
    reads that flag here and ends the episode through the normal termination
    manager (``ENV_DONES``), so demo export and per-stage eval finalize on the same
    boundary. Inert for actors that never set the flag (e.g. a learned
    ``PolicyPlanner`` during benchmark), whose episodes end on success/timeout.
    """
    flag = getattr(env, PLANNER_SEQUENCE_DONE_ATTR, None)
    if not isinstance(flag, torch.Tensor) or flag.shape[0] != env.num_envs:
        flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, PLANNER_SEQUENCE_DONE_ATTR, flag)
        _register_tensor_episode_state(env, PLANNER_SEQUENCE_DONE_ATTR, fill_value=False)
    return flag


def _current_contact_flags(
    env: ManagerBasedRLEnv,
    *,
    contact_sensor_cfgs: list[Any],
    force_threshold: float,
) -> torch.Tensor:
    num_envs = env.num_envs
    num_contacts = len(contact_sensor_cfgs)
    contact_flags = torch.zeros((num_envs, num_contacts), dtype=torch.bool, device=env.device)

    for index, sensor_cfg in enumerate(contact_sensor_cfgs):
        sensor_name = _resolve_name(collapse_single_item(sensor_cfg))
        sensor = env.scene[sensor_name]
        force_matrix = as_torch(sensor.data.force_matrix_w)
        if force_matrix is None:
            continue

        force_norm = torch.norm(force_matrix, dim=-1)
        force_per_env = torch.amax(force_norm, dim=(1, 2))
        contact_flags[:, index] = force_per_env > float(force_threshold)

    return contact_flags


def _latched_contact_flags(
    env: ManagerBasedRLEnv,
    *,
    contact_sensor_cfgs: list[Any],
    force_threshold: float,
    state_attr: str,
) -> torch.Tensor:
    num_envs = env.num_envs
    num_contacts = len(contact_sensor_cfgs)

    if (not hasattr(env, state_attr)) or (getattr(env, state_attr).shape != (num_envs, num_contacts)):
        setattr(
            env,
            state_attr,
            torch.zeros((num_envs, num_contacts), dtype=torch.bool, device=env.device),
        )
    _register_tensor_episode_state(env, state_attr, fill_value=False)

    latched_flags: torch.Tensor = getattr(env, state_attr)
    reset_mask = _episode_reset_mask(env)
    if torch.any(reset_mask):
        latched_flags[reset_mask] = False

    latched_flags |= _current_contact_flags(
        env,
        contact_sensor_cfgs=contact_sensor_cfgs,
        force_threshold=force_threshold,
    )
    return latched_flags


def _ordered_contact_progress(
    env: ManagerBasedRLEnv,
    *,
    contact_sensor_cfgs: list[Any],
    force_threshold: float,
    state_attr: str,
) -> torch.Tensor:
    num_envs = env.num_envs
    num_contacts = len(contact_sensor_cfgs)

    if (not hasattr(env, state_attr)) or (getattr(env, state_attr).shape != (num_envs,)):
        setattr(
            env,
            state_attr,
            torch.zeros((num_envs,), dtype=torch.int64, device=env.device),
        )
    _register_tensor_episode_state(env, state_attr, fill_value=0)

    ordered_progress: torch.Tensor = getattr(env, state_attr)
    reset_mask = _episode_reset_mask(env)
    if torch.any(reset_mask):
        ordered_progress[reset_mask] = 0

    contact_flags = _current_contact_flags(
        env,
        contact_sensor_cfgs=contact_sensor_cfgs,
        force_threshold=force_threshold,
    )
    for contact_index in range(num_contacts):
        expect_this_contact = ordered_progress == contact_index
        ordered_progress[expect_this_contact & contact_flags[:, contact_index]] += 1

    return ordered_progress


_BUTTON_COLOR_TOKENS: dict[str, tuple[str, ...]] = {
    "blue": ("button_blue", "blue"),
    "green": ("button_green", "green"),
    "pink": ("button_pink", "pink"),
    "yellow": ("button_yellow", "yellow"),
}


def _button_order_from_instruction(instruction: str) -> list[str]:
    """Extract the button color order from task text."""
    text = str(instruction or "").strip().lower()
    ordered: list[tuple[int, str]] = []
    for color, tokens in _BUTTON_COLOR_TOKENS.items():
        token_positions = [text.find(token.lower()) for token in tokens if text.find(token.lower()) >= 0]
        if token_positions:
            ordered.append((min(token_positions), color))
    ordered.sort(key=lambda item: item[0])
    return [color for _, color in ordered]


def _sensor_color_name(sensor_cfg: Any) -> str:
    name = _resolve_name(collapse_single_item(sensor_cfg)).lower()
    for color, tokens in _BUTTON_COLOR_TOKENS.items():
        if any(token.lower() in name for token in tokens if token.startswith("button_") or token.isascii()):
            return color
    return ""


def _match_sensor_by_substring(
    token: str,
    contact_sensor_cfgs: list[Any],
) -> Any:
    """Find exactly one sensor whose resolved name contains *token* (case-insensitive)."""
    token_lower = token.strip().lower()
    matches = [cfg for cfg in contact_sensor_cfgs if token_lower in _resolve_name(collapse_single_item(cfg)).lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        available = [_resolve_name(collapse_single_item(c)) for c in contact_sensor_cfgs]
        raise ValueError(f"sequence_order token {token!r} matched no sensor. Available: {available}")
    matched_names = [_resolve_name(collapse_single_item(c)) for c in matches]
    raise ValueError(f"sequence_order token {token!r} is ambiguous — matched {len(matches)} sensors: {matched_names}")


def _ordered_contact_sensor_cfgs(
    contact_sensor_cfgs: list[Any],
    *,
    sequence_order: list[str] | None = None,
    button_order: list[str] | None = None,
    instruction: str | None = None,
    order_from_instruction: bool = False,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> list[Any]:
    if sequence_order:
        return [_match_sensor_by_substring(token, contact_sensor_cfgs) for token in sequence_order]

    if not order_from_instruction and not button_order:
        return contact_sensor_cfgs

    resolved_order = [str(item).strip().lower().replace("button_", "") for item in (button_order or []) if item]
    if order_from_instruction:
        source_instruction = instruction
        runtime = evaluation_runtime if evaluation_runtime is not None else score_runtime
        if source_instruction is None and runtime is not None:
            env_obj = getattr(runtime, "env_obj", None)
            task = getattr(env_obj, "task", None) if env_obj is not None else None
            get_prompt = getattr(task, "get_prompt", None)
            if callable(get_prompt):
                source_instruction = get_prompt()
        if source_instruction is None:
            source_instruction = (
                getattr(evaluation_runtime, "instruction", "") if evaluation_runtime is not None else ""
            )
        parsed_order = _button_order_from_instruction(str(source_instruction or ""))
        if parsed_order:
            resolved_order = parsed_order

    if not resolved_order:
        return contact_sensor_cfgs

    by_color = {_sensor_color_name(sensor_cfg): sensor_cfg for sensor_cfg in contact_sensor_cfgs}
    reordered = [by_color[color] for color in resolved_order if color in by_color]
    if len(reordered) != len(resolved_order):
        return contact_sensor_cfgs
    return reordered


def _resolve_joint_position(
    articulation: Articulation,
    joint_name: str | None,
) -> torch.Tensor | None:
    if joint_name is None:
        return None

    try:
        joint_ids, _ = articulation.find_joints(joint_name)
    except ValueError:
        if getattr(articulation, "joint_names", None):
            joint_ids = [0]
        else:
            return None

    if not joint_ids:
        return None
    return articulation.data.joint_pos.torch[:, joint_ids[0]]


def _resolve_site_pose_env(
    env: ManagerBasedRLEnv,
    site_spec: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from isaaclab.utils import math as math_utils

    frame = site_spec.get("frame", "asset")
    pos_cfg = site_spec.get("pos", [0.0, 0.0, 0.0])
    quat_cfg = site_spec.get("quat", [0.0, 0.0, 0.0, 1.0])
    size_cfg = site_spec.get("size", [0.0, 0.0, 0.0])

    if frame == "env":
        like = env.scene.env_origins
        site_pos = _as_batched_tensor(pos_cfg, like)
        site_quat = _as_batched_tensor(quat_cfg, like)
        site_size = _as_batched_tensor(size_cfg, like)
        return site_pos, site_quat, site_size

    parent_cfg = site_spec.get("parent_cfg")
    if parent_cfg is None:
        raise ValueError(f"Asset-frame site spec missing parent_cfg: {site_spec}")

    parent_pos, parent_quat = _get_root_pose_env(env, parent_cfg)
    local_pos = _as_batched_tensor(pos_cfg, parent_pos)
    local_quat = _as_batched_tensor(quat_cfg, parent_quat)
    parent_asset = env.scene[_resolve_name(parent_cfg)]

    joint_names = site_spec.get("joint_names", [])
    joint_axes = site_spec.get("joint_axes", [])
    joint_types = site_spec.get("joint_types", [])
    if (
        hasattr(parent_asset, "find_joints")
        and hasattr(getattr(parent_asset, "data", None), "joint_pos")
        and joint_names
    ):
        for joint_name, joint_axis, joint_type in zip(joint_names, joint_axes, joint_types):
            joint_pos = _resolve_joint_position(parent_asset, joint_name)
            if joint_pos is None or joint_type != "slide":
                continue
            axis_tensor = _as_batched_tensor(joint_axis, parent_pos)
            local_pos = local_pos + axis_tensor * joint_pos.unsqueeze(-1)

    site_pos = parent_pos + math_utils.quat_apply(parent_quat, local_pos)
    site_quat = math_utils.quat_mul(parent_quat, local_quat)
    site_size = _as_batched_tensor(size_cfg, parent_pos)
    return site_pos, site_quat, site_size


# =====================================================================
# Single-object terminations
# =====================================================================


def success_placeholder(env: ManagerBasedRLEnv) -> torch.Tensor:
    """No-op success (always False). Kept for backward compatibility."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def merged(
    env: ManagerBasedRLEnv,
    _sub_term_defs: list[dict],
    mode: str = "and",
) -> torch.Tensor:
    """Aggregate inline sub-terms via the shared AND/OR/NOT checker algebra.

    Thin adapter over :func:`manaenv.terms.checker.combine` (the single
    composition primitive) — unwraps the ``{cfg, combine_mode}`` carriers and
    delegates the fold. ``mode`` may be ``and`` / ``or`` / ``not`` (NOT takes a
    single sub-term). Used as the shared implementation for :func:`success`,
    :func:`evaluation`, and available directly via ``_func: "merged"``.
    """
    if not _sub_term_defs:
        return _false_tensor(env)
    children = [d["cfg"] if isinstance(d, dict) else d for d in _sub_term_defs]
    return checker.combine(env, op=mode, children=children)


def consecutive_merged(
    env: ManagerBasedRLEnv,
    _sub_term_defs: list[dict],
    mode: str = "and",
    consecutive_steps: int = 1,
    state_key: str = "default",
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL multi-object] Hold a checker expression true for consecutive calls.

    The filter wraps the complete boolean expression rather than its children,
    so ``Home AND final state`` is one atomic stability window.  DAG and
    always-running evaluation calls use separate counters to prevent scoring
    from pre-warming a gated sink before that sink becomes active.
    """
    raw = merged(env, _sub_term_defs, mode=mode)
    if consecutive_steps <= 1:
        return raw
    runtime_kind = "evaluation" if evaluation_runtime is not None or score_runtime is not None else "dag"
    counter_attr = "_mana_consecutive_merged_" + _stable_term_counter_id(runtime_kind, str(state_key), str(mode))
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def success(
    env: ManagerBasedRLEnv,
    _sub_term_defs: list[dict],
    mode: str = "and",
) -> torch.Tensor:
    """Build the canonical task ``success`` signal from inline sub-terms."""
    return merged(env, _sub_term_defs, mode=mode)


def evaluation(
    env: ManagerBasedRLEnv,
    _sub_term_defs: list[dict] | None = None,
    mode: str = "or",
) -> torch.Tensor:
    """Episode-evaluation aggregator — same compile path as ``success``.

    The returned tensor is never consumed by the ``TerminationManager``;
    ``TerminationComponent.generate`` extracts the cfg by name and routes
    it to :class:`~manaenv.pipeline.evaluation.runtime.EvaluationRuntime`.
    """
    return merged(env, _sub_term_defs or [], mode=mode)


def latched(
    env: ManagerBasedRLEnv,
    _sub_term_defs: list[dict],
    state_key: str,
) -> torch.Tensor:
    """Latch (first-hit-and-hold) wrapper over a single momentary child term.

    The author-facing ``_latch: true`` compiles to this: it evaluates the wrapped
    child's *momentary* signal and holds it true for the rest of the episode
    (``history |= raw``), turning any term (a leaf or a whole ``_combine`` group)
    from "true only while holding" into "true once hit". The child rides the
    standard ``_sub_term_defs`` channel (always exactly one entry here), so
    :meth:`TerminationComponent.expand_runtime_term_cfg` does the child's
    ``@role`` list expansion + combine unchanged and this func only adds the hold.

    ``state_key`` is a compile-time uuid that — copied by value through configclass
    deepcopy — keys a single per-env buffer across all cloned copies (mirrors
    ``DagSuccessMachine._state_uuid``). State lives on ``env`` and self-resets at
    the episode boundary (``episode_length_buf == 0`` / ``reset_buf > 0``, like
    :func:`contact_once`) and is also registered with
    :func:`_register_tensor_episode_state` so the explicit reset path clears it.
    """
    raw = checker.combine(env, op="and", children=[d["cfg"] for d in _sub_term_defs])
    raw = raw.to(dtype=torch.bool, device=env.device).reshape(-1)

    history_attr = f"_mana_latch_{state_key}"
    if (not hasattr(env, history_attr)) or getattr(env, history_attr).shape[0] != env.num_envs:
        setattr(env, history_attr, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    _register_tensor_episode_state(env, history_attr, fill_value=False)

    history: torch.Tensor = getattr(env, history_attr)

    reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if hasattr(env, "episode_length_buf"):
        reset_mask = reset_mask | (env.episode_length_buf == 0)
    if hasattr(env, "reset_buf"):
        reset_mask = reset_mask | (env.reset_buf > 0)
    if torch.any(reset_mask):
        history[reset_mask] = False

    history[:] = history | raw
    return history


def task_success(
    env: ManagerBasedRLEnv,
    *,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """Read the already-computed task ``success`` term for evaluation use."""
    del evaluation_runtime, score_runtime
    termination_manager = getattr(env, "termination_manager", None)
    active_terms = getattr(termination_manager, "active_terms", ()) if termination_manager is not None else ()
    if termination_manager is None or "success" not in active_terms:
        return _false_tensor(env)

    success_term = termination_manager.get_term("success")
    if isinstance(success_term, torch.Tensor):
        return success_term.to(dtype=torch.bool, device=env.device)
    return torch.as_tensor(success_term, dtype=torch.bool, device=env.device).reshape(env.num_envs)


def dag_stage_latched(
    env: ManagerBasedRLEnv,
    *,
    stage_index: int,
    extras_key: str = "dag_stage_done",
) -> torch.Tensor:
    """Read one stage's gated-latched state from the composite DAG-success machine.

    A graph-composite task's ``DagSuccessMachine.evaluate`` publishes, every
    step, ``env.extras[extras_key]`` — a ``[num_envs, num_stages]`` cumulative
    bool whose column ``i`` is True once stage ``i``'s predicate has fired AND
    all its graph predecessors have latched (see ``dag_success.py``). This reader
    returns column ``stage_index`` so the composite's evaluation rubric can score
    each stage with the SAME gated truth the termination path uses — crucial for
    stages like "retract" whose raw predicate (arms-at-home) is true at episode
    start and would otherwise score 100%.

    Defensive: returns all-False when the env has no extras dict, the key is
    absent (DAG term not active), or the column is out of range — so a non-graph
    task that somehow routes here stays inert rather than crashing.
    """
    extras = getattr(env, "extras", None)
    done = extras.get(extras_key) if isinstance(extras, dict) else None
    if not isinstance(done, torch.Tensor) or done.ndim != 2 or not (0 <= int(stage_index) < done.shape[1]):
        return _false_tensor(env)
    return done[:, int(stage_index)].to(dtype=torch.bool, device=env.device)


def max_episode_steps_reached(
    env: ManagerBasedRLEnv,
    max_steps: int,
) -> torch.Tensor:
    """Terminate when episode step count reaches *max_steps*."""
    return env.episode_length_buf >= max_steps


def _apply_consecutive_filter(
    env: ManagerBasedRLEnv,
    raw_result: torch.Tensor,
    threshold: int,
    counter_attr: str,
) -> torch.Tensor:
    """Return True only after *raw_result* has been True for *threshold* consecutive calls."""
    if not hasattr(env, counter_attr):
        setattr(env, counter_attr, torch.zeros(env.num_envs, dtype=torch.int32, device=env.device))
    _register_tensor_episode_state(env, counter_attr, fill_value=0)
    counter: torch.Tensor = getattr(env, counter_attr)
    counter[:] = torch.where(raw_result, counter + 1, torch.zeros_like(counter))
    return counter >= threshold


def object_position_in_zone(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg | str,
    reference_cfg: SceneEntityCfg | str | None = None,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    z_range: list[float] | None = None,
    distance_threshold: float | None = None,
    xy_distance_threshold: float | None = None,
    z_min_offset: float | None = None,
    z_max_offset: float | None = None,
    velocity_threshold: float | None = None,
    in_reference_frame: bool = False,
    consecutive_steps: int = 1,
    counter_attr: str | None = None,
    reference_footprint_zone: bool = False,
    xy_inset_ratio: float = 0.0,
    log_prefix: str | None = None,
    log_interval: int = 0,
    use_runtime_place_beside: bool = False,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL place] Object position inside a configured zone (abs box or relative-to-reference).
    Assumes: object pose (root_pos_w); optional reference entity; optional velocity gate (root_lin_vel_w); consecutive-steps support.

    The zone can be expressed as:
    - absolute box limits in env frame (`x_range` / `y_range` / `z_range`)
    - relative limits to `reference_cfg` (`distance_threshold`, `xy_distance_threshold`,
      `x_range` / `y_range` / `z_range`, or `z_min_offset` / `z_max_offset`)
    - `reference_footprint_zone=True`: live support footprint AABB (USD-backed)
    - `use_runtime_place_beside=True`: reset-time beside band from runtime context

    When ``in_reference_frame`` is true, ``reference_cfg`` is required and the
    object-to-reference offset is expressed in the reference's local frame
    (rotated by ``root_quat_w``) before applying box / distance limits.

    Runtime-randomized pool objects are resolved to their per-env active cache slot
    (same path as ``object_dropped``), so a contact-sensor mounted on the logical
    base name is not required for pool targets.
    """
    del evaluation_runtime, score_runtime

    if reference_footprint_zone:
        if reference_cfg is None:
            raise ValueError("object_position_in_zone: reference_footprint_zone requires reference_cfg.")
        return _object_in_reference_footprint(
            env,
            object_cfg=object_cfg,
            reference_cfg=reference_cfg,
            xy_inset_ratio=xy_inset_ratio,
            z_range=z_range,
            z_min_offset=z_min_offset,
            z_max_offset=z_max_offset,
            velocity_threshold=velocity_threshold,
            consecutive_steps=consecutive_steps,
            counter_attr=counter_attr,
            log_prefix=log_prefix,
            log_interval=log_interval,
        )

    object_name = _resolve_name(collapse_single_item(object_cfg))
    obj_pos, vel = _object_pos_vel_env(env, object_name)
    if obj_pos is None:
        return _false_tensor(env)

    if reference_cfg is not None:
        reference_name = _resolve_name(collapse_single_item(reference_cfg))
        reference_pos, _ = _object_pos_vel_env(env, reference_name)
        if reference_pos is None:
            return _false_tensor(env)
        diff_w = obj_pos - reference_pos
        if in_reference_frame:
            reference_quat = _object_quat_xyzw(env, reference_name)
            if reference_quat is None:
                return _false_tensor(env)
            from isaaclab.utils import math as math_utils

            rel_pos = math_utils.quat_apply_inverse(reference_quat, diff_w)
        else:
            rel_pos = diff_w
    else:
        if in_reference_frame:
            raise ValueError("object_position_in_zone(in_reference_frame=True) requires reference_cfg.")
        reference_name = ""
        rel_pos = obj_pos

    result = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    has_position_constraint = False

    if distance_threshold is not None:
        result = result & (torch.norm(rel_pos, dim=-1) < float(distance_threshold))
        has_position_constraint = True
    if xy_distance_threshold is not None:
        result = result & (torch.norm(rel_pos[:, :2], dim=-1) < float(xy_distance_threshold))
        has_position_constraint = True
    if use_runtime_place_beside:
        result = result & _runtime_place_beside_xy_mask(env, rel_pos)
        has_position_constraint = True
    else:
        if x_range is not None:
            result = result & (rel_pos[:, 0] >= float(x_range[0])) & (rel_pos[:, 0] <= float(x_range[1]))
            has_position_constraint = True
        if y_range is not None:
            result = result & (rel_pos[:, 1] >= float(y_range[0])) & (rel_pos[:, 1] <= float(y_range[1]))
            has_position_constraint = True
    if z_range is not None:
        result = result & (rel_pos[:, 2] >= float(z_range[0])) & (rel_pos[:, 2] <= float(z_range[1]))
        has_position_constraint = True
    if z_min_offset is not None:
        result = result & (rel_pos[:, 2] >= float(z_min_offset))
        has_position_constraint = True
    if z_max_offset is not None:
        result = result & (rel_pos[:, 2] <= float(z_max_offset))
        has_position_constraint = True

    if not has_position_constraint:
        raise ValueError("object_position_in_zone requires at least one position constraint.")

    if velocity_threshold is not None and vel is not None:
        result = result & (vel < float(velocity_threshold))

    if log_prefix is not None and log_interval > 0:
        step = int(env.episode_length_buf[0].item()) if env.episode_length_buf.numel() > 0 else 0
        last_attr = "_mana_log_relzone_" + _stable_term_counter_id(
            str(object_name),
            str(reference_name),
        )
        last_step = int(getattr(env, last_attr, -log_interval))
        if step < last_step:
            last_step = -int(log_interval)
        if step - last_step >= int(log_interval):
            setattr(env, last_attr, step)
            dx = float(rel_pos[0, 0].item())
            dy = float(rel_pos[0, 1].item())
            xy_d = float(torch.norm(rel_pos[0, :2]).item())
            rz = float(rel_pos[0, 2].item())
            vel0 = float(vel[0].item()) if (velocity_threshold is not None and vel is not None) else -1.0
            if abs(dx) >= abs(dy):
                side = "far" if dx > 0 else "near"
            else:
                side = "left" if dy > 0 else "right"
            if use_runtime_place_beside:
                from manaenv.pipeline.runtime.beside_placement import get_place_beside

                bundle = get_place_beside(env, env_id=0)
                if isinstance(bundle, dict):
                    target_goal = str(bundle.get("target") or "")
                    initial = str(bundle.get("initial") or "")
                    if target_goal:
                        side = f"{side}->goal={target_goal}(spawn={initial})"
            manaprint(
                f"INFO: [{log_prefix}] step={step} pass={bool(result[0].item())} "
                f"side={side} dx={dx:+.3f} dy={dy:+.3f} xy_dist={xy_d:.3f} "
                f"(<{xy_distance_threshold}) rel_z={rz:.3f} "
                f"(z_off=[{z_min_offset},{z_max_offset}]) vel={vel0:.3f} (<{velocity_threshold})"
            )

    if consecutive_steps <= 1:
        return result
    if counter_attr is None:
        counter_attr = "_mana_object_position_in_zone_" + _stable_term_counter_id(
            str(object_name),
            str(reference_name),
            str(x_range),
            str(y_range),
            str(z_range),
            str(distance_threshold),
            str(xy_distance_threshold),
            str(z_min_offset),
            str(z_max_offset),
            str(velocity_threshold),
            str(bool(in_reference_frame)),
            str(bool(use_runtime_place_beside)),
        )
    return _apply_consecutive_filter(env, result, consecutive_steps, counter_attr=counter_attr)


def objects_outside_zone(
    env: ManagerBasedRLEnv,
    *,
    object_cfgs: SceneEntityCfg | str | list,
    reference_cfg: SceneEntityCfg | str | None = None,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    z_range: list[float] | None = None,
    in_reference_frame: bool = False,
    reference_footprint_zone: bool = False,
    xy_inset_ratio: float = 0.0,
    use_runtime_place_beside: bool = False,
    z_min_offset: float | None = None,
    z_max_offset: float | None = None,
    consecutive_steps: int = 1,
    counter_attr: str | None = None,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL place] Every listed object lies outside a configured zone.

    Position membership reuses the same abs / relative / local-frame / footprint /
    place-beside rules as :func:`object_position_in_zone`, but velocity is
    intentionally ignored: an object resting inside the box must fail even while
    still settling. Missing objects count as outside. ``object_cfgs`` keeps a
    multi-asset ``@role`` as a whole list (``_cfgs`` suffix).
    """
    del evaluation_runtime, score_runtime

    cfgs = object_cfgs if isinstance(object_cfgs, list) else [object_cfgs]
    cfgs = [c for c in cfgs if c is not None]
    if not cfgs:
        raise ValueError("objects_outside_zone requires a non-empty object_cfgs list")
    has_box = x_range is not None or y_range is not None or z_range is not None
    has_z_off = z_min_offset is not None or z_max_offset is not None
    if not (has_box or reference_footprint_zone or use_runtime_place_beside or has_z_off):
        raise ValueError(
            "objects_outside_zone requires at least one of "
            "x_range/y_range/z_range, reference_footprint_zone, "
            "use_runtime_place_beside, or z_min_offset/z_max_offset"
        )
    if in_reference_frame and reference_cfg is None:
        raise ValueError("objects_outside_zone(in_reference_frame=True) requires reference_cfg.")
    if reference_footprint_zone and reference_cfg is None:
        raise ValueError("objects_outside_zone: reference_footprint_zone requires reference_cfg.")
    if use_runtime_place_beside and reference_cfg is None:
        raise ValueError("objects_outside_zone: use_runtime_place_beside requires reference_cfg.")

    if reference_footprint_zone:
        return _objects_outside_reference_footprint(
            env,
            object_cfgs=cfgs,
            reference_cfg=reference_cfg,
            xy_inset_ratio=xy_inset_ratio,
            z_range=z_range,
            z_min_offset=z_min_offset,
            z_max_offset=z_max_offset,
            consecutive_steps=consecutive_steps,
            counter_attr=counter_attr,
        )

    reference_name = ""
    reference_pos = None
    reference_quat = None
    if reference_cfg is not None:
        reference_name = _resolve_name(collapse_single_item(reference_cfg))
        reference_pos, _ = _object_pos_vel_env(env, reference_name)
        if reference_pos is None:
            return _false_tensor(env)
        if in_reference_frame:
            reference_quat = _object_quat_xyzw(env, reference_name)
            if reference_quat is None:
                return _false_tensor(env)

    names: list[str] = []
    all_outside = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in cfgs:
        object_name = _resolve_name(collapse_single_item(cfg))
        names.append(object_name)
        obj_pos, _ = _object_pos_vel_env(env, object_name)
        if obj_pos is None:
            continue
        if reference_pos is not None:
            diff_w = obj_pos - reference_pos
            if in_reference_frame:
                from isaaclab.utils import math as math_utils

                rel_pos = math_utils.quat_apply_inverse(reference_quat, diff_w)
            else:
                rel_pos = diff_w
        else:
            rel_pos = obj_pos

        in_zone = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        if use_runtime_place_beside:
            in_zone = in_zone & _runtime_place_beside_xy_mask(env, rel_pos)
        else:
            if x_range is not None:
                in_zone = in_zone & (rel_pos[:, 0] >= float(x_range[0])) & (rel_pos[:, 0] <= float(x_range[1]))
            if y_range is not None:
                in_zone = in_zone & (rel_pos[:, 1] >= float(y_range[0])) & (rel_pos[:, 1] <= float(y_range[1]))
        if z_range is not None:
            in_zone = in_zone & (rel_pos[:, 2] >= float(z_range[0])) & (rel_pos[:, 2] <= float(z_range[1]))
        if z_min_offset is not None:
            in_zone = in_zone & (rel_pos[:, 2] >= float(z_min_offset))
        if z_max_offset is not None:
            in_zone = in_zone & (rel_pos[:, 2] <= float(z_max_offset))
        all_outside = all_outside & (~in_zone)

    if consecutive_steps <= 1:
        return all_outside
    if counter_attr is None:
        counter_attr = "_mana_objects_outside_zone_" + _stable_term_counter_id(
            ",".join(names),
            str(reference_name),
            str(x_range),
            str(y_range),
            str(z_range),
            str(z_min_offset),
            str(z_max_offset),
            str(bool(in_reference_frame)),
            str(bool(use_runtime_place_beside)),
        )
    return _apply_consecutive_filter(env, all_outside, consecutive_steps, counter_attr=counter_attr)


def _objects_outside_reference_footprint(
    env: ManagerBasedRLEnv,
    *,
    object_cfgs: list,
    reference_cfg: SceneEntityCfg | str,
    xy_inset_ratio: float,
    z_range: list[float] | None,
    z_min_offset: float | None,
    z_max_offset: float | None,
    consecutive_steps: int,
    counter_attr: str | None,
) -> torch.Tensor:
    """Every listed object lies outside the reference support footprint."""
    logical_reference = _resolve_name(collapse_single_item(reference_cfg))
    names: list[str] = []
    all_outside = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in object_cfgs:
        object_name = _resolve_name(collapse_single_item(cfg))
        names.append(object_name)
        obj_pos, _ = _object_pos_vel_env(env, object_name)
        if obj_pos is None:
            continue
        extent_cache_attr = "_mana_ref_local_ext_out_" + _stable_term_counter_id(object_name, logical_reference)
        in_zone = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for env_id in range(env.num_envs):
            inside, _debug = _object_in_reference_footprint_env(
                env,
                obj_pos,
                None,
                logical_reference,
                env_id,
                xy_inset_ratio=xy_inset_ratio,
                z_range=z_range,
                z_min_offset=z_min_offset,
                z_max_offset=z_max_offset,
                velocity_threshold=None,
                extent_cache_attr=extent_cache_attr,
            )
            in_zone[env_id] = bool(inside)
        all_outside = all_outside & (~in_zone)

    if consecutive_steps <= 1:
        return all_outside
    if counter_attr is None:
        counter_attr = "_mana_objects_outside_footprint_" + _stable_term_counter_id(
            ",".join(names),
            str(logical_reference),
            str(xy_inset_ratio),
            str(z_range),
            str(z_min_offset),
            str(z_max_offset),
        )
    return _apply_consecutive_filter(env, all_outside, consecutive_steps, counter_attr=counter_attr)


def _quat_rotate_inverse_wxyz(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` (3,) by the inverse of a wxyz quaternion -> parent-local frame."""
    w = quat_wxyz[0]
    qvec = quat_wxyz[1:4]
    t = 2.0 * torch.linalg.cross(qvec, vec)
    return vec - w * t + torch.linalg.cross(qvec, t)


def _reference_bbox_env(
    env: ManagerBasedRLEnv,
    logical_reference: str,
    env_id: int,
) -> tuple[torch.Tensor, torch.Tensor, str] | None:
    """Return ``(bbox_min, bbox_max, active_name)`` for the support footprint.

    Uses :func:`get_object_footprint_aabb_env`: the resting root-frame bound (already
    env-aligned, thin axis along env Z for a flat support) translated to the live
    root position. Reconstructing ``root-frame bbox × live quat`` instead
    double-counts the spawn rotation for Y-up supports (the live +90 deg-about-X that
    lays a Y-up plate flat is already baked into the root-frame bound, so re-applying
    it tips the flat disc onto its edge and collapses the footprint's Y span).
    """
    from manaenv.utils.obb_utils import get_object_footprint_aabb_env

    active_name = resolve_active_scene_entity_name(env, logical_reference, env_id=env_id)
    if not scene_has_asset(env.scene, active_name):
        return None
    bbox = get_object_footprint_aabb_env(
        env,
        active_name,
        env_id,
        env_origin=env.scene.env_origins[env_id],
        device=env.device,
    )
    if bbox is None:
        return None
    bbox_min, bbox_max = bbox
    return bbox_min, bbox_max, active_name


def _footprint_z_above_support(
    p_obj: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> float:
    """Object height above the support base (env Z = gravity up).

    Base (``bbox_min`` Z) works for both a flat plate (object rests just above the
    thin slab) and a deep basket (object sits inside, above the floor).
    """
    return float(p_obj[2].item()) - float(bbox_min[2].item())


def _object_in_reference_footprint_env(
    env: ManagerBasedRLEnv,
    obj_pos_env: torch.Tensor,
    obj_vel: torch.Tensor | None,
    logical_reference: str,
    env_id: int,
    *,
    xy_inset_ratio: float,
    z_range: list[float] | None,
    z_min_offset: float | None,
    z_max_offset: float | None,
    velocity_threshold: float | None,
    extent_cache_attr: str,
) -> tuple[bool, dict[str, Any] | None]:
    """One-env check: object center inside the support footprint (env frame).

    Env frame, gravity up = Z. Footprint is the support's live world-AABB X/Y
    box; height is measured above the support top face (bbox_max Z). No local
    frame, no auto up-axis detection: a flat plate's footprint is simply its
    env X/Y projection, which is what the motion planner places into.
    """
    _FP_EPS = 1.0e-4

    aabb = _reference_bbox_env(env, logical_reference, env_id)
    if aabb is None:
        return False, None
    bbox_min, bbox_max, active_name = aabb

    p_obj = obj_pos_env[env_id].to(dtype=torch.float32)
    ext_x = float((bbox_max[0] - bbox_min[0]).item())
    ext_y = float((bbox_max[1] - bbox_min[1]).item())
    ix = float(xy_inset_ratio) * ext_x
    iy = float(xy_inset_ratio) * ext_y
    a_lo, a_hi = float(bbox_min[0].item()) + ix, float(bbox_max[0].item()) - ix
    b_lo, b_hi = float(bbox_min[1].item()) + iy, float(bbox_max[1].item()) - iy
    ea = float(p_obj[0].item())
    eb = float(p_obj[1].item())
    z_above_floor = _footprint_z_above_support(p_obj, bbox_min, bbox_max)

    a_ok = (a_lo - _FP_EPS) <= ea <= (a_hi + _FP_EPS)
    b_ok = (b_lo - _FP_EPS) <= eb <= (b_hi + _FP_EPS)
    if z_range is None and z_min_offset is None and z_max_offset is None:
        z_ok = -0.03 <= z_above_floor <= 0.06
    else:
        z_ok = True
        if z_range is not None:
            z_ok = z_ok and float(z_range[0]) <= z_above_floor <= float(z_range[1])
        if z_min_offset is not None:
            z_ok = z_ok and z_above_floor >= float(z_min_offset)
        if z_max_offset is not None:
            z_ok = z_ok and z_above_floor <= float(z_max_offset)

    vel_val = float(obj_vel[env_id].item()) if obj_vel is not None else -1.0
    vel_ok = velocity_threshold is None or obj_vel is None or vel_val < float(velocity_threshold)
    inside = a_ok and b_ok and z_ok and vel_ok
    center = (bbox_min + bbox_max) * 0.5
    debug = {
        "container": active_name,
        "c_xy": (float(center[0].item()), float(center[1].item())),
        "lx": ea,
        "ly": eb,
        "x_box": (a_lo, a_hi),
        "y_box": (b_lo, b_hi),
        "z_above_floor": z_above_floor,
        "vel": vel_val,
        "x_ok": a_ok,
        "y_ok": b_ok,
        "z_ok": z_ok,
        "vel_ok": vel_ok,
        "inside": inside,
        "axis_a": "x",
        "axis_b": "y",
    }
    return inside, debug


def _object_in_reference_footprint(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg | str,
    reference_cfg: SceneEntityCfg | str,
    xy_inset_ratio: float,
    z_range: list[float] | None,
    z_min_offset: float | None,
    z_max_offset: float | None,
    velocity_threshold: float | None,
    consecutive_steps: int,
    counter_attr: str | None,
    log_prefix: str | None,
    log_interval: int,
) -> torch.Tensor:
    """Object center inside the container's local footprint; pose read every frame."""
    logical_object = _resolve_name(collapse_single_item(object_cfg))
    logical_reference = _resolve_name(collapse_single_item(reference_cfg))
    obj_pos, obj_vel = _object_pos_vel_env(env, logical_object)
    if obj_pos is None:
        return _false_tensor(env)

    extent_cache_attr = "_mana_ref_local_ext_" + _stable_term_counter_id(logical_object, logical_reference)
    raw = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    debug0: dict[str, Any] | None = None
    for env_id in range(env.num_envs):
        inside, debug = _object_in_reference_footprint_env(
            env,
            obj_pos,
            obj_vel,
            logical_reference,
            env_id,
            xy_inset_ratio=xy_inset_ratio,
            z_range=z_range,
            z_min_offset=z_min_offset,
            z_max_offset=z_max_offset,
            velocity_threshold=velocity_threshold,
            extent_cache_attr=extent_cache_attr,
        )
        raw[env_id] = inside
        if env_id == 0:
            debug0 = debug

    if log_prefix is not None and log_interval > 0:
        step = int(env.episode_length_buf[0].item()) if env.episode_length_buf.numel() > 0 else 0
        last_attr = "_mana_log_footprint_" + _stable_term_counter_id(logical_object, logical_reference)
        last_step = int(getattr(env, last_attr, -log_interval))
        if step < last_step:
            last_step = -int(log_interval)
        if step - last_step >= int(log_interval):
            setattr(env, last_attr, step)
            if debug0 is not None:
                d = debug0
                fail = (
                    ",".join(
                        axis
                        for axis, ok in (("x", d["x_ok"]), ("y", d["y_ok"]), ("z", d["z_ok"]), ("vel", d["vel_ok"]))
                        if not ok
                    )
                    or "none"
                )
                manaprint(
                    f"INFO: [{log_prefix}] step={step} container={d['container']} "
                    f"c_xy=({d['c_xy'][0]:.3f},{d['c_xy'][1]:.3f}) inside={d['inside']} fail={fail} "
                    f"| env_{d['axis_a']}={d['lx']:.3f} in [{d['x_box'][0]:.3f},{d['x_box'][1]:.3f}] ok={d['x_ok']} "
                    f"| env_{d['axis_b']}={d['ly']:.3f} in [{d['y_box'][0]:.3f},{d['y_box'][1]:.3f}] ok={d['y_ok']} "
                    f"| z_above_floor={d['z_above_floor']:.3f} ok={d['z_ok']} "
                    f"| vel={d['vel']:.3f} ok={d['vel_ok']}"
                )
            else:
                manaprint(f"INFO: [{log_prefix}] step={step} footprint_or_object_missing")

    if consecutive_steps <= 1:
        return raw
    if counter_attr is None:
        counter_attr = "_mana_obj_footprint_" + _stable_term_counter_id(logical_object, logical_reference)
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def object_in_opened_joint_region(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg | str,
    reference_cfg: SceneEntityCfg | str,
    sliding_body_name: str = "drawer",
    joint_name: str = "PrismaticJoint",
    joint_axis_local: Sequence[float] = (1.0, 0.0, 0.0),
    protruding_only: bool = False,
    xy_inset_ratio: float = 0.10,
    z_range: list[float] | None = None,
    velocity_threshold: float | None = 0.08,
    consecutive_steps: int = 3,
) -> torch.Tensor:
    """[SKILL place] Object XY inside the live opened-joint placeable tray region.

    Matches :func:`manaenv.pipeline.skills.place_runtime.opened_joint_region.resolve_opened_joint_region`
    so place success uses the same tray slab the place skill targets. Root-frame
    footprint checks miss the protruding open tray (cabinet root does not move).
    """
    from manaenv.pipeline.skills.place_runtime.opened_joint_region import resolve_opened_joint_region

    logical_object = _resolve_name(collapse_single_item(object_cfg))
    logical_reference = _resolve_name(collapse_single_item(reference_cfg))
    obj_pos, obj_vel = _object_pos_vel_env(env, logical_object)
    if obj_pos is None:
        return _false_tensor(env)

    z_lo, z_hi = (0.0, 0.15) if z_range is None else (float(z_range[0]), float(z_range[1]))
    inset = max(0.0, min(0.49, float(xy_inset_ratio)))
    raw = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for env_id in range(int(env.num_envs)):
        ref_name = resolve_active_scene_entity_name(env, logical_reference, env_id=env_id)
        if not scene_has_asset(env.scene, ref_name):
            # Fall back to logical / any drawer* articulation key.
            arts = getattr(env.scene, "articulations", None) or {}
            if logical_reference in arts:
                ref_name = logical_reference
            else:
                hits = sorted(str(k) for k in arts if str(k).startswith(f"{logical_reference}__"))
                if not hits:
                    continue
                ref_name = hits[0]
        region = resolve_opened_joint_region(
            env=env,
            object_name=ref_name,
            env_idx=env_id,
            env_origin=env.scene.env_origins[env_id],
            sliding_body_name=str(sliding_body_name),
            joint_name=str(joint_name),
            joint_axis_local=tuple(float(v) for v in joint_axis_local),
            device=env.device,
            protruding_only=bool(protruding_only),
            # Same inset semantics as place target (shrink sliding-body AABB first).
            xy_inset_ratio=inset,
        )
        if region is None:
            continue
        lo = region.open_min_xy
        hi = region.open_max_xy
        p = obj_pos[env_id]
        xy_ok = bool(
            (float(p[0].item()) >= float(lo[0].item()) - 1e-4)
            and (float(p[0].item()) <= float(hi[0].item()) + 1e-4)
            and (float(p[1].item()) >= float(lo[1].item()) - 1e-4)
            and (float(p[1].item()) <= float(hi[1].item()) + 1e-4)
        )
        z_above = float(p[2].item()) - float(region.tray_z)
        z_ok = z_lo <= z_above <= z_hi
        vel_val = float(obj_vel[env_id].item()) if obj_vel is not None else -1.0
        vel_ok = velocity_threshold is None or obj_vel is None or vel_val < float(velocity_threshold)
        raw[env_id] = xy_ok and z_ok and vel_ok

    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_obj_opened_region_" + _stable_term_counter_id(logical_object, logical_reference)
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def indexed_object_position_in_zone(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    index_log_key: str,
    reference_cfg: SceneEntityCfg | str | None = None,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    z_range: list[float] | None = None,
    distance_threshold: float | None = None,
    xy_distance_threshold: float | None = None,
    z_min_offset: float | None = None,
    z_max_offset: float | None = None,
    velocity_threshold: float | None = None,
    in_reference_frame: bool = False,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL place] Zone check on the single object selected by a reset-committed index.
    Assumes: ``index_log_key`` stores one candidate-aligned integer per env and object poses are available.

    Same zone parameters as :func:`object_position_in_zone`; ``object_names`` aligns
    with ``index_log_key`` (int ``[num_envs, 1]``).
    """

    names = [str(name) for name in object_names]
    if not names:
        raise ValueError("indexed_object_position_in_zone requires a non-empty object_names")
    idx_log = read_log(env, str(index_log_key))
    if idx_log is None:
        return _false_tensor(env)
    idx_log = idx_log.to(device=env.device, dtype=torch.int64).reshape(env.num_envs, -1)[:, 0]
    idx_log = idx_log.clamp(0, len(names) - 1)

    per_object = [
        object_position_in_zone(
            env,
            object_cfg=name,
            reference_cfg=reference_cfg,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            distance_threshold=distance_threshold,
            xy_distance_threshold=xy_distance_threshold,
            z_min_offset=z_min_offset,
            z_max_offset=z_max_offset,
            velocity_threshold=velocity_threshold,
            in_reference_frame=in_reference_frame,
            consecutive_steps=1,
        )
        for name in names
    ]
    stacked = torch.stack(per_object, dim=0)
    env_ids = torch.arange(env.num_envs, device=env.device)
    raw = stacked[idx_log, env_ids]
    if consecutive_steps <= 1:
        return raw
    ref_name = "" if reference_cfg is None else _resolve_name(collapse_single_item(reference_cfg))
    counter_attr = "_mana_indexed_object_zone_" + _stable_term_counter_id(*names, str(index_log_key), str(ref_name))
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def alphabet_active_word(env: ManagerBasedRLEnv, state_attr: str = "_mana_alphabet_word_state") -> list[str | None]:
    """Return the reset-sampled alphabet target word for per-episode logging."""
    state = getattr(env, state_attr, None)
    words = state.get("word") if isinstance(state, dict) else None
    if not isinstance(words, dict):
        return [None] * int(env.num_envs)
    return [words[env_id].upper() if env_id in words else None for env_id in range(int(env.num_envs))]


def alphabet_active_word_letters(env: ManagerBasedRLEnv, state_attr: str = "_mana_alphabet_word_state") -> torch.Tensor:
    """Return reset-sampled alphabet letter ids for a non-reset-owned episode log key."""
    state = getattr(env, state_attr, None)
    letters = state.get("letters") if isinstance(state, dict) else None
    if not isinstance(letters, dict):
        return torch.zeros((int(env.num_envs), 0), dtype=torch.int64, device=env.device)
    width = max((len(value) for value in letters.values() if isinstance(value, list)), default=0)
    out = torch.zeros((int(env.num_envs), width), dtype=torch.int64, device=env.device)
    for env_id, value in letters.items():
        if 0 <= int(env_id) < int(env.num_envs) and isinstance(value, list):
            out[int(env_id), : len(value)] = torch.as_tensor(value, dtype=torch.int64, device=env.device)
    return out


def button_active_order(
    env: ManagerBasedRLEnv, state_attr: str = "_mana_buttons_random_order_state"
) -> list[str | None]:
    """Return the reset-sampled random press order text per env for the overlay."""
    state = getattr(env, state_attr, None)
    texts = state.get("order_text") if isinstance(state, dict) else None
    if not isinstance(texts, dict):
        return [None] * int(env.num_envs)
    return [texts.get(env_id) for env_id in range(int(env.num_envs))]


def button_contact_in_order(
    env: ManagerBasedRLEnv,
    *,
    position: int,
    sensor_cfgs: Any,
    color_order: list[str],
    force_threshold: float = 0.1,
    consecutive_steps: int = 1,
    order_log_key: str = "button_press_order",
    counter_attr: str | None = None,
    state_attr: str | None = None,
) -> torch.Tensor:
    """Advance ordered button-press progress using the reset-sampled random order.

    The reset event (:func:`manaenv.terms.events.sample_button_order`) commits a
    per-env order tensor to ``order_log_key`` — 1-based indices into
    ``color_order`` (0 padded). Every step we look at the button each env is
    expected to press next and, if that button's color contact sensor is active,
    advance the env's progress. This stage's success is progress reaching
    ``position``, so gating the DAG stages enforces the sampled order.

    ``sensor_cfgs`` is the full list of per-color contact sensors (kept intact —
    the plural ``_cfgs`` name avoids the single-sensor fan-out that the special
    ``contact_sensor_cfgs`` key triggers at runtime).
    """

    num_envs = int(env.num_envs)
    device = env.device

    flat_cfgs: list[Any] = []
    for item in sensor_cfgs if isinstance(sensor_cfgs, list) else [sensor_cfgs]:
        if isinstance(item, list):
            flat_cfgs.extend(item)
        else:
            flat_cfgs.append(item)

    colors = [str(color).strip().lower().replace("button_", "") for color in (color_order or [])]

    resolved_counter = counter_attr or (
        "_mana_button_contact_in_order_"
        + _stable_term_counter_id(
            str(position),
            str(order_log_key),
            str(colors),
            *(str(_resolve_name(collapse_single_item(cfg))) for cfg in flat_cfgs),
        )
    )

    color_to_col: dict[str, int] = {}
    for col, cfg in enumerate(flat_cfgs):
        cname = _sensor_color_name(cfg)
        if cname:
            color_to_col[cname] = col

    if not colors:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    canon_to_col = torch.tensor(
        [int(color_to_col.get(color, -1)) for color in colors], dtype=torch.int64, device=device
    )

    order = read_log(env, order_log_key)
    if order is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    order = order.to(device=device, dtype=torch.int64)
    if order.dim() == 1:
        order = order.unsqueeze(-1)
    width = order.shape[1]

    contact_flags = _current_contact_flags(env, contact_sensor_cfgs=flat_cfgs, force_threshold=force_threshold)

    resolved_state = state_attr or (
        "_mana_button_order_progress_" + _stable_term_counter_id(str(order_log_key), str(colors))
    )
    if (not hasattr(env, resolved_state)) or getattr(env, resolved_state).shape != (num_envs,):
        setattr(env, resolved_state, torch.zeros(num_envs, dtype=torch.int64, device=device))
    _register_tensor_episode_state(env, resolved_state, fill_value=0)
    progress: torch.Tensor = getattr(env, resolved_state)
    reset_mask = _episode_reset_mask(env)
    if torch.any(reset_mask):
        progress[reset_mask] = 0

    order0 = order - 1
    valid_slot = (order0 >= 0) & (order0 < len(colors))
    safe_order0 = torch.where(valid_slot, order0, torch.zeros_like(order0))
    seq_cols = torch.where(valid_slot, canon_to_col[safe_order0], torch.full_like(order0, -1))

    env_ids = torch.arange(num_envs, dtype=torch.int64, device=device)
    slot = int(position) - 1
    if slot < 0 or slot >= width:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    expected_col = seq_cols[:, slot]
    has_expected = (progress == slot) & (expected_col >= 0)
    safe_col = torch.where(expected_col >= 0, expected_col, torch.zeros_like(expected_col))
    expected_contact = contact_flags[env_ids, safe_col] & has_expected
    unexpected_contact = contact_flags.clone()
    unexpected_contact[env_ids, safe_col] = False
    clean_expected_contact = expected_contact & ~unexpected_contact.any(dim=1)
    confirmed = _apply_consecutive_filter(
        env,
        clean_expected_contact,
        consecutive_steps,
        counter_attr=resolved_counter,
    )
    progress[confirmed] = int(position)

    return progress >= int(position)


def button_wrong_contact_in_order(
    env: ManagerBasedRLEnv,
    *,
    position: int | None = None,
    sensor_cfgs: Any,
    color_order: list[str],
    force_threshold: float = 0.1,
    order_log_key: str = "button_press_order",
    state_attr: str | None = None,
) -> torch.Tensor:
    """Fail when a non-target button starts a contact during an ordered stage.

    Only a rising contact edge fails. This lets the DAG enter the next stage in
    the same simulation step without treating the previous target's residual
    contact as a fresh wrong press.
    """
    num_envs = int(env.num_envs)
    device = env.device
    flat_cfgs = [
        nested
        for item in (sensor_cfgs if isinstance(sensor_cfgs, list) else [sensor_cfgs])
        for nested in (item if isinstance(item, list) else [item])
    ]
    colors = [str(color).strip().lower().replace("button_", "") for color in (color_order or [])]
    result = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if not colors or not flat_cfgs:
        return result

    color_to_col = {color: col for col, cfg in enumerate(flat_cfgs) if (color := _sensor_color_name(cfg))}
    canon_to_col = torch.tensor(
        [int(color_to_col.get(color, -1)) for color in colors], dtype=torch.int64, device=device
    )
    expected_col = torch.full((num_envs,), -1, dtype=torch.int64, device=device)
    if position is not None:
        order = read_log(env, order_log_key)
        if order is None:
            return result
        order = order.to(device=device, dtype=torch.int64)
        if order.dim() == 1:
            order = order.unsqueeze(-1)
        slot = int(position) - 1
        if slot < 0 or slot >= int(order.shape[1]):
            return result
        order_index = order[:, slot] - 1
        valid = (order_index >= 0) & (order_index < len(colors))
        safe_index = torch.where(valid, order_index, torch.zeros_like(order_index))
        expected_col = torch.where(valid, canon_to_col[safe_index], torch.full_like(order_index, -1))

    current = _current_contact_flags(env, contact_sensor_cfgs=flat_cfgs, force_threshold=force_threshold)
    resolved_state = state_attr or (
        "_mana_button_wrong_contact_edges_"
        + _stable_term_counter_id(
            str(order_log_key),
            str(colors),
            *(str(_resolve_name(collapse_single_item(cfg))) for cfg in flat_cfgs),
        )
    )
    if (not hasattr(env, resolved_state)) or getattr(env, resolved_state).shape != current.shape:
        setattr(env, resolved_state, torch.zeros_like(current))
    _register_tensor_episode_state(env, resolved_state, fill_value=False)
    previous: torch.Tensor = getattr(env, resolved_state)
    reset_mask = _episode_reset_mask(env)
    if torch.any(reset_mask):
        previous[reset_mask] = False
    rising = current & ~previous
    previous.copy_(current)

    has_expected = expected_col >= 0
    if torch.any(has_expected):
        env_ids = torch.arange(num_envs, dtype=torch.int64, device=device)
        rising[env_ids[has_expected], expected_col[has_expected]] = False
    return rising.any(dim=1)


def button_order_color_at_position(
    env: ManagerBasedRLEnv,
    *,
    position: int,
    color_order: list[str],
    order_log_key: str = "button_press_order",
    template: str = "press {color}",
) -> list[str | None]:
    """Subtask log ``value_func``: color at 1-based ``position`` in the sampled order.

    Companion to :func:`button_joint_pressed_in_order`. ``order_log_key`` holds
    1-based indices into ``color_order`` (same tensor the press success term
    advances). DAG press nodes override ``params.position`` with their stage
    index so each progress entry names the button that stage just completed.
    """
    num_envs = int(env.num_envs)
    colors = [str(color).strip().lower().replace("button_", "") for color in (color_order or [])]
    if not colors or int(position) < 1:
        return [None] * num_envs

    order = read_log(env, order_log_key)
    out: list[str | None] = [None] * num_envs
    if order is None:
        return out
    order = order.to(device=env.device, dtype=torch.int64)
    if order.dim() == 1:
        order = order.unsqueeze(-1)
    slot = int(position) - 1
    if slot >= int(order.shape[1]):
        return out
    for env_id in range(num_envs):
        idx = int(order[env_id, slot].item())
        if idx < 1 or idx > len(colors):
            continue
        color = colors[idx - 1]
        out[env_id] = template.format(color=color, name=f"button_{color}", position=int(position))
    return out


def button_joint_pressed_in_order(
    env: ManagerBasedRLEnv,
    *,
    position: int,
    button_assets: list[str],
    color_order: list[str],
    joint_names: list[str],
    press_threshold: float = 0.6,
    release_threshold: float = 0.25,
    openness_gt: bool = True,
    order_log_key: str = "button_press_order",
    consecutive_steps: int = 1,
    counter_attr: str | None = None,
    state_attr: str | None = None,
) -> torch.Tensor:
    """Advance ordered button-press progress via per-button joint 0→1→0 cycles.

    Reset (:func:`manaenv.terms.events.sample_button_press_sequence`) commits a
    with-replacement order of 1-based indices into ``color_order``. Each step we
    read openness for every asset in ``button_assets`` (same order as colors),
    look at the button each env should press next, and complete a hysteresis
    cycle (cross ``press_threshold`` then return past ``release_threshold``)
    before advancing progress. Stage success is ``progress >= position``.

    With-replacement sequences (e.g. red, red) require a full release between
    presses — a held-down joint alone cannot advance twice.
    """
    if not joint_names:
        raise ValueError("button_joint_pressed_in_order requires a non-empty joint_names list")
    if float(press_threshold) == float(release_threshold):
        raise ValueError("button_joint_pressed_in_order requires distinct press/release thresholds")

    num_envs = int(env.num_envs)
    device = env.device
    colors = [str(color).strip().lower().replace("button_", "") for color in (color_order or [])]
    assets = [str(name).strip() for name in (button_assets or []) if str(name).strip()]
    if not colors or len(assets) != len(colors):
        raise ValueError("button_joint_pressed_in_order requires button_assets aligned 1:1 with color_order")
    patterns = [p.strip() for p in joint_names]

    order = read_log(env, order_log_key)
    if order is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    order = order.to(device=device, dtype=torch.int64)
    if order.dim() == 1:
        order = order.unsqueeze(-1)
    width = int(order.shape[1])

    openness = torch.zeros((num_envs, len(assets)), dtype=torch.float32, device=device)
    for col, asset_name in enumerate(assets):
        if not scene_has_asset(env.scene, asset_name):
            continue
        articulation = env.scene[asset_name]
        joint_indices, _ = articulation.find_joints(patterns, preserve_order=True)
        if not joint_indices:
            continue
        joint_index = int(joint_indices[0])
        joint_pos = articulation.data.joint_pos[:, joint_index]
        limits = articulation.data.joint_pos_limits[0, joint_index, :]
        openness[:, col] = _normalized_joint_position_openness(joint_pos, limits[0], limits[1])

    resolved_state = state_attr or (
        "_mana_button_joint_order_"
        + _stable_term_counter_id(str(order_log_key), str(colors), str(press_threshold), str(release_threshold))
    )
    progress_attr = resolved_state + "_progress"
    armed_attr = resolved_state + "_armed"
    if (not hasattr(env, progress_attr)) or getattr(env, progress_attr).shape != (num_envs,):
        setattr(env, progress_attr, torch.zeros(num_envs, dtype=torch.int64, device=device))
    if (not hasattr(env, armed_attr)) or getattr(env, armed_attr).shape != (num_envs,):
        setattr(env, armed_attr, torch.zeros(num_envs, dtype=torch.bool, device=device))
    _register_tensor_episode_state(env, progress_attr, fill_value=0)
    _register_tensor_episode_state(env, armed_attr, fill_value=0)
    progress: torch.Tensor = getattr(env, progress_attr)
    armed: torch.Tensor = getattr(env, armed_attr)

    reset_mask = _episode_reset_mask(env)
    if torch.any(reset_mask):
        progress[reset_mask] = 0
        armed[reset_mask] = False

    order0 = order - 1
    valid_slot = (order0 >= 0) & (order0 < len(colors))
    safe_order0 = torch.where(valid_slot, order0, torch.zeros_like(order0))
    env_ids = torch.arange(num_envs, dtype=torch.int64, device=device)
    slot = int(position) - 1
    if slot < 0 or slot >= width:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    expected_col = torch.where(
        valid_slot[:, slot],
        safe_order0[:, slot],
        torch.full((num_envs,), -1, dtype=torch.int64, device=device),
    )
    has_expected = (progress == slot) & (expected_col >= 0)
    safe_col = torch.where(expected_col >= 0, expected_col, torch.zeros_like(expected_col))
    expected_open = openness[env_ids, safe_col]

    if openness_gt:
        all_pressed = openness > float(press_threshold)
        pressed_now = expected_open > float(press_threshold)
        released_now = expected_open < float(release_threshold)
    else:
        all_pressed = openness < float(press_threshold)
        pressed_now = expected_open < float(press_threshold)
        released_now = expected_open > float(release_threshold)

    all_pressed[env_ids, safe_col] = False
    no_wrong_button = ~all_pressed.any(dim=1)
    newly_armed = has_expected & (~armed) & pressed_now & no_wrong_button
    completed = has_expected & armed & released_now & no_wrong_button
    if torch.any(newly_armed):
        armed[newly_armed] = True
    if torch.any(completed):
        progress[completed] = progress[completed] + 1
        armed[completed] = False

    raw_success = progress >= int(position)
    resolved_counter = counter_attr or (
        "_mana_button_joint_in_order_"
        + _stable_term_counter_id(
            str(position),
            str(order_log_key),
            str(colors),
            str(press_threshold),
            str(release_threshold),
            *assets,
            *patterns,
        )
    )
    return _apply_consecutive_filter(
        env,
        raw_success,
        consecutive_steps,
        counter_attr=resolved_counter,
    )


def button_wrong_joint_pressed_in_order(
    env: ManagerBasedRLEnv,
    *,
    button_assets: list[str],
    color_order: list[str],
    joint_names: list[str],
    press_threshold: float = 0.6,
    release_threshold: float = 0.25,
    openness_gt: bool = True,
    order_log_key: str = "button_press_order",
    state_attr: str | None = None,
) -> torch.Tensor:
    """Fail on a newly pressed non-target joint in a button sequence."""
    if not joint_names:
        raise ValueError("button_wrong_joint_pressed_in_order requires a non-empty joint_names list")
    num_envs = int(env.num_envs)
    device = env.device
    colors = [str(color).strip().lower().replace("button_", "") for color in (color_order or [])]
    assets = [str(name).strip() for name in (button_assets or []) if str(name).strip()]
    if not colors or len(assets) != len(colors):
        raise ValueError("button_wrong_joint_pressed_in_order requires button_assets aligned 1:1 with color_order")
    patterns = [pattern.strip() for pattern in joint_names]

    order = read_log(env, order_log_key)
    if order is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    order = order.to(device=device, dtype=torch.int64)
    if order.dim() == 1:
        order = order.unsqueeze(-1)

    openness = torch.zeros((num_envs, len(assets)), dtype=torch.float32, device=device)
    for col, asset_name in enumerate(assets):
        if not scene_has_asset(env.scene, asset_name):
            continue
        articulation = env.scene[asset_name]
        joint_indices, _ = articulation.find_joints(patterns, preserve_order=True)
        if not joint_indices:
            continue
        joint_index = int(joint_indices[0])
        limits = articulation.data.joint_pos_limits[0, joint_index, :]
        openness[:, col] = _normalized_joint_position_openness(
            articulation.data.joint_pos[:, joint_index], limits[0], limits[1]
        )
    pressed = openness > float(press_threshold) if openness_gt else openness < float(press_threshold)

    progress_attr = (
        "_mana_button_joint_order_"
        + _stable_term_counter_id(str(order_log_key), str(colors), str(press_threshold), str(release_threshold))
        + "_progress"
    )
    progress = getattr(env, progress_attr, torch.zeros(num_envs, dtype=torch.int64, device=device))
    width = int(order.shape[1])
    slot = torch.clamp(progress, min=0, max=max(width - 1, 0))
    env_ids = torch.arange(num_envs, dtype=torch.int64, device=device)
    expected_col = order[env_ids, slot] - 1
    valid = (progress < width) & (expected_col >= 0) & (expected_col < len(colors))
    safe_col = torch.where(valid, expected_col, torch.zeros_like(expected_col))

    resolved_state = state_attr or (
        "_mana_button_wrong_joint_edges_"
        + _stable_term_counter_id(str(order_log_key), str(colors), str(press_threshold), *assets, *patterns)
    )
    if (not hasattr(env, resolved_state)) or getattr(env, resolved_state).shape != pressed.shape:
        setattr(env, resolved_state, torch.zeros_like(pressed))
    _register_tensor_episode_state(env, resolved_state, fill_value=False)
    previous: torch.Tensor = getattr(env, resolved_state)
    reset_mask = _episode_reset_mask(env)
    if torch.any(reset_mask):
        previous[reset_mask] = False
    rising = pressed & ~previous
    previous.copy_(pressed)
    if torch.any(valid):
        rising[env_ids[valid], safe_col[valid]] = False
    return rising.any(dim=1)


def letter_block_in_word_slot(
    env: ManagerBasedRLEnv,
    *,
    position: int,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    z_range: list[float] | None = None,
    velocity_threshold: float | None = None,
    consecutive_steps: int = 1,
    counter_attr: str | None = None,
    tensor_key: str = "target_word_letters",
    block_prefix: str = "block_alphabet_",
    state_attr: str = "_mana_alphabet_word_state",
    slot_tolerance_x: float = 0.04,
    slot_tolerance_y: float = 0.05,
    slot_tolerance_z: float | None = None,
    yaw_target_deg: float | None = None,
    yaw_tolerance_deg: float = 30.0,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """Check that the selected per-env letter block rests inside a word slot."""
    del evaluation_runtime, score_runtime

    num_envs = int(env.num_envs)
    device = env.device

    def _finish(raw: torch.Tensor) -> torch.Tensor:
        if consecutive_steps <= 1:
            return raw
        resolved_counter_attr = counter_attr
        if resolved_counter_attr is None:
            resolved_counter_attr = "_mana_letter_block_in_word_slot_" + _stable_term_counter_id(
                str(position),
                str(x_range),
                str(y_range),
                str(z_range),
                str(velocity_threshold),
                str(tensor_key),
                str(block_prefix),
                str(state_attr),
                str(slot_tolerance_x),
                str(slot_tolerance_y),
                str(slot_tolerance_z),
                str(yaw_target_deg),
                str(yaw_tolerance_deg),
            )
        return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=resolved_counter_attr)

    letters = read_log(env, tensor_key)
    if letters is None:
        return _finish(torch.zeros(num_envs, dtype=torch.bool, device=device))
    letters = letters.to(device=device, dtype=torch.int64)
    if letters.dim() == 1:
        letters = letters.unsqueeze(-1)

    column = int(position) - 1
    if column < 0 or column >= letters.shape[1]:
        return _finish(torch.zeros(num_envs, dtype=torch.bool, device=device))

    from manaenv.terms.events import alphabet_token_catalog

    letter_ids = letters[:, column].reshape(-1)
    tokens = alphabet_token_catalog()
    max_letter_id = len(tokens)
    valid = (letter_ids >= 1) & (letter_ids <= max_letter_id)

    positions = []
    velocities = []
    quaternions = []
    asset_present = []
    for token in tokens:
        name = f"{block_prefix}{token}"
        if scene_has_asset(env.scene, name):
            asset = env.scene[name]
            positions.append(asset.data.root_pos_w.torch - env.scene.env_origins)
            quaternions.append(asset.data.root_quat_w.torch)
            if velocity_threshold is not None and hasattr(asset.data, "root_lin_vel_w"):
                velocities.append(torch.norm(asset.data.root_lin_vel_w.torch, dim=-1))
            else:
                velocities.append(torch.zeros(num_envs, dtype=torch.float32, device=device))
            asset_present.append(True)
        else:
            positions.append(torch.zeros((num_envs, 3), dtype=torch.float32, device=device))
            quaternions.append(torch.zeros((num_envs, 4), dtype=torch.float32, device=device))
            quaternions[-1][:, 3] = 1.0
            velocities.append(torch.zeros(num_envs, dtype=torch.float32, device=device))
            asset_present.append(False)

    all_positions = torch.stack(positions, dim=0)
    all_velocities = torch.stack(velocities, dim=0)
    all_quaternions = torch.stack(quaternions, dim=0)
    present_mask = torch.as_tensor(asset_present, dtype=torch.bool, device=device)
    env_ids = torch.arange(num_envs, dtype=torch.int64, device=device)
    selected = torch.clamp(letter_ids - 1, min=0, max=max_letter_id - 1)
    valid = valid & present_mask[selected]
    rel_pos = all_positions[selected, env_ids]
    selected_quat = all_quaternions[selected, env_ids]

    result = valid.clone()
    has_position_constraint = False
    state = getattr(env, state_attr, None)
    slot_centers = state.get("slot_centers") if isinstance(state, dict) else None
    if isinstance(slot_centers, dict):
        center_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        centers = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        for env_id in range(num_envs):
            env_centers = slot_centers.get(env_id)
            if isinstance(env_centers, list) and column < len(env_centers):
                centers[env_id] = torch.as_tensor(env_centers[column], dtype=torch.float32, device=device)
                center_valid[env_id] = True
        if torch.any(center_valid):
            result = result & center_valid
            result = result & (torch.abs(rel_pos[:, 0] - centers[:, 0]) <= float(slot_tolerance_x))
            result = result & (torch.abs(rel_pos[:, 1] - centers[:, 1]) <= float(slot_tolerance_y))
            if slot_tolerance_z is not None:
                result = result & (torch.abs(rel_pos[:, 2] - centers[:, 2]) <= float(slot_tolerance_z))
            has_position_constraint = True
    if x_range is not None:
        result = result & (rel_pos[:, 0] >= float(x_range[0])) & (rel_pos[:, 0] <= float(x_range[1]))
        has_position_constraint = True
    if y_range is not None:
        result = result & (rel_pos[:, 1] >= float(y_range[0])) & (rel_pos[:, 1] <= float(y_range[1]))
        has_position_constraint = True
    if z_range is not None:
        result = result & (rel_pos[:, 2] >= float(z_range[0])) & (rel_pos[:, 2] <= float(z_range[1]))
        has_position_constraint = True
    if not has_position_constraint:
        raise ValueError("letter_block_in_word_slot requires at least one position constraint.")

    if velocity_threshold is not None:
        selected_velocity = all_velocities[selected, env_ids]
        result = result & (selected_velocity < float(velocity_threshold))
    if yaw_target_deg is not None:
        x, y, z, w = selected_quat[:, 0], selected_quat[:, 1], selected_quat[:, 2], selected_quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        target = torch.deg2rad(torch.tensor(float(yaw_target_deg), dtype=torch.float32, device=device))
        tolerance = torch.deg2rad(torch.tensor(float(yaw_tolerance_deg), dtype=torch.float32, device=device))
        yaw_error = torch.atan2(torch.sin(yaw - target), torch.cos(yaw - target))
        result = result & (torch.abs(yaw_error) <= tolerance)
    return _finish(result)


def alphabet_word_matches_slots(
    env: ManagerBasedRLEnv,
    *,
    tensor_key: str = "target_word_letters",
    block_prefix: str = "block_alphabet_",
    state_attr: str = "_mana_alphabet_word_state",
    slot_tolerance_x: float = 0.04,
    slot_tolerance_y: float = 0.05,
    slot_tolerance_z: float | None = None,
    yaw_target_deg: float | None = None,
    yaw_tolerance_deg: float = 30.0,
    velocity_threshold: float | None = None,
    strict: bool = True,
    consecutive_steps: int = 5,
    counter_attr: str | None = None,
) -> torch.Tensor:
    """[SKILL place] Revalidate every sampled letter in its matching word slot.

    Assumes: ``tensor_key`` contains reset-committed positive token ids with
    zero padding, and :func:`letter_block_in_word_slot` owns the per-slot pose
    contract. The sampled non-zero width is the requested N, so one term works
    for all supported word lengths.
    """
    letters = read_log(env, tensor_key)
    if letters is None:
        return _false_tensor(env)
    letters = letters.to(device=env.device, dtype=torch.int64)
    if letters.dim() == 1:
        letters = letters.unsqueeze(-1)

    active = letters > 0
    raw = active.any(dim=1)
    for column in range(int(letters.shape[1])):
        slot_ok = letter_block_in_word_slot(
            env,
            position=column + 1,
            tensor_key=tensor_key,
            block_prefix=block_prefix,
            state_attr=state_attr,
            slot_tolerance_x=slot_tolerance_x,
            slot_tolerance_y=slot_tolerance_y,
            slot_tolerance_z=slot_tolerance_z,
            yaw_target_deg=yaw_target_deg,
            yaw_tolerance_deg=yaw_tolerance_deg,
            velocity_threshold=velocity_threshold,
            consecutive_steps=1,
        )
        raw = raw & ((~active[:, column]) | slot_ok)

    if strict:
        state = getattr(env, state_attr, None)
        slot_centers = state.get("slot_centers") if isinstance(state, dict) else None
        if not isinstance(slot_centers, dict):
            raw = torch.zeros_like(raw)
        else:
            from manaenv.terms.events import alphabet_token_catalog

            distractor_in_slot = torch.zeros(int(env.num_envs), dtype=torch.bool, device=env.device)
            for token_index, token in enumerate(alphabet_token_catalog()):
                name = f"{block_prefix}{token}"
                if not scene_has_asset(env.scene, name):
                    continue
                asset = env.scene[name]
                positions = asset.data.root_pos_w.torch - env.scene.env_origins
                token_is_target = (letters == token_index + 1).any(dim=1)
                token_in_slot = torch.zeros_like(distractor_in_slot)
                for env_id in range(int(env.num_envs)):
                    centers = slot_centers.get(env_id)
                    if not isinstance(centers, list):
                        continue
                    for center in centers:
                        if not isinstance(center, (list, tuple)) or len(center) < 3:
                            continue
                        inside = (
                            abs(float(positions[env_id, 0].item()) - float(center[0])) <= float(slot_tolerance_x)
                            and abs(float(positions[env_id, 1].item()) - float(center[1])) <= float(slot_tolerance_y)
                            and float(positions[env_id, 2].item()) >= -_PARKED_POOL_SLOT_Z_MARGIN_M
                        )
                        if slot_tolerance_z is not None:
                            inside = inside and (
                                abs(float(positions[env_id, 2].item()) - float(center[2])) <= float(slot_tolerance_z)
                            )
                        if inside:
                            token_in_slot[env_id] = True
                            break
                distractor_in_slot = distractor_in_slot | ((~token_is_target) & token_in_slot)
            raw = raw & (~distractor_in_slot)

    if consecutive_steps <= 1:
        return raw
    resolved_counter = counter_attr or (
        "_mana_alphabet_word_matches_slots_"
        + _stable_term_counter_id(
            str(tensor_key),
            str(block_prefix),
            str(state_attr),
            str(slot_tolerance_x),
            str(slot_tolerance_y),
            str(slot_tolerance_z),
            str(yaw_target_deg),
            str(yaw_tolerance_deg),
            str(velocity_threshold),
            str(bool(strict)),
        )
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=resolved_counter)


def alphabet_word_and_joints_near_default(
    env: ManagerBasedRLEnv,
    *,
    asset_cfg: SceneEntityCfg | str = "robot",
    joint_names: list[str] | None = None,
    exclude_joint_name_substrings: list[str] | None = None,
    atol: float = 0.4,
    tensor_key: str = "target_word_letters",
    block_prefix: str = "block_alphabet_",
    state_attr: str = "_mana_alphabet_word_state",
    slot_tolerance_x: float = 0.04,
    slot_tolerance_y: float = 0.05,
    slot_tolerance_z: float | None = None,
    yaw_target_deg: float | None = None,
    yaw_tolerance_deg: float = 30.0,
    velocity_threshold: float | None = None,
    strict: bool = True,
    consecutive_steps: int = 10,
    counter_attr: str | None = None,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL place] Require Home and the complete clean word simultaneously.

    The consecutive filter is applied once to the whole conjunction.  Keeping
    separate DAG and evaluation counters prevents the always-running scoring
    runtime from pre-warming the DAG sink before it becomes active.
    """
    word_ok = alphabet_word_matches_slots(
        env,
        tensor_key=tensor_key,
        block_prefix=block_prefix,
        state_attr=state_attr,
        slot_tolerance_x=slot_tolerance_x,
        slot_tolerance_y=slot_tolerance_y,
        slot_tolerance_z=slot_tolerance_z,
        yaw_target_deg=yaw_target_deg,
        yaw_tolerance_deg=yaw_tolerance_deg,
        velocity_threshold=velocity_threshold,
        strict=strict,
        consecutive_steps=1,
    )
    home_ok = articulation_joints_near_default(
        env,
        asset_cfg=asset_cfg,
        joint_names=joint_names,
        exclude_joint_name_substrings=exclude_joint_name_substrings,
        atol=atol,
        consecutive_steps=1,
    )
    raw = word_ok & home_ok
    if consecutive_steps <= 1:
        return raw

    runtime_kind = "evaluation" if evaluation_runtime is not None or score_runtime is not None else "dag"
    resolved_counter = counter_attr or (
        "_mana_alphabet_word_home_"
        + _stable_term_counter_id(
            runtime_kind,
            str(asset_cfg),
            str(joint_names),
            str(exclude_joint_name_substrings),
            str(atol),
            str(tensor_key),
            str(block_prefix),
            str(state_attr),
            str(slot_tolerance_x),
            str(slot_tolerance_y),
            str(slot_tolerance_z),
            str(yaw_target_deg),
            str(yaw_tolerance_deg),
            str(velocity_threshold),
            str(bool(strict)),
        )
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=resolved_counter)


def object_near_nearest_reference_child_xy(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | str,
    reference_cfg: SceneEntityCfg | str,
    reference_child_prims: Sequence[str],
    xy_distance_threshold: float = 0.005,
    velocity_threshold: float | None = 0.05,
    consecutive_steps: int = 1,
    target_child_log_key: str | None = None,
) -> torch.Tensor:
    """[SKILL place] Object root XY within ``xy_distance_threshold`` of a dock child slot.

    Default: among ``reference_child_prims`` (e.g. ``slot_a|b|c``), pick the nearest
    by env-frame root XY. When ``target_child_log_key`` is set, read that int log
    (index into ``reference_child_prims``) and only accept that assigned slot —
    inserting into a different slot fails. Optional linear-velocity settle gate.
    """
    from manaenv.utils.obb_utils import get_asset_child_prim_point_env

    object_name = _resolve_name(collapse_single_item(object_cfg))
    reference_name = _resolve_name(collapse_single_item(reference_cfg))
    children = [str(n).strip().lstrip("/") for n in (reference_child_prims or []) if str(n).strip()]
    if not children:
        return _false_tensor(env)

    target_idx: torch.Tensor | None = None
    if target_child_log_key:

        idx_log = read_log(env, str(target_child_log_key))
        if idx_log is None:
            return _false_tensor(env)
        target_idx = idx_log.to(device=env.device, dtype=torch.int64).reshape(env.num_envs, -1)[:, 0]
        target_idx = target_idx.clamp(0, len(children) - 1)

    obj_pos, obj_vel = _object_pos_vel_env(env, object_name)
    if obj_pos is None:
        return _false_tensor(env)

    thr = float(xy_distance_threshold)
    raw = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for env_id in range(int(env.num_envs)):
        origin = env.scene.env_origins[env_id]
        obj_xy = obj_pos[env_id, :2]
        if target_idx is not None:
            check_children = [children[int(target_idx[env_id].item())]]
        else:
            check_children = children
        best_name: str | None = None
        best_dist = float("inf")
        for child in check_children:
            pt = get_asset_child_prim_point_env(
                env,
                reference_name,
                env_id,
                child,
                env_origin=origin,
                device=env.device,
            )
            if pt is None:
                continue
            delta = obj_xy - pt.reshape(3)[:2]
            dist = float(torch.linalg.norm(delta).item())
            if dist < best_dist:
                best_dist = dist
                best_name = child
        vel_ok = (
            velocity_threshold is None or obj_vel is None or float(obj_vel[env_id].item()) < float(velocity_threshold)
        )
        raw[env_id] = bool(best_name is not None and best_dist < thr and vel_ok)

    if consecutive_steps <= 1:
        return raw
    counter_parts = [object_name, reference_name]
    if target_child_log_key:
        counter_parts.append(str(target_child_log_key))
    return _apply_consecutive_filter(
        env,
        raw,
        consecutive_steps,
        counter_attr="_mana_near_child_xy_" + _stable_term_counter_id(*counter_parts),
    )


def object_orientation_near_default(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | list,
    min_quat_dot: float = 0.95,
    velocity_threshold: float = 0.05,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL orient] Live root quat stays near the scene/default spawn orientation.

    Compares ``root_quat_w`` (XYZW) to ``default_root_state`` quat (layout pose,
    e.g. scene YAML ``orientation: [0, -0.7071, 0, 0.7071]`` XYZW). Tip-over fails.
    """
    if isinstance(object_cfg, list):
        result = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for single_cfg in object_cfg:
            result = result & object_orientation_near_default(
                env,
                single_cfg,
                min_quat_dot,
                velocity_threshold,
                consecutive_steps,
            )
        return result

    name = _resolve_name(object_cfg)
    obj: RigidObject = env.scene[name]
    live = obj.data.root_quat_w
    default_state = obj.data.default_root_state
    # Isaac Lab 6.0: default_root_state = [pos(3), quat_xyzw(4), ...]
    default_quat = default_state[:, 3:7]
    live_n = live / torch.linalg.norm(live, dim=-1, keepdim=True).clamp(min=1e-8)
    def_n = default_quat / torch.linalg.norm(default_quat, dim=-1, keepdim=True).clamp(min=1e-8)
    quat_dot = torch.abs(torch.sum(live_n * def_n, dim=-1))
    orient_ok = quat_dot >= float(min_quat_dot)
    vel_norm = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    vel_ok = vel_norm < float(velocity_threshold)
    raw = orient_ok & vel_ok

    if consecutive_steps <= 1:
        return raw
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr="_mana_default_orient_counter")


def inserted_reference_child_progress(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | str,
    reference_cfg: SceneEntityCfg | str,
    reference_child_prims: Sequence[str],
    template: str = "insert {object} into dock slot {index} ({slot})",
    target_child_log_key: str | None = None,
) -> list[str | None]:
    """Subtask log ``value_func``: progress text naming the dock child slot.

    Default: nearest ``reference_child_prims`` by env-frame XY. When
    ``target_child_log_key`` is set, use the reset-sampled index instead.
    Formats ``template`` with ``object``, ``index`` (1-based), ``slot``.
    """
    from manaenv.utils.obb_utils import get_asset_child_prim_point_env

    object_name = _resolve_name(object_cfg)
    reference_name = _resolve_name(reference_cfg)
    child_names = [str(n).strip().lstrip("/") for n in (reference_child_prims or []) if str(n).strip()]
    num_envs = int(env.num_envs)
    if not child_names:
        return [None] * num_envs

    target_idx: torch.Tensor | None = None
    if target_child_log_key:

        idx_log = read_log(env, str(target_child_log_key))
        if idx_log is not None:
            target_idx = idx_log.to(device=env.device, dtype=torch.int64).reshape(num_envs, -1)[:, 0]
            target_idx = target_idx.clamp(0, len(child_names) - 1)

    obj_pos, _ = _get_root_pose_env(env, object_cfg)
    out: list[str | None] = []
    for env_id in range(num_envs):
        if target_idx is not None:
            best_index = int(target_idx[env_id].item()) + 1
            best_name = child_names[best_index - 1]
        else:
            best_name = None
            best_index = -1
            best_dist = float("inf")
            obj_xy = obj_pos[env_id, :2]
            origin = env.scene.env_origins[env_id]
            for index, child in enumerate(child_names, start=1):
                pt = get_asset_child_prim_point_env(
                    env,
                    reference_name,
                    env_id,
                    child,
                    env_origin=origin,
                    device=env.device,
                )
                if pt is None:
                    continue
                dist = float(torch.linalg.norm(pt.reshape(3)[:2] - obj_xy).item())
                if dist < best_dist:
                    best_dist = dist
                    best_name = child
                    best_index = index
        if best_name is None:
            out.append(None)
            continue
        out.append(
            str(template).format(
                object=object_name,
                index=best_index,
                slot=best_name,
            )
        )
    return out


def usb_dock_slot_label(
    env: ManagerBasedRLEnv,
    log_key: str = "target_dock_slot",
    slot_labels: Sequence[str] | None = None,
) -> list[str | None]:
    """Return the reset-sampled dock slot label (A/B/C) for overlay / logs."""

    labels = [str(x) for x in (slot_labels or ["A", "B", "C"])]
    num_envs = int(env.num_envs)
    idx_log = read_log(env, str(log_key))
    if idx_log is None:
        return [None] * num_envs
    idx = idx_log.to(dtype=torch.int64).reshape(num_envs, -1)[:, 0]
    out: list[str | None] = []
    for env_id in range(num_envs):
        i = int(idx[env_id].item())
        if not labels:
            out.append(None)
            continue
        i = max(0, min(i, len(labels) - 1))
        out.append(labels[i])
    return out


def object_orientation_aligned(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg | str,
    reference_cfg: SceneEntityCfg | str,
    target_relative_orientation: list[float] | None = None,
    angle_threshold_deg: float = 15.0,
    consecutive_steps: int = 1,
    counter_attr: str | None = None,
) -> torch.Tensor:
    """Check whether an object's relative orientation matches a target.

    Computes the relative quaternion between object and reference, then
    measures the angular difference to a target relative orientation.

    Args:
        env: The environment.
        object_cfg: The object to check.
        reference_cfg: The reference object.
        target_relative_orientation: Target relative quaternion (x, y, z, w).
            If None, defaults to identity (same orientation as reference).
        angle_threshold_deg: Maximum angular difference in degrees.
        consecutive_steps: Number of consecutive steps the condition must hold.
        counter_attr: Counter attribute name for consecutive steps tracking.

    Returns:
        Boolean tensor indicating alignment for each environment.
    """
    obj = env.scene[_resolve_name(collapse_single_item(object_cfg))]
    reference = env.scene[_resolve_name(collapse_single_item(reference_cfg))]

    from isaaclab.utils import math as math_utils

    # Get quaternions (x, y, z, w)
    obj_quat = (obj.data.root_quat_w).torch
    ref_quat = (reference.data.root_quat_w).torch

    # Compute relative quaternion: q_rel = q_ref^-1 * q_obj
    ref_quat_inv = math_utils.quat_conjugate(ref_quat)
    rel_quat = math_utils.quat_mul(ref_quat_inv, obj_quat)

    # Target relative orientation
    if target_relative_orientation is not None:
        tgt = torch.tensor(target_relative_orientation, dtype=rel_quat.dtype, device=rel_quat.device)
        tgt = tgt / torch.norm(tgt)
        # Broadcast target to match the number of envs — isaaclab's quat_mul
        # asserts q1.shape == q2.shape, so (N,4) * (1,4) would raise.
        tgt_inv = math_utils.quat_conjugate(tgt.unsqueeze(0).expand(rel_quat.shape[0], 4))
        err_quat = math_utils.quat_mul(tgt_inv, rel_quat)
        cos_half_angle = torch.clamp(torch.abs(err_quat[:, 3]), 0.0, 1.0)
    else:
        cos_half_angle = torch.clamp(torch.abs(rel_quat[:, 3]), 0.0, 1.0)

    angle_rad = 2.0 * torch.acos(cos_half_angle)
    angle_deg = torch.rad2deg(angle_rad)

    result = angle_deg < angle_threshold_deg

    if consecutive_steps <= 1:
        return result

    if counter_attr is None:
        object_name = _resolve_name(collapse_single_item(object_cfg))
        reference_name = _resolve_name(collapse_single_item(reference_cfg))
        counter_attr = "_mana_object_orientation_aligned_" + _stable_term_counter_id(
            str(object_name),
            str(reference_name),
            str(target_relative_orientation),
            str(angle_threshold_deg),
        )

    return _apply_consecutive_filter(env, result, consecutive_steps, counter_attr=counter_attr)


# =====================================================================
# Atomic single-object terminations (axis-aligned)
# =====================================================================


def object_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | list,
    minimum_height: float,
    ignore_below: float | None = None,
) -> torch.Tensor:
    """Failure when any object falls below *minimum_height*.

    ``object_cfg`` may be a list; returns True if *any* object is dropped.

    ``ignore_below`` excludes objects parked far under the floor from the check:
    an object counts as dropped only when ``ignore_below < z < minimum_height``.
    Tasks that hide unused objects at a sentinel depth (e.g. ``away_z=-10``) set
    ``ignore_below`` above that sentinel so parked objects never trip the failure.
    """
    if isinstance(object_cfg, list):
        result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for single_cfg in object_cfg:
            result = result | object_dropped(env, single_cfg, minimum_height, ignore_below)
        return result

    logical_name = _resolve_name(object_cfg)
    if getattr(env, "_schema_vnext_active_asset_slots", None):
        result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for env_id in range(env.num_envs):
            active_name = resolve_active_scene_entity_name(env, logical_name, env_id=env_id)
            obj: RigidObject = env.scene[active_name]
            z = obj.data.root_pos_w.torch[env_id, 2] - env.scene.env_origins[env_id, 2]
            dropped = z < minimum_height
            if ignore_below is not None:
                dropped = dropped & (z > ignore_below)
            result[env_id] = dropped
        return result
    obj: RigidObject = env.scene[logical_name]
    z = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    dropped = z < minimum_height
    if ignore_below is not None:
        dropped = dropped & (z > ignore_below)
    return dropped


# =====================================================================
# Contact-sensor terminations
# =====================================================================


def _flatten_contact_sensor_cfgs(
    contact_sensor_cfgs: SceneEntityCfg | str | list[SceneEntityCfg] | list[str],
) -> list[Any]:
    """Flatten nested role lists into a flat list of sensor refs."""
    if not isinstance(contact_sensor_cfgs, list):
        contact_sensor_cfgs = [contact_sensor_cfgs]
    flat_cfgs: list[Any] = []
    for item in contact_sensor_cfgs:
        if isinstance(item, list):
            flat_cfgs.extend(item)
        else:
            flat_cfgs.append(item)
    if not flat_cfgs:
        raise ValueError("contact_sensor_cfgs must contain at least one contact sensor")
    return flat_cfgs


def object_contact_destination(
    env: ManagerBasedRLEnv,
    contact_sensor_cfgs: SceneEntityCfg | str | list[SceneEntityCfg] | list[str],
    force_threshold: float = 1.0,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL contact] Filtered contact force between pickable and destination stays ABOVE threshold.
    Assumes: one or more contact sensors with filter_prim_paths to the destination (force_matrix_w); consecutive-steps.

    Multiple sensors are ORed: any sensor above threshold counts as success. This
    supports schema-vNext pool cache slots, where finalize mounts one sensor per
    slot (``id``, ``id__slot_1``, ...) and only the active slot can touch the
    destination; parked spares stay away and contribute False.
    """
    flat_cfgs = _flatten_contact_sensor_cfgs(contact_sensor_cfgs)
    contact_flags = _current_contact_flags(
        env,
        contact_sensor_cfgs=flat_cfgs,
        force_threshold=force_threshold,
    )
    raw = torch.any(contact_flags, dim=-1)

    if consecutive_steps <= 1:
        return raw
    sensor_names = tuple(_resolve_name(collapse_single_item(cfg)) for cfg in flat_cfgs)
    counter_attr = "_mana_object_contact_dest_" + _stable_term_counter_id(*sensor_names)
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def object_no_contact_destination(
    env: ManagerBasedRLEnv,
    contact_sensor_cfgs: SceneEntityCfg | str | list[SceneEntityCfg] | list[str],
    force_threshold: float = 1.0,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL contact] Filtered contact force between pickable and destination stays BELOW threshold (inverse).
    Assumes: one or more filtered contact sensors on the pickable (force_matrix_w); consecutive-steps.

    Inverse of :func:`object_contact_destination` — used to assert that an
    object is NOT in contact with a surface (e.g. glasses lifted off table).
    Multiple sensors are ANDed: every sensor must stay below threshold (pool
    slot expansion mounts one sensor per cache slot; parked spares are already
    out of contact, so the active slot dominates).
    """
    flat_cfgs = _flatten_contact_sensor_cfgs(contact_sensor_cfgs)
    contact_flags = _current_contact_flags(
        env,
        contact_sensor_cfgs=flat_cfgs,
        force_threshold=force_threshold,
    )
    # no-contact == not (any sensor in contact)
    raw = ~torch.any(contact_flags, dim=-1)

    if consecutive_steps <= 1:
        return raw
    sensor_names = tuple(_resolve_name(collapse_single_item(cfg)) for cfg in flat_cfgs)
    counter_attr = "_mana_object_no_contact_dest_" + _stable_term_counter_id(*sensor_names)
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def gripper_grasps_object(
    env: ManagerBasedRLEnv,
    *,
    contact_sensor_cfg: SceneEntityCfg | str,
    object_cfg: SceneEntityCfg | str | list,
    force_threshold: float = 1.0,
    lifted_z_min: float | None = -0.145,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL grasp] Gripper contact force above ``force_threshold`` AND object lifted above ``lifted_z_min``.
    Assumes: gripper contact sensor (net_forces_w); object pose (root_pos_w); only first object of a list checked; default lifted_z_min calibrated for the EX001-6R apple scene.

    ``object_cfg`` may be a list; only the first item is checked (lift-detection
    is meaningful only for a single tracked target). Set ``lifted_z_min=None``
    to skip the lift check (pure contact).

    Default ``lifted_z_min=-0.145`` is calibrated for apple_to_basket replays
    on the EX001-6R green-booth scene (dock surface ~-0.20, apple resting
    z ~-0.19, apple peak during human demo ~-0.134 verified via
    one-off grasp diagnostics). Tighten only after re-measuring with
    diagnostics on the new scene/object.

    Designed as the building block for arm-specific subtask grasp signals
    (see ``grasp_left_1`` / ``grasp_right_1``).
    """
    sensor_name = _resolve_name(contact_sensor_cfg)
    # Multi-embodiment OR gates may reference EX001/ARX/Franka sensors in the
    # same success group; missing sensors stay False instead of KeyError.
    try:
        sensor = env.scene[sensor_name]
    except Exception:  # noqa: BLE001 — absent embodiment sensor
        return _false_tensor(env)
    forces = as_torch(sensor.data.net_forces_w)
    if forces is None:
        force_per_env = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    else:
        norm = torch.norm(forces, dim=-1)
        while norm.dim() > 1:
            norm = torch.amax(norm, dim=-1)
        force_per_env = norm
    contact_ok = force_per_env >= float(force_threshold)

    if lifted_z_min is not None:
        first = object_cfg[0] if isinstance(object_cfg, list) else object_cfg
        logical_name = _resolve_name(first)
        pos, _ = _object_pos_vel_env(env, logical_name)
        if pos is None:
            return _false_tensor(env)
        z = pos[:, 2]
        contact_ok = contact_ok & (z >= float(lifted_z_min))

    if consecutive_steps <= 1:
        return contact_ok
    object_name = _resolve_name(object_cfg[0] if isinstance(object_cfg, list) else object_cfg)
    # Per-sensor counter: left/right grasp terms are often OR'd in the same
    # success group and both evaluate every step. A shared counter lets the
    # non-grasping hand reset the streak, so one-handed grasps never reach N.
    counter_attr = "_mana_gripper_grasps_" + _stable_term_counter_id(
        sensor_name,
        object_name,
        str(force_threshold),
        str(lifted_z_min),
        str(consecutive_steps),
    )
    return _apply_consecutive_filter(env, contact_ok, consecutive_steps, counter_attr=counter_attr)


def grasp_left_1(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg | str | list,
    force_threshold: float = 1.0,
    lifted_z_min: float | None = -0.145,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL grasp] Per-arm subtask signal: left arm grasps and lifts ``object_cfg`` (wraps gripper_grasps_object on left_gripper_contact).
    Assumes: EX001-6R ``left_gripper_contact`` sensor pre-registered on the embodiment; consumed by name as a mimic subtask-term signal.

    Reads the EX001-6R ``left_gripper_contact`` sensor pre-registered on the
    embodiment. Used by the mimic env subtask boundary parser.
    Default ``lifted_z_min=-0.145`` matches :func:`gripper_grasps_object`.
    """
    return gripper_grasps_object(
        env,
        contact_sensor_cfg="left_gripper_contact",
        object_cfg=object_cfg,
        force_threshold=force_threshold,
        lifted_z_min=lifted_z_min,
        consecutive_steps=consecutive_steps,
    )


def grasp_right_1(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg | str | list,
    force_threshold: float = 1.0,
    lifted_z_min: float | None = -0.145,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL grasp] Per-arm subtask signal: right arm grasps and lifts ``object_cfg`` (wraps gripper_grasps_object on right_gripper_contact).
    Assumes: EX001-6R ``right_gripper_contact`` sensor pre-registered on the embodiment; consumed by name as a mimic subtask-term signal.

    Default ``lifted_z_min=-0.145`` matches :func:`gripper_grasps_object`.
    """
    return gripper_grasps_object(
        env,
        contact_sensor_cfg="right_gripper_contact",
        object_cfg=object_cfg,
        force_threshold=force_threshold,
        lifted_z_min=lifted_z_min,
        consecutive_steps=consecutive_steps,
    )


def object_above_height(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    min_height: float = -0.15,
) -> torch.Tensor:
    """[SKILL lift] Object env-frame z-position >= ``min_height`` (raw absolute height, no anchor).
    Assumes: object pose (root_pos_w).

    Runtime-randomized pool objects are resolved to their per-env active cache
    slot (same path as ``object_position_in_zone`` / ``object_dropped``).
    """
    object_name = _resolve_name(collapse_single_item(object_cfg))
    pos, _ = _object_pos_vel_env(env, object_name)
    if pos is None:
        return _false_tensor(env)
    return pos[:, 2] >= min_height


# =====================================================================
# Data-collection atomic skills
#
# The following terms are task-success primitives used by selected evaluation
# suites. They are deliberately minimal and may be tightened as tasks evolve.
# =====================================================================


# =====================================================================
# Atomic skill terms (object-pose / phase independent building blocks).
#
# These four terms are the canonical primitives behind the "atomic skill
# success" pattern documented in ``wiki/architecture-plan.md`` and in
# ``manaenv/pipeline/skills/base.py``.  Each term answers a single
# geometric / contact question and is composed inside a task YAML by the
# per-skill aggregator (``pick`` / ``place`` / ``push`` / ...).
# =====================================================================


def pick(
    env: ManagerBasedRLEnv,
    _sub_term_defs: list[dict],
    mode: str = "and",
) -> torch.Tensor:
    """[ATOMIC SKILL AGGREGATOR] ``pick`` task-level success predicate.

    Semantically equivalent to :func:`success` but with an explicit name so
    task YAML self-documents intent: a ``pick`` sub-term in a task's
    ``success`` block bundles ``gripper_close_and_contact``,
    ``move_target`` (gripper near object), ``object_lifted`` and any other
    small terms that together define "the object has been picked up".

    Sister aggregators (``place``, ``push``, ``pull`` ...) follow the
    same pattern as more skills land. We split them by name instead of
    introducing a generic ``all_of`` so per-skill aggregators can later
    grow skill-specific reporting / debug hooks without affecting siblings.
    """
    return success(env, _sub_term_defs, mode=mode)


def move(
    env: ManagerBasedRLEnv,
    _sub_term_defs: list[dict],
    mode: str = "and",
) -> torch.Tensor:
    """[ATOMIC SKILL AGGREGATOR] ``move`` task-level success predicate.

    Compose object/EE position terms that define whether an intermediate
    relative move reached its intended region while keeping the object held.
    """
    return success(env, _sub_term_defs, mode=mode)


def place(
    env: ManagerBasedRLEnv,
    _sub_term_defs: list[dict],
    mode: str = "and",
) -> torch.Tensor:

    return success(env, _sub_term_defs, mode=mode)


def home(
    env: ManagerBasedRLEnv,
    _sub_term_defs: list[dict],
    mode: str = "and",
) -> torch.Tensor:

    return success(env, _sub_term_defs, mode=mode)


def touch(
    env: ManagerBasedRLEnv,
    _sub_term_defs: list[dict],
    mode: str = "and",
) -> torch.Tensor:

    return success(env, _sub_term_defs, mode=mode)


def push(
    env: ManagerBasedRLEnv,
    _sub_term_defs: list[dict],
    mode: str = "and",
) -> torch.Tensor:

    return success(env, _sub_term_defs, mode=mode)


def move_target(
    env: ManagerBasedRLEnv,
    *,
    ee_frame_key: str,
    target_pos: list[float] | tuple[float, float, float],
    max_distance: float,
    tip_offset: list[float] | tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """[SKILL reach] EE tip within ``max_distance`` of a fixed env-frame target point.
    Assumes: one EE FrameTransformer frame (target_pos_w/target_quat_w); optional tip_offset in EE local frame; object-agnostic (caller computes target_pos upstream).

    ``target_pos`` is in env-local coordinates and is broadcast to world by
    adding ``env.scene.env_origins`` per env. Tip is the EE frame origin
    optionally shifted by ``tip_offset`` (EE local frame). Object-agnostic
    by design: callers compute ``target_pos`` upstream from whatever
    semantics they need (object pose + offset, named keypoint, ...).
    """
    from isaaclab.utils import math as math_utils

    frame = env.scene[ee_frame_key]
    ee_pos_w = frame.data.target_pos_w.torch[:, 0, :]
    if tip_offset is not None and any(abs(float(v)) > 0.0 for v in tip_offset):
        ee_quat_w = frame.data.target_quat_w.torch[:, 0, :]
        offset_local = torch.tensor(list(tip_offset), dtype=ee_pos_w.dtype, device=ee_pos_w.device)
        offset_local = offset_local.unsqueeze(0).expand(ee_pos_w.shape[0], -1)
        tip_pos_w = ee_pos_w + math_utils.quat_apply(ee_quat_w, offset_local)
    else:
        tip_pos_w = ee_pos_w

    env_origins = env.scene.env_origins
    target_env = torch.tensor(list(target_pos), dtype=env_origins.dtype, device=env_origins.device)
    target_pos_w = env_origins + target_env.unsqueeze(0).expand_as(env_origins)

    distance = torch.linalg.norm(target_pos_w - tip_pos_w, dim=-1)
    return distance <= float(max_distance)


def ee_near_init_pose(
    env: ManagerBasedRLEnv,
    *,
    ee_frame_keys: list[str],
    max_distance: float,
    offset_local: list[float] | tuple[float, float, float] | None = None,
    init_state_attr: str = "_mana_init_ee_pos_w",
) -> torch.Tensor:
    """[SKILL reach] All EE frames within ``max_distance`` of their episode-init pose (+offset); ANDs across arms.
    Assumes: EE FrameTransformer frames (target_pos_w); capture_init_ee_poses event ran at reset (env._mana_init_ee_pos_w).

    Result-based success predicate for the ``prep`` / ``return`` stages whose
    target is a *dynamic* pose (the episode-start EE pose), which fixed-point
    terms like :func:`move_target` cannot express. Requires
    :func:`manaenv.terms.events.capture_init_ee_poses` to have run at reset,
    freezing each frame's episode-start world position into
    ``env.<init_state_attr>[frame_key]`` ``[num_envs, 3]``.

    ``offset_local`` is an env-frame translation added to the captured init
    position (``prep`` uses a pregrasp offset; ``return`` uses ``None`` = back
    to the start pose). The result ANDs across all arms.
    """
    init_store = getattr(env, init_state_attr, None)
    if not isinstance(init_store, dict):
        # Capture event has not run yet → not satisfied.
        return _false_tensor(env)

    offset = None
    if offset_local is not None:
        offset = torch.tensor(list(offset_local), dtype=torch.float32, device=env.device).reshape(3)

    within = torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)
    for frame_key in ee_frame_keys:
        init_pos = init_store.get(frame_key)
        if not isinstance(init_pos, torch.Tensor):
            return _false_tensor(env)
        frame = env.scene[frame_key]
        cur_pos_w = frame.data.target_pos_w.torch[:, 0, :]
        target = init_pos if offset is None else init_pos + offset.unsqueeze(0)
        distance = torch.linalg.norm(target - cur_pos_w, dim=-1)
        within = within & (distance <= float(max_distance))
    return within


def ee_near_object(
    env: ManagerBasedRLEnv,
    *,
    ee_frame_keys: list[str],
    object_cfgs: SceneEntityCfg | str | list,
    max_distance: float,
    offset_local: list[float] | tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """[SKILL reach] ANY EE frame within ``max_distance`` of the target object (+offset); ORs across arms.
    Assumes: EE FrameTransformer frames; object pool pose (root_pos_w); respects per-env pinned grasp target (EPISODE_GRASP_TARGET_ATTR) when set.

    Result-based success predicate for the ``reach`` stage: the target is the
    *live* grasp-target object pose (re-read each step; no capture needed),
    which moves per episode. Distance is full 3D (the shelf rows differ in Z, so
    XY-only terms like :func:`robot_body_near_object_xy` do not suffice).

    ANY-arm (OR) semantics, not all-arm: the planner auto-selects which arm
    grasps per episode (``touch`` skill ``arm:"auto"``), so the YAML cannot name
    the arm — reach succeeds when whichever arm is doing it gets to the bottle.
    Contrast :func:`ee_near_init_pose` (return stage), which ANDs across arms.

    ``object_cfgs`` is the grasp-target pool (the ``_cfgs`` suffix marks it as a
    whole-list param so a list-valued ``@role`` is passed in intact, NOT expanded
    into one term per element); the nearest of any listed object counts.
    ``offset_local`` is an env-frame grasp offset added to the object position.
    Compared in env-local frame (EE world pos minus env origins) so multi-env
    offsets cancel.

    **Per-episode target narrowing (task authority):** when the env carries a
    pinned grasp target (``env.<EPISODE_GRASP_TARGET_ATTR>`` = ``{env_id: name}``,
    written by the task/scene selection event), the pool is restricted to the
    pinned object FOR THAT ENV — so reach succeeds only at the bottle the task
    chose, matching the prompt and the curobo goal. Unpinned envs (or a None pin)
    fall back to the whole-pool OR. The checker is thus the single thing that
    selection narrows; nothing recomputes the target downstream.
    """
    object_cfgs = object_cfgs if isinstance(object_cfgs, list) else [object_cfgs]
    object_cfgs = [c for c in object_cfgs if c is not None]
    if not object_cfgs:
        return _false_tensor(env)

    offset = None
    if offset_local is not None:
        offset = torch.tensor(list(offset_local), dtype=torch.float32, device=env.device).reshape(3)

    pin_store = getattr(env, EPISODE_GRASP_TARGET_ATTR, None)
    pin = pin_store if isinstance(pin_store, dict) and pin_store else None

    # Target positions (env-local) + resolved name, one per candidate object.
    object_names = [_resolve_name(cfg) for cfg in object_cfgs]
    targets: list[torch.Tensor] = []
    for cfg in object_cfgs:
        obj_pos_env, _ = _get_root_pose_env(env, cfg)
        obj_pos_env = obj_pos_env.to(dtype=torch.float32, device=env.device)
        targets.append(obj_pos_env if offset is None else obj_pos_env + offset.unsqueeze(0))

    near_any = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    for frame_key in ee_frame_keys:
        frame = env.scene[frame_key]
        ee_pos_env = frame.data.target_pos_w.torch[:, 0, :] - env.scene.env_origins
        for name, target in zip(object_names, targets):
            distance = torch.linalg.norm(target - ee_pos_env, dim=-1)
            hit = distance <= float(max_distance)
            if pin is not None:
                # Object active for env e iff e is unpinned (or pinned to None →
                # fall back to whole pool) or e's pinned target equals this name.
                active = torch.tensor(
                    [pin.get(e) in (None, name) for e in range(env.num_envs)],
                    dtype=torch.bool,
                    device=env.device,
                )
                hit = hit & active
            near_any = near_any | hit
    return near_any


def gripper_close_and_contact(
    env: ManagerBasedRLEnv,
    *,
    gripper_contact_sensor: str | list[str],
    gripper_joint_name: str | list[str],
    gripper_closed_below: float = 1.0,
    force_threshold: float = 0.5,
    robot_cfg: SceneEntityCfg | str = "robot",
    arm_combine: str = "all",
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL grasp] Per-arm (gripper closed AND contact), combined across arms (all/any).
    Assumes: robot articulation with named gripper joint(s) (joint_pos); per-arm contact sensor(s) (net_forces_w); single-arm scalar or bimanual paired lists.

    Single-arm: pass scalar ``gripper_contact_sensor`` / ``gripper_joint_name``.
    Bimanual: pass paired lists (same length) — element ``i`` is arm ``i``'s
    sensor + joint. Combine across arms with ``arm_combine="all"`` (both
    arms must hold their own object — bimanual co-grasp) or ``"any"`` (at
    least one arm holds the object — sequential / either-arm pick).

    Per-arm gate ``closed_i & contact_i`` follows the original semantics:
    fingers commanded closed AND contact sensor reads above threshold.
    Object identity is intentionally absent so the term is reusable across
    pick / push / pull / etc.
    """
    sensors = [gripper_contact_sensor] if isinstance(gripper_contact_sensor, str) else list(gripper_contact_sensor)
    joints = [gripper_joint_name] if isinstance(gripper_joint_name, str) else list(gripper_joint_name)
    if not sensors or not joints:
        raise ValueError("gripper_close_and_contact: gripper_contact_sensor / gripper_joint_name must be non-empty")
    if len(sensors) != len(joints):
        raise ValueError(
            "gripper_close_and_contact: gripper_contact_sensor and gripper_joint_name must have the same length, "
            f"got sensors={len(sensors)}, joints={len(joints)}"
        )
    if arm_combine not in ("all", "any"):
        raise ValueError(f"gripper_close_and_contact: arm_combine must be 'all' or 'any', got '{arm_combine}'")

    robot = env.scene[_resolve_name(robot_cfg)]
    joint_names = list(robot.data.joint_names)

    arm_results: list[torch.Tensor] = []
    for sensor_key, joint_key in zip(sensors, joints):
        if joint_key not in joint_names:
            raise KeyError(
                f"gripper_close_and_contact: gripper_joint_name='{joint_key}' not found in "
                f"robot '{_resolve_name(robot_cfg)}' joints={joint_names}"
            )
        joint_idx = joint_names.index(joint_key)
        gripper_pos = robot.data.joint_pos.torch[:, joint_idx]
        closed = gripper_pos <= float(gripper_closed_below)

        sensor = env.scene[sensor_key]
        net_forces = as_torch(sensor.data.net_forces_w)
        if net_forces is None:
            raise RuntimeError(
                f"gripper_close_and_contact: contact sensor '{sensor_key}' has no net_forces_w; "
                "ensure the sensor is enabled and stepped before this term runs"
            )
        norm = torch.linalg.norm(net_forces, dim=-1)
        while norm.dim() > 1:
            norm = torch.amax(norm, dim=-1)
        contact = norm >= float(force_threshold)
        arm_results.append(closed & contact)

    if arm_combine == "all":
        raw = arm_results[0]
        for r in arm_results[1:]:
            raw = raw & r
    else:
        raw = arm_results[0]
        for r in arm_results[1:]:
            raw = raw | r

    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_close_and_contact_" + _stable_term_counter_id(
        ",".join(sensors),
        ",".join(joints),
        f"{float(gripper_closed_below):.4f}",
        f"{float(force_threshold):.4f}",
        arm_combine,
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def any_object_displaced_from_anchor(
    env: ManagerBasedRLEnv,
    *,
    object_cfgs: SceneEntityCfg | str | list,
    min_displacement: float = 0.025,
    xy_only: bool = True,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL multi-object] Detect any listed object displaced from its stage anchor.
    Assumes: every listed object exposes a root pose and the term is evaluated once before displacement.

    The anchor for each object is locked on the first evaluation after episode
    reset (typically the first frame of the stage that owns this term). Episode
    resets clear anchors via ``episode_length_buf <= 1``. Use param name
    ``object_cfgs`` (``_cfgs`` suffix) so a multi-asset ``@role`` stays a list.
    """
    cfgs = object_cfgs if isinstance(object_cfgs, list) else [object_cfgs]
    names = [_resolve_name(collapse_single_item(cfg)) for cfg in cfgs]
    if not names:
        return _false_tensor(env)

    n_obj = len(names)
    anchor_attr = "_mana_displace_anchor_" + _stable_term_counter_id(*names, "xy" if xy_only else "xyz")
    just_reset = env.episode_length_buf <= 1
    anchors = getattr(env, anchor_attr, None)
    if anchors is None or (isinstance(anchors, torch.Tensor) and anchors.shape != (env.num_envs, n_obj, 3)):
        anchors = torch.full((env.num_envs, n_obj, 3), float("nan"), dtype=torch.float32, device=env.device)
        setattr(env, anchor_attr, anchors)
    _register_tensor_episode_state(env, anchor_attr, fill_value=float("nan"))
    if just_reset.any():
        anchors[just_reset] = float("nan")

    positions = torch.stack([_get_root_pose_env(env, name)[0] for name in names], dim=1)
    unset = torch.isnan(anchors[:, 0, 0])
    if unset.any():
        anchors[unset] = positions[unset]

    delta = positions - anchors
    dist = torch.linalg.norm(delta[..., :2], dim=-1) if xy_only else torch.linalg.norm(delta, dim=-1)
    raw = dist >= float(min_displacement)
    result = raw.any(dim=-1)

    if consecutive_steps <= 1:
        return result
    counter_attr = "_mana_displace_counter_" + _stable_term_counter_id(*names, f"{float(min_displacement):.4f}")
    return _apply_consecutive_filter(env, result, consecutive_steps, counter_attr=counter_attr)


def object_lifted(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg | str,
    min_lift: float = 0.05,
    settle_steps: int = 0,
    contact_sensor: str | list[str] | None = None,
    force_threshold: float = 0.5,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL lift] Object env-z risen >= ``min_lift`` m above an anchor (settle-time or first-contact).
    Assumes: object pose (root_pos_w); episode_length_buf; optional contact sensor(s) (net_forces_w) for grasp-moment anchoring.

    Anchor mode:
    * ``contact_sensor=None``  — anchor is the object env-z once
      ``episode_length_buf >= settle_steps``. This avoids requiring a contact
      sensor when the desired baseline is the post-settle height.
    * ``contact_sensor=<key>`` (or list, OR-combined for bimanual) —
      anchor is locked at the *grasp moment*: the first frame any of the
      listed contact sensors reports a force above ``force_threshold``.
      Recommended pairing for ``pick``: this captures the object's z right
      when fingers first close on it, so ``min_lift`` measures the actual
      planner lift instead of being polluted by pre-grasp settling.
    Anchor resets every episode and stays NaN (term returns False) until
    the trigger fires.

    Runtime pool objects are resolved to the per-env active cache slot (same
    path as ``object_dropped`` / zone terms). ``env.scene[logical]`` alone
    watches slot 0, which is often parked below the floor.
    """
    logical_name = _resolve_name(object_cfg)
    pos, _ = _object_pos_vel_env(env, logical_name)
    if pos is None:
        return _false_tensor(env)
    z = pos[:, 2]
    just_reset = env.episode_length_buf <= 1
    settle_steps = max(0, int(settle_steps))

    sensor_keys = (
        []
        if contact_sensor is None
        else ([contact_sensor] if isinstance(contact_sensor, str) else list(contact_sensor))
    )
    anchor_attr = "_mana_lifted_anchor_z_" + _stable_term_counter_id(
        logical_name,
        "settle" if contact_sensor is None else "contact",
        ",".join(sensor_keys),
        str(settle_steps),
        f"{float(force_threshold):.6g}",
    )
    if not hasattr(env, anchor_attr):
        setattr(env, anchor_attr, torch.full_like(z, float("nan")))
    _register_tensor_episode_state(env, anchor_attr, fill_value=float("nan"))
    anchor_z: torch.Tensor = getattr(env, anchor_attr)
    anchor_z = torch.where(just_reset, torch.full_like(z, float("nan")), anchor_z)

    if contact_sensor is None:
        ready_to_anchor = env.episode_length_buf >= settle_steps
        anchor_z = torch.where(torch.isnan(anchor_z) & ready_to_anchor, z, anchor_z)
    else:
        in_contact = torch.zeros_like(z, dtype=torch.bool)
        for sensor_key in sensor_keys:
            sensor = env.scene[sensor_key]
            net_forces = as_torch(sensor.data.net_forces_w)
            if net_forces is None:
                raise RuntimeError(
                    f"object_lifted: contact sensor '{sensor_key}' has no net_forces_w; "
                    "ensure the sensor is enabled and stepped before this term runs"
                )
            norm = torch.linalg.norm(net_forces, dim=-1)
            while norm.dim() > 1:
                norm = torch.amax(norm, dim=-1)
            in_contact = in_contact | (norm >= float(force_threshold))
        first_contact = torch.isnan(anchor_z) & in_contact
        anchor_z = torch.where(first_contact, z, anchor_z)

    setattr(env, anchor_attr, anchor_z)

    anchored = ~torch.isnan(anchor_z)
    raw = anchored & ((z - torch.nan_to_num(anchor_z, nan=0.0)) >= float(min_lift))

    if consecutive_steps <= 1:
        return raw
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=anchor_attr + "_streak")


def indexed_object_lifted(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    index_log_key: str,
    min_lift: float = 0.05,
    settle_steps: int = 0,
    contact_sensor: str | list[str] | None = None,
    force_threshold: float = 0.5,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL lift] Lift check on the single object selected by a reset-committed index.
    Assumes: ``index_log_key`` stores one candidate-aligned integer per env and candidate root poses are available.

    ``object_names`` is ordered to match ``index_log_key`` (int ``[num_envs, 1]``).
    Evaluates :func:`object_lifted` per candidate and gathers the per-env target.
    """

    names = [str(name) for name in object_names]
    if not names:
        raise ValueError("indexed_object_lifted requires a non-empty object_names")
    idx_log = read_log(env, str(index_log_key))
    if idx_log is None:
        return _false_tensor(env)
    idx_log = idx_log.to(device=env.device, dtype=torch.int64).reshape(env.num_envs, -1)[:, 0]
    idx_log = idx_log.clamp(0, len(names) - 1)

    per_object = [
        object_lifted(
            env,
            object_cfg=name,
            min_lift=min_lift,
            settle_steps=settle_steps,
            contact_sensor=contact_sensor,
            force_threshold=force_threshold,
            consecutive_steps=1,
        )
        for name in names
    ]
    stacked = torch.stack(per_object, dim=0)
    env_ids = torch.arange(env.num_envs, device=env.device)
    raw = stacked[idx_log, env_ids]
    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_indexed_object_lifted_" + _stable_term_counter_id(*names, str(index_log_key))
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def grasp_target_lifted(
    env: ManagerBasedRLEnv,
    *,
    object_cfgs: SceneEntityCfg | str | list,
    sensor_suffix: str = "gripper_contact_sensor",
    min_lift: float = 0.05,
    force_threshold: float = 0.5,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL lift] The episode's pinned grasp-target object is contacted AND lifted (lift implies grasp).
    Assumes: per-object contact sensors by name convention ``{object}_{sensor_suffix}``; per-env pin (EPISODE_GRASP_TARGET_ATTR); object pose; pool+pin model mirrors ee_near_object.

    Result-based success predicate for the ``grasp`` stage. Mirrors
    :func:`ee_near_object`'s pool+pin model so it composes with the SAME
    per-episode grasp-target selection event the reach stage uses: ``object_cfgs``
    is the whole grasp-target pool (the ``_cfgs`` suffix marks it a whole-list
    param so a list-valued ``@role`` passes in intact, NOT expanded into one term
    per bottle — which is what makes a 12-object pool coexist with 30 shelf
    sensors without a length clash), and the per-env pin
    (``env.<EPISODE_GRASP_TARGET_ATTR>``) narrows it to the one bottle the task
    chose. Each candidate's gripper-contact sensor is derived by the scene's name
    convention ``f"{object_name}_{sensor_suffix}"`` (mounted on every shelf bottle
    by the stage base), so there is no parallel-list role to misalign.

    Semantics ("接触+抬起"): for the active (pinned) object the lift anchor is
    locked at the *grasp moment* — the first frame its OWN contact sensor reports
    force >= ``force_threshold`` — and success is that object's env-z having risen
    >= ``min_lift`` above the anchor. A bottle lifted without its sensor firing
    never anchors, so "lifted" here always implies "was grasped". Anchor resets
    each episode and stays NaN (term False) until contact fires. Unpinned envs
    fall back to whole-pool OR (mirrors :func:`ee_near_object`).

    The DAG gating (reach → grasp) guarantees the EE is already at the pinned
    bottle before this stage is scored, so the anchored/lifted object is the
    target — not some other bottle nudged on the shelf.
    """
    object_cfgs = object_cfgs if isinstance(object_cfgs, list) else [object_cfgs]
    object_cfgs = [c for c in object_cfgs if c is not None]
    if not object_cfgs:
        return _false_tensor(env)

    names = [_resolve_name(cfg) for cfg in object_cfgs]
    num_objects = len(names)

    pin_store = getattr(env, EPISODE_GRASP_TARGET_ATTR, None)
    pin = pin_store if isinstance(pin_store, dict) and pin_store else None

    # Per-object env-frame z + own-sensor contact (num_envs, num_objects).
    all_z = torch.zeros((env.num_envs, num_objects), dtype=torch.float32, device=env.device)
    in_contact = torch.zeros((env.num_envs, num_objects), dtype=torch.bool, device=env.device)
    for i, name in enumerate(names):
        obj = env.scene[name]
        all_z[:, i] = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        sensor = env.scene[f"{name}_{sensor_suffix}"]
        forces = as_torch(sensor.data.net_forces_w)
        if forces is None:
            continue
        norm = torch.linalg.norm(forces, dim=-1)
        while norm.dim() > 1:
            norm = torch.amax(norm, dim=-1)
        in_contact[:, i] = norm >= float(force_threshold)

    # Grasp-moment anchor per object, reset each episode (mirror object_lifted).
    anchor_attr = "_mana_grasp_target_anchor_z_" + _stable_term_counter_id(
        ",".join(names), f"{float(force_threshold):.6g}"
    )
    if not hasattr(env, anchor_attr):
        setattr(env, anchor_attr, torch.full((env.num_envs, num_objects), float("nan"), device=env.device))
    _register_tensor_episode_state(env, anchor_attr, fill_value=float("nan"))
    anchor_z: torch.Tensor = getattr(env, anchor_attr)
    just_reset = env.episode_length_buf <= 1
    anchor_z = torch.where(just_reset.unsqueeze(1), torch.full_like(anchor_z, float("nan")), anchor_z)
    first_contact = torch.isnan(anchor_z) & in_contact
    anchor_z = torch.where(first_contact, all_z, anchor_z)
    setattr(env, anchor_attr, anchor_z)

    anchored = ~torch.isnan(anchor_z)
    lifted = anchored & ((all_z - torch.nan_to_num(anchor_z, nan=0.0)) >= float(min_lift))

    # Narrow to the pinned target per env (unpinned → whole-pool OR).
    result = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    for i, name in enumerate(names):
        hit = lifted[:, i]
        if pin is not None:
            active = torch.tensor(
                [pin.get(e) in (None, name) for e in range(env.num_envs)],
                dtype=torch.bool,
                device=env.device,
            )
            hit = hit & active
        result = result | hit

    if consecutive_steps <= 1:
        return result
    return _apply_consecutive_filter(env, result, consecutive_steps, counter_attr=anchor_attr + "_streak")


def object_tilt_in_range(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | list,
    min_dot: float = 0.9,
    max_dot: float = 1.0,
    velocity_threshold: float = 0.5,
    consecutive_steps: int = 1,
    axis: str = "z",
    min_abs_dot: float | None = None,
) -> torch.Tensor:
    """[SKILL orient] Object local ``axis`` alignment with world up (dot) within [*min_dot*, *max_dot*] (e.g. upright/flat).
    Assumes: object orientation (root_quat_w, xyzw) + velocity (root_lin_vel_w); list ANDs across all.

    Computes ``dot(R @ local_axis, [0,0,1])`` from the quaternion, where ``local_axis`` is
    selected by ``axis`` (``x``/``y``/``z``). For a quaternion ``(x, y, z, w)``:
    * ``axis='z'`` -> ``1 - 2*(x² + y²)``
    * ``axis='x'`` -> ``2*(xz - wy)``
    * ``axis='y'`` -> ``2*(yz + wx)``

    ``axis`` selects which model axis is the "long"/reference axis, so assets
    whose upright direction is not the model Z (e.g. flowers authored lying along
    model +X) can still be checked. ``object_cfg`` may be a list; all objects
    must satisfy the condition. Runtime-randomized pool objects are resolved to
    their per-env active cache slot (same path as ``object_position_in_zone``).

    When ``min_abs_dot`` is set, the signed [*min_dot*, *max_dot*] window is ignored
    and success requires ``abs(up_dot) >= min_abs_dot`` (axis nearly vertical either
    polarity — useful when stem may be authored along +X or −X).

    Use cases:
    * Cup upright:  ``min_dot=0.9,  max_dot=1.0``  (local Z ≈ world Z)
    * Object flat:  ``min_dot=-0.1, max_dot=0.1``   (local Z ⊥ world Z)
    * Stem-first insert: ``axis='x', min_dot=-1.0, max_dot=-0.85`` (local X ≈ world down)
    * Stem vertical either way: ``axis='x', min_abs_dot=0.75``
    """
    if isinstance(object_cfg, list):
        result = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for single_cfg in object_cfg:
            result = result & object_tilt_in_range(
                env,
                single_cfg,
                min_dot,
                max_dot,
                velocity_threshold,
                consecutive_steps,
                axis,
                min_abs_dot,
            )
        return result

    from isaaclab.utils import math as math_utils

    # Resolve runtime pool active cache slots (same path as object_position_in_zone).
    # env.scene[logical_name] alone watches slot 0, which is often parked away.
    object_name = _resolve_name(collapse_single_item(object_cfg))
    quat = _object_quat_xyzw(env, object_name)
    if quat is None:
        return _false_tensor(env)
    _, vel = _object_pos_vel_env(env, object_name)
    rotation = math_utils.matrix_from_quat(quat)
    axis_key = str(axis).lower()
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis_key)
    if axis_index is None:
        raise ValueError(f"object_tilt_in_range: invalid axis {axis!r}, expected one of 'x', 'y', 'z'")
    up_dot = rotation[:, 2, axis_index]

    if min_abs_dot is not None:
        in_range = up_dot.abs() >= float(min_abs_dot)
    else:
        in_range = (up_dot >= min_dot) & (up_dot <= max_dot)
    if vel is None:
        raw = in_range
    else:
        raw = in_range & (vel < velocity_threshold)

    if consecutive_steps <= 1:
        return raw
    # Per-object counter: parallel place stages (or OR'd polarity checks) must not
    # share one streak, or a failing sibling term resets the count every step.
    counter_attr = "_mana_tilt_counter_" + _stable_term_counter_id(
        object_name,
        axis_key,
        str(min_dot),
        str(max_dot),
        str(min_abs_dot),
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def object_euler_xy_out_of_range(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | list,
    min_deg: float = -40.0,
    max_deg: float = 40.0,
    velocity_threshold: float | None = 0.5,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL orient] Object roll or pitch OUTSIDE ``[min_deg, max_deg]`` (negative-sense: tumble/fallen detection).
    Assumes: object orientation (root_quat_w, xyzw) + optional velocity gate; list ORs across any.
    """
    if isinstance(object_cfg, list):
        result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for single_cfg in object_cfg:
            result = result | object_euler_xy_out_of_range(
                env,
                single_cfg,
                min_deg=min_deg,
                max_deg=max_deg,
                velocity_threshold=velocity_threshold,
                consecutive_steps=consecutive_steps,
            )
        return result

    obj: RigidObject = env.scene[_resolve_name(object_cfg)]
    roll, pitch, _ = euler_xyz_from_quat_xyzw_tensor(obj.data.root_quat_w.torch)
    roll_deg = torch.rad2deg(roll)
    pitch_deg = torch.rad2deg(pitch)
    out_of_range = (roll_deg < min_deg) | (roll_deg > max_deg) | (pitch_deg < min_deg) | (pitch_deg > max_deg)
    vel_ok = _velocity_below_threshold(obj, velocity_threshold, env.num_envs, env.device)
    raw = out_of_range & vel_ok

    if consecutive_steps <= 1:
        return raw
    counter_attr = f"_mana_euler_xy_out_of_range_counter_{_stable_term_counter_id(_resolve_name(object_cfg))}"
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def object_yaw_in_range(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    min_yaw_deg: float = 80.0,
    max_yaw_deg: float = 100.0,
    velocity_threshold: float = 0.5,
    consecutive_steps: int = 1,
    yaw_ranges: list | None = None,
    signed: bool = False,
) -> torch.Tensor:
    """[SKILL orient] |yaw| of object within [*min_yaw_deg*, *max_yaw_deg*] (or any of *yaw_ranges*).
    Assumes: object orientation (root_quat_w, xyzw) + velocity (root_lin_vel_w).

    By default compares ``|yaw|`` (degrees in ``[0, 180]``). Set *signed* to compare
    the signed yaw in ``[-180, 180]`` instead (e.g. target ≈ -90°).

    If *yaw_ranges* is provided (list of ``[min, max]`` pairs), the condition
    is satisfied when the (optionally signed) yaw falls in **any** of the given
    ranges (OR logic). *min_yaw_deg* / *max_yaw_deg* are ignored when *yaw_ranges*
    is set.

    Runtime-randomized pool objects are resolved to their per-env active cache
    slot (same path as ``object_tilt_in_range`` / ``object_position_in_zone``).
    """
    object_name = _resolve_name(collapse_single_item(object_cfg))
    quat = _object_quat_xyzw(env, object_name)
    if quat is None:
        return _false_tensor(env)
    _, vel = _object_pos_vel_env(env, object_name)
    _, _, yaw = euler_xyz_from_quat_xyzw_tensor(quat)
    yaw_deg = torch.rad2deg(yaw)
    if not signed:
        yaw_deg = torch.abs(yaw_deg)

    if vel is None:
        vel_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        vel_ok = vel < velocity_threshold

    if yaw_ranges is not None:
        in_range = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for lo, hi in yaw_ranges:
            in_range = in_range | ((yaw_deg >= lo) & (yaw_deg <= hi))
    else:
        in_range = (yaw_deg >= min_yaw_deg) & (yaw_deg <= max_yaw_deg)

    raw = in_range & vel_ok

    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_yaw_counter_" + _stable_term_counter_id(
        object_name,
        str(signed),
        str(min_yaw_deg),
        str(max_yaw_deg),
        str(yaw_ranges),
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def objects_equidistant(
    env: ManagerBasedRLEnv,
    objects: list,
    distance_tolerance: float = 0.02,
    velocity_threshold: float = 0.5,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL multi-object] Consecutive objects equidistantly spaced (max−min spacing < tolerance).
    Assumes: >=3 objects with pose (root_pos_w) + velocity (root_lin_vel_w).

    Computes distances between consecutive pairs ``(objects[0]–objects[1],
    objects[1]–objects[2], …)`` and checks that
    ``max(distances) − min(distances) < distance_tolerance``,
    i.e. all spacings are approximately equal.

    *objects* is a list of scene entity names or ``SceneEntityCfg``
    instances (≥ 3 elements), resolved from role references such as
    ``["@pickable_1", "@pickable_2", "@pickable_3"]``.
    """
    if len(objects) < 3:
        raise ValueError(f"objects_equidistant requires >= 3 objects, got {len(objects)}")

    # Consecutive pairwise distances: (num_pairs, num_envs)
    pair_dists: list[torch.Tensor] = []
    for i in range(len(objects) - 1):
        a: RigidObject = env.scene[_resolve_name(objects[i])]
        b: RigidObject = env.scene[_resolve_name(objects[i + 1])]
        pos_a = a.data.root_pos_w.torch - env.scene.env_origins
        pos_b = b.data.root_pos_w.torch - env.scene.env_origins
        pair_dists.append(torch.norm(pos_a - pos_b, dim=-1))

    stacked = torch.stack(pair_dists, dim=0)  # (num_pairs, num_envs)
    spread = stacked.max(dim=0).values - stacked.min(dim=0).values
    is_equidistant = spread < distance_tolerance

    # All objects must be nearly still
    all_still = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for obj_ref in objects:
        vel = torch.norm(env.scene[_resolve_name(obj_ref)].data.root_lin_vel_w.torch, dim=-1)
        all_still = all_still & (vel < velocity_threshold)

    raw = is_equidistant & all_still
    if consecutive_steps <= 1:
        return raw
    counter_attr = f"_mana_equidistant_counter_{_stable_term_counter_id(*[_resolve_name(o) for o in objects], str(distance_tolerance))}"
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def object_inserted(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    insertion_axis: list[float] | None = None,
    object_axis: list[float] | None = None,
    axial_threshold: float = 0.03,
    lateral_threshold: float = 0.02,
    orientation_dot_min: float = 0.9,
    velocity_threshold: float = 0.5,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL insert] Object inserted into target along an axis (axial depth + lateral offset + orientation aligned).
    Assumes: object + target pose (root_pos_w) and orientation (root_quat_w, xyzw) + object velocity; optional insertion_axis (default target local Z) and object_axis (default object local Z).

    The relative position of *object* w.r.t. *target* (in the target's
    local frame) is decomposed into:

    1. **Axial distance** — absolute projection of the link-to-link offset
       onto *insertion_axis* (default: target's local Z ``[0, 0, 1]``).
       Must be **below** *axial_threshold* (the closer to zero, the deeper
       the insertion).
    2. **Lateral offset** — perpendicular deviation from the axis.  Must be
       below *lateral_threshold*.
    3. **Orientation alignment** — absolute dot product of the object's
       *object_axis* (default: object local Z ``[0, 0, 1]``) with the target's
       local Z.  Must exceed *orientation_dot_min*.  ``abs`` makes the check
       direction-agnostic (the object may be inserted either end first). Set
       *object_axis* to the object's model long axis when that is not local Z
       (e.g. pens authored along model +Y use ``[0, 1, 0]``).

    Suitable for key-in-lock, cap-on-pen, plug-in-socket, pen-in-holder, etc.
    """
    from isaaclab.utils import math as math_utils

    if insertion_axis is None:
        insertion_axis = [0.0, 0.0, 1.0]
    axis = torch.tensor(insertion_axis, dtype=torch.float32, device=env.device)
    axis = axis / torch.norm(axis)  # normalise
    if object_axis is None:
        object_axis = [0.0, 0.0, 1.0]
    obj_axis = torch.tensor(object_axis, dtype=torch.float32, device=env.device)
    obj_axis = obj_axis / torch.norm(obj_axis)  # normalise

    # Pool-aware: resolve active cache slots (pen_1__slot_N / pen_holder__slot_N)
    # the same way object_dropped / object_position_in_zone do. Logical slot0 is
    # often parked at z≈-10 while the visible asset lives on another slot.
    object_name = _resolve_name(object_cfg)
    target_name = _resolve_name(target_cfg)
    obj_pos, obj_vel = _object_pos_vel_env(env, object_name)
    tgt_pos, _ = _object_pos_vel_env(env, target_name)
    obj_quat = _object_quat_xyzw(env, object_name)
    tgt_quat = _object_quat_xyzw(env, target_name)
    if obj_pos is None or tgt_pos is None or obj_quat is None or tgt_quat is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    tgt_rotation = math_utils.matrix_from_quat(tgt_quat)

    # relative position in target's local frame
    diff_w = obj_pos - tgt_pos  # (N, 3) world frame
    local_diff = torch.bmm(tgt_rotation.transpose(1, 2), diff_w.unsqueeze(-1)).squeeze(-1)

    # 1) axial distance — link-to-link distance along insertion axis
    #    Small value = deeply inserted; must be below threshold.
    axial_dist = (local_diff * axis.unsqueeze(0)).sum(dim=-1).abs()  # (N,)
    axial_ok = axial_dist < axial_threshold

    # 2) lateral offset (perpendicular to axis)
    axial_vec = (local_diff * axis.unsqueeze(0)).sum(dim=-1, keepdim=True) * axis.unsqueeze(0)
    lateral = torch.norm(local_diff - axial_vec, dim=-1)
    lateral_ok = lateral < lateral_threshold

    # 3) orientation alignment — dot(obj_axis, tgt_local_z) in world frame.
    #    obj_axis defaults to the object's local Z but can be any model axis
    #    (e.g. a pen's model +Y long axis), rotated into world by the object quat.
    obj_rotation = math_utils.matrix_from_quat(obj_quat)
    obj_axis_w = torch.matmul(obj_rotation, obj_axis)
    tgt_z_w = tgt_rotation[:, :, 2]
    orient_dot = (obj_axis_w * tgt_z_w).sum(dim=-1).abs()
    orient_ok = orient_dot >= orientation_dot_min

    # 4) velocity
    vel_ok = (
        obj_vel < velocity_threshold
        if obj_vel is not None
        else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    )

    raw = axial_ok & lateral_ok & orient_ok & vel_ok
    if consecutive_steps <= 1:
        return raw
    counter_attr = (
        f"_mana_inserted_counter_{_stable_term_counter_id(_resolve_name(object_cfg), _resolve_name(target_cfg))}"
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def contact_once(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg | str,
    force_threshold: float = 1.0,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL contact] Latches True once a single contact sensor fires above ``force_threshold`` (per episode).
    Assumes: one contact sensor (force_matrix_w); episode_length_buf/reset_buf for the latch; dual-mode (eval vs termination).
    """
    sensor_name = _resolve_name(collapse_single_item(contact_sensor_cfg))

    contact_flags = _current_contact_flags(
        env,
        contact_sensor_cfgs=[contact_sensor_cfg],
        force_threshold=force_threshold,
    )
    if evaluation_runtime is not None or score_runtime is not None:
        return contact_flags[:, 0]

    history_attr = f"_contact_once_{sensor_name}"

    if not hasattr(env, history_attr):
        setattr(env, history_attr, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    _register_tensor_episode_state(env, history_attr, fill_value=False)

    history: torch.Tensor = getattr(env, history_attr)

    reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if hasattr(env, "episode_length_buf"):
        reset_mask = reset_mask | (env.episode_length_buf == 0)
    if hasattr(env, "reset_buf"):
        reset_mask = reset_mask | (env.reset_buf > 0)
    if torch.any(reset_mask):
        history[reset_mask] = False

    history[:] = history | contact_flags[:, 0]

    return history


def contact_count_at_least(
    env: ManagerBasedRLEnv,
    *,
    contact_sensor_cfgs: list[Any],
    min_count: int,
    force_threshold: float = 1.0,
    consecutive_steps: int = 1,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL contact] At least ``min_count`` of the given contact sensors are active (latched unless consecutive_steps>1).
    Assumes: multiple contact sensors (force_matrix_w); latches flags at episode reset.
    """
    del evaluation_runtime, score_runtime
    if consecutive_steps > 1:
        contact_flags = _current_contact_flags(
            env,
            contact_sensor_cfgs=contact_sensor_cfgs,
            force_threshold=force_threshold,
        )
        raw = contact_flags.sum(dim=1) >= int(min_count)
        sensor_id = _stable_term_counter_id(
            *(str(_resolve_name(collapse_single_item(cfg))) for cfg in contact_sensor_cfgs)
        )
        counter_attr = f"_mana_contact_count_at_least_{sensor_id}_{int(min_count)}"
        return _apply_consecutive_filter(env, raw, int(consecutive_steps), counter_attr=counter_attr)

    cfg_id = _stable_term_counter_id(
        *(str(_resolve_name(collapse_single_item(cfg))) for cfg in contact_sensor_cfgs),
        str(force_threshold),
    )
    latched_flags = _latched_contact_flags(
        env,
        contact_sensor_cfgs=contact_sensor_cfgs,
        force_threshold=force_threshold,
        state_attr=f"_latched_contact_flags_{cfg_id}",
    )
    return latched_flags.sum(dim=1) >= int(min_count)


def contact_sequence_reached(
    env: ManagerBasedRLEnv,
    *,
    contact_sensor_cfgs: list[Any],
    sequence_length: int,
    force_threshold: float = 1.0,
    sequence_order: list[str] | None = None,
    button_order: list[str] | None = None,
    instruction: str | None = None,
    order_from_instruction: bool = False,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL contact] Ordered contact sequence reached up to ``sequence_length`` (live progress).
    Assumes: multiple contact sensors; order via sequence_order/button_order or parsed from instruction (env._mana_benchmark_instruction).
    """
    if instruction is None:
        instruction = getattr(env, "_mana_benchmark_instruction", None)
    contact_sensor_cfgs = _ordered_contact_sensor_cfgs(
        contact_sensor_cfgs,
        sequence_order=sequence_order,
        button_order=button_order,
        instruction=instruction,
        order_from_instruction=order_from_instruction,
        evaluation_runtime=evaluation_runtime,
        score_runtime=score_runtime,
    )
    cfg_id = _stable_term_counter_id(
        *(str(_resolve_name(collapse_single_item(cfg))) for cfg in contact_sensor_cfgs),
        str(force_threshold),
    )
    ordered_progress = _ordered_contact_progress(
        env,
        contact_sensor_cfgs=contact_sensor_cfgs,
        force_threshold=force_threshold,
        state_attr=f"_ordered_contact_progress_{cfg_id}",
    )
    return ordered_progress >= int(sequence_length)


def _first_contact_steps(
    env: ManagerBasedRLEnv,
    *,
    contact_sensor_cfgs: list[Any],
    force_threshold: float,
    state_attr: str,
) -> torch.Tensor:
    """Track the episode step at which each sensor first becomes active.

    Returns a (num_envs, num_contacts) int64 tensor where each entry is the
    step number of the first contact (0 means never contacted yet — step
    counting starts at 1).
    """
    num_envs = env.num_envs
    num_contacts = len(contact_sensor_cfgs)

    if (not hasattr(env, state_attr)) or (getattr(env, state_attr).shape != (num_envs, num_contacts)):
        setattr(
            env,
            state_attr,
            torch.zeros((num_envs, num_contacts), dtype=torch.int64, device=env.device),
        )
    _register_tensor_episode_state(env, state_attr, fill_value=0)

    first_steps: torch.Tensor = getattr(env, state_attr)
    reset_mask = _episode_reset_mask(env)
    if torch.any(reset_mask):
        first_steps[reset_mask] = 0

    contact_flags = _current_contact_flags(
        env,
        contact_sensor_cfgs=contact_sensor_cfgs,
        force_threshold=force_threshold,
    )

    current_step = env.episode_length_buf.unsqueeze(-1).expand_as(first_steps)
    newly_contacted = contact_flags & (first_steps == 0)
    first_steps[newly_contacted] = current_step[newly_contacted]

    return first_steps


def contact_sequence_by_first_touch(
    env: ManagerBasedRLEnv,
    *,
    contact_sensor_cfgs: list[Any],
    sequence_length: int,
    force_threshold: float = 1.0,
    sequence_order: list[str] | None = None,
    button_order: list[str] | None = None,
    instruction: str | None = None,
    order_from_instruction: bool = False,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL contact] Ordered contact sequence by first-touch timestamp (ascending first-touch steps), not live contact.
    Assumes: multiple contact sensors; same ordering modes as contact_sequence_reached; tracks first-touch step per sensor.

    Records the episode step at which each sensor first exceeds force_threshold.
    Returns True when the first ``sequence_length`` sensors (in the declared
    order) all have been contacted AND their first-touch steps are strictly
    ascending — i.e. the physical placement order matches the expected sequence.

    Unlike ``contact_sequence_reached`` which can be fooled by objects that
    remain in contact (always reading True once placed), this term only cares
    about *when* each sensor was first activated.
    """
    if instruction is None:
        instruction = getattr(env, "_mana_benchmark_instruction", None)
    ordered_cfgs = _ordered_contact_sensor_cfgs(
        contact_sensor_cfgs,
        sequence_order=sequence_order,
        button_order=button_order,
        instruction=instruction,
        order_from_instruction=order_from_instruction,
        evaluation_runtime=evaluation_runtime,
        score_runtime=score_runtime,
    )

    cfg_id = _stable_term_counter_id(
        *(str(_resolve_name(collapse_single_item(c))) for c in ordered_cfgs),
        str(force_threshold),
    )
    state_attr = f"_first_contact_steps_{cfg_id}"

    first_steps = _first_contact_steps(
        env,
        contact_sensor_cfgs=ordered_cfgs,
        force_threshold=force_threshold,
        state_attr=state_attr,
    )

    seq_len = int(sequence_length)
    relevant = first_steps[:, :seq_len]

    all_touched = (relevant > 0).all(dim=-1)

    if seq_len <= 1:
        return all_touched

    strictly_ascending = (relevant[:, 1:] > relevant[:, :-1]).all(dim=-1)
    return all_touched & strictly_ascending


def articulation_joints_near_default(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | str,
    joint_names: list[str] | None = None,
    exclude_joint_name_substrings: list[str] | None = None,
    atol: float = 0.1,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL joint] Current joint positions within ``atol`` rad of the articulation default (USD/reset) pose.
    Assumes: articulation joints (joint_pos/default_joint_pos); optional joint_names subset (whole-arm homing if omitted).

    ``default_joint_pos`` matches Isaac Lab / Arena reset targets (USD initial pose).
    When *joint_names* is omitted, every DoF on the asset is checked (typical for full-arm homing).
    ``exclude_joint_name_substrings`` drops joints whose names contain any listed
    token (case-insensitive), e.g. ``["finger", "gripper"]`` so Robotiq mimic
    chains / parked-closed grippers do not block arm-home success.
    """
    name = _resolve_name(asset_cfg)
    articulation: Articulation = env.scene[name]
    q = articulation.data.joint_pos.torch
    q0 = articulation.data.default_joint_pos.torch
    if joint_names:
        joint_ids, _ = articulation.find_joints(joint_names, preserve_order=True)
        joint_ids = [int(i) for i in joint_ids]
    else:
        joint_ids = list(range(int(q.shape[-1])))
    excludes = [str(s).lower() for s in (exclude_joint_name_substrings or []) if str(s).strip()]
    if excludes:
        all_names = [str(n) for n in getattr(articulation, "joint_names", []) or []]
        if all_names:
            joint_ids = [
                int(i)
                for i in joint_ids
                if 0 <= int(i) < len(all_names) and not any(token in all_names[int(i)].lower() for token in excludes)
            ]
    if not joint_ids:
        raw = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
    else:
        diff = torch.abs(q[:, joint_ids] - q0[:, joint_ids])
        raw = (diff < atol).all(dim=-1)
    if consecutive_steps <= 1:
        return raw
    counter_key = (
        f"_mana_artic_near_def_"
        f"{_stable_term_counter_id(name, str(atol), str(joint_names), str(exclude_joint_name_substrings))}"
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_key)


def articulation_joints_near_default_after_time(
    env: ManagerBasedRLEnv,
    *,
    asset_cfg: SceneEntityCfg | str,
    joint_names: list[str] | None = None,
    exclude_joint_name_substrings: list[str] | None = None,
    atol: float = 0.1,
    after_seconds: float,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL joint] Joints within ``atol`` rad of default pose AND elapsed time > ``after_seconds`` (homed after a window).
    Assumes: articulation joints (joint_pos/default_joint_pos); a runtime with step_dt_s + frame counter (or env.episode_length_buf).
    """
    runtime = evaluation_runtime if evaluation_runtime is not None else score_runtime
    # Reuse the non-timed predicate so finger/mimic excludes stay consistent.
    base = articulation_joints_near_default(
        env,
        asset_cfg=asset_cfg,
        joint_names=joint_names,
        exclude_joint_name_substrings=exclude_joint_name_substrings,
        atol=float(atol),
        consecutive_steps=1,
    )

    step_dt_s = float(getattr(runtime, "step_dt_s", 0.0) or 0.0)
    if step_dt_s <= 0.0:
        return _false_tensor(env)

    # Prefer the evaluation runtime's own frame counter which is unaffected
    # by Isaac Lab's in-step episode_length_buf reset for done envs.
    frame_buf = getattr(runtime, "episode_elapsed_frames", None)
    if frame_buf is None:
        if not hasattr(env, "episode_length_buf"):
            return _false_tensor(env)
        frame_buf = env.episode_length_buf
    elapsed_s = frame_buf.to(dtype=torch.float32) * step_dt_s
    return base & (elapsed_s > float(after_seconds))


def _normalized_joint_position_openness(
    joint_pos: torch.Tensor,
    j_min: torch.Tensor,
    j_max: torch.Tensor,
) -> torch.Tensor:
    """Map joint angle to [0, 1] openness matching Arena ``get_normalized_joint_position``."""
    span = j_max - j_min
    eps = 1.0e-6
    safe_span = torch.where(span.abs() < eps, torch.full_like(span, eps), span)
    normalized = (joint_pos - j_min) / safe_span
    if j_min.item() < 0.0:
        normalized = 1.0 - normalized
    return torch.clamp(normalized, 0.0, 1.0)


def _resolve_articulation_name_for_env(
    env: ManagerBasedRLEnv,
    logical_name: str,
    *,
    env_id: int,
) -> str | None:
    """Resolve a logical articulation id to the active scene key for *env_id*."""
    active_name = resolve_active_scene_entity_name(env, logical_name, env_id=env_id)
    if scene_has_asset(env.scene, active_name):
        return active_name
    if scene_has_asset(env.scene, logical_name):
        return logical_name
    return None


def _joint_openness_ok_for_articulation(
    articulation: Any,
    *,
    env_id: int,
    patterns: list[str],
    threshold: float,
    openness_gt: bool,
    joint_match: str,
) -> bool:
    joint_indices, _ = articulation.find_joints(patterns, preserve_order=True)
    if not joint_indices:
        return False
    flags: list[bool] = []
    limits_t = articulation.data.joint_pos_limits.torch
    lim_env = env_id if int(limits_t.shape[0]) > env_id else 0
    for joint_index in joint_indices:
        joint_pos = articulation.data.joint_pos.torch[env_id : env_id + 1, joint_index]
        limits = limits_t[lim_env, joint_index, :]
        openness = _normalized_joint_position_openness(joint_pos, limits[0], limits[1])
        flag = openness > threshold if openness_gt else openness < threshold
        flags.append(bool(flag.reshape(-1)[0].item()))
    return all(flags) if joint_match == "all" else any(flags)


def _articulation_joint_openness_ok(
    env: ManagerBasedRLEnv,
    logical_name: str,
    patterns: list[str],
    *,
    threshold: float,
    openness_gt: bool,
    joint_match: str,
) -> torch.Tensor:
    """Per-env joint openness vs threshold; follows schema-vNext active pool slots.

    Runtime asset caches park inactive drawers under ``drawer`` / ``drawer__slot_N``.
    Reading only the logical base key leaves the checker stuck on a parked
    articulation while the operator opens the active slot.
    """
    ok = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if getattr(env, "_schema_vnext_active_asset_slots", None):
        for env_id in range(int(env.num_envs)):
            active_name = _resolve_articulation_name_for_env(env, logical_name, env_id=env_id)
            if active_name is None:
                continue
            ok[env_id] = _joint_openness_ok_for_articulation(
                env.scene[active_name],
                env_id=env_id,
                patterns=patterns,
                threshold=threshold,
                openness_gt=openness_gt,
                joint_match=joint_match,
            )
        return ok

    if not scene_has_asset(env.scene, logical_name):
        return ok
    articulation = env.scene[logical_name]
    joint_indices, _ = articulation.find_joints(patterns, preserve_order=True)
    if not joint_indices:
        return ok
    per_joint: list[torch.Tensor] = []
    for joint_index in joint_indices:
        joint_pos = articulation.data.joint_pos.torch[:, joint_index]
        limits = articulation.data.joint_pos_limits.torch[0, joint_index, :]
        openness = _normalized_joint_position_openness(joint_pos, limits[0], limits[1])
        per_joint.append(openness > threshold if openness_gt else openness < threshold)
    stacked = torch.stack(per_joint, dim=0)
    return stacked.all(dim=0) if joint_match == "all" else stacked.any(dim=0)


def joint_above_threshold(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | str,
    joint_names: list[str],
    threshold: float = 0.5,
    openness_gt: bool = True,
    joint_match: str = "all",
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL joint] Per-joint normalized openness vs ``threshold`` (all/any), Arena ``Openable``-compatible.
    Assumes: articulation with joint_pos + joint_pos_limits; joint_names are regex (fullmatch); consecutive-steps support.

    Arena ``Openable.is_open`` uses instantaneous ``openness > threshold`` with no consecutive
    framing --- that matches ``openness_gt=True`` and ``consecutive_steps=1`` here. Use
    ``openness_gt=False`` for ``openness < threshold`` (e.g. mouse buttons whose ``openness``
    drops when pressed if the idle pose reads as ``> 0.5``).

    *joint_match*: ``"all"`` — every listed joint must satisfy the comparison (AND). ``"any"`` — at
    least one (OR). Default ``"all"``.

    Optional ``consecutive_steps > 1`` only damps noise.

    Each *joint_names* entry is a regex for ``re.fullmatch`` against ``articulation.joint_names``.
    Resolves schema-vNext active pool slots (e.g. ``drawer__slot_1``) per env.
    """
    if not joint_names:
        raise ValueError("joint_above_threshold requires a non-empty joint_names list")

    jm = joint_match.strip().lower()
    if jm not in {"all", "any"}:
        raise ValueError('joint_match must be "all" or "any"')

    asset_name = _resolve_name(object_cfg)
    patterns = [p.strip() for p in joint_names]
    ok = _articulation_joint_openness_ok(
        env,
        asset_name,
        patterns,
        threshold=float(threshold),
        openness_gt=bool(openness_gt),
        joint_match=jm,
    )

    if consecutive_steps <= 1:
        return ok
    counter_key = (
        f"_mana_joint_above_{_stable_term_counter_id(asset_name, *patterns, str(openness_gt), str(threshold), jm)}"
    )
    return _apply_consecutive_filter(env, ok, consecutive_steps, counter_attr=counter_key)


def indexed_joint_above_threshold(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | str,
    joint_names: list[str],
    index_log_key: str,
    threshold: float = 0.5,
    openness_gt: bool = True,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL joint] Openness check on the single joint selected by a reset-committed index.
    Assumes: the articulation exposes all ordered joints and ``index_log_key`` stores a matching integer per env.

    ``joint_names`` is an ordered list aligned with the index (e.g. top/middle/bottom
    drawer joints). Reads ``index_log_key`` (int ``[num_envs, 1]``) and evaluates
    only ``joint_names[index]`` against ``threshold`` / ``openness_gt``.
    """

    if not joint_names:
        raise ValueError("indexed_joint_above_threshold requires a non-empty joint_names list")

    idx_log = read_log(env, str(index_log_key))
    if idx_log is None:
        return _false_tensor(env)
    idx_log = idx_log.to(device=env.device, dtype=torch.int64).reshape(env.num_envs, -1)[:, 0]
    idx_log = idx_log.clamp(0, len(joint_names) - 1)

    asset_name = _resolve_name(object_cfg)
    patterns = [p.strip() for p in joint_names]
    ok = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for env_id in range(int(env.num_envs)):
        active_name = _resolve_articulation_name_for_env(env, asset_name, env_id=env_id)
        if active_name is None:
            continue
        articulation = env.scene[active_name]
        joint_indices, _ = articulation.find_joints(patterns, preserve_order=True)
        if len(joint_indices) != len(patterns):
            raise ValueError(
                f"indexed_joint_above_threshold: expected {len(patterns)} joints matching "
                f"{patterns!r} on {active_name!r}, found {len(joint_indices)}"
            )
        joint_i = int(joint_indices[int(idx_log[env_id].item())])
        limits_t = articulation.data.joint_pos_limits.torch
        lim_env = env_id if int(limits_t.shape[0]) > env_id else 0
        joint_pos = articulation.data.joint_pos.torch[env_id : env_id + 1, joint_i]
        limits = limits_t[lim_env, joint_i, :]
        openness = _normalized_joint_position_openness(joint_pos, limits[0], limits[1])
        flag = openness > threshold if openness_gt else openness < threshold
        ok[env_id] = bool(flag.reshape(-1)[0].item())

    if consecutive_steps <= 1:
        return ok
    counter_key = (
        f"_mana_indexed_joint_above_"
        f"{_stable_term_counter_id(asset_name, str(index_log_key), *patterns, str(openness_gt), str(threshold))}"
    )
    return _apply_consecutive_filter(env, ok, consecutive_steps, counter_attr=counter_key)


def joint_reaches_sampled_openness(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | str,
    joint_names: list[str],
    tolerance: float = 0.02,
    consecutive_steps: int = 2,
    store_attr: str = "EPISODE_TURNABLE_TWIST_GOAL_ATTR",
) -> torch.Tensor:
    """[SKILL joint] Success when the knob reaches the task-pinned target openness.

    Reads per-env goals written by :func:`manaenv.terms.events.sample_turnable_twist_goal`
    (pins live in :mod:`manaenv.pipeline.skills.twist_runtime.twist_goal`).
    Envs without a pin stay False (strict — success is the sampled target, not any travel).
    """
    from manaenv.pipeline.skills.twist_runtime.twist_goal import (
        EPISODE_TURNABLE_TWIST_GOAL_ATTR,
        read_pinned_twist_goal,
    )

    if not joint_names:
        raise ValueError("joint_reaches_sampled_openness requires a non-empty joint_names list")

    resolved_store = store_attr
    if resolved_store == "EPISODE_TURNABLE_TWIST_GOAL_ATTR":
        resolved_store = EPISODE_TURNABLE_TWIST_GOAL_ATTR

    asset_name = _resolve_name(object_cfg)
    articulation = env.scene[asset_name]
    patterns = [p.strip() for p in joint_names]
    joint_indices, _ = articulation.find_joints(patterns, preserve_order=True)
    if not joint_indices:
        return _false_tensor(env)

    counter_key = (
        f"_mana_joint_sampled_{_stable_term_counter_id(asset_name, *patterns, str(tolerance), resolved_store)}"
    )

    ok = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for env_idx in range(env.num_envs):
        goal = read_pinned_twist_goal(env, env_idx, store_attr=resolved_store)
        if goal is None:
            continue
        joint_index = None
        for joint_i in joint_indices:
            names = list(getattr(articulation, "joint_names", None) or [])
            if joint_i < len(names) and names[joint_i] == goal.joint_name:
                joint_index = joint_i
                break
        if joint_index is None:
            joint_index = int(joint_indices[0])
        joint_pos = articulation.data.joint_pos.torch[env_idx, joint_index]
        from manaenv.assets.objects.object_library import KnobA

        target_joint = KnobA.display_value_to_joint_angle(goal.target_temperature)
        ok[env_idx] = (joint_pos - target_joint).abs() <= float(tolerance)

    if consecutive_steps <= 1:
        final = ok
    else:
        final = _apply_consecutive_filter(env, ok, consecutive_steps, counter_attr=counter_key)

    from manaenv.pipeline.skills.twist_runtime.twist_goal import set_episode_result

    for env_idx in range(env.num_envs):
        if bool(final[env_idx].item()):
            set_episode_result(env, env_idx, success=True)
    return final


def twist_episode_initial_temperature(env: ManagerBasedRLEnv) -> list[int | None]:
    """Per-env initial panel temperature for overlay / subtask log (``value_func``).

    Returns a Python list (not a tensor) so the evaluation overlay can snapshot the
    value: tensor ``value_func`` results go to the live log store only and never
    populate the producer ledger that ``overlay_snapshot`` reads.
    """
    from manaenv.pipeline.skills.twist_runtime.twist_goal import read_episode_record

    out: list[int | None] = []
    for env_idx in range(int(env.num_envs)):
        record = read_episode_record(env, env_idx)
        out.append(None if record is None else int(record.initial_temperature))
    return out


def twist_episode_target_temperature(env: ManagerBasedRLEnv) -> list[int | None]:
    """Per-env target panel temperature for overlay / subtask log (``value_func``).

    See :func:`twist_episode_initial_temperature` for why this returns a list.
    """
    from manaenv.pipeline.skills.twist_runtime.twist_goal import read_episode_record

    out: list[int | None] = []
    for env_idx in range(int(env.num_envs)):
        record = read_episode_record(env, env_idx)
        out.append(None if record is None else int(record.target_temperature))
    return out


def twist_episode_result_label(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env twist outcome code for subtask log: 1=passed, 0=failed, -1=pending."""
    from manaenv.pipeline.skills.twist_runtime.twist_goal import read_episode_record

    out = torch.full((env.num_envs,), -1.0, dtype=torch.float32, device=env.device)
    for env_idx in range(env.num_envs):
        record = read_episode_record(env, env_idx)
        if record is None or not str(record.result or "").strip():
            continue
        out[env_idx] = 1.0 if str(record.result).strip().lower() == "passed" else 0.0
    return out


def twist_episode_progress_message(
    env: ManagerBasedRLEnv,
    template: str = "twist the knob from {initial} to the sampled {target} temperature",
) -> list[str]:
    """Per-env progress prose with ``{initial}`` / ``{target}`` filled from the pinned goal."""
    from manaenv.pipeline.skills.twist_runtime.twist_goal import read_episode_record, render_twist_instruction_template

    out: list[str] = []
    for env_idx in range(int(env.num_envs)):
        record = read_episode_record(env, env_idx)
        if record is None:
            out.append(str(template))
            continue
        out.append(
            render_twist_instruction_template(
                str(template),
                initial=int(record.initial_temperature),
                target=int(record.target_temperature),
            )
        )
    return out


def joint_above_threshold_once(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | str,
    joint_names: list[str],
    threshold: float = 0.5,
    openness_gt: bool = True,
    joint_match: str = "all",
) -> torch.Tensor:
    """[SKILL joint] Latching ``joint_above_threshold`` — True once satisfied at any point in the episode.
    Assumes: same as joint_above_threshold; latch resets on episode reset.

    Returns ``True`` for an environment once the joint condition has been satisfied
    *at any point* during the current episode.  The latch resets when the episode resets.
    """
    if not joint_names:
        raise ValueError("joint_above_threshold_once requires a non-empty joint_names list")

    jm = joint_match.strip().lower()
    if jm not in {"all", "any"}:
        raise ValueError('joint_match must be "all" or "any"')

    asset_name = _resolve_name(object_cfg)
    patterns = [p.strip() for p in joint_names]
    history_attr = (
        f"_mana_joint_once_{_stable_term_counter_id(asset_name, *patterns, str(openness_gt), str(threshold), jm)}"
    )

    if not hasattr(env, history_attr):
        setattr(env, history_attr, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    _register_tensor_episode_state(env, history_attr, fill_value=False)

    history: torch.Tensor = getattr(env, history_attr)

    reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if hasattr(env, "episode_length_buf"):
        reset_mask = reset_mask | (env.episode_length_buf == 0)
    if hasattr(env, "reset_buf"):
        reset_mask = reset_mask | (env.reset_buf > 0)
    if torch.any(reset_mask):
        history[reset_mask] = False

    ok = _articulation_joint_openness_ok(
        env,
        asset_name,
        patterns,
        threshold=float(threshold),
        openness_gt=bool(openness_gt),
        joint_match=jm,
    )

    history[:] = history | ok
    return history


def pump_pressed_n_times(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | str,
    joint_names: list[str],
    press_threshold: float = 0.6,
    release_threshold: float = 0.25,
    openness_gt: bool = True,
    count_log_key: str = "target_press_count",
) -> torch.Tensor:
    """[SKILL joint] Count pump press-release cycles until the sampled target is reached.
    Assumes: one articulated pump joint has calibrated limits and ``count_log_key`` stores the target per env.

    A cycle is: joint openness crosses the press threshold, then returns past the
    release threshold (hysteresis). For ``pump_bottle_a``'s Z prismatic joint
    (limits ``[-0.005, 0]``), idle openness ≈ 0 and fully pressed ≈ 1, so keep
    ``openness_gt=True`` with ``press_threshold > release_threshold``.

    Defaults use ``press_threshold=0.6`` (≈3 mm of 5 mm travel): contact presses
    often stop short of 0.8 even when they look fully depressed.

    Target count is read from ``count_log_key`` (written by
    :func:`manaenv.terms.events.sample_press_button_n_times`).
    """
    if not joint_names:
        raise ValueError("pump_pressed_n_times requires a non-empty joint_names list")
    if float(press_threshold) == float(release_threshold):
        raise ValueError("pump_pressed_n_times requires distinct press_threshold and release_threshold")

    asset_name = _resolve_name(object_cfg)
    patterns = [p.strip() for p in joint_names]
    term_id = _stable_term_counter_id(
        asset_name,
        *patterns,
        str(openness_gt),
        str(press_threshold),
        str(release_threshold),
        str(count_log_key),
    )
    armed_attr = f"_mana_pump_press_armed_{term_id}"
    count_attr = f"_mana_pump_press_count_{term_id}"

    if not hasattr(env, armed_attr) or getattr(env, armed_attr).shape[0] != env.num_envs:
        setattr(env, armed_attr, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    if not hasattr(env, count_attr) or getattr(env, count_attr).shape[0] != env.num_envs:
        setattr(env, count_attr, torch.zeros(env.num_envs, dtype=torch.int64, device=env.device))
    _register_tensor_episode_state(env, armed_attr, fill_value=False)
    _register_tensor_episode_state(env, count_attr, fill_value=0)

    armed: torch.Tensor = getattr(env, armed_attr)
    count: torch.Tensor = getattr(env, count_attr)

    reset_mask = _episode_reset_mask(env)
    if torch.any(reset_mask):
        armed[reset_mask] = False
        count[reset_mask] = 0

    articulation = env.scene[asset_name]
    joint_indices, _ = articulation.find_joints(patterns, preserve_order=True)
    if not joint_indices:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Single pump joint; take the first matched index.
    joint_index = int(joint_indices[0])
    joint_pos = articulation.data.joint_pos.torch[:, joint_index]
    limits = articulation.data.joint_pos_limits.torch[0, joint_index, :]
    openness = _normalized_joint_position_openness(joint_pos, limits[0], limits[1])

    if openness_gt:
        pressed_now = openness > float(press_threshold)
        released_now = openness < float(release_threshold)
    else:
        pressed_now = openness < float(press_threshold)
        released_now = openness > float(release_threshold)

    newly_armed = (~armed) & pressed_now
    completed = armed & released_now
    if torch.any(newly_armed):
        armed[newly_armed] = True
    if torch.any(completed):
        count[completed] = count[completed] + 1
        armed[completed] = False

    target = read_log(env, str(count_log_key))
    if target is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    target = target.to(device=env.device, dtype=torch.int64).reshape(-1)
    if target.shape[0] < env.num_envs:
        padded = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        padded[: target.shape[0]] = target
        target = padded
    return count >= target[: env.num_envs]


def pump_target_press_count_label(
    env: ManagerBasedRLEnv,
    log_key: str = "target_press_count",
) -> list[str | None]:
    """Map the committed press-count target to a display string for overlay/log."""

    num_envs = int(env.num_envs)
    idx = read_log(env, str(log_key))
    if idx is None:
        return [None] * num_envs
    idx = idx.to(dtype=torch.int64).reshape(-1)
    out: list[str | None] = []
    for env_id in range(num_envs):
        value = int(idx[env_id].item()) if env_id < idx.shape[0] else -1
        out.append(str(value) if value > 0 else None)
    return out


def pump_bottle_slot_label(
    env: ManagerBasedRLEnv,
    state_attr: str = "_mana_press_button_n_times_state",
) -> list[str | None]:
    """Return the reset-sampled bottle slot label (left/center/right mid) for overlay."""
    state = getattr(env, state_attr, None)
    labels = state.get("slot_text") if isinstance(state, dict) else None
    if not isinstance(labels, dict):
        return [None] * int(env.num_envs)
    return [labels.get(env_id) for env_id in range(int(env.num_envs))]


def knob_at_target_level(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg | str,
    joint_names: list[str],
    num_levels: int,
    target_log_key: str = "target_temperature_level",
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL joint] Turnable knob is at the reset-sampled target temperature level.

    The joint's full normalized travel is split into ``num_levels`` equal bins
    (matching :func:`manaenv.terms.events.sample_knob_temperature`). The current
    bin index is compared against the per-env target committed to the live log
    store under ``target_log_key``. Absent target (store not yet written) reads as
    all-False. ``consecutive_steps`` only damps transient overshoot.
    """

    if not joint_names:
        raise ValueError("knob_at_target_level requires a non-empty joint_names list")

    name = _resolve_name(object_cfg)
    articulation: Articulation = env.scene[name]
    patterns = [p.strip() for p in joint_names]
    joint_indices, _ = articulation.find_joints(patterns, preserve_order=True)
    if not joint_indices:
        return _false_tensor(env)
    joint_index = joint_indices[0]

    joint_pos = articulation.data.joint_pos.torch[:, joint_index]
    limits = articulation.data.joint_pos_limits.torch[0, joint_index, :]
    openness = _normalized_joint_position_openness(joint_pos, limits[0], limits[1])
    levels = int(num_levels)
    # Same floor binning the asset's display OmniGraph uses (index = floor(angle /
    # step)), so "success" agrees with the digit the operator reads off the knob.
    current_level = torch.clamp(torch.floor(openness * levels), 0, levels - 1).to(torch.int64)

    target = read_log(env, str(target_log_key))
    if target is None:
        return _false_tensor(env)
    target = target.to(device=env.device, dtype=torch.int64).reshape(-1)
    raw = current_level == target

    if consecutive_steps <= 1:
        return raw
    counter_key = f"_mana_knob_level_{_stable_term_counter_id(name, *patterns, str(levels), str(target_log_key))}"
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_key)


def knob_temperature_label(
    env: ManagerBasedRLEnv,
    log_key: str = "target_temperature_level",
    level_labels: list[str] | None = None,
) -> list[str | None]:
    """Map a committed temperature level index to its display label for logging/overlay."""

    labels = [str(item) for item in (level_labels or [])]
    num_envs = int(env.num_envs)
    idx = read_log(env, str(log_key))
    if idx is None or not labels:
        return [None] * num_envs
    idx = idx.to(dtype=torch.int64).reshape(-1)
    out: list[str | None] = []
    for env_id in range(num_envs):
        level = int(idx[env_id].item()) if env_id < idx.shape[0] else -1
        out.append(labels[level] if 0 <= level < len(labels) else None)
    return out


def balance_active_plan(env: ManagerBasedRLEnv, state_attr: str = "_mana_balance_state") -> list[str | None]:
    """Return the reset-sampled per-env balance add-plan text for logging/overlay."""
    state = getattr(env, state_attr, None)
    plans = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plans, dict):
        return [None] * int(env.num_envs)
    return [plans.get(env_id) for env_id in range(int(env.num_envs))]


def _weights_in_zone_count(
    env: ManagerBasedRLEnv,
    names: list[str],
    zone: dict[str, Any],
    velocity_threshold: float | None,
) -> torch.Tensor:
    """Per-env count of the given weight objects whose env-local position is inside *zone* (and slow)."""
    device = env.device
    total = torch.zeros(env.num_envs, dtype=torch.int64, device=device)
    x_range = zone.get("x_range")
    y_range = zone.get("y_range")
    z_range = zone.get("z_range")
    for name in names:
        if not scene_has_asset(env.scene, name):
            continue
        pos, _ = _get_root_pose_env(env, name)
        inside = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        if x_range is not None:
            inside = inside & (pos[:, 0] >= float(x_range[0])) & (pos[:, 0] <= float(x_range[1]))
        if y_range is not None:
            inside = inside & (pos[:, 1] >= float(y_range[0])) & (pos[:, 1] <= float(y_range[1]))
        if z_range is not None:
            inside = inside & (pos[:, 2] >= float(z_range[0])) & (pos[:, 2] <= float(z_range[1]))
        if velocity_threshold is not None:
            inside = inside & _velocity_below_threshold(env.scene[name], velocity_threshold, env.num_envs, device)
        total = total + inside.to(torch.int64)
    return total


def weight_in_target_pan(
    env: ManagerBasedRLEnv,
    *,
    step_index: int,
    denoms: list[int],
    left_zone: dict[str, Any],
    right_zone: dict[str, Any],
    weight_prefix: str = "weight_",
    count_per_denom: int = 2,
    denom_log_key: str = "balance_add_denom",
    pan_log_key: str = "balance_add_pan",
    velocity_threshold: float | None = 0.2,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """Cumulative check that add-steps ``0..step_index`` are placed on their target pans.

    Reads the reset-committed per-env add plan (``denom_log_key`` / ``pan_log_key``,
    int tensors ``[num_envs, add_count]``; denom code = denom index + 1, pan code
    0=left/1=right). For every (denomination, pan) the required count over the
    cumulative steps is compared against the number of that denomination's weights
    currently resting in the pan's zone. True when every requirement is met.
    """

    denom_log = read_log(env, str(denom_log_key))
    pan_log = read_log(env, str(pan_log_key))
    if denom_log is None or pan_log is None:
        return _false_tensor(env)

    denom_log = denom_log.to(device=env.device, dtype=torch.int64)
    pan_log = pan_log.to(device=env.device, dtype=torch.int64)
    if denom_log.ndim == 1:
        denom_log = denom_log.unsqueeze(-1)
        pan_log = pan_log.unsqueeze(-1)
    upper = min(int(step_index) + 1, denom_log.shape[1])
    denom_slice = denom_log[:, :upper]
    pan_slice = pan_log[:, :upper]

    denoms = [int(d) for d in denoms]
    zones = {0: left_zone, 1: right_zone}
    result = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for di, denom in enumerate(denoms):
        names = [f"{weight_prefix}{denom}g_{k}" for k in range(1, int(count_per_denom) + 1)]
        for pan in (0, 1):
            required = ((denom_slice == (di + 1)) & (pan_slice == pan)).sum(dim=1).to(torch.int64)
            if not bool((required > 0).any()):
                continue
            observed = _weights_in_zone_count(env, names, zones[pan], velocity_threshold)
            result = result & (observed >= required)

    if consecutive_steps <= 1:
        return result
    counter_attr = "_mana_weight_in_pan_" + _stable_term_counter_id(
        str(denom_log_key), str(pan_log_key), str(step_index), str(denoms)
    )
    return _apply_consecutive_filter(env, result, consecutive_steps, counter_attr=counter_attr)


def balance_beam_level(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg | str,
    joint_names: list[str],
    level_atol: float = 0.05,
    vel_threshold: float | None = 0.1,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """Balance beam revolute joint within ``level_atol`` rad of its level (default) pose and steady.

    The beam's default joint position is the level state; success is |angle - default|
    < ``level_atol`` and (optionally) |angular velocity| < ``vel_threshold`` for
    ``consecutive_steps`` frames.
    """
    name = _resolve_name(object_cfg)
    articulation: Articulation = env.scene[name]
    patterns = [str(p).strip() for p in joint_names]
    joint_indices, _ = articulation.find_joints(patterns, preserve_order=True)
    if not joint_indices:
        return _false_tensor(env)
    joint_index = joint_indices[0]

    joint_pos = articulation.data.joint_pos.torch[:, joint_index]
    default_pos = articulation.data.default_joint_pos.torch[:, joint_index]
    raw = torch.abs(joint_pos - default_pos) < float(level_atol)
    if vel_threshold is not None and hasattr(articulation.data, "joint_vel"):
        joint_vel = articulation.data.joint_vel.torch[:, joint_index]
        raw = raw & (torch.abs(joint_vel) < float(vel_threshold))

    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_beam_level_" + _stable_term_counter_id(name, *patterns)
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def robot_body_near_object_xy(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg | str,
    object_cfg: SceneEntityCfg | str | list,
    body_name: str,
    xy_threshold: float = 0.10,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL reach] A robot body (e.g. EE) within ``xy_threshold`` m of an object on the XY plane (Z ignored).
    Assumes: robot articulation with named body (body_pos_w); object pose (root_pos_w); OR across an object list.

    ``object_cfg`` may be a list; returns True if the body is near *any* object in the list.

    Args:
        robot_cfg: Scene entity name or cfg for the robot articulation.
        object_cfg: Scene entity name, cfg, or list thereof for the target rigid object(s).
        body_name: Name of the rigid body on the robot to track (e.g. ``"rd001_left_gripper"``).
        xy_threshold: Distance threshold in metres on the XY plane.
        consecutive_steps: Number of consecutive steps the condition must hold.
    """
    if isinstance(object_cfg, list):
        result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for single_cfg in object_cfg:
            result = result | robot_body_near_object_xy(
                env, robot_cfg, single_cfg, body_name, xy_threshold, consecutive_steps
            )
        return result

    robot = env.scene[_resolve_name(robot_cfg)]
    obj_name = _resolve_name(object_cfg)

    body_indices, _ = robot.find_bodies([body_name])
    if not body_indices:
        manaprint(f"WARNING: robot_body_near_object_xy: body '{body_name}' not found on '{_resolve_name(robot_cfg)}'")
        return _false_tensor(env)
    body_idx = body_indices[0]

    # World-space positions, subtract env origins so multi-env offsets cancel
    body_pos_w = robot.data.body_pos_w.torch[:, body_idx, :2] - env.scene.env_origins[:, :2]
    obj_pos_w, _ = _get_root_pose_env(env, obj_name)
    obj_pos_w = obj_pos_w[:, :2]

    xy_dist = torch.norm(body_pos_w - obj_pos_w, dim=-1)
    raw = xy_dist < xy_threshold

    if consecutive_steps <= 1:
        return raw
    counter_key = (
        f"_mana_body_near_xy_{_stable_term_counter_id(_resolve_name(robot_cfg), _resolve_name(object_cfg), body_name)}"
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_key)


# =====================================================================
# Dynamic gripper-contact-order evaluation terms
#
# These terms identify objects by gripper-touch order (1st, 2nd, 3rd ...)
# rather than by fixed scene tag.  They share a common helper that
# records the episode step at which the gripper first contacts each
# candidate object, then ranks objects by that timestamp.
# =====================================================================


def _nth_touched_object_index(
    env: ManagerBasedRLEnv,
    gripper_contact_sensor_cfgs: list[Any],
    force_threshold: float,
    nth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(index, valid)`` of the n-th object touched by the gripper.

    ``index`` is ``(num_envs,)`` int64 — the position in
    *gripper_contact_sensor_cfgs* of the n-th object contacted (0-based).
    ``valid`` is ``(num_envs,)`` bool — True when at least *nth* distinct
    objects have been contacted in this episode.

    Objects are ranked by their first-touch episode step (ascending).
    Ties are broken by list order (lower index wins).
    """
    cfg_id = _stable_term_counter_id(
        *(str(_resolve_name(collapse_single_item(c))) for c in gripper_contact_sensor_cfgs),
        str(force_threshold),
        "_nth_order",
    )
    state_attr = f"_gripper_order_steps_{cfg_id}"
    first_steps = _first_contact_steps(
        env,
        contact_sensor_cfgs=gripper_contact_sensor_cfgs,
        force_threshold=force_threshold,
        state_attr=state_attr,
    )

    large = first_steps.max().item() + 1
    sortable = torch.where(first_steps > 0, first_steps, torch.full_like(first_steps, large))
    order = torch.argsort(sortable, dim=-1, stable=True)

    n = int(nth) - 1
    index = order[:, n] if n < order.shape[1] else torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
    valid = (
        (sortable.gather(1, order)[:, n] < large)
        if n < order.shape[1]
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    return index, valid


def gripper_nth_object_lifted(
    env: ManagerBasedRLEnv,
    *,
    gripper_contact_sensor_cfgs: list[Any],
    object_cfgs: list[Any],
    nth: int,
    force_threshold: float = 0.1,
    min_lift: float = 0.01,
    settle_steps: int = 0,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL multi-object] The *nth* gripper-touched object remains grasped and rises >= ``min_lift``.
    Assumes: parallel gripper-contact sensors + objects; live force and object env-z after ``settle_steps``.

    Object identity is determined dynamically: whichever object's
    gripper-contact sensor fires first is "object 1", the second is
    "object 2", and so on.  ``gripper_contact_sensor_cfgs`` and
    ``object_cfgs`` must be parallel lists of the same length.

    Lift is measured relative to the object's env-frame z captured after
    ``settle_steps``. The selected object's contact sensor must still exceed
    ``force_threshold`` when the lift threshold is reached.
    """
    del evaluation_runtime, score_runtime

    if len(gripper_contact_sensor_cfgs) != len(object_cfgs):
        raise ValueError("gripper_contact_sensor_cfgs and object_cfgs must have the same length")

    index, valid = _nth_touched_object_index(env, gripper_contact_sensor_cfgs, force_threshold, nth)
    gather_index = index.unsqueeze(1)

    contact_flags = _current_contact_flags(
        env,
        contact_sensor_cfgs=gripper_contact_sensor_cfgs,
        force_threshold=force_threshold,
    )
    contact_ok = contact_flags.gather(1, gather_index).squeeze(1)

    num_objects = len(object_cfgs)
    all_z = torch.full((env.num_envs, num_objects), float("nan"), dtype=torch.float32, device=env.device)
    for object_index, cfg in enumerate(object_cfgs):
        logical_name = _resolve_name(collapse_single_item(cfg))
        position, _ = _object_pos_vel_env(env, logical_name)
        if position is not None:
            all_z[:, object_index] = position[:, 2]

    settle_steps = max(0, int(settle_steps))
    anchor_attr = f"_nth_lifted_anchor_{_stable_term_counter_id(*(str(_resolve_name(collapse_single_item(c))) for c in object_cfgs), str(settle_steps))}"
    if not hasattr(env, anchor_attr):
        setattr(env, anchor_attr, torch.full((env.num_envs, num_objects), float("nan"), device=env.device))
    _register_tensor_episode_state(env, anchor_attr, fill_value=float("nan"))
    anchor_z: torch.Tensor = getattr(env, anchor_attr)
    just_reset = env.episode_length_buf <= 1
    anchor_z[just_reset] = float("nan")
    ready_to_anchor = (env.episode_length_buf >= settle_steps).unsqueeze(1)
    anchor_z = torch.where(torch.isnan(anchor_z) & ready_to_anchor, all_z, anchor_z)
    setattr(env, anchor_attr, anchor_z)

    current_z = all_z.gather(1, gather_index).squeeze(1)
    anchor_z_sel = anchor_z.gather(1, gather_index).squeeze(1)
    anchored = ~torch.isnan(anchor_z_sel)
    lifted = (current_z - anchor_z_sel) >= float(min_lift)

    return valid & contact_ok & anchored & lifted


def gripper_nth_object_settled(
    env: ManagerBasedRLEnv,
    *,
    gripper_contact_sensor_cfgs: list[Any],
    object_cfgs: list[Any],
    nth: int,
    force_threshold: float = 0.1,
    max_lift: float = 0.02,
    release_force_threshold: float = 0.01,
    velocity_threshold: float | None = 0.15,
    consecutive_steps: int = 5,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL multi-object] The *nth* gripper-touched object is back near the table and released.
    Assumes: parallel contact sensors and object configs expose forces, root poses, and linear velocities.

    Identity is the same touch-order ranking as
    :func:`gripper_nth_object_lifted`. Settled means env-z lift vs episode start
    is ``<= max_lift``, gripper contact force on that object is below
    ``release_force_threshold``, and (optionally) linear velocity is low.
    """
    del evaluation_runtime, score_runtime

    index, valid = _nth_touched_object_index(env, gripper_contact_sensor_cfgs, force_threshold, nth)

    num_objects = len(object_cfgs)
    all_z = torch.zeros(env.num_envs, num_objects, device=env.device)
    all_vel_ok = torch.ones(env.num_envs, num_objects, dtype=torch.bool, device=env.device)
    all_released = torch.zeros(env.num_envs, num_objects, dtype=torch.bool, device=env.device)
    for i, (obj_cfg, sensor_cfg) in enumerate(zip(object_cfgs, gripper_contact_sensor_cfgs, strict=True)):
        obj = env.scene[_resolve_name(collapse_single_item(obj_cfg))]
        all_z[:, i] = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        all_vel_ok[:, i] = _velocity_below_threshold(obj, velocity_threshold, env.num_envs, env.device)
        all_released[:, i] = ~_contact_sensor_active(env, collapse_single_item(sensor_cfg), release_force_threshold)

    anchor_attr = f"_nth_settled_anchor_{_stable_term_counter_id(*(str(_resolve_name(collapse_single_item(c))) for c in object_cfgs))}"
    if not hasattr(env, anchor_attr):
        setattr(env, anchor_attr, torch.full((env.num_envs, num_objects), float("nan"), device=env.device))
    _register_tensor_episode_state(env, anchor_attr, fill_value=float("nan"))
    anchor_z: torch.Tensor = getattr(env, anchor_attr)
    just_reset = env.episode_length_buf <= 1
    anchor_z[just_reset] = float("nan")
    anchor_z = torch.where(torch.isnan(anchor_z), all_z, anchor_z)
    setattr(env, anchor_attr, anchor_z)

    gather_idx = index.unsqueeze(1)
    current_z = all_z.gather(1, gather_idx).squeeze(1)
    anchor_z_sel = anchor_z.gather(1, gather_idx).squeeze(1)
    near_table = (current_z - anchor_z_sel) <= float(max_lift)
    released = all_released.gather(1, gather_idx).squeeze(1)
    vel_ok = all_vel_ok.gather(1, gather_idx).squeeze(1)

    raw = valid & near_table & released & vel_ok
    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_nth_settled_" + _stable_term_counter_id(
        *(str(_resolve_name(collapse_single_item(c))) for c in object_cfgs),
        str(nth),
        f"{float(max_lift):.4f}",
        f"{float(release_force_threshold):.4f}",
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def nth_object_near_mth_object(
    env: ManagerBasedRLEnv,
    *,
    gripper_contact_sensor_cfgs: list[Any],
    object_cfgs: list[Any],
    nth: int,
    mth: int,
    force_threshold: float = 0.1,
    xy_distance_threshold: float = 0.15,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL multi-object] The *nth* gripper-touched object within ``xy_distance_threshold`` (XY) of the *mth* touched object.
    Assumes: parallel gripper_contact_sensor_cfgs + object_cfgs; object pose (root_pos_w); identities resolved by touch order.

    Both object identities are resolved dynamically via gripper-contact
    order (see ``_nth_touched_object_index``).
    """
    del evaluation_runtime, score_runtime

    idx_n, valid_n = _nth_touched_object_index(env, gripper_contact_sensor_cfgs, force_threshold, nth)
    idx_m, valid_m = _nth_touched_object_index(env, gripper_contact_sensor_cfgs, force_threshold, mth)

    num_objects = len(object_cfgs)
    all_xy = torch.zeros(env.num_envs, num_objects, 2, device=env.device)
    for i, cfg in enumerate(object_cfgs):
        obj = env.scene[_resolve_name(collapse_single_item(cfg))]
        pos_env = obj.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
        all_xy[:, i, :] = pos_env

    xy_n = all_xy.gather(1, idx_n.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)).squeeze(1)
    xy_m = all_xy.gather(1, idx_m.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)).squeeze(1)

    dist = torch.norm(xy_n - xy_m, dim=-1)
    return valid_n & valid_m & (dist < float(xy_distance_threshold))


def nth_object_contact_others(
    env: ManagerBasedRLEnv,
    *,
    gripper_contact_sensor_cfgs: list[Any],
    inter_object_contact_sensor_cfgs: list[Any],
    nth: int,
    force_threshold_gripper: float = 0.1,
    force_threshold_contact: float = 0.001,
    consecutive_steps: int = 5,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL multi-object] The *nth* gripper-touched object has inter-object contact force above threshold.
    Assumes: parallel gripper_contact_sensor_cfgs + inter_object_contact_sensor_cfgs (force_matrix_w); identity by touch order.

    ``inter_object_contact_sensor_cfgs`` must be a parallel list to
    ``gripper_contact_sensor_cfgs``.  Each entry is a contact sensor that
    monitors forces between that object and the other objects (e.g.
    cup-to-cup contact).  The term checks whether the sensor
    corresponding to the *nth*-touched object reads above
    *force_threshold_contact*.
    """
    del evaluation_runtime, score_runtime

    index, valid = _nth_touched_object_index(env, gripper_contact_sensor_cfgs, force_threshold_gripper, nth)

    num_objects = len(inter_object_contact_sensor_cfgs)
    contact_flags = torch.zeros(env.num_envs, num_objects, dtype=torch.bool, device=env.device)
    for i, cfg in enumerate(inter_object_contact_sensor_cfgs):
        sensor_name = _resolve_name(collapse_single_item(cfg))
        sensor = env.scene[sensor_name]
        force_matrix = as_torch(sensor.data.force_matrix_w)
        if force_matrix is None:
            continue
        force_norm = torch.norm(force_matrix, dim=-1)
        force_per_env = torch.amax(force_norm, dim=tuple(range(1, force_norm.dim())))
        contact_flags[:, i] = force_per_env > float(force_threshold_contact)

    selected = contact_flags.gather(1, index.unsqueeze(1)).squeeze(1)
    raw = valid & selected

    if consecutive_steps <= 1:
        return raw
    counter_attr = f"_nth_contact_others_{_stable_term_counter_id(*(str(_resolve_name(collapse_single_item(c))) for c in inter_object_contact_sensor_cfgs), str(nth))}"
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def contact_mask(
    env: ManagerBasedRLEnv,
    *,
    contact_sensor_cfgs: list[Any],
    force_threshold: float = 0.01,
) -> torch.Tensor:
    """Per-frame ``[num_envs, N]`` bool mask of which sensors are in contact now.

    A stateless ``value_func`` companion to the live subtask log: declared with
    ``update_mode: or`` it latches "which objects have ever been touched" across
    frames inside the per-env log store, with no episode-state buffer of its own
    (the store's reset handles episode boundaries).
    """
    if not isinstance(contact_sensor_cfgs, list):
        contact_sensor_cfgs = [contact_sensor_cfgs]
    return _current_contact_flags(env, contact_sensor_cfgs=contact_sensor_cfgs, force_threshold=force_threshold)


def contact_remaining_all_placed(
    env: ManagerBasedRLEnv,
    *,
    contact_sensor_cfgs: list[Any],
    exclude_log_key: str,
    force_threshold: float = 0.01,
) -> torch.Tensor:
    """True when every not-yet-consumed object is currently in contact (placed).

    Reads the latched "already consumed" ``[num_envs, N]`` mask written by a
    sibling subtask-log cell (``read_log(env, exclude_log_key)``), excludes those
    columns, and requires all remaining objects to read contact this frame —
    i.e. ``(mask | contact).all()``. When the log key was never written the mask
    degrades to all-False (exclude nothing), so the term reduces to "all objects
    placed".
    """

    if not isinstance(contact_sensor_cfgs, list):
        contact_sensor_cfgs = [contact_sensor_cfgs]
    contact = _current_contact_flags(env, contact_sensor_cfgs=contact_sensor_cfgs, force_threshold=force_threshold)
    mask = read_log(env, exclude_log_key)
    if mask is None:
        mask = torch.zeros_like(contact)
    else:
        mask = mask.to(dtype=torch.bool, device=contact.device).reshape(contact.shape)
    return (mask | contact).all(dim=1)


def objects_fraction_in_relative_zone(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    reference_cfg: SceneEntityCfg | str,
    min_fraction: float = 0.8,
    xy_distance_threshold: float = 0.04,
    z_min_offset: float = -0.05,
    z_max_offset: float = 0.12,
    velocity_threshold: float | None = 0.5,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL place] At least ``min_fraction`` of objects lie in a relative zone of ``reference_cfg``.
    Assumes: the reference and candidate objects expose root poses; optional velocity checks need root velocity.

    Containment matches the basket-style check: XY distance to the reference root
    below ``xy_distance_threshold`` and relative Z in ``[z_min_offset, z_max_offset]``.
    Runtime-pool / instance objects resolve via ``_object_pos_vel_env``. Use for
    majority-transfer pour success (e.g. beans mostly inside the receiving cup).
    """
    names = [str(_resolve_name(collapse_single_item(name))) for name in object_names]
    if not names:
        raise ValueError("objects_fraction_in_relative_zone requires a non-empty object_names list")
    fraction = float(min_fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"objects_fraction_in_relative_zone: min_fraction must be in (0, 1], got {fraction}")

    reference_name = _resolve_name(collapse_single_item(reference_cfg))
    ref_pos, _ = _object_pos_vel_env(env, reference_name)
    if ref_pos is None:
        return _false_tensor(env)

    num_slots = len(names)
    in_zone = torch.zeros((env.num_envs, num_slots), dtype=torch.bool, device=env.device)
    for index, name in enumerate(names):
        obj_pos, vel = _object_pos_vel_env(env, name)
        if obj_pos is None:
            continue
        rel = obj_pos - ref_pos
        slot_ok = torch.norm(rel[:, :2], dim=-1) < float(xy_distance_threshold)
        slot_ok = slot_ok & (rel[:, 2] >= float(z_min_offset))
        slot_ok = slot_ok & (rel[:, 2] <= float(z_max_offset))
        if velocity_threshold is not None and vel is not None:
            slot_ok = slot_ok & (vel < float(velocity_threshold))
        in_zone[:, index] = slot_ok

    raw = in_zone.float().sum(dim=1) >= (fraction * float(num_slots) - 1e-6)

    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_objects_fraction_" + _stable_term_counter_id(
        ",".join(names),
        str(reference_name),
        f"{fraction:.4f}",
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def objects_count_in_relative_zone(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    reference_cfg: SceneEntityCfg | str,
    min_count: int = 1,
    xy_distance_threshold: float = 0.04,
    z_min_offset: float = -0.05,
    z_max_offset: float = 0.12,
    velocity_threshold: float | None = 0.5,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL place] At least ``min_count`` objects lie in a relative zone of ``reference_cfg``.

    Same XY/Z (and optional velocity) containment as
    ``objects_fraction_in_relative_zone``, but thresholds by absolute count —
    e.g. scooping a few beans into a target bowl (``min_count=3`` of 20).
    """
    names = [str(_resolve_name(collapse_single_item(name))) for name in object_names]
    if not names:
        raise ValueError("objects_count_in_relative_zone requires a non-empty object_names list")
    need = int(min_count)
    if need < 1:
        raise ValueError(f"objects_count_in_relative_zone: min_count must be >= 1, got {need}")

    reference_name = _resolve_name(collapse_single_item(reference_cfg))
    ref_pos, _ = _object_pos_vel_env(env, reference_name)
    if ref_pos is None:
        return _false_tensor(env)

    num_slots = len(names)
    in_zone = torch.zeros((env.num_envs, num_slots), dtype=torch.bool, device=env.device)
    for index, name in enumerate(names):
        obj_pos, vel = _object_pos_vel_env(env, name)
        if obj_pos is None:
            continue
        rel = obj_pos - ref_pos
        slot_ok = torch.norm(rel[:, :2], dim=-1) < float(xy_distance_threshold)
        slot_ok = slot_ok & (rel[:, 2] >= float(z_min_offset))
        slot_ok = slot_ok & (rel[:, 2] <= float(z_max_offset))
        if velocity_threshold is not None and vel is not None:
            slot_ok = slot_ok & (vel < float(velocity_threshold))
        in_zone[:, index] = slot_ok

    raw = in_zone.sum(dim=1) >= need

    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_objects_count_" + _stable_term_counter_id(
        ",".join(names),
        str(reference_name),
        str(need),
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def objects_cleared_from_relative_zone(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    reference_cfg: SceneEntityCfg | str,
    xy_distance_threshold: float = 0.04,
    z_min_offset: float = -0.05,
    z_max_offset: float = 0.12,
    velocity_threshold: float | None = 0.5,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL place] Detect when no listed object remains in a reference-relative zone.
    Assumes: the reference and candidate objects expose root poses; optional velocity checks need root velocity.

    Negation of a single-object / all-objects relative-zone check used for
    "reveal the hidden object" after lifting a covering cup.
    """
    covered = objects_fraction_in_relative_zone(
        env,
        object_names=object_names,
        reference_cfg=reference_cfg,
        min_fraction=1.0,
        xy_distance_threshold=xy_distance_threshold,
        z_min_offset=z_min_offset,
        z_max_offset=z_max_offset,
        velocity_threshold=velocity_threshold,
        consecutive_steps=1,
    )
    raw = ~covered
    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_objects_cleared_" + _stable_term_counter_id(
        ",".join(str(n) for n in object_names),
        str(_resolve_name(collapse_single_item(reference_cfg))),
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def cup_covers_hidden_object(
    env: ManagerBasedRLEnv,
    *,
    cup_cfg: SceneEntityCfg | str = "cup_target",
    hidden_cfg: SceneEntityCfg | str = "hidden_object",
    min_dot: float = -1.0,
    max_dot: float = -0.7,
    xy_distance_threshold: float = 0.045,
    z_min_offset: float = -0.12,
    z_max_offset: float = -0.02,
    velocity_threshold: float = 0.2,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL place] Detect an inverted cup covering a hidden object.
    Assumes: cup pose and velocity plus hidden-object pose are available in the active scene pool.

    Pool-aware: resolves active cache slots via :func:`_object_pos_vel_env` and
    per-env quaternion reads. Defaults tuned for opaque cups inverted over a
    small hidden object (``cup_p`` / ``toy_dice``; retune after AABB probe).
    """
    cup_name = _resolve_name(collapse_single_item(cup_cfg))
    hidden_name = _resolve_name(collapse_single_item(hidden_cfg))
    device = env.device
    inverted = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    for env_id in range(env.num_envs):
        active_name = resolve_active_scene_entity_name(env, cup_name, env_id=env_id)
        if not scene_has_asset(env.scene, active_name):
            continue
        obj = env.scene[active_name]
        quat = obj.data.root_quat_w.torch[env_id]
        x, y = quat[0], quat[1]
        up_dot = 1.0 - 2.0 * (x * x + y * y)
        vel = torch.norm(obj.data.root_lin_vel_w.torch[env_id])
        inverted[env_id] = (up_dot >= float(min_dot)) & (up_dot <= float(max_dot)) & (vel < float(velocity_threshold))

    under = objects_fraction_in_relative_zone(
        env,
        object_names=[hidden_name],
        reference_cfg=cup_name,
        min_fraction=1.0,
        xy_distance_threshold=xy_distance_threshold,
        z_min_offset=z_min_offset,
        z_max_offset=z_max_offset,
        velocity_threshold=velocity_threshold,
        consecutive_steps=1,
    )
    raw = inverted & under
    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_cup_covers_" + _stable_term_counter_id(cup_name, hidden_name)
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def cup_reveals_hidden_object(
    env: ManagerBasedRLEnv,
    *,
    cup_cfg: SceneEntityCfg | str = "cup_target",
    hidden_cfg: SceneEntityCfg | str = "hidden_object",
    min_cup_height: float = -0.18,
    xy_distance_threshold: float = 0.045,
    z_min_offset: float = -0.12,
    z_max_offset: float = 0.08,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL lift] Detect a lifted target cup that no longer covers the hidden object.
    Assumes: cup and hidden-object root poses are available in the active scene pool.
    """
    cup_name = _resolve_name(collapse_single_item(cup_cfg))
    hidden_name = _resolve_name(collapse_single_item(hidden_cfg))
    device = env.device
    lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    for env_id in range(env.num_envs):
        active_name = resolve_active_scene_entity_name(env, cup_name, env_id=env_id)
        if not scene_has_asset(env.scene, active_name):
            continue
        obj = env.scene[active_name]
        z = obj.data.root_pos_w.torch[env_id, 2] - env.scene.env_origins[env_id, 2]
        lifted[env_id] = z >= float(min_cup_height)

    cleared = objects_cleared_from_relative_zone(
        env,
        object_names=[hidden_name],
        reference_cfg=cup_name,
        xy_distance_threshold=xy_distance_threshold,
        z_min_offset=z_min_offset,
        z_max_offset=z_max_offset,
        velocity_threshold=None,
        consecutive_steps=1,
    )
    raw = lifted & cleared
    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_cup_reveals_" + _stable_term_counter_id(cup_name, hidden_name)
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def cups_swapped_n_times(
    env: ManagerBasedRLEnv,
    *,
    cup_a_cfg: SceneEntityCfg | str = "cup_target",
    cup_b_cfg: SceneEntityCfg | str = "cup_distractor",
    count_log_key: str = "target_swap_count",
    xy_tolerance: float = 0.08,
    velocity_threshold: float = 0.15,
    min_pair_separation: float = 0.08,
) -> torch.Tensor:
    """[SKILL multi-object] Count settled position swaps between two cups.
    Assumes: both cup poses and velocities exist and ``count_log_key`` stores the target per env.

    On first evaluation after episode reset (or when snaps are unset), latches the
    current XY of both cups without counting. A swap increments when each cup is
    within ``xy_tolerance`` of the other's latched slot, both are slow (settled),
    and the pair stays separated by ``min_pair_separation`` (rejects both cups
    collapsing to the midpoint). No height/Z gate. Snaps then update to the new
    positions. Target count is read from ``count_log_key`` (written by
    :func:`manaenv.terms.events.sample_track_object_under_cup`).
    """

    cup_a = _resolve_name(collapse_single_item(cup_a_cfg))
    cup_b = _resolve_name(collapse_single_item(cup_b_cfg))
    term_id = _stable_term_counter_id(cup_a, cup_b, str(count_log_key))
    snap_a_attr = f"_mana_cup_swap_snap_a_{term_id}"
    snap_b_attr = f"_mana_cup_swap_snap_b_{term_id}"
    init_attr = f"_mana_cup_swap_init_{term_id}"
    count_attr = f"_mana_cup_swap_count_{term_id}"
    cool_attr = f"_mana_cup_swap_cool_{term_id}"

    if not hasattr(env, snap_a_attr):
        setattr(env, snap_a_attr, torch.zeros(env.num_envs, 2, dtype=torch.float32, device=env.device))
    if not hasattr(env, snap_b_attr):
        setattr(env, snap_b_attr, torch.zeros(env.num_envs, 2, dtype=torch.float32, device=env.device))
    if not hasattr(env, init_attr):
        setattr(env, init_attr, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    if not hasattr(env, count_attr):
        setattr(env, count_attr, torch.zeros(env.num_envs, dtype=torch.int64, device=env.device))
    if not hasattr(env, cool_attr):
        setattr(env, cool_attr, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    _register_tensor_episode_state(env, snap_a_attr, fill_value=0.0)
    _register_tensor_episode_state(env, snap_b_attr, fill_value=0.0)
    _register_tensor_episode_state(env, init_attr, fill_value=False)
    _register_tensor_episode_state(env, count_attr, fill_value=0)
    _register_tensor_episode_state(env, cool_attr, fill_value=False)

    snap_a: torch.Tensor = getattr(env, snap_a_attr)
    snap_b: torch.Tensor = getattr(env, snap_b_attr)
    initialized: torch.Tensor = getattr(env, init_attr)
    count: torch.Tensor = getattr(env, count_attr)
    cooling: torch.Tensor = getattr(env, cool_attr)

    reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if hasattr(env, "episode_length_buf"):
        reset_mask = reset_mask | (env.episode_length_buf == 0)
    if hasattr(env, "reset_buf"):
        reset_mask = reset_mask | (env.reset_buf > 0)
    if torch.any(reset_mask):
        initialized[reset_mask] = False
        count[reset_mask] = 0
        cooling[reset_mask] = False

    pos_a, vel_a = _object_pos_vel_env(env, cup_a)
    pos_b, vel_b = _object_pos_vel_env(env, cup_b)
    if pos_a is None or pos_b is None:
        return _false_tensor(env)

    need_init = ~initialized
    if torch.any(need_init):
        snap_a[need_init] = pos_a[need_init, :2]
        snap_b[need_init] = pos_b[need_init, :2]
        initialized[need_init] = True
        cooling[need_init] = False

    tol = float(xy_tolerance)
    a_at_b = torch.norm(pos_a[:, :2] - snap_b, dim=-1) < tol
    b_at_a = torch.norm(pos_b[:, :2] - snap_a, dim=-1) < tol
    pair_sep = torch.norm(pos_a[:, :2] - pos_b[:, :2], dim=-1)
    separated = pair_sep >= float(min_pair_separation)
    slow = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if vel_a is not None:
        slow = slow & (vel_a < float(velocity_threshold))
    if vel_b is not None:
        slow = slow & (vel_b < float(velocity_threshold))
    # Settle via velocity only; no root-Z / table_z_max gate.
    swapped_now = a_at_b & b_at_a & separated & slow & initialized

    # Rising-edge: count once per settled swap, then require leaving the swapped
    # configuration before the next increment (cooling latch).
    newly = swapped_now & (~cooling)
    if torch.any(newly):
        count[newly] = count[newly] + 1
        snap_a[newly] = pos_a[newly, :2]
        snap_b[newly] = pos_b[newly, :2]
        cooling[newly] = True
    released = (~swapped_now) & cooling
    if torch.any(released):
        cooling[released] = False

    target = read_log(env, str(count_log_key))
    if target is None:
        return _false_tensor(env)
    target = target.to(device=env.device, dtype=torch.int64).reshape(-1)
    if target.shape[0] < env.num_envs:
        padded = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        padded[: target.shape[0]] = target
        target = padded
    return count >= target[: env.num_envs]


def track_object_under_cup_swap_count_label(
    env: ManagerBasedRLEnv,
    log_key: str = "target_swap_count",
) -> list[str | None]:
    """Map the committed swap-count target to a display string for overlay/log."""

    num_envs = int(env.num_envs)
    idx = read_log(env, str(log_key))
    if idx is None:
        return [None] * num_envs
    idx = idx.to(dtype=torch.int64).reshape(-1)
    out: list[str | None] = []
    for env_id in range(num_envs):
        value = int(idx[env_id].item()) if env_id < idx.shape[0] else -1
        out.append(str(value) if value > 0 else None)
    return out


def track_object_under_cup_role_label(
    env: ManagerBasedRLEnv,
    role: str = "target_cup",
    state_attr: str = "_mana_track_object_under_cup_state",
) -> list[str | None]:
    """Return the slot label for ``target_cup`` or ``distractor_cup`` overlay keys."""
    key = str(role)
    if key not in {"target_cup", "distractor_cup"}:
        raise ValueError(f"track_object_under_cup_role_label role must be target_cup|distractor_cup, got {key!r}")
    state = getattr(env, state_attr, None)
    labels = state.get(key) if isinstance(state, dict) else None
    if not isinstance(labels, dict):
        return [None] * int(env.num_envs)
    return [labels.get(env_id) for env_id in range(int(env.num_envs))]


def attribute_targets_in_basket(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    reference_cfg: SceneEntityCfg | str,
    target_mask_log_key: str = "attribute_target_mask",
    xy_distance_threshold: float = 0.10,
    z_min_offset: float = -0.06,
    z_max_offset: float = 0.20,
    velocity_threshold: float | None = 0.5,
    strict: bool = True,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL place] Every predicate-matching object rests in the basket (strict: non-matching stay out).

    Reads the reset-committed ``[num_envs, N]`` 0/1 target mask (column i aligns
    with ``object_names[i]``) and judges containment by each active object's
    position relative to ``reference_cfg`` (the basket). Runtime-pool objects are
    resolved to their per-env active cache slot. ``strict`` requires
    ``in_basket == mask`` on every slot (targets in AND non-targets out);
    otherwise only ``(~mask) | in_basket`` (all targets in). Absent mask reads as
    all-False. ``consecutive_steps`` damps transient overlaps.
    """

    names = [str(_resolve_name(collapse_single_item(name))) for name in object_names]
    if not names:
        raise ValueError("attribute_targets_in_basket requires a non-empty object_names list")

    reference_name = _resolve_name(collapse_single_item(reference_cfg))
    ref_pos, _ = _object_pos_vel_env(env, reference_name)
    if ref_pos is None:
        return _false_tensor(env)

    mask = read_log(env, str(target_mask_log_key))
    if mask is None:
        return _false_tensor(env)

    num_slots = len(names)
    in_basket = torch.zeros((env.num_envs, num_slots), dtype=torch.bool, device=env.device)
    for index, name in enumerate(names):
        obj_pos, vel = _object_pos_vel_env(env, name)
        if obj_pos is None:
            continue
        rel = obj_pos - ref_pos
        slot_ok = torch.norm(rel[:, :2], dim=-1) < float(xy_distance_threshold)
        slot_ok = slot_ok & (rel[:, 2] >= float(z_min_offset))
        slot_ok = slot_ok & (rel[:, 2] <= float(z_max_offset))
        if velocity_threshold is not None and vel is not None:
            slot_ok = slot_ok & (vel < float(velocity_threshold))
        in_basket[:, index] = slot_ok

    mask = mask.to(dtype=torch.bool, device=env.device).reshape(env.num_envs, -1)
    if mask.shape[1] < num_slots:
        pad = torch.zeros((env.num_envs, num_slots - mask.shape[1]), dtype=torch.bool, device=env.device)
        mask = torch.cat([mask, pad], dim=1)
    mask = mask[:, :num_slots]

    if strict:
        raw = (in_basket == mask).all(dim=1)
    else:
        raw = ((~mask) | in_basket).all(dim=1)

    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_attribute_targets_" + _stable_term_counter_id(
        str(target_mask_log_key),
        ",".join(names),
        str(reference_name),
        str(bool(strict)),
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def attribute_pick_active_label(
    env: ManagerBasedRLEnv, state_attr: str = "_mana_attribute_pick_state"
) -> list[str | None]:
    """Return the reset-sampled per-env predicate/object label for logging/overlay."""
    state = getattr(env, state_attr, None)
    labels = state.get("label") if isinstance(state, dict) else None
    if not isinstance(labels, dict):
        return [None] * int(env.num_envs)
    return [labels.get(env_id) for env_id in range(int(env.num_envs))]


# =====================================================================
# Zone terminations over a set of candidate objects (one active at a time)
# =====================================================================


_PARKED_POOL_SLOT_Z_MARGIN_M = 2.0


def _resolve_live_scene_entity_name(env: ManagerBasedRLEnv, logical_name: str, *, env_id: int) -> str | None:
    """Resolve a logical object to a physically live pool slot or logical asset."""
    logical = str(logical_name or "").strip()
    if not logical:
        return None
    try:
        active = resolve_active_scene_entity_name(env, logical, env_id=int(env_id))
    except (AttributeError, KeyError, TypeError, ValueError):
        active = logical
    candidates = tuple(dict.fromkeys((str(active), logical)))
    origins = getattr(env.scene, "env_origins", None)
    origin_z = 0.0
    if isinstance(origins, torch.Tensor):
        origin_z = float(as_torch(origins)[int(env_id), 2].item())
    for name in candidates:
        if not scene_has_asset(env.scene, name):
            continue
        data = getattr(env.scene[name], "data", None)
        root_pos = getattr(data, "root_pos_w", None)
        position = as_torch(root_pos)
        if not isinstance(position, torch.Tensor) or position.ndim == 0:
            continue
        z = position[2] if position.ndim == 1 else position[int(env_id), 2]
        if bool(torch.isfinite(z).item()) and float(z.item()) >= origin_z - _PARKED_POOL_SLOT_Z_MARGIN_M:
            return name
    return None


def _runtime_place_beside_xy_mask(env: ManagerBasedRLEnv, rel_pos: torch.Tensor) -> torch.Tensor:
    """Per-env XY band from reset-time ``outputs.place_beside`` runtime context."""
    from manaenv.pipeline.runtime.beside_placement import get_place_beside

    device = env.device
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    for env_id in range(env.num_envs):
        bundle = get_place_beside(env, env_id=int(env_id))
        if not bundle:
            continue
        x_range = bundle.get("x_range")
        y_range = bundle.get("y_range")
        if not isinstance(x_range, (list, tuple)) or not isinstance(y_range, (list, tuple)):
            continue
        if len(x_range) < 2 or len(y_range) < 2:
            continue
        dx = rel_pos[int(env_id), 0]
        dy = rel_pos[int(env_id), 1]
        mask[int(env_id)] = (
            (dx >= float(x_range[0]))
            & (dx <= float(x_range[1]))
            & (dy >= float(y_range[0]))
            & (dy <= float(y_range[1]))
        )
    return mask


def _object_pos_vel_env(
    env: ManagerBasedRLEnv,
    logical_name: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return per-env position and velocity for a live logical object."""
    device = env.device
    if getattr(env, "_schema_vnext_active_asset_slots", None):
        pos = torch.full((env.num_envs, 3), 1.0e6, dtype=torch.float32, device=device)
        vel = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
        any_found = False
        for env_id in range(env.num_envs):
            active_name = _resolve_live_scene_entity_name(env, logical_name, env_id=env_id)
            if active_name is None:
                continue
            obj = env.scene[active_name]
            pos[env_id] = as_torch(obj.data.root_pos_w)[env_id] - env.scene.env_origins[env_id]
            if hasattr(obj, "data") and hasattr(obj.data, "root_lin_vel_w"):
                vel[env_id] = torch.norm(as_torch(obj.data.root_lin_vel_w)[env_id])
            any_found = True
        return (pos, vel) if any_found else (None, None)
    if not scene_has_asset(env.scene, logical_name):
        return None, None
    pos, _ = _get_root_pose_env(env, logical_name)
    obj = env.scene[logical_name]
    vel = None
    if hasattr(obj, "data") and hasattr(obj.data, "root_lin_vel_w"):
        vel = torch.norm(as_torch(obj.data.root_lin_vel_w), dim=-1)
    return pos, vel


def _object_quat_xyzw(
    env: ManagerBasedRLEnv,
    logical_name: str,
) -> torch.Tensor | None:
    """Return per-env root quaternion ``[N, 4]`` (XYZW) for an object.

    Same active-slot resolution as :func:`_object_pos_vel_env`. Missing slots keep
    identity quaternion; returns ``None`` when no env can resolve the object.
    """
    device = env.device
    if getattr(env, "_schema_vnext_active_asset_slots", None):
        quat = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=device)
        quat[:, 3] = 1.0
        any_found = False
        for env_id in range(env.num_envs):
            active_name = _resolve_live_scene_entity_name(env, logical_name, env_id=env_id)
            if active_name is None:
                continue
            obj = env.scene[active_name]
            quat[env_id] = as_torch(obj.data.root_quat_w)[env_id]
            any_found = True
        return quat if any_found else None
    if not scene_has_asset(env.scene, logical_name):
        return None
    _, quat = _get_root_pose_env(env, logical_name)
    return quat


def _any_object_in_zone_raw(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    x_range: list[float] | None,
    y_range: list[float] | None,
    z_range: list[float] | None,
    velocity_threshold: float | None,
) -> torch.Tensor:
    """Per-env OR over *object_names* of "inside the box (and slow)".

    Tasks that materialize exactly one active object out of a candidate set (the
    rest parked far below the floor) can treat this as "the active object is in
    the zone": parked objects never satisfy an on-table box. Candidate objects may
    be runtime-randomized (``type: pool``); their per-env active slot is resolved.
    """
    if x_range is None and y_range is None and z_range is None:
        raise ValueError("any_object_in_zone requires at least one of x_range/y_range/z_range")
    device = env.device
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    for name in object_names:
        pos, vel = _object_pos_vel_env(env, name)
        if pos is None:
            continue
        inside = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        if x_range is not None:
            inside = inside & (pos[:, 0] >= float(x_range[0])) & (pos[:, 0] <= float(x_range[1]))
        if y_range is not None:
            inside = inside & (pos[:, 1] >= float(y_range[0])) & (pos[:, 1] <= float(y_range[1]))
        if z_range is not None:
            inside = inside & (pos[:, 2] >= float(z_range[0])) & (pos[:, 2] <= float(z_range[1]))
        if velocity_threshold is not None and vel is not None:
            inside = inside & (vel < float(velocity_threshold))
        result = result | inside
    return result


def any_object_in_zone(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    z_range: list[float] | None = None,
    velocity_threshold: float | None = None,
    consecutive_steps: int = 1,
    counter_attr: str | None = None,
) -> torch.Tensor:
    """[SKILL place] Any of a candidate object set rests inside a fixed env-frame box.

    Intended for tasks that activate one object out of ``object_names`` per episode
    (the rest parked below the floor); "any inside" then means the active one is in
    the zone. ``consecutive_steps`` damps transient overlaps.
    """
    raw = _any_object_in_zone_raw(env, list(object_names), x_range, y_range, z_range, velocity_threshold)
    if consecutive_steps <= 1:
        return raw
    if counter_attr is None:
        counter_attr = "_mana_any_in_zone_" + _stable_term_counter_id(
            *object_names, str(x_range), str(y_range), str(z_range), str(velocity_threshold)
        )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def any_object_in_zone_for_duration(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    z_range: list[float] | None = None,
    velocity_threshold: float | None = None,
    hold_seconds: float,
    counter_attr: str | None = None,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL place] Active object continuously inside a box for >= ``hold_seconds`` (dwell).

    Converts the dwell requirement to frames via ``step_dt_s`` (injected runtime,
    else ``env.step_dt``) so the time threshold is dt-robust, and resets the dwell
    counter whenever the object leaves the box (so a distraction window only
    counts uninterrupted dwell).
    """

    runtime = evaluation_runtime if evaluation_runtime is not None else score_runtime
    step_dt_s = _resolve_step_dt_s(env, runtime)
    if step_dt_s <= 0.0:
        return _false_tensor(env)
    raw = _any_object_in_zone_raw(env, list(object_names), x_range, y_range, z_range, velocity_threshold)
    required_frames = max(1, int(math.ceil(float(hold_seconds) / step_dt_s)))
    if counter_attr is None:
        counter_attr = "_mana_any_in_zone_hold_" + _stable_term_counter_id(
            *object_names, str(hold_seconds), str(x_range), str(y_range), str(z_range)
        )
    return _apply_consecutive_filter(env, raw, required_frames, counter_attr=counter_attr)


def any_object_in_indexed_zone(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    zones: list[dict[str, Any]],
    index_log_key: str,
    velocity_threshold: float | None = None,
    consecutive_steps: int = 1,
) -> torch.Tensor:
    """[SKILL place] Active object inside the box selected per-env by a reset-committed index.

    Reads a per-env integer index from ``index_log_key`` (committed at reset) and
    checks that some object in ``object_names`` rests inside ``zones[index]`` (fixed
    env-frame boxes ordered to match the index). Absent index reads as all-False.
    Used to enforce "return the object to the SAME slot it started on".
    """

    if not zones:
        raise ValueError("any_object_in_indexed_zone requires a non-empty zones list")
    idx_log = read_log(env, str(index_log_key))
    if idx_log is None:
        return _false_tensor(env)
    idx_log = idx_log.to(device=env.device, dtype=torch.int64).reshape(env.num_envs, -1)[:, 0]
    idx_log = idx_log.clamp(0, len(zones) - 1)

    per_zone = [
        _any_object_in_zone_raw(
            env,
            list(object_names),
            zone.get("x_range"),
            zone.get("y_range"),
            zone.get("z_range"),
            velocity_threshold,
        )
        for zone in zones
    ]
    stacked = torch.stack(per_zone, dim=0)  # (num_zones, num_envs)
    env_ids = torch.arange(env.num_envs, device=env.device)
    result = stacked[idx_log, env_ids]

    if consecutive_steps <= 1:
        return result
    counter_attr = "_mana_any_indexed_zone_" + _stable_term_counter_id(
        *object_names, str(index_log_key), str(len(zones))
    )
    return _apply_consecutive_filter(env, result, consecutive_steps, counter_attr=counter_attr)


def any_object_in_indexed_zone_for_duration(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    zones: list[dict[str, Any]],
    index_log_key: str,
    hold_seconds: float,
    velocity_threshold: float | None = None,
    counter_attr: str | None = None,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """[SKILL place] Indexed-zone dwell for >= ``hold_seconds`` (reset when object leaves).
    Assumes: ``index_log_key`` stores one zone-aligned integer per env and runtime step duration is available.

    Same index selection as :func:`any_object_in_indexed_zone`, but the consecutive
    filter length is ``ceil(hold_seconds / step_dt)`` so the wait is dt-robust.
    """

    if not zones:
        raise ValueError("any_object_in_indexed_zone_for_duration requires a non-empty zones list")

    runtime = evaluation_runtime if evaluation_runtime is not None else score_runtime
    step_dt_s = _resolve_step_dt_s(env, runtime)
    if step_dt_s <= 0.0:
        return _false_tensor(env)

    idx_log = read_log(env, str(index_log_key))
    if idx_log is None:
        return _false_tensor(env)
    idx_log = idx_log.to(device=env.device, dtype=torch.int64).reshape(env.num_envs, -1)[:, 0]
    idx_log = idx_log.clamp(0, len(zones) - 1)

    per_zone = [
        _any_object_in_zone_raw(
            env,
            list(object_names),
            zone.get("x_range"),
            zone.get("y_range"),
            zone.get("z_range"),
            velocity_threshold,
        )
        for zone in zones
    ]
    stacked = torch.stack(per_zone, dim=0)
    env_ids = torch.arange(env.num_envs, device=env.device)
    raw = stacked[idx_log, env_ids]

    required_frames = max(1, int(math.ceil(float(hold_seconds) / step_dt_s)))
    if counter_attr is None:
        counter_attr = "_mana_any_indexed_zone_hold_" + _stable_term_counter_id(
            *object_names, str(index_log_key), str(hold_seconds), str(len(zones))
        )
    return _apply_consecutive_filter(env, raw, required_frames, counter_attr=counter_attr)


def find_drawer_active_label(env: ManagerBasedRLEnv, state_attr: str = "_mana_find_drawer_state") -> list[str | None]:
    """Return the reset-sampled per-env target drawer layer label for logging/overlay."""
    state = getattr(env, state_attr, None)
    plans = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plans, dict):
        return [None] * int(env.num_envs)
    return [plans.get(env_id) for env_id in range(int(env.num_envs))]


def whack_a_mole_active_label(env: ManagerBasedRLEnv, state_attr: str = "_mana_whack_a_mole_state") -> list[str | None]:
    """Return the reset-sampled raised-peg / slot plan for logging/overlay."""
    state = getattr(env, state_attr, None)
    plans = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plans, dict):
        return [None] * int(env.num_envs)
    return [plans.get(env_id) for env_id in range(int(env.num_envs))]


def turntable_pick_active_label(
    env: ManagerBasedRLEnv, state_attr: str = "_mana_pick_item_from_turntable_state"
) -> list[str | None]:
    """Return the reset-sampled per-env turntable target label for logging/overlay."""
    state = getattr(env, state_attr, None)
    plans = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plans, dict):
        return [None] * int(env.num_envs)
    return [plans.get(env_id) for env_id in range(int(env.num_envs))]


def brick_round_trip_active_label(
    env: ManagerBasedRLEnv, state_attr: str = "_mana_brick_round_trip_state"
) -> list[str | None]:
    """Return the reset-sampled per-env brick/source-slot label for logging/overlay."""
    state = getattr(env, state_attr, None)
    plans = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plans, dict):
        return [None] * int(env.num_envs)
    return [plans.get(env_id) for env_id in range(int(env.num_envs))]


def stack_blocks_active_label(env: ManagerBasedRLEnv, state_attr: str = "_mana_stack_blocks_state") -> list[str | None]:
    """Return the reset-sampled per-env stack plan label for logging/overlay."""
    state = getattr(env, state_attr, None)
    plans = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plans, dict):
        return [None] * int(env.num_envs)
    return [plans.get(env_id) for env_id in range(int(env.num_envs))]


def stack_blocks_height_label(env: ManagerBasedRLEnv, state_attr: str = "_mana_stack_height_state") -> list[str | None]:
    """Return the reset-sampled per-env variable-height stack plan label."""
    state = getattr(env, state_attr, None)
    plans = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plans, dict):
        return [None] * int(env.num_envs)
    return [plans.get(env_id) for env_id in range(int(env.num_envs))]


def _stable_stack_height_raw(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    target_height: int | torch.Tensor = 1,
    support_cfg: SceneEntityCfg | str = "stack_base",
    x_range: Sequence[float] = (-0.035, 0.085),
    y_range: Sequence[float] = (-0.328, -0.208),
    minimum_height: float = -0.3,
    base_height_range: Sequence[float] = (0.01, 0.025),
    layer_height_range: Sequence[float] = (0.024, 0.036),
    xy_tolerance: float = 0.025,
    velocity_threshold: float = 0.08,
) -> torch.Tensor:
    """Return whether a stable, grounded vertical chain reaches the target.

    A paper-zone count is insufficient: several blocks lying side-by-side on
    the paper are not a tower. Candidates must be nearly still, the first block
    must sit at the declared support height, and every next block must overlap
    the previous block in XY while advancing by one block height in Z.
    """
    device = env.device
    n_envs = int(env.num_envs)
    raw = torch.zeros(n_envs, dtype=torch.bool, device=device)
    if isinstance(target_height, torch.Tensor):
        targets = target_height.to(device=device, dtype=torch.int64).reshape(n_envs, -1)[:, 0].clamp(min=1)
    else:
        targets = torch.full((n_envs,), max(1, int(target_height)), dtype=torch.int64, device=device)
    x_min, x_max = sorted((float(x_range[0]), float(x_range[1])))
    y_min, y_max = sorted((float(y_range[0]), float(y_range[1])))
    min_z = float(minimum_height)
    base_min, base_max = sorted((float(base_height_range[0]), float(base_height_range[1])))
    layer_min, layer_max = sorted((float(layer_height_range[0]), float(layer_height_range[1])))
    xy_limit = float(xy_tolerance)
    speed_limit = float(velocity_threshold)
    names = [str(n).strip() for n in (object_names or []) if str(n).strip()]
    if not names:
        return raw

    support_name = _resolve_name(collapse_single_item(support_cfg))
    support_pos, _support_vel = _object_pos_vel_env(env, support_name)
    if support_pos is None:
        return raw

    positions: list[torch.Tensor] = []
    velocities: list[torch.Tensor | None] = []
    for name in names:
        pos, vel = _object_pos_vel_env(env, name)
        if pos is None:
            continue
        positions.append(pos)
        velocities.append(vel)

    for env_id in range(n_envs):
        candidates: list[tuple[float, float, float]] = []
        for pos, vel in zip(positions, velocities):
            if vel is None or float(vel[env_id].item()) > speed_limit:
                continue
            x = float(pos[env_id, 0].item())
            y = float(pos[env_id, 1].item())
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue
            z = float(pos[env_id, 2].item())
            if z <= min_z:
                continue
            candidates.append((x, y, z))
        if not candidates:
            continue

        candidates.sort(key=lambda item: item[2])
        support_z = float(support_pos[env_id, 2].item())
        support_bbox = _reference_bbox_env(env, support_name, env_id)
        if support_bbox is not None:
            _bbox_min, bbox_max, _active_name = support_bbox
            support_z = float(bbox_max[2].item())
        longest = [0] * len(candidates)
        for upper_index, upper in enumerate(candidates):
            base_offset = upper[2] - support_z
            if base_min <= base_offset <= base_max:
                longest[upper_index] = 1
            for lower_index in range(upper_index):
                if longest[lower_index] <= 0:
                    continue
                lower = candidates[lower_index]
                dz = upper[2] - lower[2]
                dxy = ((upper[0] - lower[0]) ** 2 + (upper[1] - lower[1]) ** 2) ** 0.5
                if layer_min <= dz <= layer_max and dxy <= xy_limit:
                    longest[upper_index] = max(longest[upper_index], longest[lower_index] + 1)
        if longest and max(longest) >= int(targets[env_id].item()):
            raw[env_id] = True

    return raw


def stack_height_reached(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    target_height: int = 1,
    support_cfg: SceneEntityCfg | str = "stack_base",
    x_range: Sequence[float] = (-0.035, 0.085),
    y_range: Sequence[float] = (-0.328, -0.208),
    minimum_height: float = -0.3,
    base_height_range: Sequence[float] = (0.01, 0.025),
    layer_height_range: Sequence[float] = (0.024, 0.036),
    xy_tolerance: float = 0.025,
    velocity_threshold: float = 0.08,
    consecutive_steps: int = 10,
    counter_attr: str | None = None,
) -> torch.Tensor:
    """True after a grounded tower is stable for consecutive simulation steps."""
    raw = _stable_stack_height_raw(
        env,
        object_names=object_names,
        target_height=target_height,
        support_cfg=support_cfg,
        x_range=x_range,
        y_range=y_range,
        minimum_height=minimum_height,
        base_height_range=base_height_range,
        layer_height_range=layer_height_range,
        xy_tolerance=xy_tolerance,
        velocity_threshold=velocity_threshold,
    )
    if consecutive_steps <= 1:
        return raw
    if counter_attr is None:
        counter_attr = "_mana_stack_height_" + _stable_term_counter_id(
            ",".join(str(name) for name in object_names),
            str(target_height),
            str(_resolve_name(collapse_single_item(support_cfg))),
            str(xy_tolerance),
            str(velocity_threshold),
        )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def stack_height_and_joints_near_default(
    env: ManagerBasedRLEnv,
    *,
    object_names: list[str],
    asset_cfg: SceneEntityCfg | str,
    target_height: int = 1,
    target_height_log_key: str | None = None,
    support_cfg: SceneEntityCfg | str = "stack_base",
    x_range: Sequence[float] = (-0.035, 0.085),
    y_range: Sequence[float] = (-0.328, -0.208),
    minimum_height: float = -0.3,
    base_height_range: Sequence[float] = (0.01, 0.025),
    layer_height_range: Sequence[float] = (0.024, 0.036),
    xy_tolerance: float = 0.025,
    velocity_threshold: float = 0.08,
    joint_names: list[str] | None = None,
    exclude_joint_name_substrings: list[str] | None = None,
    atol: float = 0.1,
    consecutive_steps: int = 10,
    counter_attr: str | None = None,
) -> torch.Tensor:
    """Require both an intact grounded tower and an arm at its default pose.

    Tower stages latch as they are completed, but the physical tower can still
    collapse while the arm returns home. Final success therefore revalidates
    the full requested height on every Home-stage frame instead of relying only
    on the earlier stage latches.
    """
    resolved_height: int | torch.Tensor = int(target_height)
    if target_height_log_key:
        logged_height = read_log(env, str(target_height_log_key))
        if logged_height is None:
            return _false_tensor(env)
        resolved_height = logged_height

    tower_intact = _stable_stack_height_raw(
        env,
        object_names=object_names,
        target_height=resolved_height,
        support_cfg=support_cfg,
        x_range=x_range,
        y_range=y_range,
        minimum_height=minimum_height,
        base_height_range=base_height_range,
        layer_height_range=layer_height_range,
        xy_tolerance=xy_tolerance,
        velocity_threshold=velocity_threshold,
    )
    arm_home = articulation_joints_near_default(
        env,
        asset_cfg=asset_cfg,
        joint_names=joint_names,
        exclude_joint_name_substrings=exclude_joint_name_substrings,
        atol=atol,
        consecutive_steps=1,
    )
    raw = tower_intact & arm_home
    if consecutive_steps <= 1:
        return raw
    if counter_attr is None:
        counter_attr = "_mana_stack_home_" + _stable_term_counter_id(
            ",".join(str(name) for name in object_names),
            str(target_height),
            str(target_height_log_key),
            str(_resolve_name(collapse_single_item(support_cfg))),
            str(_resolve_name(collapse_single_item(asset_cfg))),
            str(atol),
        )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def tower_of_hanoi_plan_label(
    env: ManagerBasedRLEnv,
    state_attr: str = "_mana_hanoi_state",
) -> list[str | None]:
    """Return the reset-sampled per-env Hanoi plan label for logging/overlay."""
    state = getattr(env, state_attr, None)
    plans = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plans, dict):
        return [None] * int(env.num_envs)
    return [plans.get(env_id, plans.get(str(env_id))) for env_id in range(int(env.num_envs))]


def hanoi_state_matches(
    env: ManagerBasedRLEnv,
    *,
    ring_names: list[str],
    peg_names: list[str],
    expected: dict[str, list[int] | tuple[int, ...]],
    state_attr: str = "_mana_hanoi_state",
    xy_tolerance: float = 0.03,
    min_z: float = -1.0,
    velocity_threshold: float = 0.2,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """True when active rings match the expected per-peg size-rank stacks.

    ``expected`` maps peg scene id → bottom-to-top size ranks (1 = smallest /
    ``disk_1``). Ranks resolve to concrete ring names via
    ``env._mana_hanoi_state["active_rings"]`` (smallest→largest). Each active
    ring is assigned to the nearest peg in XY; parked rings (z ≤ ``min_z``) are
    ignored. Ordering on a peg is by ascending world z (bottom→top).
    """
    names = [str(n).strip() for n in (ring_names or []) if str(n).strip()]
    pegs = [str(p).strip() for p in (peg_names or []) if str(p).strip()]
    if not names or not pegs:
        return _false_tensor(env)

    expected_norm: dict[str, list[int]] = {}
    for peg, ranks in (expected or {}).items():
        key = str(peg).strip()
        if not key:
            continue
        expected_norm[key] = [int(r) for r in (ranks or [])]

    state = getattr(env, str(state_attr), None)
    active_by_env = state.get("active_rings") if isinstance(state, dict) else None
    if not isinstance(active_by_env, dict):
        active_by_env = {}

    peg_pos: dict[str, torch.Tensor] = {}
    for peg in pegs:
        pos, _ = _object_pos_vel_env(env, peg)
        if pos is None:
            return _false_tensor(env)
        peg_pos[peg] = pos

    ring_pos: dict[str, torch.Tensor] = {}
    ring_vel: dict[str, torch.Tensor | None] = {}
    for name in names:
        pos, vel = _object_pos_vel_env(env, name)
        if pos is None:
            return _false_tensor(env)
        ring_pos[name] = pos
        ring_vel[name] = vel

    device = env.device
    raw = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    tol = float(xy_tolerance)
    min_z_f = float(min_z)
    vel_thr = float(velocity_threshold)

    for env_id in range(int(env.num_envs)):
        active = active_by_env.get(env_id, active_by_env.get(str(env_id)))
        if not isinstance(active, (list, tuple)) or not active:
            # Fall back: any ring above min_z, sorted large→small by asset letter.
            on_table = [name for name in names if float(ring_pos[name][env_id, 2].item()) > min_z_f]
            active = list(reversed(sorted(on_table)))
        active = [str(n) for n in active]
        # rank 1 = smallest = active[0]
        name_to_rank = {name: index + 1 for index, name in enumerate(active)}

        stacks: dict[str, list[tuple[float, int]]] = {peg: [] for peg in pegs}
        for name in active:
            pos = ring_pos[name][env_id]
            z = float(pos[2].item())
            if z <= min_z_f:
                raw[env_id] = False
                break
            vel = ring_vel[name]
            speed = float(vel[env_id].item()) if vel is not None else 0.0
            if vel is not None and speed > vel_thr:
                raw[env_id] = False
                break
            best_peg = None
            best_dist = float("inf")
            for peg in pegs:
                peg_xy = peg_pos[peg][env_id, :2]
                dist = float(torch.linalg.norm(pos[:2] - peg_xy).item())
                if dist < best_dist:
                    best_dist = dist
                    best_peg = peg
            if best_peg is None or best_dist > tol:
                raw[env_id] = False
                break
            stacks[best_peg].append((z, int(name_to_rank[name])))
        else:
            got = {peg: [rank for _, rank in sorted(stacks[peg], key=lambda item: item[0])] for peg in pegs}
            for peg in pegs:
                if got[peg] != expected_norm.get(peg, []):
                    raw[env_id] = False
                    break

    if consecutive_steps <= 1:
        return raw
    # Per-expected counter: DAG success and per-group milestones all call this
    # term with different ``expected`` stacks. A shared counter lets a later
    # frontier stage (failing) reset an earlier stage's streak mid-verify.
    expected_key = "|".join(f"{peg}:{','.join(str(r) for r in expected_norm.get(peg, []))}" for peg in pegs)
    counter_attr = "_mana_hanoi_state_matches_counter_" + _stable_term_counter_id(expected_key)
    return _apply_consecutive_filter(
        env,
        raw,
        int(consecutive_steps),
        counter_attr=counter_attr,
    )


def specific_position_active_label(
    env: ManagerBasedRLEnv, state_attr: str = "_mana_specific_position_state"
) -> list[str | None]:
    """Return the reset-sampled grid shape and camera-ordered target cell."""
    state = getattr(env, state_attr, None)
    plans = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plans, dict):
        return [None] * int(env.num_envs)
    return [plans.get(env_id) for env_id in range(int(env.num_envs))]


def classified_object_on_color_target(
    env: ManagerBasedRLEnv,
    *,
    position: int,
    colors: list[str],
    object_prefix: str = "classify_",
    destination_prefix: str = "paper_",
    object_log_key: str = "classify_object_indices",
    color_log_key: str = "classify_color_indices",
    xy_distance_threshold: float = 0.09,
    z_min_offset: float = -0.02,
    z_max_offset: float = 0.20,
    velocity_threshold: float | None = 0.2,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL place] Reset-selected object rests on its color-matched destination.
    Assumes: reset commits 1-based object/color indices; runtime asset slots expose root pose and velocity.
    """

    object_log = read_log(env, str(object_log_key))
    color_log = read_log(env, str(color_log_key))
    if object_log is None or color_log is None:
        return _false_tensor(env)
    normalized_colors = [str(color).strip().lower() for color in colors if str(color).strip()]
    if not normalized_colors:
        return _false_tensor(env)
    sizes = {color: resolve_pool_cache_size(env, f"{object_prefix}{color}") for color in normalized_colors}
    unique_sizes = set(sizes.values())
    if len(unique_sizes) != 1:
        raise ValueError(
            "classified_object_on_color_target requires a shared scene cache_size across colors; " f"got {sizes}"
        )
    cache_size = int(next(iter(unique_sizes)))
    object_names = [
        pool_cache_slot_name(f"{object_prefix}{color}", slot)
        for color in normalized_colors
        for slot in range(cache_size)
    ]
    destination_names = [f"{destination_prefix}{color}" for color in normalized_colors]
    object_log = object_log.to(device=env.device, dtype=torch.int64).reshape(env.num_envs, -1)
    color_log = color_log.to(device=env.device, dtype=torch.int64).reshape(env.num_envs, -1)
    column = int(position) - 1
    if column < 0 or column >= object_log.shape[1] or column >= color_log.shape[1]:
        return _false_tensor(env)

    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for env_id in range(int(env.num_envs)):
        object_index = int(object_log[env_id, column].item()) - 1
        color_index = int(color_log[env_id, column].item()) - 1
        if not (0 <= object_index < len(object_names) and 0 <= color_index < len(destination_names)):
            continue
        active_name = resolve_active_scene_entity_name(env, object_names[object_index], env_id=env_id)
        obj = env.scene[active_name]
        destination = env.scene[destination_names[color_index]]
        relative = obj.data.root_pos_w.torch[env_id] - destination.data.root_pos_w.torch[env_id]
        in_zone = (
            torch.norm(relative[:2]) <= float(xy_distance_threshold)
            and relative[2] >= float(z_min_offset)
            and relative[2] <= float(z_max_offset)
        )
        if velocity_threshold is not None and hasattr(obj.data, "root_lin_vel_w"):
            in_zone = in_zone and torch.norm(obj.data.root_lin_vel_w.torch[env_id]) <= float(velocity_threshold)
        result[env_id] = bool(in_zone)

    if consecutive_steps <= 1:
        return result
    counter_attr = "_mana_classified_object_on_color_target_" + _stable_term_counter_id(
        str(position), str(object_log_key), str(color_log_key)
    )
    return _apply_consecutive_filter(env, result, consecutive_steps, counter_attr=counter_attr)


def color_classification_active_label(
    env: ManagerBasedRLEnv, state_attr: str = "_mana_color_classification_state"
) -> list[str | None]:
    """Return the reset-sampled color-classification plan for logging and overlay."""
    state = getattr(env, state_attr, None)
    labels = state.get("label") if isinstance(state, dict) else None
    if not isinstance(labels, dict):
        return [None] * int(env.num_envs)
    return [labels.get(env_id) for env_id in range(int(env.num_envs))]


def _overlay_safe_asset_name(name: str, *, max_name_length: int = 24) -> str:
    """Normalize an asset id for omni.ui overlay text.

    Isaac Sim's default overlay font often fails on non-ASCII glyphs (e.g. the
    Unicode ellipsis ``…``), so keep only printable ASCII and truncate with
    ASCII ``...``.
    """
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else "_" for ch in str(name).strip())
    cleaned = cleaned.strip("_") or "asset"
    if max_name_length > 3 and len(cleaned) > max_name_length:
        return f"{cleaned[: max_name_length - 3]}..."
    if max_name_length > 0 and len(cleaned) > max_name_length:
        return cleaned[:max_name_length]
    return cleaned


def schema_vnext_active_asset_ids_label(
    env: ManagerBasedRLEnv,
    object_names: Sequence[str],
    plan_id: str = "runtime",
    unavailable: str = "asset names unavailable",
    max_name_length: int = 24,
) -> list[str]:
    """Return reset-sampled schema-vNext asset IDs for logging/overlay."""
    state = getattr(env, "_schema_vnext_active_asset_slots", None)
    plan_state = state.get(str(plan_id)) if isinstance(state, dict) else None
    labels: list[str] = []
    for env_id in range(int(env.num_envs)):
        env_state = plan_state.get(env_id) if isinstance(plan_state, dict) else None
        lines = []
        found = 0
        for object_name in object_names:
            entry = env_state.get(str(object_name)) if isinstance(env_state, dict) else None
            asset_id = str(entry.get("asset_id") or "").strip() if isinstance(entry, dict) else ""
            if not asset_id:
                # Placeholder keeps line<->shape alignment diagnosable instead of
                # silently dropping the row when one slot entry is missing/empty.
                lines.append(f"{object_name}: <missing>")
                continue
            found += 1
            source, _, short_id = asset_id.partition(":")
            short_id = short_id or source
            # Registry ids join source and name with "__"; drop the source segment
            # whole so no orphan "_" is left behind. Colon-form ids fall back to
            # the legacy single-underscore prefix strip.
            prefix, sep, rest = short_id.partition("__")
            if sep and rest:
                short_id = rest
            else:
                for source_prefix in (f"{source}_", f"{source.split('_', 1)[0]}_"):
                    if short_id.startswith(source_prefix):
                        short_id = short_id[len(source_prefix) :]
                        break
            safe_name = _overlay_safe_asset_name(short_id, max_name_length=max_name_length)
            lines.append(f"{object_name}: {safe_name}")
        labels.append("\n".join(lines) if found else str(unavailable))
    return labels


def _resolve_cloth_prim_path(env_ns: str, cloth_name: str, cloth_prim_relpath: str | None) -> str | None:
    """Resolve the particle-cloth mesh prim under a spawned cloth USD root.

    Prefer a prim with ``PhysxParticleClothAPI`` (live simulation points). An
    explicit ``cloth_prim_relpath`` is only a fallback — a render mesh at that
    path can be valid USD but still hold rest-pose points, which breaks
    in-basket checks after the cloth has moved.
    """
    root = f"{env_ns}/{cloth_name}"
    particle_paths: list[str] = []
    explicit_paths: list[str] = []
    if cloth_prim_relpath:
        explicit_paths.append(f"{root}/{cloth_prim_relpath.lstrip('/')}")

    try:
        import omni.usd
        from pxr import Usd

        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(root)
        if root_prim and root_prim.IsValid():
            for prim in Usd.PrimRange(root_prim):
                schemas = set(prim.GetAppliedSchemas())
                if prim.HasAPI("PhysxParticleClothAPI") or "PhysxParticleClothAPI" in schemas:
                    particle_paths.append(str(prim.GetPath()))
    except Exception:
        pass

    candidates = particle_paths + explicit_paths + [f"{root}/World/cloth001_01"]

    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for path in candidates:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                return path
    except Exception:
        pass

    return candidates[0] if candidates else None


def cloth_folded_flat(
    env: ManagerBasedRLEnv,
    *,
    cloth_scene_key: str,
    max_height: float = 0.06,
    consecutive_steps: int = 10,
    cloth_prim_relpath: str | None = None,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """True when the cloth particles are all within max_height above the table.

    After folding, the cloth height profile is compact (folded layers stacked).
    We read particle world positions via SingleClothPrim and check the vertical
    range of all particles per env.  Works for num_envs=1; for multi-env the
    loop handles each independently.

    cloth_scene_key: scene entity key for the cloth asset (resolved from @role).
    cloth_prim_relpath: optional path relative to the cloth asset root
        (e.g. ``World/towel``). When omitted, auto-detect particle cloth.
    max_height: maximum allowed z-spread of cloth particles in world frame (metres).
    """
    del evaluation_runtime, score_runtime

    num_envs = env.num_envs
    result = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    try:
        from isaacsim.core.prims import SingleClothPrim

        cloth_name = _resolve_name(cloth_scene_key) if not isinstance(cloth_scene_key, str) else cloth_scene_key
        # Strip leading '@' if present (passed through from role resolution)
        cloth_name = cloth_name.lstrip("@")

        for env_idx in range(num_envs):
            # Construct the prim path for this env instance
            env_ns = f"/World/envs/env_{env_idx}"
            cloth_prim_path = _resolve_cloth_prim_path(env_ns, cloth_name, cloth_prim_relpath)
            if cloth_prim_path is None:
                result[env_idx] = False
                continue
            positions = None
            try:
                cloth = SingleClothPrim(prim_path=cloth_prim_path)
                positions = cloth.get_world_positions()  # [N, 3]
                points = _cloth_positions_to_numpy(
                    positions,
                    mesh_path=cloth_prim_path,
                    source="SingleClothPrim.get_world_positions",
                )
                if points is not None:
                    z_vals = points[:, 2]
                    z_spread = float(z_vals.max() - z_vals.min())
                    result[env_idx] = z_spread < float(max_height)
            except Exception as exc:
                _report_cloth_particle_read_error(
                    cloth_prim_path,
                    "SingleClothPrim.get_world_positions",
                    positions,
                    exc,
                )
                result[env_idx] = False

    except Exception:
        # SingleClothPrim unavailable or prim not found — never succeed
        pass

    if consecutive_steps <= 1:
        return result
    return _apply_consecutive_filter(env, result, consecutive_steps, counter_attr="_cloth_folded_flat_counter")


def _cloth_xy_pca_axes(xy: Any) -> tuple[Any, Any] | None:
    """Return (lateral, tangent) unit vectors in XY; lateral aligned toward +world Y (left)."""
    import numpy as np

    if xy is None or len(xy) < 4:
        return None
    center = xy.mean(axis=0)
    centered = xy - center
    cov = (centered.T @ centered) / max(len(xy) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    del evals
    a0 = evecs[:, 0]
    a1 = evecs[:, 1]
    world_y = np.array([0.0, 1.0], dtype=np.float32)
    if abs(float(a0 @ world_y)) >= abs(float(a1 @ world_y)):
        lateral, tangent = a0, a1
    else:
        lateral, tangent = a1, a0
    if float(lateral @ world_y) < 0.0:
        lateral = -lateral
    tn = float(np.linalg.norm(tangent))
    if tn < 1e-6:
        tangent = np.array([1.0, 0.0], dtype=np.float32)
    else:
        tangent = tangent / tn
    ln = float(np.linalg.norm(lateral))
    if ln < 1e-6:
        return None
    lateral = lateral / ln
    return lateral.astype(np.float32), tangent.astype(np.float32)


def _prim_has_particle_cloth_api(prim_path: str) -> bool:
    """True when ``prim_path`` (or any child) has PhysxParticleClothAPI."""
    try:
        import omni.usd
        from pxr import Usd
    except Exception:
        return False
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return False
        for p in Usd.PrimRange(prim):
            schemas = set(p.GetAppliedSchemas()) if hasattr(p, "GetAppliedSchemas") else set()
            if p.HasAPI("PhysxParticleClothAPI") or "PhysxParticleClothAPI" in schemas:
                return True
    except Exception:
        return False
    return False


def _iter_particle_cloth_mesh_paths(prim_path: str) -> list[str]:
    """Return PhysxParticleClothAPI mesh paths under ``prim_path`` (incl. itself)."""
    try:
        import omni.usd
        from pxr import Usd
    except Exception:
        return [prim_path]
    try:
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(prim_path) if stage is not None else None
        if not root or not root.IsValid():
            return [prim_path]
        paths: list[str] = []
        for prim in Usd.PrimRange(root):
            schemas = set(prim.GetAppliedSchemas()) if hasattr(prim, "GetAppliedSchemas") else set()
            if prim.HasAPI("PhysxParticleClothAPI") or "PhysxParticleClothAPI" in schemas:
                paths.append(str(prim.GetPath()))
        return paths or [prim_path]
    except Exception:
        return [prim_path]


def _get_isaac_physics_sim_view() -> Any | None:
    """Return the live IsaacLab/Isaac Sim physics SimulationView, if any."""
    for import_path in (
        "isaacsim.core.simulation_manager.SimulationManager",
        "isaaclab.sim.SimulationManager",
    ):
        try:
            module_name, cls_name = import_path.rsplit(".", 1)
            import importlib

            mod = importlib.import_module(module_name)
            cls = getattr(mod, cls_name)
            view = cls.get_physics_sim_view()
            if view is not None:
                return view
        except Exception:
            continue
    try:
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        view = getattr(sim, "physics_sim_view", None)
        if view is not None:
            return view
    except Exception:
        pass
    return None


def _report_cloth_particle_read_error(
    mesh_path: str,
    source: str,
    value: Any,
    exc: Exception,
) -> None:
    """Emit one actionable diagnostic per failing cloth read boundary."""
    error_key = (str(mesh_path), str(source), type(value).__name__)
    reported = getattr(_report_cloth_particle_read_error, "_reported", None)
    if reported is None:
        reported = set()
        setattr(_report_cloth_particle_read_error, "_reported", reported)
    if error_key in reported:
        return
    reported.add(error_key)
    device = getattr(value, "device", "n/a")
    manaprint(
        f"ERROR: cloth particle read failed prim={mesh_path} source={source} "
        f"type={type(value).__name__} device={device}: {exc}"
    )


def _cloth_positions_to_numpy(positions: Any, *, mesh_path: str, source: str) -> Any | None:
    """Convert one PhysX cloth-position buffer to an ``Nx3`` NumPy array."""
    import numpy as np

    try:
        arr = to_numpy_keep_shape(positions)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
    except Exception as exc:
        _report_cloth_particle_read_error(mesh_path, source, positions, exc)
        return None
    return arr if arr.shape[0] >= 4 else None


def _read_cloth_particle_points_world(prim_path: str) -> Any | None:  # noqa: C901
    """Return Nx3 world positions from PhysX particle cloth, or None.

    Must initialize ``ClothPrim`` against IsaacLab's shared
    ``SimulationManager.get_physics_sim_view()``. A bare
    ``ClothPrim.initialize()`` builds a *new* SimulationView that often has no
    valid particle-cloth handle during interactive manipulation — which previously caused
    ``particle_unavailable`` and permanently blocked in-basket success.
    """
    try:
        from isaacsim.core.prims import ClothPrim
    except Exception:
        return None

    cache = getattr(_read_cloth_particle_points_world, "_views", None)
    if cache is None:
        cache = {}
        setattr(_read_cloth_particle_points_world, "_views", cache)
    seq = getattr(_read_cloth_particle_points_world, "_seq", 0)

    def _positions_from_view(view: Any, mesh_path: str) -> Any | None:
        if view is None:
            return None
        if not bool(getattr(view, "is_physics_handle_valid", lambda: False)()):
            return None
        try:
            positions = view.get_world_positions()
        except Exception as exc:
            _report_cloth_particle_read_error(mesh_path, "ClothPrim.get_world_positions", None, exc)
            return None
        if positions is None:
            return None
        return _cloth_positions_to_numpy(positions, mesh_path=mesh_path, source="ClothPrim.get_world_positions")

    def _try_view(mesh_path: str) -> Any | None:
        nonlocal seq
        cached = cache.get(mesh_path)
        pts = _positions_from_view(cached, mesh_path)
        if pts is not None:
            return pts

        physics_sim_view = _get_isaac_physics_sim_view()
        # Unique name each recreate — Isaac prim-view registry rejects reuse of
        # a dead named ClothPrim after handle invalidation.
        seq += 1
        setattr(_read_cloth_particle_points_world, "_seq", seq)
        name = f"mana_term_cloth_{abs(hash(mesh_path)) & 0xFFFF:04x}_{seq}"
        try:
            view = ClothPrim(
                prim_paths_expr=mesh_path,
                name=name,
                reset_xform_properties=False,
            )
            if physics_sim_view is not None:
                view.initialize(physics_sim_view)
            else:
                view.initialize()
        except Exception:
            cache.pop(mesh_path, None)
            return None

        pts = _positions_from_view(view, mesh_path)
        if pts is None:
            cache.pop(mesh_path, None)
            return None
        cache[mesh_path] = view
        return pts

    # Prefer direct PhysX tensor API when shared sim view is available.
    physics_sim_view = _get_isaac_physics_sim_view()
    if physics_sim_view is not None and hasattr(physics_sim_view, "create_particle_cloth_view"):
        for mesh_path in _iter_particle_cloth_mesh_paths(prim_path):
            positions = None
            try:
                pattern = mesh_path.replace(".*", "*")
                cloth_view = physics_sim_view.create_particle_cloth_view(pattern)
                if cloth_view is None:
                    continue
                positions = cloth_view.get_positions()
                if positions is None:
                    continue
                arr = _cloth_positions_to_numpy(
                    positions,
                    mesh_path=mesh_path,
                    source="SimulationView.get_positions",
                )
                if arr is not None:
                    return arr
            except Exception as exc:
                _report_cloth_particle_read_error(mesh_path, "SimulationView.get_positions", positions, exc)
                continue

    for mesh_path in _iter_particle_cloth_mesh_paths(prim_path):
        pts = _try_view(mesh_path)
        if pts is not None:
            return pts

    # Asset-root callers: also try parent (e.g. .../towel_b).
    if "/" in prim_path.rstrip("/"):
        parent = prim_path.rstrip("/").rsplit("/", 1)[0]
        if parent and parent not in ("/World/envs", "/World"):
            for mesh_path in _iter_particle_cloth_mesh_paths(parent):
                pts = _try_view(mesh_path)
                if pts is not None:
                    return pts

    return None


def _read_cloth_points_world(prim_path: str) -> Any | None:
    """Prefer live particle positions; fall back to USD mesh × LocalToWorld."""
    pts = _read_cloth_particle_points_world(prim_path)
    if pts is not None:
        return pts
    # Never use USD rest mesh when this prim is a live particle cloth — that
    # reports the hang-bar spawn pose and falsely fails in-basket checks.
    if _prim_has_particle_cloth_api(prim_path):
        return None
    return _read_cloth_mesh_points_world(prim_path)


def cloth_edge_midpoints_aligned(
    env: ManagerBasedRLEnv,
    *,
    cloth_scene_key: str,
    max_midpoint_distance: float = 0.12,
    consecutive_steps: int = 5,
    cloth_prim_relpath: str | None = None,
    rim_fraction: float = 0.08,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """True when left/right cloth edge midpoints are within ``max_midpoint_distance``.

    Uses PCA in XY: lateral axis toward world +Y (left). Right/left rim bands
    are the outermost ``rim_fraction`` of the lateral span; each band's
    midpoint is the mean of those points. Intended as a looser fold-success
    check for interactive evaluation (midpoints only, not four corners).
    """
    del evaluation_runtime, score_runtime
    import numpy as np

    num_envs = env.num_envs
    result = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    cloth_name = _resolve_name(cloth_scene_key) if not isinstance(cloth_scene_key, str) else cloth_scene_key
    cloth_name = cloth_name.lstrip("@")
    term_id = _stable_term_counter_id(cloth_name, str(cloth_prim_relpath), "edge_mids")
    max_d = float(max_midpoint_distance)
    rim_frac = max(float(rim_fraction), 1e-3)

    for env_idx in range(num_envs):
        env_ns = f"/World/envs/env_{env_idx}"
        cloth_prim_path = _resolve_cloth_prim_path(env_ns, cloth_name, cloth_prim_relpath)
        if cloth_prim_path is None:
            continue
        pts = _read_cloth_points_world(cloth_prim_path)
        if pts is None or len(pts) < 8:
            continue

        xy = np.asarray(pts[:, :2], dtype=np.float32)
        axes = _cloth_xy_pca_axes(xy)
        if axes is None:
            continue
        lateral, _tangent = axes
        lat = (xy - xy.mean(axis=0)) @ lateral
        p_min = float(lat.min())
        p_max = float(lat.max())
        span = max(p_max - p_min, 1e-4)
        band = rim_frac * span
        right_mask = lat <= (p_min + band)
        left_mask = lat >= (p_max - band)
        if int(right_mask.sum()) < 3 or int(left_mask.sum()) < 3:
            continue
        right_mid = np.asarray(pts[right_mask], dtype=np.float32).mean(axis=0)
        left_mid = np.asarray(pts[left_mask], dtype=np.float32).mean(axis=0)
        result[env_idx] = float(np.linalg.norm(right_mid - left_mid)) <= max_d

    if consecutive_steps <= 1:
        return result
    return _apply_consecutive_filter(
        env,
        result,
        consecutive_steps,
        counter_attr=f"_cloth_edge_midpoints_aligned_{term_id}",
    )


def cloth_opposite_corners_stacked(
    env: ManagerBasedRLEnv,
    *,
    cloth_scene_key: str,
    max_corner_distance: float = 0.08,
    consecutive_steps: int = 5,
    cloth_prim_relpath: str | None = None,
    rim_fraction: float = 0.10,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """True when each side's opposite corners are stacked within ``max_corner_distance``.

    Intended for "fold the near two corners onto the far two corners":
    on the left and right rims (PCA lateral extremes), take the frontmost and
    backmost particles (PCA tangent extremes) and require both pairs to be close.
    Unfolded towels have a large rim span; a correct half-fold collapses it.
    """
    del evaluation_runtime, score_runtime
    import numpy as np

    num_envs = env.num_envs
    result = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    cloth_name = _resolve_name(cloth_scene_key) if not isinstance(cloth_scene_key, str) else cloth_scene_key
    cloth_name = cloth_name.lstrip("@")
    term_id = _stable_term_counter_id(cloth_name, str(cloth_prim_relpath), "opp_corners")
    max_d = float(max_corner_distance)
    rim_frac = max(float(rim_fraction), 1e-3)

    for env_idx in range(num_envs):
        env_ns = f"/World/envs/env_{env_idx}"
        cloth_prim_path = _resolve_cloth_prim_path(env_ns, cloth_name, cloth_prim_relpath)
        if cloth_prim_path is None:
            continue
        pts = _read_cloth_points_world(cloth_prim_path)
        if pts is None or len(pts) < 8:
            continue

        xy = np.asarray(pts[:, :2], dtype=np.float32)
        axes = _cloth_xy_pca_axes(xy)
        if axes is None:
            continue
        lateral, tangent = axes
        # Orient tangent so +tangent points away from the robot (≈ +world X / far).
        world_x = np.array([1.0, 0.0], dtype=np.float32)
        if float(tangent @ world_x) < 0.0:
            tangent = -tangent

        center = xy.mean(axis=0)
        lat = (xy - center) @ lateral
        tan = (xy - center) @ tangent
        lat_min = float(lat.min())
        lat_max = float(lat.max())
        lat_span = max(lat_max - lat_min, 1e-4)
        band = rim_frac * lat_span
        left_mask = lat >= (lat_max - band)
        right_mask = lat <= (lat_min + band)
        if int(left_mask.sum()) < 3 or int(right_mask.sum()) < 3:
            continue

        def _rim_corner_distance(mask: Any) -> float | None:
            rim_pts = np.asarray(pts[mask], dtype=np.float32)
            rim_tan = tan[mask]
            i_front = int(np.argmin(rim_tan))
            i_back = int(np.argmax(rim_tan))
            if i_front == i_back:
                return 0.0
            return float(np.linalg.norm(rim_pts[i_front] - rim_pts[i_back]))

        d_left = _rim_corner_distance(left_mask)
        d_right = _rim_corner_distance(right_mask)
        if d_left is None or d_right is None:
            continue
        result[env_idx] = d_left <= max_d and d_right <= max_d

    if consecutive_steps <= 1:
        return result
    return _apply_consecutive_filter(
        env,
        result,
        consecutive_steps,
        counter_attr=f"_cloth_opposite_corners_stacked_{term_id}",
    )


def _read_cloth_mesh_points_world(prim_path: str) -> Any | None:
    """Return Nx3 world positions from a UsdGeom.Mesh (or None on failure)."""
    try:
        import numpy as np
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception:
        return None

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if points is None or len(points) == 0:
        return None
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world = np.empty((len(points), 3), dtype=np.float32)
    for i, p in enumerate(points):
        wp = mat.Transform(p)
        world[i, 0] = float(wp[0])
        world[i, 1] = float(wp[1])
        world[i, 2] = float(wp[2])
    return world


def _resolve_rigid_asset_prim_path(asset: Any, env_idx: int) -> str | None:
    """Resolve an IsaacLab rigid asset prim path template for one env."""
    cfg = getattr(asset, "cfg", None)
    prim_path = getattr(cfg, "prim_path", None) if cfg is not None else None
    if not prim_path:
        return None
    path = str(prim_path)
    path = path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{int(env_idx)}")
    path = path.replace("env_.*", f"env_{int(env_idx)}")
    if path.startswith("envs/"):
        path = f"/World/{path}"
    return path


def _container_root_local_aabb_from_meshes(root_prim_path: str) -> tuple[Any, Any] | None:
    """AABB of container geometry in the **root prim local frame**."""
    import numpy as np

    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return None
    try:
        root_to_world = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_to_root = root_to_world.GetInverse()
    except Exception:
        return None

    pts_local: list[Any] = []
    try:
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            if points is None or len(points) == 0:
                continue
            mesh_to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for p in points:
                pw = mesh_to_world.Transform(p)
                pl = world_to_root.Transform(pw)
                pts_local.append([float(pl[0]), float(pl[1]), float(pl[2])])
    except Exception:
        return None

    if len(pts_local) < 8:
        try:
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bound = cache.ComputeWorldBound(root)
            rng = bound.ComputeAlignedRange()
            if rng.IsEmpty():
                return None
            bmin = np.array(rng.GetMin(), dtype=np.float32)
            bmax = np.array(rng.GetMax(), dtype=np.float32)
            corners = [
                [bmin[0], bmin[1], bmin[2]],
                [bmin[0], bmin[1], bmax[2]],
                [bmin[0], bmax[1], bmin[2]],
                [bmin[0], bmax[1], bmax[2]],
                [bmax[0], bmin[1], bmin[2]],
                [bmax[0], bmin[1], bmax[2]],
                [bmax[0], bmax[1], bmin[2]],
                [bmax[0], bmax[1], bmax[2]],
            ]
            for c in corners:
                pl = world_to_root.Transform(c)
                pts_local.append([float(pl[0]), float(pl[1]), float(pl[2])])
        except Exception:
            return None

    if not pts_local:
        return None
    arr = np.asarray(pts_local, dtype=np.float32)
    return arr.min(axis=0), arr.max(axis=0)


def _container_local_aabb_limits(
    container: Any,
    env_idx: int,
    *,
    use_container_aabb: bool,
    aabb_scale: float,
    xy_half_extents: list[float] | tuple[float, float],
    z_min_offset: float,
    z_max_offset: float,
) -> tuple[Any, Any] | None:
    """Root-local axis limits (min_xyz, max_xyz) for a container OBB check."""
    import numpy as np

    scale = float(max(aabb_scale, 1e-3))
    if use_container_aabb:
        prim_path = _resolve_rigid_asset_prim_path(container, env_idx)
        if prim_path:
            aabb = _container_root_local_aabb_from_meshes(prim_path)
            if aabb is not None:
                bmin, bmax = aabb
                center = 0.5 * (bmin + bmax)
                half = 0.5 * np.maximum(bmax - bmin, 1e-4) * scale
                return center - half, center + half

    hx = float(xy_half_extents[0]) * scale
    hy = float(xy_half_extents[1]) * scale
    z_lo = float(z_min_offset)
    z_hi = float(z_max_offset)
    z_mid = 0.5 * (z_lo + z_hi)
    z_half = 0.5 * (z_hi - z_lo) * scale
    return (
        np.array([-hx, -hy, z_mid - z_half], dtype=np.float32),
        np.array([hx, hy, z_mid + z_half], dtype=np.float32),
    )


def _world_points_to_container_local(
    pts_w: Any,
    root_pos_w: Any,
    root_quat_xyzw: Any,
) -> Any:
    """Transform Nx3 world points into the container root local frame."""
    import numpy as np

    pts = np.asarray(pts_w, dtype=np.float32).reshape(-1, 3)
    pos = np.asarray(root_pos_w, dtype=np.float32).reshape(3)
    quat = np.asarray(root_quat_xyzw, dtype=np.float32).reshape(4)
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    diff = pts - pos
    local = np.empty_like(diff)
    local[:, 0] = r00 * diff[:, 0] + r10 * diff[:, 1] + r20 * diff[:, 2]
    local[:, 1] = r01 * diff[:, 0] + r11 * diff[:, 1] + r21 * diff[:, 2]
    local[:, 2] = r02 * diff[:, 0] + r12 * diff[:, 1] + r22 * diff[:, 2]
    return local


def _points_inside_local_aabb(pts_local: Any, bmin: Any, bmax: Any) -> Any:
    import numpy as np

    pts = np.asarray(pts_local, dtype=np.float32).reshape(-1, 3)
    lo = np.asarray(bmin, dtype=np.float32).reshape(3)
    hi = np.asarray(bmax, dtype=np.float32).reshape(3)
    return (
        (pts[:, 0] >= lo[0])
        & (pts[:, 0] <= hi[0])
        & (pts[:, 1] >= lo[1])
        & (pts[:, 1] <= hi[1])
        & (pts[:, 2] >= lo[2])
        & (pts[:, 2] <= hi[2])
    )


def _read_cloth_points_world_with_source(prim_path: str) -> tuple[Any | None, str]:
    """Return (points, source) where source is ``particle``, ``particle_unavailable``, or ``mesh_fallback``."""
    pts = _read_cloth_particle_points_world(prim_path)
    if pts is not None:
        return pts, "particle"
    # Particle cloth with a dead physics handle: do NOT use USD rest mesh
    # (hang-bar spawn pose) — that falsely fails in-basket success checks.
    if _prim_has_particle_cloth_api(prim_path):
        return None, "particle_unavailable"
    return _read_cloth_mesh_points_world(prim_path), "mesh_fallback"


def _local_aabb_corners_world(
    bmin: Any,
    bmax: Any,
    root_pos_w: Any,
    root_quat_xyzw: Any,
) -> Any:
    """Return 8 OBB corners in world frame from root-local AABB + root pose."""
    import numpy as np

    lo = np.asarray(bmin, dtype=np.float32).reshape(3)
    hi = np.asarray(bmax, dtype=np.float32).reshape(3)
    corners_local = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], lo[2]],
            [hi[0], hi[1], hi[2]],
        ],
        dtype=np.float32,
    )
    pos = np.asarray(root_pos_w, dtype=np.float32).reshape(3)
    quat = np.asarray(root_quat_xyzw, dtype=np.float32).reshape(4)
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    world = np.empty_like(corners_local)
    world[:, 0] = r00 * corners_local[:, 0] + r01 * corners_local[:, 1] + r02 * corners_local[:, 2] + pos[0]
    world[:, 1] = r10 * corners_local[:, 0] + r11 * corners_local[:, 1] + r12 * corners_local[:, 2] + pos[1]
    world[:, 2] = r20 * corners_local[:, 0] + r21 * corners_local[:, 1] + r22 * corners_local[:, 2] + pos[2]
    return world


def _pick_scene_asset_name(env: ManagerBasedRLEnv, candidates: Sequence[str]) -> str | None:
    for name in candidates:
        key = str(name).lstrip("@").strip()
        if not key:
            continue
        if scene_has_asset(env, key):
            return key
        resolved = resolve_active_scene_entity_name(env, key)
        if resolved and scene_has_asset(env, resolved):
            return resolved
    return None


def dump_cloth_container_debug_snapshot(
    env: ManagerBasedRLEnv,
    *,
    output_dir: str | None = None,
    cloth_scene_key: str | None = None,
    container_cfg: SceneEntityCfg | str | None = None,
    cloth_prim_relpath: str | None = None,
    use_container_aabb: bool = True,
    aabb_scale: float = 1.0,
    xy_half_extents: list[float] | tuple[float, float] = (0.20, 0.20),
    z_min_offset: float = -0.05,
    z_max_offset: float = 0.30,
) -> str | None:
    """Write cloth particles / COM / container OBB debug JSON (e.g. on Enter success).

    Returns the written file path, or ``None`` on failure.
    """
    from datetime import datetime

    import numpy as np

    cloth_candidates: list[str] = []
    if cloth_scene_key is not None:
        cloth_candidates.append(
            _resolve_name(cloth_scene_key) if not isinstance(cloth_scene_key, str) else cloth_scene_key
        )
    cloth_candidates.extend(["towel_b", "cloth_1", "towel_a"])

    container_candidates: list[str] = []
    if container_cfg is not None:
        container_candidates.append(_resolve_name(collapse_single_item(container_cfg)))
    container_candidates.extend(["basket_a", "destination", "basket"])

    cloth_name = _pick_scene_asset_name(env, cloth_candidates)
    container_name = _pick_scene_asset_name(env, container_candidates)

    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tick": int(getattr(env, "common_step_counter", -1) or -1),
        "cloth_scene_key_requested": cloth_scene_key,
        "container_cfg_requested": str(container_cfg),
        "cloth_name": cloth_name,
        "container_name": container_name,
        "physics_sim_view_available": _get_isaac_physics_sim_view() is not None,
        "params": {
            "cloth_prim_relpath": cloth_prim_relpath,
            "use_container_aabb": bool(use_container_aabb),
            "aabb_scale": float(aabb_scale),
            "xy_half_extents": [float(xy_half_extents[0]), float(xy_half_extents[1])],
            "z_min_offset": float(z_min_offset),
            "z_max_offset": float(z_max_offset),
        },
        "envs": [],
    }

    container = None
    if container_name is not None:
        try:
            container = env.scene[container_name]
        except Exception as exc:
            payload["container_error"] = str(exc)

    for env_idx in range(int(env.num_envs)):
        env_ns = f"/World/envs/env_{env_idx}"
        entry: dict[str, Any] = {"env_idx": env_idx, "env_ns": env_ns}
        if cloth_name is None:
            entry["error"] = "cloth asset not found in scene"
            payload["envs"].append(entry)
            continue

        cloth_prim_path = _resolve_cloth_prim_path(env_ns, cloth_name, cloth_prim_relpath)
        entry["cloth_prim_path"] = cloth_prim_path
        if cloth_prim_path is None:
            entry["error"] = "cloth prim path unresolved"
            payload["envs"].append(entry)
            continue

        pts, source = _read_cloth_points_world_with_source(cloth_prim_path)
        entry["points_source"] = source
        entry["has_particle_cloth_api"] = _prim_has_particle_cloth_api(cloth_prim_path)
        entry["particle_cloth_mesh_paths"] = _iter_particle_cloth_mesh_paths(cloth_prim_path)
        if pts is None or len(pts) == 0:
            entry["error"] = "no cloth points"
            # Still dump container pose/OBB so Enter debug is useful when particle
            # read fails (previous dumps only said particle_unavailable).
            if container is not None:
                try:
                    cpos = container.data.root_pos_w.torch[env_idx].detach().cpu().numpy()
                    cquat = container.data.root_quat_w.torch[env_idx].detach().cpu().numpy()
                    entry["container_root_pos_w"] = np.asarray(cpos, dtype=np.float32).reshape(3).tolist()
                    entry["container_root_quat_xyzw"] = np.asarray(cquat, dtype=np.float32).reshape(4).tolist()
                    entry["container_prim_path"] = _resolve_rigid_asset_prim_path(container, env_idx)
                    limits = _container_local_aabb_limits(
                        container,
                        env_idx,
                        use_container_aabb=bool(use_container_aabb),
                        aabb_scale=float(aabb_scale),
                        xy_half_extents=xy_half_extents,
                        z_min_offset=float(z_min_offset),
                        z_max_offset=float(z_max_offset),
                    )
                    if limits is not None:
                        bmin, bmax = limits
                        entry["obb_local_aabb"] = {
                            "min": np.asarray(bmin).tolist(),
                            "max": np.asarray(bmax).tolist(),
                        }
                        entry["obb_corners_world"] = _local_aabb_corners_world(bmin, bmax, cpos, cquat).tolist()
                except Exception as exc:
                    entry["container_dump_error"] = str(exc)
            manaprint(
                f"[cloth_debug_dump] env={env_idx} src={source} error=no_points "
                f"phys_view={payload['physics_sim_view_available']} "
                f"api={entry['has_particle_cloth_api']} meshes={entry['particle_cloth_mesh_paths']}"
            )
            payload["envs"].append(entry)
            continue

        pts_arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        com_w = pts_arr.mean(axis=0)
        entry["num_points"] = int(len(pts_arr))
        entry["points_world"] = pts_arr.tolist()
        entry["com_world"] = com_w.tolist()
        entry["points_world_aabb"] = {
            "min": pts_arr.min(axis=0).tolist(),
            "max": pts_arr.max(axis=0).tolist(),
        }

        if container is None:
            entry["error"] = "container asset missing"
            payload["envs"].append(entry)
            continue

        cpos = container.data.root_pos_w.torch[env_idx].detach().cpu().numpy()
        cquat = container.data.root_quat_w.torch[env_idx].detach().cpu().numpy()
        entry["container_root_pos_w"] = np.asarray(cpos, dtype=np.float32).reshape(3).tolist()
        entry["container_root_quat_xyzw"] = np.asarray(cquat, dtype=np.float32).reshape(4).tolist()
        entry["container_prim_path"] = _resolve_rigid_asset_prim_path(container, env_idx)

        limits = _container_local_aabb_limits(
            container,
            env_idx,
            use_container_aabb=bool(use_container_aabb),
            aabb_scale=float(aabb_scale),
            xy_half_extents=xy_half_extents,
            z_min_offset=float(z_min_offset),
            z_max_offset=float(z_max_offset),
        )
        if limits is None:
            entry["error"] = "container local AABB failed"
            payload["envs"].append(entry)
            continue

        bmin, bmax = limits
        entry["obb_local_aabb"] = {"min": np.asarray(bmin).tolist(), "max": np.asarray(bmax).tolist()}
        entry["obb_corners_world"] = _local_aabb_corners_world(bmin, bmax, cpos, cquat).tolist()

        pts_local = _world_points_to_container_local(pts_arr, cpos, cquat)
        com_local = _world_points_to_container_local(com_w.reshape(1, 3), cpos, cquat).reshape(3)
        inside = _points_inside_local_aabb(pts_local, bmin, bmax)
        frac = float(inside.sum()) / float(len(pts_local))
        com_ok = bool(_points_inside_local_aabb(com_local.reshape(1, 3), bmin, bmax)[0])
        entry["points_local"] = pts_local.tolist()
        entry["com_local"] = com_local.tolist()
        entry["fraction_inside_obb"] = frac
        entry["com_inside_obb"] = com_ok
        entry["num_inside_obb"] = int(inside.sum())
        payload["envs"].append(entry)

        manaprint(
            f"[cloth_debug_dump] env={env_idx} src={source} n={len(pts_arr)} "
            f"com_w={com_w.tolist()} com_local={com_local.tolist()} "
            f"frac_inside={frac:.3f} com_inside={com_ok} "
            f"aabb=[{np.asarray(bmin).tolist()} .. {np.asarray(bmax).tolist()}] "
            f"container_pos={entry['container_root_pos_w']}"
        )

    out_dir = output_dir or os.path.join(os.path.expanduser("~"), "manaenv_debug")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.path.join(os.path.expanduser("~"), "manaenv_debug")
        os.makedirs(out_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"cloth_container_debug_{stamp}_{int(time.time() * 1000) % 1000:03d}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        manaprint(f"WARNING: [cloth_debug_dump] failed to write {out_path}: {exc}")
        return None

    manaprint(f"[cloth_debug_dump] wrote {out_path}")
    return out_path


def cloth_in_container(
    env: ManagerBasedRLEnv,
    *,
    cloth_scene_key: str,
    container_cfg: SceneEntityCfg | str,
    cloth_prim_relpath: str | None = None,
    xy_half_extents: list[float] | tuple[float, float] = (0.12, 0.12),
    z_min_offset: float = -0.02,
    z_max_offset: float = 0.18,
    min_fraction: float = 0.45,
    consecutive_steps: int = 10,
    use_container_aabb: bool = True,
    aabb_scale: float = 0.9,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """True when enough cloth points lie inside the container OBB (root-local AABB)."""
    del evaluation_runtime, score_runtime

    num_envs = env.num_envs
    result = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    cloth_name = _resolve_name(cloth_scene_key) if not isinstance(cloth_scene_key, str) else cloth_scene_key
    cloth_name = cloth_name.lstrip("@")
    container_name = _resolve_name(collapse_single_item(container_cfg))

    try:
        container = env.scene[container_name]
    except Exception:
        manaprint(f"[cloth_in_container] missing container asset {container_name!r}")
        return result

    need = float(min_fraction)
    hx = float(xy_half_extents[0])
    hy = float(xy_half_extents[1])
    debug_attr = "_mana_cloth_in_container_debug_tick"
    tick = int(getattr(env, debug_attr, 0)) + 1
    setattr(env, debug_attr, tick)
    do_debug = tick % 60 == 1

    for env_idx in range(num_envs):
        env_ns = f"/World/envs/env_{env_idx}"
        cloth_prim_path = _resolve_cloth_prim_path(env_ns, cloth_name, cloth_prim_relpath)
        if cloth_prim_path is None:
            if do_debug:
                manaprint(f"[cloth_in_container] no cloth prim under {env_ns}/{cloth_name}")
            continue
        pts, source = _read_cloth_points_world_with_source(cloth_prim_path)
        if pts is None or len(pts) == 0:
            if do_debug:
                manaprint(f"[cloth_in_container] no points from {cloth_prim_path}")
            continue
        limits = _container_local_aabb_limits(
            container,
            env_idx,
            use_container_aabb=bool(use_container_aabb),
            aabb_scale=float(aabb_scale),
            xy_half_extents=xy_half_extents,
            z_min_offset=float(z_min_offset),
            z_max_offset=float(z_max_offset),
        )
        if limits is None:
            if do_debug:
                manaprint("[cloth_in_container] failed to build container local AABB")
            continue
        bmin, bmax = limits
        cpos = container.data.root_pos_w.torch[env_idx]
        cquat = container.data.root_quat_w.torch[env_idx]
        local = _world_points_to_container_local(pts, cpos.detach().cpu().numpy(), cquat.detach().cpu().numpy())
        inside = _points_inside_local_aabb(local, bmin, bmax)
        frac = float(inside.sum()) / float(len(local))
        result[env_idx] = frac >= need
        if do_debug:
            import numpy as np

            com = np.asarray(local, dtype=np.float32).mean(axis=0)
            manaprint(
                f"[cloth_in_container] src={source} prim={cloth_prim_path} "
                f"n={len(local)} frac={frac:.3f} need={need:.3f} "
                f"com_local={com.tolist()} aabb=[{bmin.tolist()} .. {bmax.tolist()}] "
                f"ok={bool(result[env_idx])}"
            )

    if consecutive_steps <= 1:
        return result
    counter_attr = "_cloth_in_container_" + _stable_term_counter_id(
        cloth_name,
        container_name,
        f"{hx:.3f}",
        f"{hy:.3f}",
        f"{need:.3f}",
        f"aabb{int(bool(use_container_aabb))}",
        f"{float(aabb_scale):.3f}",
    )
    return _apply_consecutive_filter(env, result, consecutive_steps, counter_attr=counter_attr)


def cloth_mass_center_in_container(
    env: ManagerBasedRLEnv,
    *,
    cloth_scene_key: str,
    container_cfg: SceneEntityCfg | str,
    cloth_prim_relpath: str | None = None,
    xy_half_extents: list[float] | tuple[float, float] = (0.14, 0.14),
    z_min_offset: float = -0.05,
    z_max_offset: float = 0.25,
    consecutive_steps: int = 5,
    use_container_aabb: bool = True,
    aabb_scale: float = 0.9,
    evaluation_runtime: Any | None = None,
    score_runtime: Any | None = None,
) -> torch.Tensor:
    """True when the cloth mass center lies inside the container OBB."""
    del evaluation_runtime, score_runtime
    import numpy as np

    num_envs = env.num_envs
    result = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    cloth_name = _resolve_name(cloth_scene_key) if not isinstance(cloth_scene_key, str) else cloth_scene_key
    cloth_name = cloth_name.lstrip("@")
    container_name = _resolve_name(collapse_single_item(container_cfg))

    try:
        container = env.scene[container_name]
    except Exception:
        return result

    hx = float(xy_half_extents[0])
    hy = float(xy_half_extents[1])
    debug_attr = "_mana_cloth_com_in_container_debug_tick"
    tick = int(getattr(env, debug_attr, 0)) + 1
    setattr(env, debug_attr, tick)
    do_debug = tick % 60 == 1

    for env_idx in range(num_envs):
        env_ns = f"/World/envs/env_{env_idx}"
        cloth_prim_path = _resolve_cloth_prim_path(env_ns, cloth_name, cloth_prim_relpath)
        if cloth_prim_path is None:
            continue
        pts, source = _read_cloth_points_world_with_source(cloth_prim_path)
        if pts is None or len(pts) == 0:
            continue
        limits = _container_local_aabb_limits(
            container,
            env_idx,
            use_container_aabb=bool(use_container_aabb),
            aabb_scale=float(aabb_scale),
            xy_half_extents=xy_half_extents,
            z_min_offset=float(z_min_offset),
            z_max_offset=float(z_max_offset),
        )
        if limits is None:
            continue
        bmin, bmax = limits
        com_w = np.asarray(pts, dtype=np.float32).mean(axis=0, keepdims=True)
        cpos = container.data.root_pos_w.torch[env_idx]
        cquat = container.data.root_quat_w.torch[env_idx]
        local = _world_points_to_container_local(com_w, cpos.detach().cpu().numpy(), cquat.detach().cpu().numpy())
        ok = bool(_points_inside_local_aabb(local, bmin, bmax)[0])
        result[env_idx] = ok
        if do_debug:
            manaprint(
                f"[cloth_com_in_container] src={source} prim={cloth_prim_path} "
                f"com_local={local.reshape(3).tolist()} "
                f"aabb=[{bmin.tolist()} .. {bmax.tolist()}] ok={ok}"
            )

    if consecutive_steps <= 1:
        return result
    counter_attr = "_cloth_mass_center_in_container_" + _stable_term_counter_id(
        cloth_name,
        container_name,
        f"{hx:.3f}",
        f"{hy:.3f}",
        f"aabb{int(bool(use_container_aabb))}",
        f"{float(aabb_scale):.3f}",
    )
    return _apply_consecutive_filter(env, result, consecutive_steps, counter_attr=counter_attr)


def plug_in_charger_active_label(
    env: ManagerBasedRLEnv, state_attr: str = "_mana_plug_in_charger_state"
) -> list[str | None]:
    """Return the reset-sampled per-env USB port / face label for logging/overlay."""
    state = getattr(env, state_attr, None)
    plans = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plans, dict):
        return [None] * int(env.num_envs)
    return [plans.get(env_id) for env_id in range(int(env.num_envs))]


def usb_inserted_at_target_port(
    env: ManagerBasedRLEnv,
    *,
    object_cfg: SceneEntityCfg | str,
    base_cfg: SceneEntityCfg | str,
    port_names: list[str],
    port_log_key: str = "target_port",
    object_axis: list[float] | None = None,
    insertion_axis: list[float] | None = None,
    axial_threshold: float = 0.03,
    lateral_threshold: float = 0.01,
    orientation_dot_min: float = 0.85,
    face_dot_min: float = 0.85,
    face_axis: list[float] | None = None,
    face_world_axis: list[float] | None = None,
    base_max_displacement: float = 0.01,
    velocity_threshold: float = 0.5,
    consecutive_steps: int = 5,
    state_attr: str = "_mana_plug_in_charger_state",
) -> torch.Tensor:
    """[SKILL insert] USB seated in the sampled hub port with correct face; hub unmoved.

    Reads ``port_log_key`` (int ``[num_envs, 1]``) to select among ``port_names``,
    then requires:

    1. Axial / lateral insertion vs that port marker (same geometry as
       :func:`object_inserted`, but orientation uses a **signed** dot so the plug
       tip must point into the port).
    2. Signed face alignment: object ``face_axis`` (default local +Z) vs
       ``face_world_axis`` (default world +Z) above ``face_dot_min``.
    3. Hub root displacement from the reset snapshot below ``base_max_displacement``.
    """

    names = [str(n) for n in port_names]
    if not names:
        raise ValueError("usb_inserted_at_target_port requires non-empty port_names")

    idx_log = read_log(env, str(port_log_key))
    if idx_log is None:
        return _false_tensor(env)
    idx_log = idx_log.to(device=env.device, dtype=torch.int64).reshape(env.num_envs, -1)[:, 0]
    idx_log = idx_log.clamp(0, len(names) - 1)

    if object_axis is None:
        object_axis = [-1.0, 0.0, 0.0]
    if insertion_axis is None:
        insertion_axis = [0.0, 0.0, 1.0]
    if face_axis is None:
        face_axis = [0.0, 0.0, 1.0]
    if face_world_axis is None:
        face_world_axis = [0.0, 0.0, 1.0]

    obj: RigidObject = env.scene[_resolve_name(collapse_single_item(object_cfg))]
    base: RigidObject = env.scene[_resolve_name(collapse_single_item(base_cfg))]

    obj_axis_t = torch.tensor(object_axis, dtype=torch.float32, device=env.device)
    obj_axis_t = obj_axis_t / torch.norm(obj_axis_t)
    insert_axis_t = torch.tensor(insertion_axis, dtype=torch.float32, device=env.device)
    insert_axis_t = insert_axis_t / torch.norm(insert_axis_t)
    face_axis_t = torch.tensor(face_axis, dtype=torch.float32, device=env.device)
    face_axis_t = face_axis_t / torch.norm(face_axis_t)
    face_world_t = torch.tensor(face_world_axis, dtype=torch.float32, device=env.device)
    face_world_t = face_world_t / torch.norm(face_world_t)

    from isaaclab.utils import math as math_utils

    obj_pos = obj.data.root_pos_w.torch - env.scene.env_origins
    obj_quat = obj.data.root_quat_w.torch
    obj_rotation = math_utils.matrix_from_quat(obj_quat)
    obj_axis_w = torch.matmul(obj_rotation, obj_axis_t)
    obj_face_w = torch.matmul(obj_rotation, face_axis_t)
    face_ok = (obj_face_w * face_world_t.unsqueeze(0)).sum(dim=-1) >= float(face_dot_min)
    vel_ok = torch.norm(obj.data.root_lin_vel_w.torch, dim=-1) < float(velocity_threshold)

    per_port = []
    for name in names:
        tgt: RigidObject = env.scene[name]
        tgt_pos = tgt.data.root_pos_w.torch - env.scene.env_origins
        tgt_quat = tgt.data.root_quat_w.torch
        tgt_rotation = math_utils.matrix_from_quat(tgt_quat)
        diff_w = obj_pos - tgt_pos
        local_diff = torch.bmm(tgt_rotation.transpose(1, 2), diff_w.unsqueeze(-1)).squeeze(-1)
        axial_dist = (local_diff * insert_axis_t.unsqueeze(0)).sum(dim=-1).abs()
        axial_vec = (local_diff * insert_axis_t.unsqueeze(0)).sum(dim=-1, keepdim=True) * insert_axis_t.unsqueeze(0)
        lateral = torch.norm(local_diff - axial_vec, dim=-1)
        tgt_insert_w = torch.matmul(tgt_rotation, insert_axis_t)
        # Signed: plug tip must point along the port insertion axis.
        orient_ok = (obj_axis_w * tgt_insert_w).sum(dim=-1) >= float(orientation_dot_min)
        per_port.append((axial_dist < float(axial_threshold)) & (lateral < float(lateral_threshold)) & orient_ok)

    stacked = torch.stack(per_port, dim=0)
    env_arange = torch.arange(env.num_envs, device=env.device)
    insert_ok = stacked[idx_log, env_arange]

    base_pos = base.data.root_pos_w.torch - env.scene.env_origins
    state = getattr(env, state_attr, None)
    ref = state.get("base_pos") if isinstance(state, dict) else None
    if isinstance(ref, torch.Tensor) and ref.shape == base_pos.shape:
        base_ok = torch.norm(base_pos - ref.to(device=env.device), dim=-1) <= float(base_max_displacement)
    else:
        # Fallback: compare against default spawn pose when snapshot is missing.
        default = getattr(base.data, "default_root_state", None)
        default_tensor = getattr(default, "torch", default)
        if default_tensor is not None and default_tensor.numel() > 0:
            base_ok = torch.norm(base_pos - default_tensor[:, :3], dim=-1) <= float(base_max_displacement)
        else:
            base_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    raw = insert_ok & face_ok & vel_ok & base_ok
    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_usb_inserted_" + _stable_term_counter_id(
        *names, str(port_log_key), str(axial_threshold), str(lateral_threshold)
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def match_shape_to_slot_active_label(
    env: ManagerBasedRLEnv, state_attr: str = "_mana_match_shape_to_slot_state"
) -> list[str | None]:
    """Return the reset-sampled per-env shape/color plan for logging/overlay."""
    state = getattr(env, state_attr, None)
    plans = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plans, dict):
        return [None] * int(env.num_envs)
    return [plans.get(env_id) for env_id in range(int(env.num_envs))]


def shape_piece_in_target_slot(
    env: ManagerBasedRLEnv,
    *,
    slot_cfg: SceneEntityCfg | str,
    base_cfg: SceneEntityCfg | str,
    object_cfg: SceneEntityCfg | str | None = None,
    object_names: list[str] | None = None,
    object_log_key: str | None = None,
    object_axis: list[float] | None = None,
    insertion_axis: list[float] | None = None,
    axial_threshold: float = 0.04,
    lateral_threshold: float = 0.025,
    orientation_dot_min: float = 0.75,
    base_max_displacement: float = 0.015,
    velocity_threshold: float = 0.5,
    consecutive_steps: int = 5,
    state_attr: str = "_mana_match_shape_to_slot_state",
) -> torch.Tensor:
    """[SKILL insert] Shape piece seated in its slot marker; sorter base unmoved.

    Provide either a single ``object_cfg``, or ``object_names`` + ``object_log_key``
    (int ``[num_envs, 1]`` indexing into ``object_names``, for yaw-randomized
    left/right targets). Geometry matches :func:`object_inserted`. Also requires
    sorter ``base_cfg`` root displacement from the reset snapshot below
    ``base_max_displacement``.
    """

    if object_axis is None:
        object_axis = [0.0, 0.0, 1.0]
    if insertion_axis is None:
        insertion_axis = [0.0, 0.0, 1.0]

    names = [str(n) for n in (object_names or [])]
    if names and object_log_key:
        idx_log = read_log(env, str(object_log_key))
        if idx_log is None:
            return _false_tensor(env)
        idx_log = idx_log.to(device=env.device, dtype=torch.int64).reshape(env.num_envs, -1)[:, 0]
        idx_log = idx_log.clamp(0, len(names) - 1)
        per_object = [
            object_inserted(
                env,
                object_cfg=name,  # type: ignore[arg-type]
                target_cfg=slot_cfg,  # type: ignore[arg-type]
                insertion_axis=insertion_axis,
                object_axis=object_axis,
                axial_threshold=axial_threshold,
                lateral_threshold=lateral_threshold,
                orientation_dot_min=orientation_dot_min,
                velocity_threshold=velocity_threshold,
                consecutive_steps=1,
            )
            for name in names
        ]
        stacked = torch.stack(per_object, dim=0)
        env_arange = torch.arange(env.num_envs, device=env.device)
        insert_ok = stacked[idx_log, env_arange]
        counter_parts = (*names, str(object_log_key))
    elif object_cfg is not None:
        insert_ok = object_inserted(
            env,
            object_cfg=object_cfg,  # type: ignore[arg-type]
            target_cfg=slot_cfg,  # type: ignore[arg-type]
            insertion_axis=insertion_axis,
            object_axis=object_axis,
            axial_threshold=axial_threshold,
            lateral_threshold=lateral_threshold,
            orientation_dot_min=orientation_dot_min,
            velocity_threshold=velocity_threshold,
            consecutive_steps=1,
        )
        counter_parts = (_resolve_name(collapse_single_item(object_cfg)),)
    else:
        raise ValueError("shape_piece_in_target_slot requires object_cfg, or object_names + object_log_key")

    base: RigidObject = env.scene[_resolve_name(collapse_single_item(base_cfg))]
    base_pos = base.data.root_pos_w.torch - env.scene.env_origins
    state = getattr(env, state_attr, None)
    ref = state.get("base_pos") if isinstance(state, dict) else None
    if isinstance(ref, torch.Tensor) and ref.shape == base_pos.shape:
        base_ok = torch.norm(base_pos - ref.to(device=env.device), dim=-1) <= float(base_max_displacement)
    else:
        default = getattr(base.data, "default_root_state", None)
        default_tensor = getattr(default, "torch", default)
        if default_tensor is not None and default_tensor.numel() > 0:
            base_ok = torch.norm(base_pos - default_tensor[:, :3], dim=-1) <= float(base_max_displacement)
        else:
            base_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    raw = insert_ok & base_ok
    if consecutive_steps <= 1:
        return raw
    counter_attr = "_mana_shape_slot_" + _stable_term_counter_id(
        *counter_parts,
        _resolve_name(collapse_single_item(slot_cfg)),
        str(axial_threshold),
        str(lateral_threshold),
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=counter_attr)


def base_in_operation_zone(
    env: ManagerBasedRLEnv,
    *,
    robot_cfg: SceneEntityCfg | str = "robot",
    zone_key: str,
    context_namespace: str = "mobile_manip",
    context_plan_id: str,
    linear_speed_max: float = 0.03,
    angular_speed_max: float = 0.10,
    consecutive_steps: int = 1,
    counter_attr: str | None = None,
) -> torch.Tensor:
    """[SKILL navigation] Mobile base inside a compiled operation zone with speed gates.
    Assumes: EX001 root_pos_w/quat_w(XYZW)/lin_vel_b/ang_vel_b; zone bounds from runtime_context (no YAML IO).

    Fail-closed when context, plan, or zone bounds are missing or invalid.
    """

    from manaenv.runtime_context import get_runtime_context

    robot_name = _resolve_name(robot_cfg)
    robot = env.scene[robot_name]
    data = robot.data

    pos_w = as_torch(data.root_pos_w)
    origins = getattr(env.scene, "env_origins", None)
    if isinstance(origins, torch.Tensor):
        pos_xy = pos_w[:, :2] - origins[:, :2]
    else:
        pos_xy = pos_w[:, :2]

    quat = as_torch(data.root_quat_w)
    # Isaac Lab runtime root poses use XYZW at the simulator boundary.
    x, y, z, w = quat.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    lin_vel = as_torch(data.root_lin_vel_b)
    ang_vel = as_torch(data.root_ang_vel_b)
    lin_speed = torch.linalg.vector_norm(lin_vel[:, :2], dim=-1)
    ang_speed = ang_vel[:, 2].abs()

    namespace = str(context_namespace or "mobile_manip")
    plan_id = str(context_plan_id or "").strip()
    if not plan_id:
        raise RuntimeError("base_in_operation_zone requires context_plan_id")
    zone_name = str(zone_key or "").strip()
    if not zone_name:
        raise RuntimeError("base_in_operation_zone requires zone_key")

    inside = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for env_id in range(int(env.num_envs)):
        ctx = get_runtime_context(env, namespace=namespace, plan_id=plan_id, env_id=env_id)
        if not isinstance(ctx, dict) or not ctx:
            raise RuntimeError(
                f"base_in_operation_zone: missing runtime_context "
                f"namespace={namespace!r} plan_id={plan_id!r} env_id={env_id}"
            )
        zones = ctx.get("zones")
        if not isinstance(zones, dict) or zone_name not in zones:
            raise RuntimeError(
                f"base_in_operation_zone: zone_key {zone_name!r} missing under "
                f"runtime_context[{namespace!r}][{plan_id!r}][{env_id}].zones"
            )
        bounds = zones[zone_name]
        if not isinstance(bounds, dict):
            raise RuntimeError(f"base_in_operation_zone: invalid bounds for {zone_name!r}")
        shape = str(bounds.get("shape") or "aabb").strip().lower()

        px = float(pos_xy[env_id, 0].item())
        py = float(pos_xy[env_id, 1].item())
        speed_ok = (float(lin_speed[env_id].item()) <= float(linear_speed_max)) and (
            float(ang_speed[env_id].item()) <= float(angular_speed_max)
        )
        if shape == "circle":
            center_role = str(bounds.get("center_role") or "").strip()
            try:
                radius = float(bounds["radius_m"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"base_in_operation_zone: invalid circle bounds for {zone_name!r}: {exc}") from exc
            if not center_role or radius < 0:
                raise RuntimeError(f"base_in_operation_zone: invalid circle bounds for {zone_name!r}")
            center_xy = _live_scene_entity_xy_env_local(env, center_role, env_id=env_id)
            if center_xy is None:
                raise RuntimeError(
                    f"base_in_operation_zone: circle center_role {center_role!r} unavailable for env_id={env_id}"
                )
            inside[env_id] = bool(math.hypot(px - center_xy[0], py - center_xy[1]) <= radius and speed_ok)
            continue
        if shape != "aabb":
            raise RuntimeError(f"base_in_operation_zone: unsupported shape {shape!r} for {zone_name!r}")
        try:
            x_lo, x_hi = float(bounds["x_range"][0]), float(bounds["x_range"][1])
            y_lo, y_hi = float(bounds["y_range"][0]), float(bounds["y_range"][1])
            yaw_lo, yaw_hi = float(bounds["yaw_deg_range"][0]), float(bounds["yaw_deg_range"][1])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(f"base_in_operation_zone: invalid zone bounds for {zone_name!r}: {exc}") from exc
        if not (x_lo <= x_hi and y_lo <= y_hi and yaw_lo <= yaw_hi):
            raise RuntimeError(f"base_in_operation_zone: non-ordered bounds for {zone_name!r}")

        yaw_deg = math.degrees(float(yaw[env_id].item()))
        while yaw_deg > 180.0:
            yaw_deg -= 360.0
        while yaw_deg < -180.0:
            yaw_deg += 360.0
        xy_ok = (x_lo <= px <= x_hi) and (y_lo <= py <= y_hi)
        yaw_ok = yaw_lo <= yaw_deg <= yaw_hi
        inside[env_id] = bool(xy_ok and yaw_ok and speed_ok)

    if consecutive_steps <= 1:
        return inside
    resolved_counter = counter_attr or (
        "_mana_base_in_operation_zone_" + _stable_term_counter_id(robot_name, zone_name, namespace, plan_id)
    )
    return _apply_consecutive_filter(env, inside, consecutive_steps, counter_attr=resolved_counter)


def _live_scene_entity_xy_env_local(
    env: ManagerBasedRLEnv,
    entity_name: str,
    *,
    env_id: int,
) -> tuple[float, float] | None:
    """Read one active scene entity position in the environment-local frame."""
    active_name = _resolve_live_scene_entity_name(env, entity_name, env_id=int(env_id))
    if active_name is None:
        return None
    asset = env.scene[active_name]
    data = getattr(asset, "data", None)
    root_pos = getattr(data, "root_pos_w", None) if data is not None else None
    position = as_torch(root_pos)
    if not isinstance(position, torch.Tensor) or position.ndim == 0:
        return None
    world_xy = position[:2] if position.ndim == 1 else position[int(env_id), :2]
    origins = getattr(env.scene, "env_origins", None)
    if isinstance(origins, torch.Tensor):
        world_xy = world_xy - as_torch(origins)[int(env_id), :2]
    return float(world_xy[0].item()), float(world_xy[1].item())


def _yaw_from_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Return batched yaw from IsaacLab's XYZW root quaternion convention."""
    x, y, z, w = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _head_yaw_rad(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg | str, *, env_id: int) -> float:
    """Read the EX001 head yaw joint for one environment."""
    joint_pos = _resolve_joint_position(env.scene[_resolve_name(robot_cfg)], "head_yaw_joint")
    if joint_pos is None:
        return 0.0
    if joint_pos.ndim == 0:
        return float(joint_pos.item())
    return float(joint_pos[int(env_id)].item())


def target_in_head_hfov(
    env: ManagerBasedRLEnv,
    *,
    robot_cfg: SceneEntityCfg | str = "robot",
    target_cfgs: SceneEntityCfg | str | list[SceneEntityCfg | str] | None = None,
    target_role: SceneEntityCfg | str | None = None,
    half_fov_deg: float = 41.6,
    margin_deg: float = 3.0,
    consecutive_steps: int = 3,
    counter_attr: str | None = None,
) -> torch.Tensor:
    """Return whether one active semantic target is inside the EX001 head HFOV."""
    from manaenv.terms.head_hfov import target_in_horizontal_fov

    if target_cfgs is not None and target_role is not None:
        raise RuntimeError("target_in_head_hfov accepts either target_cfgs or target_role, not both")
    raw_targets = target_cfgs if target_cfgs is not None else target_role
    if raw_targets is None:
        raise RuntimeError("target_in_head_hfov requires target_cfgs")
    if not isinstance(raw_targets, (list, tuple)):
        raw_targets = [raw_targets]
    target_names = [_resolve_name(collapse_single_item(item)) for item in raw_targets]
    target_names = [name for name in dict.fromkeys(target_names) if name]
    if not target_names:
        raise RuntimeError("target_in_head_hfov requires at least one target")

    robot_pos, robot_quat = _get_root_pose_env(env, robot_cfg)
    robot_yaw = _yaw_from_quat_xyzw(robot_quat)
    raw = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for env_id in range(int(env.num_envs)):
        for target_name in target_names:
            target_xy = _live_scene_entity_xy_env_local(env, target_name, env_id=env_id)
            if target_xy is None:
                continue
            if target_in_horizontal_fov(
                (float(robot_pos[env_id, 0].item()), float(robot_pos[env_id, 1].item())),
                float(robot_yaw[env_id].item()),
                _head_yaw_rad(env, robot_cfg, env_id=env_id),
                target_xy,
                half_fov_deg=float(half_fov_deg),
                margin_deg=float(margin_deg),
            ):
                raw[env_id] = True
                break

    if consecutive_steps <= 1:
        return raw
    resolved_counter = counter_attr or (
        "_mana_target_in_head_hfov_" + _stable_term_counter_id(_resolve_name(robot_cfg), *target_names)
    )
    return _apply_consecutive_filter(env, raw, consecutive_steps, counter_attr=resolved_counter)
