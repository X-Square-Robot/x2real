"""Evaluation-only rollout loop for ManaEnv.

A policy planner drives an assembled environment until a fixed number of
episodes has completed. Task-owned hooks produce scores, and the optional
public recorder produces episode MP4s without exposing dataset, replay, or
teleoperation paths.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from manaenv.environments.physx_async_env import register_physics_step_probe
from manaenv.pipeline.planners.planner_base import PlannerBase
from manaenv.pipeline.runtime.env_runner import (
    ENV_DONES,
    GYM_READY,
    INITIAL_OBS,
    SESSION_END,
    SHOULD_STOP,
    GymEnvRunner,
)
from manaenv.policy.action_mode_selection import apply_embodiment_action_config
from manaenv.terms.terminations import reset_stateful_termination_terms
from manaenv.utils.camera_utils import iter_camera_cfg_items, reapply_camera_offsets_to_usd

from .episode_rebuild import register_active_only_episode_rebuild


@dataclass(frozen=True)
class EvaluationLoopResult:
    """Result returned by one policy-evaluation session."""

    num_episodes: int
    num_frames: int
    success_count: int
    error: str | None = None
    runtime_artifacts: dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedEvaluationLoop:
    """Deferred loop handle used by :class:`RolloutFactory`."""

    execute: Callable[[], EvaluationLoopResult]


def _normalize_env_ids(env_ids: Any) -> tuple[int, ...] | None:
    if env_ids is None:
        return None
    if hasattr(env_ids, "detach"):
        env_ids = env_ids.detach().cpu()
    if hasattr(env_ids, "tolist"):
        env_ids = env_ids.tolist()
    if isinstance(env_ids, int):
        return (int(env_ids),)
    return tuple(int(env_id) for env_id in env_ids)


def refresh_configured_cameras(gym_env: Any, env_ids: Any = None) -> None:
    """Reapply camera offsets after reset and refresh rendered observations."""

    assembled = getattr(gym_env, "_mana_assembled_environment", None)
    camera_bundle = getattr(getattr(assembled, "embodiment", None), "camera_config", None)
    if camera_bundle is None:
        return

    normalized_env_ids = _normalize_env_ids(env_ids)
    refreshed = reapply_camera_offsets_to_usd(camera_bundle, env=gym_env, env_ids=normalized_env_ids) > 0
    sensors = getattr(getattr(gym_env, "scene", None), "sensors", {})
    for name, _ in iter_camera_cfg_items(camera_bundle):
        if name in sensors:
            sensors[name].reset()

    sim = getattr(gym_env, "sim", None)
    if not refreshed or not bool(getattr(sim, "is_rendering", False)):
        return
    cfg = getattr(gym_env, "cfg", None)
    warmup_frames = max(int(getattr(cfg, "num_rerenders_on_reset", 1) or 1), 1)
    for _ in range(warmup_frames):
        sim.render()


def compute_post_reset_obs(gym_env: Any, env_ids: Any = None) -> Any:
    """Refresh reset-dependent state and compute policy observations."""

    refresh_configured_cameras(gym_env, env_ids=env_ids)
    try:
        from manaenv.assets.robots.components.embodiment_obs import refresh_scene_and_store_home_ee_refs

        refresh_scene_and_store_home_ee_refs(getattr(gym_env, "unwrapped", gym_env), env_ids=env_ids)
    except ImportError:
        pass
    return gym_env.observation_manager.compute()


def _reapply_embodiment_gripper_limits(embodiment: Any) -> None:
    apply_limits = getattr(embodiment, "_apply_gripper_limits", None)
    action_cfg = getattr(embodiment, "action_config", None)
    if callable(apply_limits) and action_cfg is not None:
        apply_limits(action_cfg)


def apply_planner_requested_action_mode(environment: Any, planner: Any) -> None:
    """Apply the policy's declared action contract before Gym construction."""

    embodiment = getattr(environment, "embodiment", None)
    if embodiment is None:
        return

    action_mode = None
    requested = getattr(planner, "requested_action_mode", None)
    if callable(requested):
        action_mode = requested()
    if not action_mode:
        policy_cfg = getattr(getattr(planner, "cfg", None), "policy_cfg", None)
        policy_embodiment = getattr(policy_cfg, "embodiment", None)
        action_mode = (getattr(policy_cfg, "action_mode", "") or "").strip() or (
            getattr(policy_embodiment, "action_mode", "") or ""
        ).strip()
    if not action_mode:
        return

    if action_mode in {
        "joint_position",
        "ik_eef_delta",
        "ik_eef_abs",
        "ik_eef_abs_wholebody",
    }:
        apply_embodiment_action_config(
            embodiment=embodiment,
            action_mode=action_mode,
            context="Evaluation",
        )
        _reapply_embodiment_gripper_limits(embodiment)
        return

    if action_mode == "bridge_joint":
        bridge_cfg = getattr(planner, "bridge_action_config", None)
        if not callable(bridge_cfg):
            raise RuntimeError("Policy requested bridge_joint but exposes no bridge_action_config().")
        embodiment.action_config = bridge_cfg()
        _reapply_embodiment_gripper_limits(embodiment)
        return

    if action_mode in {"pink_ik", "x2robot_ik"}:
        use_x2robot = getattr(embodiment, "use_x2robot_ik_action_config", None)
        if callable(use_x2robot):
            use_x2robot()
            return
        cfg_type = getattr(embodiment, "X2RobotIKActionsCfg", None) or getattr(
            embodiment, "PinkIKActionsCfg", None
        )
        if cfg_type is None:
            raise RuntimeError(
                f"Embodiment {type(embodiment).__name__} has no X2Robot/Pink IK action config."
            )
        embodiment.action_config = cfg_type()
        _reapply_embodiment_gripper_limits(embodiment)
        return

    raise ValueError(f"Unsupported evaluation action_mode={action_mode!r}.")


def _register_episode_budget_hooks(runner: GymEnvRunner, state: Any, target: int) -> None:
    """Admit the same final completion prefix to video recording and scoring."""

    restore_recorder: Callable[[], None] | None = None

    def remaining() -> int:
        return max(0, target - int(state.num_episodes))

    def install_recorder_limit(ctx: Any) -> None:
        nonlocal restore_recorder
        if restore_recorder is not None:
            restore_recorder()
            restore_recorder = None
        recorder = getattr(ctx.gym_env, "recorder_manager", None)
        original = getattr(recorder, "record_pre_reset", None)
        if not callable(original):
            return

        def record_pre_reset(env_ids=None, force_export_or_skip=None):
            # RGB videos finalize in recorder terms even with dataset export
            # disabled, so every pre-reset call must respect the episode budget.
            ids = _normalize_env_ids(env_ids)
            if ids is None:
                ids = tuple(range(ctx.gym_env.num_envs))
            admitted = sorted(ids)[:remaining()]
            if admitted:
                return original(admitted, force_export_or_skip=force_export_or_skip)
            return None

        def restore() -> None:
            if recorder.record_pre_reset is record_pre_reset:
                recorder.record_pre_reset = original

        recorder.record_pre_reset = record_pre_reset
        restore_recorder = restore

    def limit_completions(_ctx: Any, done: Any, *_args: Any) -> None:
        env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        done[env_ids[remaining():]] = False

    def restore_on_end(_ctx: Any) -> None:
        nonlocal restore_recorder
        if restore_recorder is not None:
            restore_recorder()
            restore_recorder = None

    # Isaac finalizes videos inside step(), before ENV_DONES. Apply the same
    # ascending-env selection before counters (priority 0) and scoring (15).
    runner.add_hook(GYM_READY, install_recorder_limit, priority=30)
    runner.add_hook(ENV_DONES, limit_completions, priority=-10)
    runner.add_hook(SESSION_END, restore_on_end, priority=-10)


def prepare_evaluation_loop(
    *,
    environment: Any,
    planner: PlannerBase,
    num_episodes: int,
    num_envs: int = 1,
    output_dir: str,
    session_support: Any | None = None,
    extra_hook_registrars: Iterable[Callable[[GymEnvRunner], None]] = (),
    max_steps_resolver: Callable[[Any], int | None] | None = None,
    external_gym_env: Any = None,
    benchmark_video_output_dir: str | None = None,
    benchmark_video_fps: float = 20.0,
    benchmark_video_camera_order: tuple[str, ...] = tuple(),
) -> PreparedEvaluationLoop:
    """Prepare a fixed-episode, evaluation-only policy rollout."""

    target = max(int(num_episodes), 1)
    environment.num_envs = max(int(num_envs), 1)
    state = SimpleNamespace(num_episodes=0, num_frames=0, success_count=0)
    runner = GymEnvRunner()

    def on_gym_ready(ctx: Any) -> None:
        state.num_envs = int(getattr(ctx.gym_env, "num_envs", environment.num_envs))

    def initial_obs(ctx: Any) -> Any:
        import torch

        gym_env = ctx.gym_env
        gym_env.reset()
        planner.initial_plan(gym_env)
        env_ids = torch.arange(gym_env.num_envs, dtype=torch.int64, device=gym_env.device)
        obs = compute_post_reset_obs(gym_env, env_ids)
        planner.on_reset(gym_env, obs, env_ids, None)
        return obs

    def on_done(ctx: Any, done: Any, obs: Any, terminated: Any, truncated: Any, actions: Any) -> None:
        del obs, actions
        gym_env = ctx.gym_env
        env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        ids = [int(value) for value in env_ids.tolist()]
        if not ids:
            return

        success_term = None
        manager = getattr(gym_env, "termination_manager", None)
        if manager is not None:
            try:
                success_term = manager.get_term("success")
            except Exception:
                success_term = None
        success_by_env: dict[int, bool] = {}
        for env_id in ids:
            if success_term is not None:
                success = bool(success_term[env_id])
            else:
                success = bool(terminated[env_id]) and not bool(truncated[env_id])
            success_by_env[env_id] = success
            state.success_count += int(success)
        state.num_episodes += len(ids)
        reset_stateful_termination_terms(gym_env, env_ids)
        post_reset_obs = compute_post_reset_obs(gym_env, env_ids)
        ctx.extra["override_obs"] = post_reset_obs
        planner.on_reset(gym_env, post_reset_obs, env_ids, success_by_env)

    def after_step(_ctx: Any, _obs: Any, _terminated: Any, _truncated: Any, _actions: Any) -> None:
        state.num_frames += int(getattr(state, "num_envs", environment.num_envs))

    def should_stop(ctx: Any) -> bool:
        return bool(ctx.stop_requested or state.num_episodes >= target)

    from manaenv.pipeline.runtime.env_runner import AFTER_STEP

    runner.add_hook(GYM_READY, on_gym_ready, priority=0)
    runner.add_hook(INITIAL_OBS, initial_obs, priority=0)
    runner.add_hook(AFTER_STEP, after_step, priority=0)
    runner.add_hook(ENV_DONES, on_done, priority=0)
    runner.add_hook(SHOULD_STOP, should_stop, priority=0)
    _register_episode_budget_hooks(runner, state, target)
    register_physics_step_probe(runner)

    if session_support is None:
        create_support = getattr(environment, "create_benchmark_session_support", None)
        if callable(create_support):
            session_support = create_support()
    if session_support is not None:
        session_support.register_hooks(runner)
        open_report = getattr(session_support, "open_evaluation_report", None)
        if not callable(open_report):
            open_report = getattr(session_support, "open_collection_report", None)
        if callable(open_report):
            open_report(
                output_dir,
                env_name=environment.name,
                prompt=environment.get_prompt() if hasattr(environment, "get_prompt") else "",
                total_episodes=target,
            )

    for registrar in extra_hook_registrars:
        registrar(runner)

    apply_planner_requested_action_mode(environment, planner)
    build_kwargs: dict[str, Any] = {"mode": "evaluate"}
    if benchmark_video_output_dir:
        build_kwargs.update(
            benchmark_video_output_dir=benchmark_video_output_dir,
            benchmark_video_fps=float(benchmark_video_fps),
            benchmark_video_camera_order=tuple(benchmark_video_camera_order),
        )
    register_active_only_episode_rebuild(
        runner,
        environment=environment,
        planner=planner,
        state=state,
        build_gym_kwargs=build_kwargs,
        episode_count=lambda value: int(value.num_episodes),
        target_count=lambda _value: target,
    )

    def execute() -> EvaluationLoopResult:
        if external_gym_env is not None:
            effective_max = max_steps_resolver(external_gym_env) if max_steps_resolver is not None else None
            ctx = runner.run_loop(
                external_gym_env,
                planner,
                max_steps=effective_max,
                use_inference_mode=True,
                close_gym_env=False,
                assembled_environment=environment,
            )
        else:
            ctx = runner.run_with_build(
                environment,
                build_kwargs,
                planner,
                max_steps_resolver=max_steps_resolver,
                use_inference_mode=True,
                create_stage=True,
                close_gym_env=True,
                assembled_environment=environment,
            )
        if ctx is None:
            return EvaluationLoopResult(0, 0, 0, error="Failed to build Gym environment")

        runtime_artifacts: dict[str, Any] = {}
        if session_support is not None:
            collect_payload = getattr(session_support, "collect_result_payload", None)
            if callable(collect_payload):
                runtime_artifacts.update(collect_payload() or {})
            collect_artifacts = getattr(session_support, "collect_evaluation_artifacts", None)
            if callable(collect_artifacts):
                runtime_artifacts.update(
                    collect_artifacts(
                        output_dir,
                        env_name=environment.name,
                        prompt=environment.get_prompt() if hasattr(environment, "get_prompt") else "",
                    )
                    or {}
                )
        return EvaluationLoopResult(
            num_episodes=int(state.num_episodes),
            num_frames=int(state.num_frames),
            success_count=int(state.success_count),
            runtime_artifacts=runtime_artifacts or None,
        )

    return PreparedEvaluationLoop(execute=execute)


__all__ = [
    "EvaluationLoopResult",
    "PreparedEvaluationLoop",
    "apply_planner_requested_action_mode",
    "compute_post_reset_obs",
    "prepare_evaluation_loop",
]
