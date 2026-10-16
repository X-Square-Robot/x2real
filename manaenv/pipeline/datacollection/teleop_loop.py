"""Manual collection hooks adapted to the public GymEnvRunner lifecycle."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import torch

from manaenv.environments.physx_async_env import register_physics_step_probe
from manaenv.pipeline.datacollection.dataset import inspect_dataset, write_collection_metadata
from manaenv.pipeline.datacollection.events import (
    MARK_SUCCESS,
    RESET_EPISODE,
    START_RECORDING,
    STOP_RECORDING,
    drain_collection_events,
)
from manaenv.pipeline.datacollection.viewports import register_collection_viewports
from manaenv.pipeline.evaluation.rollout_loop import compute_post_reset_obs
from manaenv.pipeline.evaluation.episode_rebuild import register_active_only_episode_rebuild
from manaenv.pipeline.runtime.active_instruction import read_active_instructions
from manaenv.pipeline.runtime.env_runner import (
    AFTER_ACTION,
    AFTER_STEP,
    ENV_DONES,
    GYM_READY,
    INITIAL_OBS,
    SESSION_END,
    SHOULD_STOP,
    GymEnvRunner,
)
from manaenv.policy.action_mode_selection import apply_embodiment_action_config
from manaenv.terms.terminations import reset_stateful_termination_terms
from manaenv.utils.debug_print import manaprint


def recorded_action_count(recorder: Any) -> int:
    episode = recorder.get_episode(0)
    count = getattr(episode, "recorded_count", None)
    if callable(count):
        return int(count("actions"))
    actions = getattr(episode, "data", {}).get("actions")
    if isinstance(actions, torch.Tensor) and actions.ndim:
        return int(actions.shape[0])
    return len(actions) if actions is not None else 0


def register_collection_hooks(
    runner: GymEnvRunner,
    *,
    environment: Any,
    planner: Any,
    state: Any,
    target: int,
    output_path: Path,
    on_boundary: Callable[[Any], None] | None = None,
) -> None:
    """Own export/reset events so failed or discarded attempts never become demos."""
    originals: list[tuple[Any, str, Any]] = []

    def input_active() -> bool:
        check = getattr(planner, "is_control_active", None)
        return bool(check()) if callable(check) else True

    def ready(ctx: Any) -> None:
        for recorder, name, original in reversed(originals):
            setattr(recorder, name, original)
        originals.clear()
        if int(ctx.gym_env.num_envs) != 1:
            raise ValueError("Manual collection requires exactly one environment")
        planner._init_device(ctx.gym_env)
        recorder = ctx.gym_env.recorder_manager
        for name in ("record_pre_step", "record_post_step", "record_post_physics_decimation_step"):
            original = getattr(recorder, name)
            originals.append((recorder, name, original))

            def gated(*args: Any, _original=original, **kwargs: Any) -> Any:
                if state.recording and input_active():
                    return _original(*args, **kwargs)
                return None

            setattr(recorder, name, gated)

    def reset(ctx: Any) -> Any:
        gym = ctx.gym_env
        gym.recorder_manager.reset([0])
        gym.reset()
        reset_stateful_termination_terms(gym, [0])
        obs = compute_post_reset_obs(gym, [0])
        planner.on_reset(gym, obs, torch.tensor([0], device=gym.device), None)
        planner.drain_events()
        ctx.extra["override_obs"] = obs
        return obs

    def initial(ctx: Any) -> Any:
        return reset(ctx)

    def export(ctx: Any) -> bool:
        gym = ctx.gym_env
        recorder = gym.recorder_manager
        minimum = max(int(planner.cfg.min_recorded_actions_for_export), 1)
        frames = recorded_action_count(recorder)
        if frames < minimum:
            manaprint(f"INFO: demonstration has {frames} actions; at least {minimum} are required")
            return False
        before = int(recorder.exported_successful_episode_count)
        recorder.record_pre_reset([0], force_export_or_skip=False)
        recorder.set_success_to_episodes([0], torch.tensor([True], device=gym.device))
        recorder.export_episodes([0])
        if int(recorder.exported_successful_episode_count) != before + 1:
            raise RuntimeError("Recorder did not confirm a successful demonstration export")
        active_texts = read_active_instructions(gym)
        instruction = str(active_texts[0]) if active_texts and active_texts[0] else environment.get_prompt()
        metadata = {
            "env_name": environment.name,
            "instruction": instruction,
            "collector": planner.device_name,
            "seed": getattr(gym.cfg, "seed", None),
            "action": {
                "action_mode": planner.requested_action_mode(),
                "action_dim": int(gym.action_manager.action.shape[-1]),
                "quaternion_format": "xyzw",
                "step_dt": float(gym.step_dt),
            },
        }
        current_path = getattr(state, "output_path", output_path)
        write_collection_metadata(current_path, metadata, episode_name=f"demo_{before}")
        state.success_count += 1
        manaprint(f"INFO: saved demonstration {state.success_count}/{target}: {current_path}")
        return True

    def boundary(ctx: Any) -> Any:
        obs = reset(ctx)
        if on_boundary is not None:
            on_boundary(ctx)
        return obs

    def events(ctx: Any, _obs: Any, _actions: Any) -> None:
        queued = drain_collection_events(planner)
        # A reset cancels a success queued in the same input sample.
        names = {event.name for event in queued}
        if RESET_EPISODE in names:
            state.attempts += 1
            boundary(ctx)
            ctx.extra["skip_step"] = True
            return
        if STOP_RECORDING in names:
            state.recording = False
            ctx.gym_env.recorder_manager.reset([0])
        if START_RECORDING in names:
            state.recording = True
            reset(ctx)
            ctx.extra["skip_step"] = True
            return
        if MARK_SUCCESS in names and state.recording:
            if export(ctx):
                state.attempts += 1
                boundary(ctx)
                ctx.extra["skip_step"] = True
                ctx.stop_requested = state.success_count >= target
                return
        # A disconnected master must not command the device's zero fallback.
        controller = getattr(planner, "_master_controller", None)
        waiting_for_master = controller is not None and not controller.has_fresh_input()
        paused_vr = planner.device_name == "vr" and not input_active()
        if waiting_for_master or paused_vr:
            ctx.gym_env.sim.render()
            ctx.extra["allow_indefinite_skip_step"] = True
            ctx.extra["skip_step"] = True

    def after_step(ctx: Any, _obs: Any, terminated: Any, truncated: Any, _actions: Any) -> None:
        state.num_frames += 1
        if bool(terminated[0]) or bool(truncated[0]):
            # Isaac has already reset on timeout/failure. Drop that attempt.
            state.attempts += 1
            boundary(ctx)
            ctx.extra["skip_env_dones"] = True
            ctx.extra["skip_env_dones_recovered"] = True
            return
        term = planner._success_term
        if state.recording and input_active() and term is not None:
            success = bool(torch.as_tensor(term.func(ctx.gym_env, **term.params)).any())
            if success and export(ctx):
                state.attempts += 1
                boundary(ctx)
                ctx.extra["skip_env_dones"] = True
                ctx.extra["skip_env_dones_recovered"] = True

    def done(_ctx: Any, *_args: Any) -> None:
        return None

    def end(_ctx: Any) -> None:
        for recorder, name, original in reversed(originals):
            setattr(recorder, name, original)
        originals.clear()

    runner.add_hook(GYM_READY, ready, priority=0)
    runner.add_hook(INITIAL_OBS, initial, priority=0)
    runner.add_hook(AFTER_ACTION, events, priority=0)
    runner.add_hook(AFTER_STEP, after_step, priority=0)
    runner.add_hook(ENV_DONES, done, priority=0)
    runner.add_hook(SHOULD_STOP, lambda ctx: ctx.stop_requested or state.success_count >= target, priority=0)
    runner.add_hook(SESSION_END, end, priority=0)


def run_manual_collection(
    environment: Any,
    planner: Any,
    *,
    output_dir: str,
    num_demos: int,
    external_gym_env: Any = None,
) -> dict[str, Any]:
    """Run a single manual session and verify its persisted HDF5 artifact."""
    if external_gym_env is not None:
        raise ValueError("Manual collection must build its own recording environment")
    target = int(num_demos)
    if target <= 0:
        raise ValueError("num_demos must be positive")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{environment.name}_{planner.device_name}_{uuid4().hex[:12]}.hdf5"
    environment.num_envs = 1
    apply_embodiment_action_config(
        embodiment=environment.embodiment,
        action_mode=planner.requested_action_mode(),
        context="ManualCollection",
    )
    apply_limits = getattr(environment.embodiment, "_apply_gripper_limits", None)
    if callable(apply_limits):
        apply_limits(environment.embodiment.action_config)
    state = SimpleNamespace(
        success_count=0,
        attempts=0,
        num_frames=0,
        recording=True,
        output_path=output_path,
        dataset_paths=[output_path],
    )
    runner = GymEnvRunner()
    build_kwargs = {"mode": "collect", "collection_output_path": str(output_path)}
    requires = getattr(environment, "requires_episode_scene_rebuild", None)
    rebuild_scenes = bool(requires()) if callable(requires) else False

    def next_scene(ctx: Any) -> None:
        if rebuild_scenes and state.success_count < target:
            next_path = destination / f"{output_path.stem}_scene{len(state.dataset_paths)}.hdf5"
            state.output_path = next_path
            state.dataset_paths.append(next_path)
            build_kwargs["collection_output_path"] = str(next_path)
            ctx.extra["episode_rebuild_requested"] = True

    register_collection_hooks(
        runner,
        environment=environment,
        planner=planner,
        state=state,
        target=target,
        output_path=output_path,
        on_boundary=next_scene,
    )
    register_collection_viewports(runner, environment=environment)
    register_physics_step_probe(runner)
    register_active_only_episode_rebuild(
        runner,
        environment=environment,
        planner=planner,
        state=state,
        build_gym_kwargs=build_kwargs,
        episode_count=lambda value: value.success_count,
        target_count=lambda value: target,
    )
    ctx = runner.run_with_build(
        environment,
        build_kwargs,
        planner,
        create_stage=True,
        use_inference_mode=True,
        close_gym_env=True,
        assembled_environment=environment,
    )
    artifacts = [inspect_dataset(path) for path in state.dataset_paths if path.is_file()]
    saved = sum(sum(int(episode["success"]) for episode in artifact["episodes"]) for artifact in artifacts)
    error = None
    if ctx is None:
        error = "Failed to build the collection environment"
    elif saved != state.success_count or saved < target:
        error = f"Collection ended with {saved}/{target} persisted demonstrations"
    result = {
        "env_name": environment.name,
        "actor": type(planner).__name__,
        "mode": "collection",
        "num_episodes": state.attempts,
        "success_count": saved,
        "num_frames": state.num_frames,
        "datasets": artifacts,
        "error": error,
    }
    report = output_path.with_suffix(".json")
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["collection_report_path"] = str(report)
    return result
