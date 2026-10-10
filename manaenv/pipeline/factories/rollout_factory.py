"""Policy-evaluation factory used by the public ManaEnv runtime."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Any

from manaenv.pipeline.artifacts.base import BaseArtifact
from manaenv.pipeline.artifacts.evaluation import EvalArtifact, failure_cases_from_result
from manaenv.pipeline.data_types import DataBucket
from manaenv.pipeline.datacenter.export_spec import resolve_benchmark_report_dir
from manaenv.pipeline.evaluation.rollout_loop import prepare_evaluation_loop
from manaenv.pipeline.factories.rollout_factory_cfg import RolloutOrderCFG, RolloutOrdersCFG
from manaenv.pipeline.factory import BaseFactory, BaseFactoryCFG
from manaenv.pipeline.planners.planner_base import PlannerBase, PlannerCFG, create_planner, get_planner_class
from manaenv.pipeline.runtime import runtime_flags
from manaenv.policy.action_mode_selection import clone_embodiment_for_action_mode
from manaenv.policy.mana_policy import ManaPolicyCFG
from manaenv.policy.policy_embodiment import PolicyEmbodimentCFG, bind_embodiment_to_policy_cfg
from manaenv.utils.config_io import create_config_from_dict_strict
from manaenv.utils.debug_print import manaprint
from manaenv.utils.gpu_capacity import resolve_num_parallel_envs


def _apply_episode_step_cap(gym_env: Any, max_steps: int) -> int:
    """Apply a rollout's per-episode step cap to an already-built env."""

    configured_limit = max(int(max_steps), 1)
    env_limit = int(getattr(gym_env, "max_episode_length", configured_limit))
    effective_limit = min(env_limit, configured_limit)
    if effective_limit >= env_limit:
        return env_limit

    cfg = getattr(gym_env, "cfg", None)
    step_dt = float(getattr(gym_env, "step_dt", 0.0) or 0.0)
    if cfg is None or not hasattr(cfg, "episode_length_s") or step_dt <= 0.0:
        manaprint(
            "WARNING: cannot apply rollout max_steps to the built environment; "
            f"using its {env_limit}-step episode length"
        )
        return env_limit
    cfg.episode_length_s = math.nextafter(effective_limit * step_dt, 0.0)
    return min(int(getattr(gym_env, "max_episode_length", effective_limit)), effective_limit)


def _planner_provenance(planner_cfg: Any) -> dict[str, Any]:
    if planner_cfg is None:
        return {}
    policy_cfg = getattr(planner_cfg, "policy_cfg", None)
    out: dict[str, Any] = {"class": type(planner_cfg).__name__.removesuffix("CFG")}
    if policy_cfg is None:
        return out
    remote = getattr(policy_cfg, "remote", None)
    endpoint = getattr(remote, "web_socket_remote", None) if remote is not None else None
    endpoint = endpoint or remote
    out["policy"] = {
        "name": getattr(policy_cfg, "name", None),
        "action_mode": getattr(policy_cfg, "action_mode", None),
        "server_address": getattr(endpoint, "server_address", None),
        "server_port": getattr(endpoint, "server_port", None),
    }
    return out


def _task_provenance(environment: Any) -> dict[str, Any]:
    task = getattr(environment, "task", None)
    if task is None:
        return {}
    out: dict[str, Any] = {}
    try:
        out["prompt"] = str(task.get_prompt() or "")
    except Exception:
        out["prompt"] = ""
    definition = getattr(getattr(task, "cfg", None), "definition", None)
    if isinstance(definition, dict) and definition.get("task_type"):
        out["task_type"] = str(definition["task_type"])
    return out


@dataclass
class RolloutFactoryCFG(BaseFactoryCFG):
    """Configuration for :class:`RolloutFactory`."""

    class_type: Any = None
    type: str = "default_rollout_factory"
    default_num_episodes: int = 10
    default_max_steps: int = 800

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = RolloutFactory


class RolloutFactory(BaseFactory):
    """Evaluate PolicyPlanner orders and write a benchmark report."""

    def __init__(self) -> None:
        super().__init__()
        self.cfg: RolloutFactoryCFG | None = None
        self._orders: list[RolloutOrderCFG] = []
        self._results: list[dict[str, Any]] = []
        self._artifacts: list[EvalArtifact] = []

    def setup(self, cfg: RolloutFactoryCFG | None = None) -> None:
        self.cfg = cfg if cfg is not None else RolloutFactoryCFG()
        self._orders = []
        self._results = []
        self._artifacts = []

    def receive_orders(self, orders: RolloutOrdersCFG) -> None:
        if not isinstance(orders, RolloutOrdersCFG):
            raise TypeError(f"RolloutFactory expected RolloutOrdersCFG, got {type(orders).__name__}")
        self._orders = self._expand_env_refs(orders.orders)

    @staticmethod
    def _expand_env_refs(orders: list[RolloutOrderCFG]) -> list[RolloutOrderCFG]:
        expanded: list[RolloutOrderCFG] = []
        for order in orders:
            refs = order.env_ref if isinstance(order.env_ref, (list, tuple)) else [order.env_ref]
            for ref in refs:
                if not str(ref).strip():
                    continue
                clone = copy.deepcopy(order)
                clone.env_ref = str(ref)
                expanded.append(clone)
        return expanded

    def get_order_count(self) -> int:
        return len(self._orders)

    def _build_planner(self, environment: Any, order: RolloutOrderCFG) -> PlannerBase:
        if not isinstance(order.planner, PlannerCFG):
            raise TypeError("Every public rollout order must declare a typed PolicyPlanner.")
        cfg = copy.deepcopy(order.planner)
        method = type(cfg).__name__.removesuffix("CFG")
        if method != "PolicyPlanner":
            raise ValueError(f"Public rollout supports PolicyPlanner only, got {method!r}.")

        policy_cfg = getattr(cfg, "policy_cfg", None)
        if isinstance(policy_cfg, dict):
            policy_cfg = create_config_from_dict_strict(
                policy_cfg, ManaPolicyCFG, context="rollout.planner.policy_cfg"
            )
            cfg.policy_cfg = policy_cfg
        if policy_cfg is None:
            raise ValueError("PolicyPlanner requires policy_cfg.")
        bind_embodiment_to_policy_cfg(environment=environment, policy_cfg=policy_cfg)
        embodiment_cfg = getattr(policy_cfg, "embodiment", None)
        action_mode = (getattr(policy_cfg, "action_mode", "") or "").strip() or (
            getattr(embodiment_cfg, "action_mode", "") or ""
        ).strip()
        if action_mode and isinstance(embodiment_cfg, PolicyEmbodimentCFG):
            policy_cfg.embodiment = clone_embodiment_for_action_mode(embodiment_cfg, action_mode)

        planner_cls = get_planner_class(method)
        if planner_cls is None:
            raise RuntimeError("PolicyPlanner implementation is unavailable.")
        cfg.class_type = planner_cls
        cfg.metadata.setdefault("method", method)
        return create_planner(cfg)

    def fulfill_orders(
        self,
        data_center: Any = None,
        env_name: str | set[str] | list[str] | tuple[str, ...] | None = None,
        shard: str | None = None,
        external_gym_env: Any = None,
        **_unused: Any,
    ) -> list[dict[str, Any]]:
        self.data_center = data_center or getattr(self, "data_center", None)
        if self.data_center is None:
            raise RuntimeError("RolloutFactory requires a DataCenter.")
        if env_name is None:
            env_name = shard

        selected = self._selected_env_names(env_name)
        environments = list(self.data_center.get(DataBucket.ENVIRONMENTS) or [])
        orders = [order for order in self._orders if selected is None or str(order.env_ref) in selected]
        self._results = []
        self._artifacts = []

        from manaenv.pipeline.runtime.env_runner import _simulation_app_is_running

        for index, order in enumerate(orders, start=1):
            if not _simulation_app_is_running():
                break
            matched = self._resolve_envs(environments, str(order.env_ref))
            if not matched:
                self._results.append(
                    self._failed_order_result(None, order, LookupError(f"environment {order.env_ref!r} not found"))
                )
                continue
            for environment in matched:
                manaprint(f"INFO: evaluation {index}/{len(orders)} start: {environment.name}")
                try:
                    result = self._run_evaluation(
                        environment,
                        order,
                        external_gym_env=external_gym_env,
                    )
                except Exception as exc:
                    manaprint(f"ERROR: evaluation failed for {environment.name}: {type(exc).__name__}: {exc}")
                    result = self._failed_order_result(environment, order, exc)
                self._results.append(result)
                self._artifacts.append(self._build_eval_artifact(result))

        if self._results:
            self.data_center.store_benchmark_results(self._results)
            report_dir = resolve_benchmark_report_dir(self.data_center)
            if report_dir is not None:
                self._write_benchmark_report(self._results, report_dir, self._provenance(report_dir))
        return list(self._results)

    @staticmethod
    def _selected_env_names(value: Any) -> set[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return {value} if value else None
        selected = {str(item) for item in value if str(item)}
        return selected or None

    @staticmethod
    def _resolve_envs(environments: list[Any], env_ref: str) -> list[Any]:
        prefix = f"{env_ref}_v"
        matches = [
            env
            for env in environments
            if str(getattr(env, "name", "")) == env_ref
            or (
                str(getattr(env, "name", "")).startswith(prefix)
                and str(getattr(env, "name", ""))[len(prefix) :].isdigit()
            )
        ]
        return matches

    def _run_evaluation(
        self,
        environment: Any,
        order: RolloutOrderCFG,
        *,
        external_gym_env: Any = None,
    ) -> dict[str, Any]:
        benchmark_cfg = self.data_center.get_order_cfg(DataBucket.BENCHMARKS)
        record_video = bool(getattr(benchmark_cfg, "record_video", False))
        target = max(int(order.num_episodes or self.cfg.default_num_episodes), 1)
        configured_num_envs = resolve_num_parallel_envs(order)
        report_root = resolve_benchmark_report_dir(self.data_center) or os.path.expanduser(
            "~/.manaenv/output/benchmark"
        )
        output_dir = os.path.join(report_root, environment.name)
        os.makedirs(output_dir, exist_ok=True)

        support = None
        resumed_from = 0
        if order.resume:
            from manaenv.pipeline.evaluation.report import load_evaluation_episode_jsonl

            support = environment.create_benchmark_session_support()
            report_path = os.path.join(output_dir, "evaluation_report.jsonl")
            restored_records = load_evaluation_episode_jsonl(
                report_path,
                env_name=environment.name,
            )
            resume_metadata = (order.metadata or {}).get("public_jsonl_resume")
            if isinstance(resume_metadata, dict):
                if "existing_episodes" in resume_metadata:
                    expected = resume_metadata["existing_episodes"]
                    if isinstance(expected, bool) or not isinstance(expected, int):
                        raise TypeError("public_jsonl_resume.existing_episodes must be an integer")
                    if expected != len(restored_records):
                        raise RuntimeError(
                            "evaluation report changed after the resume suite was generated: "
                            f"expected {expected} episodes, found {len(restored_records)}"
                        )
                metadata_target = resume_metadata.get("target_episodes")
                if metadata_target is not None:
                    if isinstance(metadata_target, bool) or not isinstance(metadata_target, int):
                        raise TypeError("public_jsonl_resume.target_episodes must be an integer")
                    if metadata_target != target:
                        raise ValueError(
                            f"public_jsonl_resume.target_episodes={metadata_target!r}; rollout target is {target}"
                        )
                attempt_seed = resume_metadata.get("attempt_seed")
                if attempt_seed is not None:
                    from manaenv.pipeline.runtime import master_seed as master_seed_runtime

                    if isinstance(attempt_seed, bool) or not isinstance(attempt_seed, int):
                        raise TypeError("public_jsonl_resume.attempt_seed must be an integer")
                    runtime_seed = master_seed_runtime.get_master_seed()
                    if runtime_seed != attempt_seed:
                        raise RuntimeError(
                            f"resume suite expects master seed {attempt_seed}; runtime uses {runtime_seed}"
                        )
            support.configure_resume(restored_records, total_episodes=target)
            resumed_from = len(restored_records)

        episodes_remaining = target - resumed_from

        video_output_dir: str | None = None
        video_fps = 20.0
        video_camera_order: tuple[str, ...] = tuple()
        videos_before: set[str] = set()
        if record_video:
            from manaenv.pipeline.evaluation.video_recorder import (
                camera_observation_order,
                list_episode_videos,
                resolve_video_fps,
            )

            video_output_dir = os.path.join(output_dir, "videos", "combined")
            video_fps = resolve_video_fps(environment)
            video_camera_order = camera_observation_order(environment)
            videos_before = set(list_episode_videos(video_output_dir))

        payload: dict[str, Any] = {}
        episodes_this_run = 0
        num_frames = 0
        outcome_success_count = 0
        outcome_error: str | None = None
        if episodes_remaining > 0:
            if record_video:
                from manaenv.pipeline.evaluation.video_recorder import prepare_environment_for_video

                prepare_environment_for_video(environment)

            planner = self._build_planner(environment, order)
            task = environment.task
            if order.episode_length_s is not None and task is not None:
                task.episode_length_s = float(order.episode_length_s)
            if not planner.setup(environment, task):
                raise RuntimeError("PolicyPlanner.setup() returned False.")
            if support is None:
                support = environment.create_benchmark_session_support()

            run_target = episodes_remaining if order.resume else target
            num_envs = 1 if resumed_from > 0 else configured_num_envs
            if resumed_from > 0 and configured_num_envs > 1:
                manaprint(
                    "INFO: resumed evaluation uses one parallel environment so the runtime "
                    f"produces exactly the remaining {run_target} episode(s)"
                )
            registrars = [
                self._instruction_registrar(environment, planner),
                self._overlay_registrar(order, support),
            ]
            if order.enable_hotkeys:
                registrars.append(self._reset_hotkey_registrar(support, planner, run_target))

            per_episode_cap = int(order.max_steps or self.cfg.default_max_steps)
            waves = math.ceil(run_target / max(num_envs, 1))

            def max_steps_resolver(gym_env: Any) -> int:
                return waves * _apply_episode_step_cap(gym_env, per_episode_cap) * 2

            try:
                prepared = prepare_evaluation_loop(
                    environment=environment,
                    planner=planner,
                    num_episodes=run_target,
                    num_envs=num_envs,
                    output_dir=output_dir,
                    session_support=support,
                    extra_hook_registrars=registrars,
                    max_steps_resolver=max_steps_resolver,
                    external_gym_env=external_gym_env,
                    benchmark_video_output_dir=video_output_dir,
                    benchmark_video_fps=video_fps,
                    benchmark_video_camera_order=video_camera_order,
                )
                outcome = prepared.execute()
            finally:
                planner.cleanup()

            payload = outcome.runtime_artifacts or {}
            episodes_this_run = int(outcome.num_episodes)
            num_frames = int(outcome.num_frames)
            outcome_success_count = int(outcome.success_count)
            outcome_error = outcome.error
        else:
            assert support is not None
            payload.update(support.collect_result_payload() or {})
            artifacts = support.collect_evaluation_artifacts(
                output_dir,
                env_name=environment.name,
                prompt=environment.get_prompt() if hasattr(environment, "get_prompt") else "",
            )
            payload.update(artifacts or {})

        evaluation = payload.get("evaluation") or payload.get("evaluation_summary") or {}
        num_episodes = int(evaluation.get("num_episodes", resumed_from + episodes_this_run))
        success_count = int(evaluation.get("success_count", outcome_success_count))
        success_rate = evaluation.get("success_rate")
        if success_rate is None:
            success_rate = success_count / num_episodes if num_episodes else 0.0
        debug_record_paths: list[str] = []
        recording_error: str | None = None
        if record_video and video_output_dir is not None:
            from manaenv.pipeline.evaluation.video_recorder import list_episode_videos

            all_videos = list_episode_videos(video_output_dir)
            debug_record_paths = (
                all_videos if order.resume else [path for path in all_videos if path not in videos_before]
            )
            if num_episodes > 0 and len(debug_record_paths) < num_episodes:
                recording_error = (
                    "benchmark recorder produced "
                    f"{len(debug_record_paths)} MP4 file(s) for {num_episodes} completed episode(s)"
                )
        result: dict[str, Any] = {
            "env_name": environment.name,
            "instruction": environment.get_prompt() if hasattr(environment, "get_prompt") else "",
            "actor": "PolicyPlanner",
            "mode": "evaluation",
            "num_episodes": num_episodes,
            "success_count": success_count,
            "num_frames": num_frames,
            "metrics": {
                "success_rate": round(float(success_rate), 4),
                "avg_score": evaluation.get("avg_score", 0.0),
                "per_stage_pass_rate": evaluation.get("per_stage_pass_rate", {}),
            },
            "evaluation": evaluation,
            "debug_record_paths": debug_record_paths,
            "recording_enabled": record_video,
            "error": outcome_error or recording_error,
        }
        if payload.get("evaluation_report_path"):
            result["evaluation_report_path"] = payload["evaluation_report_path"]
        if order.resume:
            result.update(
                {
                    "resume": True,
                    "resumed_from": resumed_from,
                    "episodes_this_run": episodes_this_run,
                }
            )
        if order.case_id:
            result["case_id"] = order.case_id
        return result

    @staticmethod
    def _set_policy_instruction(policy: Any, instruction: Any) -> None:
        implementation = getattr(policy, "implementation", None)
        target = implementation if implementation is not None else policy
        if hasattr(target, "_instruction"):
            target._instruction = instruction

    def _instruction_registrar(self, environment: Any, planner: PlannerBase):
        fallback = environment.get_prompt() if hasattr(environment, "get_prompt") else ""

        def live_prompt() -> str:
            getter = getattr(environment, "get_prompt", None)
            if callable(getter):
                try:
                    return str(getter() or "")
                except Exception:
                    return str(fallback or "")
            return str(fallback or "")

        def publish(ctx: Any, *_args: Any) -> None:
            policy = getattr(planner, "policy", None)
            if policy is None:
                return
            prompt = live_prompt()
            texts = None
            try:
                from manaenv.pipeline.runtime.active_instruction import read_active_instructions

                texts = read_active_instructions(ctx.gym_env)
            except Exception:
                pass
            if texts:
                self._set_policy_instruction(policy, [str(text or prompt) for text in texts])
            else:
                self._set_policy_instruction(policy, prompt)

        def register(runner: Any) -> None:
            from manaenv.pipeline.runtime.env_runner import BEFORE_STEP, ENV_DONES, LOOP_START, SESSION_START

            runner.add_hook(SESSION_START, publish, priority=20)
            runner.add_hook(LOOP_START, publish, priority=20)
            runner.add_hook(BEFORE_STEP, publish, priority=20)
            runner.add_hook(ENV_DONES, publish, priority=20)

        return register

    @staticmethod
    def _overlay_registrar(order: RolloutOrderCFG, support: Any):
        def register(runner: Any) -> None:
            evaluation_runtime = getattr(getattr(support, "task_runtime_support", None), "evaluation_runtime", None)
            if evaluation_runtime is None:
                return
            task_toggle = evaluation_runtime.overlay_evaluation_enabled
            show = task_toggle if task_toggle is not None else bool(order.show_evaluation_overlay)
            if show or evaluation_runtime.overlay_entries:
                from manaenv.pipeline.evaluation.overlay import attach_evaluation_overlay

                stream_cfg = runtime_flags.get_value("stream_viewports") or {}
                window_cfg = stream_cfg.get("score_window") if isinstance(stream_cfg, dict) else None
                attach_evaluation_overlay(
                    runner,
                    evaluation_runtime,
                    show_evaluation=show,
                    window_config=window_cfg,
                )
            interval = int(order.stage_progress_log_interval or 0)
            if interval > 0:
                from manaenv.pipeline.evaluation.runtime import attach_stage_progress_logger

                attach_stage_progress_logger(runner, evaluation_runtime, interval)

        return register

    @staticmethod
    def _reset_hotkey_registrar(support: Any, planner: PlannerBase, target: int):
        def register(runner: Any) -> None:
            pending = [False]

            def setup(_ctx: Any) -> None:
                try:
                    import carb
                    import omni

                    appwindow = omni.appwindow.get_default_app_window()
                    input_iface = carb.input.acquire_input_interface()
                    keyboard = appwindow.get_keyboard()

                    def on_key(event: Any, *_args: Any) -> bool:
                        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == "R":
                            pending[0] = True
                        return True

                    input_iface.subscribe_to_keyboard_events(keyboard, on_key)
                except Exception:
                    pass

            def reset(ctx: Any, _obs: Any) -> None:
                if not pending[0]:
                    return
                pending[0] = False
                support.handle_manual_reset(ctx, getattr(planner, "policy", None), num_episodes_target=target)

            from manaenv.pipeline.runtime.env_runner import BEFORE_STEP, GYM_READY

            runner.add_hook(GYM_READY, setup, priority=50)
            runner.add_hook(BEFORE_STEP, reset, priority=5)

        return register

    def _provenance(self, report_dir: str) -> dict[str, Any]:
        environments = self.data_center.get(DataBucket.ENVIRONMENTS) or []
        by_name = {str(getattr(env, "name", "")): env for env in environments}
        return {
            "source": "manaenv_public_evaluation",
            "report_dir": str(report_dir),
            "orders": [
                {
                    "env_ref": str(order.env_ref),
                    "num_episodes": order.num_episodes,
                    "num_parallel_envs": order.num_parallel_envs,
                    "episode_length_s": order.episode_length_s,
                    "max_steps": order.max_steps,
                    "resume": order.resume,
                    "planner": _planner_provenance(order.planner),
                    "task": _task_provenance(by_name.get(str(order.env_ref))),
                }
                for order in self._orders
            ],
        }

    @staticmethod
    def _failed_order_result(environment: Any, order: RolloutOrderCFG, exc: BaseException) -> dict[str, Any]:
        name = getattr(environment, "name", str(order.env_ref))
        return {
            "env_name": name,
            "instruction": environment.get_prompt() if environment is not None and hasattr(environment, "get_prompt") else "",
            "actor": "PolicyPlanner",
            "mode": "evaluation",
            "num_episodes": 0,
            "success_count": 0,
            "num_frames": 0,
            "metrics": {"success_rate": 0.0, "avg_score": 0.0, "per_stage_pass_rate": {}},
            "evaluation": {},
            "error": f"{type(exc).__name__}: {exc}",
        }

    @staticmethod
    def _write_benchmark_report(
        results: list[dict[str, Any]], output_dir: str, provenance: dict[str, Any]
    ) -> None:
        from manaenv.pipeline.factories.benchmark_report import build_report, save_report

        report = build_report(results, metadata=provenance)
        path = save_report(report, output_dir=output_dir)
        if path:
            manaprint(f"INFO: benchmark report -> {path}")

    @staticmethod
    def _build_eval_artifact(result: dict[str, Any]) -> EvalArtifact:
        metrics = result.get("metrics", {})
        return EvalArtifact(
            artifact_id=BaseArtifact.generate_id("eval"),
            artifact_type="eval",
            env_refs=[result.get("env_name", "")],
            num_episodes=int(result.get("num_episodes", 0)),
            success_rate=float(metrics.get("success_rate", 0.0)),
            avg_score=float(metrics.get("avg_score", 0.0) or 0.0),
            metrics=metrics,
            failure_cases=failure_cases_from_result(result),
            debug_record_paths=list(result.get("debug_record_paths", []) or []),
            created_at=BaseArtifact.now_iso(),
        )

    def get_artifacts(self) -> list[EvalArtifact]:
        return list(self._artifacts)

    def cleanup(self) -> None:
        self._orders = []
        self._results = []
        self._artifacts = []
