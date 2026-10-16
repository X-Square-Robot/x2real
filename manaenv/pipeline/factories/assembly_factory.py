"""
Assembly Factory Module

This factory assembles the complete IsaacLab Arena Environment by combining
scene, embodiment, and task configs from DataCenter.

The assembled environment is used for policy evaluation.

This is the final stage of the pipeline before execution.
"""

import argparse
import copy
import itertools
import os
import random
import traceback
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import numpy as np

from manaenv.actuators.bias_forces import sync_stable_pd_gravity_compensation_with_articulation_cfg
from manaenv.pipeline.data_types import DataBucket
from manaenv.pipeline.factories.task_factory import GENERATOR_REF_ATTR
from manaenv.pipeline.factory import BaseFactory, BaseFactoryCFG
from manaenv.pipeline.generators.scene_generator import (
    ANTIALIASING_AA_OP,
    DEFAULT_ANTIALIASING_MODE,
    normalize_antialiasing_mode,
)
from manaenv.pipeline.runtime import master_seed as master_seed_runtime
from manaenv.pipeline.runtime.env_runner import _coerce_reset_obs, _simulation_app_is_running
from manaenv.pipeline.runtime_cfg import (
    KIT_RENDER_MODE_ASYNC_HANDSHAKE,
    PHYSICS_STEP_MODE_ASYNC,
    PipelineRuntimeCFG,
    runtime_device_type,
)
from manaenv.utils.debug_print import install_third_party_noise_filter, manaprint
from manaenv.utils.physx_utils import (
    ManaPhysxContractError,
    collect_cloth_semantic_evidence,
    enable_ccd_for_all_rigid_bodies,
    find_particle_feature_prim_paths,
    require_gpu_dynamics_for_particles,
)
from manaenv.utils.viewport_utils import reapply_viewport_camera_view


@dataclass
class AssemblyFactoryCFG(BaseFactoryCFG):
    """Configuration for AssemblyFactory.

    ``num_envs`` here is the default *assembly sample count* — how many
    (scene, robot, task) combinations to draw from the Cartesian product.
    It is **not** the Isaac vectorised-env parallelism; that is configured
    via ``num_parallel_envs`` on evaluation orders.
    """

    class_type: Any = None
    type: str = "default_assembly_factory"
    num_envs: int = 1
    env_spacing: float = 30.0
    step_hz: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = AssemblyFactory


@dataclass
class AssemblyOrderCFG:
    """One assembly order binding generated scene, robot, and task products."""

    name: str = "unnamed"
    scene_generator_name: str = ""
    robot_generator_name: str = ""
    task_generator_name: str = ""
    num_envs: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssemblyOrdersCFG:
    """Typed assembly order section."""

    orders: list[AssemblyOrderCFG] = field(default_factory=list)


@dataclass
class BenchmarkSessionSupport:
    env_obj: Any
    task_runtime_support: Any = None

    def describe(self) -> str:
        if self.task_runtime_support is None:
            return ""
        return self.task_runtime_support.describe()

    def hotkey_reset_label(self) -> str:
        return "reset"

    def register_hooks(self, runner: Any) -> None:
        if self.task_runtime_support is not None:
            self.task_runtime_support.register_hooks(runner)

    def configure_resume(
        self,
        records: list[dict[str, Any]],
        *,
        total_episodes: int,
    ) -> None:
        if self.task_runtime_support is None:
            raise RuntimeError("episode resume requires task runtime support")
        configure = getattr(self.task_runtime_support, "configure_resume", None)
        if not callable(configure):
            raise TypeError("task runtime support does not implement episode resume")
        configure(records, total_episodes=total_episodes)

    def open_evaluation_report(
        self,
        output_dir: str,
        *,
        env_name: str,
        prompt: str = "",
        total_episodes: int | None = None,
    ) -> str | None:
        if self.task_runtime_support is None:
            return None
        open_report = getattr(self.task_runtime_support, "open_evaluation_report", None)
        if not callable(open_report):
            return None
        return open_report(
            output_dir,
            env_name=env_name,
            prompt=prompt,
            total_episodes=total_episodes,
        )

    def handle_manual_reset(self, ctx: Any, policy: Any, *, num_episodes_target: int | None = None) -> None:
        env_ids = list(range(int(getattr(ctx.gym_env, "num_envs", 1))))
        if self.task_runtime_support is not None and env_ids:
            self.task_runtime_support.handle_manual_reset(ctx.gym_env, env_ids, ctx.extra)

        episode_count = ctx.extra.get("episode_count", 0) + len(env_ids)
        ctx.extra["episode_count"] = episode_count
        if num_episodes_target is not None and episode_count >= num_episodes_target:
            ctx.stop_requested = True
            return

        reset_result = ctx.gym_env.reset()
        obs = _coerce_reset_obs(reset_result)
        ctx.extra["override_obs"] = obs
        policy.reset(env_ids)

    def collect_result_payload(self) -> dict[str, Any]:
        if self.task_runtime_support is None:
            return {}
        return self.task_runtime_support.collect_result_payload()

    def collect_evaluation_artifacts(
        self,
        output_dir: str,
        *,
        env_name: str,
        prompt: str = "",
    ) -> dict[str, Any] | None:
        if self.task_runtime_support is None:
            return None
        collect = getattr(self.task_runtime_support, "collect_evaluation_artifacts", None)
        if not callable(collect):
            return None
        return collect(output_dir, env_name=env_name, prompt=prompt)


# scene_layout_solver task runtime event bridge
_ENV_OWNED_TASK_RUNTIME_EVENT_DUPLICATES = {"mdp__reset_scene_to_default"}
_ENABLE_ALL_RIGID_BODY_CCD_PARAM = "enable_all_rigid_body_ccd"
_ENABLE_ALL_RIGID_BODY_SPECULATIVE_CCD_PARAM = "enable_all_rigid_body_speculative_ccd"


def _configclass_field_items(config: Any) -> list[tuple[str, type, Any]]:
    if config is None:
        return []
    fields = getattr(config, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        names = [str(name) for name in fields]
    else:
        names = [str(name) for name in vars(config)]
        if not names:
            names = [
                str(name)
                for name, value in vars(type(config)).items()
                if not str(name).startswith("_") and not callable(value)
            ]
    items: list[tuple[str, type, Any]] = []
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            items.append((name, type(value), value))
    return items


def _event_term_value_signature(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (str(key), _event_term_value_signature(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ),
        )
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_event_term_value_signature(item) for item in value))
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return ("callable", module, qualname)
    fields = getattr(value, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        names = [str(name) for name in fields]
    else:
        names = [str(name) for name in vars(value)] if hasattr(value, "__dict__") else []
    if names:
        return (
            "object",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (name, _event_term_value_signature(getattr(value, name)))
                for name in sorted(names)
                if hasattr(value, name)
            ),
        )
    return ("repr", repr(value))


def _event_terms_can_reuse_existing(name: str, env_value: Any, task_value: Any) -> bool:
    if _event_term_value_signature(env_value) == _event_term_value_signature(task_value):
        return True
    return name in _ENV_OWNED_TASK_RUNTIME_EVENT_DUPLICATES


def _is_full_scene_reset_event(name: str, value: Any) -> bool:
    event_func = getattr(value, "func", None)
    return str(name).endswith("reset_scene_to_default") or getattr(event_func, "__name__", "") == (
        "reset_scene_to_default"
    )


def _order_reset_event_terms(event_terms: Any) -> Any:
    """Run full-scene reset before every other reset event for every assembled env."""

    items = _configclass_field_items(event_terms)
    if not items:
        return event_terms
    scene_reset_items = [item for item in items if _is_full_scene_reset_event(item[0], item[2])]
    if not scene_reset_items:
        return event_terms
    remaining_items = [item for item in items if not _is_full_scene_reset_event(item[0], item[2])]
    ordered_items = [*scene_reset_items, *remaining_items]
    if [name for name, _value_type, _value in ordered_items] == [name for name, _value_type, _value in items]:
        return event_terms
    try:
        from isaaclab_arena.utils.configclass import make_configclass

        return make_configclass("SceneResetFirstEventsCfg", ordered_items)()
    except ModuleNotFoundError:
        fallback = type("SceneResetFirstEventsCfg", (), {})()
        for name, _value_type, value in ordered_items:
            setattr(fallback, name, value)
        return fallback


def _merge_task_runtime_event_terms(env_events: Any, task_events: Any) -> Any:
    task_items = _configclass_field_items(task_events)
    if not task_items:
        return env_events

    merged_items = _configclass_field_items(env_events)
    env_values = {name: value for name, _type, value in merged_items}
    seen = set(env_values)
    for name, value_type, value in task_items:
        if name in seen:
            if _event_terms_can_reuse_existing(name, env_values[name], value):
                continue
            raise ValueError(f"task.runtime_cfgs.event_terms conflicts with env_cfg.events.{name}")
        merged_items.append((name, value_type, value))
        seen.add(name)
    try:
        from isaaclab_arena.utils.configclass import make_configclass

        return make_configclass("TaskRuntimeEventsCfg", merged_items)()
    except ModuleNotFoundError:
        fallback = type("TaskRuntimeEventsCfg", (), {})()
        for name, _value_type, value in merged_items:
            setattr(fallback, name, value)
        return fallback


class AssembledEnvironment:
    """Represents an assembled Arena Environment ready for execution.

    This wraps IsaacLabArenaEnvironment and provides:
    - Environment lifecycle management
    - Data collection interface
    - Validation utilities
    """

    def __init__(
        self,
        name: str,
        scene: Any,
        embodiment: Any,
        task: Any,
        task_factory: Any = None,
        task_builder: Any = None,
        num_envs: int = 1,
        runtime_cfg: PipelineRuntimeCFG | None = None,
    ):
        self.name = name
        self.scene = scene
        self.embodiment = embodiment
        self.task = task
        self.task_factory = task_factory
        # Keep the pre-finalization task product so an active-only scene epoch
        # can re-bind role/contact/viewer runtime cfgs to the new scene.
        self.task_builder = task_builder
        self.num_envs = num_envs
        self.runtime_cfg = runtime_cfg if runtime_cfg is not None else PipelineRuntimeCFG()

        # Capture render carb settings from scene at construction time,
        # before any deep-copy / configclass __post_init__ can strip them.
        self._render_carb_settings: dict = getattr(scene, "render_carb_settings", None) or {}
        self._dlssg_multiplier: int | None = getattr(scene, "dlssg_multiplier", None)
        self._antialiasing_mode = normalize_antialiasing_mode(
            getattr(scene, "antialiasing_mode", None),
            default=DEFAULT_ANTIALIASING_MODE,
        )

        # Arena environment (created lazily)
        self._arena_env = None
        self._env_builder = None
        self._gym_env = None

        # Status
        self.is_built = False
        self.is_running = False

    def get_prompt(self) -> str:
        """Get the task prompt for this environment."""
        return self.task.get_prompt()

    def create_task_runtime_support(self, *, extra_key: str | None = None) -> Any:
        if self.task_factory is None:
            return None
        create_support = getattr(self.task_factory, "create_task_runtime_support", None)
        if not callable(create_support):
            return None
        return create_support(self, extra_key=extra_key)

    def create_benchmark_session_support(self, runtime_exporter: Any = None) -> BenchmarkSessionSupport:
        return BenchmarkSessionSupport(
            env_obj=self,
            task_runtime_support=self.create_task_runtime_support(),
        )

    def build_arena_environment(self):
        """Build the IsaacLabArenaEnvironment configclass.

        This creates the Arena environment descriptor that combines
        scene, embodiment, and task.
        """
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

        # self.task is already the Mana task instance (e.g., ManaPickAndPlaceTask)
        # which inherits from Arena task, so we can use it directly

        self._arena_env = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=self.embodiment,
            scene=self.scene,
            task=self.task,
        )

        manaprint(f"INFO: Built environment: {self.name}")
        return self._arena_env

    def build_gym_environment(
        self,
        args: argparse.Namespace = None,
        mode: str = "evaluate",
        benchmark_video_output_dir: str | None = None,
        benchmark_video_fps: float = 20.0,
        benchmark_video_camera_order: tuple[str, ...] = tuple(),
        collection_output_path: str | None = None,
    ):
        """Build the Gym environment used by the policy-evaluation loop.

        ``benchmark_video_output_dir`` enables the public episode MP4 recorder.
        It uses IsaacLab's RecorderManager lifecycle without enabling dataset
        export, replay, teleoperation, or demonstration collection.
        """
        if self._arena_env is None:
            self.build_arena_environment()

        if self._arena_env is None:
            manaprint("ERROR: Cannot build Gym env without Arena env")
            return None

        try:
            self._apply_diff_ik_compute_mode()

            from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse

            from manaenv.environments.mana_arena_env_builder import ManaArenaEnvBuilder

            if args is None:
                args = self._create_default_args()
            self._pin_physx_preset(args)
            self._reject_removed_visualizers(args)
            # Public evaluation never enters Arena's optional data-generation path.
            if hasattr(args, "mimic"):
                args.mimic = False

            # ManaArenaEnvBuilder preserves Arena's typed CFG and selects one
            # Gym entry point for the requested physics/Kit schedule. The
            # builder cfg carries args.device through parse_env_cfg() into
            # env_cfg.sim.device; missing Namespace fields use cfg defaults.
            self._env_builder = ManaArenaEnvBuilder(
                self._arena_env,
                arena_env_builder_cfg_from_argparse(args),
                physics_step_mode=self.runtime_cfg.physics_step_mode,
                kit_render_mode=self.runtime_cfg.kit_render_mode,
            )
            env_cfg_modifier = getattr(self.scene, "_manaenv_env_cfg_modifier", None)
            if callable(env_cfg_modifier):
                self._arena_env.env_cfg_callback = env_cfg_modifier
            env_name, env_cfg, env_kwargs = self._env_builder.build_registered()
            # Arena asset pose-reset events still call setLinearVelocity; rewrite
            # them to ManaEnv kinematic-safe variants before gym.make.
            from manaenv.terms.events import rewrite_arena_pose_reset_events

            rewritten_pose_events = rewrite_arena_pose_reset_events(getattr(env_cfg, "events", None))
            if rewritten_pose_events:
                manaprint(
                    f"INFO: Rewrote {rewritten_pose_events} Arena pose-reset event(s) "
                    "to ManaEnv kinematic-safe variants."
                )
            # A stream viewport is a display request, not a request to expose
            # camera observations or removed recorder terms from the robot embodiment.
            # Arena omits camera sensors when ``embodiment.enable_cameras`` is
            # false, so attach the embodiment's camera sensor cfg explicitly
            # after the builder has composed the scene.
            self._apply_stream_viewport_camera_cfg(env_cfg)
            # Arena installs the pinned PhysX preset during build_registered();
            # apply the authored solver overrides to that resolved config.
            self._apply_camera_renderer_cfg(env_cfg)
            self._apply_physx_overrides(env_cfg)
            self._apply_actuator_device(env_cfg)
            task_seed = self._resolve_task_seed()
            if task_seed is not None and hasattr(env_cfg, "seed"):
                env_cfg.seed = task_seed
            elif hasattr(env_cfg, "seed"):
                manaprint("INFO: Env seed intentionally unset; add task seed to opt into deterministic env resets.")

            if hasattr(env_cfg, "num_rerenders_on_reset"):
                env_cfg.num_rerenders_on_reset = 2

            if hasattr(self.task, "resolve_episode_length"):
                self.task.resolve_episode_length(env_cfg)
            self._apply_task_runtime_env_overrides(env_cfg)

            self._apply_mode_overrides(env_cfg, mode=mode)
            if collection_output_path:
                from manaenv.pipeline.datacollection.recorder import configure_collection_recorder

                configure_collection_recorder(env_cfg, output_path=collection_output_path)
            if benchmark_video_output_dir:
                from manaenv.pipeline.evaluation.video_recorder import configure_benchmark_video_recorder

                configure_benchmark_video_recorder(
                    env_cfg,
                    output_dir=benchmark_video_output_dir,
                    fps=benchmark_video_fps,
                    camera_order=benchmark_video_camera_order,
                )
            self._prepare_stable_pd_actuator_cfgs(env_cfg)
            self._assert_async_render_cpu_worker_contract(env_cfg)
            # Fail before gym.make so cloth/particle scenes never hit the
            # opaque OmniPhysX "Particles feature is only supported on GPU" log.
            self._assert_particle_cloth_requires_gpu()

            import gymnasium as gym

            self._gym_env = gym.make(env_name, cfg=env_cfg, **env_kwargs).unwrapped
            self._gym_env._mana_assembled_environment = self
            if collection_output_path:
                from manaenv.pipeline.datacollection.staging import install_collection_staging

                install_collection_staging(getattr(self._gym_env, "recorder_manager", None))
            self._run_post_spawn_hooks()
            from manaenv.utils.asset_utils import install_kinematic_velocity_write_guards

            guarded = install_kinematic_velocity_write_guards(self._gym_env)
            if guarded:
                manaprint(f"INFO: Installed kinematic velocity write guards on {guarded} rigid body asset(s).")
            self._assert_stage_particles_require_gpu()
            self._apply_all_rigid_body_ccd()
            # Render carb settings must be applied after env + cameras are created.
            self._apply_render_carb_settings()
            reapply_viewport_camera_view(self._gym_env, logger=manaprint)
            self.is_built = True
            install_third_party_noise_filter()

            manaprint(f"INFO: Built evaluation environment: {self.name} (mode={mode})")
            return self._gym_env

        except ManaPhysxContractError:
            # Contract violations must abort the rollout, not soft-fail to None.
            raise
        except Exception as e:
            if not _simulation_app_is_running():
                # The viewer was closed while building (e.g. a scene-epoch
                # rebuild racing a window close). Nothing is wrong with the
                # environment, so report a clean stop instead of a stack trace.
                manaprint("INFO: viewer closed while building the Gym environment; skipping build")
                return None
            manaprint(f"ERROR: Failed to build Gym environment: {e}")
            traceback.print_exc()
            return None

    def _apply_diff_ik_compute_mode(self) -> int:
        """Reapply DiffIK cadence after planners or policies replace the action config."""

        from manaenv.actions.command_rate_diff_ik import configure_diff_ik_action_cfg

        mode = getattr(self.embodiment, "diff_ik_compute_mode", "command")
        runtime_cfg = getattr(self, "runtime_cfg", None)
        use_warp = bool(
            getattr(
                runtime_cfg,
                "use_warp_diff_ik",
                getattr(self.embodiment, "use_warp_diff_ik", False),
            )
        )
        diff_ik_device = getattr(runtime_cfg, "diff_ik_device", None)
        use_cuda_graph = getattr(runtime_cfg, "diff_ik_cuda_graph", None)
        if use_cuda_graph is None:
            use_cuda_graph = getattr(runtime_cfg, "cuda_graph", None)
        return configure_diff_ik_action_cfg(
            getattr(self.embodiment, "action_config", None),
            mode,
            use_warp=use_warp,
            use_cuda_graph=use_cuda_graph,
            device=diff_ik_device,
        )

    # ------------------------------------------------------------------
    #  Env cfg normalization
    # ------------------------------------------------------------------

    _SUPPORTED_RENDER_CFG_TYPES = {"isaac_rtx"}

    def _make_renderer_cfg(self, renderer_type: str):
        """Create the renderer cfg selected by ``runtime.render_cfg``."""
        if renderer_type == "isaac_rtx":
            from isaaclab_physx.renderers import IsaacRtxRendererCfg

            return IsaacRtxRendererCfg()
        supported = ", ".join(sorted(self._SUPPORTED_RENDER_CFG_TYPES))
        raise ValueError(f"Unknown runtime.render_cfg {renderer_type!r}. Supported values: {supported}.")

    def _apply_camera_renderer_cfg(self, env_cfg) -> None:
        """Normalize camera renderer cfgs before IsaacLab validates scene requirements."""
        renderer_type = str(getattr(self.runtime_cfg, "render_cfg", "isaac_rtx") or "isaac_rtx")
        renderer_cfg = self._make_renderer_cfg(renderer_type)
        scene_cfg = getattr(env_cfg, "scene", None)
        if scene_cfg is None:
            return

        updated = 0
        for _, sensor_cfg in self._iter_dataclass_values(scene_cfg):
            current = getattr(sensor_cfg, "renderer_cfg", None)
            if current is None:
                continue
            if getattr(current, "renderer_type", None) == renderer_type:
                continue
            previous_depth = getattr(current, "depth_clipping_behavior", None)
            new_cfg = copy.deepcopy(renderer_cfg)
            if previous_depth is not None and hasattr(new_cfg, "depth_clipping_behavior"):
                new_cfg.depth_clipping_behavior = previous_depth
            sensor_cfg.renderer_cfg = new_cfg
            updated += 1
        if updated:
            manaprint(f"INFO: Applied camera render_cfg={renderer_type!r} to {updated} sensor(s)")

    def _apply_stream_viewport_camera_cfg(self, env_cfg: Any) -> None:
        """Attach display-only camera sensors without enabling robot camera terms.

        ``EmbodimentBase.get_scene_cfg`` intentionally excludes camera sensors
        when ``enable_cameras`` is false.  That flag must remain false for
        observation semantics, while a declared ``stream_viewports``
        block still needs the sensor render products used by the viewport
        preview.  This method adds only the scene camera cfg and never mutates
        the embodiment flag or its observation configs.
        """
        stream_cfg = getattr(self.runtime_cfg, "stream_viewports", None)
        if not isinstance(stream_cfg, Mapping) or not stream_cfg:
            return
        if bool(getattr(self.embodiment, "enable_cameras", False)):
            # The upstream builder already included the camera cfg in this
            # case; merging it again would duplicate scene fields.
            return

        scene_cfg = getattr(env_cfg, "scene", None)
        get_camera_cfg = getattr(self.embodiment, "get_camera_cfg", None)
        if scene_cfg is None or not callable(get_camera_cfg):
            manaprint("WARNING: stream_viewports requested but no scene/camera config is available")
            return

        camera_cfg = get_camera_cfg()
        if camera_cfg is None:
            manaprint("WARNING: stream_viewports requested but the embodiment has no camera config")
            return
        if not (is_dataclass(scene_cfg) and is_dataclass(camera_cfg)):
            manaprint("WARNING: stream_viewports camera cfg is not a configclass; skipping display-only sensors")
            return

        scene_fields = {cfg_field.name for cfg_field in fields(scene_cfg)}
        camera_fields = {cfg_field.name for cfg_field in fields(camera_cfg)}
        duplicate_fields = scene_fields.intersection(camera_fields)
        if duplicate_fields:
            names = ", ".join(sorted(duplicate_fields))
            manaprint(f"WARNING: stream_viewports camera cfg overlaps scene fields ({names}); skipping merge")
            return

        from isaaclab_arena.utils.configclass import combine_configclass_instances

        env_cfg.scene = combine_configclass_instances("SceneCfg", scene_cfg, camera_cfg)
        manaprint("INFO: Added display-only stream viewport camera sensors; robot enable_cameras remains False")

    @staticmethod
    def _iter_dataclass_values(obj):
        """Yield dataclass field values recursively, guarding against cycles."""
        stack = [obj]
        seen: set[int] = set()
        while stack:
            item = stack.pop()
            item_id = id(item)
            if item_id in seen:
                continue
            seen.add(item_id)
            if not is_dataclass(item):
                continue
            for cfg_field in fields(item):
                value = getattr(item, cfg_field.name, None)
                yield cfg_field.name, value
                if is_dataclass(value):
                    stack.append(value)

    # ------------------------------------------------------------------
    #  Evaluation env_cfg normalization
    # ------------------------------------------------------------------

    def _apply_mode_overrides(
        self,
        env_cfg,
        *,
        mode: str,
    ):
        """Apply the small set of observation settings needed by evaluation."""
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.concatenate_terms = False
        if mode not in {"evaluate", "raw", "collect"}:
            raise ValueError(f"unsupported public environment mode: {mode!r}")

    def _apply_task_runtime_env_overrides(self, env_cfg: Any) -> None:
        """Apply task runtime requirements and global reset ordering to the final env cfg."""
        self._apply_task_runtime_event_terms(env_cfg)
        if hasattr(env_cfg, "events"):
            env_cfg.events = _order_reset_event_terms(getattr(env_cfg, "events", None))

    # scene_layout_solver task runtime event bridge
    def _apply_task_runtime_event_terms(self, env_cfg: Any) -> None:
        runtime_cfgs = getattr(self.task, "runtime_cfgs", None)
        if not isinstance(runtime_cfgs, Mapping):
            return
        event_terms = runtime_cfgs.get("event_terms")
        if event_terms is None:
            return
        if not hasattr(env_cfg, "events"):
            raise ValueError("task.runtime_cfgs.event_terms requires env_cfg.events")
        env_cfg.events = _merge_task_runtime_event_terms(getattr(env_cfg, "events", None), event_terms)

    def export_to_usd(self, output_path: str) -> bool:
        """Export the entire assembled environment Stage to USD.

        Uses isaacsim's built-in save_stage for proper metadata handling
        (upAxis, metersPerUnit, etc.).

        Args:
            output_path: Path to save the USD file (.usd or .usda)

        Returns:
            True if export successful
        """
        if not self.is_built or self._gym_env is None:
            manaprint("ERROR: Environment not built. Call build_gym_environment() first.")
            return False
        try:
            from isaaclab.sim.utils import stage as stage_utils

            # Save current runtime stage as-is (exact dump).
            result = stage_utils.save_stage(output_path, save_and_reload_in_place=False)
            if result:
                manaprint(f"INFO: Exported environment to USD: {output_path}")
            else:
                manaprint(f"WARNING: save_stage returned False for: {output_path}")
            return bool(result)
        except Exception as e:
            manaprint(f"ERROR: Failed to export environment: {e}")
            traceback.print_exc()
            return False

    def settle_and_export_to_usd(self, output_path: str, settle_steps: int = 120) -> bool:
        """Build env, settle simulation, then export final stage to USD.

        This is used for reliable object placement export: objects can
        contact/snap under physics first, then we save the settled result.
        """
        env = self.get_env()
        if env is None:
            manaprint("ERROR: Cannot settle/export because Gym env build failed")
            return False

        try:
            if hasattr(env, "reset"):
                env.reset()
        except Exception as e:
            manaprint(f"WARNING: env.reset() failed before settle: {e}")

        if settle_steps > 0:
            try:
                action = self._make_zero_action(env)
                for _ in range(settle_steps):
                    env.step(action)
            except Exception as e:
                manaprint(f"WARNING: settle via env.step() failed, trying sim.step(): {e}")
                try:
                    sim = getattr(env, "sim", None)
                    if sim is None:
                        raise RuntimeError("env.sim not found")
                    for _ in range(settle_steps):
                        sim.step(render=False)
                except Exception as e2:
                    manaprint(f"ERROR: settle failed: {e2}")
                    return False

        return self.export_to_usd(output_path)

    def _make_zero_action(self, env):
        """Create zero action matching env action space shape."""
        action_space = getattr(env, "action_space", None)
        if action_space is None or not hasattr(action_space, "shape"):
            raise RuntimeError("Environment action_space missing")
        shape = action_space.shape
        dtype = getattr(action_space, "dtype", np.float32)
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def _env_cfg_physics_dt(env_cfg: Any) -> float | None:
        sim_cfg = getattr(env_cfg, "sim", None)
        try:
            dt = float(getattr(sim_cfg, "dt"))
        except (AttributeError, TypeError, ValueError):
            return None
        return dt if dt > 0.0 else None

    def _prepare_stable_pd_actuator_cfgs(self, env_cfg: Any) -> None:
        """Prepare StablePD cfgs with env sim dt and gravity policy before spawn.

        This deliberately does not read runtime_cfg.actuator: by this
        point the scene robot cfg already contains whichever actuator cfg
        objects the embodiment selected. The helper only touches cfgs exposing
        StablePD-specific attributes, so implicit/disabled StablePD paths pass
        through unchanged while shared gravity policy stays centralized.
        """
        dt = self._env_cfg_physics_dt(env_cfg)
        robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
        actuators = getattr(robot_cfg, "actuators", None)
        if not isinstance(actuators, dict):
            return

        seeded = 0
        for actuator_cfg in actuators.values():
            if not hasattr(actuator_cfg, "sim_dt"):
                continue
            if dt is not None:
                current = getattr(actuator_cfg, "sim_dt", None)
                try:
                    needs_seed = current is None or float(current) <= 0.0
                except (TypeError, ValueError):
                    needs_seed = True
                if needs_seed:
                    actuator_cfg.sim_dt = dt
                    seeded += 1
        gravity_synced = sync_stable_pd_gravity_compensation_with_articulation_cfg(robot_cfg)

        if seeded:
            manaprint(f"INFO: StablePD sim_dt default set to env sim dt={dt:g} for {seeded} actuator group(s).")
        if gravity_synced:
            enabled = next(
                getattr(a, "use_gravity_compensation")
                for a in actuators.values()
                if hasattr(a, "use_gravity_compensation")
            )
            manaprint(
                "INFO: StablePD gravity compensation "
                f"set to {enabled} from robot disable_gravity for {gravity_synced} actuator group(s)."
            )

        self._maybe_enable_newton_actuators(env_cfg, actuators)
        self._apply_per_articulation_newton_selection(env_cfg)

    def _apply_per_articulation_newton_selection(self, env_cfg: Any) -> None:
        """Route StablePD articulations through MR30's native actuator path."""
        from manaenv.actuators.newton_compat import apply_per_articulation_newton_selection

        resolved = apply_per_articulation_newton_selection(
            env_cfg,
            runtime_native=getattr(self.runtime_cfg, "native_actuators", None),
        )
        if resolved:
            summary = ", ".join(f"{path or '<root>'}={selected}" for path, selected in sorted(resolved.items()))
            manaprint(f"INFO: Newton per-articulation selection: {summary}.")

    def _maybe_enable_newton_actuators(self, env_cfg: Any, actuators: dict) -> None:
        """Enable the Newton actuator graph required by StablePD on PhysX.

        This is deliberately separate from the removed Newton *physics backend*
        support.  Isaac Lab's StablePD implementation is executed by the
        Newton actuator graph even when ``sim.physics`` is PhysX, so the graph
        flag and the small target-name compatibility shim must remain enabled
        for StablePD environments.
        """
        want = getattr(self.runtime_cfg, "native_actuators", None)
        has_stable_pd = any(type(actuator_cfg).__name__ == "StablePDActuatorCfg" for actuator_cfg in actuators.values())
        if want is None:
            want = has_stable_pd

        sim_cfg = getattr(env_cfg, "sim", None)
        if sim_cfg is None or not hasattr(sim_cfg, "use_newton_actuators"):
            if want:
                manaprint(
                    "WARNING: Newton actuator graph requested but sim cfg lacks 'use_newton_actuators'; "
                    "actuator will fall back to explicit PD (Isaac Lab version mismatch?)."
                )
            return

        if not want:
            sim_cfg.use_newton_actuators = False
            return

        from manaenv.actuators.newton_compat import install_physx_actuator_target_compat

        install_physx_actuator_target_compat()
        sim_cfg.use_newton_actuators = True
        reason = (
            "forced via runtime.native_actuators"
            if getattr(self.runtime_cfg, "native_actuators", None) is True
            else "StablePD present"
        )
        manaprint(f"INFO: Enabled sim.use_newton_actuators (Newton actuator-graph joint PD) [{reason}].")

    def _run_post_spawn_hooks(self) -> None:
        """Run post-spawn hooks registered by SceneGenerator components.

        Must be called after ``gym.make()`` so the USD stage is fully composed.
        Each component that overrides ``SceneComponent.post_spawn`` gets a chance
        to modify the live stage (e.g. deactivating embedded lights).
        """
        hooks = getattr(self.scene, "_post_spawn_hooks", None)
        if not hooks:
            return
        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()

        for hook in hooks:
            hook(stage, self.num_envs)

    def _assert_particle_cloth_requires_gpu(self) -> None:
        """Reject CPU PhysX when scene assets declare the ``cloth`` semantic tag."""
        require_gpu_dynamics_for_particles(
            device=getattr(self.runtime_cfg, "device", None),
            evidence=collect_cloth_semantic_evidence(self.scene),
        )

    def _assert_stage_particles_require_gpu(self, *, stage: Any = None) -> None:
        """Reject CPU PhysX when the composed USD stage has particle APIs."""
        if stage is None:
            from isaaclab.sim.utils import get_current_stage

            stage = get_current_stage()
        paths = find_particle_feature_prim_paths(stage)
        require_gpu_dynamics_for_particles(
            device=getattr(self.runtime_cfg, "device", None),
            evidence=[f"stage prim {path}" for path in paths],
        )

    def _apply_all_rigid_body_ccd(self, *, stage: Any = None) -> int:
        """Enable YAML-selected CCD modes on every rigid body after scene composition."""
        raw_params = getattr(self.runtime_cfg, "solver_params", None) or {}
        enable_swept = raw_params.get(_ENABLE_ALL_RIGID_BODY_CCD_PARAM) is True
        enable_speculative = raw_params.get(_ENABLE_ALL_RIGID_BODY_SPECULATIVE_CCD_PARAM) is True
        if not enable_swept and not enable_speculative:
            return 0
        if stage is None:
            from isaaclab.sim.utils import get_current_stage

            stage = get_current_stage()

        applied = enable_ccd_for_all_rigid_bodies(
            stage,
            swept=enable_swept,
            speculative=enable_speculative,
        )
        modes = (
            "swept and speculative"
            if enable_swept and enable_speculative
            else "swept" if enable_swept else "speculative"
        )
        manaprint(f"INFO: Enabled {modes} CCD on {applied} rigid-body prim(s).")
        return applied

    def _apply_render_carb_settings(self) -> None:
        """Apply render settings via carb after gym.make().

        The scene-generator env-cfg modifier applies these settings before env
        init (via ``apply_isaac_rtx_global_settings``), but many RTX parameters
        only take effect when written to ``carb.settings`` after the full init
        sequence (env + cameras). This mirrors the opensdk ``sim_app.py`` pattern.

        Re-apply YAML ``antialiasing_mode`` after USDA carb keys so local
        render presets cannot silently override the authored AA mode.
        """
        carb_settings = self._render_carb_settings
        aa_mode = getattr(self, "_antialiasing_mode", None) or "DLSS"
        try:
            import carb.settings  # type: ignore[import]

            settings_iface = carb.settings.get_settings()
            for key, value in carb_settings.items():
                # Apply DLSSG after the renderer-mode switch below; changing the
                # mode can recreate the DLSSG context and clear this setting.
                if key == "rtx-transient.dlssg.enabled":
                    continue
                # render_cfg_dict uses "rtx.xxx" keys; carb needs "/rtx/xxx"
                carb_key = "/" + key.replace(".", "/") if not key.startswith("/") else key
                settings_iface.set(carb_key, value)
            applied_via = self._apply_antialiasing_mode(settings_iface, aa_mode)
            # Renderer-mode changes can recreate DLSSG state. Reapply the authored
            # enable flag and multiplier after the mode switch so the final Kit
            # settings match the YAML profile.
            dlssg_enabled = carb_settings.get("rtx-transient.dlssg.enabled")
            if dlssg_enabled is not None:
                settings_iface.set("/rtx-transient/dlssg/enabled", dlssg_enabled)
            effective_dlssg_multiplier = self._apply_dlssg_multiplier(
                settings_iface, getattr(self, "_dlssg_multiplier", None)
            )
            dlssg_readback = None
            if dlssg_enabled is not None:
                try:
                    dlssg_readback = settings_iface.get("/rtx-transient/dlssg/enabled")
                except Exception:
                    pass
            if carb_settings:
                multiplier_note = (
                    f", dlssg=x{effective_dlssg_multiplier}" if effective_dlssg_multiplier is not None else ""
                )
                enabled_note = f", enabled={dlssg_readback!r}" if dlssg_readback is not None else ""
                manaprint(
                    f"INFO: Applied {len(carb_settings)} render carb settings "
                    f"(antialiasing={aa_mode} via {applied_via}{enabled_note}{multiplier_note})"
                )
            else:
                manaprint(f"INFO: Applied render carb settings (antialiasing={aa_mode} via {applied_via})")
        except Exception as exc:
            manaprint(f"WARNING: Failed to apply render carb settings: {exc}")

    @staticmethod
    def _apply_dlssg_multiplier(settings_iface: Any, requested_multiplier: int | None) -> int | None:
        """Apply the DLSSG frame multiplier after checking x3/x4 GPU capability."""
        if requested_multiplier is None:
            return None

        effective_multiplier = int(requested_multiplier)
        if effective_multiplier > 2:
            try:
                x3x4_supported = settings_iface.get("/rtx-transient/dlssg/x3x4supported") is True
            except Exception:
                x3x4_supported = False
            if not x3x4_supported:
                manaprint(f"WARNING: DLSSG x{effective_multiplier} is unavailable on this GPU; falling back to x2")
                effective_multiplier = 2

        interpolated_frame_count = effective_multiplier - 1
        for render_mode in ("rt", "rtpt"):
            settings_iface.set(
                f"/rtx-transient/internal/dlssg/{render_mode}/interpolatedFrameCount",
                interpolated_frame_count,
            )
        return effective_multiplier

    @staticmethod
    def _apply_antialiasing_mode(settings_iface: Any, aa_mode: str) -> str:
        """Force the authored antialiasing mode after USDA carb writes."""
        try:
            import omni.replicator.core as rep  # type: ignore[import]

            rep.settings.set_render_rtx_realtime(antialiasing=aa_mode)
            return "replicator"
        except Exception:
            aa_op = ANTIALIASING_AA_OP.get(aa_mode)
            if aa_op is not None:
                settings_iface.set("/rtx/post/aa/op", aa_op)
                return f"aa/op={aa_op}"
            return "skipped"

    def _create_default_args(self) -> argparse.Namespace:
        """Create default CLI arguments for environment building.

        Camera support depends on how the AppLauncher was started:
        - GUI mode (not headless): rendering pipeline active, cameras work
        - Headless + ``--enable_cameras``: offscreen rendering, cameras work
        - Headless without camera rendering support: no sensor render products

        ``stream_viewports`` requests display-only camera sensors and does not
        change robot observation/recording camera terms.

        We detect this by checking whether the SimulationApp was created
        with rendering enabled (viewport extension loaded).
        """
        args = argparse.Namespace()
        args.device = self.runtime_cfg.device
        args.num_envs = self.num_envs
        args.env_spacing = self._resolve_env_spacing()
        args.seed = self._resolve_task_seed()
        args.disable_fabric = self.runtime_cfg.disable_fabric
        args.presets = "physx"
        args.mimic = False
        args.solve_relations = True
        args.placement_seed = None
        args.resolve_on_reset = None
        args.random_yaw_init = False
        args.xr = False

        cameras_available = False
        try:
            import carb

            settings = carb.settings.get_settings()
            headless = settings.get("/app/window/enabled") is False
            cameras_available = (
                not headless
                or int(os.environ.get("ENABLE_CAMERAS", 0)) == 1
                or bool(getattr(self.runtime_cfg, "stream_viewports", None))
            )
        except Exception:
            cameras_available = int(os.environ.get("ENABLE_CAMERAS", 0)) == 1 or bool(
                getattr(self.runtime_cfg, "stream_viewports", None)
            )

        args.enable_cameras = cameras_available
        return args

    @staticmethod
    def _pin_physx_preset(args: argparse.Namespace) -> None:
        """Reject removed backend presets and make PhysX Arena's final authority."""

        preset = getattr(args, "presets", None)
        if preset not in (None, "physx"):
            raise ManaPhysxContractError(
                "ManaEnv's public runtime supports only the PhysX physics preset; "
                f"got {preset!r}."
            )
        args.presets = "physx"

    @staticmethod
    def _reject_removed_visualizers(args: argparse.Namespace) -> None:
        """Reject the Newton viewer while preserving public PhysX visualizers."""

        raw = getattr(args, "visualizer", None)
        if isinstance(raw, str):
            visualizers = {item.strip().lower() for item in raw.split(",") if item.strip()}
        else:
            visualizers = {str(item).strip().lower() for item in (raw or ()) if str(item).strip()}
        if "newton" in visualizers:
            raise ManaPhysxContractError(
                "The Newton visualizer is not available in ManaEnv's public PhysX runtime."
            )

    def _resolve_env_spacing(self) -> float:
        spacing = getattr(self.runtime_cfg, "world_spacing", None)
        if spacing is None:
            return 30.0
        if isinstance(spacing, (list, tuple)):
            if not spacing:
                return 30.0
            return float(spacing[0])
        return float(spacing)

    def _apply_actuator_device(self, env_cfg: Any) -> None:
        """Resolve native-actuator placement into ``SimulationCfg``."""

        robot_cfg = getattr(self.embodiment, "_mana_cfg", None)
        actuator_device = getattr(robot_cfg, "actuator_device", None)
        if actuator_device is None:
            return
        sim_cfg = getattr(env_cfg, "sim", None)
        if sim_cfg is None:
            return

        actuator_device = getattr(robot_cfg, "actuator_device", None)
        actuator_cuda_graph = getattr(robot_cfg, "actuator_cuda_graph", None)
        scene_robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
        actuators = getattr(scene_robot_cfg, "actuators", None)
        has_stable_pd = isinstance(actuators, dict) and any(
            type(actuator_cfg).__name__ == "StablePDActuatorCfg" for actuator_cfg in actuators.values()
        )
        cpu_physx = runtime_device_type(getattr(self.runtime_cfg, "device", "")) == "cpu"
        if cpu_physx and has_stable_pd:
            actuator_device = "cpu"
            actuator_cuda_graph = False
            placement_source = "CPU PhysX StablePD policy"
        else:
            placement_source = "robot cfg"
        if actuator_device is None:
            return

        sim_cfg.newton_actuator_device = str(actuator_device)
        sim_cfg.newton_actuator_cuda_graph = actuator_cuda_graph
        manaprint(
            "INFO: PhysX native actuator control configured: "
            f"device={actuator_device}, cuda_graph={actuator_cuda_graph} [{placement_source}]."
        )

    def _assert_async_render_cpu_worker_contract(self, env_cfg: Any) -> None:
        """Reject actuator placement that would put CUDA work on the physics worker."""

        runtime_cfg = self.runtime_cfg
        if (
            getattr(runtime_cfg, "kit_render_mode", None) != KIT_RENDER_MODE_ASYNC_HANDSHAKE
            or getattr(runtime_cfg, "physics_step_mode", None) != PHYSICS_STEP_MODE_ASYNC
        ):
            return

        sim_cfg = getattr(env_cfg, "sim", None)
        scene_cfg = getattr(env_cfg, "scene", None)
        if getattr(scene_cfg, "lazy_sensor_update", True) is not True:
            raise ManaPhysxContractError(
                "kit_render_mode='async_handshake' requires scene.lazy_sensor_update=true so the physics "
                "worker cannot enter RenderContext or recompute RTX sensors"
            )
        requested_device = getattr(sim_cfg, "newton_actuator_device", None) if sim_cfg is not None else None
        effective_device = requested_device or runtime_cfg.device
        if runtime_device_type(effective_device) != "cpu":
            raise ManaPhysxContractError(
                "kit_render_mode='async_handshake' requires CPU actuator execution; "
                f"resolved actuator_device={effective_device!r}"
            )
        if getattr(sim_cfg, "newton_actuator_cuda_graph", None) is True:
            raise ManaPhysxContractError(
                "kit_render_mode='async_handshake' cannot enable actuator CUDA Graphs on the physics worker"
            )

    def _apply_physx_overrides(self, env_cfg: Any) -> None:
        """Reflect ``runtime_cfg.solver_params`` onto the resolved PhysX physics cfg.

        ``solver_params`` is validated against ``PhysxCfg`` fields and applied
        directly.
        """
        sim_cfg = getattr(env_cfg, "sim", None)
        if sim_cfg is None:
            raise ManaPhysxContractError("env_cfg.sim is required by the public PhysX runtime")

        from isaaclab_physx.physics import PhysxCfg

        physics = getattr(sim_cfg, "physics", None)
        if physics is None:
            physics = PhysxCfg()
            sim_cfg.physics = physics
        elif not isinstance(physics, PhysxCfg):
            raise ManaPhysxContractError(
                "Arena resolved a non-PhysX physics backend in the public runtime: "
                f"{type(physics).__name__}."
            )

        raw_params = getattr(self.runtime_cfg, "solver_params", None)
        if not raw_params:
            return

        enable_all_rigid_body_ccd = raw_params.get(_ENABLE_ALL_RIGID_BODY_CCD_PARAM, False)
        enable_all_rigid_body_speculative_ccd = raw_params.get(_ENABLE_ALL_RIGID_BODY_SPECULATIVE_CCD_PARAM, False)
        for key, value in (
            (_ENABLE_ALL_RIGID_BODY_CCD_PARAM, enable_all_rigid_body_ccd),
            (_ENABLE_ALL_RIGID_BODY_SPECULATIVE_CCD_PARAM, enable_all_rigid_body_speculative_ccd),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"solver_params.{key} must be a bool")
        physics_device = str(getattr(self.runtime_cfg, "device", "")).lower()
        if physics_device.startswith("cuda"):
            enabled_cpu_only_keys = [
                key
                for key, enabled in (
                    (_ENABLE_ALL_RIGID_BODY_CCD_PARAM, enable_all_rigid_body_ccd),
                    (_ENABLE_ALL_RIGID_BODY_SPECULATIVE_CCD_PARAM, enable_all_rigid_body_speculative_ccd),
                )
                if enabled
            ]
            if enabled_cpu_only_keys:
                raise ValueError(f"solver_params.{enabled_cpu_only_keys[0]} requires CPU PhysX")

        valid_fields = {f.name for f in fields(type(physics))} - {"class_type"}
        applied = {}
        for key, value in raw_params.items():
            if key in {_ENABLE_ALL_RIGID_BODY_CCD_PARAM, _ENABLE_ALL_RIGID_BODY_SPECULATIVE_CCD_PARAM}:
                continue
            if key in valid_fields:
                setattr(physics, key, value)
                applied[key] = value
            else:
                manaprint(
                    f"WARNING: solver_params key {key!r} is not a field of PhysxCfg "
                    f"(valid: {sorted(valid_fields)}); dropped."
                )
        if enable_all_rigid_body_ccd:
            physics.enable_ccd = True
            applied["enable_ccd"] = True
        if applied:
            manaprint(f"INFO: PhysX solver_params overridden via runtime_cfg: {applied}.")

    def _resolve_task_seed(self) -> int | None:
        """Isaac env seed: task override → master-derived isaac_env → unset (no silent 0)."""

        task_cfg = getattr(self.task, "config", None)
        override = getattr(task_cfg, "seed", None) if task_cfg is not None else None
        if override is not None:
            try:
                return master_seed_runtime.to_isaac_seed(int(override))
            except (TypeError, ValueError):
                pass
        derived = master_seed_runtime.derive(master_seed_runtime.NS_ISAAC_ENV)
        return None if derived is None else master_seed_runtime.to_isaac_seed(derived)

    def get_env(self):
        """Get the Gym environment (build if needed)."""
        if self._gym_env is None:
            self.build_gym_environment()
        return self._gym_env

    def rebuild_scene_epoch(self, *, scene: Any | None = None, **build_kwargs: Any):
        """Recreate the Gym/physics scene at an explicit scene-epoch boundary.

        Schema-vNext scenes default to ``active_only`` and contain no inactive
        pool slots.  Changing the selected asset therefore requires rebuilding
        the whole backend topology rather than calling ``env.reset()``.  The
        caller owns asset selection; pass the next scene with ``scene=``.  The
        current scene is reused when no replacement is supplied.

        The teardown order is deliberately environment first, then
        ``SimulationContext.clear_instance()``.  Re-entering ``gym.make`` from a
        reset event would leave managers and sensors holding views into the old
        stage, so this method is intended for an outer runner/epoch controller.
        """
        if scene is not None:
            self.scene = scene

        # Keep the teardown and rebuild under one explicit lifecycle scope.
        with nullcontext():
            if self._gym_env is not None:
                try:
                    self._gym_env.close()
                except Exception as exc:  # noqa: BLE001 - continue teardown at the stage boundary
                    manaprint(f"WARNING: scene-epoch env.close() failed; continuing teardown: {exc}")
                finally:
                    self._gym_env = None

            try:
                from isaaclab.sim import SimulationContext

                SimulationContext.clear_instance()
            finally:
                self._env_builder = None
                self._arena_env = None
                self.is_built = False
                self.is_running = False

            # ``gym.make`` expects a fresh USD stage at every active-only epoch.
            # Keep this lifecycle boundary here so callers cannot accidentally
            # create a stage before the old SimulationContext has been cleared.
            try:
                from isaaclab.sim.utils import stage as stage_utils

                stage_utils.create_new_stage()
            except Exception as exc:  # pragma: no cover - Isaac-only fallback
                manaprint(f"WARNING: scene-epoch stage recreation skipped: {exc}")

            rebuilt = self.build_gym_environment(**build_kwargs)
        return rebuilt

    def requires_episode_scene_rebuild(self) -> bool:
        """Whether this environment has an outer active-only epoch producer."""
        metadata = getattr(self.scene, "metadata", {}) or {}
        return str(metadata.get("schema_vnext_runtime_topology", "")).strip().lower() == "active_only" and callable(
            getattr(self.scene, "_manaenv_scene_epoch_factory", None)
        )

    def prepare_next_scene_epoch(self, *, episode_index: int) -> Any | None:
        """Generate the next scene and re-bind the task before rebuilding Gym."""
        producer = getattr(self.scene, "_manaenv_scene_epoch_factory", None)
        if not callable(producer):
            return None
        next_scene = producer(int(episode_index))
        if next_scene is None:
            return None

        if self.task_factory is not None and self.task_builder is not None:
            builder = AssemblyFactory._clone_task_builder(self.task_builder)
            self.task = self.task_factory.finalize_task(builder, scene=next_scene, robot=self.embodiment)
        elif hasattr(self.task, "link_to_scene"):
            self.task.link_to_scene(next_scene)
        self.scene = next_scene
        return next_scene

    def to_dict(self) -> dict[str, Any]:
        """Export environment configuration as dictionary."""
        return {
            "name": self.name,
            "scene": self.scene.assets if hasattr(self.scene, "assets") else str(self.scene),
            "embodiment": getattr(self.embodiment, "name", str(self.embodiment)),
            "task": getattr(self.task, "task_type", str(self.task)),
            "prompt": self.get_prompt(),
            "is_built": self.is_built,
        }

    def cleanup(self):
        """Clean up environment resources."""
        if self._gym_env is not None:
            try:
                self._gym_env.close()
            except Exception:
                pass
            self._gym_env = None
        self._env_builder = None
        self._arena_env = None
        self.is_built = False
        self.is_running = False


class AssemblyFactory(BaseFactory):
    """Factory for assembling complete Arena environments.

    This factory reads scene/robot/task products and assembles executable
    environments for policy evaluation.
    """

    def __init__(self):
        super().__init__()
        self.cfg: AssemblyFactoryCFG = None
        self.runtime_cfg: PipelineRuntimeCFG = PipelineRuntimeCFG()
        # Assembled environments
        self._environments: dict[str, AssembledEnvironment] = {}
        # Orders for assembly
        self._orders: list[AssemblyOrderCFG] = []
        self.task_factory = None

    def setup(self, cfg: "AssemblyFactoryCFG | None" = None):
        """Setup the assembly factory with configuration."""
        self.cfg = cfg if cfg is not None else AssemblyFactoryCFG()

        self._environments = {}
        self._orders = []
        manaprint(f"INFO: AssemblyFactory setup with type: {self.cfg.type}")

    def set_runtime_cfg(self, runtime_cfg: PipelineRuntimeCFG | None) -> None:
        """Bind global pipeline runtime defaults used for env construction."""
        self.runtime_cfg = runtime_cfg if runtime_cfg is not None else PipelineRuntimeCFG()

    def receive_orders(self, orders: AssemblyOrdersCFG) -> None:
        """Receive assembly orders.

        Each order specifies which scene, robot, and task to combine.
        ``num_envs`` on an order controls how many unique (scene, robot,
        task) combinations are sampled from the Cartesian product — it
        is **not** the Isaac vectorised-env parallelism.

        Args:
            orders: Typed assembly orders with one generated scene, robot,
                and task reference per assembly environment.
        """
        if not isinstance(orders, AssemblyOrdersCFG):
            raise TypeError(f"AssemblyFactory.receive_orders expected AssemblyOrdersCFG, got {type(orders).__name__}")
        self._orders = list(orders.orders)
        for order in self._orders:
            manaprint(f"INFO: AssemblyFactory received order: {order.name}")
        manaprint(f"INFO: AssemblyFactory received {len(self._orders)} order(s)")

    def get_order_count(self) -> int:
        """Get number of pending orders."""
        return len(self._orders)

    @staticmethod
    def _product_matches_name(product: Any, generator_name: str) -> bool:
        """Check whether *product* matches *generator_name* by any known attr."""
        if hasattr(product, "_generator_name") and product._generator_name == generator_name:
            return True
        if hasattr(product, "scene_type") and product.scene_type == generator_name:
            return True
        if hasattr(product, "task_type") and product.task_type == generator_name:
            return True
        if hasattr(product, "robot_type") and product.robot_type == generator_name:
            return True
        if hasattr(product, "metadata") and isinstance(product.metadata, dict):
            if product.metadata.get("generator") == generator_name:
                return True
        return False

    def _find_all_products_by_generator_name(
        self, products: list[Any], generator_name: str, product_type: str
    ) -> list[Any]:
        """Find **all** products matching *generator_name*."""
        if not products:
            return []
        if generator_name.isdigit():
            idx = int(generator_name)
            if 0 <= idx < len(products):
                return [products[idx]]
            return []
        matched = [p for p in products if self._product_matches_name(p, generator_name)]
        if not matched:
            manaprint(f"WARNING: No {product_type} matched generator_name '{generator_name}'")
        return matched

    def assemble(self, data_center: Any, env_name: str | None = None) -> list[AssembledEnvironment]:
        """Assemble environments from DataCenter products based on orders.

        If orders exist, use them to match products by reference.
        Otherwise, create 1:1:1 mapping.

        Args:
            data_center: DataCenter instance containing products
            env_name: Optional environment name filter when assembly
                orders exist. Falls back to a name prefix only when no
                assembly orders are defined.

        Returns:
            List of assembled environments
        """
        scenes = data_center.get(DataBucket.SCENES)
        robots = data_center.get(DataBucket.ROBOTS)
        tasks = data_center.get(DataBucket.TASKS)

        if not scenes:
            manaprint("WARNING: No scenes provided for assembly")
            return []
        if not robots:
            manaprint("WARNING: No robots provided for assembly")
            return []
        if not tasks:
            manaprint("WARNING: No tasks provided for assembly")
            return []

        manaprint(f"INFO: Assembling environments from {len(scenes)} scenes, {len(robots)} robots, {len(tasks)} tasks")

        environments = []

        if self._orders:
            # Use orders to match products by generator_name
            orders = self._orders
            if env_name:
                orders = [order for order in self._orders if order.name == env_name]
                if not orders:
                    manaprint(f"WARNING: No assembly order for environment '{env_name}'")
                    return []

            manaprint(f"INFO: Using {len(orders)} assembly order(s)")
            for order in orders:
                environments.extend(
                    self._assemble_cartesian_sampled(
                        order.name,
                        [order.scene_generator_name],
                        [order.robot_generator_name],
                        [order.task_generator_name],
                        max(int(order.num_envs), 1),
                        scenes,
                        robots,
                        tasks,
                    )
                )
        else:
            # Fallback: 1:1:1 mapping
            manaprint("INFO: No orders, using 1:1:1 mapping")
            for i, (scene, robot, task_cfg) in enumerate(zip(scenes, robots, tasks)):
                name = env_name or f"mana_env_{i}"
                if hasattr(task_cfg, "task_type"):
                    name = f"{task_cfg.task_type}_env_{i}"

                env = self._assemble_single(name, scene, robot, task_cfg)
                if env:
                    environments.append(env)

        manaprint(f"INFO: Assembly complete. Total: {len(environments)} environment(s)")
        return environments

    @staticmethod
    def _deduplicate_environments(
        envs: list["AssembledEnvironment"],
    ) -> list["AssembledEnvironment"]:
        """Remove environments whose (scene assets + poses, task prompt) are identical.

        The fingerprint includes per-asset positions (rounded) so that
        environments built from the same assets but with different poses
        are kept as distinct variants.
        """
        seen: set = set()
        unique: list = []
        for env in envs:
            if hasattr(env.scene, "assets"):
                parts: list[str] = []
                for name in sorted(env.scene.assets.keys()):
                    asset = env.scene.assets[name]
                    pose = getattr(asset, "initial_pose", None)
                    if pose is not None:
                        pos = tuple(round(v, 3) for v in pose.position_xyz)
                        rot = tuple(round(v, 3) for v in pose.rotation_xyzw)
                        parts.append(f"{name}@{pos}{rot}")
                    else:
                        parts.append(name)
                scene_fp: object = tuple(parts)
            else:
                scene_fp = id(env.scene)
            prompt = env.get_prompt()
            fp = (scene_fp, prompt)
            if fp not in seen:
                seen.add(fp)
                unique.append(env)
        return unique

    def _assemble_cartesian_sampled(
        self,
        order_name: str,
        scene_refs: list[str],
        robot_refs: list[str],
        task_refs: list[str],
        sample_count: int,
        all_scenes: list[Any],
        all_robots: list[Any],
        all_tasks: list[Any],
    ) -> list[AssembledEnvironment]:
        """Build environments by sampling from the Cartesian product of
        matched scene/robot/task products.

        Strategy: shuffle the full product, take the first
        ``min(sample_count, |product|)`` unique triples, then deduplicate
        via ``_deduplicate_environments``.
        """
        scene_pool: list[Any] = []
        for gen in scene_refs or ["0"]:
            scene_pool.extend(self._find_all_products_by_generator_name(all_scenes, gen or "0", "scene"))
        robot_pool: list[Any] = []
        for gen in robot_refs or ["0"]:
            robot_pool.extend(self._find_all_products_by_generator_name(all_robots, gen or "0", "robot"))
        task_pool: list[Any] = []
        for gen in task_refs or ["0"]:
            task_pool.extend(self._find_all_products_by_generator_name(all_tasks, gen or "0", "task"))

        if not scene_pool or not robot_pool or not task_pool:
            manaprint(f"ERROR: Empty product pool for order '{order_name}', skipping")
            return []

        product = list(itertools.product(scene_pool, robot_pool, task_pool))
        random.shuffle(product)
        selected = product[:sample_count]

        envs: list[AssembledEnvironment] = []
        for idx, (scene, robot, task_cfg) in enumerate(selected):
            suffix = f"_v{idx}" if len(selected) > 1 else ""
            env = self._assemble_single(f"{order_name}{suffix}", scene, robot, task_cfg)
            if env:
                envs.append(env)

        unique = self._deduplicate_environments(envs)
        if len(unique) < len(envs):
            manaprint(f"INFO: Deduplicated {len(envs)} -> {len(unique)} environments for order '{order_name}'")
        return unique

    def _assemble_single(self, name: str, scene: Any, robot: Any, task_cfg: Any) -> AssembledEnvironment | None:
        """Assemble a single environment (num_envs always 1 at assembly time)."""
        task = self._build_task_instance(task_cfg, scene, robot)
        if task is None:
            manaprint(f"ERROR: Failed to build task for env '{name}'")
            return None

        if hasattr(task, "link_to_scene"):
            task.link_to_scene(scene)

        env = AssembledEnvironment(
            name=name,
            scene=scene,
            embodiment=robot,
            task=task,
            task_factory=self.task_factory,
            task_builder=task_cfg,
            num_envs=1,
            runtime_cfg=self.runtime_cfg,
        )
        self._environments[name] = env

        prompt = env.get_prompt()
        manaprint(f"INFO: Assembled environment '{name}'")
        manaprint(f"INFO:   Scene: {len(scene.assets) if hasattr(scene, 'assets') else 'N/A'} assets")
        manaprint(f"INFO:   Robot: {getattr(robot, 'name', str(robot))}")
        manaprint(f"INFO:   Task: {getattr(task_cfg, 'task_type', str(task_cfg))}")
        manaprint(f'INFO:   Prompt: "{prompt}"')

        return env

    def assemble_all_combinations(self, data_center: Any) -> list[AssembledEnvironment]:
        """Assemble all possible combinations from DataCenter products.

        This creates N_scenes * N_robots * N_tasks environments.
        """
        scenes = data_center.get(DataBucket.SCENES)
        robots = data_center.get(DataBucket.ROBOTS)
        tasks = data_center.get(DataBucket.TASKS)

        if not scenes or not robots or not tasks:
            manaprint("WARNING: Missing scenes/robots/tasks for combinatorial assembly")
            return []
        environments = []

        for si, scene in enumerate(scenes):
            for ri, robot in enumerate(robots):
                for ti, task_cfg in enumerate(tasks):
                    task_type = getattr(task_cfg, "task_type", "task")
                    name = f"{task_type}_s{si}_r{ri}_t{ti}"

                    task = self._build_task_instance(task_cfg, scene, robot)
                    if task is None:
                        continue

                    # Link task to scene
                    if hasattr(task, "link_to_scene"):
                        task.link_to_scene(scene)

                    env = AssembledEnvironment(
                        name=name,
                        scene=scene,
                        embodiment=robot,
                        task=task,
                        task_factory=self.task_factory,
                        task_builder=task_cfg,
                        runtime_cfg=self.runtime_cfg,
                    )
                    self._environments[name] = env
                    environments.append(env)

        manaprint(f"INFO: Combinatorial assembly complete. Total: {len(environments)} environment(s)")
        return environments

    def _build_task_instance(
        self,
        task_cfg: Any,
        scene: Any | None = None,
        robot: Any | None = None,
    ) -> Any | None:
        """Build concrete task. *task_cfg* is a ``ManaTask`` instance from DataBucket.TASKS."""
        if self.task_factory is None:
            manaprint("ERROR: AssemblyFactory has no task_factory bound for finalizing task")
            return None
        task_builder = self._clone_task_builder(task_cfg)
        return self.task_factory.finalize_task(task_builder, scene=scene, robot=robot)

    @staticmethod
    def _clone_task_builder(task_cfg: Any) -> Any:
        origin_generator = getattr(task_cfg, GENERATOR_REF_ATTR, None)
        has_origin_generator = hasattr(task_cfg, GENERATOR_REF_ATTR)
        if has_origin_generator:
            delattr(task_cfg, GENERATOR_REF_ATTR)
        try:
            cloned = copy.deepcopy(task_cfg)
        finally:
            if has_origin_generator:
                setattr(task_cfg, GENERATOR_REF_ATTR, origin_generator)

        if has_origin_generator:
            setattr(cloned, GENERATOR_REF_ATTR, origin_generator)
            deferred_sections = getattr(origin_generator, "_deferred_component_sections", None)
            if isinstance(deferred_sections, dict) and id(task_cfg) in deferred_sections:
                deferred_sections[id(cloned)] = copy.deepcopy(deferred_sections[id(task_cfg)])
        return cloned

    def get_environment(self, name: str) -> AssembledEnvironment | None:
        """Get an assembled environment by name."""
        return self._environments.get(name)

    def fulfill_orders(self, data_center: Any = None, shard: str | None = None, **kwargs) -> list[AssembledEnvironment]:
        """Fulfill assembly orders by delegating to assemble()."""
        data_center = data_center or getattr(self, "data_center", None)
        if shard is not None and "env_name" not in kwargs:
            kwargs["env_name"] = shard
        if data_center is None:
            manaprint("ERROR: data_center is required for AssemblyFactory.fulfill_orders()")
            return []
        return self.assemble(data_center, **kwargs)

    def cleanup(self):
        """Clean up all assembled environments."""
        for env in self._environments.values():
            env.cleanup()

        env_count = len(self._environments)
        self._environments.clear()
        self._orders.clear()
        manaprint(f"INFO: AssemblyFactory cleanup - released {env_count} environment(s)")
