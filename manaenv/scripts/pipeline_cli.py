"""Command-line runtime for public policy evaluation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from manaenv.pipeline.runtime import runtime_flags
from manaenv.pipeline.runtime_cfg import KIT_RENDER_MODE_SYNC, PHYSICS_STEP_MODE_SYNC
from manaenv.pipeline.translator.reflective_translator import expand_includes
from manaenv.utils.debug_print import manaprint
from manaenv.utils.isaac_extensions import (
    app_launcher_cameras_enabled,
    cameras_requested,
    configure_physics_collider_visualization_ui,
    enable_camera_sensor_extension,
    heal_kit_shadowed_site_packages,
    maybe_default_mana_experience,
    maybe_livestream3_experience,
    prepare_camera_kit_args,
    prepare_kit_async_render_args,
    prepare_mana_fabric_transform_kit_args,
    prepare_physics_collider_kit_args,
    skip_kit_window_task_registration,
    strip_window_title_version,
    suppress_simulation_manager_plugin_warnings_below_error,
    validate_kit_render_settings,
    validate_mana_fabric_transform_setting,
)
from manaenv.utils.isaac_runtime_env import resolve_isaac_path
from manaenv.utils.kit_app_runtime import close_kit_app

app_launcher = None
simulation_app = None


@dataclass(frozen=True)
class _LaunchHints:
    robot_cameras: bool = False
    stream_viewports: bool = False
    canvas_resolution: tuple[int, int] | None = None
    enable_colliders: bool = False
    physics_step_mode: str = PHYSICS_STEP_MODE_SYNC
    kit_render_mode: str = KIT_RENDER_MODE_SYNC


def _load_launch_hints(config_path: str | None) -> _LaunchHints:
    if not config_path:
        return _LaunchHints()
    path = Path(config_path).expanduser()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return _LaunchHints()
    if not isinstance(document, dict):
        return _LaunchHints()

    camera_enabled = False
    for entry in document.get("robot_orders") or []:
        if not isinstance(entry, dict):
            continue
        definition = entry.get("robot_generator") or entry
        if isinstance(definition, dict) and bool(definition.get("enable_cameras")):
            camera_enabled = True
            break

    runtime = ((document.get("pipeline") or {}).get("pipeline_runtime") or {})
    if isinstance(runtime, dict):
        try:
            runtime = expand_includes(runtime, path.resolve().parent)
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            pass
    stream = runtime.get("stream_viewports") if isinstance(runtime, dict) else None
    resolution = None
    if isinstance(stream, dict):
        try:
            width = int(stream["canvas_width"])
            height = int(stream["canvas_height"])
            if width > 0 and height > 0:
                resolution = (width, height)
        except (KeyError, TypeError, ValueError):
            pass
    return _LaunchHints(
        robot_cameras=camera_enabled,
        stream_viewports=bool(stream),
        canvas_resolution=resolution,
        enable_colliders=bool(runtime.get("enable_colliders", False)) if isinstance(runtime, dict) else False,
        physics_step_mode=str(runtime.get("physics_step_mode", PHYSICS_STEP_MODE_SYNC))
        if isinstance(runtime, dict)
        else PHYSICS_STEP_MODE_SYNC,
        kit_render_mode=str(runtime.get("kit_render_mode", KIT_RENDER_MODE_SYNC))
        if isinstance(runtime, dict)
        else KIT_RENDER_MODE_SYNC,
    )


def _parse_env_names(raw: str | None) -> set[str] | None:
    names = {item.strip() for item in str(raw or "").split(",") if item.strip()}
    return names or None


def _build_parser() -> Any:
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

    parser = get_isaaclab_arena_cli_parser()
    existing = {action.dest for action in parser._actions}
    if "enable_cameras" not in existing:
        parser.add_argument("--enable_cameras", action="store_true", default=False)
    if "seed" not in existing:
        parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", required=True, help="Evaluation suite YAML")
    parser.add_argument("--env-name", default=None, help="Comma-separated rollout env_ref filter")
    parser.add_argument(
        "--policy-config",
        default=None,
        help="Policy YAML override relative to the evaluation suite directory",
    )
    parser.add_argument("--server-address", default=None, help="Policy endpoint host override")
    parser.add_argument("--server-port", type=int, default=None, help="Policy endpoint port override")
    parser.add_argument("--enable_colliders", action="store_true", default=False)

    for action in parser._actions:
        if action.dest == "presets":
            action.choices = ("physx",)
            action.default = "physx"
            action.help = "Physics backend preset (public runtime: PhysX only)."
        if action.dest == "visualizer":
            action.help = "Visualizer backends as CSV: kit, rerun, viser, or none."
        if action.dest == "livestream" and action.choices is not None and 3 not in action.choices:
            action.choices = [*action.choices, 3]

    args = parser.parse_args()
    if args.policy_config and not _parse_env_names(args.env_name):
        parser.error("--policy-config requires --env-name so it cannot replace the full suite accidentally")
    raw_visualizers = getattr(args, "visualizer", None)
    if isinstance(raw_visualizers, str):
        raw_visualizers = raw_visualizers.split(",")
    visualizers = {str(item).strip().lower() for item in (raw_visualizers or ())}
    if "newton" in visualizers:
        parser.error("the Newton visualizer is not available in the public PhysX runtime")
    hints = _load_launch_hints(args.config)
    args._mana_launch_hints = hints
    args._robot_enable_cameras = bool(getattr(args, "enable_cameras", False))
    if hints.robot_cameras or hints.stream_viewports:
        args.enable_cameras = True
    if hints.enable_colliders:
        args.enable_colliders = True
    if getattr(args, "livestream", -1) in {2, 3} and hints.canvas_resolution:
        args.window_width, args.window_height = hints.canvas_resolution
    return args


def _ensure_isaac_app_env() -> None:
    isaac_path = os.environ.get("ISAAC_PATH") or resolve_isaac_path() or "/isaac-sim"
    os.environ.setdefault("ISAAC_PATH", isaac_path)
    os.environ.setdefault("CARB_APP_PATH", os.path.join(isaac_path, "kit"))
    os.environ.setdefault("EXP_PATH", os.path.join(isaac_path, "apps"))


def _camera_extensions_required(args: Any) -> bool:
    hints = args._mana_launch_hints
    visualizers = getattr(args, "visualizer", None) or []
    return cameras_requested(args) or hints.stream_viewports or "kit" in visualizers


def _bootstrap_isaac(args: Any) -> Any:
    global app_launcher, simulation_app

    _ensure_isaac_app_env()
    runtime_flags.set_value("livestream3", getattr(args, "livestream", -1) == 3)
    skip_kit_window_task_registration()
    if _camera_extensions_required(args):
        prepare_camera_kit_args()
    if args.enable_colliders:
        prepare_physics_collider_kit_args()
    hints = args._mana_launch_hints
    prepare_kit_async_render_args(
        hints.kit_render_mode,
        args,
        physics_step_mode=hints.physics_step_mode,
    )

    from isaaclab.app import AppLauncher

    maybe_livestream3_experience(args)
    mana_experience_active = maybe_default_mana_experience(args)
    prepare_mana_fabric_transform_kit_args(args)
    launcher = AppLauncher(args)
    app = launcher.app
    app_launcher = launcher
    simulation_app = app
    try:
        validate_kit_render_settings(
            hints.kit_render_mode,
            physics_step_mode=hints.physics_step_mode,
        )
        validate_mana_fabric_transform_setting(args)
        if mana_experience_active:
            strip_window_title_version()
            configure_physics_collider_visualization_ui(enable_colliders=args.enable_colliders)
        if app_launcher_cameras_enabled(launcher, args):
            suppress_simulation_manager_plugin_warnings_below_error()
            enable_camera_sensor_extension()
            import carb

            carb.settings.get_settings().set_bool("/isaaclab/render/offscreen", True)
        heal_kit_shadowed_site_packages()
        return app
    except Exception:
        close_kit_app(app, exit_code=1)
        app_launcher = None
        simulation_app = None
        raise


def _explicit_cli_seed(args: Any) -> int | None:
    if not any(item == "--seed" or item.startswith("--seed=") for item in sys.argv[1:]):
        return None
    value = getattr(args, "seed", None)
    return None if value is None else int(value)


def _configure_runtime(args: Any) -> None:
    from manaenv.pipeline.runtime import master_seed as master_seed_runtime

    master_seed_runtime.set_cli_seed_override(_explicit_cli_seed(args))


def _run(args: Any) -> int:
    _bootstrap_isaac(args)
    from manaenv.pipeline.pipeline import ManaPipeline, ManaPipelineCFG

    selected = _parse_env_names(args.env_name)
    pipeline = ManaPipeline()
    try:
        pipeline.setup(
            ManaPipelineCFG(
                description_yaml_path=args.config,
                policy_server_address=args.server_address,
                policy_server_port=args.server_port,
                policy_config_path=args.policy_config,
                enable_cameras=True if args._robot_enable_cameras else None,
            )
        )
        available = {str(order.env_ref) for order in pipeline.rollout_factory._orders}
        missing = sorted((selected or set()) - available)
        if missing:
            manaprint(f"ERROR: unknown evaluation environment(s): {', '.join(missing)}")
            return 2
        pipeline.warmup(env_refs=selected)
        results = pipeline.run_rollout(env_name=selected)
        failures = [item for item in results if item.get("error")]
        manaprint(f"INFO: evaluation complete: {len(results) - len(failures)} passed, {len(failures)} failed")
        return 1 if failures else 0
    except KeyboardInterrupt:
        manaprint("INFO: evaluation interrupted")
        return 130
    except Exception as exc:
        manaprint(f"ERROR: evaluation failed: {exc}")
        return 1
    finally:
        pipeline.cleanup()


def main() -> int:
    args = _build_parser()
    _configure_runtime(args)
    manaprint(f"INFO: evaluation config: {args.config}")
    return _run(args)


__all__ = ["main", "simulation_app"]
