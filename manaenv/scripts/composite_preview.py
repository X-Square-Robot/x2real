"""Stream an assembled environment composite without running a rollout.

Unlike a bare scene preview, this builds the same assembled product used by
evaluation: scene, robot, physics, materials, and camera rigs.  The Kit app is
then pumped without creating a planner, running episodes, collecting data, or
connecting to a policy service.

Run from the repository root, for example::

    python manaenv/scripts/composite_preview.py \
        --config manaenv/defination/x2real.yaml \
        --env-name artixon_arm6a_G_stack_blocks_3_id_env \
        --viz kit

``--env-name`` is the assembly generator to preview.  Omit it to preview the
first assembled environment.  Add ``--livestream 2`` for headless WebRTC;
otherwise ``--viz kit`` opens the local Mana Kit window.
"""

from __future__ import annotations

import argparse
import os
import traceback

from manaenv.pipeline.data_types import output_path
from manaenv.utils.isaac_runtime_env import (
    reexec_with_isaac_runtime_env,
    resolve_isaac_path,
)

if __name__ == "__main__":
    # Run before importing AppLauncher.  A repaired Isaac environment requires
    # a process re-exec, which must never happen during ordinary module import.
    for _isaac_env_note in reexec_with_isaac_runtime_env():
        print(f"[INFO] Isaac runtime env: {_isaac_env_note}", flush=True)

from manaenv.utils.kit_app_runtime import (
    pump_kit_app_frames,
    run_kit_app_loop,
)


def _ensure_isaac_app_env() -> None:
    """Populate the Isaac Sim variables required by AppLauncher."""
    isaac_path = os.environ.get("ISAAC_PATH")
    if not isaac_path:
        isaac_path = resolve_isaac_path() or "/isaac-sim"
        os.environ["ISAAC_PATH"] = isaac_path
    os.environ.setdefault("CARB_APP_PATH", os.path.join(isaac_path, "kit"))
    os.environ.setdefault("EXP_PATH", os.path.join(isaac_path, "apps"))


def _build_composite(config_path: str, env_name: str | None, args):
    """Assemble one environment and build its idle Gym composite."""
    from manaenv.pipeline.pipeline import ManaPipeline, ManaPipelineCFG
    from manaenv.pipeline.registry.bootstrap import bootstrap_discovery

    bootstrap_discovery()

    pipeline = ManaPipeline()
    pipeline.setup(
        ManaPipelineCFG(output_path=output_path(), description_yaml_path=config_path)
    )

    # The public pipeline selects products by explicit environment reference.
    # No rollout order or policy is instantiated by warmup.
    env_refs = {env_name} if env_name else None
    pipeline.warmup(env_refs=env_refs)

    environments = pipeline.get_environments() or []
    if not environments:
        print("WARN: no environments assembled")
        return None

    env = pipeline.get_environment(env_name) if env_name else environments[0]
    if env is None:
        available = ", ".join(str(getattr(item, "name", "?")) for item in environments)
        raise ValueError(
            f"environment {env_name!r} was not assembled; available: {available}"
        )

    print(
        f"INFO: building composite for environment {getattr(env, 'name', '?')!r} of {len(environments)}"
    )
    gym_env = env.build_gym_environment(args=args, mode="raw")
    if gym_env is None:
        print("WARN: build_gym_environment returned None")
        return None
    try:
        gym_env.reset()
    except Exception as exc:  # noqa: BLE001 - keep the initial stage visible
        print(f"WARN: gym_env.reset() failed ({exc}); rendering initial stage anyway")
    return gym_env


def _frame_composite(simulation_app) -> None:
    """Frame the active viewport on the assembled environment."""
    if not pump_kit_app_frames(simulation_app, 90):
        return
    try:
        import omni.usd
        from omni.kit.viewport.utility import (
            frame_viewport_selection,
            get_active_viewport,
        )

        stage = omni.usd.get_context().get_stage()
        target = None
        for path in ("/World/envs/env_0", "/World/envs", "/World"):
            if stage and stage.GetPrimAtPath(path).IsValid():
                target = path
                break
        if target:
            omni.usd.get_context().get_selection().set_selected_prim_paths(
                [target], True
            )
            if not pump_kit_app_frames(simulation_app, 3):
                return
            frame_viewport_selection(get_active_viewport())
            print(f"INFO: framed viewport on {target}")
    except Exception as exc:  # noqa: BLE001 - framing is best effort
        print(f"WARN: could not frame composite viewport: {exc}")


def _capture_snapshot(path: str) -> None:
    """Write the active viewport to *path* as a best-effort PNG snapshot."""
    try:
        from omni.kit.viewport.utility import (
            capture_viewport_to_file,
            get_active_viewport,
        )

        viewport = get_active_viewport()
        if viewport is None:
            return
        temporary_path = path + ".tmp.png"
        capture = capture_viewport_to_file(viewport, temporary_path)

        def _promote(*_args, **_kwargs) -> None:
            try:
                if os.path.exists(temporary_path):
                    os.replace(temporary_path, path)
            except OSError:
                pass

        if hasattr(capture, "set_completion_frame_callback"):
            try:
                capture.set_completion_frame_callback(_promote)
                return
            except Exception:  # noqa: BLE001 - fall back to immediate promotion
                _promote()
                return
        _promote()
    except Exception:  # noqa: BLE001 - snapshots must not stop preview rendering
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview an assembled environment without a policy rollout"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the pipeline YAML"
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=None,
        help="assembly generator name to preview; omit to preview the first assembled environment",
    )
    parser.add_argument(
        "--snapshot-path",
        type=str,
        default=None,
        help="periodically write the active viewport to this PNG path",
    )

    # AppLauncher starts Kit before its deferred device selection imports
    # Torch.  Preload the Conda build so Kit's bundled CUDA runtime cannot win
    # symbol resolution and make the later pipeline import fail.
    __import__("torch")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Respect the project-wide AppLauncher display contract: --viz kit opens a
    # local window, while an explicit --livestream 1/2 enables headless WebRTC.
    args.enable_cameras = True
    args.device = "cpu"
    args.disable_fabric = True

    _ensure_isaac_app_env()

    from manaenv.utils.isaac_extensions import (
        enable_camera_sensor_extension,
        heal_kit_shadowed_site_packages,
        maybe_default_mana_experience,
        prepare_camera_kit_args,
        prepare_mana_fabric_transform_kit_args,
        skip_kit_window_task_registration,
        strip_window_title_version,
        validate_mana_fabric_transform_setting,
    )

    skip_kit_window_task_registration()
    prepare_camera_kit_args()
    mana_experience_active = maybe_default_mana_experience(args)
    prepare_mana_fabric_transform_kit_args(args)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    validate_mana_fabric_transform_setting(args)
    if mana_experience_active:
        strip_window_title_version()
    enable_camera_sensor_extension()
    heal_kit_shadowed_site_packages()

    try:
        gym_env = _build_composite(args.config, args.env_name, args)
        if gym_env is not None:
            _frame_composite(simulation_app)
    except Exception as exc:  # noqa: BLE001 - retain the stream for diagnostics
        print(f"ERROR: failed to build composite preview: {exc}")
        traceback.print_exc()
        print("INFO: continuing to stream the current stage")

    snapshot_path = getattr(args, "snapshot_path", None)
    frame = 0

    def _after_update() -> None:
        nonlocal frame
        frame += 1
        if snapshot_path and frame % 30 == 0:
            _capture_snapshot(snapshot_path)

    try:
        run_kit_app_loop(simulation_app, after_update=_after_update)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
