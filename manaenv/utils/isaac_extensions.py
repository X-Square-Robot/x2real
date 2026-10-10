"""Helpers for Isaac Sim extension wiring around AppLauncher.

The OpenCV lens-distortion USD schema (``OmniLensDistortionOpenCvFisheyeAPI``
and friends) is registered by the ``omni.usd.schema.omni_lens_distortion``
extension whose ``extension.toml`` declares ``order = -100`` — it must be
loaded *before* USD finalises its plugin registry. The Isaac Lab headless
rendering kit experience does not list it as a dependency, so we have to
inject Kit ``--enable`` flags into ``sys.argv`` *before* ``AppLauncher``
instantiates ``SimulationApp``. Enabling it later via
``set_extension_enabled_immediate`` only loads the Python module: it does
**not** retroactively register the USD schema, which causes
``prim.ApplyAPI("OmniLensDistortionOpenCvFisheyeAPI")`` to fail with
"Cannot find a valid schema" the first time a fisheye camera prim is
authored on the stage.

Camera-enabled entry points must therefore call
:func:`prepare_camera_kit_args` *before* constructing ``AppLauncher`` and
:func:`enable_camera_sensor_extension` *after* ``AppLauncher`` has booted
Kit (the post-boot enable is a best-effort no-op when the kit args path
already wired the extensions in). IsaacSim 6 moved camera APIs to the
experimental RTX extension, so do not force-enable the removed legacy
``isaacsim.sensors.camera`` extension during Kit startup.

:func:`prepare_camera_kit_args`, ``AppLauncher`` return sites (when cameras are
on), and the start of :func:`enable_camera_sensor_extension` call
:func:`suppress_simulation_manager_plugin_warnings_below_error` so the Carbonite
threshold is applied before/after Kit may reset it. That raises the log
threshold for ``isaacsim.core.simulation_manager.plugin`` to ``ERROR``, which
removes the noisy benign ``No adjacent samples found for interpolation`` WARN
(emitted from native code, not Python ``sys.stderr``). Carbonite has no
per-message filter; any other ``WARN`` from that same plugin source is
suppressed as well while ``ERROR``/``FATAL`` still print.
"""

from __future__ import annotations

import builtins
import importlib
import os
import shlex
import sys
import sysconfig
from pathlib import Path

from manaenv.pipeline.runtime_cfg import (
    KIT_RENDER_MODE_ASYNC_HANDSHAKE,
    KIT_RENDER_MODE_SYNC,
    KIT_RENDER_MODES,
    PHYSICS_STEP_MODE_SYNC,
    PHYSICS_STEP_MODES,
)
from manaenv.utils.debug_print import manaprint

CAMERA_SENSOR_EXTENSION = "isaacsim.sensors.experimental.rtx"
LEGACY_CAMERA_SENSOR_EXTENSION = "isaacsim.sensors.camera"
LENS_DISTORTION_SCHEMA_EXTENSION = "omni.usd.schema.omni_lens_distortion"
PHYSICS_COLLIDER_UI_EXTENSION = "omni.physx.ui"
PHYSICS_COLLIDER_VISUALIZATION_SETTING = "/persistent/physics/visualizationDisplayColliders"
PHYSICS_COLLIDER_VISUALIZATION_UI_SETTING = "/manaenv/physics/colliderVisualizationUiEnabled"
FABRIC_RENDER_DELEGATE_TRANSFORM_SETTING = "/rtx/hydra/readTransformsFromFabricInRenderDelegate"

# Mana rendering experiences for windowed, offscreen-camera, and WebRTC runs.
MANA_EXPERIENCE_FILE = Path(__file__).resolve().parents[1] / "experience" / "mana.kit"
MANA_HEADLESS_EXPERIENCE_FILE = Path(__file__).resolve().parents[1] / "experience" / "mana.headless.kit"
WEBRTC_LIVESTREAM_EXPERIENCE_FILE = Path(__file__).resolve().parents[1] / "experience" / "mana.webrtc.kit"
LIVESTREAM3_STOCK_EXPERIENCE_ENV = "MANAENV_LIVESTREAM3_STOCK_EXPERIENCE"

SIMULATION_MANAGER_PLUGIN_LOG_SOURCE = "isaacsim.core.simulation_manager.plugin"

_REQUIRED_KIT_EXTENSIONS = (
    LENS_DISTORTION_SCHEMA_EXTENSION,
    CAMERA_SENSOR_EXTENSION,
)


def suppress_simulation_manager_plugin_warnings_below_error() -> None:
    """Suppress ``LEVEL_WARN`` from ``isaacsim.core.simulation_manager.plugin`` via Carb.

    Native CARB_LOG_* does not go through Python ``sys.stdout``/``sys.stderr``,
    so stdio wrapping cannot remove these lines. Carb only supports
    per-source thresholds: this hides all warnings from that plugin (mainly
    the interpolation spam) while keeping errors. Safe to call repeatedly
    (e.g. before and after ``AppLauncher``) in case Kit resets thresholds.
    """
    try:
        import carb.logging as carb_log
    except ModuleNotFoundError:
        return
    try:
        carb_log.acquire_logging().set_level_threshold_for_source(
            SIMULATION_MANAGER_PLUGIN_LOG_SOURCE,
            carb_log.LogSettingBehavior.OVERRIDE,
            carb_log.LEVEL_ERROR,
        )
    except Exception:
        pass


def prepare_camera_kit_args() -> None:
    """Inject ``--enable <ext>`` flags into ``sys.argv`` for Kit startup.

    Idempotent: skips an extension whose ``--enable`` pair already exists
    in ``sys.argv``. Safe to call multiple times across re-bootstrap or
    nested launchers. Must run *before* ``AppLauncher(...)`` so Kit
    consumes the flags during ``SimulationApp`` initialisation; calling
    this after Kit has started will not retroactively register USD
    schemas.
    """
    for ext in _REQUIRED_KIT_EXTENSIONS:
        already_present = any(
            sys.argv[idx] == "--enable" and idx + 1 < len(sys.argv) and sys.argv[idx + 1] == ext
            for idx in range(len(sys.argv) - 1)
        )
        if already_present:
            continue
        sys.argv += ["--enable", ext]
    suppress_simulation_manager_plugin_warnings_below_error()


def prepare_physics_collider_kit_args() -> None:
    """Load the PhysX collider UI only for an explicit collider request."""
    already_present = any(
        sys.argv[idx] == "--enable" and idx + 1 < len(sys.argv) and sys.argv[idx + 1] == PHYSICS_COLLIDER_UI_EXTENSION
        for idx in range(len(sys.argv) - 1)
    )
    if not already_present:
        sys.argv += ["--enable", PHYSICS_COLLIDER_UI_EXTENSION]


def _kit_setting_argument(path: str, value: bool) -> str:
    return f"--{path}={'true' if value else 'false'}"


def _declared_kit_arguments(args: object | None) -> list[str]:
    """Return direct and ``--kit_args`` arguments visible before AppLauncher."""
    declared = list(sys.argv)
    kit_args = getattr(args, "kit_args", "") if args is not None else ""
    if isinstance(kit_args, str) and kit_args:
        declared.extend(shlex.split(kit_args))
    return declared


def _mana_rendering_experience_selected(args: object) -> bool:
    experience = getattr(args, "experience", "")
    if not experience:
        return False
    selected_path = Path(str(experience)).expanduser().resolve()
    mana_experience_paths = {
        MANA_EXPERIENCE_FILE.resolve(),
        MANA_HEADLESS_EXPERIENCE_FILE.resolve(),
        WEBRTC_LIVESTREAM_EXPERIENCE_FILE.resolve(),
    }
    return selected_path in mana_experience_paths


def _async_handshake_kit_settings(physics_step_mode: str) -> tuple[tuple[str, bool], ...]:
    """Return the Kit settings for the environment adapter that owns one step."""
    if physics_step_mode not in PHYSICS_STEP_MODES:
        raise ValueError(f"Unsupported physics_step_mode: {physics_step_mode!r}")

    sync_physics = physics_step_mode == PHYSICS_STEP_MODE_SYNC
    return (
        ("/app/asyncRendering", True),
        ("/app/asyncRenderingLowLatency", sync_physics),
        ("/app/omni.usd/asyncHandshake", True),
        ("/renderer/asyncInit", False),
        ("/rtx/hydra/supportMultiTickRate", False),
        ("/rtx/rendering/perSensorTickTlas", False),
        # Stock cameras need Replicator to preserve app-level async rendering.
        # The Mana worker instead owns completion through explicit camera tickets.
        ("/omni/replicator/asyncRendering", sync_physics),
        *((("/exts/isaacsim.core.throttling/enable_async", False),) if sync_physics else ()),
        ("/app/renderer/waitIdle", False),
        ("/app/hydraEngine/waitIdle", False),
    )


def prepare_kit_async_render_args(
    mode: str,
    args: object | None = None,
    *,
    physics_step_mode: str = PHYSICS_STEP_MODE_SYNC,
) -> None:
    """Inject the launch-time Kit settings for the native async handshake mode.

    The settings must be present before ``AppLauncher`` creates SimulationApp.
    Existing contradictory Kit arguments are rejected so the YAML mode cannot
    silently start with a different render scheduler.
    """
    if mode not in KIT_RENDER_MODES:
        raise ValueError(f"Unsupported kit_render_mode: {mode!r}")
    if mode == KIT_RENDER_MODE_SYNC:
        return
    assert mode == KIT_RENDER_MODE_ASYNC_HANDSHAKE

    declared = _declared_kit_arguments(args)
    for path, value in _async_handshake_kit_settings(physics_step_mode):
        argument = _kit_setting_argument(path, value)
        prefix = f"--{path}="
        existing = [candidate for candidate in declared if candidate.startswith(prefix)]
        if existing:
            if any(candidate != argument for candidate in existing):
                raise ValueError(
                    f"kit_render_mode='async_handshake' conflicts with explicit Kit setting {existing[-1]!r}"
                )
            continue
        sys.argv.append(argument)
        declared.append(argument)


def validate_kit_render_settings(
    mode: str,
    *,
    physics_step_mode: str = PHYSICS_STEP_MODE_SYNC,
) -> None:
    """Verify that SimulationApp resolved every requested async render setting."""
    if mode == KIT_RENDER_MODE_SYNC:
        return
    if mode != KIT_RENDER_MODE_ASYNC_HANDSHAKE:
        raise ValueError(f"Unsupported kit_render_mode: {mode!r}")

    import carb

    settings = carb.settings.get_settings()
    expected_settings = _async_handshake_kit_settings(physics_step_mode)
    mismatches = [
        f"{path}={settings.get(path)!r} (expected {expected!r})"
        for path, expected in expected_settings
        if settings.get(path) != expected
    ]
    if mismatches:
        raise RuntimeError("Kit async render settings were not applied: " + ", ".join(mismatches))


def prepare_mana_fabric_transform_kit_args(args: object) -> bool:
    """Disable incompatible direct Fabric transform reads before RTX starts."""
    if not _mana_rendering_experience_selected(args):
        return False

    argument = _kit_setting_argument(FABRIC_RENDER_DELEGATE_TRANSFORM_SETTING, False)
    prefix = f"--{FABRIC_RENDER_DELEGATE_TRANSFORM_SETTING}="
    existing = [candidate for candidate in _declared_kit_arguments(args) if candidate.startswith(prefix)]
    if existing:
        if any(candidate != argument for candidate in existing):
            raise ValueError(
                "Mana rendering experience disables direct Fabric transform reads but conflicts with "
                f"{existing[-1]!r}"
            )
        return True
    sys.argv.append(argument)
    return True


def validate_mana_fabric_transform_setting(args: object) -> None:
    """Fail after Kit startup if direct RenderDelegate transform reads remain on."""
    if not _mana_rendering_experience_selected(args):
        return

    import carb

    resolved_value = carb.settings.get_settings().get(FABRIC_RENDER_DELEGATE_TRANSFORM_SETTING)
    if resolved_value is not False:
        raise RuntimeError(
            "Mana rendering experience direct Fabric transform reads remain enabled after startup: "
            f"{FABRIC_RENDER_DELEGATE_TRANSFORM_SETTING}={resolved_value!r}"
        )


def _requested_livestream(args: object) -> int:
    """Resolve the pre-AppLauncher livestream value from CLI or environment."""
    value = getattr(args, "livestream", -1)
    if value is None or int(value) < 0:
        value = os.environ.get("LIVESTREAM", "0")
    return int(value)


def _requested_visualizers(args: object) -> set[str]:
    """Normalize the visualizer names visible before AppLauncher starts."""
    value = getattr(args, "visualizer", None)
    if value is None:
        return set()
    if isinstance(value, str):
        value = value.split(",")
    return {str(item).strip().lower() for item in value if str(item).strip()}


def _kit_visualizer_requested(args: object) -> bool:
    """Return whether CLI or config intent requests a windowed Kit visualizer."""
    raw_visualizers = getattr(args, "visualizer", None)
    visualizer_explicit = bool(getattr(args, "visualizer_explicit", False)) or raw_visualizers is not None
    if visualizer_explicit:
        return "kit" in _requested_visualizers(args)

    visualizer_intent = getattr(args, "visualizer_intent", None)
    return isinstance(visualizer_intent, dict) and bool(visualizer_intent.get("has_kit_visualizer", False))


def _headless_without_livestream_requested(args: object, livestream: int) -> bool:
    """Mirror AppLauncher's headless intent closely enough to choose an experience."""
    if livestream != 0 or bool(getattr(args, "xr", False)):
        return False
    if bool(getattr(args, "headless", False)) or os.environ.get("HEADLESS", "0") == "1":
        return True
    # AppLauncher runs headless when neither CLI nor config requests Kit.
    return not _kit_visualizer_requested(args)


def maybe_default_mana_experience(args) -> bool:
    """Select the Mana experience matching the requested display mode.

    Explicit ``--experience`` always wins. Livestream 2 uses ``mana.webrtc``;
    the legacy Livestream-3 alias is rewritten to mode 2 before AppLauncher
    validates it. Camera-enabled headless runs use ``mana.headless``, while a
    windowed Kit visualizer uses ``mana.kit``.
    """
    livestream = _requested_livestream(args)
    legacy_livestream3 = livestream == 3
    if legacy_livestream3:
        args.livestream = 2
        livestream = 2

    if getattr(args, "experience", ""):
        return _mana_rendering_experience_selected(args)

    stock_livestream3_already_selected = bool(getattr(args, "_mana_livestream3_stock_experience", False))
    use_stock_livestream3 = stock_livestream3_already_selected or (
        legacy_livestream3
        and os.environ.get(LIVESTREAM3_STOCK_EXPERIENCE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    )
    if use_stock_livestream3:
        args._mana_livestream3_stock_experience = True
        if not stock_livestream3_already_selected:
            manaprint("INFO: Livestream-3 using stock WebRTC experience with camera-only window filtering")
        return False

    selected: Path | None = None
    description = ""
    headless_without_livestream = _headless_without_livestream_requested(args, livestream)
    if livestream == 2 and WEBRTC_LIVESTREAM_EXPERIENCE_FILE.is_file():
        selected = WEBRTC_LIVESTREAM_EXPERIENCE_FILE
        description = "Mana WebRTC"
    elif cameras_requested(args) and headless_without_livestream and MANA_HEADLESS_EXPERIENCE_FILE.is_file():
        selected = MANA_HEADLESS_EXPERIENCE_FILE
        description = "Mana headless camera"
    elif (
        livestream == 0
        and not bool(getattr(args, "xr", False))
        and not headless_without_livestream
        and _kit_visualizer_requested(args)
        and MANA_EXPERIENCE_FILE.is_file()
    ):
        selected = MANA_EXPERIENCE_FILE
        description = "Mana windowed"

    if selected is not None:
        args.experience = str(selected)
        manaprint(f"INFO: Using {description} experience: {selected}")
    return _mana_rendering_experience_selected(args)


def maybe_livestream3_experience(args) -> bool:
    """Preserve the former Livestream-3 rewrite helper for external callers."""
    if _requested_livestream(args) != 3:
        return False
    maybe_default_mana_experience(args)
    return True


def configure_physics_collider_visualization_ui(*, enable_colliders: bool = False) -> None:
    """Apply the explicit PhysX collider-visualization launch option.

    The Mana experience defaults both settings off, but an existing
    ``user.config.json`` has higher precedence. Call this after ``AppLauncher``
    so the launch value resolved from CLI/YAML wins deterministically. Display
    mode ``2`` means all colliders, matching Isaac Sim's existing helper.
    """
    import carb

    settings = carb.settings.get_settings()
    enabled = bool(enable_colliders)
    # The stable marker keeps async scheduling synchronous even when Replicator
    # temporarily clears the mutable visualization setting during capture.
    settings.set_bool(PHYSICS_COLLIDER_VISUALIZATION_UI_SETTING, enabled)
    settings.set_int(PHYSICS_COLLIDER_VISUALIZATION_SETTING, 2 if enabled else 0)
    if enabled:
        manaprint("INFO: PhysX collider display enabled by enable_colliders")


def strip_window_title_version() -> None:
    """Drop the version suffix Kit appends to the OS window title.

    Kit's native layer composes the title as ``/app/window/title`` + app
    version at window creation, so this must run *after* ``AppLauncher(...)``
    to overwrite it. Uses ``omni.kit.window.title`` when present (keeps its
    stage-suffix behaviour), else rewrites the OS title via ``carb.windowing``.
    """
    try:
        from omni.kit.window.title import get_main_window_title

        window_title = get_main_window_title()
        if window_title:
            window_title.set_app_version("")
            return
    except ImportError:
        pass  # extension not loaded in trimmed/headless experiences
    try:
        import carb.settings
        import carb.windowing
        import omni.appwindow

        window = omni.appwindow.get_default_app_window().get_window()
        if window:
            title = carb.settings.get_settings().get("/app/window/title") or "Mana"
            carb.windowing.acquire_windowing_interface().set_window_title(window, title)
    except Exception:
        pass  # headless: no OS window to retitle


def enable_camera_sensor_extension() -> None:
    """Best-effort post-launch enable for contexts that bypassed kit args.

    Useful for notebooks / tests that build ``AppLauncher`` indirectly and
    cannot mutate ``sys.argv`` beforehand. Loads the Python modules but
    cannot retroactively register USD schemas — always pair with
    :func:`prepare_camera_kit_args` when controlling the launch path.

    The Kit extension manager's ``set_extension_enabled_immediate`` is the
    primary path. ``enable_extension`` from
    ``isaacsim.core.experimental.utils.app`` is only a thin wrapper over the
    same API (the IsaacSim 6.0 replacement for the deprecated
    ``isaacsim.core.utils.extensions``), so its import is kept optional:
    when that module is not loaded it is simply skipped rather than
    aborting the launch.
    """
    suppress_simulation_manager_plugin_warnings_below_error()
    import omni.kit.app

    try:
        from isaacsim.core.experimental.utils.app import enable_extension
    except ModuleNotFoundError:
        enable_extension = None

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    for ext in _REQUIRED_KIT_EXTENSIONS:
        try:
            ext_manager.set_extension_enabled_immediate(ext, True)
        except Exception:
            pass
        if enable_extension is not None:
            try:
                enable_extension(ext)
            except Exception:
                pass


def cameras_requested(args_cli) -> bool:
    """Return whether camera rendering was requested before AppLauncher boots.

    Safe to call before ``AppLauncher`` is constructed (no Kit imports).
    Inspects the parsed CLI namespace and falls back to the
    ``ENABLE_CAMERAS=1`` environment switch.
    """
    if bool(getattr(args_cli, "enable_cameras", False)):
        return True
    return os.environ.get("ENABLE_CAMERAS", "0") == "1"


def app_launcher_cameras_enabled(app_launcher, args_cli=None) -> bool:
    """Return whether an AppLauncher resolved camera rendering as enabled."""
    if hasattr(app_launcher, "_enable_cameras"):
        return bool(app_launcher._enable_cameras)
    return cameras_requested(args_cli) if args_cli is not None else False


def skip_kit_window_task_registration() -> None:
    """Pre-set the ``isaaclab_tasks`` registration guard before Kit boots.

    ``AppLauncher._create_app()`` hides every ``.*lab.*`` module from
    ``sys.modules`` while ``SimulationApp`` starts; Kit extensions then import
    ``isaaclab_tasks`` fresh, whose ``import_packages()`` walk registers ~190
    sample gym envs manaenv never uses. Those Kit-window module copies are
    purged right after boot (to avoid dual-world cfg mixing), so any later
    organic import of a task subpackage re-executes its ``__init__`` and
    re-registers, spamming gymnasium "Overriding environment" warnings.
    Setting the ``builtins`` flag that ``isaaclab_tasks/__init__.py`` checks
    turns the Kit-window walk into a no-op; subpackages then register lazily,
    exactly once, on their first real import. Must run before
    ``AppLauncher(...)``.
    """
    builtins._isaaclab_tasks_registered = True


def heal_kit_shadowed_site_packages() -> None:
    """Re-pin ``warp`` / ``newton`` to the conda env after Kit extension load.

    IsaacSim 6.0 enables ``omni.warp.core`` (bundled warp 1.13, declared as
    python module ``warp``) and ``isaacsim.pip.newton`` (a ``pip_prebundle``
    root carrying upstream newton + deps) whenever the Newton actuator graph
    is used — which now happens even headless (``sim.use_newton_actuators``).
    Both shadow the env's site-packages stack (the pinned Warp 1.16 nightly and
    the Newton fork the pipeline is built against) through Kit's ``FastFinder``: a
    ``sys.meta_path[0]`` finder that resolves module *names* from its own
    extension-scan cache, ignoring ``sys.path`` order and the parent package's
    ``__path__``. That can serve not-yet-imported Warp submodules from
    extscache 1.13 even while ``warp`` itself is the site-packages 1.16
    module — the 1.13 code then hits cached 1.16 siblings and dies with
    ``cannot import name 'warn' from 'warp._src.utils'``.

    Heal by deregistering the shadow roots from the fast importer (path
    reordering cannot win against a meta_path finder), scrubbing any plain
    ``sys.path`` copies, purging foreign module objects, and eagerly importing
    the site-packages copies so every later ``import`` resolves from
    ``sys.modules``. Healing is scoped to packages that *actually* exist in
    site-packages: a package that ships solely via a Kit prebundle (e.g.
    ``newton`` on the IsaacSim 6.0 serving image, which has no site-packages
    copy) has nothing to re-pin to, so its Kit root is left registered and it
    keeps resolving from Kit. When nothing is shadowed this is a cheap no-op, so
    it is safe to call unconditionally right after ``AppLauncher`` boots on every
    entry point (serial and parallel SimWorker).
    """
    site_packages = sysconfig.get_paths()["purelib"]
    # Only packages used by the public PhysX/StablePD path are considered here;
    # packages that belonged solely to removed physics solvers are not re-pinned.
    pkgs = tuple(pkg for pkg in ("warp", "newton") if os.path.isdir(os.path.join(site_packages, pkg)))

    # typing_extensions (single-module): Kit prebundle copies patch typing in a way
    # that rejects torch>=2.10 generics; re-pin even when not yet imported.
    te_path = os.path.join(site_packages, "typing_extensions.py")
    te_mod = sys.modules.get("typing_extensions")
    te_foreign = os.path.isfile(te_path) and getattr(te_mod, "__file__", None) != te_path

    def _is_shadow_root(path: str) -> bool:
        # Exactly the two offenders, and only when there is a site-packages copy
        # to re-pin to. On images where e.g. ``newton`` ships *only* via Kit's
        # ``isaacsim.pip.newton`` prebundle (not in site-packages), that root is
        # the module's sole source — deregistering it would leave ``newton``
        # unimportable. Keep this narrow: a bare "pip_prebundle" match also
        # deregisters omni.kit.pip_archive-style roots and breaks their consumers
        # (e.g. aiohttp for Kit tagging extensions).
        if "warp" in pkgs and "omni.warp.core-" in path:
            return True
        if "newton" in pkgs and "isaacsim.pip.newton" in path and "pip_prebundle" in path:
            return True
        return False

    # 1) Deregister the shadow roots from Kit's fast importer and drop their
    #    cached module-name claims for our packages.
    removed_roots: list[str] = []
    try:
        from omni.ext._impl import fast_importer as _fast_importer

        removed_roots = [path for path in list(getattr(_fast_importer, "_sys_paths", {})) if _is_shadow_root(path)]
        for path in removed_roots:
            _fast_importer.remove_sys_path(path)
        cache = getattr(_fast_importer, "_cache", None)
        if removed_roots and cache is not None:
            for table in (getattr(cache, "modules", {}), getattr(cache, "namespaces", {})):
                for name in [n for n in table if n.split(".", 1)[0] in pkgs]:
                    del table[name]
        # Drop only the name claim; pip_archive roots stay (they also serve aiohttp).
        if te_foreign and cache is not None:
            for table in (getattr(cache, "modules", {}), getattr(cache, "namespaces", {})):
                table.pop("typing_extensions", None)
    except Exception as exc:  # noqa: BLE001 — healing must never kill the launch
        manaprint(f"WARNING: Kit fast-importer deregistration skipped: {exc}")

    # 2) Belt-and-suspenders: scrub plain sys.path copies of the same roots
    #    (covers profiles that run with the fast importer disabled).
    scrubbed = [p for p in sys.path if _is_shadow_root(p)]
    if scrubbed:
        sys.path[:] = [p for p in sys.path if p not in scrubbed]

    def _foreign(name: str) -> bool:
        mod = sys.modules.get(name)
        fname = getattr(mod, "__file__", None)
        return mod is not None and fname is not None and not fname.startswith(site_packages)

    foreign = [n for n in sys.modules if n.split(".", 1)[0] in pkgs and _foreign(n)]
    if not (removed_roots or scrubbed or foreign or te_foreign):
        return  # nothing shadowed: keep boot cheap

    # 3) Purge foreign module objects and eagerly import the env copies, so all
    #    later imports by the StablePD adapter (``newton`` and ``warp``)
    #    hit sys.modules first and the finder is never consulted again.
    for name in foreign:
        del sys.modules[name]
    if te_foreign:
        # re-exec re-installs the torch-compatible typing patch
        sys.modules.pop("typing_extensions", None)
    importlib.invalidate_caches()
    # Only eagerly re-pin packages that actually live in site-packages. A package
    # absent from site-packages (e.g. ``newton`` when it ships solely via Kit's
    # isaacsim.pip.newton prebundle) has nothing to re-pin to and must keep
    # resolving from Kit, so it is deliberately left out here.
    eager: list[str] = []
    if te_foreign:
        eager.append("typing_extensions")
    if "warp" in pkgs:
        eager.append("warp")
    if "newton" in pkgs:
        eager.append("newton")
    for mod_name in eager:
        importlib.import_module(mod_name)
        resolved = getattr(sys.modules[mod_name], "__file__", "") or ""
        if not resolved.startswith(site_packages):
            manaprint(f"WARNING: '{mod_name}' still resolves outside site-packages: {resolved}")
    manaprint(
        "INFO: Re-pinned warp/newton to conda site-packages "
        f"(deregistered {len(removed_roots)} Kit shadow root(s), purged {len(foreign)} split module(s))"
    )
