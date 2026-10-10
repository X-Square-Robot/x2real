"""Teleop wrist-camera tip crosshair overlay (viewport HUD).

Draws a one-shot crosshair on the Kit wrist-camera sensor previews at the
projected inner-jaw contact point used for closed-gripper aiming. Display-only;
never touches recorded observations.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

from manaenv.utils.debug_print import manaprint
from manaenv.utils.tensor_utils import as_torch

_SIDES = ("left", "right")
_CROSSHAIR_EXT_ID = "manaenv.tip_crosshair"
_DEFAULT_CROSSHAIR_NORM = (0.48, 0.52)
_DEFAULT_COLOR = 0xFFFF3333
_DEFAULT_SIZE_PX = 12
_DEFAULT_THICKNESS_PX = 2
_DEFAULT_STYLE = "cross"
_DEFAULT_ENV_REGEX = "/World/envs/env_0"
_EX001_6R_WRIST_CAMERA_FALLBACK: dict[str, str] = {
    "left": f"{_DEFAULT_ENV_REGEX}/Robot/left_arm_gripper_camera_color_frame/Left_Gripper_Camera",
    "right": f"{_DEFAULT_ENV_REGEX}/Robot/right_arm_gripper_camera_color_frame/Right_Gripper_Camera",
}
# v00.05.05 G-jaw geometry, expressed in the gripper-base frame. The active
# finger origin is x=0.077 m and the highlighted inner-nose section is
# x=0.120 m in that finger frame. Symmetry puts the closed-jaw aim point on the
# centerline; no display-pixel calibration participates in this definition.
_EX001_6R_G_FINGER_ORIGIN_BASE_X_M = 0.077
_EX001_6R_G_INNER_NOSE_SECTION_FINGER_X_M = 0.120
_EX001_6R_G_CLOSED_CONTACT_BASE = (
    _EX001_6R_G_FINGER_ORIGIN_BASE_X_M + _EX001_6R_G_INNER_NOSE_SECTION_FINGER_X_M,
    0.0,
    -0.01,
)

# Rigid v00.05.05 wrist-camera extrinsic relative to the gripper base. The
# origin comes from the USD Xforms and the pitch terms from the camera's
# orthonormal basis. Keeping this separate from the target geometry makes the
# projection pipeline explicit: geometry -> extrinsic -> intrinsic -> pixels.
_EX001_6R_WRIST_CAMERA_ORIGIN_BASE = (0.1034766137599945, 0.0, 0.0289229154586792)
_EX001_6R_WRIST_CAMERA_PITCH_SIN = 0.20791526107419317
_EX001_6R_WRIST_CAMERA_PITCH_COS = 0.9781468418391115


def tip_crosshair_enabled() -> bool:
    raw = os.environ.get("MANAENV_EX001_6R_TIP_CROSSHAIR", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _is_h_jaw_robot(gym_env: Any) -> bool:
    """True when the live robot USD is the legacy H-jaw asset."""
    try:
        scene = getattr(gym_env, "scene", None)
        robot = scene["robot"] if scene is not None else None
        spawn = getattr(getattr(robot, "cfg", None), "spawn", None)
        usd = str(getattr(spawn, "usd_path", "") or "")
    except Exception:
        return False
    return "ex001_6r_legacy" in usd


def _parse_env_norm(raw: str | None) -> tuple[float, float] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        nx, ny = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return (max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)))


def tip_crosshair_norm_override(side: str) -> tuple[float, float] | None:
    key = f"MANAENV_EX001_6R_TIP_CROSSHAIR_{side.upper()}"
    return _parse_env_norm(os.environ.get(key))


def crosshair_style_from_env() -> dict[str, Any]:
    color_raw = os.environ.get("MANAENV_EX001_6R_TIP_CROSSHAIR_COLOR", "").strip()
    color = int(color_raw, 0) if color_raw else _DEFAULT_COLOR
    size_px = int(os.environ.get("MANAENV_EX001_6R_TIP_CROSSHAIR_SIZE_PX", str(_DEFAULT_SIZE_PX)))
    thickness_px = int(os.environ.get("MANAENV_EX001_6R_TIP_CROSSHAIR_THICKNESS_PX", str(_DEFAULT_THICKNESS_PX)))
    style_raw = os.environ.get("MANAENV_EX001_6R_TIP_CROSSHAIR_STYLE", _DEFAULT_STYLE).strip().lower()
    style = "cross" if style_raw == "cross" else "dot"
    return {
        "color": color,
        "size_px": max(2, size_px),
        "thickness_px": max(1, thickness_px),
        "style": style,
    }


def map_image_norm_to_frame_px(
    norm_xy: tuple[float, float],
    *,
    image_size: tuple[int, int],
    frame_size: tuple[float, float],
) -> tuple[float, float]:
    """Map normalized image coordinates into a centered aspect-fit UI frame."""
    nx, ny = norm_xy
    image_width, image_height = image_size
    frame_width, frame_height = frame_size
    if image_width <= 0 or image_height <= 0 or frame_width <= 0 or frame_height <= 0:
        return float(nx) * frame_width, float(ny) * frame_height

    scale = min(frame_width / image_width, frame_height / image_height)
    display_width = image_width * scale
    display_height = image_height * scale
    left = (frame_width - display_width) * 0.5
    top = (frame_height - display_height) * 0.5
    return left + float(nx) * display_width, top + float(ny) * display_height


def resolve_wrist_camera_prim_paths(
    camera_bundle: Any = None,
    *,
    gym_env: Any = None,
    env_regex: str = _DEFAULT_ENV_REGEX,
) -> list[str]:
    paths: list[str] = []
    for side in _SIDES:
        resolved = _resolve_single_wrist_camera_path(
            side,
            camera_bundle=camera_bundle,
            gym_env=gym_env,
            env_regex=env_regex,
        )
        if resolved:
            paths.append(resolved)
    return paths


def _resolve_single_wrist_camera_path(
    side: str,
    *,
    camera_bundle: Any = None,
    gym_env: Any = None,
    env_regex: str = _DEFAULT_ENV_REGEX,
) -> str | None:
    attr_name = f"{side}_wrist_camera"
    if camera_bundle is not None:
        cam = getattr(camera_bundle, attr_name, None)
        prim_path = getattr(cam, "prim_path", None) if cam is not None else None
        if prim_path:
            return str(prim_path).replace("{ENV_REGEX_NS}", env_regex)

    scene = getattr(gym_env, "scene", None) if gym_env is not None else None
    sensors = getattr(scene, "sensors", None) if scene is not None else None
    if sensors is not None:
        sensor = sensors.get(attr_name)
        if sensor is not None:
            prim_path = getattr(getattr(sensor, "cfg", None), "prim_path", None)
            if prim_path:
                return str(prim_path).replace("{ENV_REGEX_NS}", env_regex)

    return _EX001_6R_WRIST_CAMERA_FALLBACK.get(side)


def _camera_cfg_for_side(camera_bundle: Any, side: str) -> Any | None:
    if camera_bundle is None:
        return None
    return getattr(camera_bundle, f"{side}_wrist_camera", None)


def _camera_cfg_image_size(camera_cfg: Any) -> tuple[int, int] | None:
    spawn = getattr(camera_cfg, "spawn", None)
    width = int(getattr(camera_cfg, "width", 0) or getattr(spawn, "width", 0) or 0)
    height = int(getattr(camera_cfg, "height", 0) or getattr(spawn, "height", 0) or 0)
    return (width, height) if width > 0 and height > 0 else None


def _fisheye_distortion(camera_cfg: Any) -> np.ndarray:
    spawn = getattr(camera_cfg, "spawn", None)
    coefficients = list(getattr(spawn, "distortion_coefficients", None) or (0.0, 0.0, 0.0, 0.0))
    coefficients.extend([0.0] * (4 - len(coefficients)))
    return np.asarray(coefficients[:4], dtype=np.float64)


def _fisheye_intrinsic_from_cfg(camera_cfg: Any, image_size: tuple[int, int]) -> np.ndarray | None:
    """Scale the lens-model calibration to the rendered image size."""
    spawn = getattr(camera_cfg, "spawn", None)
    if spawn is None:
        return None

    fx = getattr(spawn, "fx", None)
    fy = getattr(spawn, "fy", None)
    cx = getattr(spawn, "cx", None)
    cy = getattr(spawn, "cy", None)
    source_width = int(getattr(spawn, "width", 0) or getattr(camera_cfg, "width", 0) or 0)
    source_height = int(getattr(spawn, "height", 0) or getattr(camera_cfg, "height", 0) or 0)
    if any(value is None for value in (fx, fy, cx, cy)) or source_width <= 0 or source_height <= 0:
        return None

    width, height = image_size
    scale_x = width / source_width
    scale_y = height / source_height
    return np.asarray(
        [
            [float(fx) * scale_x, 0.0, float(cx) * scale_x],
            [0.0, float(fy) * scale_y, float(cy) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _prim_fisheye_calibration(
    camera: Any,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the OpenCV-fisheye model actually consumed by RTX from the camera prim."""
    prim = camera.GetPrim() if callable(getattr(camera, "GetPrim", None)) else camera
    get_attribute = getattr(prim, "GetAttribute", None)
    if not callable(get_attribute):
        return None

    def read(name: str) -> Any:
        attribute = get_attribute(name)
        if attribute is None:
            return None
        get_value = getattr(attribute, "Get", None)
        return get_value() if callable(get_value) else None

    if read("omni:lensdistortion:model") != "opencvFisheye":
        return None

    prefix = "omni:lensdistortion:opencvFisheye:"
    fx, fy = read(prefix + "fx"), read(prefix + "fy")
    cx, cy = read(prefix + "cx"), read(prefix + "cy")
    source_size = read(prefix + "imageSize")
    if any(value is None for value in (fx, fy, cx, cy, source_size)):
        return None
    try:
        source_width, source_height = int(source_size[0]), int(source_size[1])
        width, height = image_size
        if source_width <= 0 or source_height <= 0 or width <= 0 or height <= 0:
            return None
        intrinsic = np.asarray(
            [
                [float(fx) * width / source_width, 0.0, float(cx) * width / source_width],
                [0.0, float(fy) * height / source_height, float(cy) * height / source_height],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.asarray(
            [float(read(prefix + f"k{index}") or 0.0) for index in range(1, 5)],
            dtype=np.float64,
        )
    except (IndexError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(distortion)):
        return None
    return intrinsic, distortion


def _runtime_fisheye_calibration(
    sensor: Any,
    image_size: tuple[int, int],
    *,
    env_idx: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    sensor_prims = getattr(sensor, "_sensor_prims", None)
    if not sensor_prims or env_idx < 0 or env_idx >= len(sensor_prims):
        return None
    return _prim_fisheye_calibration(sensor_prims[env_idx], image_size)


def _viewport_fisheye_calibration(
    viewport_window: Any,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the active viewport camera's live USD lens schema."""
    api = getattr(viewport_window, "viewport_api", None)
    get_active_camera = getattr(api, "get_active_camera", None)
    if not callable(get_active_camera):
        return None
    try:
        camera_path = str(get_active_camera())
        from omni.usd import get_context

        stage = get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path) if stage is not None else None
    except (AttributeError, ImportError, RuntimeError, TypeError):
        return None
    return _prim_fisheye_calibration(camera_prim, image_size)


def _project_fisheye_norm(
    point_camera: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[float, float] | None:
    import cv2

    width, height = image_size
    point_camera = np.asarray(point_camera, dtype=np.float64).reshape(3)
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    if width <= 0 or height <= 0:
        return None
    if not np.all(np.isfinite(point_camera)) or not np.all(np.isfinite(intrinsic)):
        return None
    if point_camera[2] <= 1.0e-6:
        return None

    image_points, _ = cv2.fisheye.projectPoints(
        point_camera.reshape(1, 1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        intrinsic,
        distortion,
    )
    normalized_x = float(image_points[0, 0, 0]) / width
    normalized_y = float(image_points[0, 0, 1]) / height
    return (
        max(0.0, min(1.0, normalized_x)),
        max(0.0, min(1.0, normalized_y)),
    )


def _g_closed_contact_camera_ros() -> np.ndarray:
    """Transform the CAD-defined closed-jaw point through the rigid extrinsic."""
    point_x, point_y, point_z = _EX001_6R_G_CLOSED_CONTACT_BASE
    camera_x, _, camera_z = _EX001_6R_WRIST_CAMERA_ORIGIN_BASE
    delta_x = point_x - camera_x
    delta_z = point_z - camera_z
    sin_pitch = _EX001_6R_WRIST_CAMERA_PITCH_SIN
    cos_pitch = _EX001_6R_WRIST_CAMERA_PITCH_COS
    return np.asarray(
        (
            -point_y,
            -sin_pitch * delta_x - cos_pitch * delta_z,
            cos_pitch * delta_x - sin_pitch * delta_z,
        ),
        dtype=np.float64,
    )


def _project_g_tip_with_sensor(sensor: Any, *, env_idx: int) -> tuple[float, float] | None:
    data = sensor.data
    data_image_shape = getattr(data, "image_shape", None)
    sensor_image_shape = getattr(sensor, "image_shape", None)
    image_shape = data_image_shape or sensor_image_shape or (480, 640)
    image_size = (
        int(getattr(sensor.cfg, "width", 0) or image_shape[1]),
        int(getattr(sensor.cfg, "height", 0) or image_shape[0]),
    )
    # CameraData.K describes the USD pinhole film gate, while RTX renders with
    # the lens schema authored on the live camera prim. Referenced robot USDs
    # may already contain per-unit calibration, so the runtime prim is the
    # authority and CameraCfg is only a fallback when that schema is absent.
    runtime_calibration = _runtime_fisheye_calibration(sensor, image_size, env_idx=env_idx)
    if runtime_calibration is not None:
        intrinsic, distortion = runtime_calibration
    else:
        intrinsic = _fisheye_intrinsic_from_cfg(sensor.cfg, image_size)
        if intrinsic is None:
            intrinsic = as_torch(data.intrinsic_matrices)[env_idx].detach().cpu().numpy()
        distortion = _fisheye_distortion(sensor.cfg)
    return _project_fisheye_norm(
        _g_closed_contact_camera_ros(),
        intrinsic,
        distortion,
        image_size,
    )


def _viewport_texture_size(viewport_window: Any, camera_cfg: Any) -> tuple[int, int] | None:
    api = getattr(viewport_window, "viewport_api", None)
    if api is not None:
        get_resolution = getattr(api, "get_texture_resolution", None)
        try:
            resolution = get_resolution() if callable(get_resolution) else getattr(api, "resolution", None)
            if resolution is not None:
                width, height = int(resolution[0]), int(resolution[1])
                if width > 0 and height > 0:
                    return width, height
        except (IndexError, TypeError, ValueError):
            pass
    return _camera_cfg_image_size(camera_cfg)


def _project_g_tip_with_viewport(
    viewport_window: Any,
    camera_cfg: Any,
) -> tuple[float, float] | None:
    image_size = _viewport_texture_size(viewport_window, camera_cfg)
    if image_size is None:
        return None

    runtime_calibration = _viewport_fisheye_calibration(viewport_window, image_size)
    if runtime_calibration is not None:
        intrinsic, distortion = runtime_calibration
    else:
        intrinsic = _fisheye_intrinsic_from_cfg(camera_cfg, image_size)
        if intrinsic is None:
            return None
        distortion = _fisheye_distortion(camera_cfg)
    return _project_fisheye_norm(
        _g_closed_contact_camera_ros(),
        intrinsic,
        distortion,
        image_size,
    )


def compute_tip_crosshair_norm(
    gym_env: Any,
    side: str,
    *,
    env_idx: int = 0,
    viewport_window: Any = None,
    camera_bundle: Any = None,
) -> tuple[float, float]:
    """Project the closed-gripper inner contact point using a sensor or viewport."""
    override = tip_crosshair_norm_override(side)
    if override is not None:
        return override

    sensor_name = f"{side}_wrist_camera"
    scene = getattr(gym_env, "scene", None)
    sensors = getattr(scene, "sensors", None) if scene is not None else None
    sensor = sensors.get(sensor_name) if sensors is not None else None
    if sensor is not None:
        try:
            projected = _project_g_tip_with_sensor(sensor, env_idx=env_idx)
            if projected is not None:
                return projected
        except Exception as exc:
            manaprint(f"WARNING: Tip crosshair sensor projection failed for {side!r}: {exc}")

    camera_cfg = _camera_cfg_for_side(camera_bundle, side)
    if viewport_window is not None and camera_cfg is not None:
        try:
            projected = _project_g_tip_with_viewport(viewport_window, camera_cfg)
            if projected is not None:
                return projected
        except Exception as exc:
            manaprint(f"WARNING: Tip crosshair viewport projection failed for {side!r}: {exc}")

    source = "sensor and viewport" if sensor is not None else "viewport"
    manaprint(f"WARNING: Tip crosshair: {source} projection unavailable for {side!r}; using default.")
    return _DEFAULT_CROSSHAIR_NORM


def _crosshair_image_size(
    gym_env: Any,
    side: str,
    viewport_window: Any,
    camera_bundle: Any,
    *,
    env_idx: int = 0,
) -> tuple[int, int] | None:
    render_product_size = _viewport_texture_size(viewport_window, None)
    if render_product_size is not None:
        return render_product_size

    scene = getattr(gym_env, "scene", None)
    sensors = getattr(scene, "sensors", None) if scene is not None else None
    sensor = sensors.get(f"{side}_wrist_camera") if sensors is not None else None
    if sensor is not None:
        image_size = _camera_cfg_image_size(getattr(sensor, "cfg", None))
        if image_size is not None:
            return image_size

    camera_cfg = _camera_cfg_for_side(camera_bundle, side)
    if camera_cfg is None:
        return None
    return _viewport_texture_size(viewport_window, camera_cfg)


def _get_viewport_overlay_frame(viewport_window: Any, layer_name: str) -> Any:
    get_frame = getattr(viewport_window, "get_frame", None)
    if callable(get_frame):
        frame = get_frame(layer_name)
        clear = getattr(frame, "clear", None)
        if callable(clear):
            clear()
        return frame
    ui_window = getattr(viewport_window, "window", viewport_window)
    frame = getattr(ui_window, "frame", None)
    if frame is not None:
        clear = getattr(frame, "clear", None)
        if callable(clear):
            clear()
    return frame


class _CrosshairOverlayHandle:
    """Own a resize-aware crosshair frame without introducing per-frame work."""

    def __init__(
        self,
        frame: Any,
        ui: Any,
        norm_xy: tuple[float, float],
        *,
        image_size: tuple[int, int] | None,
        norm_provider: Callable[[], tuple[float, float]] | None,
        image_size_provider: Callable[[], tuple[int, int] | None] | None,
        color: int,
        size_px: int,
        thickness_px: int,
        style: str,
    ) -> None:
        self._frame = frame
        self._ui = ui
        self._norm_xy = norm_xy
        self._image_size = image_size
        self._norm_provider = norm_provider
        self._image_size_provider = image_size_provider
        self._color = color
        self._size_px = size_px
        self._thickness_px = thickness_px
        self._style = style
        self._destroyed = False
        self._refreshing = False

        set_size_changed = getattr(frame, "set_computed_content_size_changed_fn", None)
        if callable(set_size_changed):
            set_size_changed(self._on_frame_size_changed)
        self._draw()

    def _offsets(self) -> tuple[float, float] | None:
        frame_width = float(getattr(self._frame, "computed_width", 0.0) or 0.0)
        frame_height = float(getattr(self._frame, "computed_height", 0.0) or 0.0)
        if frame_width <= 0.0 or frame_height <= 0.0:
            return None
        if self._image_size is not None:
            return map_image_norm_to_frame_px(
                self._norm_xy,
                image_size=self._image_size,
                frame_size=(frame_width, frame_height),
            )
        return (
            self._norm_xy[0] * frame_width,
            self._norm_xy[1] * frame_height,
        )

    def _draw(self) -> None:
        if self._destroyed:
            return
        clear = getattr(self._frame, "clear", None)
        if callable(clear):
            clear()
        thick = max(1, int(self._thickness_px))
        if self._style == "cross":
            half = max(4, int(self._size_px))
            marker_size = half * 2
        else:
            marker_size = max(3, int(self._size_px))
        offsets = self._offsets()
        if offsets is None:
            return
        center_x, center_y = offsets
        offset_x = center_x - marker_size * 0.5
        offset_y = center_y - marker_size * 0.5
        with self._frame:
            with self._ui.Placer(
                offset_x=offset_x,
                offset_y=offset_y,
            ):
                if self._style == "cross":
                    canvas = marker_size
                    dot = max(3, thick + 1)
                    with self._ui.ZStack(width=canvas, height=canvas):
                        with self._ui.Placer(offset_x=0, offset_y=half - thick * 0.5):
                            self._ui.Rectangle(
                                width=canvas,
                                height=thick,
                                style={"background_color": self._color},
                            )
                        with self._ui.Placer(offset_x=half - thick * 0.5, offset_y=0):
                            self._ui.Rectangle(
                                width=thick,
                                height=canvas,
                                style={"background_color": self._color},
                            )
                        dot_offset = (canvas - dot) * 0.5
                        with self._ui.Placer(offset_x=dot_offset, offset_y=dot_offset):
                            self._ui.Rectangle(
                                width=dot,
                                height=dot,
                                style={"background_color": self._color},
                            )
                else:
                    dot = marker_size
                    self._ui.Rectangle(
                        width=dot,
                        height=dot,
                        style={"background_color": self._color, "border_radius": dot // 2},
                    )

    def _on_frame_size_changed(self, *_args: Any) -> None:
        if self._destroyed or self._refreshing:
            return
        self._refreshing = True
        try:
            if self._norm_provider is not None:
                self._norm_xy = self._norm_provider()
            if self._image_size_provider is not None:
                image_size = self._image_size_provider()
                if image_size is not None:
                    self._image_size = image_size
            self._draw()
        except Exception as exc:
            manaprint(f"WARNING: Tip crosshair resize refresh failed: {exc}")
        finally:
            self._refreshing = False

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        set_size_changed = getattr(self._frame, "set_computed_content_size_changed_fn", None)
        if callable(set_size_changed):
            try:
                set_size_changed(None)
            except Exception:
                pass
        clear = getattr(self._frame, "clear", None)
        if callable(clear):
            try:
                clear()
            except Exception:
                pass


def attach_tip_crosshair_to_viewport(
    viewport_window: Any,
    norm_xy: tuple[float, float],
    *,
    image_size: tuple[int, int] | None = None,
    norm_provider: Callable[[], tuple[float, float]] | None = None,
    image_size_provider: Callable[[], tuple[int, int] | None] | None = None,
    color: int = _DEFAULT_COLOR,
    size_px: int = _DEFAULT_SIZE_PX,
    thickness_px: int = _DEFAULT_THICKNESS_PX,
    style: str = _DEFAULT_STYLE,
) -> Any:
    """Attach a resize-aware HUD crosshair to a Kit viewport."""
    try:
        import omni.ui as ui
    except Exception as exc:
        manaprint(f"WARNING: Tip crosshair: cannot import omni.ui: {exc}")
        return None

    frame = _get_viewport_overlay_frame(viewport_window, _CROSSHAIR_EXT_ID)
    if frame is None:
        manaprint("WARNING: Tip crosshair: viewport window has no overlay frame.")
        return None

    return _CrosshairOverlayHandle(
        frame,
        ui,
        norm_xy,
        image_size=image_size,
        norm_provider=norm_provider,
        image_size_provider=image_size_provider,
        color=color,
        size_px=size_px,
        thickness_px=thickness_px,
        style=style,
    )


def warmup_sim_for_crosshair(gym_env: Any, *, frames: int = 8) -> None:
    sim = getattr(gym_env, "sim", None)
    if sim is None:
        return
    for _ in range(max(1, int(frames))):
        sim.render()


def _viewport_active_camera_path(viewport_window: Any) -> str | None:
    api = getattr(viewport_window, "viewport_api", None)
    get_camera = getattr(api, "get_active_camera", None)
    if not callable(get_camera):
        return None
    try:
        return str(get_camera())
    except Exception:
        return None


def match_viewports_to_wrist_sides(
    viewport_windows: list[Any],
    *,
    camera_bundle: Any = None,
    gym_env: Any = None,
    sides: tuple[str, ...] = _SIDES,
) -> dict[str, Any]:
    """Map each wrist side to the viewport whose active camera is that wrist.

    Windows come in stream_viewports YAML order (head camera may be first), so
    positional pairing is wrong; match by camera prim path instead.
    """
    cam_to_window = {
        path: window
        for window in viewport_windows
        if window is not None and (path := _viewport_active_camera_path(window))
    }
    matched: dict[str, Any] = {}
    for side in sides:
        wrist_path = _resolve_single_wrist_camera_path(side, camera_bundle=camera_bundle, gym_env=gym_env)
        window = cam_to_window.get(wrist_path) if wrist_path else None
        if window is not None:
            matched[side] = window
    return matched


def attach_tip_crosshairs_to_viewports(
    gym_env: Any,
    viewport_windows: list[Any],
    *,
    sides: tuple[str, ...] = _SIDES,
    camera_bundle: Any = None,
) -> list[Any]:
    if not tip_crosshair_enabled():
        return []
    # G-jaw CAD tip extrinsic does not apply to H-jaw; skip the HUD mark.
    if _is_h_jaw_robot(gym_env):
        manaprint("INFO: Tip crosshair disabled for H-jaw (ex001_6r_legacy).")
        return []
    if not viewport_windows:
        return []

    warmup_sim_for_crosshair(gym_env)
    style = crosshair_style_from_env()
    matched = match_viewports_to_wrist_sides(
        viewport_windows, camera_bundle=camera_bundle, gym_env=gym_env, sides=sides
    )
    scene = getattr(gym_env, "scene", None)
    sensors = getattr(scene, "sensors", None) if scene is not None else None
    handles: list[Any] = []
    for side in sides:
        window = matched.get(side)
        if window is None:
            manaprint(f"INFO: Tip crosshair: no viewport bound to the {side} wrist camera; skipped.")
            continue
        try:

            def project_norm(side: str = side, window: Any = window) -> tuple[float, float]:
                return compute_tip_crosshair_norm(
                    gym_env,
                    side,
                    viewport_window=window,
                    camera_bundle=camera_bundle,
                )

            def image_size(side: str = side, window: Any = window) -> tuple[int, int] | None:
                return _crosshair_image_size(gym_env, side, window, camera_bundle)

            norm = project_norm()
            # Sensor.data may synchronously enter RTX rendering when its buffers
            # are stale. A viewport frame-size callback can run during that same
            # render, so re-reading sensor.data from norm_provider would recurse
            # into the renderer and stop the simulation. The initial projection
            # is sufficient for the display-only marker; viewport-only fallback
            # projections remain resize-aware.
            sensor = sensors.get(f"{side}_wrist_camera") if sensors is not None else None
            norm_provider = None if sensor is not None else project_norm
            handle = attach_tip_crosshair_to_viewport(
                window,
                norm,
                image_size=image_size(),
                norm_provider=norm_provider,
                image_size_provider=image_size,
                **style,
            )
            if handle is not None:
                handles.append(handle)
                manaprint(f"INFO: Tip crosshair attached on {side} viewport at ({norm[0]:.3f}, {norm[1]:.3f}).")
        except Exception as exc:
            manaprint(f"WARNING: Failed to attach tip crosshair for {side!r}: {exc}")
    return handles
