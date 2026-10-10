"""OpenCV camera configurations using USD lens-distortion schema authoring.

This module provides configuration classes for updating existing OpenCV fisheye
and pinhole cameras in Isaac Lab environments without the deprecated Isaac Sim Camera API.

Usage:
------
There are TWO ways to use these configurations:

1. UPDATE ONLY CLIPPING RANGE (preserve original intrinsics):
   Use this when you only want to modify clipping range without changing
   the camera intrinsics already defined in the USD file.

   Example:
   ```python
   CameraCfg(
       spawn=OpenCVFisheyeCameraCfg(
           clipping_range=(0.01, 1.0e5),  # Only this will be updated
       ),
       width=640,
       height=480,
       data_types=["rgb"],
   )
   ```

2. UPDATE FULL INTRINSICS (override camera model parameters):
   Use this when you want to apply calibrated camera intrinsics.
   ALL of fx, fy, cx, cy, and distortion_coefficients must be provided.

   NOTE (Headless): This intrinsics update path requires the OpenCV lens-distortion USD
   schemas (``OmniLensDistortionOpenCvFisheyeAPI`` /
   ``OmniLensDistortionOpenCvPinholeAPI``) registered by the
   ``omni.usd.schema.omni_lens_distortion`` extension. The Isaac Lab headless rendering
   ``.kit`` experience does not list it in its dependencies, and the schema's
   ``order = -100`` requires it to load *before* USD finalises its plugin registry —
   late ``set_extension_enabled_immediate`` cannot retroactively register the schema.

   Camera-enabled entry points must therefore call
   :func:`manaenv.utils.isaac_extensions.prepare_camera_kit_args` *before* constructing
   ``AppLauncher`` so Kit picks up ``--enable omni.usd.schema.omni_lens_distortion`` and
   ``--enable isaacsim.sensors.experimental.rtx`` during ``SimulationApp`` startup.
   This module then authors lens-distortion attributes directly on the USD camera prim.

   Example:
   ```python
   CameraCfg(
       spawn=OpenCVFisheyeCameraCfg(
           clipping_range=(0.01, 1.0e5),
           width=640,
           height=480,
           fx=284.08757,
           fy=283.37961,
           cx=338.93095,
           cy=252.22276,
           distortion_coefficients=[-8.19e-03, 1.23e-02, -2.12e-02, 8.25e-03],
       ),
       width=640,
       height=480,
       data_types=["rgb"],
   )
   ```

Distortion Coefficients:
------------------------
- Fisheye (OpenCV): [k1, k2, k3, k4]
- Pinhole (OpenCV): [k1, k2, p1, p2, k3, k4, k5, k6]
  - k1-k6: radial distortion coefficients
  - p1, p2: tangential distortion coefficients

Refer in IsaacSim doc for more details.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from pxr import Sdf, Usd, UsdGeom

logger = logging.getLogger(__name__)


def isaac_rtx_renderer_cfg(
    *,
    depth_clipping_behavior: Literal["max", "zero", "none"] = "max",
) -> IsaacRtxRendererCfg:
    """Build an Isaac RTX renderer cfg with depth clipping on ``renderer_cfg``.

    Prefer this over deprecated ``CameraCfg.depth_clipping_behavior``.
    """
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    return IsaacRtxRendererCfg(depth_clipping_behavior=depth_clipping_behavior)


def _opencv_fisheye_intrinsics_complete(cfg) -> bool:
    """True when cfg carries all fields required for OpenCV fisheye API / USD lens authoring."""
    distortion_coefficients = getattr(cfg, "distortion_coefficients", None)
    try:
        has_required_coefficients = distortion_coefficients is not None and len(distortion_coefficients) >= 4
    except TypeError:
        has_required_coefficients = False
    if not has_required_coefficients:
        return False
    return all(
        [
            getattr(cfg, "fx", None) is not None,
            getattr(cfg, "fy", None) is not None,
            getattr(cfg, "cx", None) is not None,
            getattr(cfg, "cy", None) is not None,
            getattr(cfg, "width", None) is not None,
            getattr(cfg, "height", None) is not None,
        ]
    )


def _opencv_pinhole_intrinsics_complete(cfg) -> bool:
    """True when cfg carries all fields required for OpenCV pinhole USD lens authoring."""
    distortion_coefficients = getattr(cfg, "distortion_coefficients", None)
    try:
        has_required_coefficients = distortion_coefficients is not None and len(distortion_coefficients) >= 8
    except TypeError:
        has_required_coefficients = False
    if not has_required_coefficients:
        return False
    return all(
        [
            getattr(cfg, "fx", None) is not None,
            getattr(cfg, "fy", None) is not None,
            getattr(cfg, "cx", None) is not None,
            getattr(cfg, "cy", None) is not None,
            getattr(cfg, "width", None) is not None,
            getattr(cfg, "height", None) is not None,
        ]
    )


def _find_matching_paths(stage, pattern: str) -> list[str]:
    """Find all prim paths matching a regex pattern.

    Args:
        stage: USD stage to search.
        pattern: Path pattern with wildcards (e.g., "env_.*" matches "env_0", "env_1", etc.)

    Returns:
        List of matching prim paths.
    """
    regex = re.compile(f"^{pattern.replace('.*', '[^/]+').replace('*', '[^/]*')}$")
    return [str(p.GetPath()) for p in stage.Traverse() if regex.match(str(p.GetPath()))]


def _apply_local_pose(
    prim: Usd.Prim,
    translation: tuple[float, float, float] | None,
    orientation_wxyz: tuple[float, float, float, float] | None,
) -> None:
    """Apply absolute local pose overrides to an existing camera prim.

    When a field is ``None``, the corresponding authored USD value is preserved.
    When a field is provided, it overwrites the prim's local translate/orient op.
    """
    from pxr import Gf, UsdGeom

    if translation is None and orientation_wxyz is None:
        return

    xformable = UsdGeom.Xformable(prim)

    translate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps() or ():
        op_type = op.GetOpType()
        if translate_op is None and op_type == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        if orient_op is None and op_type == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translation is not None:
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        dx, dy, dz = translation
        translate_op.Set(Gf.Vec3d(float(dx), float(dy), float(dz)))

    if orientation_wxyz is not None:
        if orient_op is None:
            orient_op = xformable.AddOrientOp()
        w, x, y, z = orientation_wxyz
        absolute_q = Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))
        absolute_q.Normalize()
        orient_op.Set(absolute_q)


def _update_camera_base(prim: Usd.Prim, cfg) -> UsdGeom.Camera:
    """Update base camera parameters (clipping range, focus, f-stop).

    Only updates parameters that are explicitly set (not None) in the config.

    Args:
        prim: Camera prim to update.
        cfg: Camera configuration with optional parameters.

    Returns:
        UsdGeom.Camera object for the prim.
    """
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera(prim)
    if cfg.clipping_range:
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(*cfg.clipping_range))
    if cfg.focus_distance:
        camera.GetFocusDistanceAttr().Set(cfg.focus_distance)
    if cfg.f_stop:
        camera.GetFStopAttr().Set(cfg.f_stop)
    # ex001_6r_legacy.usd wrist cams: UsdGeom.Camera.focalLength (cm in Isaac spawn doc) with OpenCV fisheye API
    if _opencv_fisheye_intrinsics_complete(cfg) and cfg.focal_length is not None:
        camera.GetFocalLengthAttr().Set(float(cfg.focal_length))
    return camera


def _set_fisheye_via_usd(prim_path: str, cfg: OpenCVFisheyeCameraCfg, stage):
    """Set fisheye properties by authoring OpenCV lens-distortion USD attributes."""
    if not _opencv_fisheye_intrinsics_complete(cfg):
        return
    _set_opencv_fisheye_via_usd(prim_path, cfg, stage)


def _get_or_create_attr(prim: Usd.Prim, attr_name: str, attr_type: Sdf.ValueTypeName) -> Usd.Attribute:
    attr = prim.GetAttribute(attr_name)
    if not attr:
        attr = prim.CreateAttribute(attr_name, attr_type, False)
    if not attr:
        raise RuntimeError(f"Failed to create USD attribute {attr_name!r} on {prim.GetPath()}")
    return attr


def _set_attr(prim: Usd.Prim, attr_name: str, attr_type: Sdf.ValueTypeName, value) -> None:
    _get_or_create_attr(prim, attr_name, attr_type).Set(value)


def _set_legacy_ftheta_attrs(prim: Usd.Prim, cfg: OpenCVFisheyeCameraCfg) -> None:
    """Author legacy ftheta attributes when the IsaacSim camera API is unavailable."""
    from pxr import Sdf

    if not all([cfg.width, cfg.height, cfg.fx, cfg.fy, cfg.cx, cfg.cy]):
        return

    half_diag_px = math.hypot(cfg.width / 2.0, cfg.height / 2.0)
    if cfg.max_fov is not None:
        max_fov = cfg.max_fov
        ftheta_poly_b = math.radians(max_fov / 2.0) / half_diag_px
    else:
        f_equiv = (cfg.fx + cfg.fy) / 2.0
        max_fov = math.degrees(2.0 * half_diag_px / f_equiv)
        ftheta_poly_b = 1.0 / f_equiv

    attrs = [
        ("cameraProjectionType", Sdf.ValueTypeNames.Token, "fisheyePolynomial"),
        ("fthetaWidth", Sdf.ValueTypeNames.Float, float(cfg.width)),
        ("fthetaHeight", Sdf.ValueTypeNames.Float, float(cfg.height)),
        ("fthetaCx", Sdf.ValueTypeNames.Float, float(cfg.cx)),
        ("fthetaCy", Sdf.ValueTypeNames.Float, float(cfg.cy)),
        ("fthetaMaxFov", Sdf.ValueTypeNames.Float, float(max_fov)),
        ("fthetaPolyA", Sdf.ValueTypeNames.Float, 0.0),
        ("fthetaPolyB", Sdf.ValueTypeNames.Float, float(ftheta_poly_b)),
        ("fthetaPolyC", Sdf.ValueTypeNames.Float, 0.0),
        ("fthetaPolyD", Sdf.ValueTypeNames.Float, 0.0),
        ("fthetaPolyE", Sdf.ValueTypeNames.Float, 0.0),
        ("fthetaPolyF", Sdf.ValueTypeNames.Float, 0.0),
    ]
    for attr_name, attr_type, value in attrs:
        _set_attr(prim, attr_name, attr_type, value)


def _set_opencv_fisheye_via_usd(prim_path: str, cfg: OpenCVFisheyeCameraCfg, stage) -> None:
    """Author OpenCV fisheye lens-distortion USD attributes without IsaacSim Camera API."""
    from pxr import Gf, Sdf, Tf

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    schema_available = True
    try:
        prim.ApplyAPI("OmniLensDistortionOpenCvFisheyeAPI")
    except (Tf.ErrorException, RuntimeError) as exc:
        logger.warning(
            "OmniLensDistortionOpenCvFisheyeAPI schema not registered for %s; using legacy ftheta fallback: %s",
            prim_path,
            exc,
        )
        schema_available = False

    if schema_available:
        logger.debug("Setting OpenCV fisheye via USD for %s", prim_path)
        # Clear any existing ftheta projection type
        proj_type_attr = prim.GetAttribute("cameraProjectionType")
        if proj_type_attr and proj_type_attr.Get() == "fisheyePolynomial":
            logger.debug("Clearing fisheyePolynomial projection for %s", prim_path)
            proj_type_attr.Clear()

        # Set OpenCV fisheye lens distortion attributes
        attrs = [
            ("omni:lensdistortion:model", Sdf.ValueTypeNames.Token, "opencvFisheye"),
            ("omni:lensdistortion:opencvFisheye:cx", Sdf.ValueTypeNames.Float, float(cfg.cx)),
            ("omni:lensdistortion:opencvFisheye:cy", Sdf.ValueTypeNames.Float, float(cfg.cy)),
            ("omni:lensdistortion:opencvFisheye:fx", Sdf.ValueTypeNames.Float, float(cfg.fx)),
            ("omni:lensdistortion:opencvFisheye:fy", Sdf.ValueTypeNames.Float, float(cfg.fy)),
            (
                "omni:lensdistortion:opencvFisheye:imageSize",
                Sdf.ValueTypeNames.Int2,
                Gf.Vec2i(int(cfg.width), int(cfg.height)),
            ),
        ]
        for attr_name, attr_type, value in attrs:
            _set_attr(prim, attr_name, attr_type, value)

        for index, coeff_name in enumerate(("k1", "k2", "k3", "k4")):
            _set_attr(
                prim,
                f"omni:lensdistortion:opencvFisheye:{coeff_name}",
                Sdf.ValueTypeNames.Float,
                float(cfg.distortion_coefficients[index]),
            )
    else:
        logger.debug("Using ftheta fallback for %s", prim_path)
        _set_legacy_ftheta_attrs(prim, cfg)


def _set_opencv_pinhole_via_usd(prim_path: str, cfg: OpenCVPinholeCameraCfg, stage) -> None:
    """Author OpenCV pinhole lens-distortion USD attributes without IsaacSim Camera API."""
    from pxr import Gf, Sdf, Tf

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    try:
        prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI")
    except (Tf.ErrorException, RuntimeError) as exc:
        logger.warning(
            "OmniLensDistortionOpenCvPinholeAPI schema not registered for %s; authoring raw attributes: %s",
            prim_path,
            exc,
        )

    logger.debug("Setting OpenCV pinhole via USD for %s", prim_path)
    attrs = [
        ("omni:lensdistortion:model", Sdf.ValueTypeNames.Token, "opencvPinhole"),
        ("omni:lensdistortion:opencvPinhole:cx", Sdf.ValueTypeNames.Float, float(cfg.cx)),
        ("omni:lensdistortion:opencvPinhole:cy", Sdf.ValueTypeNames.Float, float(cfg.cy)),
        ("omni:lensdistortion:opencvPinhole:fx", Sdf.ValueTypeNames.Float, float(cfg.fx)),
        ("omni:lensdistortion:opencvPinhole:fy", Sdf.ValueTypeNames.Float, float(cfg.fy)),
        (
            "omni:lensdistortion:opencvPinhole:imageSize",
            Sdf.ValueTypeNames.Int2,
            Gf.Vec2i(int(cfg.width), int(cfg.height)),
        ),
    ]
    for attr_name, attr_type, value in attrs:
        _set_attr(prim, attr_name, attr_type, value)

    for index, coeff_name in enumerate(("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")):
        _set_attr(
            prim,
            f"omni:lensdistortion:opencvPinhole:{coeff_name}",
            Sdf.ValueTypeNames.Float,
            float(cfg.distortion_coefficients[index]),
        )


def _set_pinhole_via_usd(prim_path: str, cfg: OpenCVPinholeCameraCfg, stage):
    """Set pinhole properties by authoring OpenCV lens-distortion USD attributes."""
    if not _opencv_pinhole_intrinsics_complete(cfg):
        return
    _set_opencv_pinhole_via_usd(prim_path, cfg, stage)


def _update_cameras(
    prim_path: str,
    cfg,
    api_fn,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim | None:
    """Generic camera update with wildcard path support.

    Args:
        prim_path: USD path to camera. Supports wildcards like "env_.*".
        cfg: Camera configuration.
        api_fn: Function to call for setting camera-specific properties.

    Returns:
        The last updated camera prim, or None if no cameras found.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    paths = _find_matching_paths(stage, prim_path) if "*" in prim_path else [prim_path]

    result = None
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        _update_camera_base(prim, cfg)
        api_fn(path, cfg, stage)
        _apply_local_pose(prim, translation=translation, orientation_wxyz=orientation)
        result = prim
    return result


def update_opencv_fisheye_camera(
    prim_path: str, cfg: OpenCVFisheyeCameraCfg, translation=None, orientation=None
) -> Usd.Prim | None:
    return _update_cameras(prim_path, cfg, _set_fisheye_via_usd, translation=translation, orientation=orientation)


def update_opencv_pinhole_camera(
    prim_path: str, cfg: OpenCVPinholeCameraCfg, translation=None, orientation=None
) -> Usd.Prim | None:
    return _update_cameras(prim_path, cfg, _set_pinhole_via_usd, translation=translation, orientation=orientation)


@configclass
class OpenCVFisheyeCameraCfg(PinholeCameraCfg):
    """Configuration for updating an existing OpenCV fisheye camera.

    OpenCV fisheye model uses 4 distortion coefficients: [k1, k2, k3, k4]

    All parameters default to None, meaning the original USD values will be preserved.
    To update intrinsics, ALL of fx, fy, cx, cy, width, height, and distortion_coefficients must be set.
    ``focal_length`` is optional and only updates USD ``focalLength`` when explicitly provided.
    """

    func: Callable = update_opencv_fisheye_camera
    copy_from_source: bool = False

    # Image resolution in pixels (required for intrinsics update)
    width: int | None = None
    height: int | None = None
    # Focal lengths in pixels
    fx: float | None = None
    fy: float | None = None
    # Principal point in pixels
    cx: float | None = None
    cy: float | None = None
    # Fisheye distortion coefficients [k1, k2, k3, k4]
    distortion_coefficients: list[float] | None = None
    # Optional diagonal FOV used when falling back to legacy ftheta attributes
    max_fov: float | None = None
    # Base camera parameters
    clipping_range: tuple[float, float] | None = None
    # USD camera focal length in cm. Leave unset when preserving the authored focalLength.
    focal_length: float | None = None
    focus_distance: float | None = None
    f_stop: float | None = None


@configclass
class OpenCVPinholeCameraCfg(PinholeCameraCfg):
    """Configuration for updating an existing OpenCV pinhole camera.

    OpenCV pinhole model uses 8 distortion coefficients: [k1, k2, p1, p2, k3, k4, k5, k6]
    - k1, k2, k3, k4, k5, k6: radial distortion coefficients
    - p1, p2: tangential distortion coefficients

    All parameters default to None, meaning the original USD values will be preserved.
    To update intrinsics, ALL of fx, fy, cx, cy, width, height, and distortion_coefficients must be set.
    """

    func: Callable = update_opencv_pinhole_camera
    copy_from_source: bool = False

    # Image resolution in pixels (required for intrinsics update)
    width: int | None = None
    height: int | None = None
    # Focal lengths in pixels
    fx: float | None = None
    fy: float | None = None
    # Principal point in pixels
    cx: float | None = None
    cy: float | None = None
    # Pinhole distortion coefficients [k1, k2, p1, p2, k3, k4, k5, k6]
    distortion_coefficients: list[float] | None = None
    # Base camera parameters
    clipping_range: tuple[float, float] | None = None
    # USD camera focal length in cm. Leave unset when preserving the authored focalLength.
    focal_length: float | None = None
    focus_distance: float | None = None
    f_stop: float | None = None
