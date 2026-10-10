"""Nominal camera configuration for the X2Robot 5756 USB2.0 fisheye module.

Hardware source: CZB2F5756-V1 USB specification, wrist mono fisheye camera.
Document source: local camera asset bundle
Spec: 640x480@30fps MJPG, EFL 2.5 +/-5% mm. Datasheet optical FOV H137.2 V101 D179.2 (degrees).

These values are spec-derived nominal intrinsics, not per-unit calibration data.
H/V FOV are listed for cross-check against the datasheet; the simulation uses a
single equidistant focal length derived from diagonal D179.2° so fx=fy (Omniverse
assumes square pixels — see ``isaaclab.utils.sensors.convert_camera_intrinsics_to_usd``).
USD ``focal_length`` / apertures are tied to EFL 2.5 mm so the pinhole film gate
matches the OpenCV fisheye intrinsics and avoids rectangular vignetting / black wedges.
TODO: replace with per-robot/per-camera calibration once intrinsics_calibration_v2 lands.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.utils.sensors as sensor_utils

from .update_camera import OpenCVFisheyeCameraCfg, isaac_rtx_renderer_cfg

if TYPE_CHECKING:
    from isaaclab.sensors import CameraCfg

X2ROBOT_5756_FISHEYE_WIDTH = 640
X2ROBOT_5756_FISHEYE_HEIGHT = 480
X2ROBOT_5756_FISHEYE_FPS = 30
X2ROBOT_5756_FISHEYE_EFL_MM = 2.5
X2ROBOT_5756_FISHEYE_HORIZONTAL_FOV_DEG = 137.2
X2ROBOT_5756_FISHEYE_VERTICAL_FOV_DEG = 101.0
X2ROBOT_5756_FISHEYE_DIAGONAL_FOV_DEG = 179.2

# World units are meters. Wrist fisheye sees nearby gripper geometry; balance
# near-clip (avoid self-occlusion) vs far-clip (capture workspace).
X2ROBOT_5756_FISHEYE_CLIP_NEAR_M = 0.01
X2ROBOT_5756_FISHEYE_CLIP_FAR_M = 10.0

_half_w = X2ROBOT_5756_FISHEYE_WIDTH / 2.0
_half_h = X2ROBOT_5756_FISHEYE_HEIGHT / 2.0
_half_diag_px = math.hypot(_half_w, _half_h)
_theta_half_diag_rad = math.radians(X2ROBOT_5756_FISHEYE_DIAGONAL_FOV_DEG / 2.0)
# Equidistant (OpenCV fisheye): r = f * theta. Corner radius = half_diag; theta = D/2.
X2ROBOT_5756_FISHEYE_FXY = _half_diag_px / _theta_half_diag_rad
X2ROBOT_5756_FISHEYE_FX = X2ROBOT_5756_FISHEYE_FXY
X2ROBOT_5756_FISHEYE_FY = X2ROBOT_5756_FISHEYE_FXY
X2ROBOT_5756_FISHEYE_CX = X2ROBOT_5756_FISHEYE_WIDTH / 2.0
X2ROBOT_5756_FISHEYE_CY = X2ROBOT_5756_FISHEYE_HEIGHT / 2.0
X2ROBOT_5756_FISHEYE_DISTORTION_COEFFICIENTS = [0.0, 0.0, 0.0, 0.0]

_X2ROBOT_5756_INTRINSIC_MATRIX = [
    X2ROBOT_5756_FISHEYE_FXY,
    0.0,
    X2ROBOT_5756_FISHEYE_CX,
    0.0,
    X2ROBOT_5756_FISHEYE_FXY,
    X2ROBOT_5756_FISHEYE_CY,
    0.0,
    0.0,
    1.0,
]
# EFL in cm for convert_camera_intrinsics_to_usd (spec 2.5 mm nominal).
_X2ROBOT_5756_FOCAL_LENGTH_CM = X2ROBOT_5756_FISHEYE_EFL_MM / 10.0
X2ROBOT_5756_USD_PINHOLE = sensor_utils.convert_camera_intrinsics_to_usd(
    intrinsic_matrix=_X2ROBOT_5756_INTRINSIC_MATRIX,
    width=X2ROBOT_5756_FISHEYE_WIDTH,
    height=X2ROBOT_5756_FISHEYE_HEIGHT,
    focal_length=_X2ROBOT_5756_FOCAL_LENGTH_CM,
)


def make_x2robot_5756_fisheye_camera_cfg() -> CameraCfg:
    """Build a CameraCfg for the shared 5756 fisheye camera module.

    ideal equidistant fisheye: k1..k4 are zero; curvature comes from the fisheye
    projection model. Intrinsics use diagonal FOV from the datasheet.
    """
    from isaaclab.sensors import CameraCfg

    from manaenv.assets.sensors.async_camera import ManaAsyncCamera

    return CameraCfg(
        class_type=ManaAsyncCamera,
        spawn=OpenCVFisheyeCameraCfg(
            clipping_range=(X2ROBOT_5756_FISHEYE_CLIP_NEAR_M, X2ROBOT_5756_FISHEYE_CLIP_FAR_M),
            width=X2ROBOT_5756_FISHEYE_WIDTH,
            height=X2ROBOT_5756_FISHEYE_HEIGHT,
            fx=X2ROBOT_5756_FISHEYE_FX,
            fy=X2ROBOT_5756_FISHEYE_FY,
            cx=X2ROBOT_5756_FISHEYE_CX,
            cy=X2ROBOT_5756_FISHEYE_CY,
            distortion_coefficients=X2ROBOT_5756_FISHEYE_DISTORTION_COEFFICIENTS,
            max_fov=X2ROBOT_5756_FISHEYE_DIAGONAL_FOV_DEG,
        ),
        width=X2ROBOT_5756_FISHEYE_WIDTH,
        height=X2ROBOT_5756_FISHEYE_HEIGHT,
        data_types=["rgb"],
        renderer_cfg=isaac_rtx_renderer_cfg(),
    )
