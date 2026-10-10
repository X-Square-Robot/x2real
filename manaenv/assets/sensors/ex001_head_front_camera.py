"""Camera configuration for EX001 front head camera."""

from isaaclab.sensors import CameraCfg

from manaenv.assets.sensors.async_camera import ManaAsyncCamera

from .update_camera import OpenCVPinholeCameraCfg, isaac_rtx_renderer_cfg

# %YAML:1.0
# ---
# date: "2026-02-10 08:35:43"
# lens_model: pinhole_5params
# image_width: 1280
# image_height: 720
# camera_matrix: !!opencv-matrix
#    rows: 3
#    cols: 3
#    dt: d
#    data: [ 720.3193, 0., 640.3909, 0.,
#        719.8599, 351.073, 0., 0., 1. ]
# dist_coeffs: !!opencv-matrix
#    rows: 5
#    cols: 1
#    dt: d
#    data: [ -0.01953918673098087, 0.00478989025577903,
#        0.0005033669294789433, -0.00021549213852267712,
#        0. ]
# reprojection_error: 0.17702512553452684

# LensType: Pinhole OpenCV
# Refer in Isaac Lab doc for more details.

_EX001_HEADFRONT_WIDTH = 1280
_EX001_HEADFRONT_HEIGHT = 720
_EX001_HEADFRONT_FX = 720.3193
_EX001_HEADFRONT_FY = 719.8599
_EX001_HEADFRONT_CX = 640.3909
_EX001_HEADFRONT_CY = 351.073
# 5-parameter pinhole calibration mapped onto Isaac Sim's OpenCV layout:
# [k1, k2, p1, p2, k3, k4, k5, k6]
_EX001_HEADFRONT_DISTORTION = [
    -0.01953918673098087,
    0.00478989025577903,
    0.0005033669294789433,
    -0.00021549213852267712,
    0.0,
    0.0,
    0.0,
    0.0,
]

EX001_HEADFRONT_PINHOLE_COLOR_CAMERA_SIM_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    spawn=OpenCVPinholeCameraCfg(
        clipping_range=(0.01, 1.0e5),
        width=_EX001_HEADFRONT_WIDTH,
        height=_EX001_HEADFRONT_HEIGHT,
        fx=_EX001_HEADFRONT_FX,
        fy=_EX001_HEADFRONT_FY,
        cx=_EX001_HEADFRONT_CX,
        cy=_EX001_HEADFRONT_CY,
        distortion_coefficients=_EX001_HEADFRONT_DISTORTION,
    ),
    width=_EX001_HEADFRONT_WIDTH,
    height=_EX001_HEADFRONT_HEIGHT,
    data_types=["rgb"],
    renderer_cfg=isaac_rtx_renderer_cfg(),
)

EX001_HEADFRONT_PINHOLE_DEPTH_CAMERA_SIM_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    spawn=OpenCVPinholeCameraCfg(
        clipping_range=(0.01, 1.0e5),
        width=_EX001_HEADFRONT_WIDTH,
        height=_EX001_HEADFRONT_HEIGHT,
        fx=_EX001_HEADFRONT_FX,
        fy=_EX001_HEADFRONT_FY,
        cx=_EX001_HEADFRONT_CX,
        cy=_EX001_HEADFRONT_CY,
        distortion_coefficients=_EX001_HEADFRONT_DISTORTION,
    ),
    width=_EX001_HEADFRONT_WIDTH,
    height=_EX001_HEADFRONT_HEIGHT,
    data_types=["depth"],
    renderer_cfg=isaac_rtx_renderer_cfg(),
)

EX001_HEADFRONT_PINHOLE_INFRA1_CAMERA_SIM_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    spawn=OpenCVPinholeCameraCfg(
        clipping_range=(0.01, 1.0e5),
        width=_EX001_HEADFRONT_WIDTH,
        height=_EX001_HEADFRONT_HEIGHT,
        fx=_EX001_HEADFRONT_FX,
        fy=_EX001_HEADFRONT_FY,
        cx=_EX001_HEADFRONT_CX,
        cy=_EX001_HEADFRONT_CY,
        distortion_coefficients=_EX001_HEADFRONT_DISTORTION,
    ),
    width=_EX001_HEADFRONT_WIDTH,
    height=_EX001_HEADFRONT_HEIGHT,
    data_types=["rgb"],
    renderer_cfg=isaac_rtx_renderer_cfg(),
)

EX001_HEADFRONT_PINHOLE_INFRA2_CAMERA_SIM_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    spawn=OpenCVPinholeCameraCfg(
        clipping_range=(0.01, 1.0e5),
        width=_EX001_HEADFRONT_WIDTH,
        height=_EX001_HEADFRONT_HEIGHT,
        fx=_EX001_HEADFRONT_FX,
        fy=_EX001_HEADFRONT_FY,
        cx=_EX001_HEADFRONT_CX,
        cy=_EX001_HEADFRONT_CY,
        distortion_coefficients=_EX001_HEADFRONT_DISTORTION,
    ),
    width=_EX001_HEADFRONT_WIDTH,
    height=_EX001_HEADFRONT_HEIGHT,
    data_types=["rgb"],
    renderer_cfg=isaac_rtx_renderer_cfg(),
)

EX001_HEADFRONT_PINHOLE_COLOR_CAMERA_CFG = EX001_HEADFRONT_PINHOLE_COLOR_CAMERA_SIM_CFG

EX001_HEADFRONT_PINHOLE_DEPTH_CAMERA_CFG = EX001_HEADFRONT_PINHOLE_DEPTH_CAMERA_SIM_CFG

# NOTE:
# The corresponding camera prim for `EX001_HEADFRONT_PINHOLE_INFRA1_CAMERA_CFG` is deactivated in the USD file.
EX001_HEADFRONT_PINHOLE_INFRA1_CAMERA_CFG = EX001_HEADFRONT_PINHOLE_INFRA1_CAMERA_SIM_CFG

# NOTE:
# The corresponding camera prim for `EX001_HEADFRONT_PINHOLE_INFRA2_CAMERA_CFG` is deactivated in the USD file.
EX001_HEADFRONT_PINHOLE_INFRA2_CAMERA_CFG = EX001_HEADFRONT_PINHOLE_INFRA2_CAMERA_SIM_CFG
