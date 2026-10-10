"""Camera configuration for EX001 front chassis."""

from isaaclab.sensors import CameraCfg

from manaenv.assets.sensors.async_camera import ManaAsyncCamera

from .update_camera import OpenCVPinholeCameraCfg, isaac_rtx_renderer_cfg

# params
# cam_model: Pinhole
# cam_width: 1600
# cam_height: 1200
# cam_fx: 586.3132095336914
# cam_fy: 586.3132095336914
# cam_cx: 791.9939422607422
# cam_cy: 608.2356262207031
# cam_k1: -0.010871673002839088
# cam_k2: -0.0071857706643640995
# cam_p1: 0.00019181467359885573
# cam_p2: 0.00020817259792238474
# LensType: Pinhole OpenCV
# Refer in Isaac Lab doc for more details.

EX001_CHASSISFRONT_PINHOLE_COLOR_CAMERA_SIM_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    spawn=OpenCVPinholeCameraCfg(
        clipping_range=(0.01, 1.0e5),
    ),
    width=1600,
    height=1200,
    data_types=["rgb"],
    renderer_cfg=isaac_rtx_renderer_cfg(),
)

EX001_CHASSISFRONT_PINHOLE_DEPTH_CAMERA_SIM_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    spawn=OpenCVPinholeCameraCfg(
        # Clipping planes (near, far) in meters
        clipping_range=(0.01, 1.0e5),
    ),
    # image size observation in IsaacLab
    width=1600,
    height=1200,
    data_types=["depth"],
    renderer_cfg=isaac_rtx_renderer_cfg(),
)

EX001_CHASSISFRONT_PINHOLE_INFRA1_CAMERA_SIM_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    spawn=OpenCVPinholeCameraCfg(
        clipping_range=(0.01, 1.0e5),
    ),
    width=1600,
    height=1200,
    data_types=["rgb"],
    renderer_cfg=isaac_rtx_renderer_cfg(),
)

EX001_CHASSISFRONT_PINHOLE_INFRA2_CAMERA_SIM_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    spawn=OpenCVPinholeCameraCfg(
        clipping_range=(0.01, 1.0e5),
    ),
    width=1600,
    height=1200,
    data_types=["rgb"],
    renderer_cfg=isaac_rtx_renderer_cfg(),
)

# NOTE:
# The corresponding camera prim for `EX001_CHASSISFRONT_PINHOLE_COLOR_CAMERA_CFG` is deactivated in the USD file.
EX001_CHASSISFRONT_PINHOLE_COLOR_CAMERA_CFG = EX001_CHASSISFRONT_PINHOLE_COLOR_CAMERA_SIM_CFG

EX001_CHASSISFRONT_PINHOLE_DEPTH_CAMERA_CFG = EX001_CHASSISFRONT_PINHOLE_DEPTH_CAMERA_SIM_CFG

# NOTE:
# The corresponding camera prim for `EX001_CHASSISFRONT_PINHOLE_INFRA1_CAMERA_CFG` is deactivated in the USD file.
EX001_CHASSISFRONT_PINHOLE_INFRA1_CAMERA_CFG = EX001_CHASSISFRONT_PINHOLE_INFRA1_CAMERA_SIM_CFG

# NOTE:
# The corresponding camera prim for `EX001_CHASSISFRONT_PINHOLE_INFRA2_CAMERA_CFG` is deactivated in the USD file.
EX001_CHASSISFRONT_PINHOLE_INFRA2_CAMERA_CFG = EX001_CHASSISFRONT_PINHOLE_INFRA2_CAMERA_SIM_CFG
