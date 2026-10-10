"""Camera configuration for EX001 right gripper."""

from isaaclab.sensors import CameraCfg

from manaenv.assets.sensors.async_camera import ManaAsyncCamera

from .update_camera import OpenCVFisheyeCameraCfg, isaac_rtx_renderer_cfg

# 640%YAML:1.0
# ---
# date: "2025-09-16 19:24:40"
# lens_model: fisheye_opencv
# image_width: 640
# image_height: 480
# camera_matrix: !!opencv-matrix
#    rows: 3
#    cols: 3
#    dt: d
#    data: [ 284.08757, 0., 338.93094938367117, 0.,
#        283.37960864663157, 252.22276405602674, 0., 0., 1. ]
# dist_coeffs: !!opencv-matrix
#    rows: 4
#    cols: 1
#    dt: d
#    data: [ -8.1941350731747527e-03, 1.2253087646124953e-02,
#        -2.1174845358304589e-02, 8.2523135831889787e-03 ]
# reprojection_error: 2.4750796842125575e-01
# LensType: Fisheye OpenCV
# Refer in Isaac Lab doc for more details.

EX001_RIGHTGRIPPER_FISHEYE_CAMERA_SIM_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    spawn=OpenCVFisheyeCameraCfg(
        # Clipping planes (near, far) in meters
        clipping_range=(0.01, 1.0e5),
    ),
    # image size observation in IsaacLab
    width=640,
    height=480,
    data_types=["rgb"],
    renderer_cfg=isaac_rtx_renderer_cfg(),
)

EX001_RIGHTGRIPPER_FISHEYE_CAMERA_CFG = EX001_RIGHTGRIPPER_FISHEYE_CAMERA_SIM_CFG
