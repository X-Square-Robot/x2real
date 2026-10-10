"""Camera configuration for EX001 left gripper."""

from isaaclab.sensors import CameraCfg

from manaenv.assets.sensors.async_camera import ManaAsyncCamera

from .update_camera import OpenCVFisheyeCameraCfg, isaac_rtx_renderer_cfg

# IsaacLab Convension:
# https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_conventions.html

# SHPY WARNING:
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# \\\\\\\\\\DONT FUCKING CHANGE THE ORDER OF THE FARME OR ROTATION SYSTEM BELOW!!!!!!!\\\\\\\\\\\
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# SHPY Comment: coordinate system
# In Unreal Engine, the coordinate system is:
# X: Forward
# Y: Right
# Z: Up
# In Isaac Sim, the coordinate system is:
# X: Forward
# Y: Left
# Z: Up
# Therefore, when converting rotations from Unreal Engine to Isaac Sim,

# SHPY Comment: Rotation Order
# In Unreal Engine, the rotation order is ZYX
# In Isaac Sim, the rotation order is also ZYX (app.primCreation.DefaultRotationOrder = "ZYX")

# SHPY Comment: Quaternion (ALWAYS USE QUATERNION!!!!!!!!!!!!)
# In Unreal Engine, the quaternion is in the order (X, Y, Z, W)
# In Isaac Sim, the quaternion is also in the order (W, X, Y, Z)
# In URDF, the quaternion is in the order (W, X, Y, Z)
# In USD, the quaternion is in the order (W, X, Y, Z)
# In PhysX, the quaternion is in the order (X, Y, Z, W)

# SHPY Comment: Axis Convension
# https://zhuanlan.zhihu.com/p/1940774727674238008
# IsaacLab and Unreal Engine both use Tait-Bryan angles
# Proper Euler angles - DONT USE!!!

# SHPY Comment: intrinsic and extrinsic
# Intrinsic rotations: rotations about the axes of a coordinate system that is attached to the rotating body.
# Extrinsic rotations: rotations about the axes of a fixed coordinate system.
# In Unreal Engine and Isaac Sim, the rotations are Intrinsic rotations.
# Just remember if the convention is from local body or fixed world.
# Good consequence:
# ZYX intrinsic rotation is equivalent to XYZ extrinsic rotation.

# SHPY Comment: Roll Yaw Pitch
# URDF: RPY
# Roll: rotation about the forward axis
# Pitch: rotation about the right axis
# Yaw: rotation about the up axis
# Banded with the function implementation or frame convention.
# Be careful when using these terms.


# %YAML:1.0
# ---
# date: "2025-09-16 19:16:03"
# lens_model: fisheye_opencv
# image_width: 640
# image_height: 480
# camera_matrix: !!opencv-matrix
#    rows: 3
#    cols: 3
#    dt: d
#    data: [ 281.41, 0., 326.8136314159617, 0.,
#        281.67785674685077, 255.39720458939655, 0., 0., 1. ]
# dist_coeffs: !!opencv-matrix
#    rows: 4
#    cols: 1
#    dt: d
#    data: [ -8.5241697622606021e-04, 8.7306241502976903e-03,
#        -1.6219440091804906e-02, 6.9389101564615975e-03 ]
# reprojection_error: 3.0654943386213801e-01
# Calibration source is maintained with the local camera asset bundle.

# LensType: Fisheye OpenCV
# Refer fisheyeKannalaBrandtK3 in Isaac Sim doc for more details.


# The EX001 asset USD already authors its OpenCV fisheye intrinsics. Keep the
# optional intrinsic fields unset so the spawn config only overrides clipping.
EX001_LEFTGRIPPER_FISHEYE_CAMERA_SIM_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    spawn=OpenCVFisheyeCameraCfg(
        clipping_range=(0.01, 1.0e5),
    ),
    width=640,
    height=480,
    data_types=["rgb"],
    renderer_cfg=isaac_rtx_renderer_cfg(),
)

EX001_LEFTGRIPPER_FISHEYE_CAMERA_CFG = EX001_LEFTGRIPPER_FISHEYE_CAMERA_SIM_CFG
