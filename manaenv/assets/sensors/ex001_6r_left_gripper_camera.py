"""Camera configuration for EX001-6R left gripper.

Calibration source: nominal X2Robot 5756 wrist fisheye module spec
(640x480@30fps MJPG, EFL 2.5 mm, FOV H137.2 V101 D179.2). The shared
``make_x2robot_5756_fisheye_camera_cfg`` builds the ideal equidistant fisheye
intrinsics from the datasheet; both wrists reuse it until per-unit results land.

Provenance: hardware module CZB2F5756-V1; datasheet
Calibration details are maintained with the local camera asset bundle. Replace with
EX001-6R left wrist per-unit calibration once intrinsics_calibration_v2 lands.
"""

from manaenv.assets.sensors.x2robot_5756_fisheye_camera import make_x2robot_5756_fisheye_camera_cfg

EX001_6R_LEFTGRIPPER_FISHEYE_CAMERA_SIM_CFG = make_x2robot_5756_fisheye_camera_cfg()

EX001_6R_LEFTGRIPPER_FISHEYE_CAMERA_CFG = EX001_6R_LEFTGRIPPER_FISHEYE_CAMERA_SIM_CFG
