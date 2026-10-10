"""Camera configurations for Franka Panda and Franka + Robotiq.

Franka does NOT ship with camera prims in the USD, so we spawn new
pinhole cameras. Both gripper variants share the DROID-aligned three-camera
bundle (wrist + two exterior shoulders). Only the wrist mount frame differs:

- **FrankaCameraCfg**: wrist on ``panda_hand``.
- **FrankaRobotiqCameraCfg**: wrist on Robotiq ``base_link``.

Intrinsics / resolution match across variants. Wrist extrinsics stay
frame-local: Robotiq base_link and panda_hand use different axes, so the
DROID wrist offset must not be copied onto ``panda_hand``.

Isaac Lab 3.0+ ``CameraCfg.OffsetCfg.rot`` is XYZW. Values below are the 5.1
WXYZ authoring quats reordered ``(w,x,y,z) -> (x,y,z,w)``.
"""

from __future__ import annotations

from typing import ClassVar

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.utils.cameras import ArenaCameraCfg

from manaenv.assets.sensors.async_camera import ManaAsyncCamera

FRANKA_WRIST_CAMERA_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
    # Match the render cadence (sim_fps=240, render_decremental=8 -> 30 Hz).
    # update_period=0.0 marks the camera outdated on every physics substep, so
    # its annotator gets read on pre-render substeps and latches a stale frame
    # (frozen video). 1/30 s aligns the sensor refresh with the actual render,
    # matching every other manaenv embodiment's camera config.
    update_period=0.0333,
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.8,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
    ),
    offset=CameraCfg.OffsetCfg(
        # panda_hand-local forward-facing pose (not the Robotiq/DROID offset).
        # 5.1 WXYZ (0.0, 0.9962, 0.0, 0.0872) -> XYZW
        pos=(0.08, 0.0, -0.05),
        rot=(0.9962, 0.0, 0.0872, 0.0),
        convention="opengl",
    ),
)
"""Wrist camera on panda_hand; DROID-matched intrinsics / resolution."""


FRANKA_ROBOTIQ_WRIST_CAMERA_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
    # Retain the current ManaEnv render cadence while aligning the camera
    # geometry and calibration with the single-arm DROID setup.
    update_period=0.0333,
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.8,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
    ),
    offset=CameraCfg.OffsetCfg(
        # 5.1 WXYZ (-0.420, 0.570, 0.576, -0.409) -> XYZW
        pos=(0.011, -0.031, -0.074),
        rot=(0.570, 0.576, -0.409, -0.420),
        convention="opengl",
    ),
)
"""DROID-aligned wrist camera on the Robotiq 2F-85 base link."""


FRANKA_EXTERIOR_1_CAMERA_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    # The reference DROID robot is authored at the environment origin, so its
    # world-space camera pose is also the camera-rig pose relative to the robot
    # base. Parent it to Robot here to preserve that layout when a task moves
    # or rotates the Franka base.
    prim_path="{ENV_REGEX_NS}/Robot/over_shoulder_left_camera",
    update_period=0.0333,
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.1,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
    ),
    offset=CameraCfg.OffsetCfg(
        # 5.1 WXYZ (-0.393, -0.195, 0.399, 0.805) -> XYZW
        pos=(0.05, 0.57, 0.66),
        rot=(-0.195, 0.399, 0.805, -0.393),
        convention="opengl",
    ),
)
"""DROID-aligned over-shoulder-left exterior camera."""


FRANKA_EXTERIOR_2_CAMERA_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    prim_path="{ENV_REGEX_NS}/Robot/over_shoulder_right_camera",
    update_period=0.0333,
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.1,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
    ),
    offset=CameraCfg.OffsetCfg(
        # 5.1 WXYZ (0.805, 0.399, -0.195, -0.393) -> XYZW
        pos=(0.05, -0.57, 0.66),
        rot=(0.399, -0.195, -0.393, 0.805),
        convention="opengl",
    ),
)
"""DROID-aligned over-shoulder-right exterior camera."""


@configclass
class FrankaCameraCfg(ArenaCameraCfg):
    """DROID-compatible three-camera bundle for Franka + Panda hand.

    Same intrinsics / exterior layout as :class:`FrankaRobotiqCameraCfg`;
    wrist mounts on ``panda_hand``. Runtime keys: ``wrist_cam_rgb``,
    ``exterior_1_cam_rgb``, ``exterior_2_cam_rgb``.
    """

    _use_tiled_camera: ClassVar[bool] = False

    wrist_cam: CameraCfg = FRANKA_WRIST_CAMERA_CFG
    exterior_1_cam: CameraCfg = FRANKA_EXTERIOR_1_CAMERA_CFG
    exterior_2_cam: CameraCfg = FRANKA_EXTERIOR_2_CAMERA_CFG


@configclass
class FrankaRobotiqCameraCfg(ArenaCameraCfg):
    """DROID-compatible three-camera bundle for Franka + Robotiq.

    Runtime observation keys are ``wrist_cam_rgb``,
    ``exterior_1_cam_rgb``, and ``exterior_2_cam_rgb``.
    """

    _use_tiled_camera: ClassVar[bool] = False

    wrist_cam: CameraCfg = FRANKA_ROBOTIQ_WRIST_CAMERA_CFG
    exterior_1_cam: CameraCfg = FRANKA_EXTERIOR_1_CAMERA_CFG
    exterior_2_cam: CameraCfg = FRANKA_EXTERIOR_2_CAMERA_CFG
