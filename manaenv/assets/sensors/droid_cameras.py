"""Policy-calibrated camera bundle for the RoboLab DROID embodiment."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.utils.cameras import ArenaCameraCfg

from manaenv.assets.sensors.async_camera import ManaAsyncCamera

if TYPE_CHECKING:
    import isaaclab.sim as sim_utils


def _pinhole(focal_length: float) -> sim_utils.PinholeCameraCfg:
    import isaaclab.sim as sim_utils

    return sim_utils.PinholeCameraCfg(
        focal_length=focal_length,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
    )


DROID_WRIST_CAMERA_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
    update_period=0.0,
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=_pinhole(2.8),
    offset=CameraCfg.OffsetCfg(
        pos=(0.011, -0.031, -0.074),
        rot=(0.570, 0.576, -0.409, -0.420),
        convention="opengl",
    ),
)

DROID_EXTERIOR_1_CAMERA_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    prim_path="{ENV_REGEX_NS}/over_shoulder_left_camera",
    update_period=0.0,
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=_pinhole(2.1),
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, 0.57, 0.66),
        rot=(-0.195, 0.399, 0.805, -0.393),
        convention="opengl",
    ),
)

DROID_EXTERIOR_2_CAMERA_CFG = CameraCfg(
    class_type=ManaAsyncCamera,
    prim_path="{ENV_REGEX_NS}/over_shoulder_right_camera",
    update_period=0.0,
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=_pinhole(2.1),
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, -0.57, 0.66),
        rot=(0.399, -0.195, -0.393, 0.805),
        convention="opengl",
    ),
)


@configclass
class DroidCameraCfg(ArenaCameraCfg):
    _use_tiled_camera: ClassVar[bool] = False

    wrist_cam: CameraCfg = DROID_WRIST_CAMERA_CFG
    exterior_1_cam: CameraCfg = DROID_EXTERIOR_1_CAMERA_CFG
    exterior_2_cam: CameraCfg = DROID_EXTERIOR_2_CAMERA_CFG
