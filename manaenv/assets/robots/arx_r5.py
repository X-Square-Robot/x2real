"""ARX-R5 bimanual embodiment configuration for manaenv.

Dual 6-DOF arms with revolute gripper motor + mimic prismatic fingers.
Joint layout mirrors EX001-6R naming (left_arm_joint1..6) with different
gripper joint name and range.

- Arms: left_arm_joint[1-6] / right_arm_joint[1-6] (revolute)
- Gripper motor: left_gripper_motor_joint / right_gripper_motor_joint (revolute, 0~4.5 rad)
- EE frame: left_gripper_base_link / right_gripper_base_link
- Camera: left_gripper_camera_link/left_gripper_camera
- Contact: left_gripper_left_link, left_gripper_right_link
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_physx.sim.schemas import (
    PhysxArticulationRootPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)

from manaenv.actions.clamped_raw_gripper import ClampedRawGripperActionCfg
from manaenv.actions.command_rate_diff_ik import (
    CommandRateDifferentialInverseKinematicsActionCfg,
)
from manaenv.assets.resolver import resolve_runtime_asset_reference
from manaenv.assets.robots.components.embodiment_obs import (
    joint_pos_by_names,
    wallx_follow_from_init_ee_root_local,
)
from manaenv.assets.sensors.async_camera import ManaAsyncCamera
from manaenv.assets.sensors.update_camera import OpenCVFisheyeCameraCfg

if TYPE_CHECKING:
    from isaaclab.managers import ActionTermCfg
    from isaaclab_arena.utils.pose import Pose

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

_ARX_R5_GRIPPER_OPEN = 4.5
_ARX_R5_GRIPPER_CLOSE = 0.0


def _arx_r5_usd_path() -> str:
    override = os.environ.get("MANAENV_ARX_R5_USD_PATH", "").strip()
    if override:
        return override
    return "assets/robots/arx_r5/arx_r5.usd"


# ═══════════════════════════════════════════════════════════════════════════════
#  Embodiment
# ═══════════════════════════════════════════════════════════════════════════════


@register_asset
class ARXR5Embodiment(EmbodimentBase):
    """ARX-R5 bimanual robot (fixed-base, dual 6-DOF arms + revolute gripper motor)."""

    name = "ARX-R5"

    @configclass
    class SceneCfg:
        robot: ArticulationCfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=resolve_runtime_asset_reference(_arx_r5_usd_path()),
                activate_contact_sensors=True,
                semantic_tags=[("class", "arx_r5")],
                rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=PhysxArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    fix_root_link=True,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                ),
            ),
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.32165, -0.25918, -0.25),
                rot=(0.0, 0.0, 0.0, 1.0),
                joint_pos={
                    "left_arm_joint.*": 0.0,
                    "right_arm_joint.*": 0.0,
                    "left_gripper_motor_joint": 0.0,
                    "right_gripper_motor_joint": 0.0,
                },
            ),
            actuators={
                "left_arm_acts": ImplicitActuatorCfg(
                    joint_names_expr=["left_arm_joint[1-6]"],
                    effort_limit_sim=200.0,
                    stiffness=1500.0,
                    damping=150.0,
                ),
                "right_arm_acts": ImplicitActuatorCfg(
                    joint_names_expr=["right_arm_joint[1-6]"],
                    effort_limit_sim=200.0,
                    stiffness=1500.0,
                    damping=150.0,
                ),
                "left_gripper_acts": ImplicitActuatorCfg(
                    joint_names_expr=["left_gripper_motor_joint"],
                    effort_limit_sim=200.0,
                    stiffness=500.0,
                    damping=50.0,
                ),
                "right_gripper_acts": ImplicitActuatorCfg(
                    joint_names_expr=["right_gripper_motor_joint"],
                    effort_limit_sim=200.0,
                    stiffness=500.0,
                    damping=50.0,
                ),
            },
        )

        left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_gripper_base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_gripper_base_link",
                    name="end_effector",
                )
            ],
        )
        right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_gripper_base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_gripper_base_link",
                    name="right_end_effector",
                )
            ],
        )

        left_gripper_contact: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_gripper_.*_link",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
        )
        right_gripper_contact: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_gripper_.*_link",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
        )

    @configclass
    class CameraCfg(ArenaCameraCfg):
        _use_tiled_camera: ClassVar[bool] = False

        left_wrist_camera: CameraCfg = CameraCfg(
            class_type=ManaAsyncCamera,
            prim_path="{ENV_REGEX_NS}/Robot/left_gripper_camera_link/left_gripper_camera",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=OpenCVFisheyeCameraCfg(
                width=1920,
                height=1080,
                fx=650.9902530306058,
                fy=651.1395485464623,
                cx=941.1478602082152,
                cy=529.5616964687518,
                distortion_coefficients=[
                    -0.004838782348073652,
                    0.0007695240290567486,
                    -0.004532544140904039,
                    0.001519397476671249,
                ],
            ),
        )
        right_wrist_camera: CameraCfg = CameraCfg(
            class_type=ManaAsyncCamera,
            prim_path="{ENV_REGEX_NS}/Robot/right_gripper_camera_link/right_gripper_camera",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=OpenCVFisheyeCameraCfg(
                width=1920,
                height=1080,
                fx=638.1530743244554,
                fy=637.9968660747421,
                cx=960.2023882226367,
                cy=521.7980505129706,
                distortion_coefficients=[
                    -0.006487174251380216,
                    -0.0035663975460442357,
                    -0.006816629867962863,
                    0.003694025631483343,
                ],
            ),
        )
        head_camera: CameraCfg = CameraCfg(
            class_type=ManaAsyncCamera,
            prim_path="{ENV_REGEX_NS}/Robot/Head_Camera",
            update_period=0.0333,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.97,
                horizontal_aperture=5.54,
                vertical_aperture=3.09,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.084, 0.0, 0.517),
                rot=(0.21263, -0.21263, -0.67438, 0.67438),
                convention="opengl",
            ),
        )

    # ── Actions: delta IK mode ───────────────────────────────────────────────

    @configclass
    class ActionsCfg:
        arm_action: ActionTermCfg = CommandRateDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint[1-6]"],
            body_name="left_gripper_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=1.0,
        )
        gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["left_gripper_motor_joint"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_ARX_R5_GRIPPER_CLOSE,
            clamp_max=_ARX_R5_GRIPPER_OPEN,
        )
        right_arm_action: ActionTermCfg = CommandRateDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint[1-6]"],
            body_name="right_gripper_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=1.0,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["right_gripper_motor_joint"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_ARX_R5_GRIPPER_CLOSE,
            clamp_max=_ARX_R5_GRIPPER_OPEN,
        )

    # ── Actions: absolute IK mode ───────────────────────────────────────────

    @configclass
    class ActionsCfgAbsIK:
        arm_action: ActionTermCfg = CommandRateDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint[1-6]"],
            body_name="left_gripper_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            scale=1.0,
        )
        gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["left_gripper_motor_joint"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_ARX_R5_GRIPPER_CLOSE,
            clamp_max=_ARX_R5_GRIPPER_OPEN,
        )
        right_arm_action: ActionTermCfg = CommandRateDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint[1-6]"],
            body_name="right_gripper_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            scale=1.0,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["right_gripper_motor_joint"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_ARX_R5_GRIPPER_CLOSE,
            clamp_max=_ARX_R5_GRIPPER_OPEN,
        )

    # ── Actions: joint-position policy mode ──────────────────────────────────

    @configclass
    class JointActionsCfg:
        left_arm_action: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint[1-6]"],
            scale=1.0,
            use_default_offset=False,
        )
        left_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["left_gripper_motor_joint"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_ARX_R5_GRIPPER_CLOSE,
            clamp_max=_ARX_R5_GRIPPER_OPEN,
        )
        right_arm_action: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint[1-6]"],
            scale=1.0,
            use_default_offset=False,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["right_gripper_motor_joint"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_ARX_R5_GRIPPER_CLOSE,
            clamp_max=_ARX_R5_GRIPPER_OPEN,
        )

    # ── Observations ─────────────────────────────────────────────────────────

    @configclass
    class StateObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            follow_left_joint_pos = ObsTerm(
                func=joint_pos_by_names,
                params={"joint_names": [r"left_arm_joint[1-6]", "left_gripper_motor_joint"]},
            )
            follow_right_joint_pos = ObsTerm(
                func=joint_pos_by_names,
                params={"joint_names": [r"right_arm_joint[1-6]", "right_gripper_motor_joint"]},
            )
            follow1_pos = ObsTerm(
                func=wallx_follow_from_init_ee_root_local,
                params={
                    "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
                    "gripper_joint_name": "left_gripper_motor_joint",
                    "ref_key": "_left_ee_ref",
                },
            )
            follow2_pos = ObsTerm(
                func=wallx_follow_from_init_ee_root_local,
                params={
                    "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
                    "gripper_joint_name": "right_gripper_motor_joint",
                    "ref_key": "_right_ee_ref",
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class CameraObservationsCfg:
        @configclass
        class CameraObsCfg(ObsGroup):
            left_wrist_cam = ObsTerm(
                func=mdp_isaac_lab.image,
                params={
                    "sensor_cfg": SceneEntityCfg("left_wrist_camera"),
                    "data_type": "rgb",
                    "normalize": False,
                },
            )
            right_wrist_cam = ObsTerm(
                func=mdp_isaac_lab.image,
                params={
                    "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                    "data_type": "rgb",
                    "normalize": False,
                },
            )
            head_cam = ObsTerm(
                func=mdp_isaac_lab.image,
                params={
                    "sensor_cfg": SceneEntityCfg("head_camera"),
                    "data_type": "rgb",
                    "normalize": False,
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        camera_obs: CameraObsCfg = CameraObsCfg()

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        gripper_mode: str = "raw_joint",
    ):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.scene_config = self.SceneCfg()
        self.camera_config = self.CameraCfg()
        gm = (gripper_mode or "raw_joint").strip().lower()
        if gm == "joint_position":
            self.action_config = self.JointActionsCfg()
        else:
            self.action_config = self.ActionsCfgAbsIK()
        self.gripper_mode = gm
        self.observation_config = self.StateObservationsCfg()
        self._camera_observation_config = self.CameraObservationsCfg()

    def get_ee_contact_prim_paths(self, arm_mode: str | None = None) -> list[str]:
        """Return EE finger prim paths for task gripper-contact sensor filters."""
        del arm_mode
        return [
            "{ENV_REGEX_NS}/Robot/left_gripper_left_link",
            "{ENV_REGEX_NS}/Robot/left_gripper_right_link",
            "{ENV_REGEX_NS}/Robot/right_gripper_left_link",
            "{ENV_REGEX_NS}/Robot/right_gripper_right_link",
        ]

    def get_gripper_states(self) -> dict[str, float]:
        """Return the native gripper range used by policy-space conversion."""

        return {"open": _ARX_R5_GRIPPER_OPEN, "closed": _ARX_R5_GRIPPER_CLOSE}

    def get_observation_cfg(self):
        from isaaclab_arena.utils.configclass import combine_configclass_instances

        if self.enable_cameras:
            return combine_configclass_instances(
                "ObservationCfg",
                self.observation_config,
                self._camera_observation_config,
            )
        return self.observation_config
