"""EX001 whole-body embodiment (mobile base + dual 6-DOF arms + grippers + lift).

Joint/link names: arms ``*_arm_joint[1-6]``, grippers ``*_arm_gripper``
([0, 1.89]), EE links ``*_arm_gripper_base_link``, wheels
``left/right_wheel_joint``, prismatic ``lift_joint`` ([0, 0.78] m).

Absolute-IK whole-body action layout::

    [L_pos(3), L_quat(4), L_grip(1), R_pos(3), R_quat(4), R_grip(1),
     L_wheel(1), R_wheel(1), lift(1)]  = 19D

Dual-arm X2Robot IK reuses the EX001-6R Pinocchio URDF / joint chain / gains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg, PhysxRigidBodyPropertiesCfg

from manaenv.actions.clamped_raw_gripper import ClampedRawGripperActionCfg
from manaenv.actions.safe_effort_overlay import SafeEffortOverlayActionCfg
from manaenv.actions.x2robot_ik_action import X2RobotIKActionCfg
from manaenv.assets.resolver import resolve_runtime_asset_reference
from manaenv.assets.robots.components.embodiment_obs import (
    base_velocity_decomposed_odom,
    joint_pos_by_names,
    wallx_follow_from_init_ee_lift_local,
)
from manaenv.assets.robots.ex001_6r import (
    _EX001_6R_LEFT_ARM_JOINTS,
    _EX001_6R_RIGHT_ARM_JOINTS,
    _EX001_6R_X2ROBOT_IK_ARM_KD,
    _EX001_6R_X2ROBOT_IK_ARM_KP,
)
from manaenv.assets.robots.ex001_6r import _resolve_ik_urdf_paths as _resolve_ex001_arm_ik_urdf_paths
from manaenv.assets.sensors.async_camera import ManaAsyncCamera
from manaenv.assets.sensors.ex001_chassis_front_camera import EX001_CHASSISFRONT_PINHOLE_DEPTH_CAMERA_CFG
from manaenv.assets.sensors.ex001_head_front_camera import EX001_HEADFRONT_PINHOLE_COLOR_CAMERA_CFG
from manaenv.assets.sensors.ex001_imu import EX001_IMU_CFG
from manaenv.assets.sensors.ex001_left_gripper_camera import EX001_LEFTGRIPPER_FISHEYE_CAMERA_CFG
from manaenv.assets.sensors.ex001_right_gripper_camera import EX001_RIGHTGRIPPER_FISHEYE_CAMERA_CFG
from manaenv.config.ex001_prim_paths import EX001_PATHS

if TYPE_CHECKING:
    from isaaclab.managers import ActionTermCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab_arena.utils.pose import Pose

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

_EX001_V005_USD = resolve_runtime_asset_reference("assets/simready/robots/ex001/v00.05.05/ex001.usd")

_EX001_GRIPPER_OPEN = 1.89
_EX001_GRIPPER_CLOSE = 0.0

_EX001_LIFT_LOWER = 0.0
_EX001_LIFT_UPPER = 0.78
_EX001_LIFT_KP = 10000.0
_EX001_LIFT_KD = 2000.0
_EX001_LIFT_GRAVITY_FF_N = 145.0
_EX001_LIFT_GRAVITY_FF_MAX_N = 250.0
_EX001_LIFT_GRAVITY_FF_RAMP_S = 0.4


def _ex001_lift_gravity_overlay_cfg() -> SafeEffortOverlayActionCfg:
    """Return a 0-D lift gravity feed-forward action configuration."""
    return SafeEffortOverlayActionCfg(
        asset_name="robot",
        enabled=True,
        overlay_joint_names=("lift_joint",),
        effort_n=_EX001_LIFT_GRAVITY_FF_N,
        max_effort_n=_EX001_LIFT_GRAVITY_FF_MAX_N,
        ramp_s=_EX001_LIFT_GRAVITY_FF_RAMP_S,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Embodiment
# ═══════════════════════════════════════════════════════════════════════════════


@register_asset
class EX001Embodiment(EmbodimentBase):
    """EX001 whole-body mobile manipulator."""

    name = "ex001"

    # ── Scene ────────────────────────────────────────────────────────────

    @configclass
    class SceneCfg:
        robot: ArticulationCfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=_EX001_V005_USD,
                activate_contact_sensors=True,
                rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=False),
                articulation_props=PhysxArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    fix_root_link=False,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=0,
                ),
            ),
            prim_path="{ENV_REGEX_NS}/Robot",
            # Mirror USD authored joint state (ex001 v00.05.05: lift 0.379 m,
            # head_pitch -28.65 deg = -0.5 rad). Isaac Lab zeros unspecified
            # init_state.joint_pos, which otherwise drops the USD defaults on
            # reset_scene_to_default (P10D1/B1). Not a runtime joint-map event.
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                joint_pos={
                    "lift_joint": 0.379,
                    "head_pitch_joint": -0.5,
                },
            ),
            actuators={
                "wheels": ImplicitActuatorCfg(
                    joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
                    effort_limit=5000.0,
                    velocity_limit=5.0,
                    stiffness=0.0,
                    damping=5000.0,
                ),
                "lift_acts": ImplicitActuatorCfg(
                    joint_names_expr=["lift_joint"],
                    effort_limit=2000.0,
                    velocity_limit=5.0,
                    stiffness=_EX001_LIFT_KP,
                    damping=_EX001_LIFT_KD,
                ),
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
                    joint_names_expr=["left_arm_gripper"],
                    effort_limit_sim=200.0,
                    stiffness=200.0,
                    damping=30.0,
                ),
                "right_gripper_acts": ImplicitActuatorCfg(
                    joint_names_expr=["right_arm_gripper"],
                    effort_limit_sim=200.0,
                    stiffness=200.0,
                    damping=30.0,
                ),
                "head_acts": ImplicitActuatorCfg(
                    joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
                    effort_limit_sim=200.0,
                    stiffness=1500.0,
                    damping=150.0,
                ),
            },
        )

        # ---- EE frame transformers -----------------------------------------
        left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_base_link",
                    name="end_effector",
                )
            ],
        )
        right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_base_link",
                    name="right_end_effector",
                )
            ],
        )

        # ---- contact sensors ------------------------------------------------
        left_gripper_contact: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_.*_link",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
        )
        right_gripper_contact: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_.*_link",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
        )

        # ---- IMU ------------------------------------------------------------
        imu = EX001_IMU_CFG.replace(prim_path=EX001_PATHS.imu)

    # ── Cameras ──────────────────────────────────────────────────────────

    @configclass
    class CameraCfg(ArenaCameraCfg):
        _use_tiled_camera: ClassVar[bool] = False

        left_wrist_camera: CameraCfg = EX001_LEFTGRIPPER_FISHEYE_CAMERA_CFG.replace(
            prim_path=EX001_PATHS.left_gripper,
            update_period=0.0333,
            class_type=ManaAsyncCamera,
        )
        right_wrist_camera: CameraCfg = EX001_RIGHTGRIPPER_FISHEYE_CAMERA_CFG.replace(
            prim_path=EX001_PATHS.right_gripper,
            update_period=0.0333,
            class_type=ManaAsyncCamera,
        )
        head_camera: CameraCfg = EX001_HEADFRONT_PINHOLE_COLOR_CAMERA_CFG.replace(
            prim_path=EX001_PATHS.head_front_color,
            update_period=0.0333,
            class_type=ManaAsyncCamera,
        )
        chassis_camera: CameraCfg = EX001_CHASSISFRONT_PINHOLE_DEPTH_CAMERA_CFG.replace(
            prim_path=EX001_PATHS.chassis_front_depth,
            update_period=0.0333,
            class_type=ManaAsyncCamera,
        )

    # ── Actions: absolute IK (19D default) ────────────────────────────────

    @configclass
    class ActionsCfgAbsIK:
        """19D abs-IK + diff-drive base + lift.

        Layout: ``[L_pos(3), L_quat(4), L_grip(1), R_pos(3), R_quat(4),
        R_grip(1), L_wheel(1), R_wheel(1), lift(1)]``.
        """

        arm_action: ActionTermCfg = X2RobotIKActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint[1-6]"],
            body_name="left_arm_gripper_base_link",
            base_frame="base_platform_link",
            isaac_base_body="lift_link",
            ee_frame="left_arm_gripper_base_link",
            ik_joint_names=list(_EX001_6R_LEFT_ARM_JOINTS),
            command_type="pose",
            use_relative_mode=False,
            pose_command_format="quat_xyzw",
            scale=1.0,
        )
        gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["left_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_EX001_GRIPPER_CLOSE,
            clamp_max=_EX001_GRIPPER_OPEN,
        )
        right_arm_action: ActionTermCfg = X2RobotIKActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint[1-6]"],
            body_name="right_arm_gripper_base_link",
            base_frame="base_platform_link",
            isaac_base_body="lift_link",
            ee_frame="right_arm_gripper_base_link",
            ik_joint_names=list(_EX001_6R_RIGHT_ARM_JOINTS),
            command_type="pose",
            use_relative_mode=False,
            pose_command_format="quat_xyzw",
            scale=1.0,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["right_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_EX001_GRIPPER_CLOSE,
            clamp_max=_EX001_GRIPPER_OPEN,
        )
        base_action: ActionTermCfg = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel_joint", "right_wheel_joint"],
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )
        lift_action: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["lift_joint"],
            scale=1.0,
            use_default_offset=False,
        )
        lift_gravity_overlay: ActionTermCfg = _ex001_lift_gravity_overlay_cfg()

    # ── Actions: X2Robot IK (VR gen_method=2; same arms as EX001-6R) ─────

    @configclass
    class X2RobotIKActionsCfg:
        """X2Robot dual-arm IK + grippers + diff-drive + lift (19D).

        Arm IK matches EX001-6R (same URDF chain / EE / gains). Layout matches
        ``ActionsCfgAbsIK`` so external action adapters keep interleaved packing::

            [L7, Lg, R7, Rg, L_wheel, R_wheel, lift]
        """

        arm_action: ActionTermCfg = X2RobotIKActionCfg(
            asset_name="robot",  # ty: ignore[unknown-argument] -- inherited @configclass field
            joint_names=["left_arm_joint[1-6]"],
            body_name="left_arm_gripper_base_link",
            base_frame="base_platform_link",
            isaac_base_body="lift_link",
            ee_frame="left_arm_gripper_base_link",
            ik_joint_names=list(_EX001_6R_LEFT_ARM_JOINTS),
            command_type="pose",
            use_relative_mode=False,
            pose_command_format="quat_xyzw",
            scale=1.0,
        )
        gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["left_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_EX001_GRIPPER_CLOSE,
            clamp_max=_EX001_GRIPPER_OPEN,
        )
        right_arm_action: ActionTermCfg = X2RobotIKActionCfg(
            asset_name="robot",  # ty: ignore[unknown-argument] -- inherited @configclass field
            joint_names=["right_arm_joint[1-6]"],
            body_name="right_arm_gripper_base_link",
            base_frame="base_platform_link",
            isaac_base_body="lift_link",
            ee_frame="right_arm_gripper_base_link",
            ik_joint_names=list(_EX001_6R_RIGHT_ARM_JOINTS),
            command_type="pose",
            use_relative_mode=False,
            pose_command_format="quat_xyzw",
            scale=1.0,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["right_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_EX001_GRIPPER_CLOSE,
            clamp_max=_EX001_GRIPPER_OPEN,
        )
        base_action: ActionTermCfg = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel_joint", "right_wheel_joint"],
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )
        lift_action: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["lift_joint"],
            scale=1.0,
            use_default_offset=False,
        )
        lift_gravity_overlay: ActionTermCfg = _ex001_lift_gravity_overlay_cfg()

    # ── Actions: navigation (joint pos + wheel velocity) ────────────────

    @configclass
    class ActionsCfgNav:
        """Joint-position arms/lift/grippers + diff-drive wheel velocity.

        ``joint_pos`` covers every non-wheel joint with absolute position
        targets. ``base_action`` drives wheels by velocity.
        """

        joint_pos: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["^(?!left_wheel_joint$|right_wheel_joint$).*"],
            scale=1.0,
            use_default_offset=False,
        )
        base_action: ActionTermCfg = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel_joint", "right_wheel_joint"],
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )

    # ── Actions: wallx whole-body inference (21D) ────────────────────

    @configclass
    class ActionsCfgWallxWholebody:
        """21D action layout for EX001 whole-body wallx inference.

        Layout: ``[L_pos(3), L_quat(4), L_grip(1), R_pos(3), R_quat(4),
        R_grip(1), L_wheel(1), R_wheel(1), lift(1), head_yaw(1),
        head_pitch(1)]``.  Same as ``ActionsCfgAbsIK`` plus ``head_action``.
        """

        arm_action: ActionTermCfg = X2RobotIKActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint[1-6]"],
            body_name="left_arm_gripper_base_link",
            base_frame="base_platform_link",
            isaac_base_body="lift_link",
            ee_frame="left_arm_gripper_base_link",
            ik_joint_names=list(_EX001_6R_LEFT_ARM_JOINTS),
            command_type="pose",
            use_relative_mode=False,
            pose_command_format="quat_xyzw",
            scale=1.0,
        )
        gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["left_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_EX001_GRIPPER_CLOSE,
            clamp_max=_EX001_GRIPPER_OPEN,
        )
        right_arm_action: ActionTermCfg = X2RobotIKActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint[1-6]"],
            body_name="right_arm_gripper_base_link",
            base_frame="base_platform_link",
            isaac_base_body="lift_link",
            ee_frame="right_arm_gripper_base_link",
            ik_joint_names=list(_EX001_6R_RIGHT_ARM_JOINTS),
            command_type="pose",
            use_relative_mode=False,
            pose_command_format="quat_xyzw",
            scale=1.0,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["right_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_EX001_GRIPPER_CLOSE,
            clamp_max=_EX001_GRIPPER_OPEN,
        )
        base_action: ActionTermCfg = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel_joint", "right_wheel_joint"],
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )
        lift_action: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["lift_joint"],
            scale=1.0,
            use_default_offset=False,
        )
        head_action: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_yaw_joint", "head_pitch_joint"],
            scale=1.0,
            use_default_offset=False,
        )
        lift_gravity_overlay: ActionTermCfg = _ex001_lift_gravity_overlay_cfg()

    # ── Observations ─────────────────────────────────────────────────────

    @configclass
    class StateObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            joint_vel = ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            # Term names must stay root_*_w: ROS bridge / odom / TF read these keys
            # from observation_manager.compute() (see manaenv.ros_utils.sim_utils.get_root_pose).
            root_pos_w = ObsTerm(
                func=mdp.root_pos_w,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            root_quat_w = ObsTerm(
                func=mdp.root_quat_w,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )

            # ── Wallx whole-body obs (EX001 inference) ──
            follow1_pos = ObsTerm(
                func=wallx_follow_from_init_ee_lift_local,
                params={
                    "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
                    "gripper_joint_name": "left_arm_gripper",
                    "ref_key": "_left_ee_ref",
                },
            )
            follow2_pos = ObsTerm(
                func=wallx_follow_from_init_ee_lift_local,
                params={
                    "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
                    "gripper_joint_name": "right_arm_gripper",
                    "ref_key": "_right_ee_ref",
                },
            )
            velocity_decomposed_odom = ObsTerm(func=base_velocity_decomposed_odom)
            lift = ObsTerm(
                func=joint_pos_by_names,
                params={"joint_names": ["lift_joint"]},
            )
            head_pos = ObsTerm(
                func=joint_pos_by_names,
                params={"joint_names": ["head_yaw_joint", "head_pitch_joint"]},
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
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("left_wrist_camera"),
                    "data_type": "rgb",
                    "normalize": False,
                },
            )
            right_wrist_cam = ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                    "data_type": "rgb",
                    "normalize": False,
                },
            )
            head_cam = ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("head_camera"),
                    "data_type": "rgb",
                    "normalize": False,
                },
            )
            chassis_cam = ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("chassis_camera"),
                    "data_type": "depth",
                    "normalize": False,
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        camera_obs: CameraObsCfg = CameraObsCfg()

    # ── __init__ ─────────────────────────────────────────────────────────

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
    ):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.scene_config = self.SceneCfg()
        self.camera_config = self.CameraCfg() if enable_cameras else None
        self.use_abs_ik_action_config()
        self.observation_config = self.StateObservationsCfg()
        self._camera_observation_config = self.CameraObservationsCfg()

    # ── Public API ───────────────────────────────────────────────────────

    def get_observation_cfg(self):
        from isaaclab_arena.utils.configclass import combine_configclass_instances

        if self.enable_cameras:
            return combine_configclass_instances(
                "ObservationCfg",
                self.observation_config,
                self._camera_observation_config,
            )
        return self.observation_config

    def use_x2robot_ik_action_config(self) -> X2RobotIKActionsCfg:
        """Reuse EX001-6R arm IK URDF/gains; keep whole-body wheels + lift."""

        urdf_path, _ = _resolve_ex001_arm_ik_urdf_paths()
        cfg = self.X2RobotIKActionsCfg()
        cfg.arm_action.urdf_path = urdf_path
        cfg.right_arm_action.urdf_path = urdf_path
        self.action_config = cfg
        for group in ("left_arm_acts", "right_arm_acts"):
            act = self.scene_config.robot.actuators.get(group)
            if act is not None and hasattr(act, "stiffness") and hasattr(act, "damping"):
                act.stiffness = _EX001_6R_X2ROBOT_IK_ARM_KP
                act.damping = _EX001_6R_X2ROBOT_IK_ARM_KD
        return cfg

    def use_abs_ik_action_config(self) -> ActionsCfgAbsIK:
        """Configure the default absolute-IK layout with runtime-local assets."""
        urdf_path, _ = _resolve_ex001_arm_ik_urdf_paths()
        cfg = self.ActionsCfgAbsIK()
        cfg.arm_action.urdf_path = urdf_path
        cfg.right_arm_action.urdf_path = urdf_path
        self.action_config = cfg
        for group in ("left_arm_acts", "right_arm_acts"):
            actuator = self.scene_config.robot.actuators.get(group)
            if actuator is not None and hasattr(actuator, "stiffness") and hasattr(actuator, "damping"):
                actuator.stiffness = _EX001_6R_X2ROBOT_IK_ARM_KP
                actuator.damping = _EX001_6R_X2ROBOT_IK_ARM_KD
        return cfg

    def use_wallx_wholebody_action_config(self) -> ActionsCfgWallxWholebody:
        """Configure the 21-D policy action with runtime-local X2Robot IK assets."""
        urdf_path, _ = _resolve_ex001_arm_ik_urdf_paths()
        cfg = self.ActionsCfgWallxWholebody()
        cfg.arm_action.urdf_path = urdf_path
        cfg.right_arm_action.urdf_path = urdf_path
        self.action_config = cfg
        for group in ("left_arm_acts", "right_arm_acts"):
            actuator = self.scene_config.robot.actuators.get(group)
            if actuator is not None and hasattr(actuator, "stiffness") and hasattr(actuator, "damping"):
                actuator.stiffness = _EX001_6R_X2ROBOT_IK_ARM_KP
                actuator.damping = _EX001_6R_X2ROBOT_IK_ARM_KD
        return cfg

    def get_vr_gripper_clamp(self) -> dict[str, tuple[float, float]]:
        return {
            "left_arm_gripper": (_EX001_GRIPPER_CLOSE, _EX001_GRIPPER_OPEN),
            "right_arm_gripper": (_EX001_GRIPPER_CLOSE, _EX001_GRIPPER_OPEN),
        }

    def get_gripper_states(self) -> dict[str, float]:
        """Return the whole-body G gripper limits in simulator command units.

        The shared PolicySpace gripper-convention handshake reads this hook to
        convert a model's normalized ``[0, 1]`` commands before they reach the
        raw joint action.  This USD is the v00.05.05 whole-body G asset; do not
        duplicate this scale in a policy-specific adapter.
        """
        return {"closed": _EX001_GRIPPER_CLOSE, "open": _EX001_GRIPPER_OPEN}

    def get_vr_gripper_joint_names(self) -> tuple[str, str]:
        return "left_arm_gripper", "right_arm_gripper"

    def get_vr_ee_frame_names(self) -> tuple[str, str]:
        return "left_ee_frame", "right_ee_frame"

    def get_vr_lift_joint_name(self) -> str:
        return "lift_joint"

    def get_vr_lift_limits(self) -> tuple[float, float]:
        return _EX001_LIFT_LOWER, _EX001_LIFT_UPPER
