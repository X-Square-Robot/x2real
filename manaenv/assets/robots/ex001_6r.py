"""EX001-6R embodiment configuration for manaenv.

Aligned with IsaacLab-Arena EX001-6R:
- Clamped raw gripper targets [0, open], not binary open/close commands
  (``open`` depends on gripper variant: H=5.717175, G=1.89)
- FrameTransformer at gripper_base_link (offset=0)
- Contact sensors for gripper fingers

Observations (minimal):
- 14 follow joint positions (left_arm*6 + left_gripper*1 + right_arm*6 + right_gripper*1)
- Wallx wire ``follow1_pos`` / ``follow2_pos``: 7-D init-pose-relative root-local pos + Euler XYZ + raw gripper
  (policy action transforms convert returned rotations to quat XYZW internally)
- Left/right wrist + head camera RGB

Optional (opt-in): ``distance_to_image_plane`` (Z-depth in camera frame) and
``semantic_segmentation`` on the same three cameras when ``enable_cameras`` is
true and ``camera_depth_semantic=True``.
"""

from __future__ import annotations

import os
from dataclasses import fields
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
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg, PhysxRigidBodyPropertiesCfg

from manaenv.actions.clamped_raw_gripper import ClampedRawGripperActionCfg
from manaenv.actions.command_rate_diff_ik import (
    CommandRateDifferentialInverseKinematicsActionCfg,
    configure_diff_ik_action_cfg,
    normalize_diff_ik_compute_mode,
)
from manaenv.actions.x2robot_ik_action import X2RobotIKActionCfg
from manaenv.actuators.select import apply_actuator
from manaenv.assets.local_discovery import resolve_usd_path
from manaenv.assets.resolver import (
    is_asset_reference,
    resolve_asset_directory,
    resolve_asset_path,
    resolve_runtime_asset_reference,
)
from manaenv.assets.robots.arm_mode import ArmMode
from manaenv.assets.robots.components.embodiment_obs import joint_pos_by_names
from manaenv.assets.robots.components.ex001_6r_gripper_from_mimic_finger import (
    EX001_6R_G_GRIPPER_MIMIC_FINGER_LINEAR,
    EX001_6R_G_GRIPPER_MIMIC_OFFSET_DEG,
    EX001_6R_G_GRIPPER_MIMIC_SCALE,
    EX001_6R_H_GRIPPER_MIMIC_FINGER_LINEAR,
    EX001_6R_H_GRIPPER_MIMIC_OFFSET_DEG,
    EX001_6R_H_GRIPPER_MIMIC_SCALE,
    wallx_follow_from_init_ee_root_local_mimic_gripper,
)
from manaenv.assets.sensors.async_camera import ManaAsyncCamera
from manaenv.assets.sensors.ex001_6r_left_gripper_camera import EX001_6R_LEFTGRIPPER_FISHEYE_CAMERA_CFG
from manaenv.assets.sensors.ex001_6r_right_gripper_camera import EX001_6R_RIGHTGRIPPER_FISHEYE_CAMERA_CFG

if TYPE_CHECKING:
    from isaaclab.managers import ActionTermCfg
    from isaaclab_arena.utils.pose import Pose

# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

# One-line gripper switch: "H" (H-jaw legacy) or "G" (G-jaw v00.05.05).
_EX001_6R_GRIPPER_VARIANT = "G"

# Library-relative base for the EX001-6R USD builds (one subdir per version). A definition's
# ``robot_generator.usd_path`` resolves against it (see :func:`_resolve_ex001_6r_usd`).
_EX001_6R_ROBOT_BASE = "assets/simready/robots/ex001"
_EX001_6R_DEFAULT_VERSION = "v00.04.04"
_EX001_6R_USD_DIR = f"{_EX001_6R_ROBOT_BASE}/{_EX001_6R_DEFAULT_VERSION}"

_EX001_6R_PINNED_USD = f"{_EX001_6R_USD_DIR}/ex001_6r.usd"

# The legacy USD has narrower arm spacing and matches the CVPR data collection setup.
_EX001_6R_OVERRIDE_USD = f"{_EX001_6R_USD_DIR}/ex001_6r_legacy.usd"


def _resolve_ex001_6r_usd(usd_path: str) -> str:
    """Expand a short ``robot_generator.usd_path`` to its local-bundle URI.

    - bare name -> default version dir:
      ``ex001_6r_meshcollision`` -> ``{base}/v00.04.04/ex001_6r_meshcollision.usd``
    - ``version/name`` -> that version dir:
      ``v00.05.00/ex001_6r_meshcollision`` -> ``{base}/v00.05.00/ex001_6r_meshcollision.usd``
    - full URL (``://``) or local path (``/``, ``~``, ``.`` prefix) passes through unchanged.

    ``.usd`` is appended when missing.
    """
    name = usd_path.strip()
    if is_asset_reference(name) or "://" in name or name[:1] in ("/", "~", "."):
        return name
    if not name.endswith(".usd"):
        name += ".usd"
    if "/" in name:
        # ``version/file`` form -> robot base (the leading segment is the version dir).
        return f"{_EX001_6R_ROBOT_BASE}/{name}"
    return f"{_EX001_6R_USD_DIR}/{name}"


# H-jaw: legacy USD, narrower arm spacing, matches CVPR data collection; open=5.717175 rad.
_EX001_6R_H_GRIPPER_USD = f"{_EX001_6R_ROBOT_BASE}/v00.04.04/ex001_6r_legacy.usd"
# G-jaw: v00.05.05 asset; same kinematics as H except gripper open amplitude=1.89 rad.
_EX001_6R_G_GRIPPER_USD = f"{_EX001_6R_ROBOT_BASE}/v00.05.05/ex001_6r.usd"

# EE -> TCP offset (m) in the gripper frame. Measured as TCP - EE at a
# reference pose; G-jaw right arm example: EE (0.1185,-0.29,0.211),
# TCP (0.3355,-0.29,0.201) -> [0.217, 0, -0.01].
_EX001_6R_H_TCP_OFFSET: tuple[float, float, float] = (0.165, 0.0, 0.0)
_EX001_6R_G_TCP_OFFSET: tuple[float, float, float] = (0.217, 0.0, -0.01)

_EX001_6R_GRIPPER_PROFILES: dict[str, dict[str, str | float | bool | tuple[float, float, float]]] = {
    "H": {
        "usd": _EX001_6R_H_GRIPPER_USD,
        "open": 5.717175,
        "closed": 0.0,
        "tcp_offset": _EX001_6R_H_TCP_OFFSET,
        "mimic_offset_deg": EX001_6R_H_GRIPPER_MIMIC_OFFSET_DEG,
        "mimic_scale": EX001_6R_H_GRIPPER_MIMIC_SCALE,
        "mimic_finger_linear": EX001_6R_H_GRIPPER_MIMIC_FINGER_LINEAR,
    },
    "G": {
        "usd": _EX001_6R_G_GRIPPER_USD,
        "open": 1.89,
        "closed": 0.0,
        "tcp_offset": _EX001_6R_G_TCP_OFFSET,
        "mimic_offset_deg": EX001_6R_G_GRIPPER_MIMIC_OFFSET_DEG,
        "mimic_scale": EX001_6R_G_GRIPPER_MIMIC_SCALE,
        "mimic_finger_linear": EX001_6R_G_GRIPPER_MIMIC_FINGER_LINEAR,
    },
}


def _ex001_6r_gripper_profile(
    variant: str | None = None,
) -> tuple[str, dict[str, str | float | bool | tuple[float, float, float]]]:
    """Resolve an EX001-6R H/G profile, rejecting silent asset mismatches."""

    selected = (variant or _EX001_6R_GRIPPER_VARIANT).strip().upper()
    if selected not in _EX001_6R_GRIPPER_PROFILES:
        raise ValueError(f"Unsupported EX001-6R gripper_variant={variant!r}; expected 'H' or 'G'.")
    return selected, dict(_EX001_6R_GRIPPER_PROFILES[selected])


_EX001_6R_LEFT_ARM_JOINTS = tuple(f"left_arm_joint{i}" for i in range(1, 7))
_EX001_6R_RIGHT_ARM_JOINTS = tuple(f"right_arm_joint{i}" for i in range(1, 7))
# The VR action path uses X2Robot IK, but both EX001 agent bridges still use
# Pink. Keep their task parameters beside the shared URDF configuration.
_EX001_6R_LEFT_PINK_FRAME = "left_arm_gripper_base_link"
_EX001_6R_RIGHT_PINK_FRAME = "right_arm_gripper_base_link"
_EX001_6R_PINK_POSITION_COST = 6.0
_EX001_6R_PINK_ORIENTATION_COST = 3.0
_EX001_6R_PINK_LM_DAMPING = 0.01
_EX001_6R_PINK_GAIN = 1.0
_EX001_6R_IK_URDF_VERSION = "v00.05.03"
_EX001_6R_IK_URDF_NAME = "ex001_6r_legacy.urdf"
_EX001_6R_IK_ASSET_DIR = f"assets/art_assets/urdf/optimized/ex001/{_EX001_6R_IK_URDF_VERSION}"
# X2Robot IK arm gains (CartesianPositionController-aligned). Override via env.
_EX001_6R_X2ROBOT_IK_ARM_KP = float(os.environ.get("MANA_ARM_KP", "5510.1778"))
_EX001_6R_X2ROBOT_IK_ARM_KD = float(os.environ.get("MANA_ARM_KD", "63.4459"))


def _ex001_6r_usd_path() -> str:
    """USD for EX001-6R from :data:`_EX001_6R_GRIPPER_VARIANT` (``H`` / ``G``).

    Optional override: ``MANAENV_EX001_6R_USD_PATH=<url>`` or ``auto`` to fall
    back to local discovery via :func:`resolve_usd_path`. The H/G switch itself
    is code-only (see ``_EX001_6R_GRIPPER_VARIANT``).
    """
    override = os.environ.get("MANAENV_EX001_6R_USD_PATH", "").strip()
    if override == "auto":
        return resolve_usd_path("ex001_6r")
    if override:
        return override
    return str(_EX001_6R_GRIPPER_PROFILES[_EX001_6R_GRIPPER_VARIANT]["usd"])


def _resolve_ik_urdf_paths() -> tuple[str, str]:
    """Return ``(urdf_path, mesh_path)`` for EX001-6R X2Robot / Pinocchio IK."""

    root = resolve_asset_directory(_EX001_6R_IK_ASSET_DIR, must_exist=True)
    urdf = resolve_asset_path(f"{_EX001_6R_IK_ASSET_DIR}/{_EX001_6R_IK_URDF_NAME}", must_exist=True)
    return str(urdf), str(root)


# Compatibility for the existing EX001 IK adapter.
_resolve_pink_urdf_paths = _resolve_ik_urdf_paths


_EX001_6R_GRIPPER_CLOSE = 0.0
_EX001_6R_GRIPPER_OPEN = float(_EX001_6R_GRIPPER_PROFILES[_EX001_6R_GRIPPER_VARIANT]["open"])


def _ex001_6r_spawn_semantic_class() -> str:
    """Label for Isaac ``UsdFileCfg.semantic_tags`` (``("class", ...)``).

    Override with ``MANAENV_EX001_6R_SEMANTIC_CLASS`` (default ``ex001_6r``).
    Used by camera semantic segmentation so the articulated body is one class.
    """

    raw = os.environ.get("MANAENV_EX001_6R_SEMANTIC_CLASS", "").strip()
    return raw or "ex001_6r"


# ═══════════════════════════════════════════════════════════════════════════════
#  Embodiment
# ═══════════════════════════════════════════════════════════════════════════════


@register_asset
class EX0016REmbodiment(EmbodimentBase):
    """EX001-6R bimanual robot (fixed-base, dual 6-DOF arms + grippers)."""

    name = "EX001-6R"
    default_arm_mode = ArmMode.DUAL_ARM
    # Canonical action interface the scripted (skill-trajectory) planner drives for
    # this robot. The skill planner derives ``action_type`` from this when the YAML
    # leaves it unset. Distinct from the policy-level ``action_mode`` (which varies
    # across this robot's per-implementation embodiment YAMLs); this is the robot's
    # own default scripted-planning interface.
    default_planner_action_type: str = "joint"

    # ── Hardware: articulation + proprio/contact sensors ──────────────────

    @configclass
    class SceneCfg:
        # ---- articulation ---------------------------------------------------
        robot: ArticulationCfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=resolve_runtime_asset_reference(_ex001_6r_usd_path()),
                activate_contact_sensors=True,
                semantic_tags=[("class", _ex001_6r_spawn_semantic_class())],
                rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=PhysxArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    fix_root_link=True,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=4,
                ),
            ),
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.32165, -0.25918, -0.25),
                rot=(0.04894564807667012, 0.0, 0.0, 0.9988014434983336),
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
                    joint_names_expr=["left_arm_gripper"],
                    effort_limit_sim=200.0,
                    stiffness=500.0,
                    damping=50.0,
                ),
                "right_gripper_acts": ImplicitActuatorCfg(
                    joint_names_expr=["right_arm_gripper"],
                    effort_limit_sim=200.0,
                    stiffness=500.0,
                    damping=50.0,
                ),
                # The two finger joints on each gripper are USD mimic
                # followers. They still need an implicit actuator entry for
                # IsaacLab's coverage check, but must not use StablePD: the
                # near-zero-inertia follower joints remain position-driven by
                # the mimic relation.
                "gripper_mimic_acts": ImplicitActuatorCfg(
                    joint_names_expr=["left_arm_gripper_.*", "right_arm_gripper_.*"],
                    stiffness=40.0,
                    damping=40.0,
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

    # ── Cameras: separate from scene for EmbodimentBase wiring ────────────

    @configclass
    class CameraCfg(ArenaCameraCfg):
        _use_tiled_camera: ClassVar[bool] = False

        left_wrist_camera: CameraCfg = EX001_6R_LEFTGRIPPER_FISHEYE_CAMERA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_camera_color_frame/Left_Gripper_Camera",
            update_period=0.0333,
            class_type=ManaAsyncCamera,
            # offset=CameraCfg.OffsetCfg(
            #     pos=(0.0, 0.01, 0.0),
            #     rot=(0.0, 1.0, 0.0, 0.0),
            #     convention="opengl",
            # ),
        )
        right_wrist_camera: CameraCfg = EX001_6R_RIGHTGRIPPER_FISHEYE_CAMERA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_camera_color_frame/Right_Gripper_Camera",
            update_period=0.0333,
            class_type=ManaAsyncCamera,
            # offset=CameraCfg.OffsetCfg(
            #     pos=(0.0, 0.01, 0.0),
            #     rot=(0.0, 1.0, 0.0, 0.0),
            #     convention="opengl",
            # ),
        )
        head_camera: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/Head_Camera",
            update_period=0.0333,
            class_type=ManaAsyncCamera,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                horizontal_aperture=5.54,
                vertical_aperture=3.09,
                focal_length=2.97,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.084, 0.0, 0.517),
                rot=(0.21263, -0.21263, -0.67438, 0.67438),
                convention="opengl",
            ),
        )

    # ── Actions: IK mode ──────────────────────────────────────────────────

    @configclass
    class ActionsCfgAbsIK:
        """Absolute IK arms + continuous raw gripper joint targets.

        ``use_relative_mode=False`` so the IK controller receives absolute
        (pos + quat) targets in robot root frame. Per-arm action_dim is 7
        (3 pos + 4 quat). Gripper action terms consume raw joint targets
        clamped to the EX001-6R physical range.
        """

        arm_action: ActionTermCfg = CommandRateDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint[1-6]"],
            body_name="left_arm_gripper_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            scale=1.0,
        )
        gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["left_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_EX001_6R_GRIPPER_CLOSE,
            clamp_max=_EX001_6R_GRIPPER_OPEN,
        )
        right_arm_action: ActionTermCfg = CommandRateDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint[1-6]"],
            body_name="right_arm_gripper_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            scale=1.0,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["right_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_EX001_6R_GRIPPER_CLOSE,
            clamp_max=_EX001_6R_GRIPPER_OPEN,
        )

    # ── Actions: absolute X2Robot IK mode (VR gen_method=2; replaces Pink) ─

    @configclass
    class X2RobotIKActionsCfg:
        """X2Robot IK arms + continuous raw gripper joint targets.

        Layout matches the VR planner's former Pink packing
        (``gen_method=2``):
        ``[left_pos(3), left_quat(4), right_pos(3), right_quat(4),
        left_grip(1), right_grip(1)]``.

        Commands are absolute ``pos3 + quat_xyzw`` poses in the Isaac
        articulation-root frame. The controller applies measured Cartesian
        error in the URDF ``base_platform_link`` frame, so the two bases do not
        need a static calibration.
        """

        arm_action: ActionTermCfg = X2RobotIKActionCfg(
            asset_name="robot",  # ty: ignore[unknown-argument] -- inherited @configclass field
            joint_names=["left_arm_joint[1-6]"],
            body_name="left_arm_gripper_base_link",
            base_frame="base_platform_link",
            isaac_base_body="root",
            ee_frame="left_arm_gripper_base_link",
            ik_joint_names=list(_EX001_6R_LEFT_ARM_JOINTS),
            command_type="pose",
            use_relative_mode=False,
            pose_command_format="quat_xyzw",
            scale=1.0,
        )
        right_arm_action: ActionTermCfg = X2RobotIKActionCfg(
            asset_name="robot",  # ty: ignore[unknown-argument] -- inherited @configclass field
            joint_names=["right_arm_joint[1-6]"],
            body_name="right_arm_gripper_base_link",
            base_frame="base_platform_link",
            isaac_base_body="root",
            ee_frame="right_arm_gripper_base_link",
            ik_joint_names=list(_EX001_6R_RIGHT_ARM_JOINTS),
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
            clamp_min=_EX001_6R_GRIPPER_CLOSE,
            clamp_max=_EX001_6R_GRIPPER_OPEN,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["right_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_EX001_6R_GRIPPER_CLOSE,
            clamp_max=_EX001_6R_GRIPPER_OPEN,
        )

    # ── Actions: joint-position policy mode ───────────────────────────────

    @configclass
    class JointActionsCfg:
        """14-D absolute joint targets for both arms and grippers."""

        left_arm_action: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint[1-6]"],
            scale=1.0,
            use_default_offset=False,
        )
        left_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["left_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_EX001_6R_GRIPPER_CLOSE,
            clamp_max=_EX001_6R_GRIPPER_OPEN,
        )
        right_arm_action: ActionTermCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint[1-6]"],
            scale=1.0,
            use_default_offset=False,
        )
        right_gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
            asset_name="robot",
            joint_names=["right_arm_gripper"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clamp_min=_EX001_6R_GRIPPER_CLOSE,
            clamp_max=_EX001_6R_GRIPPER_OPEN,
        )

    # ── Observations ──────────────────────────────────────────────────────

    @configclass
    class StateObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            follow_left_joint_pos = ObsTerm(
                func=joint_pos_by_names,
                params={"joint_names": [r"left_arm_joint[1-6]", "left_arm_gripper"]},
            )
            follow_right_joint_pos = ObsTerm(
                func=joint_pos_by_names,
                params={"joint_names": [r"right_arm_joint[1-6]", "right_arm_gripper"]},
            )

            # Wallx wire payload. 7-D ``[pos + euler_xyz + grip]`` is relative
            # to the home EE pose (``default_joint_pos`` configuration) and
            # expressed in the robot root (base_link) frame. The x2robot
            # exporter derives its ``follow_*_position`` / ``follow_*_rotation``
            # RPY fields from these terms, so there is a single EE-pose
            # observation source. Class defaults use G's contact-aware mimic
            # inverse; __init__ stamps the active H/G open clamp and mimic law.
            follow1_pos = ObsTerm(
                func=wallx_follow_from_init_ee_root_local_mimic_gripper,
                params={
                    "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
                    "mimic_right_finger_joint_name": "left_arm_gripper_right_joint",
                    "drive_open_rad": _EX001_6R_GRIPPER_OPEN,
                    "mimic_offset_deg": EX001_6R_G_GRIPPER_MIMIC_OFFSET_DEG,
                    "mimic_scale": EX001_6R_G_GRIPPER_MIMIC_SCALE,
                    "finger_linear": EX001_6R_G_GRIPPER_MIMIC_FINGER_LINEAR,
                    "ref_key": "_left_ee_ref",
                },
            )
            follow2_pos = ObsTerm(
                func=wallx_follow_from_init_ee_root_local_mimic_gripper,
                params={
                    "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
                    "mimic_right_finger_joint_name": "right_arm_gripper_right_joint",
                    "drive_open_rad": _EX001_6R_GRIPPER_OPEN,
                    "mimic_offset_deg": EX001_6R_G_GRIPPER_MIMIC_OFFSET_DEG,
                    "mimic_scale": EX001_6R_G_GRIPPER_MIMIC_SCALE,
                    "finger_linear": EX001_6R_G_GRIPPER_MIMIC_FINGER_LINEAR,
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
                    "clone": False,
                },
            )
            right_wrist_cam = ObsTerm(
                func=mdp_isaac_lab.image,
                params={
                    "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                    "data_type": "rgb",
                    "normalize": False,
                    "clone": False,
                },
            )
            head_cam = ObsTerm(
                func=mdp_isaac_lab.image,
                params={
                    "sensor_cfg": SceneEntityCfg("head_camera"),
                    "data_type": "rgb",
                    "normalize": False,
                    "clone": False,
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        camera_obs: CameraObsCfg = CameraObsCfg()

    # ── __init__ ──────────────────────────────────────────────────────────

    def _apply_optional_camera_depth_semantic(self) -> None:
        """Enable Isaac Z-depth + semantic outputs on wrist/head cameras and ``camera_obs``.

        Depth uses ``distance_to_image_plane`` (distance along camera +Z to the image plane),
        not Euclidean ``distance_to_camera`` (ray length).
        """
        from isaaclab_arena.utils.configclass import make_configclass

        for sensor_name in ("left_wrist_camera", "right_wrist_camera", "head_camera"):
            cam = getattr(self.camera_config, sensor_name)
            dtypes = list(getattr(cam, "data_types", []) or [])
            for dt in ("distance_to_image_plane", "semantic_segmentation"):
                if dt not in dtypes:
                    dtypes.append(dt)
            cam.data_types = dtypes

        base_cam_obs = self.CameraObservationsCfg().camera_obs
        sensor_map = (
            ("left_wrist_cam", "left_wrist_camera"),
            ("right_wrist_cam", "right_wrist_camera"),
            ("head_cam", "head_camera"),
        )
        modal_pairs = (
            ("distance_to_image_plane", "distance_to_image_plane"),
            ("semantic_segmentation", "semantic_segmentation"),
        )

        field_list: list[tuple[str, type, ObsTerm]] = []
        for f in fields(type(base_cam_obs)):
            field_list.append((f.name, f.type, getattr(base_cam_obs, f.name)))

        for cam_key, sensor_name in sensor_map:
            for suffix, data_type in modal_pairs:
                term_name = f"{cam_key}_{suffix}"
                field_list.append(
                    (
                        term_name,
                        ObsTerm,
                        ObsTerm(
                            func=mdp_isaac_lab.image,
                            params={
                                "sensor_cfg": SceneEntityCfg(sensor_name),
                                "data_type": data_type,
                                "normalize": False,
                            },
                        ),
                    )
                )

        def _cam_post_init(self):
            self.enable_corruption = False
            self.concatenate_terms = False

        camera_obs_cls = make_configclass(
            "CameraObsCfg",
            field_list,
            bases=(ObsGroup,),
            namespace={"__post_init__": _cam_post_init},
        )
        self._camera_observation_config.camera_obs = camera_obs_cls()

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        gripper_mode: str = "raw_joint",
        gripper_variant: str | None = None,
        diff_ik_compute_mode: str = "command",
        actuator: str = "stablepd",
        actuator_params: dict | None = None,
        camera_depth_semantic: bool = False,
        usd_path: str | None = None,
    ):
        diff_ik_compute_mode = normalize_diff_ik_compute_mode(diff_ik_compute_mode)
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.gripper_variant, self._gripper_profile = _ex001_6r_gripper_profile(gripper_variant)
        self._gripper_open = float(self._gripper_profile["open"])
        self.scene_config = self.SceneCfg()
        # A named H/G profile takes precedence over the generic spawn override
        # so geometry and action limits always describe the same gripper.
        if gripper_variant is not None:
            self.scene_config.robot.spawn.usd_path = resolve_runtime_asset_reference(
                str(self._gripper_profile["usd"])
            )
        elif usd_path:
            self.scene_config.robot.spawn.usd_path = resolve_runtime_asset_reference(
                _resolve_ex001_6r_usd(usd_path)
            )
        apply_actuator(
            self.scene_config.robot,
            actuator,
            actuator_params=actuator_params,
        )
        self.camera_config = self.CameraCfg()
        self.diff_ik_compute_mode = diff_ik_compute_mode
        self.action_config = self.ActionsCfgAbsIK()
        configure_diff_ik_action_cfg(self.action_config, self.diff_ik_compute_mode)
        gm = (gripper_mode or "raw_joint").strip().lower()
        if gm == "joint_position":
            self.action_config = self.JointActionsCfg()
        self._apply_gripper_limits(self.action_config)
        self.gripper_mode = gm
        self.observation_config = self.StateObservationsCfg()
        self._apply_gripper_observation_profile(self.observation_config)
        self._camera_observation_config = self.CameraObservationsCfg()
        if enable_cameras and camera_depth_semantic:
            self._apply_optional_camera_depth_semantic()

    def use_x2robot_ik_action_config(self) -> X2RobotIKActionsCfg:
        """Build X2Robot IK action config and switch arm StablePD gains to real-aligned Kp/Kd."""

        urdf_path, _ = _resolve_ik_urdf_paths()
        cfg = self.X2RobotIKActionsCfg()
        cfg.arm_action.urdf_path = urdf_path
        cfg.right_arm_action.urdf_path = urdf_path
        self.action_config = cfg
        self._apply_gripper_limits(cfg)

        for group in ("left_arm_acts", "right_arm_acts"):
            act = self.scene_config.robot.actuators.get(group)
            if act is not None and hasattr(act, "stiffness") and hasattr(act, "damping"):
                act.stiffness = _EX001_6R_X2ROBOT_IK_ARM_KP
                act.damping = _EX001_6R_X2ROBOT_IK_ARM_KD
        return cfg

    def get_gripper_variant(self) -> str:
        """Return the active H/G gripper variant key."""
        return self.gripper_variant

    def get_gripper_variant_profile(self) -> dict:
        """Return a copy of the active gripper profile (usd/open/tcp_offset/...)."""
        return dict(self._gripper_profile)

    def get_gripper_tcp_offset(self) -> tuple[float, float, float]:
        """Return EE->TCP offset (m) for the active gripper variant."""
        raw = self._gripper_profile.get("tcp_offset", _EX001_6R_H_TCP_OFFSET)
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            return (float(raw[0]), float(raw[1]), float(raw[2]))
        return _EX001_6R_H_TCP_OFFSET

    def map_gripper_command(self, normalized: float) -> float:
        """Map a normalized gripper command to the active variant's joint radians.

        Contract: ``0.0`` = fully closed, ``1.0`` = fully open. Maps onto
        ``[closed, open]`` for the active variant (H: open=5.717175, G: open=1.89).
        """
        open_val = float(self._gripper_profile.get("open", self._gripper_open))
        closed_val = float(self._gripper_profile.get("closed", _EX001_6R_GRIPPER_CLOSE))
        frac = min(1.0, max(0.0, float(normalized)))
        return closed_val + frac * (open_val - closed_val)

    @staticmethod
    def _resolve_arm_sides(arm_mode: ArmMode | str | None) -> list[str]:
        if arm_mode is None:
            arm_mode = ArmMode.DUAL_ARM
        value = arm_mode.value if isinstance(arm_mode, ArmMode) else str(arm_mode)
        if value in {ArmMode.LEFT.value, "left_arm"}:
            return ["left"]
        if value in {ArmMode.RIGHT.value, "right_arm", "robot"}:
            return ["right"]
        if value == ArmMode.SINGLE_ARM.value:
            return ["left"]
        if value == ArmMode.DUAL_ARM.value:
            return ["left", "right"]
        raise ValueError(f"Unsupported EX001-6R arm mode: {arm_mode!r}")

    def get_ee_frame_name(self, arm_mode: ArmMode | str | None = None) -> str:
        sides = self._resolve_arm_sides(arm_mode or ArmMode.LEFT)
        return f"{sides[0]}_ee_frame" if len(sides) == 1 else ""

    def get_ee_frame_names(self, arm_mode: ArmMode | str | None = None) -> list[str]:
        return [f"{side}_ee_frame" for side in self._resolve_arm_sides(arm_mode or self.get_arm_mode())]

    def get_command_body_name(self) -> str:
        return "left_arm_gripper_base_link"

    def get_command_body_names(self, arm_mode: ArmMode | str | None = None) -> list[str]:
        return [f"{side}_arm_gripper_base_link" for side in self._resolve_arm_sides(arm_mode or self.get_arm_mode())]

    def get_ee_contact_sensor_names(self, arm_mode: ArmMode | str | None = None) -> list[str]:
        return [f"{side}_gripper_contact" for side in self._resolve_arm_sides(arm_mode or self.get_arm_mode())]

    def get_ee_contact_prim_paths(self, arm_mode: ArmMode | str | None = None) -> list[str]:
        paths: list[str] = []
        for side in self._resolve_arm_sides(arm_mode or self.get_arm_mode()):
            paths.extend(
                [
                    f"{{ENV_REGEX_NS}}/Robot/{side}_arm_gripper_left_link",
                    f"{{ENV_REGEX_NS}}/Robot/{side}_arm_gripper_right_link",
                ]
            )
        return paths

    def get_observation_cfg(self):
        from isaaclab_arena.utils.configclass import combine_configclass_instances

        if self.enable_cameras:
            return combine_configclass_instances(
                "ObservationCfg",
                self.observation_config,
                self._camera_observation_config,
            )
        return self.observation_config

    def get_default_curobo_arm_configs(self) -> dict[str, dict[str, object]]:
        """Return the robot's per-arm CuRobo *identity* (the EE frame each arm targets).

        Only genuine per-arm robot identity lives here. The cuRobo *planner* config —
        ``robot_config_file``, ``collision_spheres_file``, ``world_config_file`` and
        the execution knobs (``max_planning_attempts``, seeds, thresholds) — is a
        planner concern authored in the skill YAML ``left_curobo``/``right_curobo``,
        merged OVER these at planner setup (``_merge_embodiment_curobo_defaults``;
        YAML wins per key). EX001-6R uses split per-arm robot files (the arms are not
        interchangeable), declared in the YAML.

        cuRobo v0.8.0 dropped USD kinematics, so the maintained URDF (the same one
        X2Robot IK consumes) is injected here via ``urdf_path`` / ``asset_root_path``;
        the evaluator materializes them into the schema-2.0 robot yaml.
        """
        urdf_path, asset_root_path = _resolve_ik_urdf_paths()
        shared = {"urdf_path": urdf_path, "asset_root_path": asset_root_path}
        return {
            "left": {**shared, "ee_link_name": "left_arm_gripper_base_link"},
            "right": {**shared, "ee_link_name": "right_arm_gripper_base_link"},
        }

    def get_gripper_states(self) -> dict[str, float]:
        """Map robot-agnostic gripper symbols to this robot's native joint value.

        ``left_arm_gripper`` is a revolute drive joint; values are radians.
        Fully-open amplitude follows :data:`_EX001_6R_GRIPPER_VARIANT`
        (H=5.717175, G=1.89).
        """
        return {"open": self._gripper_open, "closed": _EX001_6R_GRIPPER_CLOSE}

    def _apply_gripper_limits(self, action_config: object) -> None:
        """Set H/G-native limits on whichever EX001 action config is active."""

        for field_name in (
            "gripper_action",
            "right_gripper_action",
            "master_left_gripper_action",
            "master_right_gripper_action",
        ):
            term = getattr(action_config, field_name, None)
            if term is not None:
                term.clamp_min = _EX001_6R_GRIPPER_CLOSE
                term.clamp_max = self._gripper_open

    def _apply_gripper_observation_profile(self, observation_config: object) -> None:
        """Stamp the active H/G mimic law and open range onto gripper feedback."""

        for term in (
            observation_config.policy.follow1_pos,
            observation_config.policy.follow2_pos,
        ):
            term.func = wallx_follow_from_init_ee_root_local_mimic_gripper
            term.params["drive_open_rad"] = self._gripper_open
            term.params["mimic_offset_deg"] = float(self._gripper_profile["mimic_offset_deg"])
            term.params["mimic_scale"] = float(self._gripper_profile["mimic_scale"])
            term.params["finger_linear"] = bool(self._gripper_profile["mimic_finger_linear"])
