"""EX001-6R gripper channel: drive-equivalent scalar from mimic finger joint.

Why mimic inverse (not the drive joint)?
----------------------------------------
EX001-6R grippers are **mimic-coupled** in USD:

- **Drive joint** (action target): ``left_arm_gripper`` / ``right_arm_gripper``,
  range ``[0, open]`` rad (open = 5.717175 for H-jaw, 1.89 for G-jaw). This is
  what ``ClampedRawGripperAction`` and the master trigger command.
- **Finger joints** (passive mimic): ``*_arm_gripper_left_joint`` /
  ``*_arm_gripper_right_joint``. Isaac Sim enforces the USD mimic constraint, so
  their simulated angles follow the drive joint in free motion but **stall on
  object contact** — the drive joint keeps closing toward its target while the
  finger is physically blocked. That stall is the grasp signal.

Reading ``left_arm_gripper`` directly makes the recorded ``follow_*_gripper``
equal to the command: it never reflects a grasp, so master and follow gripper
curves overlap exactly. To recover contact, read the **right-finger** mimic
joint and invert the USD mimic law back to the drive-equivalent scalar that the
x2robot / wallx gripper channel already uses.

H and G use different USD mimic laws:

    H (prismatic fingers, meters): ``finger_m = drive_deg * (-0.00015)``
    G (revolute fingers, degrees): ``finger_deg = drive_deg * (-0.5) + 4.1827``

H evidence that ``0.00015`` is **m/deg**, not m/rad:

- USD drive ``physics:upperLimit = 327.57`` (Isaac revolute limits are degrees
  = 5.717175 rad).
- Right-finger travel is ``[-0.04875, 0]`` m.
- ``327.57 * 0.00015 ≈ 0.049`` m, matching that travel. ``5.717 * 0.00015``
  would be only 0.86 mm.

Isaac Lab ``joint_pos`` reports prismatic joints in meters and revolute
joints in radians. H must invert meters with a degree-valued drive; it must
not ``rad2deg`` the finger (that cancels into an m/rad divide and saturates).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from manaenv.assets.robots.components.embodiment_obs import get_ee_root_local, get_or_init_ee_ref, gripper_joint_pos
from manaenv.utils.math_utils import euler_xyz_from_quat_xyzw_tensor, quat_conjugate_xyzw_tensor, quat_mul_xyzw_tensor

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg

# H: PhysX ``physxMimicJoint:rotY:gearing = 0.00015`` on the right prismatic
# finger; sign is negative because that finger travels [-0.04875, 0] m.
EX001_6R_H_GRIPPER_MIMIC_OFFSET_DEG = 0.0
EX001_6R_H_GRIPPER_MIMIC_SCALE = -0.00015
EX001_6R_H_GRIPPER_MIMIC_FINGER_LINEAR = True
# G: revolute right-finger law from feat/ex001-composite-teleop-manus.
EX001_6R_G_GRIPPER_MIMIC_OFFSET_DEG = 4.1827
EX001_6R_G_GRIPPER_MIMIC_SCALE = -0.5
EX001_6R_G_GRIPPER_MIMIC_FINGER_LINEAR = False
EX001_6R_DRIVE_GRIPPER_MIN_RAD = 0.0

# Right-finger mimic joints (negative multiplier) matching the laws above.
EX001_6R_GRIPPER_MIMIC_RIGHT_FINGER_JOINTS = (
    "left_arm_gripper_right_joint",
    "right_arm_gripper_right_joint",
)


def drive_gripper_from_mimic_right_finger_position(
    finger_pos: torch.Tensor,
    drive_open_rad: float,
    mimic_offset_deg: float,
    mimic_scale: float,
    finger_linear: bool = False,
) -> torch.Tensor:
    """Invert a passive finger position into a drive-equivalent scalar.

    ``finger_linear=True`` (H): ``finger_pos`` is meters, ``mimic_scale`` is
    meters per drive-degree. ``finger_linear=False`` (G): ``finger_pos`` is
    radians and is converted to degrees before applying a deg/deg scale.
    """

    if finger_linear:
        drive_deg = (finger_pos - mimic_offset_deg) / mimic_scale
    else:
        finger_deg = finger_pos * (180.0 / math.pi)
        drive_deg = (finger_deg - mimic_offset_deg) / mimic_scale
    drive_rad = drive_deg * (math.pi / 180.0)
    return torch.clamp(drive_rad, EX001_6R_DRIVE_GRIPPER_MIN_RAD, float(drive_open_rad))


def drive_gripper_from_mimic_right_finger(
    env,
    mimic_right_finger_joint_name: str,
    drive_open_rad: float,
    mimic_offset_deg: float,
    mimic_scale: float,
    finger_linear: bool = False,
) -> torch.Tensor:
    """Read one passive finger joint and return its drive-equivalent value."""

    finger_pos = gripper_joint_pos(env, mimic_right_finger_joint_name)
    return drive_gripper_from_mimic_right_finger_position(
        finger_pos,
        drive_open_rad,
        mimic_offset_deg,
        mimic_scale,
        finger_linear=finger_linear,
    )


def wallx_follow_from_init_ee_root_local_mimic_gripper(
    env,
    ee_frame_cfg: SceneEntityCfg,
    mimic_right_finger_joint_name: str,
    drive_open_rad: float,
    mimic_offset_deg: float,
    mimic_scale: float,
    ref_key: str,
    finger_linear: bool = False,
) -> torch.Tensor:
    """7-D wallx ``follow*_pos`` with contact-aware (mimic-inverted) gripper.

    Identical to
    :func:`~manaenv.assets.robots.components.embodiment_obs.wallx_follow_from_init_ee_root_local`
    for the pose channels (init-EE-relative pos + Euler XYZ in ``base_link``);
    the gripper slot is the mimic-inverted drive-equivalent scalar so a grasp
    shows up as divergence from the commanded gripper target.
    """
    pos, quat = get_ee_root_local(env, ee_frame_cfg)
    ref_pos, ref_quat = get_or_init_ee_ref(env, ref_key, pos, quat)
    relative_pos = pos - ref_pos
    relative_quat = quat_mul_xyzw_tensor(quat, quat_conjugate_xyzw_tensor(ref_quat))
    roll, pitch, yaw = euler_xyz_from_quat_xyzw_tensor(relative_quat)
    relative_euler = torch.stack([roll, pitch, yaw], dim=-1)
    grip = drive_gripper_from_mimic_right_finger(
        env,
        mimic_right_finger_joint_name,
        drive_open_rad,
        mimic_offset_deg,
        mimic_scale,
        finger_linear=finger_linear,
    )
    return torch.cat([relative_pos, relative_euler, grip], dim=-1)
