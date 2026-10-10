"""
Robot Configuration and Base Module

This module defines robot configurations for the ManaEnv pipeline and base classes
for robot embodiments that integrate with IsaacLab-Arena.

Architecture:
    Asset (isaaclab_arena)
        └── EmbodimentBase (isaaclab_arena)
                └── ManaRobotBase (manaenv)
                        └── ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

if TYPE_CHECKING:
    from isaaclab_arena.utils.pose import Pose


@dataclass
class ManaRobotCFG:
    """Configuration for ManaEnv robot embodiments.

    This dataclass provides a standardized configuration interface for
    all ManaEnv robot embodiments.
    """

    # Robot identification
    name: str = "mana_robot"
    tags: list[str] = field(default_factory=lambda: ["embodiment", "mana_robot"])

    # Robot type: 0=mobile, 1=fixed_arm, 2=humanoid, 3=quadruped
    robot_type: int = 0

    # USD model path (relative or absolute)
    usd_path: str = None

    # Initial pose
    initial_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    initial_orientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # xyzw quaternion

    # Physics settings
    enable_self_collision: bool = False
    fix_root_link: bool = False
    disable_gravity: bool = False

    # Simulation parameters
    solver_position_iteration_count: int = 8
    solver_velocity_iteration_count: int = 0
    max_linear_velocity: float = 1000.0
    max_angular_velocity: float = 1000.0

    # Camera settings
    enable_cameras: bool = False
    camera_configs: list[dict] = field(default_factory=list)

    # Actuator groups (dict of actuator_name -> actuator_config)
    actuators: dict = field(default_factory=dict)

    # Joint initial states (dict of joint_name -> value)
    initial_joint_positions: dict = field(default_factory=dict)
    initial_joint_velocities: dict = field(default_factory=dict)

    # Metadata
    metadata: dict = field(default_factory=dict)


class ManaRobotBase(EmbodimentBase):
    """Base class for all ManaEnv robot embodiments.

    This class extends IsaacLab-Arena's EmbodimentBase to provide:
    - Standardized robot configuration via ManaRobotCFG
    - Common utilities for robot setup and management
    - Integration with ManaEnv's pipeline system

    Subclasses should:
    1. Set the `name` class attribute
    2. Override `__init__` to set up scene_config, action_config, etc.
    3. Optionally override config getter methods for custom behavior
    """

    name: str = "mana_robot_base"
    tags: list[str] = ["embodiment", "mana_robot"]

    robot_cfg: ManaRobotCFG | None = None

    def __init__(
        self,
        cfg: ManaRobotCFG | None = None,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
    ):
        """Initialize the ManaRobot.

        Args:
            cfg: Robot configuration. If None, uses default configuration.
            enable_cameras: Whether to enable camera sensors.
            initial_pose: Initial pose of the robot in the scene.
        """
        from isaaclab_arena.utils.pose import Pose

        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)

        self.robot_cfg = cfg if cfg is not None else ManaRobotCFG()

        if initial_pose is None and self.robot_cfg.initial_position is not None:
            self.initial_pose = Pose(
                position_xyz=tuple(self.robot_cfg.initial_position),
                rotation_xyzw=tuple(self.robot_cfg.initial_orientation),
            )

        self.scene_config = None
        self.camera_config = None
        self.action_config = None
        self.observation_config = None
        self.event_config = None
        self.reward_config = None
        self.curriculum_config = None
        self.command_config = None
        self.termination_cfg = None

    def get_robot_cfg(self) -> ManaRobotCFG:
        """Get the robot configuration."""
        if self.robot_cfg is None:
            raise RuntimeError("Robot configuration is not initialized")
        return self.robot_cfg

    def get_usd_path(self) -> str:
        """Get the USD model path for this robot."""
        return self.robot_cfg.usd_path

    def get_actuator_config(self) -> dict:
        """Get the actuator configuration dictionary."""
        return self.robot_cfg.actuators

    def is_fixed_base(self) -> bool:
        """Check if the robot has a fixed base."""
        return self.robot_cfg.fix_root_link

    def get_robot_type_name(self) -> str:
        """Get human-readable robot type name."""
        type_names = {
            0: "mobile",
            1: "fixed_arm",
            2: "humanoid",
            3: "quadruped",
        }
        return type_names.get(self.robot_cfg.robot_type, "unknown")

    def validate_config(self) -> bool:
        """Validate the robot configuration.

        Returns:
            True if configuration is valid, False otherwise.
        """
        if self.robot_cfg.usd_path is None:
            return False
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, type={self.get_robot_type_name()})>"
