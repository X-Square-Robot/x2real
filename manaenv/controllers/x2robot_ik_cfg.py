"""Configuration for X2RobotIKController."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class X2RobotIKControllerCfg:
    """Configuration for the X2Robot IK controller.

    Mirrors the C++ CartesianPositionController parameters from
    x2robot_controllers-v26.06.02.00. Runtime is GPU-batched only.
    """

    urdf_path: str = ""
    base_frame: str = "base_link"
    ee_frame: str = "link6"
    joint_names: list[str] = field(
        default_factory=lambda: [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
    )

    command_type: str = "pose"
    use_relative_mode: bool = False

    jacobian_damping: float = 0.0005
    convergence_vel_eps: float = 0.005
    dt: float = 0.005

    armijo_alpha_init: float = 0.2
    armijo_c: float = 0.1
    armijo_beta: float = 0.5
    armijo_alpha_min: float = 1e-4

    # Stop updating when cartesian error is below this threshold.
    # Prevents float32 noise from destabilizing a converged solution.
    gpu_convergence_error_eps: float = 5e-3

    position_limit_low: list[float] | None = None
    position_limit_high: list[float] | None = None
