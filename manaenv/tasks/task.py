"""Domain randomization and constraint configuration for tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class PositionRandomizationCfg:
    """Position randomization around an initial pose."""

    AVAILABLE_METHODS: ClassVar[tuple[str, ...]] = ("uniform_offset",)

    method: str = "uniform_offset"
    range_x: list = field(default_factory=lambda: [0.0, 0.0])
    range_y: list = field(default_factory=lambda: [0.0, 0.0])
    range_z: list = field(default_factory=lambda: [0.0, 0.0])

    def __post_init__(self) -> None:
        if self.method not in self.AVAILABLE_METHODS:
            raise ValueError(
                f"Unsupported object_position randomization method '{self.method}'. "
                f"Available methods: {list(self.AVAILABLE_METHODS)}"
            )

    def to_range_dict(self) -> dict:
        """Convert to the dict format expected by event functions."""
        return {
            "x": list(self.range_x),
            "y": list(self.range_y),
            "z": list(self.range_z),
        }


@dataclass
class OrientationRandomizationCfg:
    """Orientation (yaw) randomization."""

    AVAILABLE_METHODS: ClassVar[tuple[str, ...]] = ("uniform_yaw",)

    method: str = "uniform_yaw"
    yaw_range: list = field(default_factory=lambda: [-3.14159, 3.14159])

    def __post_init__(self) -> None:
        if self.method not in self.AVAILABLE_METHODS:
            raise ValueError(
                f"Unsupported object_orientation randomization method '{self.method}'. "
                f"Available methods: {list(self.AVAILABLE_METHODS)}"
            )


@dataclass
class DomainRandomizationCfg:
    """Domain randomization configuration for a task.

    Controls how objects, lighting, and textures are randomized
    during episode resets. This config is passed to ManaTask classes
    which override get_events_cfg() to inject the corresponding
    IsaacLab EventTermCfg entries.

    Future extensions: texture, lighting, camera, physics params.
    """

    object_position: PositionRandomizationCfg | None = None
    object_orientation: OrientationRandomizationCfg | None = None

    @property
    def any_enabled(self) -> bool:
        """Check if any randomization is enabled."""
        return self.object_position is not None or self.object_orientation is not None
