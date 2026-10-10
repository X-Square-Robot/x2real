from enum import Enum


class ArmMode(str, Enum):  # noqa: UP042 - StrEnum would change the legacy string representation.
    """Arm mode used by ManaEnv robot embodiments."""

    SINGLE_ARM = "single_arm"
    DUAL_ARM = "dual_arm"
    LEFT = "left"
    RIGHT = "right"

    def get_other_arm(self) -> "ArmMode":
        if self not in (ArmMode.LEFT, ArmMode.RIGHT):
            raise AssertionError(f"Arm mode {self} is not a bimanual arm mode")
        return ArmMode.RIGHT if self == ArmMode.LEFT else ArmMode.LEFT
