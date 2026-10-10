"""Touchable affordance for contact-sensor based interaction."""

from __future__ import annotations

from isaaclab_arena.affordances.affordance_base import AffordanceBase


class Touchable(AffordanceBase):
    """Asset capability described by contact sensor materialization metadata."""

    def __init__(
        self,
        contact_sensor_suffix: str = "contact_sensor",
        contact_against: list[str] | None = None,
        force_threshold: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.contact_sensor_suffix = contact_sensor_suffix
        self.contact_against = list(contact_against or [])
        self.force_threshold = force_threshold

    @property
    def contact_sensor_name(self) -> str:
        return f"{self.name}_{self.contact_sensor_suffix}"
