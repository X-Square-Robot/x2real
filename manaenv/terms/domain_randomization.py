"""Domain randomization builder functions.

Domain randomization is realized through ``EventTermCfg`` terms that
run on ``"reset"`` (or ``"interval"``).  The component collects these
into ``event_terms`` so they enter the arena pipeline via
``get_events_cfg()``.

Each function here returns an ``EventTermCfg`` referencing the
corresponding function in ``manaenv.terms.events``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg


def uniform_offset(
    asset_cfg: Any = None,
    range_x: list | None = None,
    range_y: list | None = None,
    range_z: list | None = None,
    mode: str = "reset",
    **kwargs,
) -> EventTermCfg:
    """Uniform position offset randomization as an event term."""
    from isaaclab.managers import EventTermCfg

    from manaenv.terms.events import randomize_object_pose

    params: dict = dict(kwargs)
    if asset_cfg is not None:
        params["asset_cfg"] = asset_cfg
    position_range: dict = {}
    if range_x is not None:
        position_range["x"] = range_x
    if range_y is not None:
        position_range["y"] = range_y
    if range_z is not None:
        position_range["z"] = range_z
    if position_range:
        params["position_range"] = position_range
    return EventTermCfg(func=randomize_object_pose, mode=mode, params=params)


def uniform_yaw(yaw_range: list | None = None, asset_cfg: Any = None, mode: str = "reset", **kwargs) -> EventTermCfg:
    """Uniform yaw rotation randomization as an event term."""
    from isaaclab.managers import EventTermCfg

    from manaenv.terms.events import randomize_object_orientation

    params: dict = dict(kwargs)
    if asset_cfg is not None:
        params["asset_cfg"] = asset_cfg
    if yaw_range is not None:
        params["yaw_range"] = yaw_range
    return EventTermCfg(func=randomize_object_orientation, mode=mode, params=params)
