"""Helpers for StablePD gravity-compensation policy."""

from __future__ import annotations

import dataclasses


def gravity_compensation_enabled_from_articulation_cfg(articulation_cfg) -> bool:
    """Return the StablePD gravity-compensation default for an articulation cfg.

    StablePD must not compensate gravity for articulations whose rigid bodies
    have gravity disabled in PhysX, otherwise the controller injects an
    unopposed gravity-compensation torque and the robot can float.
    """
    rigid_props = getattr(getattr(articulation_cfg, "spawn", None), "rigid_props", None)
    return not bool(getattr(rigid_props, "disable_gravity", False))


def stable_pd_cfg_supports_gravity_compensation() -> bool:
    """Whether the installed ``StablePDActuatorCfg`` accepts ``use_gravity_compensation``.

    Upstream Isaac Lab's StablePDActuatorCfg does not expose this field (older / forked
    builds did). Constructor call sites must pass the kwarg only when this returns True,
    mirroring the ``hasattr`` guards already used elsewhere for this attribute.
    """

    try:
        from isaaclab.actuators import StablePDActuatorCfg
    except Exception:
        return False
    return any(f.name == "use_gravity_compensation" for f in dataclasses.fields(StablePDActuatorCfg))


def sync_stable_pd_gravity_compensation_with_articulation_cfg(articulation_cfg) -> int:
    """Apply the articulation gravity flag to all StablePD actuator cfgs.

    Args:
        articulation_cfg: IsaacLab ``ArticulationCfg``-like object with
            ``spawn.rigid_props.disable_gravity`` and an ``actuators`` dict.

    Returns:
        Number of actuator cfg objects updated.
    """
    actuators = getattr(articulation_cfg, "actuators", None)
    if not isinstance(actuators, dict):
        return 0

    enabled = gravity_compensation_enabled_from_articulation_cfg(articulation_cfg)
    updated = 0
    for actuator_cfg in actuators.values():
        if hasattr(actuator_cfg, "use_gravity_compensation"):
            actuator_cfg.use_gravity_compensation = enabled
            updated += 1
    return updated
