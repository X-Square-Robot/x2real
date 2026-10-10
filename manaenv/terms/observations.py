"""Observation builder functions for declarative ObservationComponent.

Each function returns an ``ObservationTermCfg`` (``ObsTerm``) referencing
the underlying ``mdp`` function.  The component collects these into an
``ObservationGroupCfg`` and wraps the group(s) into a dynamic configclass
for ``combine_configclass_instances``.

Functions whose name starts with ``mdp.`` are resolved from
``isaaclab.envs.mdp`` automatically by the component's ``resolve_func``.
Functions defined here act as thin adapters that construct ``ObsTerm``
objects with the correct ``func`` and ``params``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.managers import ObservationTermCfg as ObsTerm


def joint_pos(asset_cfg: Any = None, **kwargs) -> ObsTerm:
    """Joint position observation (relative)."""
    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm

    params = dict(kwargs)
    if asset_cfg is not None:
        params["asset_cfg"] = asset_cfg
    return ObsTerm(func=mdp.joint_pos_rel, params=params if params else {})


def joint_vel(asset_cfg: Any = None, **kwargs) -> ObsTerm:
    """Joint velocity observation (relative)."""
    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm

    params = dict(kwargs)
    if asset_cfg is not None:
        params["asset_cfg"] = asset_cfg
    return ObsTerm(func=mdp.joint_vel_rel, params=params if params else {})


def last_action(**kwargs) -> ObsTerm:
    """Last action taken by the agent."""
    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm

    return ObsTerm(func=mdp.last_action, params=kwargs if kwargs else {})
