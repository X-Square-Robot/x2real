"""Command builder functions for declarative CommandsComponent.

Each function returns a command-term configuration (typically a
``CommandTermCfg`` subclass) that the component collects into a
dynamic configclass for ``combine_configclass_instances``.

IsaacLab commands are highly varied (UniformPoseCommand, etc.), so
builder functions here act as adapters that accept YAML params and
return the appropriate cfg type.
"""

from __future__ import annotations

from typing import Any


def uniform_pose_command(target_cfg: Any = None, **kwargs) -> Any:
    """Build a UniformPoseCommandCfg for pose-based commands."""
    from isaaclab.envs.mdp.commands import UniformPoseCommandCfg

    return UniformPoseCommandCfg(**kwargs)
