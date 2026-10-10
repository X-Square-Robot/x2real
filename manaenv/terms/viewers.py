"""Viewer builder functions for declarative KitVisualizerCfg construction.

Each function takes keyword params (from YAML) and returns a ``KitVisualizerCfg``.
The ``lookat_object`` param is an asset name (string) that gets resolved
to an ``Asset`` via ``scene[name]`` at runtime; conversion from the
``@role`` string to a concrete asset name is done earlier by the component.

These are *deferred* — stored as (func, params) by ``ViewerComponent`` and
invoked once the scene is available.

IsaacLab 3.0 ``KitVisualizerCfg`` accepts ``eye`` / ``lookat`` only. Do not
forward the retired ``ViewerCfg.origin_type`` field from YAML or Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_visualizers.kit import KitVisualizerCfg


def _kit_visualizer_cfg(*, eye: tuple[float, float, float], lookat: tuple[float, float, float]) -> KitVisualizerCfg:
    from isaaclab_visualizers.kit import KitVisualizerCfg

    return KitVisualizerCfg(eye=eye, lookat=lookat)


def _vec3(value: Any) -> tuple[float, float, float]:
    vector = tuple(float(item) for item in value)
    if len(vector) != 3:
        raise ValueError(f"viewer vector must contain three values, got {vector!r}")
    return vector[0], vector[1], vector[2]


def look_at_object(lookat_object: Any, offset: list[float] | None = None) -> KitVisualizerCfg:
    """Position the viewport camera looking at an object with an offset.

    Works with both an ``Asset`` instance (has ``get_initial_pose()``) and
    a plain string name (falls back to default).

    Args:
        lookat_object: Asset instance or asset name string.
        offset: [x, y, z] camera offset from the object. Defaults to
            ``[-1.5, -1.5, 1.5]``.

    Returns:
        Configured ``KitVisualizerCfg``.
    """
    if offset is None:
        offset = [-1.5, -1.5, 1.5]
    offset_xyz = _vec3(offset)

    initial_pose = None
    if hasattr(lookat_object, "get_initial_pose"):
        initial_pose = lookat_object.get_initial_pose()

    if initial_pose is None:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        return KitVisualizerCfg()
    if hasattr(initial_pose, "get_midpoint"):
        initial_pose = initial_pose.get_midpoint()

    lookat = _vec3(initial_pose.position_xyz)
    eye = tuple(lookat[index] + offset_xyz[index] for index in range(3))
    return _kit_visualizer_cfg(eye=eye, lookat=lookat)


def look_at_position(lookat_position: list[float], offset: list[float] | None = None) -> KitVisualizerCfg:
    """Position the viewport camera looking at a fixed env-local point."""
    if offset is None:
        offset = [-1.5, -1.5, 1.5]
    lookat = _vec3(lookat_position)
    offset_xyz = _vec3(offset)
    eye = tuple(lookat[index] + offset_xyz[index] for index in range(3))
    return _kit_visualizer_cfg(eye=eye, lookat=lookat)


def default_viewer(**kwargs) -> KitVisualizerCfg:
    """Return a default ``KitVisualizerCfg`` — no object dependency."""
    del kwargs
    from isaaclab_visualizers.kit import KitVisualizerCfg

    return KitVisualizerCfg()
