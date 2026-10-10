"""Robot camera-config discovery for the global lazy registry.

Robots that do not define their own ``camera_config`` get one attached by
robot tag. The cfg classes are heavy (IsaacSim ``@configclass``), so they are
resolved lazily through ``global_registry.get_impl`` rather than a module-level
class dict.
"""

from __future__ import annotations

from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = ["ROBOT_CAMERA_DOMAIN", "ROBOT_TAG_TO_SYMBOL", "discover_robot_camera"]

ROBOT_CAMERA_DOMAIN: str = "robot_camera"

# robot_cfg.name -> camera cfg symbol.
ROBOT_TAG_TO_SYMBOL: dict[str, str] = {
    "droid": "DroidCameraCfg",
    "franka": "FrankaCameraCfg",
}

# symbol -> hosting module.
_ROBOT_CAMERA_SYMBOLS: dict[str, str] = {
    "DroidCameraCfg": "manaenv.assets.sensors.droid_cameras",
    "FrankaCameraCfg": "manaenv.assets.sensors.franka_cameras",
}


def discover_robot_camera(symbol: str) -> ComponentSpec | None:
    """Resolve a robot camera-cfg symbol to its class."""
    module_path = _ROBOT_CAMERA_SYMBOLS.get(symbol)
    if module_path is None:
        return None
    return ComponentSpec(module_path=module_path, class_name=symbol, symbol=symbol)


register_provider(ROBOT_CAMERA_DOMAIN, discover_robot_camera)
