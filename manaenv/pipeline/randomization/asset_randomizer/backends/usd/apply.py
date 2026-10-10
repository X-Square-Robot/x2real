from __future__ import annotations

from typing import Any

from ...materials.channels import MaterialChannel


def infer_usd_value(value: Any) -> Any:
    try:
        from pxr import Gf
    except ImportError:
        return value
    if isinstance(value, tuple) and len(value) == 3:
        return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))
    return value


def channel_value_type(channel: str) -> Any:
    from pxr import Sdf

    if channel == MaterialChannel.BASE_COLOR:
        return Sdf.ValueTypeNames.Color3f
    return Sdf.ValueTypeNames.Float
