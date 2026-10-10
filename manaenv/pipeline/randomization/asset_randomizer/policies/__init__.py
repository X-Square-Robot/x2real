from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import UnsupportedPolicyError
from .base import MaterialPolicy
from .channel_matrix import ChannelMatrixPolicy
from .composite import CompositePolicy
from .material_local_adjustment import MaterialLocalAdjustmentPolicy
from .material_template import MaterialTemplatePolicy


def create_policy(config: Mapping[str, Any]) -> MaterialPolicy:
    policy_type = str(config.get("type", "")).strip()
    if policy_type == "channel_matrix":
        return ChannelMatrixPolicy.from_config(config)
    if policy_type == "material_template":
        return MaterialTemplatePolicy.from_config(config)
    if policy_type == "material_local_adjustment":
        return MaterialLocalAdjustmentPolicy.from_config(config)
    if policy_type == "composite":
        return CompositePolicy.from_config(config)
    raise UnsupportedPolicyError(f"unsupported material policy type: {policy_type!r}")


__all__ = [
    "ChannelMatrixPolicy",
    "CompositePolicy",
    "MaterialLocalAdjustmentPolicy",
    "MaterialPolicy",
    "MaterialTemplatePolicy",
    "create_policy",
]
