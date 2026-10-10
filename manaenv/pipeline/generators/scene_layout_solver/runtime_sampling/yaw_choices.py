from __future__ import annotations

from typing import Any

from ..static_solve.domains.providers import object_scene_yaw_choices
from .meta import (
    META_SEMANTIC_POOL,
    META_SEMANTIC_POOL_KIND,
    META_SOURCE_SCENE_FIELD,
    META_WEIGHT_SCOPE,
    META_WEIGHT_TOTAL,
    META_ZERO_WEIGHT_EXCLUDED,
    POOL_YAW_CHOICE,
)


def object_yaw_choice_payloads(problem: Any, object_name: str) -> tuple[dict[str, Any], ...]:
    choices = object_scene_yaw_choices(problem, object_name)
    return tuple(
        {
            "id": str(choice.yaw_id),
            "yaw": float(choice.yaw),
            "weight": float(choice.weight),
        }
        for choice in choices
        if float(choice.weight) > 0.0
    )


def object_yaw_choices_by_object(problem: Any, object_names: list[str]) -> dict[str, tuple[dict[str, Any], ...]]:
    return {name: choices for name in object_names if (choices := object_yaw_choice_payloads(problem, name))}


def yaw_choice_semantic_meta(object_name: str, yaw_choices: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    if not yaw_choices:
        return {}
    return {
        META_SEMANTIC_POOL: True,
        META_SEMANTIC_POOL_KIND: POOL_YAW_CHOICE,
        META_SOURCE_SCENE_FIELD: "objects[].yaw_samples",
        META_WEIGHT_SCOPE: "yaw",
        META_WEIGHT_TOTAL: sum(float(choice.get("weight", 1.0)) for choice in yaw_choices),
        META_ZERO_WEIGHT_EXCLUDED: True,
        "yaw_choice_pool_id": f"{object_name}:yaw_samples",
        "yaw_choice_count": len(yaw_choices),
    }
