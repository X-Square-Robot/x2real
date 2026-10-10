"""Relative beside-placement direction: infer spawn relation, sample target dir.

Env tabletop frame (top-down, operator view):
  +x = far from operator (上 / up)
  -x = near operator          (下 / down)
  +y = operator's left        (左 / left)
  -y = operator's right       (右 / right)

On reset, given pickable A and reference block B spawn positions:
  1. Infer which side A starts on relative to B (dominant axis of A - B).
  2. Uniformly sample one of the other three directions as the place goal.
  3. Expose offset, success zone, and instruction phrase for place skill /
     termination / runtime instruction to consume via runtime context.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from manaenv.runtime_context import get_runtime_context
from manaenv.utils.debug_print import manaprint

Direction = str  # "left" | "right" | "up" | "down"

ALL_DIRECTIONS: tuple[Direction, ...] = ("left", "right", "up", "down")

_DIRECTION_OFFSETS: dict[Direction, tuple[float, float, float]] = {
    "left": (0.0, 0.20, 0.02),
    "right": (0.0, -0.20, 0.02),
    "up": (0.20, 0.0, 0.02),
    "down": (-0.20, 0.0, 0.02),
}

_DIRECTION_PHRASES: dict[Direction, str] = {
    "left": "to the left of",
    "right": "to the right of",
    "up": "above",
    "down": "below",
}


def infer_initial_direction(dx: float, dy: float, *, min_axis_delta: float = 0.01) -> Direction:
    """Map A-B offset to the dominant cardinal side (ties prefer x / up-down)."""
    if abs(dx) < min_axis_delta and abs(dy) < min_axis_delta:
        return "right"
    if abs(dx) >= abs(dy):
        return "up" if dx > 0 else "down"
    return "left" if dy > 0 else "right"


def sample_target_direction(initial: Direction, rng: random.Random) -> Direction:
    """Pick uniformly among the three directions other than *initial*."""
    choices = [d for d in ALL_DIRECTIONS if d != initial]
    return rng.choice(choices)


def direction_offset(direction: Direction, *, gap: float = 0.20) -> tuple[float, float, float]:
    base = _DIRECTION_OFFSETS[direction]
    scale = float(gap) / 0.20 if gap != 0.20 else 1.0
    return (base[0] * scale, base[1] * scale, base[2])


def direction_zone_ranges(
    direction: Direction,
    *,
    gap: float = 0.20,
    half_width: float = 0.07,
    cross_half_width: float = 0.08,
) -> dict[str, list[float]]:
    """Relative (object - block) XY bands for ``object_position_in_zone``."""
    g = float(gap)
    hw = float(half_width)
    cw = float(cross_half_width)
    if direction == "left":
        return {"x_range": [-cw, cw], "y_range": [g - hw, g + hw]}
    if direction == "right":
        return {"x_range": [-cw, cw], "y_range": [-g - hw, -g + hw]}
    if direction == "up":
        return {"x_range": [g - hw, g + hw], "y_range": [-cw, cw]}
    return {"x_range": [-g - hw, -g + hw], "y_range": [-cw, cw]}


def direction_phrase(direction: Direction) -> str:
    return _DIRECTION_PHRASES[direction]


def select_place_beside_direction(
    *,
    object_xy: tuple[float, float],
    block_xy: tuple[float, float],
    gap: float = 0.20,
    zone_half_width: float = 0.07,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Full per-episode beside-placement bundle stored under runtime context."""
    rng = rng or random.Random()
    dx = float(object_xy[0]) - float(block_xy[0])
    dy = float(object_xy[1]) - float(block_xy[1])
    initial = infer_initial_direction(dx, dy)
    target = sample_target_direction(initial, rng)
    zone = direction_zone_ranges(target, gap=gap, half_width=zone_half_width)
    return {
        "initial": initial,
        "target": target,
        "gap": float(gap),
        "offset": list(direction_offset(target, gap=gap)),
        "x_range": zone["x_range"],
        "y_range": zone["y_range"],
        "phrase": direction_phrase(target),
        "spawn_dx": dx,
        "spawn_dy": dy,
    }


def definition_beside_placement(task_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    definition = task_payload.get("definition") if isinstance(task_payload.get("definition"), Mapping) else {}
    spec = definition.get("beside_placement") if isinstance(definition, Mapping) else None
    return dict(spec) if isinstance(spec, Mapping) else None


def attach_place_beside_to_context(
    context: dict[str, Any],
    *,
    placements: Mapping[str, Mapping[str, Any]],
    spec: Mapping[str, Any],
    rng: random.Random,
) -> None:
    pickable = str(spec.get("pickable") or "object").strip()
    reference = str(spec.get("reference") or "block").strip()
    pick_pose = placements.get(pickable)
    ref_pose = placements.get(reference)
    if not isinstance(pick_pose, Mapping) or not isinstance(ref_pose, Mapping):
        return
    pick_pos = pick_pose.get("position") or []
    ref_pos = ref_pose.get("position") or []
    if len(pick_pos) < 2 or len(ref_pos) < 2:
        return
    bundle = select_place_beside_direction(
        object_xy=(float(pick_pos[0]), float(pick_pos[1])),
        block_xy=(float(ref_pos[0]), float(ref_pos[1])),
        gap=float(spec.get("gap", 0.20)),
        zone_half_width=float(spec.get("zone_half_width", 0.07)),
        rng=rng,
    )
    context.setdefault("outputs", {})["place_beside"] = bundle

    manaprint(
        "INFO: [place_beside] "
        f"spawn=({bundle.get('spawn_dx'):+.3f},{bundle.get('spawn_dy'):+.3f}) "
        f"initial={bundle.get('initial')} target={bundle.get('target')} "
        f"gap={bundle.get('gap')} offset={bundle.get('offset')} phrase='{bundle.get('phrase')}'"
    )


def get_place_beside(env: Any, *, env_id: int, plan_id: str = "runtime") -> dict[str, Any] | None:

    context = get_runtime_context(env, plan_id=plan_id, env_id=int(env_id))
    if not isinstance(context, Mapping):
        return None
    outputs = context.get("outputs")
    if not isinstance(outputs, Mapping):
        return None
    bundle = outputs.get("place_beside")
    return dict(bundle) if isinstance(bundle, Mapping) else None


def runtime_place_offset(
    env: Any,
    env_idx: int,
    *,
    plan_id: str = "runtime",
    fallback: tuple[float, float, float] | None = None,
) -> tuple[float, float, float] | None:
    bundle = get_place_beside(env, env_id=int(env_idx), plan_id=plan_id)
    if bundle and isinstance(bundle.get("offset"), (list, tuple)) and len(bundle["offset"]) >= 3:
        off = bundle["offset"]
        return (float(off[0]), float(off[1]), float(off[2]))
    return fallback


__all__ = [
    "ALL_DIRECTIONS",
    "attach_place_beside_to_context",
    "definition_beside_placement",
    "direction_offset",
    "direction_phrase",
    "direction_zone_ranges",
    "get_place_beside",
    "infer_initial_direction",
    "runtime_place_offset",
    "sample_target_direction",
    "select_place_beside_direction",
]
