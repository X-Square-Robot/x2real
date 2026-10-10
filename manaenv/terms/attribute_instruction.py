"""Pure helpers for pick_by_attribute instruction text (no sim imports)."""

from __future__ import annotations

import random
from typing import Any

_DEFAULT_CONFUSION_COLORS = ("red", "green", "blue")
_DEFAULT_CONFUSION_SHAPES = ("ball", "cube", "cylinder", "box")


def resolve_confusion_count(count: Any, rng: random.Random) -> int:
    """Resolve an int or ``[lo, hi]`` range into a non-negative draw count."""
    if isinstance(count, (list, tuple)):
        values = [int(v) for v in count]
        if not values:
            return 0
        if len(values) == 1:
            return max(0, values[0])
        lo, hi = min(values[0], values[1]), max(values[0], values[1])
        return rng.randint(lo, hi)
    return max(0, int(count))


def format_confusion_phrase(
    template: str,
    *,
    rng: random.Random,
    colors: list[str] | tuple[str, ...],
    shapes: list[str] | tuple[str, ...],
) -> str:
    """Fill optional ``{color}`` / ``{shape}`` slots in one distractor phrase."""
    mapping: dict[str, str] = {}
    if "{color}" in template:
        if not colors:
            raise ValueError("confusion phrase uses {color} but confusion_colors is empty")
        mapping["color"] = str(rng.choice(list(colors)))
    if "{shape}" in template:
        if not shapes:
            raise ValueError("confusion phrase uses {shape} but confusion_shapes is empty")
        mapping["shape"] = str(rng.choice(list(shapes)))
    if not mapping:
        return str(template)
    return str(template).format(**mapping)


def compose_attribute_instruction_with_confusion(
    base_instruction: str,
    *,
    confusion_phrases: list[str] | tuple[str, ...],
    rng: random.Random,
    confusion_count: Any = 1,
    confusion_colors: list[str] | tuple[str, ...] | None = None,
    confusion_shapes: list[str] | tuple[str, ...] | None = None,
    confusion_placement: str = "random",
) -> str:
    """Insert random distractor phrases around the real attribute-pick instruction.

    Distractors may mention colors/shapes (via ``{color}`` / ``{shape}`` templates)
    but must not change the committed target mask — only the policy-facing text.
    ``confusion_placement`` is ``before`` / ``after`` / ``around`` / ``random``.
    """
    base = str(base_instruction or "").strip()
    bank = [str(p).strip() for p in confusion_phrases if str(p).strip()]
    if not bank or not base:
        return base

    count = min(resolve_confusion_count(confusion_count, rng), len(bank))
    if count <= 0:
        return base

    colors = tuple(confusion_colors) if confusion_colors else _DEFAULT_CONFUSION_COLORS
    shapes = tuple(confusion_shapes) if confusion_shapes else _DEFAULT_CONFUSION_SHAPES
    chosen = [format_confusion_phrase(p, rng=rng, colors=colors, shapes=shapes) for p in rng.sample(bank, count)]

    placement = str(confusion_placement or "random").strip().lower()
    if placement == "random":
        placement = rng.choice(("before", "after", "around"))
    if placement == "before":
        parts = chosen + [base]
    elif placement == "after":
        parts = [base] + chosen
    elif placement == "around":
        split = rng.randint(0, len(chosen))
        parts = chosen[:split] + [base] + chosen[split:]
    else:
        raise ValueError(
            f"unsupported confusion_placement={confusion_placement!r}; " "expected before|after|around|random"
        )
    return " ".join(parts)
