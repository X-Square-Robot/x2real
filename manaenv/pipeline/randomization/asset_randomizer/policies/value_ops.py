from __future__ import annotations

import colorsys
from collections.abc import Mapping, Sequence
from typing import Any

from ..materials.channels import COLOR_CHANNELS, SCALAR_CHANNELS, MaterialChannel, RandomizationOp
from ..model import MaterialSlot
from ..rng import RandomStream

Color = tuple[float, float, float]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_color(value: Sequence[Any]) -> Color:
    if len(value) != 3:
        raise ValueError("color must have exactly three channels")
    floats = tuple(float(channel) for channel in value)
    if any(channel > 1.0 for channel in floats):
        floats = tuple(channel / 255.0 for channel in floats)
    return clamp01(floats[0]), clamp01(floats[1]), clamp01(floats[2])


def normalize_scalar(value: Any) -> float:
    return clamp01(float(value))


def normalize_channel_value(channel: str, value: Any) -> Color | float:
    if channel in COLOR_CHANNELS:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError(f"{channel} value must be a color triplet")
        return normalize_color(value)
    if channel in SCALAR_CHANNELS:
        return normalize_scalar(value)
    raise ValueError(f"{channel} does not support value randomization")


def sample_constant_value(
    *,
    slot: MaterialSlot,
    channel: str,
    op: str,
    config: Mapping[str, Any],
    rng: RandomStream,
) -> Color | float:
    if op == RandomizationOp.SET_VALUE:
        if "value" not in config and "color" not in config:
            raise ValueError(f"{channel}.{op} requires value")
        return normalize_channel_value(channel, config.get("value", config.get("color")))
    if op == RandomizationOp.CHOICE:
        values = tuple(config.get("values") or config.get("colors") or config.get("palette") or ())
        if not values:
            raise ValueError(f"{channel}.{op} requires non-empty values")
        return normalize_channel_value(channel, rng.choice(values))
    if op == RandomizationOp.SAMPLE_UNIFORM:
        return sample_uniform_value(channel=channel, config=config, rng=rng)
    if op == RandomizationOp.HSV_JITTER and channel == MaterialChannel.BASE_COLOR:
        base = slot.channel_states[channel].value or config.get("default_color") or (0.8, 0.8, 0.8)
        return hsv_jitter(base=base, config=config, rng=rng)
    if op == RandomizationOp.JITTER_RELATIVE:
        state = slot.channel_states.get(channel)
        base = (state.value if state is not None else None) or config.get("default") or _default_value(channel)
        return jitter_relative_value(channel=channel, base=base, config=config, rng=rng)
    if op == RandomizationOp.CLAMP:
        state = slot.channel_states.get(channel)
        base = (state.value if state is not None else None) or config.get("default") or _default_value(channel)
        return clamp_channel_value(channel=channel, value=base, config=config)
    raise ValueError(f"unsupported constant operation for sampling: {channel}.{op}")


def sample_uniform_value(*, channel: str, config: Mapping[str, Any], rng: RandomStream) -> Color | float:
    if channel == MaterialChannel.BASE_COLOR:
        low = normalize_color(config.get("min") or config.get("low") or (0.0, 0.0, 0.0))
        high = normalize_color(config.get("max") or config.get("high") or (1.0, 1.0, 1.0))
        return (
            rng.uniform(low[0], high[0]),
            rng.uniform(low[1], high[1]),
            rng.uniform(low[2], high[2]),
        )
    bounds = config.get("range", (0.0, 1.0))
    return normalize_scalar(rng.uniform(float(bounds[0]), float(bounds[1])))


def hsv_jitter(*, base: Any, config: Mapping[str, Any], rng: RandomStream) -> Color:
    r, g, b = normalize_color(base)
    hue, sat, value = colorsys.rgb_to_hsv(r, g, b)
    hue_delta = float(config.get("hue_delta", 0.08))
    saturation_range = config.get("saturation_range", (0.5, 1.0))
    value_range = config.get("value_range", (0.4, 1.0))
    hue = (hue + rng.uniform(-hue_delta, hue_delta)) % 1.0
    sat = rng.uniform(float(saturation_range[0]), float(saturation_range[1]))
    value = rng.uniform(float(value_range[0]), float(value_range[1]))
    red, green, blue = colorsys.hsv_to_rgb(hue, sat, value)
    return clamp01(red), clamp01(green), clamp01(blue)


def texture_scale_value(*, channel: str, config: Mapping[str, Any]) -> Color | float:
    value = config.get("value", config.get("scale", config.get("color", 1.0)))
    return normalize_texture_numeric_value(channel=channel, value=value)


def texture_bias_value(*, channel: str, config: Mapping[str, Any]) -> Color | float:
    value = config.get("value", config.get("bias", config.get("color", 0.0)))
    return normalize_texture_numeric_value(channel=channel, value=value, clamp=False)


def texture_replace_value(config: Mapping[str, Any]) -> dict[str, Any]:
    uri = config.get("uri", config.get("path", config.get("texture", config.get("value"))))
    if not uri:
        raise ValueError("texture.replace requires uri/path/texture/value")
    payload = {"uri": str(uri)}
    for key in ("color_space", "role"):
        if key in config:
            payload[key] = config[key]
    return payload


def normalize_texture_numeric_value(*, channel: str, value: Any, clamp: bool = True) -> Color | float:
    if channel in COLOR_CHANNELS:
        if isinstance(value, (int, float)):
            scalar = float(value)
            return clamp01(scalar) if clamp else scalar
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if clamp:
                return normalize_color(value)
            if len(value) != 3:
                raise ValueError("color must have exactly three channels")
            return float(value[0]), float(value[1]), float(value[2])
        raise ValueError(f"{channel} texture numeric value must be scalar or color triplet")
    if channel in SCALAR_CHANNELS:
        scalar = float(value)
        return clamp01(scalar) if clamp else scalar
    raise ValueError(f"{channel} does not support texture value transforms")


def normal_strength_scale_value(config: Mapping[str, Any]) -> float:
    return normalize_scalar(config.get("value", config.get("scale", 1.0)))


def jitter_relative_value(
    *,
    channel: str,
    base: Any,
    config: Mapping[str, Any],
    rng: RandomStream,
) -> Color | float:
    low, high = _relative_bounds(config)
    if channel in COLOR_CHANNELS:
        color = normalize_color(base)
        return (
            clamp01(color[0] * rng.uniform(low, high)),
            clamp01(color[1] * rng.uniform(low, high)),
            clamp01(color[2] * rng.uniform(low, high)),
        )
    scalar = normalize_scalar(base)
    return normalize_scalar(scalar * rng.uniform(low, high))


def clamp_channel_value(*, channel: str, value: Any, config: Mapping[str, Any]) -> Color | float:
    if channel in COLOR_CHANNELS:
        color = normalize_color(value)
        low = normalize_color(config.get("min") or config.get("low") or (0.0, 0.0, 0.0))
        high = normalize_color(config.get("max") or config.get("high") or (1.0, 1.0, 1.0))
        return (
            max(low[0], min(high[0], color[0])),
            max(low[1], min(high[1], color[1])),
            max(low[2], min(high[2], color[2])),
        )
    bounds = config.get("range", (config.get("min", 0.0), config.get("max", 1.0)))
    scalar = normalize_scalar(value)
    return max(float(bounds[0]), min(float(bounds[1]), scalar))


def _relative_bounds(config: Mapping[str, Any]) -> tuple[float, float]:
    bounds = config.get("scale_range", config.get("relative_range"))
    if bounds is not None:
        return float(bounds[0]), float(bounds[1])
    amount = float(config.get("amount", config.get("delta", 0.1)))
    return 1.0 - amount, 1.0 + amount


def _default_value(channel: str) -> Color | float:
    if channel in COLOR_CHANNELS:
        return (0.5, 0.5, 0.5)
    if channel in SCALAR_CHANNELS:
        return 0.5
    raise ValueError(f"{channel} does not support value randomization")
