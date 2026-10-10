from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .channels import MaterialChannel, RandomizationOp


class PackedOrmChannel:
    OCCLUSION = "occlusion"
    ROUGHNESS = "roughness"
    METALLIC = "metallic"


@dataclass(frozen=True)
class PackedOrmLayout:
    layout_id: str
    channel_indices: Mapping[str, int]

    def index_for(self, channel: str) -> int:
        key = _packed_channel_name(channel)
        if key not in self.channel_indices:
            raise ValueError(f"packed ORM layout {self.layout_id!r} does not provide {key}")
        return int(self.channel_indices[key])


@dataclass(frozen=True)
class PackedOrmBinding:
    uri: str
    layout: PackedOrmLayout = field(default_factory=lambda: DEFAULT_ORM_RGB_LAYOUT)
    color_space: str = "raw"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "uri": self.uri,
            "layout": self.layout.layout_id,
            "packing": "orm",
            "channels": dict(self.layout.channel_indices),
            "color_space": self.color_space,
        }
        for key in ("role", "normal_format"):
            if key in self.metadata:
                payload[key] = self.metadata[key]
        return payload


DEFAULT_ORM_RGB_LAYOUT = PackedOrmLayout(
    layout_id="orm_rgb",
    channel_indices={
        PackedOrmChannel.OCCLUSION: 0,
        PackedOrmChannel.ROUGHNESS: 1,
        PackedOrmChannel.METALLIC: 2,
    },
)

ORM_RGB_LAYOUT_ALIASES = frozenset({"orm", "orm_rgb", "r_ao_g_roughness_b_metallic"})
RGB_BAND_INDICES = {
    "r": 0,
    "red": 0,
    "g": 1,
    "green": 1,
    "b": 2,
    "blue": 2,
    "a": 3,
    "alpha": 3,
}


def normalize_packed_orm_binding(value: Any) -> PackedOrmBinding:
    data = _as_mapping(value)
    uri = _uri(data)
    if not uri:
        raise ValueError("packed ORM binding requires uri/path/texture/value")
    layout = normalize_packed_orm_layout(data)
    return PackedOrmBinding(
        uri=uri,
        layout=layout,
        color_space=str(data.get("color_space") or data.get("colorspace") or "raw"),
        metadata=data,
    )


def normalize_packed_orm_layout(value: Any) -> PackedOrmLayout:
    data = value if isinstance(value, Mapping) else {}
    layout_id = str(data.get("layout") or data.get("packing") or data.get("format") or "orm_rgb").strip().lower()
    if layout_id not in ORM_RGB_LAYOUT_ALIASES:
        raise ValueError(f"unsupported packed ORM layout: {layout_id}")
    channel_indices = data.get("channels") or data.get("packed_channels")
    if channel_indices is None:
        return DEFAULT_ORM_RGB_LAYOUT
    normalized = _normalize_channel_indices(channel_indices)
    if normalized != dict(DEFAULT_ORM_RGB_LAYOUT.channel_indices):
        raise ValueError(f"unsupported packed ORM channel mapping: {normalized}")
    return DEFAULT_ORM_RGB_LAYOUT


def packed_orm_replace_payload(value: Any) -> dict[str, Any]:
    return normalize_packed_orm_binding(value).to_payload()


def packed_orm_set_channel_value_payload(*, channel: str, value: Any, layout: Any = None) -> dict[str, Any]:
    packed_channel = _packed_channel_name(channel)
    if packed_channel == PackedOrmChannel.OCCLUSION:
        raise ValueError("occlusion packed ORM editing is not supported in V1")
    payload = {
        "layout": normalize_packed_orm_layout({"layout": layout or "orm_rgb"}).layout_id,
        "packed_channel": packed_channel,
        "value": _normalize_scalar(value),
    }
    return payload


def packed_orm_transform_channel_payload(
    *,
    channel: str,
    transform: str,
    value: Any,
    layout: Any = None,
) -> dict[str, Any]:
    packed_channel = _packed_channel_name(channel)
    if packed_channel == PackedOrmChannel.OCCLUSION:
        raise ValueError("occlusion packed ORM editing is not supported in V1")
    transform = str(transform or "").strip()
    if transform not in {
        RandomizationOp.SCALE,
        RandomizationOp.BIAS,
        RandomizationOp.CLAMP,
    }:
        raise ValueError(f"unsupported packed ORM channel transform: {transform}")
    payload = {
        "layout": normalize_packed_orm_layout({"layout": layout or "orm_rgb"}).layout_id,
        "packed_channel": packed_channel,
        "transform": transform,
        "value": _normalize_transform_value(transform=transform, value=value),
    }
    return payload


def packed_orm_disable_values_payload(value: Any) -> dict[str, float]:
    data = _as_mapping(value)
    raw_values = data.get("values")
    values = raw_values if isinstance(raw_values, Mapping) else data
    if MaterialChannel.ROUGHNESS not in values or MaterialChannel.METALLIC not in values:
        raise ValueError("orm.packed_texture.disable_and_set_values requires roughness and metallic values")
    return {
        MaterialChannel.ROUGHNESS: _normalize_scalar(values[MaterialChannel.ROUGHNESS]),
        MaterialChannel.METALLIC: _normalize_scalar(values[MaterialChannel.METALLIC]),
    }


def write_packed_orm_channel_value_texture(
    *,
    source_path: Path,
    output_dir: Path,
    channel: str,
    value: Any,
    layout: Any = None,
) -> Path:
    return _write_packed_orm_texture(
        source_path=source_path,
        output_dir=output_dir,
        channel=channel,
        operation=RandomizationOp.SET_CHANNEL_VALUE,
        transform=RandomizationOp.SET_VALUE,
        value=_normalize_scalar(value),
        layout=normalize_packed_orm_layout({"layout": layout or "orm_rgb"}),
    )


def write_packed_orm_channel_transform_texture(
    *,
    source_path: Path,
    output_dir: Path,
    channel: str,
    transform: str,
    value: Any,
    layout: Any = None,
) -> Path:
    transform = str(transform or "").strip()
    if transform not in {
        RandomizationOp.SCALE,
        RandomizationOp.BIAS,
        RandomizationOp.CLAMP,
    }:
        raise ValueError(f"unsupported packed ORM channel transform: {transform}")
    return _write_packed_orm_texture(
        source_path=source_path,
        output_dir=output_dir,
        channel=channel,
        operation=RandomizationOp.TRANSFORM_CHANNEL,
        transform=transform,
        value=_normalize_transform_value(transform=transform, value=value),
        layout=normalize_packed_orm_layout({"layout": layout or "orm_rgb"}),
    )


def _write_packed_orm_texture(
    *,
    source_path: Path,
    output_dir: Path,
    channel: str,
    operation: str,
    transform: str,
    value: Any,
    layout: PackedOrmLayout,
) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL is required for packed ORM texture transforms") from exc

    source_path = Path(source_path)
    output_path = _generated_packed_orm_path(
        source_path=source_path,
        output_dir=Path(output_dir),
        channel=channel,
        operation=operation,
        transform=transform,
        value=value,
        layout=layout,
    )
    if output_path.is_file():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source_image:
        image = _editable_packed_image(source_image)
        channels = list(image.split())
        channel_index = layout.index_for(channel)
        if channel_index >= min(len(channels), 3):
            raise ValueError(f"packed ORM texture has no RGB band for {channel}")
        channels[channel_index] = channels[channel_index].point(_lut(transform=transform, value=value))
        Image.merge(image.mode, channels).save(output_path)
    return output_path


def _generated_packed_orm_path(
    *,
    source_path: Path,
    output_dir: Path,
    channel: str,
    operation: str,
    transform: str,
    value: Any,
    layout: PackedOrmLayout,
) -> Path:
    source_stat = source_path.stat()
    payload = {
        "source": str(source_path.resolve()),
        "size": int(source_stat.st_size),
        "mtime_ns": int(source_stat.st_mtime_ns),
        "layout": layout.layout_id,
        "channel": _packed_channel_name(channel),
        "operation": operation,
        "transform": transform,
        "value": _jsonable(value),
    }
    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return output_dir / f"{source_path.stem}_{_packed_channel_name(channel)}_{operation}_{digest}.png"


def _editable_packed_image(image: Any) -> Any:
    if image.mode in {"RGB", "RGBA"}:
        return image.copy()
    if image.mode in {"L", "LA"}:
        raise ValueError("packed ORM texture must expose at least RGB channels")
    if "A" in image.mode:
        return image.convert("RGBA")
    return image.convert("RGB")


def _lut(*, transform: str, value: Any) -> list[int]:
    if transform == RandomizationOp.SET_VALUE:
        byte = _byte(_normalize_scalar(value) * 255.0)
        return [byte for _ in range(256)]
    if transform == RandomizationOp.SCALE:
        amount = float(value)
        return [_byte(channel_value * amount) for channel_value in range(256)]
    if transform == RandomizationOp.BIAS:
        amount = float(value)
        return [_byte(channel_value + amount * 255.0) for channel_value in range(256)]
    if transform == RandomizationOp.CLAMP:
        low, high = _clamp_bounds(value)
        return [_byte(max(low, min(high, channel_value / 255.0)) * 255.0) for channel_value in range(256)]
    raise ValueError(f"unsupported packed ORM transform: {transform}")


def _normalize_transform_value(*, transform: str, value: Any) -> Any:
    if transform == RandomizationOp.CLAMP:
        low, high = _clamp_bounds(value)
        return {"min": low, "max": high}
    if transform == RandomizationOp.SCALE:
        return max(0.0, float(value))
    if transform == RandomizationOp.BIAS:
        return float(value)
    raise ValueError(f"unsupported packed ORM transform: {transform}")


def _clamp_bounds(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        if "range" in value:
            bounds = value["range"]
        else:
            bounds = (value.get("min", 0.0), value.get("max", 1.0))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        bounds = value
    else:
        raise ValueError("packed ORM clamp requires range or min/max")
    if len(bounds) != 2:
        raise ValueError("packed ORM clamp range must have exactly two values")
    low = _normalize_scalar(bounds[0])
    high = _normalize_scalar(bounds[1])
    if low > high:
        raise ValueError("packed ORM clamp min must be <= max")
    return low, high


def _normalize_channel_indices(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        normalized: dict[str, int] = {}
        for key, item in value.items():
            band = str(key).strip().lower()
            if band in RGB_BAND_INDICES:
                normalized[_packed_channel_name(item)] = RGB_BAND_INDICES[band]
            else:
                normalized[_packed_channel_name(key)] = _channel_index(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {_packed_channel_name(item): index for index, item in enumerate(value)}
    raise ValueError("packed ORM channels must be a mapping or sequence")


def _channel_index(value: Any) -> int:
    band = str(value).strip().lower()
    if band in RGB_BAND_INDICES:
        return RGB_BAND_INDICES[band]
    return int(value)


def _packed_channel_name(channel: Any) -> str:
    value = str(channel).strip().lower()
    aliases = {
        "ao": PackedOrmChannel.OCCLUSION,
        "ambient_occlusion": PackedOrmChannel.OCCLUSION,
        "occlusion": PackedOrmChannel.OCCLUSION,
        MaterialChannel.ROUGHNESS: PackedOrmChannel.ROUGHNESS,
        "rough": PackedOrmChannel.ROUGHNESS,
        MaterialChannel.METALLIC: PackedOrmChannel.METALLIC,
        "metal": PackedOrmChannel.METALLIC,
    }
    if value not in aliases:
        raise ValueError(f"unsupported packed ORM channel: {channel}")
    return aliases[value]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        return {"uri": str(value)}
    raise ValueError(f"packed ORM payload must be a mapping or URI string, got {type(value).__name__}")


def _uri(data: Mapping[str, Any]) -> str:
    for key in ("uri", "path", "texture", "value"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _normalize_scalar(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def _byte(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    return value
