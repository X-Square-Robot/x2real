from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import gettempdir
from typing import Any

DEFAULT_GENERATED_TEXTURE_DIR = Path(gettempdir()) / "manaenv_asset_randomizer" / "generated_textures"


def generated_texture_path(
    *,
    source_path: Path,
    output_dir: Path,
    channel: str,
    operation: str,
    value: Any,
) -> Path:
    source_stat = source_path.stat()
    payload = {
        "source": str(source_path.resolve()),
        "size": int(source_stat.st_size),
        "mtime_ns": int(source_stat.st_mtime_ns),
        "channel": channel,
        "operation": operation,
        "value": _jsonable(value),
    }
    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return output_dir / f"{source_path.stem}_{channel}_{operation}_{digest}.png"


def write_value_transformed_texture(
    *,
    source_path: Path,
    output_dir: Path,
    channel: str,
    operation: str,
    value: Any,
) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL is required for generated texture value transforms") from exc

    output_path = generated_texture_path(
        source_path=source_path,
        output_dir=output_dir,
        channel=channel,
        operation=operation,
        value=value,
    )
    if output_path.is_file():
        return output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source_image:
        image = _editable_image(source_image)
        transformed = _transform_image_values(image=image, operation=operation, value=value)
        transformed.save(output_path)
    return output_path


def _editable_image(image: Any) -> Any:
    if image.mode in {"L", "LA", "RGB", "RGBA"}:
        return image.copy()
    if "A" in image.mode:
        return image.convert("RGBA")
    return image.convert("RGB")


def _transform_image_values(*, image: Any, operation: str, value: Any) -> Any:
    channels = image.split()
    alpha = None
    color_channels = channels
    if image.mode in {"LA", "RGBA"}:
        alpha = channels[-1]
        color_channels = channels[:-1]
    values = _per_channel_values(value=value, count=len(color_channels))
    transformed_channels = []
    for channel_image, amount in zip(color_channels, values, strict=True):
        lut = _lut(operation=operation, amount=float(amount))
        transformed_channels.append(channel_image.point(lut))
    if alpha is not None:
        transformed_channels.append(alpha)
    return _merge_like(image.mode, transformed_channels)


def _per_channel_values(*, value: Any, count: int) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return tuple(float(value) for _ in range(count))
    try:
        values = tuple(float(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"texture transform value must be scalar or sequence, got {type(value).__name__}") from exc
    if len(values) == 1:
        return tuple(values[0] for _ in range(count))
    if len(values) < count:
        raise ValueError(f"texture transform value has {len(values)} channels, expected at least {count}")
    return values[:count]


def _lut(*, operation: str, amount: float) -> list[int]:
    if operation == "scale":
        return [_byte(value * amount) for value in range(256)]
    if operation == "bias":
        return [_byte(value + amount * 255.0) for value in range(256)]
    raise ValueError(f"unsupported generated texture operation: {operation}")


def _byte(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _merge_like(mode: str, channels: list[Any]) -> Any:
    from PIL import Image

    return Image.merge(mode, channels)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    return value
