from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .channels import ChannelSource
from .library import MaterialAsset

TEXTURE_URI_KEYS = ("uri", "path", "texture", "value")


@dataclass(frozen=True)
class MaterialChannelBinding:
    channel: str
    source: str
    value: Any = None
    uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def texture_payload(self) -> dict[str, Any]:
        payload = {"uri": str(self.uri or "")}
        for key in ("color_space", "role", "normal_format"):
            if key in self.metadata:
                payload[key] = self.metadata[key]
        return payload

    @property
    def is_local_texture(self) -> bool:
        if not self.uri:
            return False
        return "://" not in self.uri

    @property
    def local_texture_path(self) -> Path | None:
        if not self.is_local_texture:
            return None
        return Path(str(self.uri)).expanduser()


def material_channel_bindings(
    material: MaterialAsset,
) -> tuple[MaterialChannelBinding, ...]:
    bindings: list[MaterialChannelBinding] = []
    for channel, raw_binding in material.channels.items():
        bindings.append(_channel_binding(str(channel), raw_binding))
    return tuple(bindings)


def _channel_binding(channel: str, raw_binding: Any) -> MaterialChannelBinding:
    if not isinstance(raw_binding, Mapping):
        return MaterialChannelBinding(
            channel=channel,
            source=ChannelSource.CONSTANT,
            value=raw_binding,
            metadata={},
        )

    data = dict(raw_binding)
    source = str(data.get("source") or data.get("source_type") or "").strip()
    uri = _texture_uri(
        data,
        include_value=source in {ChannelSource.TEXTURE, ChannelSource.PACKED_TEXTURE},
    )
    if not source:
        uri = _texture_uri(data, include_value=False)
        if uri:
            source = ChannelSource.TEXTURE
        elif "packing" in data or "packed_channels" in data:
            source = ChannelSource.PACKED_TEXTURE
        else:
            source = ChannelSource.CONSTANT
    if source in {ChannelSource.TEXTURE, ChannelSource.PACKED_TEXTURE} and not uri:
        uri = _texture_uri(data, include_value=True)
    if source == ChannelSource.CONSTANT:
        uri = None

    value = data.get("value", data.get("color"))
    return MaterialChannelBinding(
        channel=channel,
        source=source,
        value=value,
        uri=uri,
        metadata=data,
    )


def _texture_uri(data: Mapping[str, Any], *, include_value: bool) -> str | None:
    keys = TEXTURE_URI_KEYS if include_value else tuple(key for key in TEXTURE_URI_KEYS if key != "value")
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
    return None
