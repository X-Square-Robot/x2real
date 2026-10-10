"""Image encoder discovery for the global lazy registry.

``video.encode`` in the observation YAML maps onto these encoder symbols;
``ObservationCodec`` resolves the class via ``global_registry.get_impl``
instead of a local callable dict.
"""

from __future__ import annotations

from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

__all__ = ["IMAGE_ENCODER_DOMAIN", "ENCODE_TO_SYMBOL", "discover_image_encoder"]

IMAGE_ENCODER_DOMAIN: str = "image_encoder"

_MODULE: str = "manaenv.policy.image_encoders"

# video.encode value -> encoder symbol.
ENCODE_TO_SYMBOL: dict[str, str] = {
    "jpeg_b64": "JpegB64ImageEncoder",
    "raw_numpy": "RawNumpyImageEncoder",
}

_IMAGE_ENCODER_SYMBOLS: dict[str, str] = {
    "JpegB64ImageEncoder": _MODULE,
    "RawNumpyImageEncoder": _MODULE,
}


def discover_image_encoder(symbol: str) -> ComponentSpec | None:
    """Resolve an image-encoder symbol to its class."""
    module_path = _IMAGE_ENCODER_SYMBOLS.get(symbol)
    if module_path is None:
        return None
    return ComponentSpec(module_path=module_path, class_name=symbol, symbol=symbol)


register_provider(IMAGE_ENCODER_DOMAIN, discover_image_encoder)
