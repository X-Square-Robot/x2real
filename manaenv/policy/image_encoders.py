"""Image encoders for :class:`~manaenv.policy.observation_codec.ObservationCodec`.

Each encoder is a small callable class built from the observation YAML
``video`` block (so per-encoder knobs like JPEG ``quality`` are bound at
construction). ``video.encode`` selects the class via the ``image_encoder``
discovery domain; ``ObservationCodec`` resolves it through ``global_registry``
instead of a local callable dict.
"""

from __future__ import annotations

from typing import Any

from manaenv.utils.tensor_utils import tensor_to_jpeg_b64, to_numpy_keep_shape


class JpegB64ImageEncoder:
    """Encode an image tensor/array to a base64 JPEG string."""

    def __init__(self, video_cfg: dict[str, Any] | None = None) -> None:
        cfg = video_cfg or {}
        self._quality = int(cfg.get("jpeg_quality", 95))

    def __call__(self, value: Any) -> Any:
        return tensor_to_jpeg_b64(value, quality=self._quality)


class RawNumpyImageEncoder:
    """Pass an image tensor/array through to numpy without shape changes."""

    def __init__(self, video_cfg: dict[str, Any] | None = None) -> None:
        pass

    def __call__(self, value: Any) -> Any:
        return to_numpy_keep_shape(value)


__all__ = ["JpegB64ImageEncoder", "RawNumpyImageEncoder"]
