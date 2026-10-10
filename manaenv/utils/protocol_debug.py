"""Helpers for readable wire-protocol debug output."""

from __future__ import annotations

from pprint import pformat
from typing import Any

import numpy as np

from manaenv.utils.tensor_utils import as_torch

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in runtime env
    torch = None


_IMAGE_CONTAINER_KEYS = {"images", "views", "camera_obs"}


def _looks_like_image_array(value: np.ndarray) -> bool:
    if value.ndim < 2:
        return False
    if value.ndim >= 3 and (value.shape[-1] in (1, 3, 4) or value.shape[0] in (1, 3, 4)):
        return True
    return False


def _format_array(value: np.ndarray, *, include_values: bool = True) -> str:
    if not include_values:
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    threshold = min(int(value.size), 64)
    return (
        f"ndarray(shape={value.shape}, dtype={value.dtype}, "
        f"values={np.array2string(value, precision=6, threshold=threshold)})"
    )


def _to_debug_object(value: Any, *, hide_image_data: bool = False, in_image_context: bool = False) -> Any:
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for k, v in value.items():
            key_lower = str(k).lower()
            child_image_ctx = in_image_context or (hide_image_data and key_lower in _IMAGE_CONTAINER_KEYS)
            result[k] = _to_debug_object(v, hide_image_data=hide_image_data, in_image_context=child_image_ctx)
        return result
    if isinstance(value, list):
        return [_to_debug_object(v, hide_image_data=hide_image_data, in_image_context=in_image_context) for v in value]
    if isinstance(value, tuple):
        return tuple(
            _to_debug_object(v, hide_image_data=hide_image_data, in_image_context=in_image_context) for v in value
        )
    value = as_torch(value)
    if torch is not None and isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        include_values = not (in_image_context or (hide_image_data and _looks_like_image_array(arr)))
        return _format_array(arr, include_values=include_values)
    if isinstance(value, np.ndarray):
        include_values = not (in_image_context or (hide_image_data and _looks_like_image_array(value)))
        return _format_array(value, include_values=include_values)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str):
        if in_image_context:
            return f"str(len={len(value)})"
        if len(value) > 160:
            return f"str(len={len(value)}, preview={value[:160]!r})"
        return value
    if isinstance(value, bytes):
        if in_image_context:
            return f"bytes(len={len(value)})"
        return f"bytes(len={len(value)})"
    return value


def format_debug_payload(payload: dict[str, Any], *, hide_image_data: bool = False) -> str:
    return pformat(_to_debug_object(payload, hide_image_data=hide_image_data), width=120, sort_dicts=False)
