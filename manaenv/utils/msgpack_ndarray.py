"""Safe ndarray object-hook compatibility for PolicySpace msgpack traffic.

ManaEnv's canonical wire uses ``__ndarray__/dtype/shape/data``. Some deployed gateways still emit
the older ``msgpack-numpy`` ``nd/type/kind/shape/data`` envelope. Decode both numeric forms without
calling ``msgpack_numpy.decode``: that decoder accepts pickled object arrays, which must never be
unpickled from an external model service.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_MAX_ARRAY_NDIM = 16


def _numeric_dtype(value: Any) -> np.dtype[Any] | None:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError):
        return None
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufc":
        return None
    return dtype


def _numeric_array(data: Any, dtype_value: Any, shape_value: Any) -> Any:
    dtype = _numeric_dtype(dtype_value)
    if dtype is None or not isinstance(data, (bytes, bytearray, memoryview)):
        return None
    if not isinstance(shape_value, (list, tuple)) or len(shape_value) > _MAX_ARRAY_NDIM:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape_value):
        return None
    shape = tuple(shape_value)
    expected_size = int(dtype.itemsize) * int(np.prod(shape, dtype=object))
    if expected_size != len(data):
        return None
    try:
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    except (TypeError, ValueError):
        return None


def unpack_ndarray(value: dict[Any, Any]) -> Any:
    """Decode one canonical or legacy numeric ndarray/scalar object-hook value."""
    if b"__ndarray__" in value:
        decoded = _numeric_array(
            value.get(b"data"),
            value.get(b"dtype"),
            value.get(b"shape"),
        )
        return value if decoded is None else decoded
    if b"__npgeneric__" in value:
        dtype = _numeric_dtype(value.get(b"dtype"))
        if dtype is None:
            return value
        try:
            return dtype.type(value[b"data"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return value

    if b"nd" not in value:
        return value
    # msgpack-numpy uses pickle for object arrays and descriptor reconstruction for structured
    # arrays. Neither belongs in the PolicySpace numeric tensor contract; leave them undecoded so
    # the normal action-shape validation rejects the response.
    if value.get(b"kind") in (b"O", b"V"):
        return value
    try:
        dtype = _numeric_dtype(value[b"type"])
        if dtype is None:
            return value
        if value[b"nd"] is True:
            decoded = _numeric_array(value.get(b"data"), dtype, value.get(b"shape"))
            return value if decoded is None else decoded
        if value[b"nd"] is False:
            data = value[b"data"]
            if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) != dtype.itemsize:
                return value
            return np.frombuffer(data, dtype=dtype)[0]
    except (KeyError, TypeError, ValueError):
        return value
    return value


__all__ = ["unpack_ndarray"]
