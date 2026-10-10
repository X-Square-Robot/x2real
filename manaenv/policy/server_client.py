"""WebSocket inference client for remote policy servers.

Sync client communicating via msgpack over WebSocket. ``client_kind``
selects the wire codec:

  * ``"wallx"``  — msgpack-numpy encoding; decoder is a superset that
    also handles the openpi ``__ndarray__`` envelope so wallx servers
    can accept data packed by either codec.
  * ``"openpi"`` — bare ``__ndarray__`` / ``__npgeneric__`` envelope
    (no msgpack-numpy dependency on the server side).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from manaenv.utils.debug_print import manaprint
from manaenv.utils.protocol_debug import format_debug_payload

try:
    import msgpack

    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False

try:
    import msgpack_numpy

    _HAS_MSGPACK_NUMPY = True
except ImportError:
    msgpack_numpy = None  # type: ignore[assignment]
    _HAS_MSGPACK_NUMPY = False

try:
    import websockets  # noqa: F401 - dependency availability gate for the explicit error below

    from manaenv.utils.pinned_endpoint_transport import abort_websocket, connect_websocket, send_websocket_bounded

    _HAS_WEBSOCKETS = True
except ImportError:
    connect_websocket = None  # type: ignore[assignment]
    send_websocket_bounded = None  # type: ignore[assignment]
    abort_websocket = None  # type: ignore[assignment]
    _HAS_WEBSOCKETS = False


_DEFAULT_TIMEOUT_SECONDS = 10.0


# -- numpy codec helpers --


def _pack_numpy(obj: Any) -> dict[bytes, Any]:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    raise TypeError(f"Unknown type: {type(obj)}")


def _unpack_numpy(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _unpack_wallx(obj: dict[Any, Any]) -> Any:
    """Wallx decoder: tries msgpack_numpy first, falls back to openpi envelope.

    This makes wallx a superset decoder — it can read data packed by either
    codec, which matters when mixed clients hit the same server.
    """
    if _HAS_MSGPACK_NUMPY:
        try:
            decoded = msgpack_numpy.decode(obj)  # type: ignore[union-attr]
            if decoded is not obj:
                return decoded
        except Exception:
            pass
    return _unpack_numpy(obj)


# -- codec dispatch --

_CODEC_HOOKS = {
    "openpi": (_pack_numpy, _unpack_numpy),
    "wallx": (
        msgpack_numpy.encode if _HAS_MSGPACK_NUMPY else _pack_numpy,  # type: ignore[union-attr]
        _unpack_wallx,
    ),
}


class PolicyServerClient:
    """Sync WebSocket client for msgpack-based policy inference servers.

    Parameters
    ----------
    client_kind:
        ``"wallx"`` (default) or ``"openpi"`` — selects pack/unpack hooks
        from ``_CODEC_HOOKS``.
    log_io:
        Per-inference "Sending/Received N bytes" lines. Cheap; safe to
        leave on in production.
    log_payload:
        Pretty-print full request/response dict per inference. Very verbose.
        MUST stay False in long benchmarks — full-payload dumps flood Ray
        actor stdout and break the raylet log RPC channel
        (ActorUnavailableError: Socket closed rpc_code: 14).
    """

    def __init__(
        self,
        address: str = "localhost",
        port: int = 8000,
        client_kind: str = "wallx",
        log_io: bool = False,
        log_payload: bool = False,
        timeout: float | None = None,
        require_public_endpoint: bool = False,
        endpoint: str | None = None,
    ):
        kind = client_kind.strip().lower()
        if kind not in _CODEC_HOOKS:
            raise ValueError(f"unsupported client_kind={client_kind!r} (expected {tuple(_CODEC_HOOKS)})")
        self._client_kind = kind
        self._pack_hook, self._unpack_hook = _CODEC_HOOKS[kind]

        self.address = address
        self.port = port
        self.uri = str(endpoint or f"ws://{address}:{port}")
        self._connection = None
        self._metadata: dict[str, Any] = {}
        self._connected = False
        self._log_io = log_io
        self._log_payload = log_payload
        self._timeout: float | None = float(timeout) if timeout is not None else None
        self._require_public_endpoint = bool(require_public_endpoint)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def connect(self) -> dict[str, Any]:
        if not _HAS_WEBSOCKETS:
            raise ImportError("websockets is required: pip install websockets")
        if not _HAS_MSGPACK:
            raise ImportError("msgpack is required: pip install msgpack")
        assert abort_websocket is not None and connect_websocket is not None

        manaprint(f"INFO: [PolicyServerClient] Connecting to {self.uri}")
        timeout = _DEFAULT_TIMEOUT_SECONDS if self._timeout is None else max(0.0, self._timeout)
        deadline = time.monotonic() + timeout
        previous, self._connection = self._connection, None
        self._connected = False
        self._metadata = {}
        if previous is not None:
            abort_websocket(previous)
        connection = None
        try:
            connection = connect_websocket(
                self.uri,
                timeout_seconds=timeout,
                max_size=None,
                require_public_endpoint=self._require_public_endpoint,
            )
            self._connection = connection
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"PolicyServerClient metadata handshake exceeded its {timeout:g}-second deadline")
            metadata_bytes = connection.recv(timeout=remaining)
            metadata = self._unpack_response(metadata_bytes)
            if not isinstance(metadata, dict):
                raise TypeError("PolicyServerClient metadata must decode to an object")
            self._metadata = metadata
        except BaseException:
            self._connection = None
            self._connected = False
            self._metadata = {}
            if connection is not None:
                abort_websocket(connection)
            raise
        self._connected = True
        manaprint(f"INFO: [PolicyServerClient] Connected to {self.uri}")
        if self._metadata:
            if self._log_payload:
                manaprint(f"INFO: [PolicyServerClient] Server metadata:\n{format_debug_payload(self._metadata)}")
            else:
                manaprint(f"INFO: [PolicyServerClient] Server metadata: {self._metadata}")
        return self._metadata

    def ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def predict(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("Not connected. Call connect() first.")

        request_bytes = self._pack_request(observation)
        if self._log_io:
            manaprint(f"INFO: [PolicyServerClient] Sending {len(request_bytes)} bytes to {self.uri}")
        try:
            assert send_websocket_bounded is not None
            timeout = _DEFAULT_TIMEOUT_SECONDS if self._timeout is None else max(0.0, self._timeout)
            deadline = time.monotonic() + timeout
            send_websocket_bounded(
                self._connection,
                request_bytes,
                timeout_seconds=timeout,
                operation="PolicyServerClient",
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"PolicyServerClient request timed out after {timeout:g} seconds")
            response_bytes = self._connection.recv(timeout=remaining)
        except Exception:
            conn = self._connection
            self._connection = None
            self._connected = False
            if conn is not None:
                assert abort_websocket is not None
                abort_websocket(conn)
            raise
        if isinstance(response_bytes, str):
            raise RuntimeError(f"Server error: {response_bytes}")
        response = self._unpack_response(response_bytes)
        if self._log_io:
            manaprint(f"INFO: [PolicyServerClient] Received {len(response_bytes)} bytes from {self.uri}")
        if self._log_payload:
            manaprint(f"INFO: [PolicyServerClient] Response payload:\n{format_debug_payload(response)}")
        return response

    def _pack_request(self, payload: dict[str, Any]) -> bytes:
        return msgpack.packb(payload, default=self._pack_hook)

    def _unpack_response(self, payload: bytes) -> dict[str, Any]:
        return msgpack.unpackb(payload, object_hook=self._unpack_hook)

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception as e:
                manaprint(f"WARNING: [PolicyServerClient] Error closing: {e}")
        self._connection = None
        self._connected = False

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"PolicyServerClient({self.uri}, kind={self._client_kind}, {status})"
