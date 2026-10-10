"""WebSocket transport that validates DNS answers and dials numeric addresses.

Passing the original URI to ``websockets`` preserves the HTTP ``Host`` header, URL path, TLS SNI,
and certificate verification. Passing an already-connected numeric socket prevents the library
from resolving a participant-controlled hostname a second time after the SSRF decision.
"""

from __future__ import annotations

import errno
import inspect
import ipaddress
import logging
import socket
import threading
import time
from typing import Any
from urllib.parse import urlparse

from websockets.sync.client import connect as _connect

_DEFAULT_OPEN_TIMEOUT_SECONDS = 10.0
_WEBSOCKET_PROTOCOL_LOGGER = logging.Logger("manaenv.pinned_endpoint_transport.websocket", level=logging.CRITICAL + 1)
_WEBSOCKET_PROTOCOL_LOGGER.disabled = True
_WEBSOCKET_PROTOCOL_LOGGER.propagate = False
_WEBSOCKET_PROTOCOL_LOGGER.addHandler(logging.NullHandler())
_TRANSIENT_NETWORK_ERRNOS = {
    errno.ECONNABORTED,
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.EHOSTDOWN,
    errno.EHOSTUNREACH,
    errno.ENETDOWN,
    errno.ENETRESET,
    errno.ENETUNREACH,
    errno.ETIMEDOUT,
}


def _transient_network_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, socket.gaierror):
        return exc.errno == getattr(socket, "EAI_AGAIN", None)
    return isinstance(exc, OSError) and exc.errno in _TRANSIENT_NETWORK_ERRNOS


def abort_websocket(websocket: Any) -> None:
    """Abort a synchronous WebSocket without waiting for the closing handshake.

    Error paths must not call ``close()``: supported ``websockets`` releases wait up to their
    ``close_timeout`` for the peer, which would make a request or reconnect deadline advisory.
    ``ClientConnection.close_socket`` is available throughout the supported websockets 11/12
    range.  The raw-socket and daemon-close fallbacks keep lightweight test doubles and foreign
    WebSocket implementations fail-safe without putting a graceful close back on the caller's
    critical path.
    """
    close_socket = getattr(websocket, "close_socket", None)
    if callable(close_socket):
        try:
            close_socket()
        except Exception:
            pass
        return

    raw_socket = getattr(websocket, "socket", None)
    if raw_socket is not None:
        try:
            raw_socket.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            raw_socket.close()
        except Exception:
            pass
        return

    close = getattr(websocket, "close", None)
    if callable(close):

        def close_in_background() -> None:
            try:
                close()
            except Exception:
                pass

        threading.Thread(
            target=close_in_background,
            name="policy-space-abort",
            daemon=True,
        ).start()


def send_websocket_bounded(
    websocket: Any,
    payload: bytes,
    *,
    timeout_seconds: float | None,
    operation: str = "PolicySpace",
) -> None:
    """Send on a synchronous WebSocket without allowing peer backpressure to pin the caller.

    ``websockets.sync`` doesn't expose a send deadline.  A peer can therefore complete the
    handshake and metadata exchange, stop reading, and hold an OD probe or Factory task forever
    once the kernel send buffer fills.  Run the blocking send in a daemon worker and tear down the
    socket when the caller's deadline expires.  The worker may unwind later, but it can no longer
    retain a live connection or block process shutdown.
    """
    timeout = _DEFAULT_OPEN_TIMEOUT_SECONDS if timeout_seconds is None else max(0.0, float(timeout_seconds))
    completed = threading.Event()
    outcome: list[BaseException] = []

    def send() -> None:
        try:
            websocket.send(payload)
        except BaseException as exc:  # noqa: BLE001 - replay the transport error in the caller
            outcome.append(exc)
        finally:
            completed.set()

    threading.Thread(target=send, name="policy-space-send", daemon=True).start()
    if not completed.wait(timeout):
        abort_websocket(websocket)
        raise TimeoutError(f"{operation} send timed out after {timeout:g} seconds")
    if outcome:
        raise outcome[0]


def _endpoint(endpoint: str) -> tuple[str, int, bool]:
    try:
        parsed = urlparse(str(endpoint or ""))
        port = parsed.port
    except ValueError as exc:
        raise ValueError("PolicySpace endpoint has an invalid port") from exc
    scheme = parsed.scheme.lower()
    if (
        scheme not in {"ws", "wss"}
        or not parsed.hostname
        or port is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("PolicySpace endpoint must be a ws:// or wss:// URL with an explicit port")
    return parsed.hostname, int(port), scheme == "wss"


def _is_public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    # ``ipaddress.is_global`` is True for multicast in supported Python releases even though a
    # multicast group can never be a participant's unicast policy server. Keep every prohibited
    # class explicit rather than relying on one evolving stdlib aggregate property.
    return bool(
        address.is_global
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_private
        and not getattr(address, "is_site_local", False)
    )


def _resolved_addresses(
    host: str,
    port: int,
    *,
    require_public_endpoint: bool,
    timeout_seconds: float,
) -> list[tuple[Any, ...]]:
    """Resolve once and validate the complete answer set before any socket is opened."""
    completed = threading.Event()
    answer_box: list[list[tuple[Any, ...]]] = []
    error_box: list[Exception] = []

    def resolve() -> None:
        try:
            answer_box.append(socket.getaddrinfo(host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM))
        except (socket.gaierror, UnicodeError, OSError) as exc:
            error_box.append(exc)
        finally:
            completed.set()

    # libc's resolver has no portable per-call deadline. A daemon worker keeps a stuck resolver
    # from pinning the Factory task; it may finish later, but its result is never consumed.
    threading.Thread(target=resolve, name="policy-space-dns", daemon=True).start()
    if not completed.wait(max(0.0, timeout_seconds)):
        raise TimeoutError(f"PolicySpace endpoint DNS resolution timed out for {host!r}")
    if error_box:
        error = error_box[0]
        if _transient_network_error(error):
            raise ConnectionError(f"PolicySpace endpoint DNS is temporarily unavailable for {host!r}") from error
        if isinstance(error, UnicodeError):
            raise ValueError(f"PolicySpace endpoint hostname is invalid: {host!r}") from error
        raise OSError(f"PolicySpace endpoint DNS resolution failed for {host!r}") from error
    answers = answer_box[0]

    resolved: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for family, kind, protocol, _canonical_name, sockaddr in answers:
        if family not in {socket.AF_INET, socket.AF_INET6} or not sockaddr:
            continue
        address = str(sockaddr[0])
        if require_public_endpoint and not _is_public_ip(address):
            raise ValueError(f"PolicySpace endpoint must resolve only to public IP addresses; got {address}")
        key = (family, kind or socket.SOCK_STREAM, protocol, tuple(sockaddr))
        if key not in seen:
            seen.add(key)
            resolved.append(key)
    if not resolved:
        raise OSError(f"PolicySpace endpoint DNS returned no usable TCP address for {host!r}")
    return resolved


def _connect_kwargs(
    *,
    raw_socket: socket.socket,
    hostname: str,
    secure: bool,
    timeout_seconds: float | None,
    max_size: int | None,
    additional_headers: dict[str, str] | None,
) -> dict[str, Any]:
    supported = inspect.signature(_connect).parameters
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in supported.values())

    def accepts(name: str) -> bool:
        return name in supported or accepts_kwargs

    kwargs: dict[str, Any] = {
        "sock": raw_socket,
        "open_timeout": timeout_seconds,
        "max_size": max_size,
        "compression": None,
    }
    if secure and accepts("server_hostname"):
        kwargs["server_hostname"] = hostname
    if accepts("proxy"):
        kwargs["proxy"] = None
    if accepts("ping_interval"):
        kwargs["ping_interval"] = None
    if accepts("logger"):
        # websockets logs every handshake header at DEBUG, including caller-supplied Authorization
        # and API-key headers. A detached disabled logger keeps those values out of application and
        # root handlers while preserving the authenticated handshake itself.
        kwargs["logger"] = _WEBSOCKET_PROTOCOL_LOGGER
    if additional_headers:
        if accepts("additional_headers"):
            kwargs["additional_headers"] = dict(additional_headers)
        elif accepts("extra_headers"):
            kwargs["extra_headers"] = dict(additional_headers)
        else:
            raise RuntimeError("installed websockets client cannot send endpoint authentication headers")
    return kwargs


def connect_websocket(
    endpoint: str,
    *,
    timeout_seconds: float | None,
    max_size: int | None,
    require_public_endpoint: bool,
    additional_headers: dict[str, str] | None = None,
) -> Any:
    """Connect without a DNS check/use race.

    Resolution occurs in the caller's own network namespace. Every returned address is validated
    before dialing starts; each candidate is then reached through its numeric ``sockaddr`` and the
    connected socket is handed to ``websockets`` with the untouched URI. Reconnect callers invoke
    this function again, so legitimate DNS rotation remains supported without trusting stale
    answers from another host or network.
    """
    hostname, port, secure = _endpoint(endpoint)
    open_timeout = _DEFAULT_OPEN_TIMEOUT_SECONDS if timeout_seconds is None else max(0.0, float(timeout_seconds))
    deadline = time.monotonic() + open_timeout
    candidates = _resolved_addresses(
        hostname,
        port,
        require_public_endpoint=require_public_endpoint,
        timeout_seconds=open_timeout,
    )
    last_error: Exception | None = None
    for family, kind, protocol, sockaddr in candidates:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"PolicySpace connection timed out after {open_timeout:g} seconds") from last_error
        raw_socket = socket.socket(family, kind, protocol)
        try:
            raw_socket.settimeout(remaining)
            raw_socket.connect(sockaddr)
            # ``websockets`` removes its own TCP timeout only when it creates the socket. With a
            # caller-supplied plain-WS socket, leaving the connect timeout installed would make an
            # otherwise healthy long-lived connection fail after that many idle seconds.
            raw_socket.settimeout(None)
            return _connect(
                endpoint,
                **_connect_kwargs(
                    raw_socket=raw_socket,
                    hostname=hostname,
                    secure=secure,
                    timeout_seconds=remaining,
                    max_size=max_size,
                    additional_headers=additional_headers,
                ),
            )
        except Exception as exc:
            last_error = exc
            raw_socket.close()
    assert last_error is not None
    if _transient_network_error(last_error) and not isinstance(last_error, (ConnectionError, TimeoutError)):
        raise ConnectionError(f"PolicySpace endpoint is temporarily unreachable: {hostname}:{port}") from last_error
    raise last_error


__all__ = ["abort_websocket", "connect_websocket", "send_websocket_bounded"]
