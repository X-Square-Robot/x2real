"""WebSocketRemote: transport-only PolicyRemote backed by sync WebSocket+msgpack.

Pool mode when ``server_endpoints`` is set; single-endpoint otherwise.
``client_kind`` selects codec: ``"wallx"`` (default) or ``"openpi"``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from manaenv.policy.policy_client_pool import PolicyClientPool
from manaenv.policy.policy_remote.base import PolicyRemoteBase, PolicyRemoteCFG
from manaenv.policy.server_client import PolicyServerClient
from manaenv.utils.debug_print import manaprint

_TRANSPORT_ERROR_KEYWORDS = (
    "connect",
    "connection",
    "refused",
    "timed out",
    "timeout",
    "closed",
    "going away",
    "handshake",
    "reset by peer",
    "broken pipe",
    "eof",
    "1001",
    "all endpoints failed",
    "cannot connect",
    "not connected",
)


def _is_connection_failure(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    exc_type = type(exc).__name__.lower()
    if "connectionclosed" in exc_type or "websocket" in exc_type:
        return True
    msg = str(exc).lower()
    return any(k in msg for k in _TRANSPORT_ERROR_KEYWORDS)


class WebSocketRemote(PolicyRemoteBase):
    """Sync WebSocket + msgpack PolicyRemote with retry loop."""

    def __init__(self) -> None:
        super().__init__()
        self._client: PolicyServerClient | PolicyClientPool | None = None

    def load(self, cfg: WebSocketRemoteCFG) -> None:
        super().load(cfg)
        log_io = cfg.log_io if cfg.log_io is not None else False
        log_payload = cfg.log_payload if cfg.log_payload is not None else False
        timeout = cfg.timeout_seconds if cfg.timeout_seconds else None
        if cfg.server_endpoints:
            endpoints = [(ep.get("address", "localhost"), int(ep.get("port", 8000))) for ep in cfg.server_endpoints]
            self._client = PolicyClientPool(
                endpoints,
                strategy=cfg.lb_strategy,
                client_kind=cfg.client_kind,
                log_io=log_io,
                log_payload=log_payload,
                timeout=timeout,
                require_public_endpoint=cfg.require_public_endpoint,
            )
            manaprint(
                f"INFO: [WebSocketRemote] pool with {len(endpoints)} endpoints "
                f"(client_kind={cfg.client_kind}, strategy={cfg.lb_strategy}, "
                f"log_io={log_io}, log_payload={log_payload})"
            )
        else:
            self._client = PolicyServerClient(
                cfg.server_address,
                cfg.server_port,
                client_kind=cfg.client_kind,
                log_io=log_io,
                log_payload=log_payload,
                timeout=timeout,
                require_public_endpoint=cfg.require_public_endpoint,
                endpoint=cfg.endpoint,
            )
            manaprint(
                f"INFO: [WebSocketRemote] single endpoint {cfg.endpoint or f'ws://{cfg.server_address}:{cfg.server_port}'} "
                f"(client_kind={cfg.client_kind}, log_io={log_io}, log_payload={log_payload})"
            )

    def get_actions(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("WebSocketRemote.load() must be called before get_actions()")
        cfg = self.cfg
        assert isinstance(cfg, WebSocketRemoteCFG)
        retry_forever = cfg.retry_forever
        retry_attempts = max(cfg.retry_attempts, 1)
        retry_interval = max(float(cfg.retry_interval_seconds), 0.0)
        # Hard wall-clock cap on consecutive-failure reconnects, applied even in
        # retry_forever mode: transient blips still recover, but a persistently
        # unreachable server fails the run cleanly instead of spinning forever
        # (which hangs the sim and leaks the GPU). 0 disables the cap.
        retry_deadline = max(float(cfg.retry_max_seconds), 0.0)

        attempt = 1
        first_failure_at: float | None = None
        while True:
            try:
                self._client.ensure_connected()
                response = self._client.predict(request)
                if attempt > 1:
                    manaprint(f"INFO: [WebSocketRemote] inference recovered after {attempt - 1} retries")
                if isinstance(response, dict) and "error" in response:
                    raise RuntimeError(f"Remote policy server returned error: {response['error']}")
                return response
            except Exception as exc:
                is_transport = _is_connection_failure(exc)
                if not is_transport:
                    raise
                if not retry_forever:
                    self._client.close()
                now = time.monotonic()
                if first_failure_at is None:
                    first_failure_at = now
                elapsed = now - first_failure_at
                if attempt <= 5 or attempt % 10 == 0:
                    manaprint(f"WARNING: [WebSocketRemote] connect attempt {attempt} failed: {exc}")
                deadline_hit = retry_deadline > 0 and elapsed >= retry_deadline
                if (not retry_forever and attempt >= retry_attempts) or deadline_hit:
                    if deadline_hit:
                        manaprint(
                            f"ERROR: [WebSocketRemote] giving up after {elapsed:.0f}s of failed reconnects "
                            f"to ws://{cfg.server_address}:{cfg.server_port} — is the policy server up?"
                        )
                    raise
                if retry_interval > 0:
                    if attempt <= 5 or attempt % 10 == 0:
                        manaprint(f"INFO: [WebSocketRemote] reconnecting in {retry_interval:.1f}s")
                    time.sleep(retry_interval)
                attempt += 1

    def cleanup(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                manaprint(f"WARNING: [WebSocketRemote] error closing client: {exc}")
        self._client = None


@dataclass
class WebSocketRemoteCFG(PolicyRemoteCFG):
    """Serving yaml schema for :class:`WebSocketRemote`.

    Logging:
      ``log_io``      — per-inference "Sending/Received N bytes" (cheap, safe).
      ``log_payload`` — pretty-print full request/response dict per inference.
                        MUST stay False in long benchmarks — full-payload dumps
                        flood Ray actor stdout and break the raylet log RPC
                        channel (ActorUnavailableError: Socket closed rpc_code: 14).
      ``debug``       — deprecated YAML compat alias; ignored at runtime.
                        Existing serving YAMLs may still carry ``debug: false``.

    Retry:
      ``retry_forever``  True (default) keeps client alive and relies on
      ``ensure_connected()`` for reconnection.  False actively closes client
      on each failure and gives up after ``retry_attempts``.
    """

    name: str = ""
    server_address: str = "localhost"
    server_port: int = 8000
    endpoint: str | None = None
    server_endpoints: list[dict[str, Any]] | None = None
    lb_strategy: str = "round_robin"
    client_kind: str = "wallx"
    # Public evaluation config generation enables this; local/operator configs remain compatible.
    require_public_endpoint: bool = False
    # Accepted from YAML for backward compat; not used at runtime.
    debug: bool = False
    log_io: bool | None = None
    log_payload: bool | None = None
    retry_attempts: int = 3
    retry_interval_seconds: float = 2.0
    retry_forever: bool = True
    # Wall-clock cap (seconds) on consecutive-failure reconnects, honored even when
    # retry_forever is True; 0 disables it. Bounds the "server never comes up" hang.
    retry_max_seconds: float = 120.0
    timeout_seconds: float | None = 3.0
    metadata: dict[str, Any] = field(default_factory=dict)
    class_type: type = WebSocketRemote
