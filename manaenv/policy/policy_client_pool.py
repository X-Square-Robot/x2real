"""Multi-endpoint policy client pool with load balancing and failover.

Wraps multiple :class:`PolicyServerClient` behind a single ``predict()``
interface. Enabled when ``server_endpoints`` is specified in serving YAML.
"""

from __future__ import annotations

import os
import time
from typing import Any

from manaenv.policy.server_client import PolicyServerClient
from manaenv.utils.debug_print import manaprint


class PolicyClientPool:
    """Round-robin / least-latency pool of ``PolicyServerClient`` connections."""

    def __init__(
        self,
        endpoints: list[tuple[str, int]],
        strategy: str = "round_robin",
        max_retries: int = 2,
        health_check_interval: float = 30.0,
        client_kind: str = "wallx",
        log_io: bool = False,
        log_payload: bool = False,
        timeout: float | None = None,
        require_public_endpoint: bool = False,
    ):
        if not endpoints:
            raise ValueError("PolicyClientPool requires at least one endpoint")

        self._endpoints = list(endpoints)
        self._strategy = strategy
        self._max_retries = min(max_retries, len(endpoints) - 1)
        self._health_check_interval = health_check_interval
        self._log_io = log_io

        self._clients: list[PolicyServerClient] = [
            PolicyServerClient(
                addr,
                port,
                client_kind=client_kind,
                log_io=log_io,
                log_payload=log_payload,
                timeout=timeout,
                require_public_endpoint=require_public_endpoint,
            )
            for addr, port in endpoints
        ]
        self._healthy: list[bool] = [True] * len(self._clients)
        self._latencies: list[float] = [0.0] * len(self._clients)
        self._attempt_counts: list[int] = [0] * len(self._clients)
        self._success_counts: list[int] = [0] * len(self._clients)
        self._failure_counts: list[int] = [0] * len(self._clients)
        self._total_latency: list[float] = [0.0] * len(self._clients)
        self._last_health_check: list[float] = [0.0] * len(self._clients)
        self._rr_index: int = 0
        self._request_seq: int = 0
        self._metadata: dict[str, Any] | None = None
        self._pid = os.getpid()

    @property
    def metadata(self) -> dict[str, Any] | None:
        return self._metadata

    @property
    def connected(self) -> bool:
        return any(c.connected for c in self._clients)

    # -- connection management --

    def ensure_connected(self) -> None:
        if self.connected:
            return
        for i, client in enumerate(self._clients):
            try:
                meta = client.connect()
                self._healthy[i] = True
                if self._metadata is None:
                    self._metadata = meta
                manaprint(f"INFO: [PolicyClientPool] Connected to {client.uri}")
                return
            except Exception as exc:
                self._healthy[i] = False
                manaprint(f"WARNING: [PolicyClientPool] Failed to connect {client.uri}: {exc}")
        raise ConnectionError("PolicyClientPool: cannot connect to any endpoint")

    # -- predict (with failover) --

    def predict(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.ensure_connected()

        idx = self._select_endpoint()
        tried: set[int] = set()

        for _ in range(self._max_retries + 1):
            if idx in tried:
                idx = self._next_healthy(exclude=tried)
                if idx is None:
                    break
            tried.add(idx)

            client = self._clients[idx]
            if not client.connected:
                try:
                    client.connect()
                    self._healthy[idx] = True
                except Exception:
                    self._healthy[idx] = False
                    idx = self._next_healthy(exclude=tried)
                    if idx is None:
                        break
                    continue

            try:
                t0 = time.monotonic()
                self._request_seq += 1
                self._attempt_counts[idx] += 1
                request_seq = self._request_seq
                if self._log_io:
                    manaprint(
                        f"INFO: [PolicyClientPool] route request={request_seq} pid={self._pid} "
                        f"endpoint={client.uri} index={idx + 1}/{len(self._clients)} "
                        f"strategy={self._strategy} endpoint_attempts={self._attempt_counts[idx]}"
                    )
                result = client.predict(observation)
                latency = time.monotonic() - t0
                self._latencies[idx] = latency
                self._success_counts[idx] += 1
                self._total_latency[idx] += latency
                if self._log_io:
                    server_timing = self._format_server_timing(result)
                    timing_suffix = f" server_timing={server_timing}" if server_timing else ""
                    manaprint(
                        f"INFO: [PolicyClientPool] complete request={request_seq} pid={self._pid} "
                        f"endpoint={client.uri} latency_ms={latency * 1000:.1f} "
                        f"endpoint_successes={self._success_counts[idx]}{timing_suffix}"
                    )
                return result
            except Exception as exc:
                manaprint(f"WARNING: [PolicyClientPool] Endpoint {client.uri} failed: {exc}")
                self._failure_counts[idx] += 1
                self._healthy[idx] = False
                client.close()

        raise RuntimeError("PolicyClientPool: all endpoints failed")

    # -- cleanup --

    def close(self) -> None:
        if self._log_io:
            self._log_summary()
        for client in self._clients:
            client.close()

    # -- endpoint selection --

    def _select_endpoint(self) -> int:
        if self._strategy == "least_latency":
            return self._pick_least_latency()
        return self._pick_round_robin()

    def _pick_round_robin(self) -> int:
        n = len(self._clients)
        for _ in range(n):
            idx = self._rr_index % n
            self._rr_index += 1
            if self._healthy[idx]:
                return idx
        return 0

    def _pick_least_latency(self) -> int:
        best_idx, best_lat = 0, float("inf")
        for i, (healthy, lat) in enumerate(zip(self._healthy, self._latencies)):
            if healthy and lat < best_lat:
                best_idx, best_lat = i, lat
        return best_idx

    def _next_healthy(self, exclude: set[int] | None = None) -> int | None:
        now = time.monotonic()
        exclude = exclude or set()
        for i in range(len(self._clients)):
            if i in exclude:
                continue
            if self._healthy[i]:
                return i
            if now - self._last_health_check[i] > self._health_check_interval:
                self._last_health_check[i] = now
                try:
                    self._clients[i].connect()
                    self._healthy[i] = True
                    return i
                except Exception:
                    pass
        return None

    @staticmethod
    def _format_server_timing(result: dict[str, Any]) -> str:
        if not isinstance(result, dict):
            return ""
        timing = result.get("server_timing")
        if not isinstance(timing, dict) or not timing:
            return ""
        parts = []
        for key in sorted(timing):
            value = timing[key]
            if isinstance(value, (int, float)):
                parts.append(f"{key}={float(value):.1f}")
            else:
                parts.append(f"{key}={value}")
        return ",".join(parts)

    def _log_summary(self) -> None:
        if not any(self._attempt_counts):
            return
        parts = []
        for i, client in enumerate(self._clients):
            avg_ms = self._total_latency[i] * 1000.0 / self._success_counts[i] if self._success_counts[i] else 0.0
            parts.append(
                f"{client.uri}:attempts={self._attempt_counts[i]},success={self._success_counts[i]},"
                f"fail={self._failure_counts[i]},last_ms={self._latencies[i] * 1000:.1f},"
                f"avg_ms={avg_ms:.1f},healthy={self._healthy[i]}"
            )
        manaprint(f"INFO: [PolicyClientPool] summary pid={self._pid} strategy={self._strategy} " + " | ".join(parts))

    def __repr__(self) -> str:
        eps = [f"{a}:{p}" for a, p in self._endpoints]
        return f"PolicyClientPool({eps}, strategy={self._strategy})"
