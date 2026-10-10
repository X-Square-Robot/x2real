"""PolicySpaceImpl: WebSocket bridge to an external PolicySpace server.

Replaces the former ``PolicyAdapterImpl → manaenv.sdk`` path with a clean
client-server split:

  * The external PolicySpace server owns the model, its dependencies,
    checkpoint loading, frame history, and inference logic.
  * This Impl owns only the sim-side obs encoding, action decoding, and
    the WebSocket transport — no model code is imported into the Isaac
    Sim process.

Wire protocol: msgpack-encoded dicts over WebSocket (same codec as
openpi/wallx). The Policy Space server implements:

  * ``initialize_episode(episode_info)`` — prepare episode state
  * ``ingest_observation(obs_dict)`` — buffer one frame when required
  * ``infer_actions(obs_dict) -> {"actions": ndarray[T, D]}`` — run inference
  * ``finalize_episode(episode_info)`` — release episode state

The obs dict sent to the server follows the ManaEnv standardized format::

    {
        "images": {"<camera>": ndarray[H,W,3] uint8, ...},
        "state": {"<field>": ndarray float32, ...},
        "instruction": str,
        "step": int,
        "episode_id": str,
    }

Action decoding reuses the existing ``EEAbsoluteIKTransform`` (for
``delta_ee``/``absolute_ee`` types) or direct joint mapping (for
``delta_joint``/``absolute_joint``), same as the wallx path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, ClassVar

import numpy as np
import torch
from policy_space_protocol.contracts import SchemaPair, schema_pairs_from_metadata, select_schema_pair
from policy_space_protocol.protocol import (
    SESSION_MODE_MULTIPLEXED_SERIAL,
    SESSION_OWNER_BUSY_MARKER,
    metadata_mismatches,
)

from manaenv.policy import gripper_convention
from manaenv.policy.policy_embodiment import PolicyEmbodimentCFG
from manaenv.policy.policy_implementation.base import PolicyImplementationBase, PolicyImplementationCFG
from manaenv.utils.debug_print import manaprint
from manaenv.utils.msgpack_ndarray import unpack_ndarray
from manaenv.utils.pinned_endpoint_transport import abort_websocket, connect_websocket, send_websocket_bounded
from manaenv.utils.tensor_utils import normalize_env_ids

_MODEL_IO_PREFIX = "MANA_MODEL_IO "
# Successful inference already proves the rollout is alive; bound that existing
# event instead of adding a timer or a second heartbeat protocol.
_MODEL_IO_HEARTBEAT_SECONDS = 60.0
_INTERNAL_ENV_IDX_KEY = "__manaenv_env_idx"
_INTERNAL_EPISODE_CONTEXT_KEY = "__manaenv_episode_context"
_ACTION_SCHEMA_DIMS = {
    "bimanual_joint_position@1": 14,
    "bimanual_joint_position_26d@1": 26,
    "bimanual_ee_absolute@1": 14,
    "single_arm_joint_position@1": 8,
    "mobile_semantic_20d@1": 20,
}
_RUNTIME_HEADERS_FD_ENV = "MANA_POLICYSPACE_ADDITIONAL_HEADERS_FD"
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}$")
_FORBIDDEN_RUNTIME_HEADERS = {
    "connection",
    "content-length",
    "cookie",
    "host",
    "proxy-authorization",
    "sec-websocket-key",
    "sec-websocket-protocol",
    "sec-websocket-version",
    "transfer-encoding",
    "upgrade",
}
_RUNTIME_HEADERS_CACHE: dict[str, str] | None = None


def _runtime_additional_headers() -> dict[str, str]:
    """Read attempt auth from an inherited memfd without logging or exporting its values."""
    global _RUNTIME_HEADERS_CACHE
    if _RUNTIME_HEADERS_CACHE is not None:
        return dict(_RUNTIME_HEADERS_CACHE)
    raw_descriptor = os.environ.get(_RUNTIME_HEADERS_FD_ENV)
    if raw_descriptor is None:
        return {}
    descriptor = -1
    try:
        descriptor = int(raw_descriptor)
        if descriptor < 3:
            raise ValueError
        encoded = os.pread(descriptor, 20_001, 0)
        if not encoded or len(encoded) > 20_000:
            raise ValueError
        raw = encoded.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("PolicySpace runtime authentication headers are invalid") from exc
    finally:
        os.environ.pop(_RUNTIME_HEADERS_FD_ENV, None)
        if descriptor >= 3:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if "\x00" in raw:
        raise RuntimeError("PolicySpace runtime authentication headers are invalid")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("PolicySpace runtime authentication headers are invalid") from exc
    if not isinstance(document, dict) or not 1 <= len(document) <= 8:
        raise RuntimeError("PolicySpace runtime authentication headers are invalid")
    headers: dict[str, str] = {}
    for name, value in document.items():
        if (
            not isinstance(name, str)
            or _HEADER_NAME.fullmatch(name) is None
            or name.casefold() in _FORBIDDEN_RUNTIME_HEADERS
            or not isinstance(value, str)
            or not value
            or len(value) > 16_384
            or "\r" in value
            or "\n" in value
            or "\x00" in value
        ):
            raise RuntimeError("PolicySpace runtime authentication headers are invalid")
        headers[name] = value
    _RUNTIME_HEADERS_CACHE = dict(headers)
    return headers


def _emit_model_io(event: str, **fields: Any) -> None:
    """Emit one bounded structured line captured by the Factory live-log channel."""
    print(
        _MODEL_IO_PREFIX + json.dumps({"event": event, **fields}, ensure_ascii=True, separators=(",", ":")),
        flush=True,
    )


def _is_camera_field(name: str) -> bool:
    lowered = name.lower()
    return any(s in lowered for s in ("_cam", "_image", "_rgb", "view", "camera"))


def _to_rgb_uint8(raw: Any) -> np.ndarray | None:
    """Convert any image tensor to HWC uint8 RGB."""
    from manaenv.utils.tensor_utils import to_numpy_keep_shape

    arr = to_numpy_keep_shape(raw)
    if arr is None:
        return None
    if arr.size == 0 or not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        return None
    if not np.isfinite(arr).all():
        return None
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        return None
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _select_camera_env_item(raw: Any, env_idx: int | None) -> Any:
    """Select a batched camera frame without falling back to another environment."""
    if env_idx is None:
        return raw
    shape = getattr(raw, "shape", None)
    if shape is None:
        try:
            shape = np.shape(raw)
        except Exception:
            return None
    if len(shape) != 4:
        return raw
    idx = int(env_idx)
    if idx < 0 or idx >= int(shape[0]):
        return None
    return raw[idx]


def _resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an RGB frame before it crosses the policy WebSocket."""
    if height <= 0 or width <= 0 or rgb.shape[:2] == (height, width):
        return rgb
    import cv2

    return np.ascontiguousarray(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA))


def _summarize_images(images: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build compact fingerprints for the exact frames sent at an inference refill."""
    summaries: dict[str, dict[str, Any]] = {}
    for name in sorted(images):
        image = np.ascontiguousarray(np.asarray(images[name]))
        summaries[name] = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "digest": hashlib.blake2s(image.view(np.uint8), digest_size=8).hexdigest(),
            "mean": round(float(image.mean()), 3),
            "std": round(float(image.std()), 3),
        }
    return summaries


def _is_retryable_transport_error(exc: Exception) -> bool:
    """Return whether a failed request can be retried on a fresh WebSocket."""
    try:
        from websockets.exceptions import ConnectionClosed
    except ImportError:  # pragma: no cover - every supported websocket runtime provides this
        connection_closed: tuple[type[BaseException], ...] = ()
    else:
        connection_closed = (ConnectionClosed,)
    if isinstance(exc, connection_closed + (ConnectionError, EOFError, TimeoutError)):
        return True
    if isinstance(exc, RuntimeError) and SESSION_OWNER_BUSY_MARKER in str(exc):
        return True

    # A participant endpoint commonly sits behind a tunnel or reverse proxy. During a restart it
    # can answer the WebSocket handshake with a transient HTTP response rather than refusing TCP.
    # websockets <=12 exposes ``status_code`` on InvalidStatusCode; >=13 exposes
    # ``response.status_code`` on InvalidStatus. Limit retries to explicit transient statuses so
    # authentication, routing, URI and TLS failures remain immutable-input failures.
    try:
        from websockets.exceptions import InvalidHandshake
    except ImportError:  # pragma: no cover - every supported websocket runtime provides this
        return False
    if not isinstance(exc, InvalidHandshake):
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        status_code = int(status)
    except (TypeError, ValueError):
        return False
    return status_code in {408, 425, 429} or 500 <= status_code <= 599


class _TransientPolicySpaceConnectionError(ConnectionError):
    """Connection establishment failed for a transport reason that may recover.

    Keeping this distinct from ``RuntimeError`` is important: PolicySpace uses RuntimeError for
    deterministic server/application errors too.  Retrying those for the full reconnect budget
    only burns Factory time and can never change the result.
    """


class _PolicySpaceClient:
    """Msgpack WebSocket client for PolicySpace servers."""

    def __init__(
        self,
        endpoint: str,
        timeout_seconds: float,
        *,
        max_message_size: int = 64 * 1024 * 1024,
        require_public_endpoint: bool = False,
        additional_headers: dict[str, str] | None = None,
    ) -> None:
        import msgpack

        self._msgpack = msgpack
        self._timeout = timeout_seconds
        self._max_message_size = max(1024, int(max_message_size))
        deadline = time.monotonic() + timeout_seconds
        self._ws = connect_websocket(
            endpoint,
            timeout_seconds=timeout_seconds,
            max_size=self._max_message_size,
            require_public_endpoint=require_public_endpoint,
            additional_headers=additional_headers,
        )
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("PolicySpace metadata handshake exceeded its deadline")
            raw = self._ws.recv(timeout=remaining)
            if not isinstance(raw, bytes):
                raise TypeError("PolicySpace metadata must be msgpack bytes")
            metadata = self._unpack(raw)
            if not isinstance(metadata, dict):
                raise TypeError("PolicySpace metadata must decode to an object")
            self._server_metadata = metadata
        except BaseException:
            websocket, self._ws = self._ws, None
            abort_websocket(websocket)
            raise

    @staticmethod
    def _pack_array(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }
        if isinstance(value, np.generic):
            return {
                b"__npgeneric__": True,
                b"data": value.item(),
                b"dtype": value.dtype.str,
            }
        return value

    @staticmethod
    def _unpack_array(value: dict) -> Any:
        return unpack_ndarray(value)

    def _pack(self, value: Any) -> bytes:
        return self._msgpack.packb(value, default=self._pack_array)

    def _unpack(self, value: bytes) -> Any:
        return self._msgpack.unpackb(value, object_hook=self._unpack_array)

    def _send_bounded(self, payload: bytes, timeout_seconds: float) -> None:
        send_websocket_bounded(
            self._ws,
            payload,
            timeout_seconds=timeout_seconds,
            operation="PolicySpace",
        )

    def request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> Any:
        msg = dict(payload)
        if endpoint:
            msg["endpoint"] = endpoint
        timeout = self._timeout if timeout_seconds is None else max(0.0, float(timeout_seconds))
        deadline = time.monotonic() + timeout
        encoded = self._pack(msg)
        if len(encoded) > self._max_message_size:
            raise ValueError(f"PolicySpace request exceeds max_message_size={self._max_message_size}")
        self._send_bounded(encoded, timeout)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"PolicySpace request timed out after {timeout:g} seconds")
        response = self._ws.recv(timeout=remaining)
        if isinstance(response, str):
            raise RuntimeError(f"PolicySpace server error: {response}")
        return self._unpack(response)

    def close(self) -> None:
        websocket, self._ws = self._ws, None
        if websocket is None:
            return
        try:
            websocket.close()
        except Exception:
            pass

    def abort(self) -> None:
        websocket, self._ws = self._ws, None
        if websocket is not None:
            abort_websocket(websocket)


class PolicySpaceImpl(PolicyImplementationBase):
    """WebSocket bridge to a PolicySpace model server.

    Lifecycle:
      load(cfg) → connect to server, validate metadata
      bind(embodiment) → build obs encoder + action decoder from embodiment
      observe(obs) → optionally forward every frame to server (ingest_observation)
      encode_request(obs) → build standardized obs dict
      get_actions(request) → call server infer_actions, return raw response
      decode_response(raw) → convert policy actions to sim tensor
      initialize_episode() / finalize_episode() → manage episode state
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_dim: int = 0
        self._client: _PolicySpaceClient | None = None
        self._episode_step: int = 0
        self._episode_id: str = ""
        self._server_episode_active: bool = False
        self._server_instance_id: str = ""
        self._server_episode_instance_id: str = ""
        self._episode_steps: dict[int, int] = {}
        self._episode_ids: dict[int, str] = {}
        self._known_env_ids: set[int] = set()
        self._episode_state_lock = Lock()
        self._instruction: str | list[str] = ""
        self._camera_map: dict[str, str] = {}
        self._state_fields: list[str] = []
        self._action_type: str = "delta_joint"
        self._wire_protocol: str = "policy_space"
        self._image_height: int = 0
        self._image_width: int = 0
        self._log_timing: bool = False
        self._observation_buffered: bool = False
        self._ingest_observation_each_step: bool = True
        self._stateless_per_request: bool = False
        self._control_rate_hz: int = 0
        self._observation_rate_hz: int = 0
        self._observation_step: int = 0
        self._observe_requests: int = 0
        self._observe_seconds: float = 0.0
        self._successful_inference_log_at: dict[str, float] = {}
        self._model_io_log_lock = Lock()
        self._ee_transform: Any = None
        self._reconnect_attempts: int = 30
        self._initial_connect_attempts: int = 120
        self._reconnect_backoff_seconds: float = 2.0
        self._reconnect_backoff_max_seconds: float = 30.0
        self._reconnect_budget_seconds: float = 600.0
        self._initial_connect_budget_seconds: float = 2700.0
        self._ever_connected: bool = False
        self._gripper_convention: str = gripper_convention.RAW
        self._gripper_limits: tuple[float, float] | None = None
        self._gripper_wire_indices: list[int] | None = None
        self._gripper_state_indices: dict[str, int] = {}
        self._gripper_resolved: bool = False
        # ``AsyncChunkScheduler`` performs inference on its worker while an episode transition can
        # arrive from the simulation thread. ``websockets.sync`` permits concurrent sends, but
        # never two concurrent ``recv`` calls on one connection. Keep the complete connection,
        # lifecycle, request, and reconnect transaction single-owner so a late RTC refill cannot
        # interleave with a lifecycle operation and corrupt request/response pairing.
        self._request_lock = Lock()
        self._additional_headers: dict[str, str] = {}
        self._require_schema_negotiation: bool = False
        self._observation_schema: str = ""
        self._action_schema: str = ""
        self._embodiment_schema: str = ""

    def load(self, cfg: PolicySpaceImplCFG) -> None:
        super().load(cfg)
        self._endpoint = cfg.endpoint
        self._additional_headers = _runtime_additional_headers()
        self._policy_name = cfg.policy_name or ""
        self._timeout_seconds = cfg.timeout_ms / 1000.0
        self._max_message_size = max(1024, int(cfg.max_message_size))
        self._finalize_timeout_seconds = max(0.1, float(cfg.finalize_timeout_ms) / 1000.0)
        self._action_type = cfg.action_type or "delta_joint"
        self._wire_protocol = cfg.wire_protocol or "policy_space"
        self._require_public_endpoint = bool(cfg.require_public_endpoint)
        self._reconnect_budget_seconds = max(1.0, float(cfg.reconnect_budget_seconds))
        self._initial_connect_budget_seconds = max(1.0, float(cfg.initial_connect_budget_seconds))
        self._image_height = int(cfg.image_height or 0)
        self._image_width = int(cfg.image_width or 0)
        self._log_timing = bool(cfg.log_timing)
        self._ingest_observation_each_step = bool(cfg.ingest_observation_each_step)
        self._stateless_per_request = bool(cfg.stateless_per_request)
        self._control_rate_hz = int(cfg.control_rate_hz or 0)
        self._observation_rate_hz = int(cfg.observation_rate_hz or self._control_rate_hz)
        self._require_schema_negotiation = bool(cfg.require_schema_negotiation)
        self._observation_schema = str(cfg.observation_schema or "").strip()
        self._action_schema = str(cfg.action_schema or "").strip()
        self._embodiment_schema = str(cfg.embodiment_schema or "").strip()
        configured_schemas = [self._observation_schema, self._action_schema, self._embodiment_schema]
        if self._require_schema_negotiation and not all(configured_schemas):
            raise ValueError(
                "PolicySpaceImpl schema negotiation requires observation_schema, "
                "action_schema, and embodiment_schema"
            )
        if any(configured_schemas) and not all(configured_schemas):
            raise ValueError("PolicySpaceImpl schema fields must be configured together")
        if self._action_schema and self._action_schema not in _ACTION_SCHEMA_DIMS:
            raise ValueError(f"Unsupported Policy Space action schema: {self._action_schema}")
        if self._control_rate_hz < 0 or self._observation_rate_hz < 0:
            raise ValueError("PolicySpaceImpl control and observation rates must be non-negative")
        if self._observation_rate_hz and not self._control_rate_hz:
            raise ValueError("PolicySpaceImpl observation_rate_hz requires control_rate_hz")
        if self._observation_rate_hz > self._control_rate_hz:
            raise ValueError("PolicySpaceImpl observation_rate_hz cannot exceed control_rate_hz")
        self._observation_step = 0
        self._reconnect_attempts = max(1, int(cfg.reconnect_attempts))
        self._initial_connect_attempts = max(1, int(cfg.initial_connect_attempts))
        self._reconnect_backoff_seconds = max(0.0, float(cfg.reconnect_backoff_seconds))
        self._reconnect_backoff_max_seconds = max(
            self._reconnect_backoff_seconds, float(cfg.reconnect_backoff_max_seconds)
        )
        if self._wire_protocol not in {"policy_space", "openpi"}:
            raise ValueError(
                "PolicySpaceImpl wire_protocol must be 'policy_space' or 'openpi', " f"got {self._wire_protocol!r}"
            )
        if bool(self._image_height) != bool(self._image_width):
            raise ValueError("PolicySpaceImpl image_height and image_width must be configured together")
        self._instruction = cfg.instruction or ""
        manaprint(f"INFO: [PolicySpaceImpl] configured endpoint={cfg.endpoint} " f"action_type={self._action_type}")

    def bind(self, embodiment: PolicyEmbodimentCFG) -> None:
        super().bind(embodiment)
        self.action_dim = int(embodiment.action_dim)
        self._camera_map = {}
        self._state_sim_keys: list[str] = []
        self._state_field_map: dict[str, str] = {}
        for policy_name, sim_name in embodiment.obs_field_map.items():
            if _is_camera_field(policy_name) or _is_camera_field(sim_name):
                self._camera_map[sim_name] = policy_name
            else:
                self._state_sim_keys.append(sim_name)
                self._state_field_map[policy_name] = sim_name

        self._bind_gripper_layout(embodiment)

        if self._action_type in ("delta_ee", "absolute_ee"):
            self._build_ee_transform(embodiment)
        elif self._action_type == "ex001_wholebody_policy_wire":
            self._bind_ex001_wholebody_policy_wire(embodiment)

    def _bind_gripper_layout(self, embodiment: PolicyEmbodimentCFG) -> None:
        """Record the embodiment's gripper units and wire layout for conversion.

        Only the embodiment-owned half is known here. The policy's declared
        convention arrives on the server metadata handshake, so the two are
        joined later in :meth:`_resolve_gripper_convention`.
        """
        metadata = embodiment.metadata or {}
        self._gripper_limits = gripper_convention.limits_from_metadata(metadata)
        self._gripper_wire_indices = gripper_convention.read_wire_indices(metadata)
        self._gripper_state_indices = gripper_convention.read_state_indices(metadata)
        self._gripper_resolved = False

    def _resolve_gripper_convention(self, server_metadata: dict[str, Any]) -> None:
        """Join the policy's declared gripper convention with the sim's units.

        Runs once per connection, right after the metadata handshake. A policy
        that declares a normalized convention but cannot be converted must fail
        here rather than at the first action: a wrongly-scaled gripper command
        is silently clamped into a legal range by the action term, so there is
        no later point at which the mistake becomes visible.
        """
        convention = gripper_convention.normalize_convention(server_metadata.get(gripper_convention.CONVENTION_KEY))
        if self._action_type == "ex001_wholebody_policy_wire" and convention != gripper_convention.NORMALIZED_01:
            raise RuntimeError(
                "ex001_wholebody_policy_wire requires server metadata "
                f"{gripper_convention.CONVENTION_KEY}={gripper_convention.NORMALIZED_01!r}"
            )
        self._gripper_convention = convention
        self._gripper_resolved = True
        if convention == gripper_convention.RAW:
            return

        policy = self._policy_name or server_metadata.get("policy_name") or "<unknown>"
        if self._gripper_limits is None:
            raise RuntimeError(
                f"Policy '{policy}' declares {gripper_convention.CONVENTION_KEY}={convention!r}, but the bound "
                "embodiment reports no gripper range. Implement `get_gripper_states() -> {'open':..,'closed':..}` "
                "on the embodiment (see manaenv/assets/robots/ex001_6r.py) so its native units are known."
            )
        if not self._gripper_wire_indices:
            raise RuntimeError(
                f"Policy '{policy}' declares {gripper_convention.CONVENTION_KEY}={convention!r}, but the "
                f"policy-embodiment yaml declares no `{gripper_convention.WIRE_INDICES_KEY}`. Add the gripper "
                "column indices of the action wire under the matching action_mode in "
                "manaenv/defination/embodiments/policy_space/<Embodiment>.yaml."
            )
        closed, open_ = self._gripper_limits
        manaprint(
            f"INFO: [PolicySpaceImpl] gripper convention={convention} "
            f"sim_range=[{closed:g}, {open_:g}] wire_idx={list(self._gripper_wire_indices)} "
            f"state_idx={self._gripper_state_indices or '{}'}"
        )

    def _gripper_conversion_active(self) -> bool:
        return (
            self._gripper_convention == gripper_convention.NORMALIZED_01
            and self._gripper_limits is not None
            and bool(self._gripper_wire_indices)
        )

    def _build_ee_transform(self, embodiment: PolicyEmbodimentCFG) -> None:
        from manaenv.policy.action_transforms import EEAbsoluteIKTransform

        action_config = {
            "response_keys": [{"name": "follow1_pos"}, {"name": "follow2_pos"}],
            "slot_layout": {k: dict(v) for k, v in embodiment.slot_layout.items()},
            "ee_frames": dict(embodiment.ee_frames),
            "wire_pose_reference": embodiment.wire_pose_reference,
            "trajectory_semantics": {"first_frame": "prediction_only"},
            "robot_entity": "robot",
        }
        cfg = self.cfg
        horizon = int(getattr(cfg, "action_horizon", 24))
        self._ee_transform = EEAbsoluteIKTransform(
            action_horizon=horizon,
            action_dim=int(embodiment.action_dim),
            action_config=action_config,
        )

    def _bind_ex001_wholebody_policy_wire(self, embodiment: PolicyEmbodimentCFG) -> None:
        """Validate the fixed EX001 whole-body policy wire before connection.

        This action type is intentionally model-neutral.  It accepts only the
        public versioned policy wire and maps it to the existing
        EX001 21-D action layout; it does not import OpenPI or any model code.
        """
        from manaenv.policy.contracts import ex001_wholebody_policy_wire as wholebody_wire

        if int(embodiment.action_dim) != wholebody_wire.ENV_ACTION_DIM:
            raise ValueError(
                "ex001_wholebody_policy_wire requires the EX001 whole-body "
                f"{wholebody_wire.ENV_ACTION_DIM}D env action, got {embodiment.action_dim}"
            )
        metadata = embodiment.metadata or {}
        if metadata.get("action_wire_schema") != wholebody_wire.WIRE_SCHEMA:
            raise ValueError(
                "ex001_wholebody_policy_wire requires embodiment metadata "
                f"action_wire_schema={wholebody_wire.WIRE_SCHEMA!r}"
            )
        indices = gripper_convention.read_wire_indices(metadata)
        if list(indices or []) != [6, 13]:
            raise ValueError(
                "ex001_wholebody_policy_wire requires gripper_wire_indices=[6, 13] on its normalized 20D wire"
            )

    def observe(self, observation: dict[str, Any], env_idx: int | None = None) -> None:
        """Forward control frames only for models that consume observation history."""
        if self._wire_protocol == "openpi" or not self._ingest_observation_each_step:
            # OpenPI's websocket server performs inference for every message;
            # it has no separate observation-ingestion operation. Some Policy
            # Space models (for example WallX) are also single-frame:
            # infer_actions already carries the current observation, so sending
            # full camera payloads on every executed action would be redundant.
            return
        step = self._observation_step
        self._observation_step += 1
        if self._observation_rate_hz < self._control_rate_hz and step > 0:
            current_native_step = step * self._observation_rate_hz // self._control_rate_hz
            previous_native_step = (step - 1) * self._observation_rate_hz // self._control_rate_hz
            if current_native_step == previous_native_step:
                return
        obs = self._build_obs(observation, env_idx)
        started = time.perf_counter()
        try:
            self._request_with_reconnect("ingest_observation", obs)
            self._observation_buffered = True
            self._observe_requests += 1
            self._observe_seconds += time.perf_counter() - started
        except Exception as exc:
            manaprint(f"WARNING: [PolicySpaceImpl] ingest_observation failed: {exc}")
            _emit_model_io("ingest_observation_error", error=f"{type(exc).__name__}: {exc}"[:512])
            raise

    def finalize_episode(
        self,
        env_ids: Any = None,
        outcomes: dict[int, bool] | None = None,
    ) -> None:
        """Best-effort completion of the server-side episode.

        Evaluation results are already owned by the simulator. A failed final
        notification therefore closes the stale socket and lets the mandatory
        initialize operation reconnect, without changing the recorded result.
        """
        ids = normalize_env_ids(env_ids)
        with self._episode_state_lock:
            known_before = set(self._known_env_ids)
            target_ids = set(known_before) if ids is None else set(ids) & known_before
            if not target_ids and not (ids is None and self._server_episode_active):
                return
            finalizes_all_known = ids is None or target_ids == known_before
            server_episode_id = self._episode_id
            final_step = self._episode_step
            per_env_steps = {str(env_idx): self._episode_steps.get(env_idx, 0) for env_idx in sorted(target_ids)}

        if self._wire_protocol == "openpi":
            return
        if not self._ingest_observation_each_step and not finalizes_all_known:
            return
        # The simulator has ended this episode. A replacement service instance
        # may therefore initialize the next episode even when the best-effort
        # final notification cannot be delivered.
        self._server_episode_active = False
        self._server_episode_instance_id = ""
        self._observation_buffered = False
        client = self._client
        if client is None:
            return

        payload: dict[str, Any] = {
            "episode_id": server_episode_id,
            "final_step": final_step,
            "env_steps": per_env_steps,
        }
        if outcomes:
            payload["outcomes"] = {
                str(env_idx): bool(outcomes[env_idx]) for env_idx in sorted(target_ids) if env_idx in outcomes
            }
        try:
            with self._request_lock:
                client.request(
                    "finalize_episode",
                    payload,
                    timeout_seconds=self._finalize_timeout_seconds,
                )
            _emit_model_io("finalize_episode", episode_id=server_episode_id, status="ok")
        except Exception as exc:
            self._abort_client()
            manaprint(f"WARNING: [PolicySpaceImpl] finalize_episode failed: {exc}")
            _emit_model_io(
                "finalize_episode",
                episode_id=server_episode_id,
                status="error",
                error=f"{type(exc).__name__}: {exc}"[:512],
            )

    def initialize_episode(self, env_ids: Any = None) -> None:
        self._observation_step = 0
        ids = normalize_env_ids(env_ids)
        with self._episode_state_lock:
            known_before = set(self._known_env_ids)
            episode_env_ids = set(known_before) if ids is None else set(ids)
            self._known_env_ids.update(episode_env_ids)
            initializes_all_known = ids is None or (bool(episode_env_ids) and episode_env_ids == self._known_env_ids)
            first_episode = not known_before
            starts_server_episode = first_episode or initializes_all_known or self._ingest_observation_each_step

            if starts_server_episode:
                self._episode_step = 0
                self._episode_id = uuid.uuid4().hex[:12]
            for index, env_idx in enumerate(sorted(episode_env_ids)):
                episode_id = self._episode_id if starts_server_episode and index == 0 else uuid.uuid4().hex[:12]
                self._episode_steps[env_idx] = 0
                self._episode_ids[env_idx] = episode_id

            server_episode_id = self._episode_id
        self._observe_requests = 0
        self._observation_buffered = False
        self._observe_seconds = 0.0
        if self._wire_protocol == "openpi":
            # The A800 Cosmos3 OpenPI server is stateless per request.
            return
        if not self._ingest_observation_each_step and not (first_episode or initializes_all_known):
            # Single-frame services such as WallX expose one process-global
            # episode lifecycle. A partial vector-environment transition must
            # not clear the model while sibling environments are still running;
            # RTC episode state is client-side.
            return
        try:
            initialize_payload = {
                "episode_id": server_episode_id,
                "instruction": self.resolve_instruction(),
                **self._schema_selection_payload(),
            }
            self._request_with_reconnect(
                "initialize_episode",
                initialize_payload,
            )
            self._server_episode_active = True
            _emit_model_io("initialize_episode", episode_id=server_episode_id, status="ok")
        except Exception as exc:
            manaprint(f"WARNING: [PolicySpaceImpl] initialize_episode failed: {exc}")
            _emit_model_io("initialize_episode", episode_id=server_episode_id, status="error", error=str(exc)[:512])
            raise

    def encode_request(self, observation: dict[str, Any], env_idx: int | None = None) -> dict[str, Any]:
        request = {"obs": self._build_obs(observation, env_idx)}
        if env_idx is not None:
            request[_INTERNAL_ENV_IDX_KEY] = int(env_idx)
        return request

    def get_actions(self, request: dict[str, Any]) -> dict[str, Any]:
        request = dict(request)
        raw_env_idx = request.pop(_INTERNAL_ENV_IDX_KEY, None)
        env_idx = None if raw_env_idx is None else int(raw_env_idx)
        episode_id, episode_step = self._episode_state(env_idx)
        obs = request.get("obs", {})
        obs["step"] = episode_step
        obs["episode_id"] = episode_id
        rtc_enabled = bool(obs.get("use_rtc"))
        rtc_prefix = obs.get("prev_action_chunk")
        rtc_prefix_shape = list(np.asarray(rtc_prefix).shape) if rtc_prefix is not None else None
        rtc_delay = obs.get("inference_delay") if rtc_enabled else None
        if self._log_timing:
            _emit_model_io(
                "observation_refill",
                policy_name=self._policy_name,
                episode_id=episode_id,
                step=episode_step,
                history_updates=self._observe_requests,
                instruction=str(obs.get("instruction") or "")[:240],
                images=_summarize_images(obs.get("images", {})),
                rtc_enabled=rtc_enabled,
                rtc_delay=rtc_delay,
                rtc_prefix_shape=rtc_prefix_shape,
            )
        started = time.perf_counter()
        try:
            if self._wire_protocol == "openpi":
                result = self._request_with_reconnect("", self._build_openpi_obs(obs))
                if isinstance(result, dict) and "action" in result:
                    result["actions"] = result["action"]
            else:
                result = self._request_with_reconnect("infer_actions", obs)
        except Exception as exc:
            _emit_model_io(
                "infer_actions",
                step=episode_step,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                status="error",
                error=f"{type(exc).__name__}: {exc}"[:512],
            )
            raise
        wholebody_wire = self._action_type == "ex001_wholebody_policy_wire"
        expected_key = "action_chunk" if wholebody_wire else "actions"
        if not isinstance(result, dict) or expected_key not in result:
            raise RuntimeError(f"PolicySpace server response missing {expected_key!r}; got {type(result)}")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        now = time.monotonic()
        with self._model_io_log_lock:
            previous_log_at = self._successful_inference_log_at.get(episode_id)
            emit_success_log = (
                self._log_timing or previous_log_at is None or now - previous_log_at >= _MODEL_IO_HEARTBEAT_SECONDS
            )
            if emit_success_log:
                self._successful_inference_log_at[episode_id] = now
        if emit_success_log:
            fields: dict[str, Any] = {
                "episode_id": episode_id,
                "step": episode_step,
                "latency_ms": round(elapsed_ms, 2),
                "status": "ok",
            }
            if self._log_timing:
                fields.update(
                    {
                        "policy_name": str(getattr(self.cfg, "policy_name", "") or ""),
                        "action_type": str(getattr(self.cfg, "action_type", "") or ""),
                        "action_shape": list(np.asarray(result[expected_key]).shape),
                        "rtc_enabled": rtc_enabled,
                        "rtc_delay": rtc_delay,
                    }
                )
            _emit_model_io("infer_actions", **fields)
        self._observe_requests = 0
        self._observe_seconds = 0.0
        self._advance_episode_step(env_idx, episode_id, 1)
        result[_INTERNAL_EPISODE_CONTEXT_KEY] = (env_idx, episode_id)
        return result

    def _episode_state(self, env_idx: int | None) -> tuple[str, int]:
        with self._episode_state_lock:
            if env_idx is None:
                return self._episode_id, self._episode_step
            env_idx = int(env_idx)
            self._known_env_ids.add(env_idx)
            self._episode_ids.setdefault(env_idx, self._episode_id)
            self._episode_steps.setdefault(env_idx, self._episode_step)
            return self._episode_ids[env_idx], self._episode_steps[env_idx]

    def _advance_episode_step(self, env_idx: int | None, episode_id: str, amount: int) -> None:
        if amount == 0:
            return
        with self._episode_state_lock:
            if env_idx is None:
                if self._episode_id == episode_id:
                    self._episode_step += int(amount)
                return
            env_idx = int(env_idx)
            if self._episode_ids.get(env_idx) == episode_id:
                self._episode_steps[env_idx] = self._episode_steps.get(env_idx, 0) + int(amount)

    @staticmethod
    def _build_openpi_obs(obs: dict[str, Any]) -> dict[str, Any]:
        """Translate the standardized embodiment payload to Cosmos3 OpenPI keys."""
        images = obs.get("images", {})
        state = obs.get("state", {})
        required_images = {
            "wrist": "observation/wrist_image_left",
            "exterior_1": "observation/exterior_image_1_left",
            "exterior_2": "observation/exterior_image_2_left",
        }
        missing = [name for name in required_images if name not in images]
        if missing:
            raise RuntimeError("Cosmos3 OpenPI request is missing DROID camera observations: " + ", ".join(missing))
        for name in ("joint_position", "gripper_position"):
            if name not in state:
                raise RuntimeError(f"Cosmos3 OpenPI request is missing state field: {name}")

        payload = {
            "prompt": str(obs.get("instruction") or ""),
            "observation/joint_position": np.asarray(state["joint_position"], dtype=np.float32),
            "observation/gripper_position": np.asarray(state["gripper_position"], dtype=np.float32),
        }
        payload.update({target: images[source] for source, target in required_images.items()})
        return payload

    def decode_response(
        self,
        raw: dict[str, Any],
        env: Any = None,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        context = raw.pop(_INTERNAL_EPISODE_CONTEXT_KEY, None)
        response_episode_id = None
        if isinstance(context, (tuple, list)) and len(context) == 2:
            context_env_idx, response_episode_id = context
            if env_idx is None and context_env_idx is not None:
                env_idx = int(context_env_idx)
        if self._action_type == "ex001_wholebody_policy_wire":
            return self._decode_ex001_wholebody_policy_wire(
                raw,
                env=env,
                env_idx=env_idx,
                response_episode_id=response_episode_id,
            )

        actions = raw.get("actions")
        if actions is None:
            return None
        arr = np.asarray(actions)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] < 1:
            raise RuntimeError(f"PolicySpace actions must have non-empty shape [T,D], got {arr.shape}")
        if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
            raise RuntimeError(f"PolicySpace actions must be real numeric values, got {arr.dtype}")
        if not np.isfinite(arr).all():
            raise RuntimeError("PolicySpace actions contain NaN/Inf")
        expected_wire_dim = _ACTION_SCHEMA_DIMS.get(self._action_schema)
        if expected_wire_dim is not None and arr.shape[1] != expected_wire_dim:
            raise RuntimeError(
                f"PolicySpace action schema {self._action_schema} requires width "
                f"{expected_wire_dim}, got {arr.shape[1]}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            arr = np.asarray(arr, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise RuntimeError("PolicySpace actions exceed the finite float32 range")

        # Convert before decoding: EEAbsoluteIKTransform copies the wire gripper
        # column into the sim action verbatim, and the joint path passes the whole
        # row through, so this is the last point where the gripper channel is still
        # identifiable by index.
        if self._gripper_conversion_active():
            assert self._gripper_limits is not None
            arr = gripper_convention.convert_wire_actions(
                arr,
                limits=self._gripper_limits,
                indices=self._gripper_wire_indices or [],
            )

        if self._ee_transform is not None:
            return self._decode_ee(arr, env, env_idx)

        chunk = torch.as_tensor(arr, dtype=torch.float32)
        device = getattr(env, "device", "cpu") if env else "cpu"
        if response_episode_id is None:
            response_episode_id, _ = self._episode_state(env_idx)
        self._advance_episode_step(env_idx, response_episode_id, int(chunk.shape[0]) - 1)
        return chunk.to(device=device)

    @staticmethod
    def _read_mobile_home_ee_reference(
        env: Any,
        *,
        attr_name: str,
        env_idx: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read one env's reset-captured command-frame home EE pose.

        EX001 whole-body stores this pose in ``lift_link`` coordinates. The
        versioned wire is home-relative, so composing with this per-env value
        produces the absolute target in the same frame consumed by the IK
        action. Keep environment access at the simulator boundary and pass
        only NumPy data into the Isaac-free action contract.
        """
        if env is None:
            raise RuntimeError("ex001_wholebody_policy_wire requires a live sim env to resolve init-EE references")
        reference = getattr(env, attr_name, None)
        if not isinstance(reference, tuple) or len(reference) != 2:
            raise RuntimeError(
                f"ex001_wholebody_policy_wire requires reset-captured env.{attr_name}; "
                "ensure the embodiment init-EE observation terms are installed"
            )
        index = 0 if env_idx is None else int(env_idx)

        def _as_numpy(value: Any) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value[index], dtype=np.float32)

        return _as_numpy(reference[0]), _as_numpy(reference[1])

    def _decode_ex001_wholebody_policy_wire(
        self,
        raw: dict[str, Any],
        *,
        env: Any = None,
        env_idx: int | None = None,
        response_episode_id: str | None = None,
    ) -> torch.Tensor:
        """Decode a model-neutral policy-wire response into the EX001 21-D action.

        Gripper conversion deliberately happens first while columns 6/13 are
        still identifiable on the policy wire. ``policy_wire_to_env21`` then
        performs the sole home-EE composition, Euler-to-XYZW and odom-to-wheel
        conversion.
        """
        from manaenv.policy.contracts import ex001_wholebody_policy_wire as wholebody_wire

        if not isinstance(raw, dict):
            raise TypeError("ex001_wholebody_policy_wire response must be a mapping")
        arr = wholebody_wire.validate_wire_envelope(raw)

        if self._gripper_conversion_active():
            assert self._gripper_limits is not None
            arr = gripper_convention.convert_wire_actions(
                arr,
                limits=self._gripper_limits,
                indices=self._gripper_wire_indices or [],
            )

        left_home_pos, left_home_quat = self._read_mobile_home_ee_reference(
            env, attr_name="_left_ee_ref", env_idx=env_idx
        )
        right_home_pos, right_home_quat = self._read_mobile_home_ee_reference(
            env, attr_name="_right_ee_ref", env_idx=env_idx
        )
        decoded = np.stack(
            [
                wholebody_wire.policy_wire_to_env21(
                    row,
                    left_init_ee_pos=left_home_pos,
                    left_init_ee_quat_xyzw=left_home_quat,
                    right_init_ee_pos=right_home_pos,
                    right_init_ee_quat_xyzw=right_home_quat,
                )
                for row in arr
            ],
            axis=0,
        )
        device = getattr(env, "device", "cpu") if env is not None else "cpu"
        if response_episode_id is None:
            response_episode_id, _ = self._episode_state(env_idx)
        self._advance_episode_step(env_idx, response_episode_id, int(decoded.shape[0]) - 1)
        return torch.as_tensor(decoded, dtype=torch.float32, device=device)

    def _decode_ee(self, arr: np.ndarray, env: Any, env_idx: int | None) -> torch.Tensor:
        if env is None:
            raise RuntimeError("PolicySpaceImpl: delta_ee/absolute_ee requires live sim env")
        follow1 = arr[:, 0:7]
        follow2 = arr[:, 7:14]
        response = {"follow1_pos": follow1, "follow2_pos": follow2}
        device = getattr(env, "device", "cpu")
        chunk = self._ee_transform(response, env=env, device=device, env_idx=env_idx)
        if chunk is None:
            raise RuntimeError("EEAbsoluteIKTransform returned None")
        return chunk

    def cleanup(self) -> None:
        if self._server_episode_active:
            self.finalize_episode()
        client, self._client = self._client, None
        if client is not None:
            client.close()

    def _abort_client(self) -> None:
        self._observation_buffered = False
        client, self._client = self._client, None
        if client is not None:
            client.abort()

    def _request_with_reconnect(self, endpoint: str, payload: dict[str, Any]) -> Any:
        with self._request_lock:
            return self._request_with_reconnect_locked(endpoint, payload)

    def _request_with_reconnect_locked(self, endpoint: str, payload: dict[str, Any]) -> Any:
        """Run a request, reconnecting with backoff for as long as the budget allows.

        A remote policy server can drop mid-episode (restart, OOM kill, network blip). Both
        failure modes must be survived, and they surface differently: the in-flight request
        raises a ``ConnectionClosed``, while a server that is down raises from
        ``_ensure_client``. Retrying only the first left an evaluation dying on the second.

        Each attempt re-establishes the socket and replays ``initialize_episode`` so the server's episode
        state matches the sim's before the request is retried. The episode itself continues —
        the sim is not stepped while we wait, so the policy resumes from the same observation.
        """

        last_error: Exception | None = None
        delay = self._reconnect_backoff_seconds
        initial_connection = not self._ever_connected
        reconnect_budget_seconds = (
            self._initial_connect_budget_seconds if initial_connection else self._reconnect_budget_seconds
        )
        reconnect_attempts = self._initial_connect_attempts if initial_connection else self._reconnect_attempts
        deadline = time.monotonic() + reconnect_budget_seconds
        # The server-side lifecycle belongs to this PolicySpaceImpl connection,
        # not to one vector-environment item carried in an inference payload.
        # Reclaim the connection episode after a socket drop so one client can
        # continue serving several independently resetting Isaac environments.
        request_episode_id = self._episode_id
        wire_payload = dict(payload)
        wire_payload["request_id"] = uuid.uuid4().hex
        attempts_used = 0

        def operation_timeout() -> float:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"PolicySpace reconnect budget of {reconnect_budget_seconds:g} seconds was exhausted"
                )
            return min(self._timeout_seconds, remaining)

        for attempt in range(1, reconnect_attempts + 1):
            attempts_used = attempt
            try:
                reconnected = self._client is None and attempt > 1
                client = self._ensure_client(timeout_seconds=operation_timeout())
                # Claim the same server-side episode after a fresh socket. A
                # same-process service resumes it without clearing history.
                if (
                    reconnected
                    and self._wire_protocol == "policy_space"
                    and endpoint != "initialize_episode"
                    and request_episode_id
                ):
                    initialize_payload: dict[str, Any] = {
                        "episode_id": request_episode_id,
                        "instruction": self.resolve_instruction(),
                        **self._schema_selection_payload(),
                    }
                    server_metadata = client._server_metadata
                    if (
                        server_metadata.get("session_mode") == SESSION_MODE_MULTIPLEXED_SERIAL
                        and self._server_episode_instance_id
                        and server_metadata.get("server_instance_id") == self._server_episode_instance_id
                    ):
                        initialize_payload["resume_only"] = True
                    client.request(
                        "initialize_episode",
                        initialize_payload,
                        timeout_seconds=operation_timeout(),
                    )
                    self._server_episode_instance_id = str(server_metadata["server_instance_id"])
                request_payload = wire_payload
                if (
                    endpoint == "infer_actions"
                    and self._wire_protocol == "policy_space"
                    and self._ingest_observation_each_step
                    and not self._stateless_per_request
                    and self._observation_buffered
                ):
                    request_payload = {key: value for key, value in wire_payload.items() if key != "images"}
                result = client.request(endpoint, request_payload, timeout_seconds=operation_timeout())
                if endpoint == "initialize_episode":
                    self._server_episode_instance_id = str(client._server_metadata["server_instance_id"])
            except Exception as exc:
                if not _is_retryable_transport_error(exc):
                    raise
                last_error = exc
                self._abort_client()
                if attempt >= reconnect_attempts or time.monotonic() >= deadline:
                    break
                manaprint(
                    f"WARNING: [PolicySpaceImpl] connection lost on {endpoint or 'inference'} "
                    f"(attempt {attempt}/{reconnect_attempts}); "
                    f"retrying in {delay:.1f}s: {exc}"
                )
                _emit_model_io(
                    "reconnect",
                    endpoint=endpoint or "inference",
                    status="retrying",
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}"[:512],
                )
                if delay > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(delay, remaining))
                delay = min(delay * 2, self._reconnect_backoff_max_seconds)
                continue
            if attempt > 1:
                manaprint(f"INFO: [PolicySpaceImpl] reconnected after {attempt} attempt(s)")
                _emit_model_io("reconnect", endpoint=endpoint or "inference", status="ok", attempt=attempt)
            return result

        _emit_model_io(
            "reconnect",
            status="exhausted",
            phase="initial_connect" if initial_connection else "reconnect",
            endpoint=endpoint or "inference",
            attempts=attempts_used,
            attempt_limit=reconnect_attempts,
            budget_seconds=reconnect_budget_seconds,
            error=f"{type(last_error).__name__}: {last_error}"[:512] if last_error is not None else "unknown",
        )
        raise RuntimeError(
            f"PolicySpace at {self._endpoint} stayed unreachable across "
            f"{attempts_used}/{reconnect_attempts} reconnect attempts: {last_error}"
        ) from last_error

    def _ensure_client(self, *, timeout_seconds: float | None = None) -> _PolicySpaceClient:
        if self._client is not None:
            return self._client
        try:
            self._client = _PolicySpaceClient(
                self._endpoint,
                self._timeout_seconds if timeout_seconds is None else timeout_seconds,
                max_message_size=self._max_message_size,
                require_public_endpoint=self._require_public_endpoint,
                additional_headers=self._additional_headers,
            )
        except Exception as exc:
            if _is_retryable_transport_error(exc):
                raise _TransientPolicySpaceConnectionError(
                    f"Could not connect to PolicySpace at {self._endpoint}: {exc}"
                ) from exc
            # URI/DNS policy rejection, malformed metadata and protocol/application errors are
            # deterministic for this immutable Factory job.  Preserve their type and fail now.
            raise
        server_metadata = self._client._server_metadata
        if self._wire_protocol == "policy_space":
            try:
                self._validate_server_metadata(server_metadata)
                self._accept_server_instance(server_metadata)
            except Exception:
                self._abort_client()
                raise
        if not self._gripper_resolved:
            self._resolve_gripper_convention(server_metadata)
        server_policy = server_metadata.get("policy_name", "")
        if self._policy_name and server_policy and server_policy != self._policy_name:
            manaprint(
                f"WARNING: [PolicySpaceImpl] policy_name mismatch: "
                f"yaml declares '{self._policy_name}', server reports '{server_policy}'"
            )
        effective_name = server_policy or self._policy_name or "unknown"
        manaprint(f"INFO: [PolicySpaceImpl] connected to {self._endpoint} " f"(policy={effective_name})")
        self._ever_connected = True
        return self._client

    def _validate_server_metadata(self, server_metadata: dict[str, Any]) -> None:
        admission_error = str(server_metadata.get("error") or "").strip()
        if admission_error:
            message = f"Policy Space connection admission failed: {admission_error}"
            if server_metadata.get("retryable") is True:
                raise _TransientPolicySpaceConnectionError(message)
            raise RuntimeError(message)
        if metadata_mismatches(server_metadata):
            raise RuntimeError(
                "Policy Space protocol v2 is required; restart the policy service "
                "with initialize_episode, ingest_observation, infer_actions, and finalize_episode support"
            )
        if self._require_schema_negotiation:
            policy_pairs = schema_pairs_from_metadata(server_metadata)
            bridge_pair = SchemaPair(
                self._observation_schema,
                self._action_schema,
                (self._embodiment_schema,),
            )
            select_schema_pair(
                policy_pairs,
                [bridge_pair],
                embodiment=self._embodiment_schema,
            )

    def _schema_selection_payload(self) -> dict[str, str]:
        if not self._observation_schema:
            return {}
        return {
            "observation_schema": self._observation_schema,
            "action_schema": self._action_schema,
            "embodiment_schema": self._embodiment_schema,
        }

    def _accept_server_instance(self, server_metadata: dict[str, Any]) -> None:
        """Accept a connection only when episode state can be resumed safely."""
        instance_id = str(server_metadata["server_instance_id"])
        previous = self._server_instance_id
        if (
            previous
            and instance_id != previous
            and self._server_episode_active
            and self._ingest_observation_each_step
            and not self._stateless_per_request
        ):
            raise RuntimeError(
                "Policy Space service restarted during an active episode; "
                "the policy history cannot be resumed safely"
            )
        self._server_instance_id = instance_id

    def requires_serial_envs(self) -> bool:
        """Return whether one server-side history is shared by the rollout."""
        return (
            self._wire_protocol == "policy_space"
            and self._ingest_observation_each_step
            and not self._stateless_per_request
        )

    def _ensure_gripper_resolved_for_state(self) -> None:
        """Guarantee the convention is known before any state leaves for the model.

        The convention rides the metadata handshake, so it is only known once a
        client exists. ``initialize_episode`` normally establishes that first, but an
        observation can legitimately be built before any request has been sent
        (``ingest_observation_each_step: false`` policies, or the ``openpi`` path that
        skips episode initialization). Connecting here keeps the first frame from being sent
        in the wrong units — silently, and for the whole episode.

        Only forced when the embodiment actually declares a gripper slot in its
        state, so policies with no gripper state keep their existing timing.
        """
        if self._gripper_resolved or not self._gripper_state_indices:
            return
        self._ensure_client()

    def _build_obs(self, observation: dict[str, Any], env_idx: int | None) -> dict[str, Any]:
        from manaenv.utils.tensor_utils import select_env_item, to_numpy_keep_shape

        self._ensure_gripper_resolved_for_state()

        images: dict[str, Any] = {}
        camera_obs = observation.get("camera_obs", {})
        if not isinstance(camera_obs, Mapping):
            raise RuntimeError("PolicySpace observation camera_obs must be a mapping")
        invalid_cameras: list[str] = []
        for sim_name, policy_name in self._camera_map.items():
            raw = camera_obs.get(sim_name)
            if raw is None:
                invalid_cameras.append(f"{sim_name} (missing)")
                continue
            rgb = _to_rgb_uint8(_select_camera_env_item(raw, env_idx))
            if rgb is None:
                invalid_cameras.append(f"{sim_name} (invalid RGB image)")
                continue
            images[policy_name] = _resize_rgb(rgb, self._image_height, self._image_width)
        if invalid_cameras:
            raise RuntimeError(
                "PolicySpace observation requires every mapped camera in the same control step; "
                f"unavailable cameras: {', '.join(invalid_cameras)}"
            )

        state: dict[str, Any] = {}
        policy_obs = observation.get("policy", {}) or {}
        for policy_name, sim_name in self._state_field_map.items():
            val = policy_obs.get(sim_name)
            if val is None:
                continue
            item = select_env_item(val, env_idx)
            arr = to_numpy_keep_shape(item)
            if arr is None:
                continue
            flat = np.asarray(arr, dtype=np.float32).flatten()
            # The simulator reports the gripper joint in native units, but a
            # normalized-convention model has only ever seen [0, 1] state. Skipping
            # this direction feeds it out-of-distribution input, which openloop
            # replay cannot detect (there the state comes from the dataset) and which
            # surfaces only as a lower closed-loop success rate.
            grip_idx = self._gripper_state_indices.get(policy_name)
            if grip_idx is not None and self._gripper_conversion_active():
                assert self._gripper_limits is not None
                flat = gripper_convention.convert_state_field(
                    flat,
                    limits=self._gripper_limits,
                    index=grip_idx,
                )
            state[policy_name] = flat

        episode_id, episode_step = self._episode_state(env_idx)
        return {
            "images": images,
            "state": state,
            "instruction": self.resolve_instruction(env_idx),
            "step": episode_step,
            "episode_id": episode_id,
        }


@dataclass
class PolicySpaceImplCFG(PolicyImplementationCFG):
    """CFG for PolicySpaceImpl."""

    YAML_DISPATCH_KEY: ClassVar[str] = "policy_space"

    name: str = ""
    policy_name: str = ""
    endpoint: str = "ws://127.0.0.1:8001"
    timeout_ms: int = 120000
    max_message_size: int = 64 * 1024 * 1024
    finalize_timeout_ms: int = 2000
    action_type: str = "delta_ee"
    action_horizon: int = 24
    instruction: str = ""
    wire_protocol: str = "policy_space"
    # Public challenge / Factory workloads set this through a non-overridable catalog binding.
    # Local development keeps the historical private-endpoint behavior unless explicitly enabled.
    require_public_endpoint: bool = False
    # One request may reconnect several times, but the immutable Factory task must have a wall-clock
    # ceiling as well as an attempt count. This includes DNS, connection, episode initialization
    # replay, and inference.
    reconnect_budget_seconds: float = 600.0
    # A participant's Policy Space service may still be cold-starting when the
    # Factory simulator first reaches initialize_episode. Give that first
    # admission a longer budget without weakening the bounded recovery window
    # after a connection has already succeeded.
    initial_connect_budget_seconds: float = 2700.0
    initial_connect_attempts: int = 120
    image_height: int = 0
    image_width: int = 0
    # Diagnostic mode restores per-refill fingerprints and per-call inference
    # details. Formal benchmark configs keep it disabled.
    log_timing: bool = False
    ingest_observation_each_step: bool = True
    # The simulator may execute an interpolated action chunk faster than the
    # model's native observation stream. When both rates are set, observe()
    # deterministically forwards only the native-rate frames.
    control_rate_hz: int = 0
    observation_rate_hz: int = 0
    # Pi05/OpenPI inference is stateless per request. This relaxes history and
    # restart constraints, while Policy Space v2 still uses its episode lifecycle.
    stateless_per_request: bool = False
    require_schema_negotiation: bool = False
    observation_schema: str = ""
    action_schema: str = ""
    embodiment_schema: str = ""
    # A remote policy server restarting mid-evaluation used to abort the whole run: one
    # reconnect attempt, and if the socket was not back yet the RuntimeError propagated out
    # of the rollout, discarding every task still to run. Keep retrying instead — a long
    # benchmark is worth more than a fast failure, and a server that never returns still
    # ends the run once the budget is spent.
    reconnect_attempts: int = 30
    reconnect_backoff_seconds: float = 2.0
    reconnect_backoff_max_seconds: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)
    class_type: type = PolicySpaceImpl
