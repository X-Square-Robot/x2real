"""Persist wire-protocol images from remote policy requests (debug / audit).

Two usage patterns:

1. **Wire interceptor** (recommended): declare under ``client.wire_interceptors``
   in the policy YAML.  The translator resolves ``save_request_images`` from
   ``manaenv.policy.wire_interceptors`` which delegates here.

2. **Programmatic**: create a :class:`RequestImageSink` with a
   :class:`RequestImageSinkConfig`, call :meth:`configure` then
   :meth:`save_wire_images` per inference step.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from manaenv.utils.debug_print import manaprint
from manaenv.utils.tensor_utils import to_numpy_keep_shape

if TYPE_CHECKING:
    from manaenv.policy.observation_codec import ObservationCodec

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


def _sanitize_filename_part(name: str, max_len: int = 120) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name)).strip("._-")
    return (s[:max_len] if s else "view") or "view"


def _iter_wire_image_items_by_key(model_input: dict[str, Any], image_key: str) -> list[tuple[str, Any]]:
    """Extract ``(view_name, payload)`` pairs from the wire request."""
    views = model_input.get(image_key)
    if isinstance(views, dict):
        return list(views.items())
    return []


def _iter_wire_image_items(model_input: dict[str, Any], codec: ObservationCodec) -> list[tuple[str, Any]]:
    return _iter_wire_image_items_by_key(model_input, codec.image_out_key)


def _payload_to_bgr_uint8(payload: Any) -> np.ndarray | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        try:
            raw = base64.b64decode(payload, validate=False)
        except Exception:
            return None
        if cv2 is None:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return bgr

    payload = to_numpy_keep_shape(payload)
    if not isinstance(payload, np.ndarray):
        return None
    rgb = np.asarray(payload)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.ndim not in (2, 3):
        return None
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0).clip(0, 255)
        rgb = rgb.astype(np.uint8)
    if cv2 is None:
        return None
    if rgb.ndim == 3 and rgb.shape[2] == 3:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb_hwc = np.transpose(rgb, (1, 2, 0))
        return cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
    if rgb.ndim == 2:
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return None


@dataclass
class RequestImageSinkConfig:
    enabled: bool = False
    output_dir: str = "generated_data/policy_request_images"
    session_name: str = ""
    every_n: int = 1
    image_key: str = "views"


class RequestImageSink:
    """Write JPEGs for each view in the wire payload (before msgpack send)."""

    def __init__(self, config: RequestImageSinkConfig) -> None:
        self._config = config
        self._session_dir: Path | None = None
        self._warned_no_codec = False
        self._query_counter: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    @property
    def session_path(self) -> Path | None:
        return self._session_dir

    def configure(self) -> None:
        if not self._config.enabled:
            return
        base_dir = Path(self._config.output_dir).expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        session = self._config.session_name.strip() or "request_images"
        self._session_dir = base_dir / f"{session}_{stamp}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        manaprint(f"INFO: [RequestImageSink] session dir: {self._session_dir}")

    def _save_images_by_key(
        self,
        model_input: dict[str, Any],
        image_key: str,
        *,
        query_index: int,
        env_idx: int = 0,
    ) -> None:
        """Core save logic using an explicit image key string."""
        if not self._config.enabled or self._session_dir is None:
            return
        every_n = max(int(self._config.every_n), 1)
        if query_index % every_n != 0:
            return
        items = _iter_wire_image_items_by_key(model_input, image_key)
        if not items:
            return
        if cv2 is None:
            manaprint("WARNING: [RequestImageSink] opencv not available; cannot save images")
            return

        q = int(query_index)
        e = int(env_idx)
        for view_name, payload in items:
            bgr = _payload_to_bgr_uint8(payload)
            if bgr is None:
                manaprint(f"WARNING: [RequestImageSink] skip view '{view_name}' (unsupported or decode failed)")
                continue
            safe = _sanitize_filename_part(view_name)
            path = self._session_dir / f"q{q:06d}_e{e}_{safe}.jpg"
            if not cv2.imwrite(str(path), bgr):
                manaprint(f"WARNING: [RequestImageSink] imwrite failed for {path}")

    def save_wire_images(
        self,
        model_input: dict[str, Any],
        codec: ObservationCodec | None,
        *,
        query_index: int,
        env_idx: int = 0,
    ) -> None:
        if not self._config.enabled or self._session_dir is None:
            return
        if codec is None:
            if not self._warned_no_codec:
                manaprint("WARNING: [RequestImageSink] save_request_images enabled but no ObservationCodec; skipping")
                self._warned_no_codec = True
            return
        self._save_images_by_key(model_input, codec.image_out_key, query_index=query_index, env_idx=env_idx)

    # -- Wire interceptor interface (on_request / on_response / reset) --------

    def on_request(self, request: dict[str, Any], *, env_idx: int | None) -> dict[str, Any]:
        """Save images from the wire request, then return it unchanged."""
        if not self._config.enabled or self._session_dir is None:
            return request
        self._save_images_by_key(
            request,
            self._config.image_key,
            query_index=self._query_counter,
            env_idx=int(env_idx) if env_idx is not None else 0,
        )
        self._query_counter += 1
        return request

    def on_response(self, raw: dict[str, Any], *, env_idx: int | None) -> dict[str, Any]:
        return raw

    def reset(self, env_ids: Any = None) -> None:
        pass
