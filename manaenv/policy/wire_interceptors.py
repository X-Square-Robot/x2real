"""Wire interceptors — CFG-reflected per-call request/response rewrites
listed by the ``client.wire_interceptors`` segment of a policy yaml
(PLAN.md §10.5).

Each entry is an explicit CFG key, e.g.
``wire_interceptors: [{save_request_images: {every_n: 1}}]``, reflected
through the global lazy registry. ``ManaPolicy`` instantiates
``cfg.class_type()``, calls ``load(cfg)``, and invokes two methods on
every refill:

  * ``on_request(request, *, env_idx) -> request``
  * ``on_response(raw, *, env_idx) -> raw``

The legacy ``ee_pose_normalize`` interceptor used to capture the
episode-start EE pose per env on the wire and add it back on the
response trajectory. That responsibility now lives at the observation
layer (the WallX ``follow*_pos`` ObsTerms emit root-frame values relative
to the episode init EE pose), and the matching reassembly happens inside
:class:`manaenv.policy.action_transforms.EEAbsoluteIKTransform`
(``wire_pose_reference='init_ee_pose'``); no wire-side normaliser is needed
any more.

:class:`SaveRequestImages` is kept here for ad-hoc debug runs that need
to dump the JPEG payload that goes to the policy server.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

from manaenv.policy.chunk_scheduler.base import REFILL_CONTEXT_KEY


@dataclass
class WireInterceptorBaseCFG:
    """Base CFG for wire interceptor plugins."""

    class_type: type | None = None


class WireInterceptorBase(ABC):
    """Runtime base for wire interceptors (stateful per policy instance)."""

    def __init__(self) -> None:
        self.cfg: WireInterceptorBaseCFG | None = None

    def load(self, cfg: WireInterceptorBaseCFG) -> None:
        self.cfg = cfg

    @abstractmethod
    def on_request(self, request: dict[str, Any], *, env_idx: int | None) -> dict[str, Any]:
        """Rewrite (or observe) the outgoing wire request."""

    @abstractmethod
    def on_response(self, raw: dict[str, Any], *, env_idx: int | None) -> dict[str, Any]:
        """Rewrite (or observe) the raw wire response."""

    def reset(self, env_ids: Any = None) -> None:
        """Optional per-env reset hook."""


class SaveRequestImages(WireInterceptorBase):
    """Wire interceptor that persists request images as JPEGs.

    YAML form::

        client:
          wire_interceptors:
          - save_request_images:
              output_dir: "~/Desktop/manaenv_output/policy_request_images"
              every_n: 1
              image_key: views

    The interceptor is transparent — ``on_request`` saves images then
    returns the request unchanged; ``on_response`` is a passthrough.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sink = None

    def load(self, cfg: WireInterceptorBaseCFG) -> None:
        super().load(cfg)
        from manaenv.policy.request_image_sink import RequestImageSink, RequestImageSinkConfig

        config = RequestImageSinkConfig(
            enabled=True,
            output_dir=str(getattr(cfg, "output_dir", "~/Desktop/manaenv_output/policy_request_images")),
            session_name=str(getattr(cfg, "session_name", "")),
            every_n=int(getattr(cfg, "every_n", 1)),
            image_key=str(getattr(cfg, "image_key", "views")),
        )
        self._sink = RequestImageSink(config)
        self._sink.configure()

    def on_request(self, request: dict[str, Any], *, env_idx: int | None) -> dict[str, Any]:
        assert self._sink is not None, "SaveRequestImages.load() must run before use"
        return self._sink.on_request(request, env_idx=env_idx)

    def on_response(self, raw: dict[str, Any], *, env_idx: int | None) -> dict[str, Any]:
        assert self._sink is not None, "SaveRequestImages.load() must run before use"
        return self._sink.on_response(raw, env_idx=env_idx)

    def reset(self, env_ids: Any = None) -> None:
        if self._sink is not None:
            self._sink.reset(env_ids)


@dataclass
class SaveRequestImagesCFG(WireInterceptorBaseCFG):
    output_dir: str = "~/Desktop/manaenv_output/policy_request_images"
    every_n: int = 1
    session_name: str = ""
    image_key: str = "views"
    class_type: type = SaveRequestImages


class WallXRtc(WireInterceptorBase):
    """Inject Wall-X RTC prefix fields for async overlapping chunk refills.

    The first refill is deliberately vanilla.  Its raw ``[state, action...]``
    trajectory is cached per environment.  On a later async refill the
    scheduler supplies how many post-interpolation rows have advanced and how
    many will overlap inference; this interceptor converts both counts back to
    model-action rows and sends the remaining old trajectory as RTC context.

    Any missing/misaligned state falls back to vanilla inference.  Replaying an
    incorrectly shifted prefix is less safe than losing RTC for one refill.
    """

    def __init__(self) -> None:
        super().__init__()
        self._previous_chunks: dict[int, np.ndarray] = {}
        self._env_generations: dict[int, int] = {}
        self._pending_generations: dict[int, int] = {}
        self._state_lock = Lock()
        self._temporal_multiplier = 1.0
        self._max_delay = 8
        self._model_action_horizon = 32
        self._num_flow_steps = 0

    def load(self, cfg: WireInterceptorBaseCFG) -> None:
        super().load(cfg)
        self._temporal_multiplier = float(getattr(cfg, "temporal_multiplier", 1.0))
        self._max_delay = int(getattr(cfg, "max_delay", 8))
        self._model_action_horizon = int(getattr(cfg, "model_action_horizon", 32))
        self._num_flow_steps = int(getattr(cfg, "num_flow_steps", 0))
        if self._temporal_multiplier < 1.0:
            raise ValueError("wall_x_rtc.temporal_multiplier must be >= 1")
        if self._max_delay < 1:
            raise ValueError("wall_x_rtc.max_delay must be >= 1")
        if self._model_action_horizon < 1:
            raise ValueError("wall_x_rtc.model_action_horizon must be >= 1")
        if self._max_delay > self._model_action_horizon:
            raise ValueError("wall_x_rtc.max_delay cannot exceed model_action_horizon")

    @staticmethod
    def _env_key(env_idx: int | None) -> int:
        return 0 if env_idx is None else int(env_idx)

    def _to_model_rows(self, value: Any) -> int | None:
        try:
            scaled = float(value) / self._temporal_multiplier
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        rounded = int(round(scaled))
        # RTC alignment must be exact.  The shipped 48-row/1.5x setup maps
        # advanced=36 -> 24 model rows and overlap=12 -> 8 model rows.
        if abs(scaled - rounded) > 1e-6:
            return None
        return rounded

    @staticmethod
    def _sensor_state_from_request(request: dict[str, Any]) -> np.ndarray | None:
        obs = request.get("obs")
        if not isinstance(obs, dict):
            return None
        state = obs.get("state")
        if not isinstance(state, dict):
            return None
        left = state.get("follow1_pos")
        right = state.get("follow2_pos")
        if left is None or right is None:
            return None
        row = np.concatenate(
            [np.asarray(left, dtype=np.float32).reshape(-1), np.asarray(right, dtype=np.float32).reshape(-1)]
        )
        if row.shape != (14,) or not np.isfinite(row).all():
            return None
        return row

    def on_request(self, request: dict[str, Any], *, env_idx: int | None) -> dict[str, Any]:
        request = dict(request)
        env_key = self._env_key(env_idx)
        with self._state_lock:
            # Record every request, including vanilla/first requests, so a
            # response arriving after reset cannot repopulate an old episode.
            self._pending_generations[env_key] = self._env_generations.get(env_key, 0)
            previous = self._previous_chunks.get(env_key)
        context = request.pop(REFILL_CONTEXT_KEY, None)
        if not isinstance(context, dict) or not bool(context.get("crop_overlap")):
            return request

        if previous is None:
            return request

        advanced = self._to_model_rows(context.get("advanced_rows"))
        delay = self._to_model_rows(context.get("overlap_rows"))
        if advanced is None or delay is None or delay <= 0:
            return request
        delay = min(delay, self._max_delay)
        if advanced < 0 or advanced > self._model_action_horizon:
            return request

        shifted = np.asarray(previous[advanced:], dtype=np.float32).copy()
        # RTC layout needs one state/reference row followed by at least D action
        # rows.  A server returning 32 instead of the desktop contract's 33 rows
        # therefore falls back instead of silently pinning D-1 actions.
        if shifted.ndim != 2 or shifted.shape[1] != 14 or shifted.shape[0] < delay + 1:
            return request

        sensor_state = self._sensor_state_from_request(request)
        if sensor_state is not None:
            shifted[0] = sensor_state

        obs = request.get("obs")
        if not isinstance(obs, dict):
            return request
        rtc_obs = dict(obs)
        rtc_obs.update(
            {
                "use_rtc": True,
                "prev_action_chunk": np.ascontiguousarray(shifted),
                "inference_delay": delay,
                "rtc_mode": "training_time",
            }
        )
        if self._num_flow_steps > 0:
            rtc_obs["num_flow_steps"] = self._num_flow_steps
        request["obs"] = rtc_obs
        return request

    def on_response(self, raw: dict[str, Any], *, env_idx: int | None) -> dict[str, Any]:
        env_key = self._env_key(env_idx)
        with self._state_lock:
            request_generation = self._pending_generations.pop(env_key, None)
            generation_is_current = request_generation == self._env_generations.get(env_key, 0)
        if isinstance(raw, dict):
            # This hook runs after PolicySpaceServer has normalized any
            # model-specific response (for example WallX follow1_pos / follow2_pos)
            # into the public {"actions": (H, D)} wire contract.
            value = raw.get("actions")
            if value is not None:
                chunk = np.asarray(value, dtype=np.float32)
                if (
                    chunk.ndim == 2
                    and chunk.shape == (self._model_action_horizon + 1, 14)
                    and np.isfinite(chunk).all()
                    and generation_is_current
                ):
                    with self._state_lock:
                        # Reset may have raced shape validation above; verify the
                        # generation once more while committing the cache.
                        if request_generation == self._env_generations.get(env_key, 0):
                            self._previous_chunks[env_key] = np.ascontiguousarray(chunk.copy())
        return raw

    def reset(self, env_ids: Any = None) -> None:
        with self._state_lock:
            if env_ids is None:
                keys = set(self._previous_chunks) | set(self._pending_generations) | set(self._env_generations)
            else:
                if hasattr(env_ids, "detach"):
                    env_ids = env_ids.detach().cpu().numpy()
                keys = {int(env_idx) for env_idx in np.asarray(env_ids).reshape(-1)}
            for env_key in keys:
                self._env_generations[env_key] = self._env_generations.get(env_key, 0) + 1
                self._previous_chunks.pop(env_key, None)
                self._pending_generations.pop(env_key, None)


@dataclass
class WallXRtcCFG(WireInterceptorBaseCFG):
    """Wall-X RTC client settings, expressed in model/action time units."""

    temporal_multiplier: float = 1.0
    max_delay: int = 8
    model_action_horizon: int = 32
    num_flow_steps: int = 0
    class_type: type = WallXRtc
