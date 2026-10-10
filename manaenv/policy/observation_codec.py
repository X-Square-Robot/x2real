"""YAML-driven observation adapter runtime.

``ObservationCodec`` converts simulator-native observations into the wallx
desktop wire envelope ``{views: ..., state: ..., instruction: ...}``.

All output key names, image encoding, state dtype, and key mappings are
read from the observation-space YAML at construction time. Sim-side state
keys are filtered against ``state_keys``; sim-side camera keys are filtered
against ``camera_key_map`` / ``camera_keys`` (both populated by
:func:`~manaenv.policy.policy_implementation.wallx._build_codec_config`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from manaenv.utils.tensor_utils import select_env_item, tensor_to_numpy


class ObservationCodec:
    """Data-driven observation adapter built from obs YAML fields."""

    def __init__(self, obs_config: dict[str, Any]) -> None:
        self._image_out_key: str = obs_config.get("obs_image_key", "views")
        self._state_out_key: str = obs_config.get("state_key", "state")
        self._instruction_out_key: str = obs_config.get("instruction_key", "instruction")
        language_cfg = obs_config.get("language") or {}
        self._language_out_key: str = language_cfg.get("instruction_key", self._instruction_out_key)

        video_cfg = obs_config.get("video") or {}
        self._camera_keys: list[str] = video_cfg.get("camera_keys", [])
        self._camera_key_map: dict[str, str] = video_cfg.get("camera_key_map") or obs_config.get("camera_key_map") or {}
        self._image_encoder: Callable = _build_image_encoder(video_cfg)

        state_cfg = obs_config.get("state") or {}
        self._state_keys: list[str] = state_cfg.get("state_keys", [])
        self._state_dtype: str = state_cfg.get("dtype", "float32")
        self._state_key_map: dict[str, str] = state_cfg.get("state_key_map") or {}
        self._allowed_state_sources: set[str] = set(self._state_keys) | set(self._state_key_map.keys())
        self._allowed_camera_sources: set[str] = set(self._camera_key_map.keys()) or set(self._camera_keys)

    def _extract_images(self, observation: dict[str, Any], env_idx: int | None = None) -> dict[str, Any]:
        images: dict[str, Any] = {}
        camera_obs = observation.get("camera_obs", {})
        if not isinstance(camera_obs, dict):
            raise TypeError(f"[ObsCodec] observation['camera_obs'] must be a dict, got {type(camera_obs).__name__}")
        for cam, rgb in camera_obs.items():
            if self._allowed_camera_sources and cam not in self._allowed_camera_sources:
                continue
            encoded = self._image_encoder(select_env_item(rgb, env_idx))
            if encoded is not None:
                out_name = self._camera_key_map.get(cam, cam)
                images[out_name] = encoded
        return images

    def _extract_state(self, observation: dict[str, Any], env_idx: int | None = None) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for key, val in observation.get("policy", {}).items():
            if self._allowed_state_sources and key not in self._allowed_state_sources:
                continue
            converted = tensor_to_numpy(select_env_item(val, env_idx))
            if converted is not None:
                out_key = self._state_key_map.get(key, key)
                state[out_key] = converted
        return state

    def encode(
        self,
        observation: dict[str, Any],
        instruction: str = "",
        env_idx: int | None = None,
    ) -> dict[str, Any]:
        """Serialize a simulator-native observation into the wire envelope."""
        return {
            self._image_out_key: self._extract_images(observation, env_idx=env_idx),
            self._state_out_key: self._extract_state(observation, env_idx=env_idx),
            self._instruction_out_key: instruction,
        }

    @property
    def image_out_key(self) -> str:
        return self._image_out_key

    @property
    def state_out_key(self) -> str:
        return self._state_out_key

    @property
    def instruction_out_key(self) -> str:
        return self._instruction_out_key

    @property
    def language_out_key(self) -> str:
        return self._language_out_key

    @property
    def state_key_map(self) -> dict[str, str]:
        return dict(self._state_key_map)

    @property
    def camera_key_map(self) -> dict[str, str]:
        return dict(self._camera_key_map)


def _build_image_encoder(video_cfg: dict[str, Any]) -> Callable:
    """Resolve ``video.encode`` to an encoder instance via the global registry."""
    from manaenv.pipeline.registry.bootstrap import bootstrap_discovery
    from manaenv.pipeline.registry.global_provider import global_registry
    from manaenv.policy.image_encoder_discovery import ENCODE_TO_SYMBOL

    encode_name: str = video_cfg.get("encode", "jpeg_b64")
    symbol = ENCODE_TO_SYMBOL.get(encode_name)
    if symbol is None:
        raise ValueError(f"Unknown image encoder '{encode_name}'. Available: {list(ENCODE_TO_SYMBOL)}")
    bootstrap_discovery()
    encoder_cls = global_registry.get_impl(symbol)
    if not callable(encoder_cls):
        raise RuntimeError(f"Image encoder registry symbol {symbol!r} is not callable")
    return encoder_cls(video_cfg)


def create_observation_codec(obs_config: dict[str, Any] | None) -> ObservationCodec | None:
    """Factory helper — returns ``None`` when *obs_config* is empty."""
    if not obs_config:
        return None
    return ObservationCodec(obs_config)
