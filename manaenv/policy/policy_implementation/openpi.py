"""OpenpiImpl: PolicyImplementation for the OpenPI wire protocol.

Single-frame, stateless wire codec (PLAN.md §10.5):

  * :meth:`encode_request` -> flat top-level envelope
    ``{cam_high: img, cam_left_wrist: img, ..., state: vec, prompt: str}``
    built from :class:`PolicyEmbodimentCFG.obs_field_map`. State is the
    concatenated joint vector, optionally indexed by
    ``embodiment.metadata['active_joint_indices']``.
  * :meth:`decode_response` -> action chunk tensor read from
    ``raw[action_response_key]``, shape ``(T, action_dim)``.
  * :meth:`get_actions` is **not** overridden; OpenPI is remote-only and
    requires a remote CFG section in the policy yaml.

Chunk buffering, refill scheduling, horizon clipping and chunk
postprocessing **do not** live here — they are handled by ManaPolicy via
the policy yaml ``client:`` segment (``chunk_postprocessors.horizon_clip``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import torch

from manaenv.policy.policy_embodiment import PolicyEmbodimentCFG
from manaenv.policy.policy_implementation.base import PolicyImplementationBase, PolicyImplementationCFG
from manaenv.utils.debug_print import manaprint
from manaenv.utils.tensor_utils import select_env_item, tensor_to_jpeg_b64, tensor_to_numpy, to_numpy_keep_shape

_CAMERA_KEY_RE = re.compile(r"^(camera_|.*(_cam|_image|_rgb|_high|_low|_wrist)$)", re.IGNORECASE)


def _is_camera_field(name: str) -> bool:
    return bool(_CAMERA_KEY_RE.match(name))


def _resolve_image_encoder(encode_name: str, jpeg_quality: int):
    if encode_name == "raw_numpy":
        return to_numpy_keep_shape
    if encode_name == "jpeg_b64":

        def _enc(rgb):
            return tensor_to_jpeg_b64(rgb, quality=jpeg_quality)

        return _enc
    raise ValueError(f"OpenpiImpl: unknown image encoder '{encode_name}'; expected 'raw_numpy' or 'jpeg_b64'")


class OpenpiImpl(PolicyImplementationBase):
    """OpenPI wire codec. Remote-only (no local-inference path).

    PLAN.md §10.5: ``encode_request`` builds the OpenPI flat envelope;
    ``decode_response`` reads ``raw[action_response_key]`` and returns
    ``(T, action_dim)``. ``get_actions`` is not overridden — ``ManaPolicy.load``
    raises if remote serving config is missing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_dim: int = 0
        # Single broadcast string at load; the runtime may swap in a per-env
        # list (active-instruction store), resolved via ``resolve_instruction``.
        self._instruction: str | list[str] = ""
        self._image_encoder = None
        self._action_response_key: str = "actions"
        self._sim_state_keys: list[str] = []
        self._camera_to_camera_out: dict[str, str] = {}
        self._active_joint_indices: list[int] | None = None
        self._instruction_out_key: str = "prompt"
        self._state_out_key: str = "state"

    def load(self, cfg: OpenpiImplCFG) -> None:
        super().load(cfg)
        self._instruction = cfg.instruction or ""
        self._action_response_key = cfg.action_response_key or "actions"
        self._image_encoder = _resolve_image_encoder(cfg.image_encode, cfg.image_jpeg_quality)
        self._instruction_out_key = cfg.instruction_out_key or "prompt"
        self._state_out_key = cfg.state_out_key or "state"

    def bind(self, embodiment: PolicyEmbodimentCFG) -> None:
        super().bind(embodiment)
        cam_map: dict[str, str] = {}
        sim_state_keys: list[str] = []
        for model_name, sim_name in embodiment.obs_field_map.items():
            if _is_camera_field(model_name) or _is_camera_field(sim_name):
                cam_map[sim_name] = model_name
            else:
                sim_state_keys.append(sim_name)
        self._camera_to_camera_out = cam_map
        self._sim_state_keys = sim_state_keys
        self.action_dim = int(embodiment.action_dim) or self.action_dim
        active = embodiment.metadata.get("active_joint_indices") if embodiment.metadata else None
        if isinstance(active, list) and active:
            self._active_joint_indices = [int(x) for x in active]
        else:
            self._active_joint_indices = None

    def encode_request(self, observation: dict[str, Any], env_idx: int | None = None) -> dict[str, Any]:
        if self._image_encoder is None:
            raise RuntimeError("OpenpiImpl.load() must be called before encode_request()")
        request: dict[str, Any] = {}
        camera_obs = observation.get("camera_obs", {}) or {}
        for sim_cam, out_cam in self._camera_to_camera_out.items():
            rgb = camera_obs.get(sim_cam)
            if rgb is None:
                continue
            encoded = self._image_encoder(select_env_item(rgb, env_idx))
            if encoded is not None:
                request[out_cam] = encoded
        state_vec = self._build_state_vector(observation, env_idx=env_idx)
        if state_vec is not None:
            request[self._state_out_key] = state_vec
        request[self._instruction_out_key] = self.resolve_instruction(env_idx)
        return request

    def decode_response(
        self,
        raw: dict[str, Any],
        env: Any = None,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        chunk = raw.get(self._action_response_key)
        if chunk is None:
            manaprint(
                f"WARNING: [OpenpiImpl] response missing key '{self._action_response_key}'; "
                f"available={list(raw.keys())}"
            )
            return None
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] <= 0:
            return None
        device = getattr(env, "device", "cpu") if env is not None else "cpu"
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    def _build_state_vector(self, observation: dict[str, Any], env_idx: int | None = None) -> np.ndarray | None:
        policy = observation.get("policy", {}) or {}
        chunks: list[np.ndarray] = []
        for key in self._sim_state_keys:
            val = policy.get(key)
            if val is None:
                continue
            arr = tensor_to_numpy(select_env_item(val, env_idx))
            if arr is None:
                continue
            arr_np = np.asarray(arr, dtype=np.float32).flatten()
            chunks.append(arr_np)
        if not chunks:
            return None
        state = np.concatenate(chunks, axis=0)
        if self._active_joint_indices:
            state = state[self._active_joint_indices]
        return state


@dataclass
class OpenpiImplCFG(PolicyImplementationCFG):
    """CFG for :class:`OpenpiImpl`. Wire-only fields (PLAN.md §10.4 + §10.5).

      * ``instruction``         - language instruction (default ``""``)
      * ``action_response_key`` - response dict key carrying the action chunk
                                  (default ``"actions"``)
      * ``image_encode``        - ``raw_numpy`` (default) or ``jpeg_b64``
      * ``image_jpeg_quality``  - quality when ``image_encode='jpeg_b64'``
      * ``state_out_key``       - wire envelope state key (default ``"state"``)
      * ``instruction_out_key`` - wire envelope instruction key (default ``"prompt"``)

    Chunk-pipeline knobs (``action_chunk_length``, ``action_horizon``) live
    in the policy yaml ``client:`` segment.
    """

    YAML_DISPATCH_KEY: ClassVar[str] = "openpi"

    name: str = ""
    instruction: str = ""
    action_response_key: str = "actions"
    image_encode: str = "raw_numpy"
    image_jpeg_quality: int = 95
    state_out_key: str = "state"
    instruction_out_key: str = "prompt"
    metadata: dict[str, Any] = field(default_factory=dict)
    class_type: type = OpenpiImpl
