"""WallxImpl: PolicyImplementation for the wallx wire protocol.

Single-frame, stateless wire codec (PLAN.md §10.5):

  * :meth:`encode_request` -> ``{views, state, instruction}`` envelope built
    from the simulator-native batched observation via :class:`ObservationCodec`.
  * :meth:`decode_response` -> action chunk tensor produced by
    :class:`ActionTransform` (delta IK / absolute EE keyframes / joint
    raw, depending on ``embodiment.action_mode``).
  * :meth:`get_actions` is **not** overridden; wallx is remote-only and
    requires a remote CFG section in the policy yaml. ManaPolicy.load
    enforces this at load time.

Chunk buffering, refill scheduling, EE-pose normalization, and temporal
interpolation **do not** live here — they are handled by ManaPolicy via
the policy yaml ``client:`` segment and the reflection modules
``manaenv.policy.refill_policies`` / ``manaenv.policy.wire_interceptors`` /
``manaenv.policy.chunk_postprocessors``.

Bind-time adapter
-----------------
The :class:`ObservationCodec` / :class:`ActionTransform` factories still
expect dict shapes (``obs_image_key`` / ``state_key`` / ``video`` /
``state`` / ``response_keys`` / ``slot_layout`` / ``ee_frames`` /
``command_space``); :meth:`bind` adapts an enriched
:class:`~manaenv.policy.policy_embodiment.PolicyEmbodimentCFG` to those
dicts. The only protocol constants that remain hard-coded are the wallx
wire envelope keys (``views`` / ``state``) — these are not deployment knobs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from manaenv.policy.action_transforms import ActionTransform, create_action_transform
from manaenv.policy.observation_codec import ObservationCodec, create_observation_codec
from manaenv.policy.policy_embodiment import PolicyEmbodimentCFG
from manaenv.policy.policy_implementation.base import PolicyImplementationBase, PolicyImplementationCFG

# wallx wire envelope keys (fixed by the wallx server contract; not a
# per-deployment knob and never appear in PolicyEmbodimentCFG).
_WALLX_IMAGE_OUT_KEY = "views"
_WALLX_STATE_OUT_KEY = "state"

# action_mode -> ActionTransform command_space (PLAN.md §10.4 PolicyEmbodimentCFG schema)
_ACTION_MODE_TO_COMMAND_SPACE: dict[str, str] = {
    "ik_eef_delta": "eef_delta",
    "ik_eef_abs": "eef_abs",
    "ik_eef_abs_wholebody": "eef_abs_wholebody",
    "ee_abs_ik": "ee_abs_ik",
    "pink_ik": "eef_abs",
    "joint_pos": "joint_raw",
    "joint_position": "joint_raw",
}

# Heuristic: classify an embodiment field key as "camera" when its name
# suggests a camera (prefixed ``camera_`` or suffixed ``_cam`` / ``_image``
# / ``_rgb``). All other keys are treated as state.
_CAMERA_KEY_RE = re.compile(r"^(camera_|.*(_cam|_image|_rgb)$)", re.IGNORECASE)


def _is_camera_field(name: str) -> bool:
    return bool(_CAMERA_KEY_RE.match(name))


_SLICE_RE = re.compile(r"^(?P<base>[^\[]+)(?:\[(?P<start>\d+):(?P<end>\d+)\])?$")


def _parse_action_field(field_spec: str) -> tuple[str, int | None, int | None]:
    m = _SLICE_RE.match(field_spec)
    if m is None:
        raise ValueError(f"Invalid action_field_map key: {field_spec!r} (expected 'name' or 'name[start:end]')")
    base = m.group("base").strip()
    start = m.group("start")
    end = m.group("end")
    return base, (int(start) if start is not None else None), (int(end) if end is not None else None)


def _build_codec_config(embodiment: PolicyEmbodimentCFG) -> dict[str, Any]:
    """Adapt ``PolicyEmbodimentCFG`` -> legacy ObservationCodec input dict."""
    state_keys: list[str] = []
    state_key_map: dict[str, str] = {}
    camera_key_map: dict[str, str] = {}
    for model_name, sim_name in embodiment.obs_field_map.items():
        if _is_camera_field(model_name) or _is_camera_field(sim_name):
            camera_key_map[sim_name] = model_name
        else:
            state_keys.append(sim_name)
            if sim_name != model_name:
                state_key_map[sim_name] = model_name
    return {
        "obs_image_key": _WALLX_IMAGE_OUT_KEY,
        "state_key": _WALLX_STATE_OUT_KEY,
        "instruction_key": embodiment.instruction_key,
        "video": {
            "camera_key_map": camera_key_map,
            "encode": embodiment.image_encode,
            "jpeg_quality": embodiment.image_jpeg_quality,
        },
        "state": {
            "state_keys": state_keys,
            "state_key_map": state_key_map,
            "dtype": embodiment.state_dtype,
        },
        "language": {"instruction_key": embodiment.instruction_key},
    }


def _build_action_transform_config(
    embodiment: PolicyEmbodimentCFG,
    action_horizon: int,
    transform_params: dict[str, Any],
) -> dict[str, Any]:
    """Adapt PolicyEmbodimentCFG -> legacy ActionTransform input dict."""
    command_space = _ACTION_MODE_TO_COMMAND_SPACE.get(embodiment.action_mode)
    if command_space is None:
        raise ValueError(
            f"WallxImpl: unsupported embodiment.action_mode={embodiment.action_mode!r}; "
            f"supported: {sorted(_ACTION_MODE_TO_COMMAND_SPACE)}"
        )

    per_key_ranges: dict[str, list[int]] = {}
    base_order: list[str] = []
    for spec in embodiment.action_field_map.keys():
        base, start, end = _parse_action_field(spec)
        if base not in per_key_ranges:
            per_key_ranges[base] = [start if start is not None else 0, end if end is not None else 0]
            base_order.append(base)
        else:
            cur = per_key_ranges[base]
            if start is not None:
                cur[0] = min(cur[0], start)
            if end is not None:
                cur[1] = max(cur[1], end)

    response_keys: list[dict[str, Any]] = [{"name": base, "slots": [0, per_key_ranges[base][1]]} for base in base_order]

    result = {
        "command_space": command_space,
        "action_dim": embodiment.action_dim,
        "action_horizon": action_horizon,
        "response_keys": response_keys,
        "slot_layout": {k: dict(v) for k, v in embodiment.slot_layout.items()},
        "ee_frames": dict(embodiment.ee_frames),
        "wire_pose_reference": embodiment.wire_pose_reference,
        "transform_params": dict(transform_params),
    }
    if embodiment.metadata:
        result["metadata"] = dict(embodiment.metadata)
    return result


class WallxImpl(PolicyImplementationBase):
    """Wallx wire codec. Remote-only (no local-inference path).

    PLAN.md §10.5: ``encode_request`` builds the wallx wire envelope;
    ``decode_response`` runs :class:`ActionTransform` to produce a chunk
    tensor of shape ``(T, action_dim)``. ``get_actions`` is not
    overridden — ``ManaPolicy.load`` raises if remote serving config is missing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_dim: int = 0
        self.action_horizon: int = 16
        self._codec: ObservationCodec | None = None
        self._action_transform: ActionTransform | None = None
        # Single broadcast string at load; the runtime may swap in a per-env
        # list (active-instruction store), resolved via ``resolve_instruction``.
        self._instruction: str | list[str] = ""

    def load(self, cfg: WallxImplCFG) -> None:
        super().load(cfg)
        self._instruction = cfg.instruction or ""
        self.action_horizon = int(cfg.action_horizon)

    def bind(self, embodiment: PolicyEmbodimentCFG) -> None:
        super().bind(embodiment)
        cfg = self.cfg
        assert isinstance(cfg, WallxImplCFG)
        codec_config = _build_codec_config(embodiment)
        self._codec = create_observation_codec(codec_config)
        action_config = _build_action_transform_config(
            embodiment,
            action_horizon=self.action_horizon,
            transform_params={"pos_gain": cfg.pos_gain, "rot_gain": cfg.rot_gain},
        )
        self.action_dim = int(action_config["action_dim"])
        self._action_transform = create_action_transform(action_config)

    def encode_request(self, observation: dict[str, Any], env_idx: int | None = None) -> dict[str, Any]:
        if self._codec is None:
            raise RuntimeError("WallxImpl.bind() must be called before encode_request()")
        return self._codec.encode(observation, instruction=self.resolve_instruction(env_idx), env_idx=env_idx)

    def decode_response(
        self,
        raw: dict[str, Any],
        env: Any = None,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        if self._action_transform is None:
            raise RuntimeError("WallxImpl.bind() must be called before decode_response()")
        device = getattr(env, "device", "cpu") if env is not None else "cpu"
        return self._action_transform(raw, env=env, device=device, env_idx=env_idx)

    def reset(self, env_ids: Any = None) -> None:
        if self._action_transform is not None:
            self._action_transform.reset(env_ids)


@dataclass
class WallxImplCFG(PolicyImplementationCFG):
    """CFG for :class:`WallxImpl`. Wire-only fields (PLAN.md §10.4 + §10.5).

      * ``instruction``      - language instruction (default ``""``)
      * ``action_horizon``   - max horizon kept from each chunk (default ``16``)
      * ``pos_gain`` / ``rot_gain`` - delta scaling factors (default ``1.0``)

    Chunk-pipeline knobs (``action_chunk_length``, ``interpolation_multiplier``,
    ``ee_pose_normalize``) live in the policy yaml ``client:`` segment;
    EE frame names live in :class:`PolicyEmbodimentCFG.ee_frames`.
    """

    YAML_DISPATCH_KEY: ClassVar[str] = "wallx"

    name: str = ""
    instruction: str = ""
    action_horizon: int = 16
    pos_gain: float = 1.0
    rot_gain: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    class_type: type = WallxImpl
