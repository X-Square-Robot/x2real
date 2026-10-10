"""DPImageImpl: PolicyImplementation for image-conditioned Diffusion Policy.

Local in-process inference adapter for any image-conditioned Diffusion
Policy checkpoint produced by ``BaseWorkspace`` (policy-schema-contract.md).
The checkpoint's Hydra config decides the concrete ``BasePolicy`` subclass
at runtime.

Single-frame, batched-aware wire codec:

  * :meth:`encode_request` -> ``{"batch": batch_dict}`` where
    ``batch_dict`` follows the DP convention
    ``{"obs": {dp_key: (B, 1, C, H, W) or (B, 1, D)}, "instruction": [str]*B}``.
    The batch is built from :class:`PolicyEmbodimentCFG.obs_field_map`
    and resized to ``target_image_size``.
  * :meth:`decode_response` -> action chunk tensor read from
    ``raw["action"]``, shape ``(num_envs, T, action_dim)`` —
    drives ``client.refill_fn=global_sync``.
  * :meth:`get_actions` is overridden: forwards the batch to
    ``inner.predict_action(batch)`` and returns ``{"action": chunk}``.
    No remote serving config is used.

Lazy import contract: the upstream ``diffusion_policy`` package and its
``dill`` loader live in ``3rd/diffusion_policy/`` and are not part of the
manaenv conda env by default. They are imported only inside
:meth:`load`, so that discovering this implementation through the global
registry does not require those packages to be installed.

YAML schema (``defination/policy/<name>@v1.0.0.yaml``)::

    dp_image_impl:
      checkpoint_path: /abs/path/to/dp.ckpt
      device: cuda:0
      num_inference_steps: 16
      use_agent_pos: true
      state_key: robot_joint_pos
      target_image_size: [480, 640]
      instruction: "Pick up the target object."
    client:
      chunk_executor:
        global_sync: {}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from manaenv.policy.policy_embodiment import PolicyEmbodimentCFG
from manaenv.policy.policy_implementation.base import PolicyImplementationBase, PolicyImplementationCFG
from manaenv.utils.debug_print import manaprint


def _load_dp_model_from_checkpoint(checkpoint_path: str, device: str, num_inference_steps: int | None):
    """Load a DP policy from a ``.ckpt`` produced by ``BaseWorkspace``."""
    try:
        import dill
        from diffusion_policy.workspace.base_workspace import BaseWorkspace
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "DPImageImpl: 'diffusion_policy' / 'dill' packages are not importable. "
            "Install them (or add 3rd/diffusion_policy to PYTHONPATH) before loading dp_image policies."
        ) from exc

    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"DPImageImpl: checkpoint not found: {ckpt}")
    payload = torch.load(ckpt.open("rb"), pickle_module=dill)
    workspace = BaseWorkspace(payload["cfg"])
    workspace.load_payload(payload, exclude_keys=tuple(), include_keys=tuple())
    model = workspace.model
    model.eval()
    model.to(device)
    if num_inference_steps is not None and hasattr(model, "num_inference_steps"):
        model.num_inference_steps = num_inference_steps
    return model


class DPImageImpl(PolicyImplementationBase):
    """DP-image local-inference adapter. Overrides ``get_actions``.

    The inner DP model exposes ``predict_action(batch) -> {"action": chunk}``
    where chunk is ``(B, T, action_dim)`` (batched per env). We drive it
    from ``ManaPolicy`` via ``client.refill_fn=global_sync`` so the inner
    model batches every env in a single forward pass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_dim: int = 0
        self._inner_policy: Any | None = None
        self._device: str = "cuda:0"
        self._target_image_size: tuple[int, int] = (480, 640)
        self._use_agent_pos: bool = False
        self._sim_state_key: str = "robot_joint_pos"
        # Single broadcast string at load; the runtime may swap in a per-env
        # list (active-instruction store), resolved via ``resolve_instruction_list``.
        self._instruction: str | list[str] = ""
        self._camera_to_dp: dict[str, str] = {}

    def load(self, cfg: DPImageImplCFG) -> None:
        super().load(cfg)
        self._device = str(cfg.device)
        self._target_image_size = tuple(cfg.target_image_size)
        self._use_agent_pos = bool(cfg.use_agent_pos)
        self._sim_state_key = cfg.state_key
        self._instruction = cfg.instruction
        self._inner_policy = _load_dp_model_from_checkpoint(
            cfg.checkpoint_path,
            self._device,
            cfg.num_inference_steps if cfg.num_inference_steps > 0 else None,
        )
        inferred_dim = getattr(self._inner_policy, "action_dim", 0)
        self.action_dim = int(inferred_dim) if inferred_dim else int(cfg.action_dim)
        manaprint(
            f"INFO: [DPImageImpl] loaded DP image checkpoint '{cfg.checkpoint_path}' " f"(action_dim={self.action_dim})"
        )

    def bind(self, embodiment: PolicyEmbodimentCFG) -> None:
        super().bind(embodiment)
        cam_map: dict[str, str] = {}
        state_keys: list[str] = []
        for model_name, sim_name in embodiment.obs_field_map.items():
            if any(suffix in model_name.lower() for suffix in ("_cam", "_image", "_rgb", "view")):
                cam_map[sim_name] = model_name
            else:
                state_keys.append(sim_name)
        self._camera_to_dp = cam_map
        if state_keys and not self._sim_state_key:
            self._sim_state_key = state_keys[0]
        if not self.action_dim and embodiment.action_dim:
            self.action_dim = int(embodiment.action_dim)

    def encode_request(self, observation: dict[str, Any], env_idx: int | None = None) -> dict[str, Any]:
        batch: dict[str, Any] = {"obs": {}}
        camera_obs = observation.get("camera_obs", {}) or {}
        target_h, target_w = int(self._target_image_size[0]), int(self._target_image_size[1])
        num_envs = 1
        for sim_cam, dp_key in self._camera_to_dp.items():
            rgb = camera_obs.get(sim_cam)
            if rgb is None:
                continue
            if rgb.dtype == torch.uint8:
                rgb = rgb.float()
            rgb = rgb.permute(0, 3, 1, 2)
            if rgb.shape[-2:] != (target_h, target_w):
                rgb = F.interpolate(rgb, size=(target_h, target_w), mode="bilinear", align_corners=False)
            num_envs = max(num_envs, int(rgb.shape[0]))
            batch["obs"][dp_key] = rgb.unsqueeze(1).to(self._device)
        if self._use_agent_pos and self._sim_state_key:
            policy = observation.get("policy", {}) or {}
            state = policy.get(self._sim_state_key)
            if state is not None:
                state = state.float() if hasattr(state, "float") else state
                num_envs = max(num_envs, int(state.shape[0]))
                batch["obs"]["agent_pos"] = state.unsqueeze(1).to(self._device)
        instruction_list = self.resolve_instruction_list(num_envs)
        if instruction_list:
            batch["instruction"] = instruction_list
        return {"batch": batch}

    def decode_response(
        self,
        raw: dict[str, Any],
        env: Any = None,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        chunk = raw.get("action")
        if chunk is None:
            manaprint(f"WARNING: [DPImageImpl] response missing 'action'; available={list(raw.keys())}")
            return None
        if not isinstance(chunk, torch.Tensor):
            chunk = torch.as_tensor(chunk, dtype=torch.float32)
        return chunk

    def get_actions(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._inner_policy is None:
            raise RuntimeError("DPImageImpl.load() must be called before get_actions()")
        batch = request.get("batch")
        if batch is None:
            raise RuntimeError("DPImageImpl.get_actions: request missing 'batch' key")
        with torch.no_grad():
            result = self._inner_policy.predict_action(batch)
        if not isinstance(result, dict) or "action" not in result:
            raise RuntimeError("DPImageImpl.get_actions: inner policy did not return 'action' key in result")
        return {"action": result["action"]}

    def cleanup(self) -> None:
        self._inner_policy = None


@dataclass
class DPImageImplCFG(PolicyImplementationCFG):
    """CFG for :class:`DPImageImpl`. Wire / inference fields only.

    Chunk-pipeline knobs live in the policy yaml ``client:`` segment.
    """

    YAML_DISPATCH_KEY: ClassVar[str] = "dp_image"

    name: str = ""
    checkpoint_path: str = ""
    device: str = "cuda:0"
    action_dim: int = 0
    num_inference_steps: int = 16
    use_agent_pos: bool = False
    state_key: str = "robot_joint_pos"
    target_image_size: tuple[int, int] = (480, 640)
    instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    class_type: type = DPImageImpl
