"""Gr00tImpl: PolicyImplementation for GR00T closed-loop inference.

Local in-process inference adapter wrapping
:class:`isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy`
(PLAN.md §10.4 + §10.5).

Single-frame, batched-aware wire codec:

  * :meth:`encode_request` -> ``{"observation": observation,
    "camera_name": pov_cam_name_sim}``. The observation is forwarded as-is;
    GR00T's inner policy reads multi-env tensors directly.
  * :meth:`decode_response` -> ``(num_envs, T, action_dim)`` tensor read
    from ``raw["actions"]`` — drives ``client.refill_fn=global_sync``
    (one batched forward fills every env).
  * :meth:`get_actions` is overridden: forwards the request straight to
    ``inner.get_action_chunk(observation, camera_name)`` and returns
    ``{"actions": tensor}``. No remote serving config is used.

Lazy import contract: the upstream ``isaaclab_arena_gr00t`` and GR00T model
packages are heavy and may be absent in some development machines. They are
therefore imported only inside :meth:`load`, never at module top-level, so
that discovering this implementation through the global registry does not
require those packages to be installed.

YAML schema (``defination/policy/<name>@v1.0.0.yaml``)::

    gr00t_impl:
      model_path: ~/models/GR00T/...
      language_instruction: "Pick up the target object."
      embodiment_tag: gr1
      task_mode_name: gr1_tabletop_manipulation
      data_config: fourier_gr1_arms_only
      policy_joints_config_path: /path/to/policy_joint_space.yaml
      action_joints_config_path: /path/to/14dof_joint_space.yaml
    state_joints_config_path: /path/to/14dof_joint_space.yaml
    pov_cam_name_sim: head_cam
    target_image_size: [256, 256, 3]
    client:
      chunk_executor:
        global_sync: {}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch

from manaenv.policy.policy_embodiment import PolicyEmbodimentCFG
from manaenv.policy.policy_implementation.base import PolicyImplementationBase, PolicyImplementationCFG
from manaenv.utils.debug_print import manaprint


class Gr00tImpl(PolicyImplementationBase):
    """GR00T local-inference adapter. Overrides ``get_actions``.

    The inner :class:`Gr00tClosedloopPolicy` exposes ``get_action_chunk``
    (one batched forward returning ``(num_envs, T, action_dim)``); we
    drive it from ``ManaPolicy`` via ``client.refill_fn=global_sync``,
    which lets the inner model batch every env in one call rather than
    looping ``per_env_independent``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_dim: int = 0
        self._inner_policy: Any | None = None
        self._device: str = "cuda:0"
        self._pov_cam_name_sim: str = ""

    def load(self, cfg: Gr00tImplCFG) -> None:
        super().load(cfg)
        try:
            from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyConfig
            from isaaclab_arena_gr00t.policy.gr00t_closedloop_policy import Gr00tClosedloopPolicy
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Gr00tImpl.load(): isaaclab_arena_gr00t package is not importable. "
                "Install it (it ships with isaaclabarena) or remove gr00t policy yamls "
                "from the active pipeline."
            ) from exc

        # Upstream ``pov_cam_name_sim`` is now ``list[str]`` (was a single str);
        # normalise so a plain-string yaml value keeps working.
        pov_cams = cfg.pov_cam_name_sim
        pov_cams_list = [pov_cams] if isinstance(pov_cams, str) else list(pov_cams)

        gr00t_cfg = Gr00tClosedloopPolicyConfig(
            language_instruction=cfg.language_instruction,
            model_path=cfg.model_path,
            action_horizon=cfg.action_horizon,
            action_chunk_length=cfg.action_chunk_length,
            embodiment_tag=cfg.embodiment_tag,
            denoising_steps=cfg.denoising_steps,
            data_config=cfg.data_config,
            modality_config_path=cfg.modality_config_path or None,
            original_image_size=tuple(cfg.original_image_size),
            target_image_size=tuple(cfg.target_image_size),
            policy_joints_config_path=Path(cfg.policy_joints_config_path),
            task_mode_name=cfg.task_mode_name,
            action_joints_config_path=Path(cfg.action_joints_config_path),
            state_joints_config_path=Path(cfg.state_joints_config_path),
            policy_device=cfg.policy_device,
            pov_cam_name_sim=pov_cams_list,
        )
        for extra_field, extra_value in cfg.policy_config_extra.items():
            if hasattr(gr00t_cfg, extra_field):
                setattr(gr00t_cfg, extra_field, extra_value)

        # Programmatic entry point provided by upstream for embedding frameworks
        # (replaces the previous ``__new__`` + ``_initGr00tClosedloopPolicy`` hack).
        inner = Gr00tClosedloopPolicy.from_policy_config(
            gr00t_cfg,
            num_envs=cfg.num_envs,
            policy_device=cfg.policy_device,
        )
        # ``get_action_chunk`` asserts a task description is set; fall back to the
        # config's ``language_instruction`` when none is supplied externally.
        inner.set_task_description(None)
        self._inner_policy = inner
        self._device = str(cfg.policy_device)
        self._pov_cam_name_sim = pov_cams_list[0] if pov_cams_list else ""
        self.action_dim = int(getattr(inner, "action_dim", 0))
        manaprint(
            f"INFO: [Gr00tImpl] loaded GR00T model from '{cfg.model_path}' "
            f"(action_dim={self.action_dim}, embodiment_tag={cfg.embodiment_tag})"
        )

    def bind(self, embodiment: PolicyEmbodimentCFG) -> None:
        super().bind(embodiment)

    def encode_request(self, observation: dict[str, Any], env_idx: int | None = None) -> dict[str, Any]:
        return {"observation": observation, "camera_name": self._pov_cam_name_sim}

    def decode_response(
        self,
        raw: dict[str, Any],
        env: Any = None,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        actions = raw.get("actions")
        if actions is None:
            manaprint(f"WARNING: [Gr00tImpl] response missing 'actions'; available={list(raw.keys())}")
            return None
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        return actions

    def get_actions(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._inner_policy is None:
            raise RuntimeError("Gr00tImpl.load() must be called before get_actions()")
        observation = request.get("observation")
        camera_name = request.get("camera_name") or self._pov_cam_name_sim
        if observation is None:
            raise RuntimeError("Gr00tImpl.get_actions: request missing 'observation' key")
        actions = self._inner_policy.get_action_chunk(observation, camera_name)
        return {"actions": actions}

    def reset(self, env_ids: Any = None) -> None:
        if self._inner_policy is None:
            return
        if hasattr(self._inner_policy, "reset"):
            self._inner_policy.reset(env_ids)

    def cleanup(self) -> None:
        self._inner_policy = None


@dataclass
class Gr00tImplCFG(PolicyImplementationCFG):
    """CFG for :class:`Gr00tImpl`.

    Mirrors :class:`isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config.Gr00tClosedloopPolicyConfig`
    plus a few orchestration fields (``num_envs`` / ``policy_device`` /
    ``pov_cam_name_sim``).

    Chunk-pipeline knobs (``client.chunk_executor.global_sync``) live in
    the policy yaml ``client:`` segment; ``action_chunk_length`` here
    selects the inner GR00T chunk length and is consumed both by the
    inner model and by ``client.chunk_executor.global_sync.execute_horizon``.

    ``policy_config_extra`` is an escape hatch for upstream
    ``Gr00tClosedloopPolicyConfig`` fields that are not modelled here yet
    (avoids a hard dependency on every upstream field).
    """

    YAML_DISPATCH_KEY: ClassVar[str] = "gr00t"

    name: str = ""
    model_path: str = ""
    language_instruction: str = ""
    embodiment_tag: str = "gr1"
    task_mode_name: str = "gr1_tabletop_manipulation"
    data_config: str = "fourier_gr1_arms_only"
    modality_config_path: str = ""
    action_horizon: int = 16
    action_chunk_length: int = 16
    denoising_steps: int = 4
    original_image_size: tuple[int, int, int] = (480, 640, 3)
    target_image_size: tuple[int, int, int] = (256, 256, 3)
    policy_joints_config_path: str = ""
    action_joints_config_path: str = ""
    state_joints_config_path: str = ""
    pov_cam_name_sim: str = "head_cam"
    num_envs: int = 1
    policy_device: str = "cuda:0"
    policy_config_extra: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    class_type: type = Gr00tImpl
