"""Zero-dimensional action term for safe feed-forward effort overlays."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

from manaenv.actuators.safe_effort_overlay import SafeEffortOverlay


class SafeEffortOverlayAction(ActionTerm):
    """Apply an opt-in effort overlay after position/velocity action terms."""

    cfg: "SafeEffortOverlayActionCfg"

    def __init__(self, cfg: "SafeEffortOverlayActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._overlay = SafeEffortOverlay(
            enabled=cfg.enabled,
            joint_names=cfg.overlay_joint_names,
            effort_n=cfg.effort_n,
            max_effort_n=cfg.max_effort_n,
            ramp_s=cfg.ramp_s,
        )
        self._overlay.bind(num_envs=self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions
        self._processed_actions = actions

    def apply_actions(self) -> None:
        self._overlay.apply(self._asset, dt=float(self._env.cfg.sim.dt))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._overlay.reset(self._asset, env_ids)


@configclass
class SafeEffortOverlayActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SafeEffortOverlayAction
    enabled: bool = False
    overlay_joint_names: tuple[str, ...] = ()
    effort_n: float = 0.0
    max_effort_n: float = 0.0
    ramp_s: float = 0.0
