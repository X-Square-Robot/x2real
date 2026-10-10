"""No-op action channels used only to align external policy interfaces."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass


class NoopAction(ActionTerm):
    """Consume a fixed-width action slice without changing simulation state."""

    cfg: "NoopActionCfg"

    def __init__(self, cfg: "NoopActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return int(self.cfg.action_dim)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

    def apply_actions(self) -> None:
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class NoopActionCfg(ActionTermCfg):
    """Config for :class:`NoopAction`."""

    class_type: type[ActionTerm] = NoopAction

    action_dim: int = 0
