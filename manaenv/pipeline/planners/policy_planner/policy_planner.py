"""Batched policy planner used by public task evaluation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..planner_base import PlannerBase, PlannerCFG

if TYPE_CHECKING:
    from manaenv.policy.mana_policy import ManaPolicyCFG


@dataclass
class PolicyPlannerCFG(PlannerCFG):
    """Settings for PolicyPlanner."""

    policy_cfg: "ManaPolicyCFG | None" = None


class PolicyPlanner(PlannerBase):
    """Delegate batched actions and resets to a ``ManaPolicy``."""

    CFG = PolicyPlannerCFG

    def __init__(self):
        super().__init__()
        self.policy = None
        self._done = False
        self._success = False

    def setup(self, env: Any, task: Any) -> bool:
        """Set up the policy from the pre-built ``policy_cfg``.

        ``policy_cfg`` is part of the normalized evaluation ``planner`` block;
        the rollout factory binds its embodiment before setup.
        """
        from manaenv.policy.mana_policy import create_policy

        settings: PolicyPlannerCFG = self.cfg
        policy_cfg = settings.policy_cfg

        if policy_cfg is None:
            raise RuntimeError(
                "PolicyPlanner requires policy_cfg from Translator. "
                "Ensure the rollout order has a valid policy entry "
                "(implementation + optional remote)."
            )

        self.policy = create_policy(policy_cfg)
        if int(getattr(env, "num_envs", 1)) > 1 and self.policy.requires_serial_envs():
            raise RuntimeError(
                "History-dependent Policy Space policies require num_parallel_envs: 1 "
                "so episode history cannot be shared across environments"
            )
        return True

    def requested_action_mode(self) -> str | None:
        """The policy owns the sim action layout; apply it before gym env build."""
        policy_cfg = getattr(self.cfg, "policy_cfg", None)
        if policy_cfg is None:
            return None
        if isinstance(policy_cfg, dict):
            mode = policy_cfg.get("action_mode")
            embodiment = policy_cfg.get("embodiment")
        else:
            mode = getattr(policy_cfg, "action_mode", None)
            embodiment = getattr(policy_cfg, "embodiment", None)
        if isinstance(mode, str) and mode.strip():
            return mode.strip()
        if isinstance(embodiment, dict):
            mode = embodiment.get("action_mode")
        else:
            mode = getattr(embodiment, "action_mode", None)
        return mode.strip() if isinstance(mode, str) and mode.strip() else None

    # ------------------------------------------------------------------
    #  Batch-first overrides — bypass per-instance loop
    # ------------------------------------------------------------------

    def get_actions(self, env: Any, obs: dict[str, Any]) -> torch.Tensor:
        """Delegate directly to ManaPolicy.get_actions. (N, act_dim)."""
        if self.policy is None:
            act_dim = env.action_manager.action.shape[-1]
            return torch.zeros(env.num_envs, act_dim, device=env.device)
        return self.policy.get_actions(env, obs)

    def on_reset(self, env: Any, obs: dict[str, Any], env_ids: torch.Tensor, env_id_to_success=None) -> None:
        if self.policy is not None:
            outcomes = None
            if env_id_to_success is not None:
                outcomes = {int(env_id): bool(success) for env_id, success in env_id_to_success.items()}
            self.policy.reset(env_ids, episode_outcomes=outcomes)

    def initial_plan(self, env: Any, settle_steps: int = 30) -> None:
        """Policy doesn't need settling."""

    def cleanup(self) -> None:
        if self.policy is not None:
            self.policy.cleanup()
        self.policy = None
