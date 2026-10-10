"""Differential IK evaluated once per environment command."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.managers.action_manager import ActionTerm

_SUPPORTED_DIFF_IK_COMPUTE_MODES = {"command", "physics"}


def normalize_diff_ik_compute_mode(value: str) -> str:
    """Return a validated DiffIK evaluation cadence."""

    mode = str(value or "command").strip().lower()
    if mode not in _SUPPORTED_DIFF_IK_COMPUTE_MODES:
        supported = ", ".join(sorted(_SUPPORTED_DIFF_IK_COMPUTE_MODES))
        raise ValueError(f"Unsupported diff_ik_compute_mode {value!r}; expected one of: {supported}.")
    return mode


class CommandRateDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Compute DiffIK once per command and reuse its joint target between commands."""

    cfg: CommandRateDifferentialInverseKinematicsActionCfg

    def __init__(self, cfg: CommandRateDifferentialInverseKinematicsActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._cached_joint_pos_des: torch.Tensor | None = None
        self._ik_update_pending = True

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._ik_update_pending = True

    def apply_actions(self) -> None:
        if self._ik_update_pending or self._cached_joint_pos_des is None:
            super().apply_actions()
            self._cached_joint_pos_des = self._asset.data.joint_pos_target.torch[:, self._joint_ids].clone()
            self._ik_update_pending = False
            return

        self._asset.set_joint_position_target_index(
            target=self._cached_joint_pos_des,
            joint_ids=self._joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._ik_update_pending = True


@configclass
class CommandRateDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for command-rate differential inverse kinematics."""

    class_type: type[ActionTerm] = CommandRateDifferentialInverseKinematicsAction


def configure_diff_ik_action_cfg(
    action_cfg: object,
    compute_mode: str,
    *,
    use_warp: bool = False,
    use_cuda_graph: bool | None = None,
    device: str | None = None,
) -> int:
    """Apply a DiffIK cadence and implementation to every DiffIK action term."""

    mode = normalize_diff_ik_compute_mode(compute_mode)
    if use_warp:
        from manaenv.actions.warp_diff_ik import (
            WarpCommandRateDifferentialInverseKinematicsAction,
            WarpDifferentialInverseKinematicsAction,
        )

        class_type = (
            WarpCommandRateDifferentialInverseKinematicsAction
            if mode == "command"
            else WarpDifferentialInverseKinematicsAction
        )
    else:
        class_type = (
            CommandRateDifferentialInverseKinematicsAction if mode == "command" else DifferentialInverseKinematicsAction
        )
    configured = 0
    for term_cfg in vars(action_cfg).values() if action_cfg is not None else ():
        if isinstance(term_cfg, DifferentialInverseKinematicsActionCfg):
            term_cfg.class_type = class_type
            if use_warp:
                # IsaacLab's standard DiffIK cfg has no backend-specific graph
                # field. Keep the ManaEnv runtime choice on the term without
                # modifying the read-only third-party configclass.
                term_cfg._mana_use_cuda_graph = use_cuda_graph
                term_cfg._mana_device = device
            configured += 1
    return configured


__all__ = [
    "CommandRateDifferentialInverseKinematicsAction",
    "CommandRateDifferentialInverseKinematicsActionCfg",
    "configure_diff_ik_action_cfg",
    "normalize_diff_ik_compute_mode",
]
