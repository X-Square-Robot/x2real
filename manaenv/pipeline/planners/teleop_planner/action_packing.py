"""Pure action-layout helpers shared by Vuer active and hold paths."""

from __future__ import annotations

import torch


def bimanual_pose_parts(
    left_pos: torch.Tensor,
    left_quat: torch.Tensor,
    left_gripper: torch.Tensor,
    right_pos: torch.Tensor,
    right_quat: torch.Tensor,
    right_gripper: torch.Tensor,
    *,
    interleaved_grippers: bool,
) -> tuple[torch.Tensor, ...]:
    """Order bimanual pose tensors for EX001 poses-first or CX002 interleaved layouts."""
    if interleaved_grippers:
        return left_pos, left_quat, left_gripper, right_pos, right_quat, right_gripper
    return left_pos, left_quat, right_pos, right_quat, left_gripper, right_gripper
