"""Safe opt-in feed-forward joint-effort overlay.

The overlay owns no action channels.  It is reasserted every physics step and
cleared on reset, so a controller cannot accidentally retain a stale effort.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def clamp_effort(effort_n: float, max_effort_n: float) -> float:
    """Clamp an effort to the configured symmetric safe bound."""
    limit = abs(float(max_effort_n))
    return max(-limit, min(limit, float(effort_n))) if limit > 0.0 else 0.0


def ramp_scale(elapsed_s: float, ramp_s: float) -> float:
    """Return a bounded linear ramp; non-positive duration is immediate."""
    return 1.0 if ramp_s <= 0.0 else max(0.0, min(1.0, float(elapsed_s) / float(ramp_s)))


class SafeEffortOverlay:
    """Ramped, clamped effort writer for named articulation joints."""

    def __init__(
        self,
        *,
        enabled: bool,
        joint_names: Sequence[str],
        effort_n: float,
        max_effort_n: float,
        ramp_s: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.joint_names = tuple(str(name) for name in joint_names if str(name))
        self.effort_n = float(effort_n)
        self.max_effort_n = float(max_effort_n)
        self.ramp_s = float(ramp_s)
        self._elapsed_s: torch.Tensor | None = None

    def bind(self, *, num_envs: int, device: torch.device | str) -> None:
        self._elapsed_s = torch.zeros(int(num_envs), dtype=torch.float32, device=device)

    def reset(self, robot: Any, env_ids: Sequence[int] | None = None) -> None:
        self._write(robot, 0.0, env_ids)
        if self._elapsed_s is not None:
            self._elapsed_s[self._rows(env_ids, device=self._elapsed_s.device)] = 0.0

    def apply(self, robot: Any, *, dt: float) -> None:
        if self._elapsed_s is None:
            self.bind(
                num_envs=int(robot.data.joint_effort_target.shape[0]), device=robot.data.joint_effort_target.device
            )
        if not self.enabled:
            self.reset(robot)
            return
        assert self._elapsed_s is not None
        self._elapsed_s += max(0.0, float(dt))
        scale = (
            torch.clamp(self._elapsed_s / self.ramp_s, 0.0, 1.0)
            if self.ramp_s > 0.0
            else torch.ones_like(self._elapsed_s)
        )
        limit = abs(self.max_effort_n)
        effort = (scale * self.effort_n).clamp(-limit, limit) if limit > 0.0 else torch.zeros_like(scale)
        self._write(robot, effort, None)

    def _rows(self, env_ids: Sequence[int] | None, *, device: torch.device) -> slice | torch.Tensor:
        return slice(None) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=device)

    def _joint_ids(self, robot: Any) -> list[int]:
        ids, _ = robot.find_joints(list(self.joint_names))
        return [int(index) for index in ids]

    def _write(self, robot: Any, effort: float | torch.Tensor, env_ids: Sequence[int] | None) -> None:
        ids = self._joint_ids(robot)
        if not ids:
            return
        target = robot.data.joint_effort_target.clone()
        rows = self._rows(env_ids, device=target.device)
        if isinstance(effort, torch.Tensor):
            target[:, ids] = effort.to(device=target.device, dtype=target.dtype).reshape(-1, 1)
        else:
            target[rows, ids] = float(effort)
        robot.set_joint_effort_target(target)
