"""Chunk postprocessors — CFG-reflected plugins listed by the
``client.chunk_postprocessors`` segment of a policy yaml (PLAN.md §10.5).

Each entry is an explicit CFG key, e.g.
``chunk_postprocessors: [{temporal_interpolate: {multiplier: 1.5}}]``,
reflected through the global lazy registry. ``ManaPolicy`` instantiates
``cfg.class_type()``, calls ``load(cfg)``, and applies the callable
instances in declaration order to the chunk produced by
:meth:`PolicyImplementationBase.decode_response`, **before** the chunk
is written into the per-env :class:`ChunkBuffer`.

The two shipped postprocessors cover every chunk shaping needed in
tree:

  * :class:`TemporalInterpolate` — upsample the chunk along time by
    a multiplier (may be non-integer, e.g. 1.5). Used by wallx with
    absolute EE keyframes when interpolation multiplier > 1.
  * :class:`HorizonClip` — clip the chunk to a maximum number of rows.
    Replaces the per-impl ``action_horizon`` clipping that lived inside
    ``OpenpiImpl._parse_response``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from manaenv.utils.tensor_utils import upsample_temporal_rows_numpy


@dataclass
class ChunkPostprocessorBaseCFG:
    """Base CFG for chunk postprocessor plugins."""

    class_type: type | None = None


class ChunkPostprocessorBase(ABC):
    """Runtime base for chunk postprocessors."""

    def __init__(self) -> None:
        self.cfg: ChunkPostprocessorBaseCFG | None = None

    def load(self, cfg: ChunkPostprocessorBaseCFG) -> None:
        self.cfg = cfg

    @abstractmethod
    def __call__(self, chunk: torch.Tensor) -> torch.Tensor:
        """Return the reshaped chunk."""


class TemporalInterpolate(ChunkPostprocessorBase):
    """Upsample the time dimension by ``multiplier``.

    ``multiplier`` can be a non-integer (e.g. 1.5); the target row count is
    ``round(T * multiplier)``. Last-dim 7/14 uses Euler NLERP; 8/16 uses
    XYZW NLERP. Other widths stay component-linear.

    Input/Output shapes: ``(T, action_dim)`` or ``(num_envs, T, action_dim)``.
    """

    def load(self, cfg: ChunkPostprocessorBaseCFG) -> None:
        super().load(cfg)
        multiplier = float(getattr(cfg, "multiplier", 1.0))
        if multiplier < 1.0:
            raise ValueError(f"temporal_interpolate: multiplier must be >= 1, got {multiplier}")
        self._multiplier = multiplier

    def __call__(self, chunk: torch.Tensor) -> torch.Tensor:
        multiplier = self._multiplier
        if multiplier == 1:
            return chunk
        device = chunk.device
        np_chunk = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
        if np_chunk.ndim == 2:
            up = upsample_temporal_rows_numpy(np_chunk, multiplier)
        elif np_chunk.ndim == 3:
            up = np.stack(
                [upsample_temporal_rows_numpy(np_chunk[i], multiplier) for i in range(np_chunk.shape[0])],
                axis=0,
            )
        else:
            raise ValueError(f"temporal_interpolate: chunk must be (T, D) or (N, T, D), got shape={tuple(chunk.shape)}")
        return torch.as_tensor(up, dtype=chunk.dtype, device=device)


@dataclass
class TemporalInterpolateCFG(ChunkPostprocessorBaseCFG):
    multiplier: float = 1.0
    class_type: type = TemporalInterpolate


class HorizonClip(ChunkPostprocessorBase):
    """Clip the chunk along the time axis to at most ``horizon`` rows.

    Input/Output shapes: ``(T, action_dim)`` or ``(num_envs, T, action_dim)``.
    """

    def load(self, cfg: ChunkPostprocessorBaseCFG) -> None:
        super().load(cfg)
        horizon = int(getattr(cfg, "horizon", 1))
        if horizon < 1:
            raise ValueError(f"horizon_clip: horizon must be >= 1, got {horizon}")
        self._horizon = horizon

    def __call__(self, chunk: torch.Tensor) -> torch.Tensor:
        horizon = self._horizon
        if chunk.ndim == 2:
            return chunk[: min(int(chunk.shape[0]), horizon)]
        if chunk.ndim == 3:
            return chunk[:, : min(int(chunk.shape[1]), horizon)]
        raise ValueError(f"horizon_clip: chunk must be (T, D) or (N, T, D), got shape={tuple(chunk.shape)}")


@dataclass
class HorizonClipCFG(ChunkPostprocessorBaseCFG):
    horizon: int = 1
    class_type: type = HorizonClip
