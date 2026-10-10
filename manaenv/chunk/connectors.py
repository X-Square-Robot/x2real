"""Chunk-boundary connection helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch


def blend_chunk_start_to_last_command(
    chunk: torch.Tensor,
    *,
    last_command: torch.Tensor | None,
    slot_layout: Mapping[str, Mapping[str, Sequence[int]]],
) -> torch.Tensor:
    """Blend each position slot in a new chunk to the last executed command.

    The first row is anchored exactly at ``last_command`` and the correction
    decays linearly to zero at the final row. Orientation and non-position
    slots are left unchanged.
    """

    if chunk.ndim != 2 or int(chunk.shape[0]) == 0 or last_command is None:
        return chunk

    pos_slices = list(
        _iter_position_slices(
            slot_layout,
            action_dim=int(chunk.shape[1]),
        )
    )
    if not pos_slices:
        return chunk

    out = chunk.clone()
    anchor = (
        last_command.detach()
        .reshape(-1)
        .to(
            device=out.device,
            dtype=out.dtype,
        )
    )
    weights = torch.linspace(
        1.0,
        0.0,
        int(out.shape[0]),
        dtype=out.dtype,
        device=out.device,
    )

    for pos_slice in pos_slices:
        anchor_pos = anchor[pos_slice]
        offset = anchor_pos - out[0, pos_slice]
        out[:, pos_slice] = out[:, pos_slice] + weights[:, None] * offset

    return out


def _iter_position_slices(
    slot_layout: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    action_dim: int,
) -> Iterable[slice]:
    for slots in slot_layout.values():
        # EE policies expose Cartesian ``pos`` slots; absolute-joint policies
        # expose one ``joint_pos`` slice. Both are continuous position targets
        # and benefit from the same boundary correction.
        pos = slots.get("pos", slots.get("joint_pos"))
        if pos is None:
            continue
        yield _parse_slice(pos, action_dim=action_dim)


def _parse_slice(value: Sequence[int], *, action_dim: int) -> slice:
    if len(value) != 2:
        raise ValueError(f"slot_layout pos slice must have two entries, got {value!r}")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end <= start or end > action_dim:
        raise ValueError(f"slot_layout pos slice [{start}, {end}] is outside " f"action_dim={action_dim}")
    return slice(start, end)
