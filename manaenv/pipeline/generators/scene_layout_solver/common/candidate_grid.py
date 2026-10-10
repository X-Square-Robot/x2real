from __future__ import annotations

import math

from .constants import GEOM_EPS
from .geometry import Bounds2D, Range1D


def adaptive_cell_step(
    bounds: Bounds2D, footprint: tuple[float, float], *, min_step: float = 0.005, max_step: float = 0.05
) -> float:
    spans = [value for value in (bounds.x.length, bounds.y.length) if value > GEOM_EPS]
    fps = [value for value in footprint if value > GEOM_EPS]
    if not spans:
        return min_step
    span_step = min(spans) / 12.0
    fp_step = min(fps) if fps else span_step
    return max(min_step, min(max_step, span_step, fp_step))


def axis_cells(axis: Range1D, step: float) -> list[Range1D]:
    if axis.length <= GEOM_EPS:
        return [axis]
    out: list[Range1D] = []
    pos = axis.min
    step = max(1e-6, float(step))
    while pos < axis.max - 1e-12:
        nxt = min(axis.max, pos + step)
        out.append(Range1D(pos, nxt))
        pos = nxt
    return out or [axis]


def grid_dims(count: int) -> list[tuple[int, int]]:
    base_cols = max(1, int(math.ceil(math.sqrt(count))))
    base_rows = max(1, int(math.ceil(count / base_cols)))
    raw = [
        (1, count),
        (count, 1),
        (base_cols, base_rows),
        (base_rows, base_cols),
        (base_cols + 1, base_rows),
        (base_cols, base_rows + 1),
        (base_cols + 1, base_rows + 1),
    ]
    out: list[tuple[int, int]] = []
    for cols, rows in raw:
        if cols * rows < count:
            continue
        item = (cols, rows)
        if item not in out:
            out.append(item)
    return out or [(max(1, count), 1)]


def rectangular_grid(bounds: Bounds2D, *, cols: int, rows: int) -> list[Bounds2D]:
    cells: list[Bounds2D] = []
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    sx = bounds.x.length / cols
    sy = bounds.y.length / rows
    for ix in range(cols):
        for iy in range(rows):
            cells.append(
                Bounds2D(
                    Range1D(bounds.x.min + sx * ix, bounds.x.min + sx * (ix + 1)),
                    Range1D(bounds.y.min + sy * iy, bounds.y.min + sy * (iy + 1)),
                )
            )
    return cells
