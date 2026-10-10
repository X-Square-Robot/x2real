from __future__ import annotations

import random
from itertools import combinations

from ...candidate_grid import rectangular_grid
from ...constants import GEOM_EPS
from ...geometry import Bounds2D, Range1D
from ..geometry import accepted_cell, case_is_safe
from ..models import PartitionCase, PartitionCell, PartitionProblem
from ..policy import PERMUTED_UNIFORM_GRID, UNIFORM_GRID, WEIGHTED_GRID
from .common import (
    allocate_lengths,
    boundaries,
    cells_from_named_raw,
    cells_from_raw_assignment,
    select_cells,
    select_coords,
    strategy_dims,
)


def uniform_grid_cases(
    problem: PartitionProblem,
    *,
    rng: random.Random,
    budget: int,
    tier: str = UNIFORM_GRID,
    permute: bool = False,
    strip_only: bool = False,
) -> list[PartitionCase]:
    if strip_only and (problem.align_axis or problem.order_axis or problem.grid_step is not None):
        constrained = constrained_line_grid_cases(problem, rng=rng, budget=budget, tier=tier)
        if constrained:
            return constrained
    dims = strategy_dims(problem.count, strip_only=strip_only)
    out: list[PartitionCase] = []
    attempts = 0
    max_attempts = max(64, budget * 16)
    while len(out) < budget and attempts < max_attempts:
        attempts += 1
        cols, rows = dims[attempts % len(dims)]
        raw_cells = rectangular_grid(problem.bounds, cols=cols, rows=rows)
        if len(raw_cells) < problem.count:
            continue
        selected_cells = select_cells(raw_cells, problem.count, rng=rng)
        selected_cells = _jittered_grid_cells(problem, selected_cells, rng=rng)
        names = list(problem.names)
        if permute:
            rng.shuffle(names)
        cells = cells_from_raw_assignment(problem, names, selected_cells)
        if cells is None:
            continue
        out.append(PartitionCase(cells=cells, tier=tier, topology=f"{cols}x{rows}"))
    return out


def _jittered_grid_cells(problem: PartitionProblem, cells: list[Bounds2D], *, rng: random.Random) -> list[Bounds2D]:
    method = str(problem.sample_method or "").strip().lower()
    if method != "jittered_grid" or problem.jitter_ratio <= 0.0:
        return cells
    ratio = max(0.0, min(0.95, float(problem.jitter_ratio)))
    out: list[Bounds2D] = []
    for cell in cells:
        x_len = cell.x.length * (1.0 - ratio)
        y_len = cell.y.length * (1.0 - ratio)
        x_slack = max(0.0, cell.x.length - x_len) * 0.5
        y_slack = max(0.0, cell.y.length - y_len) * 0.5
        cx = cell.x.center + rng.uniform(-x_slack, x_slack)
        cy = cell.y.center + rng.uniform(-y_slack, y_slack)
        out.append(
            Bounds2D(
                Range1D(cx - x_len * 0.5, cx + x_len * 0.5),
                Range1D(cy - y_len * 0.5, cy + y_len * 0.5),
            )
        )
    return out


def permuted_uniform_grid_cases(problem: PartitionProblem, *, rng: random.Random, budget: int) -> list[PartitionCase]:
    return uniform_grid_cases(
        problem, rng=rng, budget=budget, tier=PERMUTED_UNIFORM_GRID, permute=True, strip_only=False
    )


def constrained_line_grid_cases(
    problem: PartitionProblem, *, rng: random.Random, budget: int, tier: str
) -> list[PartitionCase]:
    """Generate runtime-safe line/order cases without weakening align/order semantics.

    Cells are point center domains on a deterministic grid. This keeps reset-time
    sampling cheap while making inherited align/order constraints explicit in
    the compiled case pool.
    """

    names = list(problem.names)
    if not names:
        return []
    align_axis = problem.align_axis if problem.align_axis in {"x", "y"} else None
    order_axis = problem.order_axis if problem.order_axis in {"x", "y"} else None
    primary_axis = order_axis or ("y" if align_axis == "x" else "x")
    secondary_axis = align_axis or ("x" if primary_axis == "y" else "y")
    primary_values = _axis_grid_values(problem, primary_axis)
    secondary_values = _axis_grid_values(problem, secondary_axis)
    if len(primary_values) < len(names) or not secondary_values:
        return []

    out: list[PartitionCase] = []
    primary_sequences = list(combinations(primary_values, len(names)))
    if not primary_sequences:
        return []
    rng.shuffle(primary_sequences)
    secondary_sample = list(secondary_values)
    rng.shuffle(secondary_sample)
    max_attempts = max(budget * 8, 64)
    attempts = 0
    for secondary in secondary_sample:
        for primary_seq in primary_sequences:
            attempts += 1
            if attempts > max_attempts or len(out) >= budget:
                return out
            raw_cells: dict[str, PartitionCell] = {}
            for name, primary in zip(names, primary_seq):
                x = secondary if primary_axis == "y" else primary
                y = primary if primary_axis == "y" else secondary
                profile = problem.profiles[name]
                cell = accepted_cell(problem, PartitionCell(name, Range1D(x, x), Range1D(y, y), profile.z))
                if cell is None:
                    raw_cells = {}
                    break
                raw_cells[name] = cell
            if raw_cells and case_is_safe(problem, raw_cells):
                out.append(
                    PartitionCase(cells=raw_cells, tier=tier, topology=f"line_{primary_axis}_aligned_{secondary_axis}")
                )
    return out


def _axis_grid_values(problem: PartitionProblem, axis: str) -> list[float]:
    bounds = problem.bounds.x if axis == "x" else problem.bounds.y
    if bounds.length <= GEOM_EPS:
        return [bounds.center]
    step = float(problem.grid_step or 0.0)
    if step <= GEOM_EPS:
        count = max(problem.count + 2, 4)
        step = bounds.length / float(count - 1)
    origin = _grid_origin(problem, axis)
    start_index = int((bounds.min - origin) // step) - 1
    values: list[float] = []
    index = start_index
    while origin + step * index <= bounds.max + step + GEOM_EPS:
        value = origin + step * index
        if bounds.min - GEOM_EPS <= value <= bounds.max + GEOM_EPS:
            values.append(0.0 if abs(value) < GEOM_EPS else value)
        index += 1
    return values or [bounds.center]


def _grid_origin(problem: PartitionProblem, axis: str) -> float:
    snap = problem.grid_snap or {}
    origin = snap.get("origin") if isinstance(snap, dict) else None
    if isinstance(origin, (list, tuple)) and len(origin) >= 2:
        return float(origin[0 if axis == "x" else 1])
    if isinstance(origin, (int, float)):
        return float(origin)
    return 0.0


def weighted_grid_cases(problem: PartitionProblem, *, rng: random.Random, budget: int) -> list[PartitionCase]:
    dims = [dim for dim in strategy_dims(problem.count, strip_only=False) if min(dim) > 1]
    if not dims:
        dims = strategy_dims(problem.count, strip_only=False)
    out: list[PartitionCase] = []
    attempts = 0
    max_attempts = max(96, budget * 20)
    while len(out) < budget and attempts < max_attempts:
        attempts += 1
        cols, rows = dims[attempts % len(dims)]
        coords = [(ix, iy) for ix in range(cols) for iy in range(rows)]
        if len(coords) < problem.count:
            continue
        selected = select_coords(coords, problem.count, rng=rng, require_full_axes=True)
        if len(selected) < problem.count:
            continue
        names = sorted(problem.names, key=lambda name: (-problem.profiles[name].area, name))
        if attempts % 3:
            rng.shuffle(names)
        raw_cells = _weighted_grid_raw_cells(problem, dict(zip(names, selected)), cols=cols, rows=rows, rng=rng)
        cells = cells_from_named_raw(problem, raw_cells)
        if cells is None:
            continue
        out.append(PartitionCase(cells=cells, tier=WEIGHTED_GRID, topology=f"{cols}x{rows}"))
    return out


def _weighted_grid_raw_cells(
    problem: PartitionProblem,
    assignment: dict[str, tuple[int, int]],
    *,
    cols: int,
    rows: int,
    rng: random.Random,
) -> dict[str, Bounds2D]:
    col_weights = [0.0 for _ in range(cols)]
    row_weights = [0.0 for _ in range(rows)]
    min_size = max(0.0, problem.min_spacing)
    for name, (ix, iy) in assignment.items():
        fp = problem.profiles[name].footprint
        col_weights[ix] = max(col_weights[ix], fp[0] + min_size)
        row_weights[iy] = max(row_weights[iy], fp[1] + min_size)
    col_weights = [
        value if value > GEOM_EPS else min(col_weights) if any(col_weights) else 1.0 for value in col_weights
    ]
    row_weights = [
        value if value > GEOM_EPS else min(row_weights) if any(row_weights) else 1.0 for value in row_weights
    ]
    col_widths = allocate_lengths(problem.bounds.x.length, col_weights, rng=rng)
    row_heights = allocate_lengths(problem.bounds.y.length, row_weights, rng=rng)
    xs = boundaries(problem.bounds.x.min, col_widths)
    ys = boundaries(problem.bounds.y.min, row_heights)
    return {
        name: Bounds2D(Range1D(xs[ix], xs[ix + 1]), Range1D(ys[iy], ys[iy + 1]))
        for name, (ix, iy) in assignment.items()
    }
