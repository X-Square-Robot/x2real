from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..geometry import Bounds2D, Range1D

CellAcceptFn = Callable[[str, Range1D, Range1D], bool]
CellClipFn = Callable[[str, Range1D, Range1D], tuple[Range1D, Range1D] | None]


@dataclass(frozen=True)
class ObjectProfile:
    name: str
    footprint: tuple[float, float]
    footprint_bounds: Bounds2D
    z: tuple[float, float]

    @property
    def area(self) -> float:
        return max(0.0, self.footprint[0]) * max(0.0, self.footprint[1])

    def swept_x(self, cell: PartitionCell) -> Range1D:
        return Range1D(cell.x.min + self.footprint_bounds.x.min, cell.x.max + self.footprint_bounds.x.max)

    def swept_y(self, cell: PartitionCell) -> Range1D:
        return Range1D(cell.y.min + self.footprint_bounds.y.min, cell.y.max + self.footprint_bounds.y.max)


@dataclass(frozen=True)
class PartitionProblem:
    names: tuple[str, ...]
    bounds: Bounds2D
    profiles: dict[str, ObjectProfile]
    min_spacing: float = 0.0
    spacing_mode: str = "footprint_gap"
    jitter_ratio: float = 0.0
    sample_method: str = "random"
    grid_step: float | None = None
    grid_snap: dict | None = None
    align_axis: str | None = None
    order_axis: str | None = None
    boundary_spacing: float | None = None
    accept_cell: CellAcceptFn | None = None
    clip_cell: CellClipFn | None = None
    obstacles: tuple[PartitionObstacle, ...] = ()

    @property
    def count(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class PartitionCell:
    object_name: str
    x: Range1D
    y: Range1D
    z: tuple[float, float]

    @property
    def center(self) -> tuple[float, float]:
        return (self.x.center, self.y.center)

    @property
    def area(self) -> float:
        return max(0.0, self.x.length) * max(0.0, self.y.length)

    @property
    def short_edge(self) -> float:
        return min(max(0.0, self.x.length), max(0.0, self.y.length))


@dataclass(frozen=True)
class PartitionCase:
    cells: dict[str, PartitionCell]
    tier: str
    topology: str
    quality: float = 0.0
    feature: dict[str, tuple[float, float]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def with_score(self, quality: float, metrics: dict[str, float]) -> PartitionCase:
        return PartitionCase(
            cells=dict(self.cells),
            tier=self.tier,
            topology=self.topology,
            quality=float(quality),
            feature={name: cell.center for name, cell in self.cells.items()},
            metrics=dict(metrics),
        )


@dataclass(frozen=True)
class StrategyPolicy:
    strategies: tuple[str, ...]
    relation_kind: str = "region_sample"


@dataclass(frozen=True)
class PartitionObstacle:
    obstacle_id: str
    x: Range1D
    y: Range1D
    source_object: str = ""
    source_domain: str = ""
