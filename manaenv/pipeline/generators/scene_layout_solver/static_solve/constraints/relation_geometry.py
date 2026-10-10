from __future__ import annotations

from dataclasses import dataclass

from ...assets.bridge import LayoutAssetRecord
from ...common.geometry import Bounds2D, ManualSupportSurface, Pose, Range1D, projected_footprint_bounds


@dataclass(frozen=True)
class FootprintRanges:
    x: Range1D
    y: Range1D

    def axis(self, axis: str) -> Range1D:
        return self.x if str(axis) == "x" else self.y


def footprint_ranges_on_surface(
    *,
    record: LayoutAssetRecord,
    pose: Pose,
    surface: ManualSupportSurface,
) -> FootprintRanges:
    """Return occupied support-local x/y ranges for an object pose.

    The pose position is the authored object origin, not necessarily the center
    of the support footprint. The projected footprint bounds are therefore
    origin-relative and must be offset by the origin's support-local position.
    """

    origin_x, origin_y = surface.world_to_local_xy(pose.position)
    footprint = projected_footprint_bounds(record.local_bounds, pose.quaternion, surface.x_axis, surface.y_axis)
    return FootprintRanges(
        x=Range1D(float(origin_x) + footprint.x.min, float(origin_x) + footprint.x.max),
        y=Range1D(float(origin_y) + footprint.y.min, float(origin_y) + footprint.y.max),
    )


def footprint_axis_span(
    *,
    record: LayoutAssetRecord,
    pose: Pose,
    surface: ManualSupportSurface,
    axis: str,
) -> Range1D:
    return footprint_ranges_on_surface(record=record, pose=pose, surface=surface).axis(axis)


def footprint_gap(left: Range1D, right: Range1D) -> float:
    """Return positive gap between 1D ranges, or negative overlap depth."""

    return max(float(right.min) - float(left.max), float(left.min) - float(right.max))


def footprint_ranges_separated(
    left: FootprintRanges,
    right: FootprintRanges,
    *,
    min_gap: float,
) -> bool:
    gap = float(min_gap)
    return footprint_gap(left.x, right.x) >= gap or footprint_gap(left.y, right.y) >= gap


def bounds2d_from_ranges(ranges: FootprintRanges) -> Bounds2D:
    return Bounds2D(ranges.x, ranges.y)
