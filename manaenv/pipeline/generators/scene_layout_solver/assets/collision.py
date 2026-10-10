from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from ..common.constants import GEOM_EPS
from ..common.geometry import Bounds3D, Pose, bounds_overlap, union_aabb, world_aabb_for_pose


class CollisionProxy(Protocol):
    def local_aabb(self) -> Bounds3D:
        raise NotImplementedError

    def world_aabb(self, pose: Pose) -> Bounds3D:
        raise NotImplementedError

    def narrow_boxes(self) -> list[Bounds3D]:
        raise NotImplementedError

    @property
    def has_narrow_proxy(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class AabbProxy:
    bounds: Bounds3D

    def local_aabb(self) -> Bounds3D:
        return self.bounds

    def world_aabb(self, pose: Pose) -> Bounds3D:
        return world_aabb_for_pose(self.bounds, pose)

    def narrow_boxes(self) -> list[Bounds3D]:
        return [self.bounds]

    @property
    def has_narrow_proxy(self) -> bool:
        return False


@dataclass(frozen=True)
class MultiBoxProxy:
    boxes: list[Bounds3D]

    def local_aabb(self) -> Bounds3D:
        return union_aabb(self.boxes)

    def world_aabb(self, pose: Pose) -> Bounds3D:
        return union_aabb(world_aabb_for_pose(box, pose) for box in self.boxes)

    def narrow_boxes(self) -> list[Bounds3D]:
        return list(self.boxes)

    @property
    def has_narrow_proxy(self) -> bool:
        return True


@dataclass(frozen=True)
class CollisionObject:
    name: str
    proxy: CollisionProxy
    pose: Pose


def proxy_from_geometry(local_bounds: Bounds3D, boxes: Iterable[Bounds3D] | None = None) -> CollisionProxy:
    box_list = list(boxes or [])
    if box_list:
        return MultiBoxProxy(box_list)
    return AabbProxy(local_bounds)


def objects_collide(a: CollisionObject, b: CollisionObject, *, eps: float = GEOM_EPS, mode: str = "multi_box") -> bool:
    if not bounds_overlap(a.proxy.world_aabb(a.pose), b.proxy.world_aabb(b.pose), eps=eps):
        return False
    if mode == "aabb":
        return True
    if a.proxy.has_narrow_proxy or b.proxy.has_narrow_proxy:
        a_boxes = [world_aabb_for_pose(box, a.pose) for box in a.proxy.narrow_boxes()]
        b_boxes = [world_aabb_for_pose(box, b.pose) for box in b.proxy.narrow_boxes()]
        for a_box in a_boxes:
            for b_box in b_boxes:
                if bounds_overlap(a_box, b_box, eps=eps):
                    return True
        return False
    return True
