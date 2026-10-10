"""DataBase - shared in-memory storage for pipeline products."""

from typing import TYPE_CHECKING, Any, Optional

from manaenv.pipeline.data_types import DataBucket

if TYPE_CHECKING:
    from isaaclab_arena.assets.registries import AssetRegistry


class DataBase:
    """Base storage and query layer for pipeline data.

    All product buckets are derived from :class:`DataBucket` — adding a new
    enum member automatically creates the corresponding storage container.

    Provides:
    - Enum-indexed product buckets (scenes, robots, tasks, …)
    - Export order CFG storage (configured by DataFormatter)
    - Unified asset_registry access for all pipeline components
    """

    def __init__(self):
        self._asset_registry: AssetRegistry | None = None
        self._order_cfgs: dict[DataBucket, Any] = {}
        self._buckets: dict[DataBucket, Any] = {}
        self.reset_storage()

    # ==================== Asset Registry ====================

    @property
    def asset_registry(self) -> Optional["AssetRegistry"]:
        """Global asset registry — single source of truth for all factories."""
        return self._asset_registry

    @asset_registry.setter
    def asset_registry(self, registry: "AssetRegistry") -> None:
        self._asset_registry = registry

    # ==================== Storage ====================

    def reset_storage(self) -> None:
        """Reset all product buckets. Container types derived from DataBucket."""
        self._buckets = {b: b.new_container() for b in DataBucket}

    def _resolve_bucket(self, bucket: DataBucket):
        """Return the underlying container for *bucket*."""
        return self._buckets[bucket]

    # ==================== Order CFGs ====================

    def get_order_cfg(self, bucket: DataBucket) -> Any:
        """Get export order CFG for a bucket, or None if not set."""
        return self._order_cfgs.get(bucket)

    def set_order_cfg(self, bucket: DataBucket, cfg) -> None:
        """Set export order CFG for a bucket."""
        self._order_cfgs[bucket] = cfg

    def pop_order_cfg(self, bucket: DataBucket) -> Any:
        """Remove and return export order CFG for a bucket (or None)."""
        return self._order_cfgs.pop(bucket, None)

    # ==================== Retrieval ====================

    def get(self, bucket: DataBucket, index: int | None = None) -> Any:
        """Unified retrieval interface using enum-indexed buckets.

        - ``index is None``: return the whole bucket.
        - ``index is int``: return one element for list buckets (or None if out of range).
        """
        target = self._resolve_bucket(bucket)
        if index is None:
            return target
        if isinstance(target, list) and 0 <= index < len(target):
            return target[index]
        return None

    def count(self, bucket: DataBucket) -> int:
        """Unified count interface using enum-indexed buckets."""
        return len(self._resolve_bucket(bucket))

    # ==================== Query helpers ====================

    def collect_available_assets(self) -> dict[str, dict[str, list[str]]]:
        """Enumerate all registered assets from AssetRegistry, categorised by tag."""
        if self._asset_registry is None:
            return {
                "objects": {"names": [], "tags": []},
                "scenes": {"names": [], "tags": []},
                "robots": {"names": [], "tags": []},
            }

        obj_names, scene_names, robot_names = [], [], []
        obj_tags, scene_tags, robot_tags = set(), set(), set()

        for name, comp in self._asset_registry._components.items():
            tags = comp.tags or []
            if "embodiment" in tags:
                robot_names.append(name)
                robot_tags.update(tags)
            elif "background" in tags:
                scene_names.append(name)
                scene_tags.update(tags)
            else:
                obj_names.append(name)
                obj_tags.update(tags)

        return {
            "objects": {"names": obj_names, "tags": sorted(obj_tags)},
            "scenes": {"names": scene_names, "tags": sorted(scene_tags)},
            "robots": {"names": robot_names, "tags": sorted(robot_tags)},
        }

    def collect_task_constraints(self) -> dict[str, Any]:
        """Aggregate scene/robot constraints from stored tasks."""
        merged_scene: dict[str, Any] = {}
        merged_robot: dict[str, Any] = {}

        for task in self.get(DataBucket.TASKS):
            sc = getattr(task, "scene_constraints", {}) or {}
            for role, c in sc.items():
                if role not in merged_scene:
                    merged_scene[role] = c

            rc = getattr(task, "robot_constraints", {}) or {}
            for role, c in rc.items():
                if role not in merged_robot:
                    merged_robot[role] = c

        return {"scene": merged_scene, "robot": merged_robot}
