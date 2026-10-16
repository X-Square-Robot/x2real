"""In-memory products and artifact locations for the public pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from manaenv.pipeline.data_types import DataBucket
from manaenv.pipeline.datacenter.database import DataBase
from manaenv.pipeline.datacenter.export_spec import BenchmarkExportCFG, DatasetExportCFG
from manaenv.pipeline.datacenter.output_orders import OutputOrderCFG, OutputOrdersCFG
from manaenv.utils.debug_print import manaprint


@dataclass
class DataCenterCFG:
    class_type: Any = None
    type: str = "default_data_center"

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = DataCenter


class DataCenter(DataBase):
    """Central storage for generated products and evaluation summaries.

    No dataset cleaner, monitor, or external asset IO is attached to this
    object. USD/asset resolution happens in the local asset registry; the
    rollout factory writes evaluation reports and optional episode videos.
    """

    def __init__(self) -> None:
        super().__init__()

    def setup(self, cfg: Any = None) -> None:
        self.reset_storage()
        self._order_cfgs.clear()
        manaprint("INFO: Evaluation DataCenter initialized")

    def receive_orders(self, orders: OutputOrdersCFG | Iterable[OutputOrderCFG] | None) -> None:
        if orders is None:
            return
        if isinstance(orders, OutputOrdersCFG):
            entries = orders.orders
        elif isinstance(orders, dict):
            entries = [OutputOrderCFG(**orders)]
        else:
            entries = list(orders)
        for raw in entries:
            if isinstance(raw, dict):
                raw = OutputOrderCFG(**raw)
            if not isinstance(raw, OutputOrderCFG):
                raise TypeError(f"unsupported output order: {type(raw).__name__}")
            kind = str(raw.type or "benchmark").strip().lower()
            if kind == "datagen":
                self.set_order_cfg(
                    DataBucket.DATASETS,
                    DatasetExportCFG(enabled=raw.enabled, save_path=raw.save_path, format=raw.format or "hdf5"),
                )
                continue
            if kind not in {"benchmark", "evaluation", "report"}:
                raise ValueError(f"unsupported public output type: {raw.type!r}")
            cfg = BenchmarkExportCFG(
                enabled=raw.enabled,
                save_path=raw.save_path,
                report_format=raw.report_format or raw.format,
                record_video=raw.record_video,
            )
            self.set_order_cfg(DataBucket.BENCHMARKS, cfg)

    # Product storage -------------------------------------------------
    def _store(self, bucket: DataBucket, values: Iterable[Any] | None) -> None:
        target = self.get(bucket)
        if values is not None:
            for value in values:
                if not any(existing is value for existing in target):
                    target.append(value)

    def store_scenes(self, scenes: List[Any]) -> None:
        self._store(DataBucket.SCENES, scenes)

    def store_robots(self, robots: List[Any]) -> None:
        self._store(DataBucket.ROBOTS, robots)

    def store_tasks(self, tasks: List[Any]) -> None:
        self._store(DataBucket.TASKS, tasks)

    def store_environments(self, environments: List[Any]) -> None:
        self._store(DataBucket.ENVIRONMENTS, environments)

    def store_policy_cfgs(self, policy_cfgs: List[Any]) -> None:
        self._store(DataBucket.POLICY_CFGS, policy_cfgs)

    def store_policies(self, policies: List[Any]) -> None:
        self._store(DataBucket.POLICIES, policies)

    def store_benchmark_results(self, results: List[Any]) -> None:
        target = self.get(DataBucket.BENCHMARKS)
        target.clear()
        target.extend(results)

    # Report sink -----------------------------------------------------
    def save(self, buckets: list[DataBucket] | None = None, prefix: str | None = None) -> Dict[str, List[str]]:
        """Return report and dataset paths already produced by the runners."""
        selected = buckets or [DataBucket.BENCHMARKS]
        if buckets is None and self.get(DataBucket.DATASETS):
            selected = [*selected, DataBucket.DATASETS]
        paths: list[str] = []
        for item in self.get(DataBucket.BENCHMARKS):
            path = item.get("evaluation_report_path") if isinstance(item, dict) else None
            if path:
                paths.append(str(path))
        result = {bucket.value: paths for bucket in selected if bucket is DataBucket.BENCHMARKS}
        if DataBucket.DATASETS in selected:
            result[DataBucket.DATASETS.value] = [
                str(artifact["path"])
                for item in self.get(DataBucket.DATASETS)
                for artifact in item.get("datasets", [])
                if artifact.get("path")
            ]
        return result

    def clear(self) -> None:
        self.reset_storage()

    def cleanup(self) -> None:
        self.clear()
        self._order_cfgs.clear()


__all__ = ["DataCenter", "DataCenterCFG"]
