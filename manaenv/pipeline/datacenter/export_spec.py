"""Public evaluation report and episode-video output configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from manaenv.pipeline.data_types import BenchmarkFormat, DataBucket, output_path


def _coerce_report_format(value: Any) -> BenchmarkFormat:
    if value is None:
        return BenchmarkFormat.JSON
    if isinstance(value, BenchmarkFormat):
        return value
    try:
        return BenchmarkFormat(str(value).strip().lower())
    except ValueError as exc:
        allowed = tuple(item.value for item in BenchmarkFormat)
        raise ValueError(f"report_format={value!r} not one of {allowed}") from exc


@dataclass
class BenchmarkExportCFG:
    """Destination and format for one evaluation report and its videos."""

    enabled: bool = True
    save_path: str | None = None
    report_format: BenchmarkFormat | str = BenchmarkFormat.JSON
    record_video: bool = False
    # ``format`` is accepted because older YAML reflection callers used it.
    format: BenchmarkFormat | str | None = None

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.save_path = str(self.save_path or output_path("benchmark"))
        self.report_format = _coerce_report_format(self.report_format if self.format is None else self.format)
        self.record_video = bool(self.record_video)


def resolve_benchmark_report_dir(data_center: Any) -> str | None:
    """Return the local report directory when the benchmark sink is enabled."""

    if data_center is None:
        return None
    cfg = data_center.get_order_cfg(DataBucket.BENCHMARKS)
    if cfg is None or not bool(getattr(cfg, "enabled", True)):
        return None
    save_path = getattr(cfg, "save_path", None)
    if not save_path:
        return None
    return os.path.expanduser(os.path.expandvars(str(save_path)))


__all__ = ["BenchmarkExportCFG", "resolve_benchmark_report_dir"]
