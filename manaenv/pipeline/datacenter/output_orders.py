"""Typed output orders for the evaluation-only pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OutputOrderCFG:
    """One benchmark report and optional episode-video sink declaration."""

    type: str = "benchmark"
    enabled: bool = True
    save_path: str | None = None
    format: str | None = None
    report_format: str | None = None
    record_video: bool = False

    def as_order_dict(self) -> dict[str, object]:
        values = {
            "type": self.type,
            "enabled": self.enabled,
            "save_path": self.save_path,
            "format": self.format,
            "report_format": self.report_format,
            "record_video": self.record_video,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass
class OutputOrdersCFG:
    """Typed ``output`` section."""

    orders: list[OutputOrderCFG] = field(default_factory=list)


__all__ = ["OutputOrderCFG", "OutputOrdersCFG"]
