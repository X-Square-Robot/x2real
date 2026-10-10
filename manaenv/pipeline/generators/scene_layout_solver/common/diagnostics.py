from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LayoutDiagnostic:
    code: str
    message: str
    target: str = ""
    severity: str = "error"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "target": self.target,
            "severity": self.severity,
            "meta": dict(self.meta),
        }


class LayoutSpecError(ValueError):
    def __init__(self, diagnostic: LayoutDiagnostic):
        parts = [diagnostic.code]
        if diagnostic.message:
            parts.append(diagnostic.message)
        if diagnostic.target:
            parts.append(f"target={diagnostic.target}")
        if diagnostic.meta:
            detail = ", ".join(f"{key}={value!r}" for key, value in sorted(diagnostic.meta.items()))
            if detail:
                parts.append(detail)
        super().__init__(": ".join(parts))
        self.diagnostic = diagnostic


def error(code: str, message: str, *, target: str = "", **meta: Any) -> LayoutSpecError:
    return LayoutSpecError(LayoutDiagnostic(code=code, message=message, target=target, meta=dict(meta)))
