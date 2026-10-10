from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = True
    max_surface_range_tier: int = 0
    surface_range_expand_m: float = 0.08
    support_budget_floor: int = 256
    support_budget_ceiling: int = 20000

    @classmethod
    def from_dict(cls, data: dict | None) -> RecoveryConfig:
        payload = data if isinstance(data, dict) else {}
        return cls(
            enabled=bool(payload.get("enabled", True)),
            max_surface_range_tier=max(0, min(2, int(payload.get("max_surface_range_tier", 0) or 0))),
            surface_range_expand_m=max(0.0, float(payload.get("surface_range_expand_m", 0.08) or 0.0)),
            support_budget_floor=max(1, int(payload.get("support_budget_floor", 256) or 256)),
            support_budget_ceiling=max(1, int(payload.get("support_budget_ceiling", 20000) or 20000)),
        )


@dataclass(frozen=True)
class RecoveryAttempt:
    tier: int
    surface_range_tier: str
    surface_range_expand_m: float

    def to_json(self) -> dict:
        return {
            "tier": int(self.tier),
            "surface_range_tier": self.surface_range_tier,
            "surface_range_expand_m": float(self.surface_range_expand_m),
        }


def recovery_attempts(config: RecoveryConfig) -> list[RecoveryAttempt]:
    if not config.enabled:
        return [RecoveryAttempt(tier=0, surface_range_tier="strict", surface_range_expand_m=0.0)]
    tiers = [RecoveryAttempt(tier=0, surface_range_tier="strict", surface_range_expand_m=0.0)]
    if config.max_surface_range_tier >= 1:
        tiers.append(
            RecoveryAttempt(
                tier=1,
                surface_range_tier="expand",
                surface_range_expand_m=float(config.surface_range_expand_m),
            )
        )
    if config.max_surface_range_tier >= 2:
        tiers.append(
            RecoveryAttempt(
                tier=2,
                surface_range_tier="surface",
                surface_range_expand_m=float(config.surface_range_expand_m),
            )
        )
    return tiers
