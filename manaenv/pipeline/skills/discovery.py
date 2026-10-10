"""Skill CFG discovery for top-level pipeline sections."""

from __future__ import annotations

from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

SKILL_CONFIG_DOMAIN: str = "skill_config"


def discover_skill_config(symbol: str) -> ComponentSpec | None:
    """Resolve the top-level skill_config section."""
    if symbol != "SkillConfig":
        return None
    return ComponentSpec(
        module_path="manaenv.pipeline.skills.config",
        class_name="SkillConfig",
        symbol=symbol,
    )


register_provider(SKILL_CONFIG_DOMAIN, discover_skill_config)
