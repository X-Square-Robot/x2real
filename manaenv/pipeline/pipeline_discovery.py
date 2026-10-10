"""Pipeline-level CFG discovery for the global lazy registry."""

from __future__ import annotations

from manaenv.pipeline.registry.global_provider import register_provider
from manaenv.pipeline.registry.lazy import ComponentSpec

PIPELINE_DOMAIN: str = "pipeline"
PIPELINE_SELECTION_DOMAIN: str = "pipeline_selection"

_PIPELINE_SECTION_SPECS: dict[str, tuple[str, str]] = {
    "PipelineSelection": ("manaenv.pipeline.pipeline_cfg", "PipelineSelection"),
    "PipelineDag": ("manaenv.pipeline.pipeline_cfg", "PipelineDag"),
    "PipelineDagNode": ("manaenv.pipeline.pipeline_cfg", "PipelineDagNode"),
}


def discover_pipeline(symbol: str) -> ComponentSpec | None:
    """Resolve the top-level PipelineCFG for ``pipeline:`` YAML reflection."""
    if symbol != "Pipeline":
        return None
    return ComponentSpec(
        module_path="manaenv.pipeline.pipeline_cfg",
        class_name="Pipeline",
        symbol=symbol,
    )


def discover_pipeline_sections(symbol: str) -> ComponentSpec | None:
    """Resolve pipeline orchestration CFG sections (top-level and nested)."""
    spec = _PIPELINE_SECTION_SPECS.get(symbol)
    if spec is None:
        return None
    module_path, class_name = spec
    return ComponentSpec(
        module_path=module_path,
        class_name=class_name,
        symbol=symbol,
    )


register_provider(PIPELINE_DOMAIN, discover_pipeline)
register_provider(PIPELINE_SELECTION_DOMAIN, discover_pipeline_sections)
