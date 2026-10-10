"""CONSTRAINT scene generator workflow."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any

from manaenv.pipeline.generators.scene_generator import SceneGenerationCompilation, SceneGenerator, SceneGeneratorCFG
from manaenv.pipeline.registry.global_provider import global_registry


@dataclass
class ConstraintSceneGenerationCompilation(SceneGenerationCompilation):
    """CONSTRAINT scene compilation with the resolved template cfg."""

    template_scene_cfg: object | None = None


class ConstraintSceneGenerator(SceneGenerator):
    """Template-driven scene generator that reuses the base component flow."""

    def compile(self, **kwargs: object) -> ConstraintSceneGenerationCompilation:
        cfg: ConstraintSceneGeneratorCFG = self.cfg
        if not cfg.template:
            raise ValueError(
                f"{type(self).__name__} requires a non-empty `template` field; "
                "use SceneGenerator for component-only scene workflows."
            )
        compiled = super().compile(**kwargs)
        return ConstraintSceneGenerationCompilation(
            component_cfgs=compiled.component_cfgs,
            variant_index=compiled.variant_index,
            components=compiled.components,
            template_scene_cfg=cfg.build_template_scene_cfg(self._resolve_template_cfg_class(cfg.template)),
        )

    def instancing(self, compiled: ConstraintSceneGenerationCompilation, **kwargs: object):
        del kwargs
        template_cfg = compiled.template_scene_cfg
        template_cls = getattr(template_cfg, "class_type", None)
        if template_cls is None:
            raise TypeError(
                f"Template scene cfg {type(template_cfg).__name__} has no class_type; "
                "ensure the *SceneCFG dataclass declares its peer class_type."
            )
        template = template_cls(replace(template_cfg))
        rng = random.Random(self._variant_seed(compiled.variant_index))
        base_assets = list(template.generate(asset_registry=self._asset_registry, rng=rng) or [])
        return self._instantiate_scene(compiled, base_assets=base_assets)

    @staticmethod
    def _resolve_template_cfg_class(template: str) -> type:
        if not isinstance(template, str) or not template:
            raise ValueError(f"template must be a non-empty string, got {template!r}")
        cfg_class = global_registry.get_cfg(template)
        if cfg_class is None:
            raise ValueError(
                f"ConstraintSceneGenerator could not resolve template {template!r}; "
                "template must match an existing *Scene class name exactly."
            )
        return cfg_class


@dataclass
class ConstraintSceneGeneratorCFG(SceneGeneratorCFG):
    """CFG for the unified CONSTRAINT scene workflow."""

    class_type: type[SceneGenerator] | None = ConstraintSceneGenerator
    template: str = ""
    template_payload: dict[str, Any] = field(default_factory=dict)

    def build_template_scene_cfg(self, scene_cfg_class: type) -> object:
        return scene_cfg_class(**self.template_payload)
