from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG


class ViewerComponent(TaskComponent):
    """Builds ``KitVisualizerCfg`` objects from a flat dict of named terms.

    YAML example::

        viewer:
          terms:
            look_at_object:
              lookat_object: "@pickable"
              offset: [-1.5, -1.5, 1.5]
    """

    def generate(self, cfg: ViewerComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        roles = roles or {}
        resolved_terms: dict[str, dict[str, Any]] = {}
        for func_name, spec in cfg.terms.items():
            params = dict(spec or {})
            if roles:
                params = self.resolve_roles(params, roles)
            resolved_terms[func_name] = params
        return {"default_visualizer_cfg": {"terms": resolved_terms}}


@dataclass
class ViewerComponentCFG(TaskComponentCFG):
    """Flat viewer configuration — function name is the key.

    YAML example::

        viewer:
          terms:
            look_at_object:
              lookat_object: "@pickable"
              offset: [-1.5, -1.5, 1.5]
    """

    class_type: type = ViewerComponent
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)
