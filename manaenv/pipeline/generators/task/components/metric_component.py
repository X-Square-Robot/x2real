from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG
from manaenv.pipeline.generators.task.term_discovery import resolve_task_term


class MetricComponent(TaskComponent):
    """Builds ``MetricBase`` instances from a flat dict of named terms.

    Each entry in ``cfg.terms`` declares a metric builder function (the
    key) and its params (everything under it).  ``@role`` references in
    params are expanded.  The function is called immediately to produce
    a ``MetricBase`` instance — same pattern as TerminationComponent /
    EventComponent / ViewerComponent.

    ``get_metrics()`` returns a **list**, so the component collects all
    built instances into a list under the ``"metrics"`` key.

    YAML example::

        metrics:
          terms:
            success_rate: {}
            object_moved_rate:
              object_cfg: "@pickable"
    """

    def generate(self, cfg: MetricComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        roles = roles or {}
        results = []
        for func_name, spec in cfg.terms.items():
            params = dict(spec or {})
            if roles:
                params = self.resolve_roles(params, roles)
            func = resolve_task_term("metric", func_name)
            results.append(func(**params))
        return {"metrics": results}


@dataclass
class MetricComponentCFG(TaskComponentCFG):
    """Flat metric configuration — function name is the key.

    YAML example::

        metrics:
          terms:
            success_rate: {}
            object_moved_rate:
              object_cfg: "@pickable"
    """

    class_type: type = MetricComponent
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)
