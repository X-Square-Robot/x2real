"""Planner-skills task component — embeds a stage's cuRobo skill sequence.

Under the graph-task migration, each graph node (stage) is the single source of
truth for *its own* skills: the ``planner_skills_component`` on a stage YAML
carries the ordered skill specs the :class:`SkillTrajectoryPlanner` runs for
that node. The staged-collection DFS runs exactly one node per episode, so a
node's skills form one self-contained segment (typically ``[<ops>, stay]``,
with the boundary ``stay`` pose the snapshot is captured at).

Unlike gating/merge components this output is **kept per-node**: it lands in the
leaf task's ``runtime_cfgs["planner_skills"]`` and is read back off each
``_composite_children`` entry by the planner at setup. The composite's own
merged ``runtime_cfgs`` does not aggregate it (see
:mod:`graph_compose` ``DEFER_TO_P3_SECTIONS``).

YAML example::

    component_sections:
      planner_skills_component:
        skills:
          - type: stay
          - type: pick
            params: {grasp_label_asset: brick_yellow, object_name: "@sortable_1"}
          - type: place
            params: {object_name: "@sortable_1", target: {object_name: "@destination_1"}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG


class PlannerSkillsComponent(TaskComponent):
    """Passes through a stage's ordered skill specs (``@role`` refs resolved)."""

    def generate(self, cfg: PlannerSkillsComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        roles = roles or {}
        out: list[dict[str, Any]] = []
        for item in cfg.skills or []:
            if not isinstance(item, dict):
                continue
            spec = dict(item)
            params = spec.get("params") or spec.get("skill_params")
            if isinstance(params, dict) and roles:
                spec["params"] = self.resolve_roles(params, roles)
            elif isinstance(params, dict):
                spec["params"] = dict(params)
            out.append(spec)
        return {"planner_skills": out}


@dataclass
class PlannerSkillsComponentCFG(TaskComponentCFG):

    class_type: type = PlannerSkillsComponent
    skills: list[dict[str, Any]] = field(default_factory=list)
