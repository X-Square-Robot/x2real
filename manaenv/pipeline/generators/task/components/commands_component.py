from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG
from manaenv.pipeline.generators.task.term_discovery import resolve_task_term


class CommandsComponent(TaskComponent):
    """Builds command configuration as a dynamic configclass.

    YAML example::

        command:
          terms:
            uniform_pose_command:
              target_cfg: "@destination"
    """

    def generate(self, cfg: CommandsComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        from isaaclab_arena.utils.configclass import make_configclass

        roles = roles or {}
        term_fields = []
        for func_name, spec in cfg.terms.items():
            params = dict(spec or {})
            if roles:
                params = self.resolve_roles(params, roles)
            func = resolve_task_term("command", func_name)
            term_cfg = func(**params)
            field_name = func_name.replace(".", "__")
            term_fields.append((field_name, type(term_cfg), term_cfg))

        if not term_fields:
            return {"commands_cfg": None}

        CommandsCfgCls = make_configclass("CommandsCfg", term_fields)
        return {"commands_cfg": CommandsCfgCls()}


@dataclass
class CommandsComponentCFG(TaskComponentCFG):

    class_type: type = CommandsComponent
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)
