"""Base component interface — shared by all pipeline components.

Mirrors BaseGenerator / BaseFactory as the third pillar of the pipeline.
Every component has a CFG with ``priority`` for execution ordering:
lower numbers run first (0 = highest, default 2 = normal consumer).

Roles-update protocol
---------------------
``generate()`` returns ``Dict[str, Any]``.  If the dict contains a
``_roles_update`` key whose value is a **non-empty** dict, the caller
(typically TaskGenerator) merges those entries into the shared ``roles``
dict so that later components can reference them via ``@role``.
An empty or absent ``_roles_update`` is silently ignored.

Use :func:`extract_roles_update` to pop the update from the result dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

ROLES_UPDATE_KEY = "_roles_update"


def extract_roles_update(result: dict[str, Any]) -> dict[str, Any]:
    """Pop and return the roles-update payload from a component result.

    Returns an empty dict when the key is absent or the value is falsy.
    """
    update = result.pop(ROLES_UPDATE_KEY, None)
    if update and isinstance(update, dict):
        return update
    return {}


class BaseComponent(ABC):

    def __init__(self, cfg: BaseComponentCFG | None = None):
        self.cfg = cfg

    @abstractmethod
    def generate(self, cfg: BaseComponentCFG, **kwargs) -> dict[str, Any]:
        pass


@dataclass
class BaseComponentCFG:

    class_type: type[BaseComponent] = None
    priority: int = 2
