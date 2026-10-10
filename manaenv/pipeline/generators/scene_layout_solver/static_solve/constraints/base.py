from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...common.geometry import Pose
from ...common.ids import NodeId

if TYPE_CHECKING:
    from ..domains.manager import DomainManager
    from ..ir import CompiledLayoutIR


class HardFactor:
    priority: int = 100

    def check(self, assignment: dict[NodeId, Pose]) -> bool:
        raise NotImplementedError

    def forward_check(
        self,
        node_id: NodeId,
        assignment: dict[NodeId, Pose],
        ir: CompiledLayoutIR,
        domain_mgr: DomainManager,
    ) -> bool | None:
        return None

    def describe_failure(self, assignment: dict[NodeId, Pose]) -> dict[str, Any]:
        return {"factor": type(self).__name__}
