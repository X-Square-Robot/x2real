from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ...common.diagnostics import LayoutDiagnostic
from ...common.geometry import Pose
from ...common.ids import NodeId
from ..constraints.evaluator import ConstraintEvaluator
from ..domains.manager import DomainManager
from .alternatives import ComponentAlternative, ComponentAlternativeProvider
from .model import SupportSurfaceComponent


@dataclass
class AlternativeSearchTrace:
    diagnostics: tuple[LayoutDiagnostic, ...] = ()
    rejection_counts: dict[str, int] = field(default_factory=dict)
    unknown_rejection_count: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)


class SupportSurfaceTransactionSolver:
    def __init__(
        self,
        *,
        domain_mgr: DomainManager,
        evaluator: ConstraintEvaluator,
        max_component_steps: int,
    ) -> None:
        self.domain_mgr = domain_mgr
        self.evaluator = evaluator
        self.max_component_steps = max(1, int(max_component_steps))
        self.trace = AlternativeSearchTrace()

    def iter_component_alternatives(
        self,
        component: SupportSurfaceComponent,
        assignment: dict[NodeId, Pose],
    ) -> Iterator[ComponentAlternative]:
        provider = ComponentAlternativeProvider(
            domain_mgr=self.domain_mgr,
            evaluator=self.evaluator,
            max_component_steps=self.max_component_steps,
        )
        yielded = 0
        for alternative in provider.iter_component_alternatives(component, assignment):
            yielded += 1
            yield alternative
        self.trace.evidence["yielded_alternatives"] = yielded
        if yielded == 0:
            self.trace.unknown_rejection_count = True
