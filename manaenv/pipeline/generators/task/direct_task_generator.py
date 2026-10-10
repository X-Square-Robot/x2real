"""DIRECT task generator workflow.

DIRECT is now the default component-section workflow provided by
:class:`manaenv.pipeline.generators.task_generator.TaskGenerator`.
This module intentionally keeps only the public Direct subclass and CFG
symbol so Translator / global registry routing can continue to select
``DirectTaskGenerator`` while the compiler infrastructure lives in the
base class.
"""

from __future__ import annotations

from dataclasses import dataclass

from manaenv.pipeline.generators.task_generator import TaskGenerator, TaskGeneratorCFG


class DirectTaskGenerator(TaskGenerator):
    """Public DIRECT workflow shell."""


@dataclass
class DirectTaskGeneratorCFG(TaskGeneratorCFG):
    """CFG for the DIRECT task workflow."""

    class_type: type[TaskGenerator] | None = DirectTaskGenerator
