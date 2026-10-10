"""Common enums and paths for the public evaluation pipeline."""

import os
from enum import Enum, IntEnum


class GenMethod(IntEnum):
    """Generation method enumeration.

    Defines how components (scene/robot/task) are generated:
        DIRECT (0): Use config directly, no generation needed
        TAGS (1): Match by semantic tags [TODO]
        CONSTRAINT (2): Match by constraints [TODO]
        AI (3): LLM structured task generation (action primitives + tags)

    Usage:
        gen_method: 0  # in YAML
        GenMethod.DIRECT  # in Python
    """

    DIRECT = 0  # Name/config based, no generation
    TAGS = 1  # Tag-based matching
    CONSTRAINT = 2  # Constraint-based matching
    AI = 3  # LLM structured task generation


class DataBucket(Enum):
    SCENES = "scenes"
    ROBOTS = "robots"
    TASKS = "tasks"
    ENVIRONMENTS = "environments"
    POLICY_CFGS = "policy_cfgs"
    POLICIES = "policies"
    BENCHMARKS = "benchmarks"

    def new_container(self) -> list:
        """Create the empty list container for this bucket."""
        return []


class BenchmarkFormat(Enum):
    """Export format for benchmark reports."""

    JSON = "json"
    MARKDOWN = "markdown"


DEFAULT_OUTPUT_ROOT = os.getenv("MANAENV_OUTPUT_ROOT", "~/.manaenv/output")


def output_path(subdir: str = "") -> str:
    """Build output path under the configurable root."""
    root = DEFAULT_OUTPUT_ROOT.rstrip("/")
    if not subdir:
        return f"{root}/"
    return f"{root}/{subdir.strip('/')}/"
