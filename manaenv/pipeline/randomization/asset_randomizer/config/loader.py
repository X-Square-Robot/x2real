from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..model import RandomizationSpec
from .schema import parse_randomization_spec


def load_spec_from_dict(data: Mapping[str, Any]) -> RandomizationSpec:
    return parse_randomization_spec(data)
