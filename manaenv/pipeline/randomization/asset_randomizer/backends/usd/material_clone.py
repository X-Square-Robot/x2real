from __future__ import annotations

from typing import Any


def clone_material_for_instance(*, stage: Any, material_path: str, instance_path: str) -> str:
    del stage, instance_path
    raise NotImplementedError(f"material clone is not implemented yet for {material_path}")
