from __future__ import annotations

from typing import Any


def prim_path(prim: Any) -> str:
    return str(prim.GetPath())


def prim_name(prim: Any) -> str:
    return str(prim.GetName())
