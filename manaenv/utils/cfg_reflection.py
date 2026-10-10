"""Reflective yaml-dict → typed-CFG populator.

Single entry point for materializing typed config dataclasses from
YAML-derived dictionaries.

Contract:

* ``populate_cfg(cls, yaml_dict)`` calls ``cls(**kwargs)`` with kwargs
  derived from ``yaml_dict``. The cfg class's ``__init__`` signature
  IS the schema -- adding/removing/renaming a field means updating the
  one ``__init__``, nothing else.
* Unknown yaml keys raise ``TypeError``: yaml typos fail loudly at
  translator time, not silently at sim time.
* When a parameter's annotation resolves to a class with its own
  ``__init__`` AND the yaml value is a ``dict``, we recurse:
  ``populate_cfg(annotation_cls, yaml_subdict)``. This is what gives
  nested configuration objects their typed shape without per-field code.
* No alias layer. yaml keys map 1:1 to ``__init__`` kwarg names; if a
  field needs a name that shadows a Python builtin (e.g. ``format``),
  use the builtin name directly -- shadowing is local to ``__init__``
  and harmless.

Non-goals:

* Not a general-purpose schema validator. Per-field value validation
  (range, choice, cross-field) lives in the cfg class's
  ``__post_init__`` (or, for non-dataclass cfgs like ``BaseExportCFG``,
  in custom validation methods).
* No coercion. yaml-loaded values must already be the right Python
  type; PyYAML hands back ``bool`` for ``true``/``false`` and ``int``
  for digits. String→bool/int helpers are intentionally absent so that
  yaml type mistakes surface immediately.
"""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _resolved_init_hints(cls: type) -> dict[str, Any]:
    """Return ``cls.__init__`` annotations with string forward-refs evaluated.

    Falls back to the raw signature annotations on resolution failure
    (e.g. cfg classes that import behind ``TYPE_CHECKING`` guards).
    """
    try:
        return typing.get_type_hints(cls.__init__)
    except Exception:
        return {}


def _strip_optional(annotation: Any) -> Any:
    """Unwrap ``X | None`` / ``Optional[X]`` to ``X``.

    Returns the annotation unchanged if it isn't a Union, or if it's a
    multi-arm union with more than one non-None member (we don't support
    polymorphic cfg fields -- the choice would have to live in yaml itself).
    """
    origin = typing.get_origin(annotation)
    union_origins = {typing.Union}
    if hasattr(types, "UnionType"):
        union_origins.add(types.UnionType)
    if origin not in union_origins:
        return annotation
    non_none = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
    if len(non_none) == 1:
        return non_none[0]
    return annotation


def _is_typed_cfg_class(annotation: Any) -> bool:
    """Heuristic: a typed sub-cfg is a non-builtin class with an ``__init__``.

    We deliberately do NOT require ``@dataclass`` -- ``BaseExportCFG``
    subclasses are plain classes too. We DO exclude ``dict``, ``list``,
    ``str``, ``int``, ``bool``, ``float``, ``None``, ``Path``, and the
    typing module's generic aliases (``Dict[str, Any]`` etc.) because
    those are pass-through primitives.
    """
    if annotation is None or annotation is type(None):
        return False
    if not inspect.isclass(annotation):
        return False
    if annotation in (dict, list, tuple, set, str, int, bool, float, bytes, Path):
        return False
    return True


def populate_cfg(cls: type, yaml_dict: Mapping[str, Any] | None) -> Any:
    """Construct ``cls`` from ``yaml_dict`` via ``__init__`` reflection.

    Args:
        cls: Target cfg class (dataclass or plain class with ``__init__``).
        yaml_dict: Yaml-derived dict (post-PyYAML load). ``None`` is
            treated as an empty dict (uses cfg defaults).

    Returns:
        ``cls(**kwargs)`` instance.

    Raises:
        TypeError: when ``yaml_dict`` contains a key that is not a
            kwarg of ``cls.__init__``. The error message lists both the
            offending key and the allowed key set so users can fix the
            yaml without grep.
    """
    if yaml_dict is None:
        return cls()

    if not isinstance(yaml_dict, Mapping):
        raise TypeError(
            f"populate_cfg({cls.__name__}, ...) expected a Mapping, "
            f"got {type(yaml_dict).__name__}. yaml block must be a dict."
        )

    sig = inspect.signature(cls.__init__)
    init_params = {
        name: param
        for name, param in sig.parameters.items()
        if name != "self" and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    resolved_hints = _resolved_init_hints(cls)

    kwargs: dict[str, Any] = {}
    for yaml_key, value in yaml_dict.items():
        if yaml_key not in init_params:
            raise TypeError(
                f"{cls.__name__}: unknown yaml key {yaml_key!r}. "
                f"Allowed keys: {sorted(init_params)!r}. "
                "Add the field to the cfg class's __init__ if this is a "
                "new knob; otherwise fix the typo in your yaml."
            )

        if isinstance(value, Mapping):
            annotation = resolved_hints.get(yaml_key, init_params[yaml_key].annotation)
            annotation = _strip_optional(annotation)
            if _is_typed_cfg_class(annotation):
                value = populate_cfg(annotation, value)

        kwargs[yaml_key] = value

    return cls(**kwargs)
