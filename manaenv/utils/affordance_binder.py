"""Reflection-driven binder for cfg dataclass slots typed as ``Asset`` / ``Affordance``.

Implements the Mana **asset-subscription protocol**: a yaml ``@<tag>``
reference means *subscribe to every scene asset carrying that tag*;
the cfg field's annotation determines how many hits the slot may
absorb.

* ``Asset`` (single) annotation
    Demand exactly one match. Multi-match raises with the candidate
    list so authors can either rename a tag, add a discriminating
    tag, or switch the slot to ``List[Asset]``. Zero matches also
    raise (a typo in the tag must surface eagerly).
* ``List[Asset]`` annotation
    Subscribe to every match. Zero hits resolve to an empty list and
    do not raise — empty subscriptions are a legal protocol state.

The same protocol applies to inline ``object_reference`` mappings
(materialised into the scene by :func:`inline_object_reference`)
and already-bound asset instances; the only difference is that those
inputs are intrinsically single-valued and therefore are wrapped in a
singleton list when the slot is ``List[Asset]``.

Ownership: the originating ``TaskGenerator.finalize`` calls
``bind_typed_slots(cfg, scene)`` once per task right before
constructing the runtime task object. The cfg's typed annotations
are the contract; the binder never writes outside the cfg dataclass.

Bindable bases are pluggable so unit tests can pass a stub class set
that doesn't drag IsaacLab into the import path.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Iterable, Mapping
from typing import Any

_DEFAULT_BASES_CACHE: tuple[type, ...] | None = None


def _default_bindable_bases() -> tuple[type, ...]:
    """Lazy import the production-time bindable bases from Arena.

    Cached after first successful import. Callers may pass stub bases via
    ``bindable_bases=`` when importing Arena is undesirable.
    """
    global _DEFAULT_BASES_CACHE
    if _DEFAULT_BASES_CACHE is not None:
        return _DEFAULT_BASES_CACHE
    from isaaclab_arena.affordances.affordance_base import AffordanceBase
    from isaaclab_arena.assets.asset import Asset

    _DEFAULT_BASES_CACHE = (Asset, AffordanceBase)
    return _DEFAULT_BASES_CACHE


def _union_origins() -> tuple[Any, ...]:
    union_origins: list[Any] = [typing.Union]
    union_type = getattr(types, "UnionType", None)
    if union_type is not None:
        union_origins.append(union_type)
    return tuple(union_origins)


def _peel_optional(annotation: Any) -> Any:
    """Unwrap ``X | None`` / ``Optional[X]`` to ``X`` (single non-None arm only)."""
    origin = typing.get_origin(annotation)
    if origin not in _union_origins():
        return annotation
    non_none = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
    if len(non_none) == 1:
        return non_none[0]
    return annotation


def _peel_list(annotation: Any) -> tuple[Any, bool]:
    """Return ``(element_type, is_list)``; preserves annotation when not a list/tuple."""
    origin = typing.get_origin(annotation)
    if origin in (list, tuple):
        args = typing.get_args(annotation)
        if args:
            return args[0], True
    return annotation, False


def _resolve_field_annotation(field: dataclasses.Field, owner_cls: type) -> Any:
    """Resolve a dataclass field's annotation to a runtime type object.

    Falls back to ``field.type`` when ``typing.get_type_hints`` cannot
    evaluate forward references (cfg classes that reference IsaacLab
    types behind ``TYPE_CHECKING`` guards).
    """
    try:
        hints = typing.get_type_hints(owner_cls, include_extras=False)
    except Exception:
        hints = {}
    return hints.get(field.name, field.type)


def _find_bindable_arm(annotation: Any, bindable_bases: tuple[type, ...]) -> type | None:
    """Find a class arm in *annotation* that subclasses one of *bindable_bases*.

    Handles ``Asset``, ``Asset | str``, ``Optional[Asset]``,
    ``list[Asset]`` and ``list[Asset | str]`` shapes.
    """
    annotation = _peel_optional(annotation)
    annotation, _ = _peel_list(annotation)
    annotation = _peel_optional(annotation)
    origin = typing.get_origin(annotation)
    if origin in _union_origins():
        for arg in typing.get_args(annotation):
            arg, _ = _peel_list(_peel_optional(arg))
            if isinstance(arg, type) and any(issubclass(arg, b) for b in bindable_bases):
                return arg
        return None
    if isinstance(annotation, type) and any(issubclass(annotation, b) for b in bindable_bases):
        return annotation
    return None


def is_bindable_field(
    field: dataclasses.Field,
    owner_cls: type,
    *,
    bindable_bases: Iterable[type] | None = None,
) -> bool:
    """Return ``True`` when *field* is a binder-managed slot."""
    bases = tuple(bindable_bases) if bindable_bases is not None else _default_bindable_bases()
    annotation = _resolve_field_annotation(field, owner_cls)
    return _find_bindable_arm(annotation, bases) is not None


def field_target_type(
    field: dataclasses.Field,
    owner_cls: type,
    *,
    bindable_bases: Iterable[type] | None = None,
) -> type | None:
    """Return the target type used for ``isinstance`` validation, or ``None``."""
    bases = tuple(bindable_bases) if bindable_bases is not None else _default_bindable_bases()
    annotation = _resolve_field_annotation(field, owner_cls)
    return _find_bindable_arm(annotation, bases)


def field_is_list(field: dataclasses.Field, owner_cls: type) -> bool:
    annotation = _resolve_field_annotation(field, owner_cls)
    annotation = _peel_optional(annotation)
    origin = typing.get_origin(annotation)
    if origin in _union_origins():
        return any(_peel_list(_peel_optional(arg))[1] for arg in typing.get_args(annotation))
    _, is_list = _peel_list(annotation)
    return is_list


def _scene_assets(scene: Any) -> Any:
    """Return the assets mapping carried by *scene*, or ``None``."""
    if scene is None:
        return None
    if isinstance(scene, Mapping):
        return scene
    return getattr(scene, "assets", None)


def _resolve_string(
    value: str,
    scene: Any,
    *,
    field_name: str,
    expect_list: bool = False,
) -> Any:
    """Resolve ``"@tag"`` / ``"name"`` per the asset-subscription protocol.

    Lookup order: exact ``scene.assets`` dict key, exact ``asset.name``
    match, then ``asset.tags`` membership.

    When ``expect_list`` is ``True`` (the binder field is annotated as
    ``List[Asset]``), every tag match is returned. The ordering is
    insertion order of ``scene.assets``. Zero matches resolve to an
    empty list.

    When ``expect_list`` is ``False`` (single ``Asset`` annotation), the
    function demands exactly one tag match. Zero matches and >1 matches
    both raise :class:`KeyError`, with the candidate list in the
    message so authors can disambiguate.
    """
    assets = _scene_assets(scene)
    if assets is None:
        if expect_list:
            return []
        raise ValueError(f"{field_name}: scene has no assets mapping; cannot resolve {value!r}.")
    lookup_key = value[1:] if value.startswith("@") else value
    if not lookup_key:
        raise ValueError(f"{field_name}: empty lookup key {value!r}.")
    if isinstance(assets, Mapping) and lookup_key in assets:
        result = assets[lookup_key]
        return [result] if expect_list else result
    iterable = assets.values() if isinstance(assets, Mapping) else iter(assets)
    asset_list = list(iterable)
    for asset in asset_list:
        if getattr(asset, "name", None) == lookup_key:
            return [asset] if expect_list else asset
    matches: list[Any] = []
    for asset in asset_list:
        tags = getattr(asset, "tags", None) or []
        if lookup_key in tags:
            matches.append(asset)

    if expect_list:
        return matches

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = [getattr(a, "name", "<unnamed>") for a in matches]
        raise KeyError(
            f"{field_name}: tag {value!r} matched {len(matches)} assets ({names!r}); "
            f"declare the slot as List[Asset] to subscribe to all of them, or add a "
            f"discriminating tag to narrow the match."
        )
    keys = list(assets) if isinstance(assets, Mapping) else [getattr(a, "name", "?") for a in asset_list]
    raise KeyError(f"{field_name}: cannot resolve {value!r} from scene assets (keys={keys!r}).")


def _check_target(result: Any, target_type: type | None, field_name: str) -> None:
    if target_type is None:
        return
    if not isinstance(target_type, type):
        return
    if not isinstance(result, target_type):
        raise TypeError(
            f"{field_name}: resolved object of type {type(result).__name__} is not an "
            f"instance of {target_type.__name__}"
        )


def inline_object_reference(
    spec: Mapping[str, Any],
    scene: Any,
    *,
    field_name: str = "<unknown>",
    default_parent: Any = None,
) -> Any:
    """Construct an ``ObjectReference`` from an inline yaml spec and register it into *scene*.

    The spec carries a ``type: object_reference`` discriminator
    (default), a ``prim_path`` pointing at a prim inside a parent
    background USD, an optional ``parent_asset`` reference (defaults
    to ``"@background"``), and an optional ``object_type``
    (``rigid`` / ``articulation`` / ``base``). The constructed
    reference is added to ``scene.assets`` (via ``scene.add_asset``
    when available, otherwise direct dict insertion) so downstream
    factories pick it up at env-build time.
    """
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference

    spec_type = spec.get("type", "object_reference")
    if spec_type != "object_reference":
        raise ValueError(
            f"{field_name}: unsupported inline asset spec type {spec_type!r}; " "expected 'object_reference'."
        )
    name = spec.get("name") or field_name
    prim_path = spec.get("prim_path")
    if not prim_path:
        raise ValueError(f"{field_name}: object_reference spec requires a 'prim_path'.")

    parent_ref = spec.get("parent_asset", "@background")
    parent_asset = default_parent
    if parent_asset is None:
        parent_asset = _resolve_string(parent_ref, scene, field_name=f"{field_name}.parent_asset")

    object_type_str = str(spec.get("object_type", "rigid")).lower()
    object_type_map = {
        "rigid": ObjectType.RIGID,
        "articulation": ObjectType.ARTICULATION,
        "base": ObjectType.BASE,
    }
    if object_type_str not in object_type_map:
        raise ValueError(
            f"{field_name}: object_reference spec has unsupported object_type={object_type_str!r}; "
            f"expected one of {sorted(object_type_map)}"
        )
    object_type = object_type_map[object_type_str]

    ref_kwargs: dict[str, Any] = {
        "name": name,
        "prim_path": prim_path,
        "object_type": object_type,
        "parent_asset": parent_asset,
    }
    tags = spec.get("tags")
    if tags is not None:
        ref_kwargs["tags"] = list(tags)

    obj_ref = ObjectReference(**ref_kwargs)

    assets = _scene_assets(scene)
    if isinstance(assets, dict) and obj_ref.name not in assets:
        if scene is not None and hasattr(scene, "add_asset"):
            scene.add_asset(obj_ref)
        else:
            assets[obj_ref.name] = obj_ref
    elif scene is not None and hasattr(scene, "add_asset") and not isinstance(assets, dict):
        scene.add_asset(obj_ref)
    return obj_ref


def resolve(
    value: Any,
    scene: Any,
    *,
    target_type: type | None,
    field_name: str = "<unknown>",
    bindable_bases: Iterable[type] | None = None,
    expect_list: bool = False,
) -> Any:
    """Resolve a single value (the cfg slot's raw payload) to a bindable instance.

    ``expect_list`` reflects the field annotation: ``True`` for
    ``List[Asset]`` slots (every input form fans out to a list), ``False``
    for single-``Asset`` slots (each input form returns one asset, with
    multi-match tag references raising — see :func:`_resolve_string`).
    """
    bases = tuple(bindable_bases) if bindable_bases is not None else _default_bindable_bases()

    if isinstance(value, list):
        if not expect_list:
            raise TypeError(
                f"{field_name}: list payload {value!r} cannot bind to a single-Asset slot; "
                "annotate the field as List[Asset] to accept multiple values."
            )
        out: list[Any] = []
        for i, item in enumerate(value):
            resolved = resolve(
                item,
                scene,
                target_type=target_type,
                field_name=f"{field_name}[{i}]",
                bindable_bases=bases,
                expect_list=False,
            )
            out.append(resolved)
        return out

    if isinstance(value, bases):
        _check_target(value, target_type, field_name)
        return [value] if expect_list else value

    if isinstance(value, Mapping):
        result = inline_object_reference(value, scene, field_name=field_name)
        _check_target(result, target_type, field_name)
        return [result] if expect_list else result

    if isinstance(value, str):
        result = _resolve_string(value, scene, field_name=field_name, expect_list=expect_list)
        if expect_list:
            for item in result:
                _check_target(item, target_type, field_name)
        else:
            _check_target(result, target_type, field_name)
        return result

    if value is None:
        if expect_list:
            return []
        raise ValueError(f"{field_name}: cannot resolve None to a bindable instance.")

    raise TypeError(f"{field_name}: unsupported value type {type(value).__name__}: {value!r}")


def bind_typed_slots(
    cfg: Any,
    scene: Any,
    *,
    bindable_bases: Iterable[type] | None = None,
) -> list[str]:
    """Resolve every bindable field of *cfg* against *scene* in-place.

    Returns the list of field names that were written. Fields whose
    annotation is not bindable are left untouched, including
    ``class_type`` / ``runtime_cfgs`` / etc.
    """
    if not dataclasses.is_dataclass(cfg):
        raise TypeError(f"bind_typed_slots expected a dataclass instance, got {type(cfg).__name__}")
    bases = tuple(bindable_bases) if bindable_bases is not None else _default_bindable_bases()
    bound: list[str] = []
    for f in dataclasses.fields(cfg):
        target = field_target_type(f, type(cfg), bindable_bases=bases)
        if target is None:
            continue
        is_list = field_is_list(f, type(cfg))
        raw_value = getattr(cfg, f.name)
        resolved = resolve(
            raw_value,
            scene,
            target_type=target,
            field_name=f.name,
            bindable_bases=bases,
            expect_list=is_list,
        )
        setattr(cfg, f.name, resolved)
        bound.append(f.name)
    return bound
