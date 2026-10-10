from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

from ..assets.semantic_index import is_non_strict_asset_reference, semantic_ref_from_asset_pool_item
from ..common.diagnostics import error
from ..common.reproducibility import stable_rng, stable_seed

INSTANCE_REF_PREFIX = "@instances:"
INSTANCE_ALIASES_KEY = "__instance_aliases"
AssetTagResolver = Callable[[str], list[str]]


@dataclass(frozen=True)
class _AssetPoolCandidate:
    asset_name: str
    weight: float
    raw_index: int
    weighted: bool


@dataclass(frozen=True)
class _AssetPoolSourceItem:
    raw_item: Any
    raw_index: int
    target: str


@dataclass(frozen=True)
class _AssetPoolGroup:
    group_id: str
    asset_names: tuple[str, ...]
    weight: float
    raw_index: int
    target: str
    weighted: bool
    raw_item: Any


@dataclass
class _RuntimeAssetPoolGroup:
    group: _AssetPoolGroup
    remaining_assets: list[str]


def materialize_scene_payload(
    payload: Mapping[str, Any] | None,
    *,
    asset_tag_resolver: AssetTagResolver | None = None,
) -> dict[str, Any]:
    """Materialize public scene YAML authoring sugar into concrete objects/nodes.

    This is a front-end adaptor only. Solver/runtime internals continue to
    consume explicit object and node names.
    """

    out = deepcopy(dict(payload or {}))
    objects, aliases = _materialize_scene_objects(
        out.get("objects"),
        seed=_scene_seed(out),
        asset_tag_resolver=asset_tag_resolver,
    )
    if objects is not None:
        out["objects"] = objects
    layout = out.get("layout")
    if isinstance(layout, Mapping):
        out["layout"] = materialize_layout_payload(dict(layout), instance_aliases=aliases)
    return out


def materialize_task_payload(
    payload: Mapping[str, Any] | None, *, instance_aliases: Mapping[str, list[str]] | None = None
) -> dict[str, Any]:
    """Expand instance refs in task runtime authoring fields.

    Only object-list authoring fields are expanded. The task semantics remain
    unchanged after materialization.
    """

    out = deepcopy(dict(payload or {}))
    aliases = _normalise_aliases(instance_aliases)
    if not aliases:
        return out
    event = out.get("event")
    terms = event.get("terms") if isinstance(event, Mapping) else None
    if isinstance(terms, Mapping):
        out["event"] = {str(key): value for key, value in event.items()}
        out["event"]["terms"] = materialize_task_event_terms(terms, instance_aliases=aliases)
    return out


def materialize_task_event_terms(
    terms: Mapping[str, Any] | None,
    *,
    instance_aliases: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    out = deepcopy(dict(terms or {}))
    aliases = _normalise_aliases(instance_aliases)
    if not aliases:
        return out
    for raw_term in out.values():
        if not isinstance(raw_term, dict):
            continue
        sampling = raw_term.get("sampling")
        domains = sampling.get("domains") if isinstance(sampling, Mapping) else None
        if not isinstance(domains, list):
            continue
        for domain in domains:
            if not isinstance(domain, dict):
                continue
            if "nodes" in domain:
                domain["nodes"] = _expand_ref_list(domain.get("nodes"), aliases, target="runtime_sampling.nodes")
    return out


def instance_aliases_from_layout(payload: Mapping[str, Any] | None) -> dict[str, list[str]]:
    raw = (payload or {}).get(INSTANCE_ALIASES_KEY) if isinstance(payload, Mapping) else None
    return _normalise_aliases(raw if isinstance(raw, Mapping) else None)


def instance_aliases_from_scene(
    payload: Mapping[str, Any] | None,
    *,
    asset_tag_resolver: AssetTagResolver | None = None,
) -> dict[str, list[str]]:
    scene_payload = dict(payload or {}) if isinstance(payload, Mapping) else {}
    _, aliases = _materialize_scene_objects(
        scene_payload.get("objects"),
        seed=_scene_seed(scene_payload),
        asset_tag_resolver=asset_tag_resolver,
    )
    return aliases


def materialize_layout_payload(
    payload: dict[str, Any] | None,
    *,
    instance_aliases: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Convert public YAML authoring fields into the solver's internal payload.

    Public ManaEnv scene YAML uses ``layout.nodes`` to avoid overloading the
    top-level ``objects`` asset declarations. The solver schema still consumes
    an ``objects`` list of layout nodes, so this function is the narrow adaptor
    between the two representations.
    """

    out = deepcopy(dict(payload or {}))
    aliases = _normalise_aliases(instance_aliases) or instance_aliases_from_layout(out)
    if aliases:
        out[INSTANCE_ALIASES_KEY] = aliases
        _expand_layout_relations(out, aliases)
    if "nodes" not in out:
        return out
    if out.get("objects"):
        raise error("layout_nodes_objects_conflict", "layout must not define both nodes and objects")
    raw_nodes = out.pop("nodes")
    out["objects"] = _node_list(raw_nodes, instance_aliases=aliases)
    return out


def _node_list(raw_nodes: Any, *, instance_aliases: Mapping[str, list[str]] | None = None) -> list[dict[str, Any]]:
    aliases = _normalise_aliases(instance_aliases)
    if isinstance(raw_nodes, dict):
        nodes = []
        for name, raw in raw_nodes.items():
            if raw is None:
                raw = {}
            if not isinstance(raw, dict):
                raise error("layout_node_invalid", "layout node must be a mapping", target=str(name))
            raw_instance_ref = raw.get("instances_of")
            if raw_instance_ref is not None:
                instance_names = _alias_names(str(raw_instance_ref), aliases, target=str(name))
                base = {
                    key: deepcopy(value) for key, value in raw.items() if key not in {"instances_of", "node_overrides"}
                }
                overrides = raw.get("node_overrides")
                if overrides is not None and not isinstance(overrides, Mapping):
                    raise error(
                        "layout_node_overrides_invalid", "layout node_overrides must be a mapping", target=str(name)
                    )
                for instance_name in instance_names:
                    node = deepcopy(base)
                    node.update(deepcopy((overrides or {}).get(instance_name, {})))
                    node.setdefault("name", str(instance_name))
                    nodes.append(node)
                continue
            node = dict(raw)
            node.setdefault("name", str(name))
            nodes.append(node)
        return nodes
    if isinstance(raw_nodes, list):
        nodes = []
        for index, raw in enumerate(raw_nodes):
            if not isinstance(raw, dict):
                raise error("layout_node_invalid", "layout node must be a mapping", target=f"node_{index}")
            if not raw.get("name"):
                raise error("layout_node_name_missing", "layout node list entries require name", target=f"node_{index}")
            nodes.append(dict(raw))
        return nodes
    raise error("layout_nodes_invalid", "layout.nodes must be a mapping or list")


def _materialize_scene_objects(
    raw_objects: Any,
    *,
    seed: int = 0,
    asset_tag_resolver: AssetTagResolver | None = None,
) -> tuple[list[dict[str, Any]] | None, dict[str, list[str]]]:
    if raw_objects is None:
        return None, {}
    raw_list = raw_objects if isinstance(raw_objects, list) else [raw_objects]
    objects: list[dict[str, Any]] = []
    aliases: dict[str, list[str]] = {}
    for index, raw in enumerate(raw_list):
        if not isinstance(raw, Mapping):
            continue
        item = deepcopy(dict(raw))
        if _has_asset_pool_authoring(item):
            pool_objects, pool_aliases = _materialize_asset_pool_object(
                item,
                index=index,
                scene_seed=seed,
                asset_tag_resolver=asset_tag_resolver,
            )
            objects.extend(pool_objects)
            aliases.update(pool_aliases)
            continue
        raw_instances = item.pop("instances", None)
        if raw_instances is None:
            objects.append(item)
            continue
        if not isinstance(raw_instances, Mapping):
            raise error(
                "scene_object_instances_invalid",
                "object instances must be a mapping",
                target=str(item.get("name") or index),
            )
        base_name = str(item.get("name") or item.get("tags") or f"object_{index}").strip()
        if not base_name:
            raise error(
                "scene_object_instance_base_missing",
                "object instances require object name or tags",
                target=f"object_{index}",
            )
        names = _instance_names(base_name, raw_instances)
        alias_id = str(raw_instances.get("id") or base_name).strip()
        aliases[alias_id] = names
        aliases[base_name] = names
        overrides = raw_instances.get("overrides")
        if overrides is not None and not isinstance(overrides, Mapping):
            raise error(
                "scene_object_instance_overrides_invalid", "instance overrides must be a mapping", target=base_name
            )
        semantic_tag_templates = raw_instances.get("semantic_tag_templates")
        if semantic_tag_templates is not None and not isinstance(semantic_tag_templates, Mapping):
            raise error(
                "scene_object_instance_semantic_tag_templates_invalid",
                "semantic_tag_templates must be a mapping",
                target=base_name,
            )
        for offset, name in enumerate(names):
            instance = deepcopy(item)
            instance["name"] = name
            _apply_semantic_tag_templates(
                instance,
                semantic_tag_templates,
                base_name=base_name,
                raw_instances=raw_instances,
                offset=offset,
            )
            instance.update(deepcopy((overrides or {}).get(name, {})))
            objects.append(instance)
    return objects, aliases


def _materialize_asset_pool_object(
    item: dict[str, Any],
    *,
    index: int,
    scene_seed: int,
    asset_tag_resolver: AssetTagResolver | None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    target = str(item.get("id") or item.get("name") or index)
    source_items = _asset_pool_source_items(item, target=target)
    groups = _asset_pool_groups(source_items, asset_tag_resolver=asset_tag_resolver)
    if not groups:
        raise error(
            "scene_object_asset_pool_empty",
            "asset_pool must contain at least one asset name",
            target=target,
        )
    raw_count = item.get("pool_select_count")
    if raw_count is None:
        raise error(
            "scene_object_pool_select_count_missing",
            "asset_pool requires pool_select_count",
            target=target,
        )
    count = int(raw_count)
    if count <= 0:
        raise error(
            "scene_object_pool_select_count_invalid",
            "pool_select_count must be positive",
            target=target,
            count=count,
        )
    pool_size = sum(len(group.asset_names) for group in groups)
    if count > pool_size:
        raise error(
            "scene_object_pool_select_count_too_large",
            "pool_select_count cannot exceed asset_pool size",
            target=target,
            count=count,
            pool_size=pool_size,
        )

    base_name = str(item.get("id") or item.get("name") or item.get("asset_pool_alias") or f"asset_pool_{index}").strip()
    if not base_name:
        base_name = f"asset_pool_{index}"
    selected = _select_asset_pool_candidates(
        groups,
        count=count,
        seed=_asset_pool_seed(item, scene_seed=scene_seed, index=index, base_name=base_name),
        base_name=base_name,
    )
    name_template = str(item.get("pool_name_template") or item.get("name_template") or f"{base_name}_{{i}}")
    start = int(item.get("pool_start_index", item.get("start_index", 0)) or 0)
    zero_pad = int(item.get("pool_zero_pad", item.get("zero_pad", 0)) or 0)
    selected_objects: list[dict[str, Any]] = []
    names: list[str] = []
    for offset, asset_name in enumerate(selected):
        raw_index = start + offset
        formatted_index = f"{raw_index:0{zero_pad}d}" if zero_pad > 0 else str(raw_index)
        name = name_template.format(
            i=formatted_index,
            index=formatted_index,
            raw_index=raw_index,
            ordinal=offset + 1,
            asset=asset_name,
            base=base_name,
        )
        clone = deepcopy(item)
        for key in (
            "id",
            "asset_pool",
            "asset_pool_tag",
            "asset_pool_tag_weight",
            "asset_pool_alias",
            "pool_select_count",
            "pool_seed",
            "pool_name_template",
            "pool_start_index",
            "pool_zero_pad",
            "name_template",
            "start_index",
            "zero_pad",
        ):
            clone.pop(key, None)
        clone["id"] = str(name)
        clone["name"] = str(name)
        clone["gen_method"] = 0
        clone["asset"] = asset_name
        clone["tags"] = asset_name
        selected_objects.append(clone)
        names.append(str(name))
    aliases: dict[str, list[str]] = {}
    aliases[base_name] = names
    explicit_alias = str(item.get("asset_pool_alias") or "").strip()
    if explicit_alias:
        aliases[explicit_alias] = names
    return selected_objects, aliases


def asset_pool_candidate_asset_names(
    item: Mapping[str, Any],
    *,
    target: str = "",
    asset_tag_resolver: AssetTagResolver | None = None,
) -> list[tuple[str, Any]]:
    """Return every concrete asset id that can be selected from an asset pool."""

    source_items = _asset_pool_source_items(dict(item), target=target or str(item.get("name") or "asset_pool"))
    groups = _asset_pool_groups(source_items, asset_tag_resolver=asset_tag_resolver)
    out: list[tuple[str, Any]] = []
    for group in groups:
        out.extend((asset_name, group.raw_item) for asset_name in group.asset_names)
    return out


def _has_asset_pool_authoring(item: Mapping[str, Any]) -> bool:
    return item.get("asset_pool") is not None or bool(str(item.get("asset_pool_tag") or "").strip())


def _asset_pool_source_items(item: dict[str, Any], *, target: str) -> list[_AssetPoolSourceItem]:
    source_items: list[_AssetPoolSourceItem] = []
    raw_pool = item.get("asset_pool")
    raw_pool_tag = str(item.get("asset_pool_tag") or "").strip()
    if raw_pool is not None:
        if not isinstance(raw_pool, list):
            raise error(
                "scene_object_asset_pool_invalid",
                "asset_pool must be a list",
                target=target,
            )
        if not raw_pool and not raw_pool_tag:
            raise error(
                "scene_object_asset_pool_invalid",
                "asset_pool must be a non-empty list",
                target=target,
            )
        for raw_index, raw_item in enumerate(raw_pool):
            source_items.append(
                _AssetPoolSourceItem(
                    raw_item=raw_item,
                    raw_index=raw_index,
                    target=f"{target}.asset_pool[{raw_index}]",
                )
            )
    if raw_pool_tag:
        tag_item: dict[str, Any] = {"tag": raw_pool_tag}
        if "asset_pool_tag_weight" in item:
            tag_item["weight"] = item.get("asset_pool_tag_weight")
        source_items.append(
            _AssetPoolSourceItem(
                raw_item=tag_item,
                raw_index=len(source_items),
                target=f"{target}.asset_pool_tag",
            )
        )
    if not source_items:
        raise error(
            "scene_object_asset_pool_invalid",
            "asset_pool must be a non-empty list or asset_pool_tag must be set",
            target=target,
        )
    return source_items


def _asset_pool_groups(
    source_items: list[_AssetPoolSourceItem],
    *,
    asset_tag_resolver: AssetTagResolver | None,
) -> list[_AssetPoolGroup]:
    groups: list[_AssetPoolGroup] = []
    for source in source_items:
        raw_item = source.raw_item
        if is_non_strict_asset_reference(raw_item):
            raise error(
                "non_strict_semantic_asset_reference",
                "weak, approximate, placeholder, or fallback asset references are not allowed",
                target=source.target,
                source_kind="asset_pool",
                payload=dict(raw_item) if isinstance(raw_item, Mapping) else repr(raw_item),
            )
        tag = _asset_pool_item_tag(raw_item)
        group_asset_names = _asset_pool_item_group_assets(raw_item, target=source.target)
        asset_name = "" if group_asset_names else semantic_ref_from_asset_pool_item(raw_item)
        if sum(bool(value) for value in (tag, group_asset_names, asset_name)) > 1:
            raise error(
                "scene_object_asset_pool_item_ambiguous",
                "asset_pool item must provide exactly one of asset id, tag, or assets group",
                target=source.target,
                item=repr(raw_item),
            )
        weight = _asset_pool_item_weight(raw_item, target=source.target)
        if tag:
            assets = _resolve_asset_pool_tag(
                tag,
                target=source.target,
                asset_tag_resolver=asset_tag_resolver,
            )
            groups.append(
                _AssetPoolGroup(
                    group_id=f"tag:{tag}",
                    asset_names=tuple(assets),
                    weight=weight,
                    raw_index=source.raw_index,
                    target=source.target,
                    weighted=True,
                    raw_item=raw_item,
                )
            )
            continue
        if group_asset_names:
            group_id = _asset_pool_item_group_id(raw_item, fallback=f"group_{source.raw_index}")
            groups.append(
                _AssetPoolGroup(
                    group_id=f"group:{group_id}",
                    asset_names=tuple(group_asset_names),
                    weight=weight,
                    raw_index=source.raw_index,
                    target=source.target,
                    weighted=True,
                    raw_item=raw_item,
                )
            )
            continue
        if not asset_name:
            raise error(
                "scene_object_asset_pool_item_invalid",
                "asset_pool item must provide an asset id or tag",
                target=source.target,
                item=repr(raw_item),
            )
        groups.append(
            _AssetPoolGroup(
                group_id=f"asset:{asset_name}",
                asset_names=(asset_name,),
                weight=weight,
                raw_index=source.raw_index,
                target=source.target,
                weighted=isinstance(raw_item, Mapping) or weight != 1.0,
                raw_item=raw_item,
            )
        )
    _reject_duplicate_asset_pool_assets(groups)
    return groups


def _asset_pool_item_tag(raw_item: Any) -> str:
    if not isinstance(raw_item, Mapping):
        return ""
    return str(raw_item.get("tag") or "").strip()


def _asset_pool_item_group_assets(raw_item: Any, *, target: str) -> list[str]:
    if not isinstance(raw_item, Mapping):
        return []
    raw_assets = raw_item.get("assets")
    if raw_assets is None:
        raw_assets = raw_item.get("asset_ids")
    if raw_assets is None:
        return []
    if not isinstance(raw_assets, list) or not raw_assets:
        raise error(
            "scene_object_asset_pool_group_assets_invalid",
            "asset_pool group assets must be a non-empty list",
            target=target,
            item=repr(raw_item),
        )
    assets: list[str] = []
    for index, raw_asset in enumerate(raw_assets):
        if is_non_strict_asset_reference(raw_asset):
            raise error(
                "non_strict_semantic_asset_reference",
                "weak, approximate, placeholder, or fallback asset references are not allowed",
                target=f"{target}.assets[{index}]",
                source_kind="asset_pool_group",
                payload=dict(raw_asset) if isinstance(raw_asset, Mapping) else repr(raw_asset),
            )
        asset_name = semantic_ref_from_asset_pool_item(raw_asset)
        if not asset_name:
            raise error(
                "scene_object_asset_pool_group_asset_invalid",
                "asset_pool group assets must contain concrete asset ids",
                target=f"{target}.assets[{index}]",
                item=repr(raw_asset),
            )
        assets.append(asset_name)
    return assets


def _asset_pool_item_group_id(raw_item: Any, *, fallback: str) -> str:
    if not isinstance(raw_item, Mapping):
        return fallback
    for key in ("group", "group_id", "id", "name"):
        value = str(raw_item.get(key) or "").strip()
        if value:
            return value
    return fallback


def _resolve_asset_pool_tag(
    tag: str,
    *,
    target: str,
    asset_tag_resolver: AssetTagResolver | None,
) -> list[str]:
    if asset_tag_resolver is None:
        raise error(
            "scene_object_asset_pool_tag_resolver_missing",
            "asset_pool tag expansion requires an asset tag resolver",
            target=target,
            tag=tag,
        )
    try:
        raw_assets = asset_tag_resolver(tag)
    except Exception as exc:  # noqa: BLE001
        raise error(
            "scene_object_asset_pool_tag_resolve_failed",
            "asset_pool tag expansion failed",
            target=target,
            tag=tag,
            reason=str(exc),
        ) from exc
    assets = sorted({str(asset).strip() for asset in (raw_assets or []) if str(asset).strip()})
    if not assets:
        raise error(
            "scene_object_asset_pool_tag_empty",
            "asset_pool tag did not resolve to any assets",
            target=target,
            tag=tag,
        )
    return assets


def _asset_pool_item_weight(raw_item: Any, *, target: str) -> float:
    if not isinstance(raw_item, Mapping) or "weight" not in raw_item:
        return 1.0
    raw_weight = raw_item.get("weight")
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError) as exc:
        raise error(
            "scene_object_asset_pool_weight_invalid",
            "asset_pool item weight must be a positive number",
            target=target,
            weight=raw_weight,
        ) from exc
    if not isfinite(weight) or weight <= 0.0:
        raise error(
            "scene_object_asset_pool_weight_invalid",
            "asset_pool item weight must be a positive number",
            target=target,
            weight=raw_weight,
        )
    return weight


def _reject_duplicate_asset_pool_assets(groups: list[_AssetPoolGroup]) -> None:
    seen: dict[str, _AssetPoolGroup] = {}
    for group in groups:
        for asset_name in group.asset_names:
            if asset_name in seen:
                raise error(
                    "scene_object_asset_pool_candidate_duplicate",
                    "asset_pool candidates must resolve to unique asset ids",
                    target=group.target,
                    asset_id=asset_name,
                    first_group=seen[asset_name].group_id,
                    duplicate_group=group.group_id,
                )
            seen[asset_name] = group


def _select_asset_pool_candidates(groups: list[_AssetPoolGroup], *, count: int, seed: int, base_name: str) -> list[str]:
    if all(len(group.asset_names) == 1 for group in groups):
        candidates = [
            _AssetPoolCandidate(
                asset_name=group.asset_names[0],
                weight=group.weight,
                raw_index=group.raw_index,
                weighted=group.weighted,
            )
            for group in groups
        ]
        return _select_flat_asset_pool_candidates(candidates, count=count, seed=seed, base_name=base_name)

    return _select_grouped_asset_pool_candidates(groups, count=count, seed=seed, base_name=base_name)


def _select_flat_asset_pool_candidates(
    candidates: list[_AssetPoolCandidate], *, count: int, seed: int, base_name: str
) -> list[str]:
    if not any(candidate.weighted for candidate in candidates):
        canonical_pool = sorted(candidate.asset_name for candidate in candidates)
        return stable_rng(
            seed,
            "scene_asset_pool",
            base_name,
            payload={"pool": canonical_pool, "count": count},
        ).sample(canonical_pool, count)

    canonical_candidates = sorted(candidates, key=lambda candidate: (candidate.asset_name, candidate.raw_index))
    rng = stable_rng(
        seed,
        "scene_asset_pool",
        base_name,
        payload={
            "pool": [
                {"asset": candidate.asset_name, "weight": candidate.weight, "index": candidate.raw_index}
                for candidate in canonical_candidates
            ],
            "count": count,
        },
    )
    remaining = list(canonical_candidates)
    selected: list[str] = []
    for _ in range(count):
        total_weight = sum(candidate.weight for candidate in remaining)
        threshold = rng.random() * total_weight
        cumulative = 0.0
        for index, candidate in enumerate(remaining):
            cumulative += candidate.weight
            if threshold < cumulative or index == len(remaining) - 1:
                selected.append(candidate.asset_name)
                del remaining[index]
                break
    return selected


def _select_grouped_asset_pool_candidates(
    groups: list[_AssetPoolGroup], *, count: int, seed: int, base_name: str
) -> list[str]:
    canonical_groups = sorted(groups, key=lambda group: (group.group_id, group.raw_index))
    rng = stable_rng(
        seed,
        "scene_asset_pool",
        base_name,
        payload={
            "groups": [
                {
                    "group": group.group_id,
                    "weight": group.weight,
                    "assets": list(group.asset_names),
                    "index": group.raw_index,
                }
                for group in canonical_groups
            ],
            "count": count,
        },
    )
    remaining = [
        _RuntimeAssetPoolGroup(group=group, remaining_assets=sorted(group.asset_names)) for group in canonical_groups
    ]
    selected: list[str] = []
    for _ in range(count):
        active_groups = [group for group in remaining if group.remaining_assets]
        total_weight = sum(group.group.weight for group in active_groups)
        threshold = rng.random() * total_weight
        cumulative = 0.0
        selected_group = active_groups[-1]
        for group in active_groups:
            cumulative += group.group.weight
            if threshold < cumulative:
                selected_group = group
                break
        asset_index = rng.randrange(len(selected_group.remaining_assets))
        selected.append(selected_group.remaining_assets.pop(asset_index))
    return selected


def _asset_pool_seed(item: Mapping[str, Any], *, scene_seed: int, index: int, base_name: str) -> int:
    raw = item.get("pool_seed")
    if raw is not None:
        return int(raw)
    return stable_seed(int(scene_seed), "asset_pool", index, base_name)


def _scene_seed(payload: Mapping[str, Any]) -> int:
    layout = payload.get("layout")
    if isinstance(layout, Mapping) and layout.get("seed") is not None:
        return int(layout.get("seed"))
    definition = payload.get("definition")
    if isinstance(definition, Mapping) and definition.get("seed") is not None:
        return int(definition.get("seed"))
    return 0


def _instance_names(base_name: str, raw_instances: Mapping[str, Any]) -> list[str]:
    if isinstance(raw_instances.get("names"), list):
        names = [str(value).strip() for value in raw_instances.get("names") or [] if str(value).strip()]
        if not names:
            raise error("scene_object_instance_names_empty", "instance names must not be empty", target=base_name)
        return names
    count = int(raw_instances.get("count", 0) or 0)
    if count <= 0:
        raise error(
            "scene_object_instance_count_invalid", "instance count must be positive", target=base_name, count=count
        )
    start = int(raw_instances.get("start_index", 0) or 0)
    zero_pad = int(raw_instances.get("zero_pad", 0) or 0)
    template = str(raw_instances.get("name_template") or f"{base_name}_{{i}}")
    names = []
    for offset in range(count):
        index = start + offset
        formatted_index = f"{index:0{zero_pad}d}" if zero_pad > 0 else str(index)
        names.append(
            template.format(
                i=formatted_index,
                index=formatted_index,
                raw_index=index,
                base=base_name,
            )
        )
    return names


def _apply_semantic_tag_templates(
    instance: dict[str, Any],
    raw_templates: Any,
    *,
    base_name: str,
    raw_instances: Mapping[str, Any],
    offset: int,
) -> None:
    if not isinstance(raw_templates, Mapping):
        return
    raw_append = raw_templates.get("append")
    if raw_append is None:
        return
    append_templates = raw_append if isinstance(raw_append, list) else [raw_append]
    start = int(raw_instances.get("start_index", 0) or 0)
    zero_pad = int(raw_instances.get("zero_pad", 0) or 0)
    raw_index = start + offset
    formatted_index = f"{raw_index:0{zero_pad}d}" if zero_pad > 0 else str(raw_index)
    ordinal = int(raw_templates.get("ordinal_start", 1) or 1) + offset
    context = {
        "base": base_name,
        "name": str(instance.get("name") or ""),
        "i": formatted_index,
        "index": formatted_index,
        "raw_index": raw_index,
        "ordinal": ordinal,
    }
    tags = [str(tag) for tag in instance.get("semantic_tags", []) or [] if str(tag).strip()]
    for template in append_templates:
        tag = str(template).format(**context).strip()
        if tag and tag not in tags:
            tags.append(tag)
    instance["semantic_tags"] = tags


def _expand_layout_relations(payload: dict[str, Any], aliases: Mapping[str, list[str]]) -> None:
    for key in ("regions", "region_sample", "scatter_groups", "scatter_group", "volume_groups", "volume_group"):
        raw_items = payload.get(key)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if isinstance(item, dict) and "objects" in item:
                item["objects"] = _expand_ref_list(item.get("objects"), aliases, target=f"layout.{key}.objects")
    variants = payload.get("variants")
    if isinstance(variants, dict) and "objects" in variants:
        variants["objects"] = _expand_ref_list(variants.get("objects"), aliases, target="layout.variants.objects")


def _expand_ref_list(raw: Any, aliases: Mapping[str, list[str]], *, target: str) -> list[str]:
    if isinstance(raw, str):
        values: list[Any] = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        raise error("instance_ref_list_invalid", "instance references require a string or list", target=target)
    out: list[str] = []
    for value in values:
        if isinstance(value, str) and value.startswith(INSTANCE_REF_PREFIX):
            out.extend(_alias_names(value[len(INSTANCE_REF_PREFIX) :], aliases, target=target))
        elif isinstance(value, str):
            token = value.strip()
            if token:
                out.append(token)
        else:
            raise error("instance_ref_invalid", "instance reference entries must be strings", target=target)
    return out


def _alias_names(alias: str, aliases: Mapping[str, list[str]], *, target: str) -> list[str]:
    key = str(alias).strip()
    if key.startswith(INSTANCE_REF_PREFIX):
        key = key[len(INSTANCE_REF_PREFIX) :].strip()
    names = aliases.get(key)
    if not names:
        raise error("instance_alias_unknown", "unknown instance alias", target=target, alias=key)
    return list(names)


def _normalise_aliases(raw: Mapping[str, list[str]] | None) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for key, values in (raw or {}).items():
        if isinstance(values, str):
            names = [values]
        else:
            names = [str(value) for value in values or [] if str(value).strip()]
        if names:
            aliases[str(key)] = names
    return aliases


def authoring_layout_from_template(
    template: Any,
    *,
    example_group: str = "",
    source_name: str = "",
) -> dict[str, Any]:
    """Convert a native SceneTemplate layout into public ManaEnv YAML layout.

    Internal solver payload uses ``objects`` as layout nodes. Public scene YAML
    uses ``layout.nodes`` to avoid conflicting with top-level asset
    declarations. This is the single serialization boundary for template
    builders.
    """

    out = deepcopy(template.layout)
    objects = out.pop("objects", [])
    nodes: dict[str, dict[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, dict) or not obj.get("name"):
            continue
        node = dict(obj)
        name = str(node.pop("name"))
        nodes[name] = node
    out["nodes"] = nodes
    migration = dict(out.get("migration") or {})
    if example_group:
        migration.setdefault("example_group", example_group)
    if source_name:
        migration.setdefault("source_scene", source_name)
    migration.setdefault("template_mode", template.mode)
    out["migration"] = migration
    return out
