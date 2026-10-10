from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from .manaenv_bridge.asset_tags import resolve_manaenv_asset_ids_by_tag
from .runner import LayoutSolveResult, layout_fingerprint, solve_and_apply
from .scene_spec.authoring import AssetTagResolver, materialize_scene_payload

LOGGER = logging.getLogger(__name__)


def layout_enabled(config: Any) -> bool:
    metadata = getattr(config, "metadata", {}) or {}
    layout_payload = metadata.get("layout") if isinstance(metadata, dict) else None
    return isinstance(layout_payload, dict) and bool(layout_payload.get("enabled", False))


def layout_payload_from_config(config: Any) -> dict[str, Any]:
    metadata = getattr(config, "metadata", {}) or {}
    layout_payload = metadata.get("layout") if isinstance(metadata, dict) else None
    return dict(layout_payload or {}) if isinstance(layout_payload, dict) else {}


def materialize_manaenv_scene_config(
    scene_config: Mapping[str, Any] | None,
    *,
    asset_tag_resolver: AssetTagResolver | None = None,
) -> dict[str, Any]:
    """Materialize public scene authoring fields before ManaEnv component parsing.

    ManaEnv's translator sees top-level ``objects`` before the layout solver
    hook runs. Public scene YAML may use ``objects.instances`` and layout
    ``@instances:...`` refs, so the formal pipeline must expand those fields
    at the translator boundary instead of relying on local review tools.
    """

    return materialize_scene_payload(
        scene_config or {},
        asset_tag_resolver=asset_tag_resolver or resolve_manaenv_asset_ids_by_tag,
    )


def solve_assets_for_scene_generator(
    generator: Any,
    assets: Mapping[str, Any] | Iterable[Any],
    layout_payload: dict[str, Any] | None = None,
) -> LayoutSolveResult:
    payload = dict(layout_payload or layout_payload_from_config(getattr(generator, "config", None)))
    if not payload or not payload.get("enabled"):
        return LayoutSolveResult(success=True)
    payload = _inject_background_poses(generator, payload)
    _store_materialized_layout_payload(generator, payload)

    key = layout_fingerprint(payload)
    state = getattr(generator, "_scene_layout_solver_state", None)
    if not isinstance(state, dict):
        state = {}
    variant_index = int(state.get(key, 0))
    nums = getattr(getattr(generator, "config", None), "nums", None)
    result = solve_and_apply(payload, assets, variant_index=variant_index, nums=nums)
    if result.success:
        state[key] = variant_index + 1
        setattr(generator, "_scene_layout_solver_state", state)
    setattr(generator, "_last_scene_layout_solver_result", result)
    return result


def _inject_background_poses(generator: Any, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("background_poses"):
        return payload
    poses: dict[str, dict[str, list[float]]] = {}
    for component in _generator_components(generator):
        if not _is_background_component(component):
            continue
        cfg = getattr(component, "cfg", None)
        if cfg is None:
            continue
        tag = str(getattr(cfg, "asset_tag", "") or "").strip()
        if not tag:
            continue
        position = _float_list(getattr(cfg, "initial_position", None), length=3)
        orientation = _float_list(getattr(cfg, "initial_orientation", None), length=4)
        if position is None or orientation is None:
            continue
        poses[tag] = {"position": position, "orientation": orientation}
    if not poses:
        LOGGER.debug("scene_layout_solver found no background component poses to inject")
        return payload
    out = dict(payload)
    out["background_poses"] = poses
    return out


def _is_background_component(component: Any) -> bool:
    cfg = getattr(component, "cfg", None)
    if bool(getattr(component, "is_background", False)) or bool(getattr(cfg, "is_background", False)):
        return True
    if bool(getattr(component, "is_background_component", False)) or bool(
        getattr(cfg, "is_background_component", False)
    ):
        return True
    return "background" in type(component).__name__.lower()


def _store_materialized_layout_payload(generator: Any, payload: dict[str, Any]) -> None:
    """Persist the solver-consumed layout payload for downstream task compile.

    SceneFactory copies ``gen_cfg.metadata`` onto the produced Scene after
    ``SceneGenerator.generate()``.  Updating the generator config here lets
    TaskFactory.finalize_task compile runtime sampling with the same
    background pose/materialized layout that static scene solving consumed.
    """

    config = getattr(generator, "config", None)
    metadata = getattr(config, "metadata", None)
    if not isinstance(metadata, dict):
        return
    metadata["layout"] = dict(payload)


def _generator_components(generator: Any) -> list[Any]:
    if hasattr(generator, "sorted_components"):
        try:
            return list(generator.sorted_components())
        except Exception as exc:
            LOGGER.warning(
                "scene_layout_solver sorted_components failed; falling back to components: %s",
                exc,
            )
    return list(getattr(generator, "components", []) or [])


def _float_list(raw: Any, *, length: int) -> list[float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        return None
    return [float(value) for value in raw]
