from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from tempfile import gettempdir
from typing import Any

from ..api import randomize_asset_appearance
from ..backends.usd import UsdSceneBackend
from ..backends.usd.oqc_bridge import path_within, semantic_tags_for_prim
from ..config.wrapper import compile_material_randomization_config
from ..reporting import concise_randomization_report
from ..rng import context_for_reset

DEFAULT_RUNTIME_CACHE_ROOT = Path(gettempdir()) / "manaenv_asset_randomizer" / "runtime_cache"
REPORT_ATTR = "_mana_material_randomization_reports"
RESET_COUNTS_ATTR = "_mana_material_randomization_reset_counts"
_WARNED_RUNTIME_KEYS: set[str] = set()


def attach_material_randomization_event_for_task(task: Any) -> None:
    runtime_cfgs = getattr(task, "runtime_cfgs", None)
    if not isinstance(runtime_cfgs, dict):
        return
    material_cfg = runtime_cfgs.get("material_randomization_cfg")
    if not isinstance(material_cfg, Mapping) or not bool(material_cfg.get("enabled", True)):
        return
    raw_config = material_cfg.get("config")
    if not isinstance(raw_config, Mapping):
        return

    event_terms = runtime_cfgs.get("event_terms")
    field_items = _event_field_items(event_terms)

    if any(name == "material_randomization" for name, _type, _value in field_items):
        return

    from isaaclab.managers import EventTermCfg
    from isaaclab_arena.utils.configclass import make_configclass

    task_definition = getattr(task, "definition", {}) if isinstance(getattr(task, "definition", {}), dict) else {}
    base_seed = material_cfg.get("seed")
    if base_seed is None:
        base_seed = task_definition.get("seed", 0)
    runtime_options = material_cfg.get("runtime") if isinstance(material_cfg.get("runtime"), Mapping) else {}
    field_items.append(
        (
            "material_randomization",
            EventTermCfg,
            EventTermCfg(
                func=randomize_materials_on_reset,
                mode="reset",
                params={
                    "config": deepcopy(dict(raw_config)),
                    "base_seed": int(base_seed or 0),
                    "task_id": task_definition.get("task_type"),
                    "cache_root": runtime_options.get("cache_root"),
                    "report_limit": int(runtime_options.get("report_limit", 64)),
                    "reset_renderer_accumulation_after_apply": bool(
                        runtime_options.get("reset_renderer_accumulation_after_apply", True)
                    ),
                    "reset_rtx_sensors_after_apply": bool(runtime_options.get("reset_rtx_sensors_after_apply", True)),
                },
            ),
        )
    )
    runtime_cfgs["event_terms"] = make_configclass("EventsCfg", field_items)()


def _event_field_items(event_terms: Any) -> list[tuple[str, type, Any]]:
    if event_terms is None:
        return []
    fields = getattr(event_terms, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        return [(name, type(getattr(event_terms, name)), getattr(event_terms, name)) for name in fields]
    return [(name, type(value), value) for name, value in vars(event_terms).items()]


def randomize_materials_on_reset(
    env: Any,
    env_ids: Any,
    *,
    config: Mapping[str, Any],
    base_seed: int = 0,
    task_id: str | None = None,
    cache_root: str | None = None,
    report_limit: int = 64,
    reset_renderer_accumulation_after_apply: bool = True,
    reset_rtx_sensors_after_apply: bool = True,
) -> None:
    if env_ids is None:
        return
    stage = _current_stage()
    env_id_list = _env_id_list(env_ids)
    if not env_id_list:
        return
    reset_counts = _reset_counts(env)
    for env_id in env_id_list:
        reset_index = int(reset_counts.get(env_id, 0))
        reset_counts[env_id] = reset_index + 1
        env_config = _config_for_env(config, env=env, stage=stage, env_id=env_id)
        compiled = compile_material_randomization_config(env_config)
        if not compiled.spec.enabled:
            continue
        scoped_reset_index = 0 if compiled.spec.seed_scope == "run" else reset_index
        context = context_for_reset(
            base_seed=int(base_seed),
            reset_index=scoped_reset_index,
            env_id=env_id,
            task_id=task_id,
        )
        backend = UsdSceneBackend(
            stage,
            generated_texture_dir=_generated_texture_dir(env=env, cache_root=cache_root),
        )
        report = randomize_asset_appearance(
            backend=backend,
            spec=compiled.spec,
            context=context,
            library=compiled.libraries,
        )
        _append_report(
            env,
            {
                "env_id": env_id,
                "reset_index": reset_index,
                "report": concise_randomization_report(report),
            },
            report_limit=report_limit,
        )
    if reset_renderer_accumulation_after_apply:
        _reset_renderer_accumulation()
    if reset_rtx_sensors_after_apply:
        _reset_rtx_sensors(env, env_id_list)


def _current_stage() -> Any:
    from isaaclab.sim.utils import get_current_stage

    return get_current_stage()


def _env_id_list(env_ids: Any) -> list[int]:
    if hasattr(env_ids, "detach"):
        env_ids = env_ids.detach().cpu().tolist()
    if isinstance(env_ids, int):
        return [int(env_ids)]
    if isinstance(env_ids, Sequence):
        return [int(item) for item in env_ids]
    return [int(env_ids)]


def _reset_counts(env: Any) -> dict[int, int]:
    counts = getattr(env, RESET_COUNTS_ATTR, None)
    if not isinstance(counts, dict):
        counts = {}
        setattr(env, RESET_COUNTS_ATTR, counts)
    return counts


def _reset_rtx_sensors(env: Any, env_ids: Sequence[int]) -> None:
    scene = getattr(env, "scene", None)
    sensors = getattr(scene, "sensors", None)
    if not isinstance(sensors, Mapping):
        return
    for sensor in sensors.values():
        if not hasattr(sensor, "render_product_paths"):
            continue
        reset = getattr(sensor, "reset", None)
        if callable(reset):
            reset(env_ids)


def _reset_renderer_accumulation() -> bool:
    """Reset renderer history without changing steady-state render settings."""
    context = _current_usd_context()
    if context is None:
        _warn_once(
            "renderer_accumulation_no_context",
            "Unable to reset renderer accumulation after material randomization: omni.usd context is unavailable.",
        )
        return False
    for method_name in ("reset_renderer_accumulation", "resetRendererAccumulation"):
        reset = getattr(context, method_name, None)
        if callable(reset):
            reset()
            return True
    _warn_once(
        "renderer_accumulation_no_api",
        "Unable to reset renderer accumulation after material randomization: omni.usd context has no reset API.",
    )
    return False


def _current_usd_context() -> Any | None:
    try:
        import omni.usd  # type: ignore[import]
    except Exception:
        return None
    get_context = getattr(omni.usd, "get_context", None)
    if not callable(get_context):
        return None
    return get_context()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_RUNTIME_KEYS:
        return
    _WARNED_RUNTIME_KEYS.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _non_negative_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


def _config_for_env(config: Mapping[str, Any], *, env: Any, stage: Any, env_id: int) -> dict[str, Any]:
    env_config = deepcopy(dict(config))
    for section_name in ("material_randomization_profiles", "advanced_rules"):
        entries = env_config.get(section_name)
        if isinstance(entries, Mapping):
            _rewrite_entry_targets(entries, env=env, stage=stage, env_id=env_id)
        elif isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
            for entry in entries:
                if isinstance(entry, Mapping):
                    _rewrite_entry_targets(entry, env=env, stage=stage, env_id=env_id)
    return env_config


def _rewrite_entry_targets(entry: Mapping[str, Any], *, env: Any, stage: Any, env_id: int) -> None:
    targets = entry.get("targets")
    if not isinstance(targets, dict):
        return
    prim_paths = list(_as_string_list(targets.get("prim_paths")))
    for name in _as_string_list(targets.pop("asset_names", None)):
        prim_paths.extend(_prim_paths_for_asset_name(env=env, stage=stage, asset_name=name, env_id=env_id))
    for name in _as_string_list(targets.pop("asset_ids", None)):
        prim_paths.extend(_prim_paths_for_asset_name(env=env, stage=stage, asset_name=name, env_id=env_id))
    semantic_tags = tuple(
        _as_string_list(targets.pop("semantic_tags", None)) + _as_string_list(targets.pop("semantic_tags_any", None))
    )
    if semantic_tags:
        prim_paths.extend(_prim_paths_for_semantic_tags(stage=stage, env_id=env_id, tags=semantic_tags))
    if prim_paths:
        targets["prim_paths"] = sorted(set(_expand_env_tokens(path, env_id=env_id) for path in prim_paths))


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value]
    return [str(value)]


def _prim_paths_for_asset_name(*, env: Any, stage: Any, asset_name: str, env_id: int) -> list[str]:
    scene = getattr(env, "scene", None)
    if scene is not None:
        try:
            asset = scene[asset_name]
            prim_path = getattr(getattr(asset, "cfg", None), "prim_path", None)
            if isinstance(prim_path, str) and prim_path:
                return [_expand_env_tokens(prim_path, env_id=env_id)]
        except (KeyError, TypeError, AttributeError):
            pass
    env_root = f"/World/envs/env_{env_id}"
    matches: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path_within(path, env_root) and prim.GetName() == asset_name:
            matches.append(path)
    return matches


def _prim_paths_for_semantic_tags(*, stage: Any, env_id: int, tags: tuple[str, ...]) -> list[str]:
    wanted = set(tags)
    env_root = f"/World/envs/env_{env_id}"
    matches: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path_within(path, env_root):
            continue
        if wanted.intersection(semantic_tags_for_prim(prim)):
            matches.append(path)
    return matches


def _expand_env_tokens(path: str, *, env_id: int) -> str:
    return (
        str(path)
        .replace("{ENV_REGEX_NS}", f"envs/env_{env_id}")
        .replace("env_.*", f"env_{env_id}")
        .replace("{ENV_NS}", f"envs/env_{env_id}")
    )


def _generated_texture_dir(*, env: Any, cache_root: str | None) -> Path:
    del env
    if cache_root:
        root = Path(cache_root).expanduser()
    else:
        root = DEFAULT_RUNTIME_CACHE_ROOT
    path = root / "asset_randomizer" / "generated_textures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_report(env: Any, item: dict[str, Any], *, report_limit: int) -> None:
    reports = getattr(env, REPORT_ATTR, None)
    if not isinstance(reports, list):
        reports = []
        setattr(env, REPORT_ATTR, reports)
    reports.append(item)
    if report_limit > 0 and len(reports) > report_limit:
        del reports[: len(reports) - report_limit]
