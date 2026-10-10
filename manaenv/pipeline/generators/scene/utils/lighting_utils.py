"""Utilities for managing lights embedded in USD assets and light cfg helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from manaenv.utils.debug_print import manaprint

if TYPE_CHECKING:
    from pxr import Usd

_ENV_PRIM_PREFIX = "/World/envs/env_"


def dome_lights_disabled() -> bool:
    """True when ``MANAENV_DISABLE_DOME_LIGHT`` requests an all-dome-off preview."""
    return os.environ.get("MANAENV_DISABLE_DOME_LIGHT", "").strip().lower() in {"1", "true", "yes"}


def tabletop_rect_lights_enabled() -> bool:
    """True when ``MANAENV_ENABLE_TABLETOP_RECT_LIGHT`` keeps local desk RectLights.

    Default is off: benchmark wrappers author a high-intensity
    ``lighting:role=tabletop_local_key`` RectLight that fights the shared global
    DomeLight. Opt back in with ``MANAENV_ENABLE_TABLETOP_RECT_LIGHT=1``.
    """
    return os.environ.get("MANAENV_ENABLE_TABLETOP_RECT_LIGHT", "").strip().lower() in {"1", "true", "yes"}


def build_rect_light_scene_config(
    *,
    color: Sequence[float],
    intensity: float,
    width: float,
    height: float,
    position: Sequence[float],
    orientation: Sequence[float] | None,
    prim_path: str,
    entry_name: str = "rect_light",
) -> SimpleNamespace:
    """Build a scene config namespace with one RectLight asset entry."""
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.spawners.lights import LightCfg
    from isaaclab.utils.configclass import configclass

    @configclass
    class RectLightCfg(LightCfg):
        prim_type: str = "RectLight"
        width: float = 1.0
        height: float = 1.0

    init_state_kwargs: dict[str, tuple[float, ...]] = {
        "pos": tuple(float(v) for v in position),
    }
    if orientation:
        init_state_kwargs["rot"] = tuple(float(v) for v in orientation)

    return SimpleNamespace(
        _is_scene_config=True,
        entries={
            entry_name: AssetBaseCfg(
                prim_path=prim_path,
                spawn=RectLightCfg(
                    color=tuple(float(v) for v in color),
                    intensity=float(intensity),
                    width=float(width),
                    height=float(height),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(**init_state_kwargs),
            )
        },
    )


def deactivate_embedded_lights(stage: Usd.Stage, root_prim_path: str) -> list[str]:
    """Traverse *root_prim_path* and deactivate any UsdLux light prims found.

    Background USD files sometimes embed global lights (DomeLight, DistantLight, etc.)
    which conflict with the Python-side lighting managed by ``LightingComponent``.
    Calling this function after the prims have been spawned ensures those embedded
    lights are suppressed without modifying the source USD files.

    Args:
        stage: The active USD stage.
        root_prim_path: Prim path to start the traversal from (e.g.
            ``"/World/envs/env_0/packing_table"``).

    Returns:
        List of prim paths that were deactivated.
    """
    from pxr import Usd, UsdLux

    deactivated: list[str] = []
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        manaprint(f"WARNING: deactivate_embedded_lights – prim not found: {root_prim_path}")
        return deactivated

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdLux.LightAPI):
            prim.SetActive(False)
            path_str = str(prim.GetPath())
            deactivated.append(path_str)

    return deactivated


def env_index_from_prim_path(prim_path: str) -> int | None:
    """Return the ``/World/envs/env_N`` index owning *prim_path*, else ``None``."""
    if not prim_path.startswith(_ENV_PRIM_PREFIX):
        return None
    remainder = prim_path[len(_ENV_PRIM_PREFIX) :]
    index_str = remainder.split("/", 1)[0]
    return int(index_str) if index_str.isdigit() else None


def _silence_dome_prim(prim: Any, *, usd_lux: Any) -> str:
    """Zero intensity then deactivate; return the silenced prim path."""
    usd_lux.DomeLight(prim).CreateIntensityAttr(0.0)
    prim.SetActive(False)
    return str(prim.GetPath())


def _silence_rect_prim(prim: Any, *, usd_lux: Any) -> str:
    """Zero intensity then deactivate; return the silenced prim path."""
    usd_lux.RectLight(prim).CreateIntensityAttr(0.0)
    prim.SetActive(False)
    return str(prim.GetPath())


def silence_tabletop_rect_lights(
    stage: Usd.Stage,
    *,
    usd_lux: Any = None,
    enabled: bool | None = None,
) -> list[str]:
    """Silence ``lighting:role=tabletop_local_key`` RectLights (default on).

    Benchmark background wrappers embed a ceiling RectLight per env. With a
    shared global DomeLight those local keys add harsh, often over-exposed desk
    fill and are not needed for dome-only preview/eval. Intensity is zeroed
    before deactivation so a silenced prim contributes nothing if the render
    delegate retains it.

    Pass ``enabled=True`` or set ``MANAENV_ENABLE_TABLETOP_RECT_LIGHT=1`` to keep
    the authored RectLights.

    Returns:
        Prim paths that were silenced.
    """
    if enabled is None:
        enabled = tabletop_rect_lights_enabled()
    if enabled:
        return []
    if usd_lux is None:
        from pxr import UsdLux

        usd_lux = UsdLux

    silenced: list[str] = []
    for prim in stage.Traverse() or ():
        if not prim.IsActive() or not prim.IsDefined() or prim.IsInPrototype():
            continue
        if not prim.IsA(usd_lux.RectLight):
            continue
        role_attr = prim.GetAttribute("lighting:role")
        if not role_attr.IsValid() or str(role_attr.Get() or "") != "tabletop_local_key":
            continue
        silenced.append(_silence_rect_prim(prim, usd_lux=usd_lux))
    if silenced:
        manaprint(
            f"INFO: Silenced {len(silenced)} tabletop RectLight(s) "
            f"(dome-only; set MANAENV_ENABLE_TABLETOP_RECT_LIGHT=1 to keep): "
            f"{', '.join(silenced)}"
        )
    return silenced


def collapse_cloned_dome_lights(
    stage: Usd.Stage,
    *,
    num_envs: int,
    usd_lux: Any = None,
    silence_all: bool | None = None,
) -> list[str]:
    """Silence env-cloned DomeLights so every env shares one global sky.

    A DomeLight is an infinite light: it has no position and no distance falloff,
    so a dome cloned under ``/World/envs/env_N`` does not light only that env.
    Every clone lights the whole stage and their illumination adds up, which both
    multiplies exposure by the env count and mixes each clone's HDRI. Backgrounds
    that embed their own dome therefore need collapsing back to a single light
    once the stage has been cloned across envs.

    Keeps the dome under the lowest env index and silences the rest. Intensity is
    zeroed before deactivation so a silenced clone contributes nothing even if the
    render delegate retains a deactivated prim.

    Domes outside ``/World/envs`` (e.g. a ``LightingComponent`` dome at
    ``/World/DomeLight``) are left untouched unless ``silence_all`` /
    ``MANAENV_DISABLE_DOME_LIGHT`` is set (preview: turn every DomeLight off so
    only local tabletop lights remain). A stage where one env owns more than one
    dome is left untouched as well, so genuinely ambiguous authoring still fails
    closed downstream instead of being silently masked here.

    Returns:
        Prim paths that were silenced.
    """
    if silence_all is None:
        silence_all = dome_lights_disabled()
    if usd_lux is None:
        from pxr import UsdLux

        usd_lux = UsdLux

    if silence_all:
        # Include inactive traversal? Only active domes contribute; Traverse
        # already skips inactive. Also catch stage-scoped /World domes.
        silenced: list[str] = []
        for prim in list(stage.Traverse() or ()):
            if not prim.IsActive() or not prim.IsDefined() or prim.IsInPrototype():
                continue
            if not prim.IsA(usd_lux.DomeLight):
                continue
            silenced.append(_silence_dome_prim(prim, usd_lux=usd_lux))
        if silenced:
            manaprint(
                f"INFO: MANAENV_DISABLE_DOME_LIGHT — silenced {len(silenced)} DomeLight(s): " f"{', '.join(silenced)}"
            )
        return silenced

    if int(num_envs) < 2:
        return []

    domes_by_env: dict[int, list[tuple[str, Any]]] = {}
    for prim in stage.Traverse() or ():
        if not prim.IsActive() or not prim.IsDefined() or prim.IsInPrototype():
            continue
        if not prim.IsA(usd_lux.DomeLight):
            continue
        env_index = env_index_from_prim_path(str(prim.GetPath()))
        if env_index is None:
            continue
        domes_by_env.setdefault(env_index, []).append((str(prim.GetPath()), prim))

    if len(domes_by_env) < 2:
        return []
    if any(len(entries) > 1 for entries in domes_by_env.values()):
        manaprint(
            "WARNING: collapse_cloned_dome_lights left the stage untouched: "
            "at least one env owns multiple DomeLights, which is an authoring error."
        )
        return []

    kept_env, *cloned_envs = sorted(domes_by_env)
    kept_path = domes_by_env[kept_env][0][0]
    silenced = []
    for env_index in cloned_envs:
        _path, prim = domes_by_env[env_index][0]
        silenced.append(_silence_dome_prim(prim, usd_lux=usd_lux))

    manaprint(f"INFO: Collapsed {len(silenced)} cloned DomeLight(s) into shared global sky '{kept_path}'")
    return silenced
