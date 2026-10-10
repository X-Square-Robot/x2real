from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from manaenv.pipeline.generators.scene.component import SceneComponent, SceneComponentCFG
from manaenv.pipeline.generators.scene.utils.lighting_utils import build_rect_light_scene_config
from manaenv.utils.debug_print import manaprint


class LightingComponent(SceneComponent):

    def __init__(self, cfg: LightingComponentCFG | None = None):
        super().__init__(cfg)

    def generate(self, cfg: LightingComponentCFG) -> Any:
        if cfg is not None:
            self.cfg = cfg
        c = self.cfg
        prim_path = self._resolve_prim_path(c)
        event_entries = self._build_event_entries(c, prim_path)
        if c.type == "domelight":
            from isaaclab.assets import AssetBaseCfg
            from isaaclab.sim.spawners.lights import DomeLightCfg

            dome_kwargs: dict[str, Any] = {
                "color": tuple(float(v) for v in c.color),
                "intensity": float(c.intensity),
            }
            if c.exposure is not None:
                dome_kwargs["exposure"] = float(c.exposure)
            texture_file = c.texture_file or self._resolve_texture_file(c.texture)
            if texture_file:
                dome_kwargs["texture_file"] = texture_file

            return SimpleNamespace(
                _is_scene_config=True,
                entries={
                    "dome_light": AssetBaseCfg(
                        prim_path=prim_path,
                        spawn=DomeLightCfg(**dome_kwargs),
                    )
                },
                event_entries=event_entries,
            )
        if c.type == "rectlight":
            result = build_rect_light_scene_config(
                color=c.color,
                intensity=c.intensity,
                width=c.width,
                height=c.height,
                position=c.position,
                orientation=c.orientation,
                prim_path=prim_path,
            )
            result.event_entries = event_entries
            return result
        return None

    @staticmethod
    def _resolve_texture_file(texture_cfg: dict[str, Any]) -> str:
        if not texture_cfg:
            return ""
        try:
            from manaenv.assets.local_discovery import resolve_texture_paths

            candidates = resolve_texture_paths(
                gen_method=int(texture_cfg.get("gen_method", 1)),
                tags=texture_cfg.get("tags"),
                category=texture_cfg.get("category", "skybox"),
            )
        except Exception as exc:
            manaprint(f"WARNING: Failed to resolve light texture {texture_cfg}: {exc}")
            return ""
        if not candidates:
            manaprint(f"WARNING: No light texture matched {texture_cfg}")
            return ""
        return candidates[0] if candidates else ""

    @staticmethod
    def _resolve_prim_path(cfg: LightingComponentCFG) -> str:
        if cfg.prim_path:
            return cfg.prim_path
        if cfg.type == "domelight":
            return "/World/DomeLight" if cfg.scope == "world" else "{ENV_REGEX_NS}/DomeLight"
        if cfg.type == "rectlight":
            return "/World/RectLight" if cfg.scope == "world" else "{ENV_REGEX_NS}/RectLight"
        return "/World/Light"

    @staticmethod
    def _build_event_entries(cfg: LightingComponentCFG, prim_path: str) -> dict[str, Any]:
        randomization = dict(cfg.randomization or {})
        texture_randomization = dict(cfg.texture_randomization or {})
        randomize_enabled = bool(randomization.get("enabled", False))
        texture_randomize_enabled = bool(texture_randomization.get("enabled", False))
        if not randomize_enabled and not texture_randomize_enabled:
            return {}

        from isaaclab.managers import EventTermCfg

        from manaenv.terms.events import randomize_scene_lighting

        params: dict[str, Any] = {"prim_paths": [prim_path]}
        for key in ("intensity_range", "exposure_range", "color_range", "rotation_rpy_range"):
            if key in randomization:
                params[key] = randomization[key]

        if texture_randomize_enabled:
            if texture_randomization.get("candidates"):
                params["texture_file_candidates"] = texture_randomization["candidates"]
            resolved_candidates = LightingComponent._resolve_texture_candidates(texture_randomization)
            if resolved_candidates:
                params.setdefault("texture_file_candidates", [])
                params["texture_file_candidates"].extend(resolved_candidates)

        term_cfg = EventTermCfg(
            func=randomize_scene_lighting,
            mode=randomization.get("mode", "reset"),
            params=params,
        )
        return {"randomize_scene_lighting": term_cfg}

    @staticmethod
    def _resolve_texture_candidates(texture_cfg: dict[str, Any]) -> list[str]:
        if not texture_cfg:
            return []
        try:
            from manaenv.assets.local_discovery import resolve_texture_paths

            return resolve_texture_paths(
                gen_method=int(texture_cfg.get("gen_method", 1)),
                tags=texture_cfg.get("tags"),
                category=texture_cfg.get("category", "skybox"),
            )
        except Exception as exc:
            manaprint(f"WARNING: Failed to resolve light texture candidates {texture_cfg}: {exc}")
            return []


@dataclass
class LightingComponentCFG(SceneComponentCFG):
    class_type: builtins.type = LightingComponent

    type: str = "domelight"
    # DomeLight is global scene illumination; default ``world`` places it under
    # ``/World/DomeLight`` so all env clones share one sky. Use ``env`` only for
    # per-clone lights (e.g. rect lights under ``{ENV_REGEX_NS}/...``).
    scope: str = "world"
    prim_path: str = ""
    intensity: float = 1.0
    exposure: float | None = None
    color: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    texture_file: str = ""
    texture: dict[str, Any] = field(default_factory=dict)
    texture_randomization: dict[str, Any] = field(default_factory=dict)
    position: list[float] = field(default_factory=lambda: [0.0, -0.3, 1.8])
    orientation: list[float] = field(default_factory=list)
    width: float = 1.5
    height: float = 1.0
    randomize: bool = False
    randomization: dict[str, Any] = field(default_factory=dict)
