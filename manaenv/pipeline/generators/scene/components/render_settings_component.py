"""Render-settings component for scene generation.

YAML example (schema-vNext)::

    render_settings:
      asset: "green_booth_noblock"         # resolved from the local asset root
      # usda_path: "/explicit/path/to/file.settings.usda"  # optional override
      antialiasing_mode: DLAA             # Off | FXAA | DLSS | TAA | DLAA (default DLSS)
      enable_dlssg: true                  # interactive Kit/WebRTC only; defaults to x2
      # dlssg_multiplier: 4               # optional x3/x4 override with GPU capability check
      # dlss_mode: 2                      # only with DLSS: 0 Performance | 1 Balanced | 2 Quality | 3 Auto
      carb_settings:                      # optional advanced Isaac RTX settings
        rtx.reflections.enabled: true

Legacy scene order::

    render_settings_component:
      asset: green_booth
      antialiasing_mode: DLSS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from manaenv.pipeline.generators.scene.component import SceneComponent, SceneComponentCFG
from manaenv.pipeline.generators.scene_generator import DEFAULT_ANTIALIASING_MODE, normalize_antialiasing_mode
from manaenv.utils.debug_print import manaprint

_NAMED_RENDER_SETTING_KEYS = {
    "enable_dlssg": "rtx-transient.dlssg.enabled",
    "dlss_mode": "rtx.post.dlss.execMode",
}


def _normalize_render_setting_key(key: str) -> tuple[str, str]:
    """Return dotted renderer and slash-prefixed Carb forms of one setting key."""
    normalized = str(key).strip()
    if not normalized:
        raise ValueError("render carb setting keys must be non-empty")
    if normalized.startswith("/"):
        dotted = normalized[1:].replace("/", ".")
    else:
        dotted = normalized.replace(":", ".").replace("/", ".")
    return dotted, "/" + dotted.replace(".", "/")


def _normalize_inline_carb_settings(settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize public inline settings for IsaacLab and direct Carb application."""
    if not isinstance(settings, dict):
        raise TypeError(f"carb_settings must be a mapping, got {type(settings).__name__}")
    render_cfg_dict: dict[str, Any] = {}
    carb_dict: dict[str, Any] = {}
    for key, value in settings.items():
        render_key, carb_key = _normalize_render_setting_key(key)
        render_cfg_dict[render_key] = value
        carb_dict[carb_key] = value
    return render_cfg_dict, carb_dict


def _usda_value_to_python(value: Any) -> Any:
    """Convert USD layer value to Python native type for carb."""
    if hasattr(value, "Get"):
        value = value.Get()
    if isinstance(value, (bool, int, float, str)):
        return value
    for cast in (float, bool, int):
        try:
            return cast(value)
        except (TypeError, ValueError):
            continue
    return value


def load_render_settings_from_usda(
    usda_path: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read ``customLayerData.renderSettings`` from a ``.settings.usda`` file."""
    if not usda_path:
        return {}, {}
    try:
        from pxr import Sdf  # type: ignore[import]

        layer = Sdf.Layer.FindOrOpen(str(usda_path))
        if not layer:
            return {}, {}
        render_dict = layer.customLayerData.get("renderSettings") or {}
        if not isinstance(render_dict, dict) or not render_dict:
            return {}, {}
        render_cfg_dict: dict[str, Any] = {}
        carb_dict: dict[str, Any] = {}
        for key, value in render_dict.items():
            py_val = _usda_value_to_python(value)
            render_cfg_dict[key.replace("rtx:", "rtx.").replace(":", ".")] = py_val
            carb_dict["/" + key.replace(":", "/")] = py_val
        return render_cfg_dict, carb_dict
    except Exception as exc:  # noqa: BLE001
        manaprint(f"WARNING: Failed to load render settings from '{usda_path}': {exc}")
        return {}, {}


def _filter_unknown_carb_settings(
    render_cfg_dict: dict[str, Any],
    carb_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop carb keys that the running IsaacSim build does not recognize."""
    if not render_cfg_dict and not carb_dict:
        return render_cfg_dict, carb_dict
    try:
        import carb.settings  # type: ignore[import]

        settings_iface = carb.settings.get_settings()
    except Exception as exc:  # noqa: BLE001
        manaprint(f"DEBUG: carb settings unavailable -- skipping render-settings filter ({exc})")
        return render_cfg_dict, carb_dict

    filtered_render: dict[str, Any] = {}
    filtered_carb: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in render_cfg_dict.items():
        if "_" in key:
            carb_key = "/" + key.replace("_", "/")
        elif "." in key:
            carb_key = "/" + key.replace(".", "/")
        else:
            carb_key = "/" + key
        try:
            present = settings_iface.get(carb_key) is not None
        except Exception:
            present = False
        if present:
            filtered_render[key] = value
        else:
            dropped.append(carb_key)
    for key, value in carb_dict.items():
        try:
            present = settings_iface.get(key) is not None
        except Exception:
            present = False
        if present:
            filtered_carb[key] = value
    if dropped:
        manaprint(
            f"WARNING: dropped {len(dropped)} unrecognized render-settings key(s) "
            f"for this IsaacSim build (e.g. {dropped[0]})"
        )
    return filtered_render, filtered_carb


def _resolve_render_settings_path(asset: str, usda_path: str | None = None) -> str | None:
    """Resolve a render-settings USDA path from *asset* or *usda_path*."""
    if usda_path:
        return usda_path

    if not asset:
        return None

    # Render settings files follow the convention {name}.settings.usda.
    try:
        from manaenv.assets.local_discovery import resolve_usd_path

        url = resolve_usd_path(f"{asset}.settings")
        manaprint(f"INFO: Resolved render settings '{asset}' -> {url}")
        return url
    except KeyError:
        manaprint(f"WARNING: Local render settings '{asset}.settings' not found")
    except Exception as exc:  # noqa: BLE001
        manaprint(f"WARNING: Failed to resolve render settings '{asset}': {exc}")
    return None


class RenderSettingsComponent(SceneComponent):
    """Loads USDA and inline render settings and injects them into sim config."""

    def __init__(self, cfg: RenderSettingsComponentCFG | None = None):
        super().__init__(cfg)

    def generate(self, cfg: RenderSettingsComponentCFG | None = None) -> SimpleNamespace | None:
        if cfg is not None:
            self.cfg = cfg

        resolved = _resolve_render_settings_path(self.cfg.asset, self.cfg.usda_path)
        render_cfg_dict, carb_dict = load_render_settings_from_usda(resolved)

        inline_render, inline_carb = _normalize_inline_carb_settings(self.cfg.carb_settings)
        render_cfg_dict.update(inline_render)
        carb_dict.update(inline_carb)

        for field_name, render_key in _NAMED_RENDER_SETTING_KEYS.items():
            value = getattr(self.cfg, field_name)
            if value is None:
                continue
            _, carb_key = _normalize_render_setting_key(render_key)
            render_cfg_dict[render_key] = value
            carb_dict[carb_key] = value

        if not render_cfg_dict and not carb_dict:
            source = f"'{resolved}'" if resolved else "YAML"
            manaprint(f"WARNING: RenderSettingsComponent: no settings loaded from {source}")
            return None

        render_cfg_dict, carb_dict = _filter_unknown_carb_settings(render_cfg_dict, carb_dict)
        source = f"'{resolved}' and YAML overrides" if resolved else "YAML"
        manaprint(f"INFO: Loaded {len(render_cfg_dict)} render settings from {source}")
        if self.cfg.enable_dlssg:
            manaprint(
                "WARNING: DLSS Frame Generation is intended for interactive Kit/WebRTC display; "
                "it does not affect offscreen evaluation observations."
            )
        return SimpleNamespace(
            _is_render_settings=True,
            render_cfg_dict=render_cfg_dict,
            carb_dict=carb_dict,
            dlssg_multiplier=self.cfg.dlssg_multiplier,
        )


@dataclass
class RenderSettingsComponentCFG(SceneComponentCFG):
    """Configuration for render settings component."""

    class_type: type = RenderSettingsComponent
    usda_path: str = ""
    # Empty keeps SceneGeneratorCFG.antialiasing_mode; non-empty overrides it.
    antialiasing_mode: str = ""
    # IsaacLab's named Isaac RTX global settings. None preserves the app/preset value.
    enable_dlssg: bool | None = None
    dlssg_multiplier: int | None = None
    dlss_mode: int | None = None
    # Advanced escape hatch matching IsaacRtxRendererGlobalSettingsCfg.carb_settings.
    carb_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dlssg_multiplier is None and self.enable_dlssg is True:
            self.dlssg_multiplier = 2
        if self.dlssg_multiplier is not None:
            if (
                isinstance(self.dlssg_multiplier, bool)
                or not isinstance(self.dlssg_multiplier, int)
                or self.dlssg_multiplier not in (2, 3, 4)
            ):
                raise ValueError(f"dlssg_multiplier must be one of 2, 3, or 4; got {self.dlssg_multiplier!r}")
            if self.enable_dlssg is False:
                raise ValueError("enable_dlssg cannot be false when dlssg_multiplier is set")
            self.enable_dlssg = True
        if str(self.antialiasing_mode or "").strip():
            self.antialiasing_mode = normalize_antialiasing_mode(
                self.antialiasing_mode, default=DEFAULT_ANTIALIASING_MODE
            )
        if self.enable_dlssg is not None and not isinstance(self.enable_dlssg, bool):
            raise TypeError(f"enable_dlssg must be a boolean or null, got {self.enable_dlssg!r}")
        if self.dlss_mode is not None:
            if (
                isinstance(self.dlss_mode, bool)
                or not isinstance(self.dlss_mode, int)
                or self.dlss_mode not in range(4)
            ):
                raise ValueError(f"dlss_mode must be one of 0, 1, 2, or 3; got {self.dlss_mode!r}")
        if not isinstance(self.carb_settings, dict):
            raise TypeError(f"carb_settings must be a mapping, got {type(self.carb_settings).__name__}")
