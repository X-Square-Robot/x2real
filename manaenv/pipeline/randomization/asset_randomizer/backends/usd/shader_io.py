from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ...materials.channels import MaterialChannel


@dataclass(frozen=True)
class ShaderChannelIO:
    constant_inputs: tuple[str, ...] = ()
    texture_inputs: tuple[str, ...] = ()
    strength_inputs: tuple[str, ...] = ()
    enable_flags: dict[str, Any] = field(default_factory=dict)
    enable_values: dict[str, Any] = field(default_factory=dict)
    disable_flags: dict[str, Any] = field(default_factory=dict)
    disable_values: dict[str, Any] = field(default_factory=dict)
    native_scale_inputs: tuple[str, ...] = ()
    native_bias_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShaderModelIO:
    shader_model: str
    channels: dict[str, ShaderChannelIO]
    preview_only: bool = False


OMNIPBR_CHANNELS: dict[str, ShaderChannelIO] = {
    MaterialChannel.BASE_COLOR: ShaderChannelIO(
        constant_inputs=("diffuse_color_constant", "diffuse_tint"),
        texture_inputs=("diffuse_texture",),
        disable_values={
            "diffuse_tint": (1.0, 1.0, 1.0),
            "albedo_add": 0.0,
            "albedo_brightness": 1.0,
        },
        native_scale_inputs=("diffuse_tint", "albedo_brightness"),
        native_bias_inputs=("albedo_add",),
    ),
    MaterialChannel.ROUGHNESS: ShaderChannelIO(
        constant_inputs=("reflection_roughness_constant",),
        texture_inputs=("reflectionroughness_texture",),
        enable_values={"reflection_roughness_texture_influence": 1.0},
        disable_values={"reflection_roughness_texture_influence": 0.0},
    ),
    MaterialChannel.METALLIC: ShaderChannelIO(
        constant_inputs=("metallic_constant",),
        texture_inputs=("metallic_texture",),
        enable_values={"metallic_texture_influence": 1.0},
        disable_values={"metallic_texture_influence": 0.0},
    ),
    MaterialChannel.SPECULAR: ShaderChannelIO(
        constant_inputs=("specular_level",),
    ),
    MaterialChannel.OPACITY: ShaderChannelIO(
        constant_inputs=("opacity_constant",),
        texture_inputs=("opacity_texture",),
        enable_flags={"enable_opacity_texture": True, "enable_opacity": True},
        disable_flags={"enable_opacity_texture": False},
    ),
    MaterialChannel.NORMAL: ShaderChannelIO(
        texture_inputs=("normalmap_texture",),
        strength_inputs=("bump_factor",),
        enable_values={"bump_factor": 1.0},
    ),
    MaterialChannel.ORM: ShaderChannelIO(
        texture_inputs=("ORM_texture",),
        enable_flags={"enable_ORM_texture": True},
        disable_flags={"enable_ORM_texture": False},
    ),
}


OMNIGLASS_CHANNELS: dict[str, ShaderChannelIO] = {
    MaterialChannel.BASE_COLOR: ShaderChannelIO(
        constant_inputs=("glass_color",),
        texture_inputs=("glass_color_texture",),
        enable_flags={"use_glass_color_texture": True},
        disable_flags={"use_glass_color_texture": False},
    ),
    MaterialChannel.ROUGHNESS: ShaderChannelIO(
        constant_inputs=("frosting_roughness",),
        texture_inputs=("roughness_texture",),
        enable_values={"roughness_texture_influence": 1.0},
        disable_values={"roughness_texture_influence": 0.0},
    ),
    MaterialChannel.OPACITY: ShaderChannelIO(
        constant_inputs=("cutout_opacity",),
        texture_inputs=("cutout_opacity_texture",),
    ),
    MaterialChannel.NORMAL: ShaderChannelIO(
        texture_inputs=("normal_map_texture",),
        strength_inputs=("normal_map_strength",),
        enable_values={"normal_map_strength": 1.0},
    ),
}


USD_PREVIEW_CHANNELS: dict[str, ShaderChannelIO] = {
    MaterialChannel.BASE_COLOR: ShaderChannelIO(
        constant_inputs=("diffuseColor",),
        texture_inputs=("diffuseColor",),
    ),
    MaterialChannel.METALLIC: ShaderChannelIO(constant_inputs=("metallic",)),
    MaterialChannel.ROUGHNESS: ShaderChannelIO(constant_inputs=("roughness",)),
    MaterialChannel.OPACITY: ShaderChannelIO(constant_inputs=("opacity",)),
    MaterialChannel.NORMAL: ShaderChannelIO(texture_inputs=("normal",)),
}


SHADER_MODELS: dict[str, ShaderModelIO] = {
    "OmniPBR": ShaderModelIO("OmniPBR", OMNIPBR_CHANNELS),
    "OmniPBR_ClearCoat": ShaderModelIO("OmniPBR_ClearCoat", OMNIPBR_CHANNELS),
    "OmniGlass": ShaderModelIO("OmniGlass", OMNIGLASS_CHANNELS),
    "UsdPreviewSurface": ShaderModelIO("UsdPreviewSurface", USD_PREVIEW_CHANNELS, preview_only=True),
}

SHADER_CONSTANT_INPUTS: dict[str, dict[str, tuple[str, ...]]] = {
    shader_model: {
        channel: io.constant_inputs + io.strength_inputs
        for channel, io in model.channels.items()
        if io.constant_inputs or io.strength_inputs
    }
    for shader_model, model in SHADER_MODELS.items()
}


SHADER_TEXTURE_INPUTS: dict[str, dict[str, tuple[str, ...]]] = {
    shader_model: {channel: io.texture_inputs for channel, io in model.channels.items() if io.texture_inputs}
    for shader_model, model in SHADER_MODELS.items()
}


TEXTURE_INPUT_NAMES = frozenset(
    input_name
    for shader_inputs in SHADER_TEXTURE_INPUTS.values()
    for names in shader_inputs.values()
    for input_name in names
) | frozenset(
    {
        "diffuse_texture",
        "base_color_texture",
        "albedo_texture",
        "normalmap_texture",
        "normal_texture",
        "ORM_texture",
    }
)


def canonical_shader_model(
    shader_id: str,
    input_names: Iterable[str] = (),
    *,
    effective_shader_type: str | None = None,
) -> str:
    if effective_shader_type in SHADER_MODELS:
        return str(effective_shader_type)
    if "UsdPreviewSurface" in shader_id:
        return "UsdPreviewSurface"
    if "OmniPBR_ClearCoat" in shader_id:
        return "OmniPBR_ClearCoat"
    if "OmniGlass" in shader_id:
        return "OmniGlass"
    if "OmniPBR" in shader_id or "OmniPBRBase" in shader_id:
        return "OmniPBR"
    names = set(input_names)
    for shader_model in (
        "OmniPBR_ClearCoat",
        "OmniPBR",
        "OmniGlass",
        "UsdPreviewSurface",
    ):
        known = set()
        for candidates in SHADER_CONSTANT_INPUTS.get(shader_model, {}).values():
            known.update(candidates)
        for candidates in SHADER_TEXTURE_INPUTS.get(shader_model, {}).values():
            known.update(candidates)
        for io in SHADER_MODELS[shader_model].channels.values():
            known.update(io.disable_flags)
            known.update(io.disable_values)
            known.update(io.enable_flags)
            known.update(io.enable_values)
            known.update(io.native_scale_inputs)
            known.update(io.native_bias_inputs)
        if names.intersection(known):
            return shader_model
    return shader_id or "unknown"


def shader_model_io(shader_model: str) -> ShaderModelIO | None:
    return SHADER_MODELS.get(shader_model)


def channel_io(shader_model: str, channel: str) -> ShaderChannelIO | None:
    model = shader_model_io(shader_model)
    if model is None:
        return None
    return model.channels.get(channel)


def channel_input_names(shader_model: str) -> dict[str, tuple[str, ...]]:
    return SHADER_CONSTANT_INPUTS.get(shader_model, {})


def channel_texture_input_names(shader_model: str) -> dict[str, tuple[str, ...]]:
    return SHADER_TEXTURE_INPUTS.get(shader_model, {})


def read_input_value(shader: Any, input_names: tuple[str, ...]) -> Any:
    for input_name in input_names:
        shader_input = shader.GetInput(input_name)
        if shader_input:
            value = shader_input.Get()
            if value is not None:
                try:
                    return tuple(float(component) for component in value)
                except TypeError:
                    return value
    return None


def input_has_connected_source(shader_input: Any) -> bool:
    try:
        return bool(shader_input.HasConnectedSource())
    except Exception:
        try:
            sources, invalid = shader_input.GetConnectedSources()
            return bool(sources or invalid)
        except Exception:
            return False


def input_connected_source_path(shader_input: Any) -> str | None:
    try:
        sources, invalid = shader_input.GetConnectedSources()
    except Exception:
        return None
    source = sources[0] if sources else None
    if source and source.source and source.source.GetPrim().IsValid():
        return str(source.source.GetPrim().GetPath())
    if invalid:
        return str(invalid[0])
    return None


def input_asset_path(shader_input: Any) -> str | None:
    value = shader_input.Get()
    if value is None:
        return None
    path = str(getattr(value, "path", None) or value).strip()
    return path or None


def input_resolved_asset_path(shader_input: Any) -> str | None:
    value = shader_input.Get()
    if value is None:
        return None
    path = str(getattr(value, "resolvedPath", None) or "").strip()
    return path or None


def has_texture_source(shader: Any, input_names: Iterable[str] = TEXTURE_INPUT_NAMES) -> bool:
    for input_name in input_names:
        shader_input = shader.GetInput(input_name)
        if not shader_input:
            continue
        if input_has_connected_source(shader_input):
            return True
        value = shader_input.Get()
        if value is not None:
            return True
    return False
