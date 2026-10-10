from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import ConfigError
from ..materials.adjustments import MaterialLocalAdjustment, MaterialLocalAdjustmentLibrary
from ..materials.channels import ChannelSource, MaterialChannel
from ..materials.library import MaterialAsset, MaterialLibrary, RandomizationLibraries
from ..model import RandomizationSpec
from .schema import parse_randomization_spec

FORBIDDEN_V1_KEYS = frozenset({"material_randomization_strategy", "operation_mix"})
PROFILE_TYPES = frozenset({"material_template", "material_local_adjustment"})


@dataclass(frozen=True)
class CompiledMaterialRandomization:
    spec: RandomizationSpec
    libraries: RandomizationLibraries
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)


def compile_task_material_randomization(data: Mapping[str, Any]) -> CompiledMaterialRandomization:
    """Compile a ManaEnv task-local material randomization wrapper.

    The expected public shape is ``task.material_randomization``. A direct
    ``material_randomization`` mapping is accepted for adapter tests that have
    already extracted the task payload.
    """

    return compile_material_randomization_config(_extract_material_randomization(data))


def compile_material_randomization_config(config: Mapping[str, Any]) -> CompiledMaterialRandomization:
    if not isinstance(config, Mapping):
        raise ConfigError("task.material_randomization must be a mapping")
    _reject_forbidden_v1_keys(config)
    cache = _parse_cache_config(config.get("cache"))
    libraries = _parse_libraries(config.get("material_libraries"))
    rules = [
        _compile_profile(profile, libraries=libraries)
        for profile in _mapping_sequence(config.get("material_randomization_profiles"))
    ]
    rules.extend(_compile_advanced_rule(rule) for rule in _mapping_sequence(config.get("advanced_rules")))
    spec = parse_randomization_spec(
        {
            "schema_version": int(config.get("schema_version", 1)),
            "asset_randomization": {
                "enabled": bool(config.get("enabled", True)),
                "seed_scope": str(config.get("seed_scope", "reset")),
                "rules": rules,
            },
        }
    )
    return CompiledMaterialRandomization(
        spec=spec,
        libraries=libraries,
        metadata={"cache": cache},
    )


def _extract_material_randomization(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ConfigError("material randomization input must be a mapping")
    if "task" in data:
        task = data["task"]
        if isinstance(task, Mapping):
            config = task.get("material_randomization")
            if config is None:
                raise ConfigError("task.material_randomization is required")
            if not isinstance(config, Mapping):
                raise ConfigError("task.material_randomization must be a mapping")
            return config
        if isinstance(task, Sequence) and not isinstance(task, (str, bytes, bytearray)):
            candidates = [
                item["material_randomization"]
                for item in task
                if isinstance(item, Mapping) and "material_randomization" in item
            ]
            if len(candidates) != 1:
                raise ConfigError("task list must contain exactly one material_randomization entry")
            if not isinstance(candidates[0], Mapping):
                raise ConfigError("task.material_randomization must be a mapping")
            return candidates[0]
        raise ConfigError("task must be a mapping or task list")
    if "material_randomization" in data:
        config = data["material_randomization"]
        if not isinstance(config, Mapping):
            raise ConfigError("material_randomization must be a mapping")
        return config
    if "material_libraries" in data or "material_randomization_profiles" in data:
        return data
    raise ConfigError("expected task.material_randomization")


def _reject_forbidden_v1_keys(value: Any, *, path: str = "task.material_randomization") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}"
            if key_text in FORBIDDEN_V1_KEYS:
                raise ConfigError(f"{item_path} is not supported in V1")
            _reject_forbidden_v1_keys(item, path=item_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_forbidden_v1_keys(item, path=f"{path}[{index}]")


def _parse_cache_config(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"generated_textures": "run_cache"}
    if not isinstance(raw, Mapping):
        raise ConfigError("task.material_randomization.cache must be a mapping")
    cache = dict(raw)
    generated_textures = str(cache.get("generated_textures", "run_cache"))
    if generated_textures != "run_cache":
        raise ConfigError("cache.generated_textures must be run_cache in V1")
    cache["generated_textures"] = generated_textures
    return cache


def _parse_libraries(raw: Any) -> RandomizationLibraries:
    if raw is None:
        return RandomizationLibraries()
    if not isinstance(raw, Mapping):
        raise ConfigError("material_libraries must be a mapping")
    return RandomizationLibraries(
        material_library=_parse_material_template_library(raw.get("templates")),
        adjustment_library=_parse_material_local_adjustment_library(raw.get("local_adjustments")),
    )


def _parse_material_template_library(raw: Any) -> MaterialLibrary | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return MaterialLibrary(library_id=raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return MaterialLibrary(
            library_id="material_templates_v1",
            materials=tuple(_parse_material_template(item) for item in raw),
        )
    if not isinstance(raw, Mapping):
        raise ConfigError("material_libraries.templates must be a string, list, or mapping")
    library_id = str(raw.get("id") or raw.get("library_id") or raw.get("name") or "material_templates_v1")
    items = raw.get("materials", raw.get("material_templates", raw.get("templates", ())))
    return MaterialLibrary(
        library_id=library_id,
        materials=tuple(_parse_material_template(item) for item in _mapping_sequence(items)),
    )


def _parse_material_template(raw: Mapping[str, Any]) -> MaterialAsset:
    if not isinstance(raw, Mapping):
        raise ConfigError("material template entries must be mappings")
    material_id = str(raw.get("id") or raw.get("material_id") or "").strip()
    if not material_id:
        raise ConfigError("material template requires id")
    family = str(raw.get("family") or "").strip()
    tags = _string_tuple(raw.get("tags"))
    if family and family not in tags:
        tags = (*tags, family)
    compatibility = dict(raw.get("compatibility") or {})
    requirements = raw.get("requirements")
    if requirements is not None:
        if not isinstance(requirements, Mapping):
            raise ConfigError(f"material template {material_id} requirements must be a mapping")
        compatibility.update(requirements)
    channels = raw.get("channels")
    if channels is None:
        channels = _template_channels_from_sources(material_id=material_id, sources=raw.get("sources"))
    elif not isinstance(channels, Mapping):
        raise ConfigError(f"material template {material_id} channels must be a mapping")
    return MaterialAsset(
        material_id=material_id,
        tags=tags,
        shader_model=_optional_string(raw.get("shader_family") or raw.get("shader_model")),
        usd_material_ref=_optional_string(raw.get("usd_material_ref")),
        channels=dict(channels),
        compatibility=compatibility,
    )


def _template_channels_from_sources(*, material_id: str, sources: Any) -> dict[str, Any]:
    if not isinstance(sources, Mapping):
        raise ConfigError(f"material template {material_id} requires sources or channels")
    channels: dict[str, Any] = {}
    constants = sources.get("constants")
    if constants is not None:
        if not isinstance(constants, Mapping):
            raise ConfigError(f"material template {material_id} sources.constants must be a mapping")
        for channel, value in constants.items():
            channels[str(channel)] = {"source": ChannelSource.CONSTANT, "value": value}
    textures = sources.get("textures")
    if textures is not None:
        if not isinstance(textures, Mapping):
            raise ConfigError(f"material template {material_id} sources.textures must be a mapping")
        for channel, texture in textures.items():
            channels[str(channel)] = _texture_binding(str(channel), texture, source=ChannelSource.TEXTURE)
    pbr_set = sources.get("pbr_set")
    if pbr_set is not None:
        if not isinstance(pbr_set, Mapping):
            raise ConfigError(f"material template {material_id} sources.pbr_set must be a mapping")
        channels.update(_pbr_set_bindings(pbr_set))
    if not channels:
        raise ConfigError(f"material template {material_id} defines no channel sources")
    return channels


def _pbr_set_bindings(pbr_set: Mapping[str, Any]) -> dict[str, Any]:
    raw_color_space = pbr_set.get("color_space")
    color_space = raw_color_space if isinstance(raw_color_space, Mapping) else {}
    raw_packing = pbr_set.get("packing")
    packing = raw_packing if isinstance(raw_packing, Mapping) else {}
    bindings: dict[str, Any] = {}
    for channel in (
        MaterialChannel.BASE_COLOR,
        MaterialChannel.ROUGHNESS,
        MaterialChannel.METALLIC,
        MaterialChannel.OPACITY,
        MaterialChannel.NORMAL,
        MaterialChannel.ORM,
    ):
        value = pbr_set.get(channel)
        if not value:
            continue
        if channel == MaterialChannel.ORM:
            binding = _texture_binding(channel, value, source=ChannelSource.PACKED_TEXTURE)
            binding["packing"] = "orm"
            binding["layout"] = packing.get("orm") or pbr_set.get("layout") or "orm_rgb"
        else:
            binding = _texture_binding(channel, value, source=ChannelSource.TEXTURE)
        if channel in color_space:
            binding["color_space"] = color_space[channel]
        if channel == MaterialChannel.NORMAL and pbr_set.get("normal_format"):
            binding["normal_format"] = pbr_set["normal_format"]
        bindings[channel] = binding
    return bindings


def _texture_binding(channel: str, raw: Any, *, source: str) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        binding = dict(raw)
        binding["source"] = source
        if "uri" not in binding:
            for key in ("path", "texture", "value"):
                if binding.get(key):
                    binding["uri"] = binding[key]
                    break
        if not binding.get("uri"):
            raise ConfigError(f"{channel} texture binding requires uri")
        return binding
    return {"source": source, "uri": str(raw)}


def _parse_material_local_adjustment_library(
    raw: Any,
) -> MaterialLocalAdjustmentLibrary | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return MaterialLocalAdjustmentLibrary(library_id=raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return MaterialLocalAdjustmentLibrary(
            library_id="material_local_adjustments_v1",
            adjustments=tuple(_parse_material_local_adjustment(item) for item in raw),
        )
    if not isinstance(raw, Mapping):
        raise ConfigError("material_libraries.local_adjustments must be a string, list, or mapping")
    library_id = str(raw.get("id") or raw.get("library_id") or raw.get("name") or "material_local_adjustments_v1")
    items = raw.get(
        "adjustments",
        raw.get("material_local_adjustments", raw.get("local_adjustments", ())),
    )
    return MaterialLocalAdjustmentLibrary(
        library_id=library_id,
        adjustments=tuple(_parse_material_local_adjustment(item) for item in _mapping_sequence(items)),
    )


def _parse_material_local_adjustment(raw: Mapping[str, Any]) -> MaterialLocalAdjustment:
    if not isinstance(raw, Mapping):
        raise ConfigError("material local adjustment entries must be mappings")
    adjustment_id = str(raw.get("id") or raw.get("adjustment_id") or "").strip()
    if not adjustment_id:
        raise ConfigError("material local adjustment requires id")
    channels = raw.get("channels")
    if not isinstance(channels, Mapping):
        raise ConfigError(f"material local adjustment {adjustment_id} channels must be a mapping")
    return MaterialLocalAdjustment(
        adjustment_id=adjustment_id,
        tags=_string_tuple(raw.get("tags")),
        family=_optional_string(raw.get("family")),
        channels=dict(channels),
        identity_preserving=bool(raw.get("identity_preserving", True)),
        slotwise_safe=bool(raw.get("slotwise_safe", True)),
        metadata=dict(raw.get("metadata") or {}),
    )


def _compile_profile(profile: Mapping[str, Any], *, libraries: RandomizationLibraries) -> dict[str, Any]:
    profile_id = str(profile.get("id") or profile.get("name") or "").strip()
    if not profile_id:
        raise ConfigError("material_randomization_profiles entries require id")
    policy_type = str(profile.get("type") or "").strip()
    if policy_type == "channel_matrix":
        raise ConfigError("channel_matrix is debug-only in V1; use advanced_rules")
    if policy_type not in PROFILE_TYPES:
        raise ConfigError(f"profile {profile_id} type must be one of {sorted(PROFILE_TYPES)}")
    policy = _compile_profile_policy(profile_id=profile_id, profile=profile, libraries=libraries)
    return {
        "name": profile_id,
        "targets": _compile_targets(profile.get("targets")),
        "policy": policy,
        "apply": dict(profile.get("apply") or {}),
    }


def _compile_profile_policy(
    *,
    profile_id: str,
    profile: Mapping[str, Any],
    libraries: RandomizationLibraries,
) -> dict[str, Any]:
    policy_type = str(profile.get("type") or "").strip()
    policy: dict[str, Any] = {
        "type": policy_type,
        "id": profile_id,
        "profile_id": profile_id,
    }
    if policy_type == "material_template":
        policy["library"] = str(
            profile.get("library")
            or profile.get("library_id")
            or (libraries.material_library.library_id if libraries.material_library else "")
        )
        _copy_keys(
            src=profile,
            dst=policy,
            keys=(
                "template_id",
                "template_ids",
                "material_id",
                "material_ids",
                "template_tags_any",
                "template_tags_all",
                "material_tags_any",
                "material_tags_all",
            ),
        )
        return policy
    policy["library"] = str(
        profile.get("library")
        or profile.get("library_id")
        or (libraries.adjustment_library.library_id if libraries.adjustment_library else "")
    )
    _copy_keys(
        src=profile,
        dst=policy,
        keys=(
            "adjustment_id",
            "adjustment_ids",
            "local_adjustment_id",
            "local_adjustment_ids",
            "adjustment_tags_any",
            "adjustment_tags_all",
            "tags_any",
            "tags_all",
        ),
    )
    return policy


def _compile_advanced_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(rule, Mapping):
        raise ConfigError("advanced_rules entries must be mappings")
    name = str(rule.get("name") or rule.get("id") or "").strip()
    if not name:
        raise ConfigError("advanced_rules entries require name")
    policy = rule.get("policy")
    if not isinstance(policy, Mapping):
        raise ConfigError(f"advanced rule {name} requires policy mapping")
    policy_type = str(policy.get("type") or "").strip()
    if policy_type != "channel_matrix":
        raise ConfigError(f"advanced rule {name} must use debug-only channel_matrix policy")
    return {
        "name": name,
        "targets": _compile_targets(rule.get("targets")),
        "policy": dict(policy),
        "apply": dict(rule.get("apply") or {}),
    }


def _compile_targets(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError("profile targets must be a mapping")
    if raw.get("semantic_tags_all"):
        raise ConfigError("semantic_tags_all requires ManaEnv adapter-side target expansion in V1")
    targets: dict[str, Any] = {}
    if raw.get("asset_ids") is not None:
        targets["asset_names"] = _string_tuple(raw.get("asset_ids"))
    if raw.get("asset_names") is not None:
        targets["asset_names"] = _string_tuple(raw.get("asset_names"))
    if raw.get("prim_paths") is not None:
        targets["prim_paths"] = _string_tuple(raw.get("prim_paths"))
    if raw.get("semantic_tags_any") is not None:
        targets["semantic_tags"] = _string_tuple(raw.get("semantic_tags_any"))
    if raw.get("semantic_tags") is not None:
        targets["semantic_tags"] = _string_tuple(raw.get("semantic_tags"))
    if "material_slots" in raw:
        targets["material_slots"] = raw["material_slots"]
    return targets


def _mapping_sequence(raw: Any) -> tuple[Mapping[str, Any], ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        result: list[Mapping[str, Any]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ConfigError(f"entry {index} must be a mapping")
            result.append(item)
        return tuple(result)
    raise ConfigError(f"expected mapping or list of mappings, got {type(raw).__name__}")


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    raise ConfigError(f"expected string or list of strings, got {type(value).__name__}")


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _copy_keys(*, src: Mapping[str, Any], dst: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in src:
            dst[key] = src[key]
