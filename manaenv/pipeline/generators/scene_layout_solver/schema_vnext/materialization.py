from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..common.diagnostics import error
from ..common.weighted import weighted_choice
from .parser import AssetSelectorSpec, GroupSpec, VariantChoiceSpec, VariantSpec, VNextSceneSpec

AssetTagResolver = Callable[[str], list[str] | tuple[str, ...]]


@dataclass(frozen=True)
class AssetPoolBindingPlan:
    target_type: str
    target_id: str
    candidate_id: str
    assets: tuple[str, ...]
    asset_scales: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    scale_normalization: Mapping[str, float] = field(default_factory=dict)

    @property
    def candidate_ref(self) -> str:
        if self.target_type == "background":
            return f"background.{self.candidate_id}"
        return f"{self.target_id}.{self.candidate_id}"


@dataclass(frozen=True)
class MaterializationChoicePlan:
    variant_id: str
    choice_id: str
    weight: float
    bindings: tuple[AssetPoolBindingPlan, ...]

    @property
    def target_keys(self) -> tuple[str, ...]:
        return tuple(_target_key(binding.target_type, binding.target_id) for binding in self.bindings)


@dataclass(frozen=True)
class MaterializationVariantPlan:
    variant_id: str
    variant_type: str
    target_keys: tuple[str, ...]
    choices: tuple[MaterializationChoicePlan, ...]


@dataclass(frozen=True)
class AssignmentChoicePlan:
    variant_id: str
    choice_id: str
    weight: float
    group_id: str
    assignment_id: str
    bindings: Mapping[str, str]
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AssignmentVariantPlan:
    variant_id: str
    group_id: str
    choices: tuple[AssignmentChoicePlan, ...]


@dataclass(frozen=True)
class SceneVariantPlan:
    materialization: tuple[MaterializationVariantPlan, ...] = ()
    assignments: tuple[AssignmentVariantPlan, ...] = ()
    pool_targets: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def assignment_variant_groups(self) -> set[str]:
        return {variant.group_id for variant in self.assignments}


@dataclass(frozen=True)
class SceneMaterializationSample:
    variant_choices: Mapping[str, str]
    background_asset: str
    object_assets: Mapping[str, str]
    selected_candidate_refs: Mapping[str, str]
    object_scales: Mapping[str, float] = field(default_factory=dict)
    object_spawn_masses: Mapping[str, float] = field(default_factory=dict)
    object_asset_scales: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    background_asset_scale: tuple[float, float, float] | None = None
    selected_assignments: Mapping[str, str] = field(default_factory=dict)


def build_scene_variant_plan(scene: VNextSceneSpec) -> SceneVariantPlan:
    _validate_background_surface_contract(scene)
    pool_targets = _pool_target_registry(scene)
    materialization = _build_materialization_variants(scene)
    _validate_materialization_target_coverage(pool_targets, materialization)
    assignments = _build_assignment_variants(scene)
    return SceneVariantPlan(
        materialization=materialization,
        assignments=assignments,
        pool_targets=pool_targets,
    )


def sample_scene_materialization(
    scene: VNextSceneSpec,
    *,
    rng: random.Random,
    asset_tag_resolver: AssetTagResolver | None = None,
) -> SceneMaterializationSample:
    plan = build_scene_variant_plan(scene)
    variant_choices: dict[str, str] = {}
    selected_candidate_refs: dict[str, str] = {}
    object_asset_scales: dict[str, tuple[float, float, float]] = {}
    background_asset_scale: tuple[float, float, float] | None = None
    background_asset = ""
    if scene.background.asset.selector_type != "pool":
        background_asset = _resolve_non_pool_asset(
            scene.background.asset,
            rng=rng,
            asset_tag_resolver=asset_tag_resolver,
            target="background",
        )
    object_assets = {
        object_id: _resolve_non_pool_asset(
            obj.asset, rng=rng, asset_tag_resolver=asset_tag_resolver, target=f"objects.{object_id}.asset"
        )
        for object_id, obj in scene.objects.items()
        if obj.asset.selector_type != "pool"
    }

    for variant in plan.materialization:
        choice = weighted_choice(variant.choices, rng=rng, weight_fn=lambda item: item.weight)
        if choice is None:
            raise error(
                "schema_vnext_materialization_weights_all_zero",
                "materialization choices require positive weights",
                target=variant.variant_id,
            )
        variant_choices[variant.variant_id] = choice.choice_id
        for binding in choice.bindings:
            concrete_asset = _uniform_asset(binding.assets, rng=rng, target=binding.candidate_ref)
            selected_candidate_refs[_target_key(binding.target_type, binding.target_id)] = binding.candidate_ref
            if binding.target_type == "background":
                background_asset = concrete_asset
                background_asset_scale = binding.asset_scales.get(concrete_asset)
            else:
                object_assets[binding.target_id] = concrete_asset
                scale = binding.asset_scales.get(concrete_asset)
                if scale is not None:
                    object_asset_scales[binding.target_id] = scale

    selected_assignments: dict[str, str] = {}
    for variant in plan.assignments:
        choice = weighted_choice(variant.choices, rng=rng, weight_fn=lambda item: item.weight)
        if choice is None:
            raise error(
                "schema_vnext_assignment_variant_weights_all_zero",
                "assignment choices require positive weights",
                target=variant.variant_id,
            )
        variant_choices[variant.variant_id] = choice.choice_id
        selected_assignments[variant.group_id] = choice.assignment_id

    missing = sorted(set(scene.objects) - set(object_assets))
    if (
        scene.background.asset.selector_type == "pool"
        and _target_key("background", "background") not in selected_candidate_refs
    ):
        # Allow a background pool without an explicit background_pool_choice
        # variant. Sample uniformly over every concrete asset declared in the
        # pool.
        pool_assets = tuple(
            asset_id for candidate in scene.background.asset.candidates for asset_id in candidate.assets
        )
        if not pool_assets:
            raise error(
                "schema_vnext_pool_target_unmaterialized",
                "background pool target was not materialized",
                target="background",
            )
        background_asset = _uniform_asset(pool_assets, rng=rng, target="background")
        owning_candidate = next(
            candidate.candidate_id
            for candidate in scene.background.asset.candidates
            if background_asset in candidate.assets
        )
        selected_candidate_refs[_target_key("background", "background")] = f"background.{owning_candidate}"
    if missing:
        raise error(
            "schema_vnext_pool_target_unmaterialized",
            "object pool target was not materialized",
            target="objects",
            ids=missing,
        )
    object_assets = _expand_object_assets_for_instances(scene, object_assets)
    object_asset_scales = _expand_object_scales_for_instances(scene, object_asset_scales)
    for concrete_id, axis in _object_axis_scales(scene).items():
        object_asset_scales.setdefault(concrete_id, axis)
    selected_candidate_refs = _expand_candidate_refs_for_instances(scene, selected_candidate_refs)
    object_scales = _sample_object_scales(scene, rng=rng)
    object_spawn_masses = _sample_object_spawn_masses(scene, rng=rng)

    return SceneMaterializationSample(
        variant_choices=variant_choices,
        background_asset=background_asset,
        object_assets=object_assets,
        selected_candidate_refs=selected_candidate_refs,
        object_scales=object_scales,
        object_spawn_masses=object_spawn_masses,
        object_asset_scales=object_asset_scales,
        background_asset_scale=background_asset_scale,
        selected_assignments=selected_assignments,
    )


def _build_materialization_variants(scene: VNextSceneSpec) -> tuple[MaterializationVariantPlan, ...]:
    variants: list[MaterializationVariantPlan] = []
    controlled_targets: dict[str, str] = {}
    for variant in scene.variants.materialization:
        choices = tuple(_materialization_choice_plan(scene, variant, choice) for choice in variant.choices)
        if not choices:
            raise error(
                "schema_vnext_variant_choices_empty",
                "materialization variant requires choices",
                target=variant.variant_id,
            )
        first_targets = choices[0].target_keys
        for choice in choices:
            if choice.target_keys != first_targets:
                raise error(
                    "schema_vnext_materialization_choice_target_mismatch",
                    "all choices in a materialization variant must bind the same targets",
                    target=variant.variant_id,
                )
        for target_key in first_targets:
            if target_key in controlled_targets:
                raise error(
                    "schema_vnext_materialization_target_multi_owner",
                    "pool target cannot be controlled by multiple materialization variants",
                    target=target_key,
                    variants=[controlled_targets[target_key], variant.variant_id],
                )
            controlled_targets[target_key] = variant.variant_id
        variants.append(
            MaterializationVariantPlan(
                variant_id=variant.variant_id,
                variant_type=variant.variant_type,
                target_keys=first_targets,
                choices=choices,
            )
        )
    return tuple(variants)


def _materialization_choice_plan(
    scene: VNextSceneSpec,
    variant: VariantSpec,
    choice: VariantChoiceSpec,
) -> MaterializationChoicePlan:
    bindings: list[AssetPoolBindingPlan] = []
    for ref in choice.refs:
        target_id, candidate_id = ref.split(".", 1)
        if target_id == "background":
            asset = scene.background.asset
            target_type = "background"
            concrete_target_id = "background"
        else:
            obj = scene.objects[target_id]
            asset = obj.asset
            target_type = "object"
            concrete_target_id = target_id
        assets = _candidate_assets(asset, candidate_id, target=ref)
        asset_scales = _candidate_asset_scales(asset, candidate_id, target=ref)
        bindings.append(
            AssetPoolBindingPlan(
                target_type=target_type,
                target_id=concrete_target_id,
                candidate_id=candidate_id,
                assets=assets,
                asset_scales=asset_scales,
                scale_normalization=_candidate_scale_normalization(asset, candidate_id, target=ref),
            )
        )
    return MaterializationChoicePlan(
        variant_id=variant.variant_id,
        choice_id=choice.choice_id,
        weight=choice.weight,
        bindings=tuple(bindings),
    )


def _build_assignment_variants(scene: VNextSceneSpec) -> tuple[AssignmentVariantPlan, ...]:
    variants: list[AssignmentVariantPlan] = []
    controlled_groups: dict[str, str] = {}
    for variant in scene.variants.assignments:
        group_id = str(variant.group)
        if group_id in controlled_groups:
            raise error(
                "schema_vnext_assignment_variant_group_multi_owner",
                "slot group cannot be controlled by multiple assignment variants",
                target=group_id,
                variants=[controlled_groups[group_id], variant.variant_id],
            )
        controlled_groups[group_id] = variant.variant_id
        group = scene.layout.groups[group_id]
        choices = tuple(_assignment_choice_plan(variant, choice, group) for choice in variant.choices)
        variants.append(AssignmentVariantPlan(variant_id=variant.variant_id, group_id=group_id, choices=choices))
    return tuple(variants)


def _assignment_choice_plan(
    variant: VariantSpec,
    choice: VariantChoiceSpec,
    group: GroupSpec,
) -> AssignmentChoicePlan:
    assignment_id = choice.refs[0]
    assignment_by_id = {item.assignment_id: item for item in group.assignments}
    assignment = assignment_by_id.get(assignment_id)
    if assignment is None:
        raise error(
            "schema_vnext_assignment_variant_generated_unsupported",
            "scene assignment variants require explicit assignments",
            target=variant.variant_id,
        )
    return AssignmentChoicePlan(
        variant_id=variant.variant_id,
        choice_id=choice.choice_id,
        weight=choice.weight,
        group_id=group.group_id,
        assignment_id=assignment_id,
        bindings=assignment.bindings,
        params=assignment.params,
    )


def _pool_target_registry(scene: VNextSceneSpec) -> dict[str, tuple[str, ...]]:
    targets: dict[str, tuple[str, ...]] = {}
    if scene.background.asset.selector_type == "pool":
        targets[_target_key("background", "background")] = tuple(sorted(scene.background.asset.candidate_ids))
    for object_id, obj in scene.objects.items():
        if obj.asset.selector_type == "pool":
            targets[_target_key("object", object_id)] = tuple(sorted(obj.asset.candidate_ids))
    return targets


def _expand_object_assets_for_instances(
    scene: VNextSceneSpec,
    object_assets: Mapping[str, str],
) -> dict[str, str]:
    expanded: dict[str, str] = {}
    for object_id, obj in scene.objects.items():
        asset_id = object_assets.get(object_id)
        if asset_id is None:
            continue
        for concrete_id in obj.materialized_ids:
            expanded[concrete_id] = asset_id
    return expanded


def _expand_candidate_refs_for_instances(
    scene: VNextSceneSpec,
    selected_candidate_refs: Mapping[str, str],
) -> dict[str, str]:
    expanded = dict(selected_candidate_refs)
    for object_id, obj in scene.objects.items():
        ref = selected_candidate_refs.get(_target_key("object", object_id))
        if not ref:
            continue
        for instance_id in obj.instance_ids:
            expanded[_target_key("object", instance_id)] = ref
    return expanded


def _sample_object_scales(scene: VNextSceneSpec, *, rng: random.Random) -> dict[str, float]:
    scales: dict[str, float] = {}
    for object_id, obj in scene.objects.items():
        if obj.scale is None or obj.scale.scale_range is None:
            continue
        value = _sample_scale(obj.scale, rng=rng)
        for concrete_id in obj.materialized_ids:
            scales[concrete_id] = value
    node_scale_owners: dict[str, str] = {}
    for node_id, node in scene.layout.nodes.items():
        if node.scale is None or node.scale.scale_range is None:
            continue
        object_id = str(node.object_id)
        previous = node_scale_owners.get(object_id)
        if previous is not None:
            raise error(
                "schema_vnext_node_scale_multi_owner",
                "a concrete object can receive at most one node-level scale override",
                target=node_id,
                object=object_id,
                previous_node=previous,
            )
        node_scale_owners[object_id] = node_id
        scales[object_id] = _sample_scale(node.scale, rng=rng)
    return scales


def _object_axis_scales(scene: VNextSceneSpec) -> dict[str, tuple[float, float, float]]:
    """Collect explicit non-uniform per-axis object scales, keyed by concrete id."""
    axis_scales: dict[str, tuple[float, float, float]] = {}
    for _object_id, obj in scene.objects.items():
        axis = obj.scale.axis_scale if obj.scale is not None else None
        if axis is None:
            continue
        for concrete_id in obj.materialized_ids:
            axis_scales[concrete_id] = axis
    for _node_id, node in scene.layout.nodes.items():
        axis = node.scale.axis_scale if node.scale is not None else None
        if axis is None:
            continue
        axis_scales[str(node.object_id)] = axis
    return axis_scales


def _sample_scale(scale: Any, *, rng: random.Random) -> float:
    left, right = scale.scale_range
    if abs(float(left) - float(right)) <= 1e-12:
        return float(left)
    return float(rng.uniform(float(left), float(right)))


def _sample_object_spawn_masses(scene: VNextSceneSpec, *, rng: random.Random) -> dict[str, float]:
    masses: dict[str, float] = {}
    for _object_id, obj in scene.objects.items():
        if obj.spawn_mass is None:
            continue
        value = _sample_mass(obj.spawn_mass, rng=rng)
        for concrete_id in obj.materialized_ids:
            masses[concrete_id] = value
    return masses


def _sample_mass(mass: Any, *, rng: random.Random) -> float:
    left, right = mass.mass_range
    if abs(float(left) - float(right)) <= 1e-12:
        return float(left)
    return float(rng.uniform(float(left), float(right)))


def _expand_object_scales_for_instances(
    scene: VNextSceneSpec,
    object_asset_scales: Mapping[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float]]:
    expanded: dict[str, tuple[float, float, float]] = {}
    for object_id, obj in scene.objects.items():
        scale = object_asset_scales.get(object_id)
        if scale is None:
            continue
        for concrete_id in obj.materialized_ids:
            expanded[concrete_id] = scale
    return expanded


def _validate_materialization_target_coverage(
    pool_targets: Mapping[str, tuple[str, ...]],
    variants: tuple[MaterializationVariantPlan, ...],
) -> None:
    controlled = {target_key for variant in variants for target_key in variant.target_keys}
    # Background pools may omit ``background_pool_choice``; sampling falls back to
    # uniform selection over the declared concrete assets.
    missing = sorted(key for key in (set(pool_targets) - controlled) if key != "background")
    extra = sorted(controlled - set(pool_targets))
    if missing:
        raise error(
            "schema_vnext_pool_target_unmaterialized",
            "pool targets must be controlled by variants.materialization",
            target="variants.materialization",
            targets=missing,
        )
    if extra:
        raise error(
            "schema_vnext_materialization_target_not_pool",
            "materialization variant target must be an asset pool",
            target="variants.materialization",
            targets=extra,
        )


def _validate_background_surface_contract(scene: VNextSceneSpec) -> None:
    required = set(scene.background.required_surfaces).union(scene.background.surface_overrides)
    missing: list[str] = []
    for surface in scene.layout.surfaces.values():
        if surface.ref and surface.ref.startswith("background."):
            surface_name = surface.ref.split(".", 1)[1]
            if surface_name not in required:
                missing.append(surface_name)
    if missing:
        raise error(
            "schema_vnext_background_surface_contract_missing",
            "layout surface refs must be declared in background.required_surfaces",
            target="background.required_surfaces",
            surfaces=sorted(set(missing)),
        )


def _candidate_assets(asset: AssetSelectorSpec, candidate_id: str, *, target: str) -> tuple[str, ...]:
    for candidate in asset.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate.assets
    raise error("schema_vnext_unknown_reference", "unknown asset pool candidate", target=target, ref=candidate_id)


def _candidate_asset_scales(
    asset: AssetSelectorSpec,
    candidate_id: str,
    *,
    target: str,
) -> Mapping[str, tuple[float, float, float]]:
    for candidate in asset.candidates:
        if candidate.candidate_id == candidate_id:
            return dict(candidate.asset_scales)
    raise error("schema_vnext_unknown_reference", "unknown asset pool candidate", target=target, ref=candidate_id)


def _candidate_scale_normalization(
    asset: AssetSelectorSpec,
    candidate_id: str,
    *,
    target: str,
) -> Mapping[str, float]:
    for candidate in asset.candidates:
        if candidate.candidate_id == candidate_id:
            return dict(candidate.scale_normalization)
    raise error("schema_vnext_unknown_reference", "unknown asset pool candidate", target=target, ref=candidate_id)


def _resolve_non_pool_asset(
    asset: AssetSelectorSpec,
    *,
    rng: random.Random,
    asset_tag_resolver: AssetTagResolver | None,
    target: str,
) -> str:
    if asset.selector_type == "asset_id":
        return str(asset.asset_id)
    if asset.selector_type == "tag":
        if asset_tag_resolver is None:
            raise error(
                "schema_vnext_tag_resolver_missing", "tag asset materialization requires a resolver", target=target
            )
        assets = tuple(str(item).strip() for item in asset_tag_resolver(str(asset.tag)) if str(item).strip())
        if not assets:
            raise error(
                "schema_vnext_tag_resolver_empty",
                "tag asset resolver returned no candidates",
                target=target,
                tag=asset.tag,
            )
        return _uniform_asset(assets, rng=rng, target=target)
    if asset.selector_type == "pool":
        raise error(
            "schema_vnext_pool_target_unmaterialized", "pool asset requires variants.materialization", target=target
        )
    raise error(
        "schema_vnext_asset_type_unsupported",
        "unsupported asset selector type",
        target=target,
        type=asset.selector_type,
    )


def _uniform_asset(assets: tuple[str, ...], *, rng: random.Random, target: str) -> str:
    if not assets:
        raise error("schema_vnext_pool_assets_empty", "asset candidate contains no concrete assets", target=target)
    return assets[rng.randrange(len(assets))]


def _target_key(target_type: str, target_id: str) -> str:
    if target_type == "background":
        return "background"
    return f"object:{target_id}"
