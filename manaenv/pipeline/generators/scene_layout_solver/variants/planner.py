from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from ..common.diagnostics import error
from ..common.reproducibility import stable_rng, stable_seed
from ..common.weighted import WeightError, parse_nonnegative_weight, weighted_choice
from ..scene_spec.schema import SceneLayoutSpec, SlotSpec, SurfaceRangeSpec


@dataclass(frozen=True)
class ObjectPlacementOverride:
    surface: str | None = None
    surface_range: SurfaceRangeSpec | None = None
    slot_id: str | None = None
    yaw: tuple[float, float] | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LayoutCase:
    case_id: str
    object_overrides: dict[str, ObjectPlacementOverride] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


class VariantPlanner:
    def __init__(self, spec: SceneLayoutSpec, *, nums: int | None = None) -> None:
        self.spec = spec
        self.nums = nums

    def plan(self) -> list[LayoutCase]:
        variant = self.spec.variants
        vtype = variant.type
        limit = self._limit()
        if vtype in {"", "fixed", "fixed_cases"}:
            return self._fixed_cases(limit=limit)
        elif vtype == "permutation":
            cases = self._permutation_cases()
        elif vtype == "cartesian_product":
            cases = self._cartesian_cases()
        elif vtype == "random_seeded":
            cases = self._random_seeded_cases()
        elif vtype == "stratified_slots":
            cases = self._stratified_slot_cases()
        else:
            raise error("variant_type_unsupported", "unsupported variant planner type", variant_type=vtype)
        if limit is not None:
            return cases[: max(0, int(limit))]
        return cases

    def _limit(self) -> int | None:
        if self.nums is not None:
            return int(self.nums)
        if self.spec.variants.nums is not None:
            return int(self.spec.variants.nums)
        return None

    def _slot_by_id(self) -> dict[str, SlotSpec]:
        return {slot.slot_id: slot for slot in self.spec.variants.slots}

    def _case_from_slot_assignment(
        self,
        case_id: str,
        assignment: dict[str, SlotSpec],
        *,
        params: dict[str, Any] | None = None,
    ) -> LayoutCase:
        return LayoutCase(
            case_id=case_id,
            object_overrides={
                obj: ObjectPlacementOverride(
                    surface=slot.surface,
                    surface_range=slot.surface_range,
                    slot_id=slot.slot_id,
                )
                for obj, slot in assignment.items()
            },
            params=dict(params or {}),
        )

    def fixed_case_candidates(self) -> list[LayoutCase]:
        """Return authored fixed cases once, preserving authored case weights."""

        variant = self.spec.variants
        if not variant.cases:
            return [LayoutCase(case_id="case_000")]
        slots = self._slot_by_id()
        out: list[LayoutCase] = []
        for idx, raw_case in enumerate(variant.cases):
            overrides: dict[str, ObjectPlacementOverride] = {}
            raw_bindings = raw_case.get("bindings") or raw_case.get("objects") or {}
            if not isinstance(raw_bindings, dict):
                raise error("fixed_case_bindings_invalid", "fixed case bindings must be a dict", target=f"case_{idx}")
            for obj, raw_binding in raw_bindings.items():
                if isinstance(raw_binding, str):
                    slot = slots.get(raw_binding)
                    if slot is None:
                        raise error(
                            "fixed_case_slot_unknown",
                            "fixed case references unknown slot",
                            target=str(obj),
                            slot=raw_binding,
                        )
                    overrides[str(obj)] = ObjectPlacementOverride(
                        surface=slot.surface,
                        surface_range=slot.surface_range,
                        slot_id=slot.slot_id,
                        yaw=slot.yaw,
                    )
                elif isinstance(raw_binding, dict):
                    slot_id = raw_binding.get("slot")
                    slot = slots.get(str(slot_id)) if slot_id else None
                    overrides[str(obj)] = ObjectPlacementOverride(
                        surface=str(raw_binding.get("surface") or (slot.surface if slot else "") or "") or None,
                        surface_range=SurfaceRangeSpec.from_dict(raw_binding.get("surface_range"))
                        or (slot.surface_range if slot else None),
                        slot_id=str(slot_id) if slot_id else None,
                        yaw=_binding_yaw(raw_binding, slot),
                        params=dict(raw_binding.get("params") or {}),
                    )
                else:
                    raise error(
                        "fixed_case_binding_invalid",
                        "fixed case object binding must be a slot id or mapping",
                        target=str(obj),
                    )
            case_id = str(raw_case.get("id") or f"case_{idx:03d}")
            weight = self._fixed_case_weight(raw_case, target=case_id)
            params = dict(raw_case.get("params") or {})
            params.setdefault("source_case_id", case_id)
            params["weight"] = weight
            out.append(
                LayoutCase(
                    case_id=case_id,
                    object_overrides=overrides,
                    params=params,
                )
            )
        return out

    def _fixed_cases(self, *, limit: int | None) -> list[LayoutCase]:
        variant = self.spec.variants
        candidates = self.fixed_case_candidates()
        if not self._fixed_cases_have_explicit_weight():
            return candidates[: max(0, int(limit))] if limit is not None else candidates
        if limit is None:
            raise error(
                "variant_weighted_fixed_cases_nums_missing",
                "weighted fixed_cases require nums for deterministic static planning",
            )
        if limit <= 0:
            return []
        positive = [case for case in candidates if _case_weight(case) > 0.0]
        if not positive:
            raise error("variant_weighted_fixed_cases_empty", "weighted fixed_cases have no positive-weight cases")
        base_seed = int(variant.seed if variant.seed is not None else self.spec.seed)
        weight_payload = {
            "case_ids": [case.case_id for case in candidates],
            "weights": [_case_weight(case) for case in candidates],
        }
        out: list[LayoutCase] = []
        for draw_index in range(int(limit)):
            rng = stable_rng(
                base_seed,
                "variant_weighted_fixed_case",
                draw_index,
                payload=weight_payload,
            )
            selected = weighted_choice(positive, rng=rng, weight_fn=_case_weight)
            if selected is None:
                raise error("variant_weighted_fixed_cases_empty", "weighted fixed_cases have no selectable cases")
            params = dict(selected.params)
            source_case_id = str(params.get("source_case_id") or selected.case_id)
            params.update(
                {
                    "source_case_id": source_case_id,
                    "source_weight": _case_weight(selected),
                    "weighted_draw_index": draw_index,
                }
            )
            params.setdefault(
                "seed",
                stable_seed(
                    base_seed,
                    "variant_weighted_fixed_case_seed",
                    draw_index,
                    payload={"source_case_id": source_case_id},
                ),
            )
            out.append(
                LayoutCase(
                    case_id=f"weighted_{draw_index:03d}:{source_case_id}",
                    object_overrides=dict(selected.object_overrides),
                    params=params,
                )
            )
        return out

    def _fixed_case_weight(self, raw_case: dict[str, Any], *, target: str) -> float:
        try:
            return parse_nonnegative_weight(raw_case.get("weight"), default=1.0)
        except WeightError as exc:
            raise error("variant_case_weight_invalid", str(exc), target=target) from exc

    def _fixed_cases_have_explicit_weight(self) -> bool:
        return any(isinstance(item, dict) and "weight" in item for item in self.spec.variants.cases)

    def _permutation_cases(self) -> list[LayoutCase]:
        variant = self.spec.variants
        objects = list(variant.objects)
        slots = list(variant.slots)
        if not objects:
            raise error("permutation_objects_missing", "permutation variant requires objects")
        if len(objects) != len(slots):
            raise error("permutation_cardinality_mismatch", "permutation objects and slots must have same length")
        out: list[LayoutCase] = []
        for idx, perm in enumerate(itertools.permutations(slots, len(objects))):
            assignment = {obj: slot for obj, slot in zip(objects, perm)}
            out.append(self._case_from_slot_assignment(f"perm_{idx:03d}", assignment))
        return out

    def _cartesian_cases(self) -> list[LayoutCase]:
        axes = self.spec.variants.axes
        if not axes:
            return [LayoutCase(case_id="cart_000")]
        keys = sorted(str(k) for k in axes.keys())
        values = [list(axes[k]) for k in keys]
        out = []
        for idx, combo in enumerate(itertools.product(*values)):
            out.append(LayoutCase(case_id=f"cart_{idx:03d}", params={k: v for k, v in zip(keys, combo)}))
        return out

    def _random_seeded_cases(self) -> list[LayoutCase]:
        variant = self.spec.variants
        nums = int(self._limit() if self._limit() is not None else 1)
        base_seed = int(variant.seed if variant.seed is not None else self.spec.seed)
        slots = list(variant.slots)
        objects = list(variant.objects)
        out: list[LayoutCase] = []
        for idx in range(max(0, nums)):
            rng = stable_rng(
                base_seed,
                "variant_random_seeded",
                idx,
                payload={"objects": objects, "slots": [slot.slot_id for slot in slots]},
            )
            case_seed = stable_seed(
                base_seed,
                "variant_case_seed",
                idx,
                payload={"objects": objects, "slots": [slot.slot_id for slot in slots]},
            )
            if slots and objects:
                sampled_slots = list(slots)
                rng.shuffle(sampled_slots)
                assignment = {obj: sampled_slots[i % len(sampled_slots)] for i, obj in enumerate(objects)}
                out.append(self._case_from_slot_assignment(f"random_{idx:03d}", assignment, params={"seed": case_seed}))
            else:
                out.append(LayoutCase(case_id=f"random_{idx:03d}", params={"seed": case_seed}))
        return out

    def _stratified_slot_cases(self) -> list[LayoutCase]:
        variant = self.spec.variants
        objects = list(variant.objects)
        slots = list(variant.slots)
        nums = int(self._limit() if self._limit() is not None else max(1, len(slots)))
        if not objects or not slots:
            return [LayoutCase(case_id=f"stratified_{idx:03d}") for idx in range(nums)]
        out: list[LayoutCase] = []
        for idx in range(nums):
            assignment = {obj: slots[(idx + obj_idx) % len(slots)] for obj_idx, obj in enumerate(objects)}
            out.append(self._case_from_slot_assignment(f"stratified_{idx:03d}", assignment))
        return out


def _case_weight(case: LayoutCase) -> float:
    try:
        return float(case.params.get("weight", 1.0))
    except (TypeError, ValueError):
        return 0.0


def _binding_yaw(raw_binding: dict[str, Any], slot: SlotSpec | None) -> tuple[float, float] | None:
    if raw_binding.get("yaw") is None:
        return slot.yaw if slot is not None else None
    raw = raw_binding.get("yaw")
    if isinstance(raw, (int, float)):
        value = float(raw)
        return (value, value)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (float(raw[0]), float(raw[1]))
    raise error("fixed_case_binding_yaw_invalid", "fixed case binding yaw must be a scalar or [min, max] range")
