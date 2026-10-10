from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..common.diagnostics import error
from .parser import SelectorSpec


@dataclass(frozen=True)
class SelectorResult:
    selected_id: str
    selector_id: str
    selector_type: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class SelectorRuntimeState:
    counters: dict[tuple[str, str, int], int] = field(default_factory=dict)
    sequences: dict[tuple[str, str, int], tuple[str, ...]] = field(default_factory=dict)


def select_candidate(
    candidate_ids: Sequence[str],
    selector: SelectorSpec,
    *,
    selector_id: str,
    runtime_sample_id: str,
    env_id: int = 0,
    rng: random.Random,
    state: SelectorRuntimeState | None = None,
) -> SelectorResult:
    candidates = _candidate_tuple(candidate_ids, target=selector_id)
    if selector.selector_type == "uniform":
        selected = candidates[rng.randrange(len(candidates))]
        return SelectorResult(
            selected_id=selected,
            selector_id=selector_id,
            selector_type="uniform",
            metadata={"candidate_count": len(candidates)},
        )
    if selector.selector_type == "weighted":
        selected = _weighted_select(candidates, selector, rng=rng, target=selector_id)
        return SelectorResult(
            selected_id=selected,
            selector_id=selector_id,
            selector_type="weighted",
            metadata={
                "weights": {
                    candidate: selector.weights.get(candidate, selector.default_weight) for candidate in candidates
                }
            },
        )
    if selector.selector_type == "schedule":
        runtime_state = state if state is not None else SelectorRuntimeState()
        selected, index, sequence = _schedule_select(
            candidates,
            selector,
            selector_id=selector_id,
            runtime_sample_id=runtime_sample_id,
            env_id=env_id,
            rng=rng,
            state=runtime_state,
        )
        return SelectorResult(
            selected_id=selected,
            selector_id=selector_id,
            selector_type="schedule",
            metadata={
                "mode": selector.mode,
                "index": index,
                "sequence_length": len(sequence),
                "runtime_sample_id": runtime_sample_id,
                "env_id": env_id,
            },
        )
    raise error(
        "schema_vnext_selector_type_unsupported",
        "unsupported selector type",
        target=selector_id,
        type=selector.selector_type,
    )


def _candidate_tuple(candidate_ids: Sequence[str], *, target: str) -> tuple[str, ...]:
    candidates = tuple(str(candidate_id) for candidate_id in candidate_ids)
    if not candidates:
        raise error("schema_vnext_selector_candidates_empty", "selector requires candidates", target=target)
    duplicates = sorted({candidate_id for candidate_id in candidates if candidates.count(candidate_id) > 1})
    if duplicates:
        raise error(
            "schema_vnext_selector_candidate_duplicate",
            "selector candidates must be unique",
            target=target,
            ids=duplicates,
        )
    return candidates


def _weighted_select(candidates: tuple[str, ...], selector: SelectorSpec, *, rng: random.Random, target: str) -> str:
    weights = [selector.weights.get(candidate, selector.default_weight) for candidate in candidates]
    if not any(weight > 0.0 for weight in weights):
        raise error(
            "schema_vnext_selector_weights_all_zero",
            "selector requires at least one positive effective weight",
            target=target,
        )
    total = sum(weight for weight in weights if weight > 0.0)
    threshold = rng.random() * total
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights, strict=True):
        if weight <= 0.0:
            continue
        cumulative += weight
        if cumulative >= threshold:
            return candidate
    return candidates[-1]


def _schedule_select(
    candidates: tuple[str, ...],
    selector: SelectorSpec,
    *,
    selector_id: str,
    runtime_sample_id: str,
    env_id: int,
    rng: random.Random,
    state: SelectorRuntimeState,
) -> tuple[str, int, tuple[str, ...]]:
    key = (str(runtime_sample_id), str(selector_id), int(env_id))
    sequence = state.sequences.get(key)
    if sequence is None:
        sequence = _schedule_sequence(candidates, selector, rng=rng, target=selector_id)
        state.sequences[key] = sequence
    index = state.counters.get(key, 0)
    if index < len(sequence):
        selected = sequence[index]
    elif selector.on_exhausted == "repeat":
        selected = sequence[index % len(sequence)]
    elif selector.on_exhausted == "hold_last":
        selected = sequence[-1]
    else:
        raise error(
            "schema_vnext_selector_schedule_exhausted",
            "schedule selector is exhausted",
            target=selector_id,
            index=index,
        )
    state.counters[key] = index + 1
    return selected, index, sequence


def _schedule_sequence(
    candidates: tuple[str, ...],
    selector: SelectorSpec,
    *,
    rng: random.Random,
    target: str,
) -> tuple[str, ...]:
    candidate_set = set(candidates)
    if selector.mode == "cycle":
        _validate_schedule_ids(selector.order, candidate_set, target=target)
        return selector.order
    if selector.mode == "blocks":
        sequence: list[str] = []
        for block in selector.blocks:
            _validate_schedule_ids((block.candidate_id,), candidate_set, target=target)
            sequence.extend([block.candidate_id] * block.count)
        return tuple(sequence)
    if selector.mode == "counts":
        order = selector.order or tuple(sorted(selector.counts))
        _validate_schedule_ids(order, candidate_set, target=target)
        sequence = [candidate_id for candidate_id in order for _ in range(selector.counts[candidate_id])]
        if selector.order_policy == "shuffle_once":
            rng.shuffle(sequence)
        return tuple(sequence)
    raise error(
        "schema_vnext_selector_schedule_mode_unsupported",
        "unsupported schedule mode",
        target=target,
        mode=selector.mode,
    )


def _validate_schedule_ids(ids: Sequence[str], candidate_set: set[str], *, target: str) -> None:
    unknown = sorted(set(ids) - candidate_set)
    if unknown:
        raise error(
            "schema_vnext_selector_unknown_candidate",
            "selector references unknown candidates",
            target=target,
            ids=unknown,
        )
