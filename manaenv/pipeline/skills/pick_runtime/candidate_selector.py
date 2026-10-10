"""Candidate ranking/cache state for PickSkill."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CandidateSelectorState:
    """Mutable per-skill candidate cache and cooldown state."""

    blocked_candidates: set[int] = field(default_factory=set)
    succeeded_candidates: set[int] = field(default_factory=set)
    succeeded_candidate_order: list[int] = field(default_factory=list)
    candidate_plan_failures: dict[int, int] = field(default_factory=dict)
    candidate_cooldowns: dict[int, int] = field(default_factory=dict)

    def mark_failed(self, idx: int) -> bool:
        if idx < 0 or idx in self.blocked_candidates:
            return False
        self.blocked_candidates.add(idx)
        return True

    def mark_succeeded(self, idx: int) -> bool:
        if idx < 0:
            return False
        self.candidate_plan_failures.pop(idx, None)
        self.candidate_cooldowns.pop(idx, None)
        if idx in self.succeeded_candidates:
            return False
        self.succeeded_candidates.add(idx)
        self.succeeded_candidate_order.append(idx)
        return True

    def mark_unreachable(self, idx: int, *, cooldown: int) -> int:
        if idx < 0:
            return 0
        failures = self.candidate_plan_failures.get(idx, 0) + 1
        self.candidate_plan_failures[idx] = failures
        if idx in self.succeeded_candidates and cooldown > 0:
            self.candidate_cooldowns[idx] = max(self.candidate_cooldowns.get(idx, 0), int(cooldown))
        return failures

    def reset_episode(self) -> None:
        self.blocked_candidates.clear()
        self.candidate_plan_failures.clear()
        self.candidate_cooldowns.clear()


def rank_candidates(
    *,
    state: CandidateSelectorState,
    candidate_indices: torch.Tensor,
    scores: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    if candidate_indices.numel() == 0:
        return candidate_indices
    top_k = max(int(top_k), 1)
    n = int(candidate_indices.numel())
    device = candidate_indices.device
    if n <= top_k:
        perm = torch.randperm(n, device=device)
        return candidate_indices[perm]

    if state.succeeded_candidates:
        winner_order = [idx for idx in state.succeeded_candidate_order if idx in state.succeeded_candidates]
        winner_tensor = torch.tensor(winner_order, dtype=candidate_indices.dtype, device=device)
        is_winner = torch.isin(candidate_indices, winner_tensor)
        candidate_set = set(candidate_indices[is_winner].detach().cpu().tolist())
        winner_idx = torch.tensor(
            [idx for idx in winner_order if idx in candidate_set], dtype=candidate_indices.dtype, device=device
        )
        loser_idx = candidate_indices[~is_winner]
    else:
        winner_idx = candidate_indices[:0]
        loser_idx = candidate_indices

    explore_reserve = 1 if loser_idx.numel() > 0 else 0
    winner_quota = max(0, top_k - explore_reserve)
    winner_picks = winner_idx[:winner_quota] if winner_idx.numel() > winner_quota else winner_idx

    remaining = top_k - int(winner_picks.numel())
    if remaining > 0 and loser_idx.numel() > 0:
        loser_scores = scores[loser_idx].to(dtype=torch.float32)
        weights = torch.clamp(loser_scores, min=1.0e-3)
        n_to_draw = min(remaining, int(loser_idx.numel()))
        sampled = torch.multinomial(weights, n_to_draw, replacement=False)
        loser_picks = loser_idx[sampled]
    else:
        loser_picks = loser_idx[:0]
    return torch.cat([winner_picks, loser_picks], dim=0)


def tick_cooldowns(state: CandidateSelectorState) -> None:
    for idx in list(state.candidate_cooldowns):
        remaining = int(state.candidate_cooldowns[idx]) - 1
        if remaining <= 0:
            state.candidate_cooldowns.pop(idx, None)
        else:
            state.candidate_cooldowns[idx] = remaining


def apply_candidate_cooldowns(
    *,
    state: CandidateSelectorState,
    candidate_indices: torch.Tensor,
    log_fn,
) -> torch.Tensor:
    if not state.candidate_cooldowns or candidate_indices.numel() == 0:
        return candidate_indices
    active = sorted(idx for idx, ttl in state.candidate_cooldowns.items() if int(ttl) > 0)
    if not active:
        state.candidate_cooldowns.clear()
        return candidate_indices

    cooldown_tensor = torch.tensor(active, dtype=candidate_indices.dtype, device=candidate_indices.device)
    keep_mask = ~torch.isin(candidate_indices, cooldown_tensor)
    survived = candidate_indices[keep_mask]
    removed = int((~keep_mask).sum().item())
    if removed > 0 and survived.numel() > 0:
        log_fn(f"[pick] cooldown removed {removed} temporarily unreachable successful candidate(s): {active}")
        tick_cooldowns(state)
        return survived
    if removed > 0 and survived.numel() == 0:
        log_fn("[pick] all filtered candidates are cooling down; clearing cooldowns to avoid an empty batch")
        state.candidate_cooldowns.clear()
        return candidate_indices
    tick_cooldowns(state)
    return candidate_indices


def winner_candidates(*, state: CandidateSelectorState, candidate_indices: torch.Tensor) -> torch.Tensor:
    if not state.succeeded_candidate_order or candidate_indices.numel() == 0:
        return candidate_indices[:0]
    candidate_set = set(candidate_indices.detach().cpu().tolist())
    winners = [idx for idx in state.succeeded_candidate_order if idx in candidate_set]
    if not winners:
        return candidate_indices[:0]
    return torch.tensor(winners, dtype=candidate_indices.dtype, device=candidate_indices.device)
