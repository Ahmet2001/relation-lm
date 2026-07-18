from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def causal_anchor_history_mask(anchor_index: Tensor, sequence_length: int) -> Tensor:
    """Return ``[B,T,S]`` mask of positions used as anchors up to each query."""

    if anchor_index.ndim != 2:
        raise ValueError("anchor_index must have shape [batch, query_length]")
    return F.one_hot(anchor_index, num_classes=sequence_length).cumsum(1).bool()


def partner_distance_zones(partner_index: Tensor) -> Tensor:
    """Map partners to near=0, middle=1 and far=2 by relative backward distance."""

    if partner_index.ndim != 3:
        raise ValueError("partner_index must have shape [batch, query_length, k]")
    query_length = partner_index.size(1)
    query_position = torch.arange(
        query_length, device=partner_index.device, dtype=partner_index.dtype
    )[None, :, None]
    distance = query_position - partner_index
    relative = distance.float() / (query_position.float() + 1.0)
    return torch.where(
        relative < 1.0 / 3.0,
        torch.zeros_like(partner_index),
        torch.where(
            relative < 2.0 / 3.0,
            torch.ones_like(partner_index),
            torch.full_like(partner_index, 2),
        ),
    )


def summarize_partner_dynamics(
    anchor_index: Tensor,
    partner_index: Tensor,
    partner_weights: Tensor,
    active_partner_mask: Tensor,
    relation_vectors: Tensor | None = None,
) -> dict[str, Any]:
    """Summarize partner reuse, influence and near/middle/far distribution.

    When relation vectors are provided, influence is measured as
    ``weight * ||relation_vector||``. This is a contribution-magnitude proxy;
    it is not a causal attribution because different vectors can cancel.
    """

    if partner_index.shape != partner_weights.shape:
        raise ValueError("partner_index and partner_weights must have equal shapes")
    if partner_index.shape != active_partner_mask.shape:
        raise ValueError("active_partner_mask must match partner_index")
    batch_size, query_length, partner_count = partner_index.shape
    anchor_history = causal_anchor_history_mask(anchor_index, query_length)
    zones = partner_distance_zones(partner_index)
    active = active_partner_mask.bool()
    weights = partner_weights.float() * active

    if relation_vectors is None:
        influence = weights
    else:
        if relation_vectors.shape[:3] != partner_index.shape:
            raise ValueError("relation_vectors must start with [batch, query, k]")
        influence = weights * relation_vectors.float().norm(dim=-1)
    influence_share = influence / influence.sum(-1, keepdim=True).clamp_min(1.0e-12)

    history_collision = anchor_history.gather(2, partner_index).logical_and(active)
    rank_weight = weights.sum((0, 1))
    rank_influence = (influence_share * active).sum((0, 1))
    zone_rows: dict[str, dict[str, float]] = {}
    for zone_id, name in enumerate(("near", "middle", "far")):
        zone_mask = zones.eq(zone_id) & active
        zone_rows[name] = {
            "selection_event_fraction": float(zone_mask.sum() / active.sum().clamp_min(1)),
            "softmax_weight_fraction": float(
                (weights * zone_mask).sum() / weights.sum().clamp_min(1.0e-12)
            ),
            "influence_fraction": float(
                (influence_share * zone_mask).sum()
                / influence_share.sum().clamp_min(1.0e-12)
            ),
        }

    shared_partner_fractions: list[float] = []
    mean_owner_counts: list[float] = []
    maximum_owner_count = 0
    shared_event_count = 0
    total_event_count = int(active.sum())
    anchor_cpu = anchor_index.detach().cpu()
    partner_cpu = partner_index.detach().cpu()
    active_cpu = active.detach().cpu()
    for batch in range(batch_size):
        owners: dict[int, set[int]] = defaultdict(set)
        for query in range(query_length):
            anchor = int(anchor_cpu[batch, query])
            for rank in range(partner_count):
                if bool(active_cpu[batch, query, rank]):
                    owners[int(partner_cpu[batch, query, rank])].add(anchor)
        if owners:
            shared_ids = {partner for partner, value in owners.items() if len(value) >= 2}
            shared_partner_fractions.append(len(shared_ids) / len(owners))
            mean_owner_counts.append(sum(map(len, owners.values())) / len(owners))
            maximum_owner_count = max(maximum_owner_count, max(map(len, owners.values())))
            for query in range(query_length):
                for rank in range(partner_count):
                    if (
                        bool(active_cpu[batch, query, rank])
                        and int(partner_cpu[batch, query, rank]) in shared_ids
                    ):
                        shared_event_count += 1

    entropy = -(weights.clamp_min(1.0e-12).log() * weights).sum(-1)
    valid_query = active.any(-1)
    return {
        "mean_active_partners": float(active.sum(-1)[valid_query].float().mean()),
        "mean_effective_partners": float(torch.exp(entropy)[valid_query].mean()),
        "weight_share_by_rank": (rank_weight / rank_weight.sum().clamp_min(1.0e-12)).tolist(),
        "influence_share_by_rank": (
            rank_influence / rank_influence.sum().clamp_min(1.0e-12)
        ).tolist(),
        "partner_is_previous_or_current_anchor_fraction": float(
            history_collision.sum() / active.sum().clamp_min(1)
        ),
        "mean_fraction_unique_partners_shared_by_multiple_anchors": sum(
            shared_partner_fractions
        )
        / max(1, len(shared_partner_fractions)),
        "mean_distinct_anchor_owners_per_partner": sum(mean_owner_counts)
        / max(1, len(mean_owner_counts)),
        "maximum_distinct_anchor_owners_for_one_partner": maximum_owner_count,
        "selection_event_fraction_using_cross_anchor_shared_partner": shared_event_count
        / max(1, total_event_count),
        "distance_zones": zone_rows,
    }
