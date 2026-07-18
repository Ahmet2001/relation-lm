from __future__ import annotations

import statistics
from collections import defaultdict

from torch import Tensor


def summarize_partner_structure(
    anchor_index: Tensor,
    partner_index: Tensor,
    active: Tensor,
    weight: Tensor,
    relation: Tensor | None = None,
) -> dict:
    """Summarize anchor/partner role reuse and relative-distance zones.

    Args:
        anchor_index: ``[B,T]`` selected anchor positions.
        partner_index: ``[B,T,K]`` selected partner positions.
        active: ``[B,T,K]`` valid partner mask.
        weight: ``[B,T,K]`` normalized partner weights.
        relation: optional ``[B,T,K,D]`` relation vectors. When supplied, the
            report includes weighted relation-norm contribution by zone/rank.
    """
    anchor = anchor_index.detach().cpu()
    partner = partner_index.detach().cpu()
    active_cpu = active.detach().cpu()
    weight_cpu = weight.detach().float().cpu()
    relation_cpu = relation.detach().float().cpu() if relation is not None else None
    batch, length = anchor.shape
    k_max = partner.size(-1)

    active_count = 0
    global_collision = 0
    causal_collision = 0
    shared_selection_count = 0
    shared_positions = 0
    total_partner_positions = 0
    anchors_per_shared: list[int] = []
    reused_anchor_counts: list[int] = []
    repeated_anchor_jaccards: list[float] = []

    zone_counts = {name: 0 for name in ('near', 'middle', 'far')}
    zone_weights = {name: 0.0 for name in ('near', 'middle', 'far')}
    zone_contributions = {name: 0.0 for name in ('near', 'middle', 'far')}
    rank_counts = [0] * k_max
    rank_weights = [0.0] * k_max
    rank_contributions = [0.0] * k_max

    for batch_index in range(batch):
        all_anchors = set(int(value) for value in anchor[batch_index].tolist())
        seen_anchors: set[int] = set()
        partner_to_anchors: dict[int, set[int]] = defaultdict(set)
        selections: list[int] = []
        partner_sets_by_anchor: dict[int, list[set[int]]] = defaultdict(list)

        for time in range(length):
            current_anchor = int(anchor[batch_index, time])
            seen_anchors.add(current_anchor)
            current_partners: set[int] = set()
            for rank in range(k_max):
                if not bool(active_cpu[batch_index, time, rank]):
                    continue
                current_partner = int(partner[batch_index, time, rank])
                current_weight = float(weight_cpu[batch_index, time, rank])
                contribution = current_weight
                if relation_cpu is not None:
                    contribution *= float(relation_cpu[batch_index, time, rank].norm())

                active_count += 1
                selections.append(current_partner)
                current_partners.add(current_partner)
                partner_to_anchors[current_partner].add(current_anchor)
                global_collision += int(current_partner in all_anchors)
                causal_collision += int(current_partner in seen_anchors)

                ratio = (time - current_partner) / max(1, time)
                zone = 'near' if ratio <= 1 / 3 else ('middle' if ratio <= 2 / 3 else 'far')
                zone_counts[zone] += 1
                zone_weights[zone] += current_weight
                zone_contributions[zone] += contribution
                rank_counts[rank] += 1
                rank_weights[rank] += current_weight
                rank_contributions[rank] += contribution
            partner_sets_by_anchor[current_anchor].append(current_partners)

        shared = {
            position: anchor_set
            for position, anchor_set in partner_to_anchors.items()
            if len(anchor_set) > 1
        }
        shared_positions += len(shared)
        total_partner_positions += len(partner_to_anchors)
        anchors_per_shared.extend(len(value) for value in shared.values())
        shared_selection_count += sum(position in shared for position in selections)

        for sets in partner_sets_by_anchor.values():
            reused_anchor_counts.append(len(sets))
            for left, right in zip(sets[:-1], sets[1:], strict=False):
                union = left | right
                if union:
                    repeated_anchor_jaccards.append(len(left & right) / len(union))

    total_weight = sum(zone_weights.values())
    total_contribution = sum(zone_contributions.values())
    return {
        'active_partner_selections': active_count,
        'global_anchor_role_collision_rate': global_collision / max(1, active_count),
        'causal_seen_anchor_collision_rate': causal_collision / max(1, active_count),
        'shared_partner_selection_rate': shared_selection_count / max(1, active_count),
        'shared_partner_position_rate': shared_positions / max(1, total_partner_positions),
        'mean_distinct_anchors_per_shared_partner': statistics.mean(anchors_per_shared)
        if anchors_per_shared
        else 0.0,
        'max_distinct_anchors_per_shared_partner': max(anchors_per_shared, default=0),
        'mean_anchor_reuse_count': statistics.mean(reused_anchor_counts)
        if reused_anchor_counts
        else 0.0,
        'mean_partner_set_jaccard_for_reused_anchor': statistics.mean(
            repeated_anchor_jaccards
        )
        if repeated_anchor_jaccards
        else 0.0,
        'zones': {
            zone: {
                'selection_fraction': zone_counts[zone] / max(1, active_count),
                'weight_mass_fraction': zone_weights[zone] / max(total_weight, 1e-12),
                'contribution_norm_fraction': zone_contributions[zone]
                / max(total_contribution, 1e-12),
            }
            for zone in ('near', 'middle', 'far')
        },
        'rank_profile': [
            {
                'rank': rank + 1,
                'mean_weight_when_active': rank_weights[rank] / max(1, rank_counts[rank]),
                'mean_weighted_relation_norm': rank_contributions[rank]
                / max(1, rank_counts[rank]),
                'selection_fraction': rank_counts[rank] / max(1, active_count),
            }
            for rank in range(k_max)
        ],
    }
