from __future__ import annotations

import torch

from relation_lm.analysis.partner_dynamics import (
    causal_anchor_history_mask,
    partner_distance_zones,
    summarize_partner_dynamics,
)
from relation_lm.models.stacked_relation import (
    RelationBlockConfig,
    StackedRelationLayers,
)


def test_causal_anchor_history_mask() -> None:
    anchors = torch.tensor([[0, 1, 1, 3]])
    mask = causal_anchor_history_mask(anchors, 4)
    assert mask.shape == (1, 4, 4)
    assert mask[0, 0].tolist() == [True, False, False, False]
    assert mask[0, 2].tolist() == [True, True, False, False]
    assert mask[0, 3].tolist() == [True, True, False, True]


def test_partner_dynamics_summary_and_zones() -> None:
    anchors = torch.tensor([[0, 0, 1, 1]])
    partners = torch.tensor([[[0, 0], [1, 0], [2, 0], [3, 0]]])
    weights = torch.tensor([[[0.0, 0.0], [0.7, 0.3], [0.6, 0.4], [0.8, 0.2]]])
    active = weights > 0
    vectors = torch.ones(1, 4, 2, 3)
    zones = partner_distance_zones(partners)
    assert zones.shape == partners.shape
    summary = summarize_partner_dynamics(anchors, partners, weights, active, vectors)
    assert summary["mean_active_partners"] == 2.0
    assert 0.0 <= summary["partner_is_previous_or_current_anchor_fraction"] <= 1.0
    assert set(summary["distance_zones"]) == {"near", "middle", "far"}


def test_stacked_relation_layers_are_causal() -> None:
    torch.manual_seed(7)
    config = RelationBlockConfig(
        d_model=16,
        gate_dim=4,
        relation_dim=4,
        hidden_dim=16,
        k_max=4,
    )
    model = StackedRelationLayers([config, config]).eval()
    hidden = torch.randn(2, 8, 16)
    memory = torch.randn(2, 8, 16)
    changed_hidden = hidden.clone()
    changed_memory = memory.clone()
    changed_hidden[:, 5:] = torch.randn_like(changed_hidden[:, 5:])
    changed_memory[:, 5:] = torch.randn_like(changed_memory[:, 5:])
    output = model(
        hidden,
        memory,
        exclude_anchor_history_from_partners=True,
    )
    changed_output = model(
        changed_hidden,
        changed_memory,
        exclude_anchor_history_from_partners=True,
    )
    assert torch.allclose(output[:, :5], changed_output[:, :5], atol=1.0e-6, rtol=0.0)
