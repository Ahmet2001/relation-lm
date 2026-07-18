from __future__ import annotations

import torch

from relation_lm.analysis import summarize_partner_structure


def test_partner_structure_summary() -> None:
    anchor = torch.tensor([[0, 0, 2, 2]])
    partner = torch.tensor([[[1, 2], [1, 2], [1, 3], [1, 3]]])
    active = torch.ones_like(partner, dtype=torch.bool)
    weight = torch.full(partner.shape, 0.5)
    relation = torch.ones(1, 4, 2, 3)
    report = summarize_partner_structure(anchor, partner, active, weight, relation)
    assert report['active_partner_selections'] == 8
    assert report['shared_partner_selection_rate'] > 0
    assert report['global_anchor_role_collision_rate'] > 0
    assert abs(sum(row['selection_fraction'] for row in report['zones'].values()) - 1) < 1e-6
