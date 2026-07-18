from __future__ import annotations

import torch

from relation_lm.models import (
    MultiAnchorRelationBlock,
    MultiRelationBlockConfig,
    RelationBlockStack,
)


def test_shapes_roles_and_causality() -> None:
    torch.manual_seed(7)
    cfg = MultiRelationBlockConfig(
        d_model=48,
        gate_dim=16,
        relation_dim=24,
        anchors_per_block=3,
        partners_per_anchor=4,
    )
    stack = RelationBlockStack(cfg, num_blocks=2).eval()
    hidden = torch.randn(2, 12, 48)
    memory = torch.randn(2, 12, 48)
    output, diagnostics = stack(hidden, memory)
    assert output.shape == hidden.shape
    assert len(diagnostics) == 2
    for row in diagnostics:
        anchors = row["anchor_index"]
        anchor_active = row["anchor_active"]
        partners = row["partner_index"]
        partner_active = row["partner_active"]
        assert anchors.shape == (2, 12, 3)
        assert partners.shape == (2, 12, 3, 4)
        for batch in range(2):
            for time in range(12):
                active_anchors = anchors[batch, time][anchor_active[batch, time]]
                anchor_set = set(int(value) for value in active_anchors)
                assert len(anchor_set) == min(3, time + 1)
                assert all(int(value) <= time for value in active_anchors)
                active_partners = partners[batch, time][partner_active[batch, time]]
                assert all(int(value) <= time for value in active_partners)
                assert not any(int(value) in anchor_set for value in active_partners)

    altered_hidden = hidden.clone()
    altered_memory = memory.clone()
    altered_hidden[:, 8:] = torch.randn_like(altered_hidden[:, 8:])
    altered_memory[:, 8:] = torch.randn_like(altered_memory[:, 8:])
    changed, _ = stack(altered_hidden, altered_memory)
    assert float((output[:, :8] - changed[:, :8]).detach().abs().max()) < 1.0e-5


def test_single_block_backward() -> None:
    cfg = MultiRelationBlockConfig(
        d_model=32,
        gate_dim=8,
        relation_dim=16,
        anchors_per_block=2,
        partners_per_anchor=3,
    )
    block = MultiAnchorRelationBlock(cfg)
    hidden = torch.randn(2, 10, 32, requires_grad=True)
    memory = torch.randn(2, 10, 32, requires_grad=True)
    output, diagnostics = block(hidden, memory)
    loss = output.square().mean() + diagnostics["partner_weight"].square().mean()
    loss.backward()
    assert hidden.grad is not None
    assert memory.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(memory.grad).all()
