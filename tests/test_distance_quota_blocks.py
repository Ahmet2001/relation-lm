from __future__ import annotations

import torch

from relation_lm.models.distance_quota_blocks import (
    DistanceQuotaRelationBlock,
    DistanceQuotaRelationStack,
    QuotaRelationBlockConfig,
)


def _config(quota: tuple[int, int, int]) -> QuotaRelationBlockConfig:
    return QuotaRelationBlockConfig(
        d_model=16,
        gate_dim=8,
        relation_dim=8,
        ff_mult=2,
        num_anchors=4,
        zone_quotas=quota,
    )


def test_distance_quota_block_is_causal_and_excludes_anchors() -> None:
    torch.manual_seed(3)
    block = DistanceQuotaRelationBlock(_config((2, 1, 1))).eval()
    query = torch.randn(2, 12, 16)
    memory = torch.randn(2, 12, 16)
    output, info = block(query, memory, return_diagnostics=True)

    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert [(info["slot_zones"] == zone).sum().item() for zone in range(3)] == [2, 1, 1]
    violations = (
        info["partner_positions"].unsqueeze(-1)
        == info["anchor_positions"].unsqueeze(-2).unsqueeze(-2)
    ) & info["partner_valid"].unsqueeze(-1)
    assert not violations.any()

    altered_query = query.clone()
    altered_memory = memory.clone()
    altered_query[:, 8:] = torch.randn_like(altered_query[:, 8:])
    altered_memory[:, 8:] = torch.randn_like(altered_memory[:, 8:])
    changed = block(altered_query, altered_memory)
    assert torch.allclose(output[:, :8], changed[:, :8], atol=1e-6, rtol=0.0)


def test_distance_quota_auxiliary_losses_backpropagate() -> None:
    torch.manual_seed(4)
    block = DistanceQuotaRelationBlock(_config((1, 2, 1)))
    query = torch.randn(2, 16, 16, requires_grad=True)
    memory = torch.randn(2, 16, 16, requires_grad=True)
    output, info = block(query, memory, return_diagnostics=True)
    loss = (
        output.square().mean()
        + 0.02 * info["cross_anchor_overlap_loss"]
        + 0.1 * info["zone_balance_loss"]
    )
    loss.backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert memory.grad is not None and torch.isfinite(memory.grad).all()


def test_distance_quota_stack_has_layer_specific_roles() -> None:
    stack = DistanceQuotaRelationStack(
        [_config((2, 1, 1)), _config((1, 2, 1)), _config((1, 1, 2))]
    )
    query = torch.randn(1, 12, 16)
    memory = torch.randn(1, 12, 16)
    output, diagnostics = stack(query, memory, return_diagnostics=True)
    assert output.shape == query.shape
    assert len(diagnostics) == 3
    observed = [
        [(row["slot_zones"] == zone).sum().item() for zone in range(3)]
        for row in diagnostics
    ]
    assert observed == [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
