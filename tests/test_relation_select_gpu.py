from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_relation_select_matches_two_kernel_path() -> None:
    from relation_lm.kernels.relation_select import relation_select_packed
    from relation_lm.kernels.triton_select import anchor_select, partner_topk

    torch.manual_seed(29)
    device = torch.device("cuda")
    batch, context, blocks = 4, 512, 64
    anchor_query = torch.randn(batch, 64, device=device)
    anchor_keys = torch.randn(batch, context, 64, device=device)
    anchor_router_query = torch.randn(batch, 64, device=device)
    anchor_mean = torch.randn(batch, blocks, 4, 16, device=device)
    anchor_std = torch.rand(batch, blocks, 4, 16, device=device)
    anchor_gamma = torch.randn(4, device=device)
    anchor_bias = torch.randn(4, 16, device=device)
    partner_base = torch.randn(batch, 128, device=device)
    partner_anchor_cache = torch.randn(batch, context, 128, device=device)
    partner_keys = torch.randn(batch, context, 64, device=device)
    partner_mean = torch.randn(batch, blocks, 4, 16, device=device)
    partner_std = torch.rand(batch, blocks, 4, 16, device=device)
    partner_gamma = torch.randn(4, device=device)
    partner_bias = torch.randn(4, 16, device=device)
    position = torch.tensor(context - 1, device=device, dtype=torch.long)

    anchor, anchor_score = anchor_select(
        anchor_query,
        anchor_keys,
        anchor_router_query,
        anchor_mean,
        anchor_std,
        anchor_gamma,
        anchor_bias,
        position,
        16,
        2,
        False,
    )
    batch_index = torch.arange(batch, device=device)
    contribution = partner_anchor_cache[batch_index, anchor.long()]
    partner, partner_score = partner_topk(
        (partner_base[:, :64] + contribution[:, :64]).contiguous(),
        partner_keys,
        (partner_base[:, 64:] + contribution[:, 64:]).contiguous(),
        partner_mean,
        partner_std,
        partner_gamma,
        partner_bias,
        position,
        anchor,
        16,
        2,
        8,
        False,
    )
    fused = relation_select_packed(
        anchor_query,
        anchor_keys,
        anchor_router_query,
        anchor_mean,
        anchor_std,
        anchor_gamma,
        anchor_bias,
        partner_base,
        partner_anchor_cache,
        partner_keys,
        partner_mean,
        partner_std,
        partner_gamma,
        partner_bias,
        position,
        16,
        2,
        8,
        False,
    )
    assert torch.equal(anchor, fused[0])
    assert torch.equal(partner, fused[2])
    assert torch.allclose(anchor_score, fused[1], atol=2e-4, rtol=0)
    assert torch.allclose(partner_score, fused[3], atol=2e-4, rtol=0)
