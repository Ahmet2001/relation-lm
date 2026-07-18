from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from relation_lm.kernels.relation_reduce import (
    factor_relation_first_layer,
    relation_hidden_cache_update,
    relation_hidden_cached,
    relation_norm_reduce,
)

pytestmark = pytest.mark.gpu


def _reference_context(
    raw: torch.Tensor,
    scores: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    position: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    relations = F.layer_norm(
        raw,
        (raw.size(-1),),
        norm_weight,
        norm_bias,
        eps,
    )
    k_limit = int(math.ceil(math.log2(float(position.item()) + 2.0)))
    k_limit = max(1, min(k_limit, scores.size(1)))
    ranks = torch.arange(scores.size(1), device=scores.device)[None]
    active = torch.isfinite(scores) & ranks.lt(k_limit)
    safe = scores.masked_fill(~active, -1.0e4)
    weights = safe.softmax(-1) * active
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0e-9)
    return (weights.unsqueeze(-1) * relations).sum(1)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("k_out", [5, 8])
def test_relation_reduce_kernels(batch: int, k_out: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(20260818 + batch)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    context = 512
    relation_dim = 128
    hidden_dim = 2304
    output_dim = 576
    position = torch.tensor(context - 1, device=device, dtype=torch.long)

    operand_cache = torch.randn(batch, context, relation_dim, device=device)
    first_weight = torch.randn(hidden_dim, 4 * relation_dim, device=device) * 0.02
    first_bias = torch.randn(hidden_dim, device=device) * 0.01
    factors = factor_relation_first_layer(first_weight, first_bias, relation_dim)
    expanded_cache = F.linear(
        operand_cache,
        factors.expanded_weight,
        factors.expanded_bias,
    )
    anchor_index = torch.randint(0, context, (batch,), device=device, dtype=torch.int32)
    partner_index = torch.randint(
        0,
        context,
        (batch, k_out),
        device=device,
        dtype=torch.int32,
    )
    anchor_index[0] = position.to(torch.int32)
    partner_index[:, 0] = position.to(torch.int32)

    batch_index = torch.arange(batch, device=device)
    anchor = operand_cache[batch_index, anchor_index.long()]
    partner = operand_cache[batch_index[:, None], partner_index.long()]
    anchor_expanded = anchor[:, None].expand(-1, k_out, -1)
    features = torch.cat(
        (
            anchor_expanded,
            partner,
            anchor_expanded * partner,
            anchor_expanded - partner,
        ),
        dim=-1,
    )
    reference_hidden = F.gelu(F.linear(features, first_weight, first_bias))
    factorized_hidden = F.gelu(
        F.linear(anchor, factors.anchor_weight, factors.first_bias)[:, None]
        + F.linear(partner, factors.partner_weight)
        + F.linear(anchor_expanded * partner, factors.product_weight)
    )
    assert torch.allclose(
        factorized_hidden, reference_hidden, atol=2.0e-4, rtol=2.0e-5
    )

    cached_hidden = relation_hidden_cached(
        expanded_cache,
        operand_cache,
        anchor_index,
        partner_index,
        factors.product_weight,
    )
    torch.cuda.synchronize()
    assert torch.allclose(cached_hidden, factorized_hidden, atol=2.0e-4, rtol=2.0e-5)

    mutable_cache = expanded_cache.clone()
    mutable_cache[:, position.item()].zero_()
    updated_hidden = relation_hidden_cache_update(
        mutable_cache,
        operand_cache,
        anchor_index,
        partner_index,
        factors.anchor_weight,
        factors.partner_weight,
        factors.product_weight,
        factors.first_bias,
        position,
    )
    torch.cuda.synchronize()
    assert torch.allclose(updated_hidden, factorized_hidden, atol=2.0e-4, rtol=2.0e-5)
    assert torch.allclose(
        mutable_cache[:, position.item()],
        expanded_cache[:, position.item()],
        atol=2.0e-4,
        rtol=2.0e-5,
    )

    second_weight = torch.randn(output_dim, hidden_dim, device=device) * 0.02
    second_bias = torch.randn(output_dim, device=device) * 0.01
    norm_weight = torch.randn(output_dim, device=device) * 0.05 + 1.0
    norm_bias = torch.randn(output_dim, device=device) * 0.01
    raw = F.linear(cached_hidden, second_weight, second_bias)
    scores = torch.randn(batch, k_out, device=device)
    scores[:, -1] = -torch.inf
    eps = 1.0e-5
    reference_context = _reference_context(
        raw,
        scores,
        norm_weight,
        norm_bias,
        position,
        eps,
    )
    fused_context = relation_norm_reduce(
        raw,
        scores,
        norm_weight,
        norm_bias,
        position,
        eps,
    )
    torch.cuda.synchronize()
    assert torch.allclose(fused_context, reference_context, atol=2.0e-4, rtol=2.0e-5)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("k_out", [5, 6, 7])
def test_relation_norm_reduce_non_power_of_two_budget(
    batch: int,
    k_out: int,
) -> None:
    """Triton rows are padded to a power of two and masked back to K."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(20260830 + 10 * batch + k_out)
    device = torch.device("cuda")
    output_dim = 576
    raw = torch.randn(batch, k_out, output_dim, device=device)
    scores = torch.randn(batch, k_out, device=device)
    scores[:, -1] = -torch.inf
    norm_weight = torch.randn(output_dim, device=device) * 0.05 + 1.0
    norm_bias = torch.randn(output_dim, device=device) * 0.01
    position = torch.tensor(511, device=device, dtype=torch.long)
    eps = 1.0e-5
    expected = _reference_context(
        raw,
        scores,
        norm_weight,
        norm_bias,
        position,
        eps,
    )
    actual = relation_norm_reduce(
        raw,
        scores,
        norm_weight,
        norm_bias,
        position,
        eps,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=2.0e-4, rtol=2.0e-5)
