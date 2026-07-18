from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor


@dataclass(frozen=True)
class FactorizedRelationWeights:
    """First Relation-MLP layer split into reusable token contributions."""

    anchor_weight: Tensor
    partner_weight: Tensor
    product_weight: Tensor
    first_bias: Tensor
    expanded_weight: Tensor
    expanded_bias: Tensor


def factor_relation_first_layer(
    first_weight: Tensor,
    first_bias: Tensor,
    relation_dim: int,
) -> FactorizedRelationWeights:
    """Factor ``W[a,p,a*p,a-p]+b`` into anchor, partner, and product terms.

    The identity is exact::

        (W_a + W_d) a + (W_p - W_d) p + W_m (a * p) + b

    ``expanded_weight`` projects every cached operand into concatenated
    ``[anchor_base, partner_base]`` contributions. Its bias is applied only to
    the anchor half so the original first-layer bias appears exactly once.
    """
    if first_weight.ndim != 2 or first_bias.ndim != 1:
        raise ValueError("first_weight must be 2D and first_bias must be 1D")
    if first_weight.size(1) != 4 * int(relation_dim):
        raise ValueError("first_weight input dimension must equal 4 * relation_dim")
    anchor_raw, partner_raw, product_weight, difference = first_weight.split(
        int(relation_dim), dim=1
    )
    anchor_weight = (anchor_raw + difference).contiguous()
    partner_weight = (partner_raw - difference).contiguous()
    product_weight = product_weight.contiguous()
    expanded_weight = torch.cat((anchor_weight, partner_weight), dim=0).contiguous()
    expanded_bias = torch.cat((first_bias, torch.zeros_like(first_bias)), dim=0).contiguous()
    return FactorizedRelationWeights(
        anchor_weight=anchor_weight,
        partner_weight=partner_weight,
        product_weight=product_weight,
        first_bias=first_bias.contiguous(),
        expanded_weight=expanded_weight,
        expanded_bias=expanded_bias,
    )


def recommended_relation_cache_mode(batch_size: int) -> str:
    """Return the empirically best cache-update mode for verified decode cases.

    The current benchmark validates batch sizes 1 and 8 at context 512. Batch 1
    benefits from fusing the current-token cache update into the relation-hidden
    kernel. Batch 8 benefits from a separate cuBLAS cache projection.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return "fused_update" if batch_size == 1 else "separate_update"


@triton.jit
def _relation_hidden_cached_kernel(
    EXPANDED_CACHE,
    OPERAND_CACHE,
    ANCHOR_INDEX,
    PARTNER_INDEX,
    PRODUCT_WEIGHT,
    OUT,
    MAX_T: tl.constexpr,
    RELATION_D: tl.constexpr,
    HIDDEN_D: tl.constexpr,
    K_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_index = tl.program_id(0)
    output_block = tl.program_id(1)
    rows = tl.arange(0, BLOCK_M)
    columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    relation_columns = tl.arange(0, RELATION_D)
    valid_row = rows < K_OUT
    valid_column = columns < HIDDEN_D
    expanded_stride = 2 * HIDDEN_D

    anchor_index = tl.load(ANCHOR_INDEX + batch_index).to(tl.int32)
    partner_index = tl.load(
        PARTNER_INDEX + batch_index * K_OUT + rows,
        mask=valid_row,
        other=0,
    ).to(tl.int32)
    anchor_base = tl.load(
        EXPANDED_CACHE
        + (batch_index * MAX_T + anchor_index) * expanded_stride
        + columns,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)
    partner_base = tl.load(
        EXPANDED_CACHE
        + (batch_index * MAX_T + partner_index[:, None]) * expanded_stride
        + HIDDEN_D
        + columns[None, :],
        mask=valid_row[:, None] & valid_column[None, :],
        other=0.0,
    ).to(tl.float32)
    anchor_operand = tl.load(
        OPERAND_CACHE
        + (batch_index * MAX_T + anchor_index) * RELATION_D
        + relation_columns
    ).to(tl.float32)
    partner_operand = tl.load(
        OPERAND_CACHE
        + (batch_index * MAX_T + partner_index[:, None]) * RELATION_D
        + relation_columns[None, :],
        mask=valid_row[:, None],
        other=0.0,
    ).to(tl.float32)
    product = partner_operand * anchor_operand[None, :]
    product_weight = tl.load(
        PRODUCT_WEIGHT
        + columns[None, :] * RELATION_D
        + relation_columns[:, None],
        mask=valid_column[None, :],
        other=0.0,
    ).to(tl.float32)
    hidden = tl.dot(product, product_weight, input_precision="ieee")
    hidden += anchor_base[None, :] + partner_base
    hidden = 0.5 * hidden * (1.0 + tl.erf(hidden * 0.7071067811865476))
    tl.store(
        OUT
        + (batch_index * K_OUT + rows[:, None]) * HIDDEN_D
        + columns[None, :],
        hidden,
        mask=valid_row[:, None] & valid_column[None, :],
    )


@torch.library.triton_op("relation_lm::relation_hidden_cached", mutates_args=())
def relation_hidden_cached(
    expanded_cache: Tensor,
    operand_cache: Tensor,
    anchor_index: Tensor,
    partner_index: Tensor,
    product_weight: Tensor,
) -> Tensor:
    """Gather relation operands and compute the factorized first MLP layer."""
    batch = operand_cache.size(0)
    k_out = partner_index.size(1)
    relation_dim = operand_cache.size(2)
    hidden_dim = product_weight.size(0)
    output = torch.empty(
        (batch, k_out, hidden_dim), device=operand_cache.device, dtype=torch.float32
    )
    block_n = 64
    torch.library.wrap_triton(_relation_hidden_cached_kernel)[
        (batch, triton.cdiv(hidden_dim, block_n))
    ](
        expanded_cache,
        operand_cache,
        anchor_index,
        partner_index,
        product_weight,
        output,
        MAX_T=operand_cache.size(1),
        RELATION_D=relation_dim,
        HIDDEN_D=hidden_dim,
        K_OUT=k_out,
        BLOCK_M=16,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return output


@triton.jit
def _relation_hidden_cache_update_kernel(
    EXPANDED_CACHE,
    OPERAND_CACHE,
    ANCHOR_INDEX,
    PARTNER_INDEX,
    ANCHOR_WEIGHT,
    PARTNER_WEIGHT,
    PRODUCT_WEIGHT,
    FIRST_BIAS,
    POSITION,
    OUT,
    MAX_T: tl.constexpr,
    RELATION_D: tl.constexpr,
    HIDDEN_D: tl.constexpr,
    K_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_index = tl.program_id(0)
    output_block = tl.program_id(1)
    rows = tl.arange(0, BLOCK_M)
    columns = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    relation_columns = tl.arange(0, RELATION_D)
    valid_row = rows < K_OUT
    valid_column = columns < HIDDEN_D
    expanded_stride = 2 * HIDDEN_D
    position = tl.load(POSITION).to(tl.int32)

    current_operand = tl.load(
        OPERAND_CACHE
        + (batch_index * MAX_T + position) * RELATION_D
        + relation_columns
    ).to(tl.float32)
    anchor_weight = tl.load(
        ANCHOR_WEIGHT
        + columns[:, None] * RELATION_D
        + relation_columns[None, :],
        mask=valid_column[:, None],
        other=0.0,
    ).to(tl.float32)
    partner_weight = tl.load(
        PARTNER_WEIGHT
        + columns[:, None] * RELATION_D
        + relation_columns[None, :],
        mask=valid_column[:, None],
        other=0.0,
    ).to(tl.float32)
    current_anchor = tl.sum(anchor_weight * current_operand[None, :], axis=1)
    current_anchor += tl.load(FIRST_BIAS + columns, mask=valid_column, other=0.0)
    current_partner = tl.sum(partner_weight * current_operand[None, :], axis=1)
    tl.store(
        EXPANDED_CACHE
        + (batch_index * MAX_T + position) * expanded_stride
        + columns,
        current_anchor,
        mask=valid_column,
    )
    tl.store(
        EXPANDED_CACHE
        + (batch_index * MAX_T + position) * expanded_stride
        + HIDDEN_D
        + columns,
        current_partner,
        mask=valid_column,
    )

    anchor_index = tl.load(ANCHOR_INDEX + batch_index).to(tl.int32)
    partner_index = tl.load(
        PARTNER_INDEX + batch_index * K_OUT + rows,
        mask=valid_row,
        other=0,
    ).to(tl.int32)
    cached_anchor = tl.load(
        EXPANDED_CACHE
        + (batch_index * MAX_T + anchor_index) * expanded_stride
        + columns,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)
    cached_partner = tl.load(
        EXPANDED_CACHE
        + (batch_index * MAX_T + partner_index[:, None]) * expanded_stride
        + HIDDEN_D
        + columns[None, :],
        mask=valid_row[:, None] & valid_column[None, :],
        other=0.0,
    ).to(tl.float32)
    anchor_base = tl.where(anchor_index == position, current_anchor, cached_anchor)
    partner_base = tl.where(
        partner_index[:, None] == position,
        current_partner[None, :],
        cached_partner,
    )
    anchor_operand = tl.load(
        OPERAND_CACHE
        + (batch_index * MAX_T + anchor_index) * RELATION_D
        + relation_columns
    ).to(tl.float32)
    partner_operand = tl.load(
        OPERAND_CACHE
        + (batch_index * MAX_T + partner_index[:, None]) * RELATION_D
        + relation_columns[None, :],
        mask=valid_row[:, None],
        other=0.0,
    ).to(tl.float32)
    product = partner_operand * anchor_operand[None, :]
    product_weight = tl.load(
        PRODUCT_WEIGHT
        + columns[None, :] * RELATION_D
        + relation_columns[:, None],
        mask=valid_column[None, :],
        other=0.0,
    ).to(tl.float32)
    hidden = tl.dot(product, product_weight, input_precision="ieee")
    hidden += anchor_base[None, :] + partner_base
    hidden = 0.5 * hidden * (1.0 + tl.erf(hidden * 0.7071067811865476))
    tl.store(
        OUT
        + (batch_index * K_OUT + rows[:, None]) * HIDDEN_D
        + columns[None, :],
        hidden,
        mask=valid_row[:, None] & valid_column[None, :],
    )


@torch.library.triton_op(
    "relation_lm::relation_hidden_cache_update",
    mutates_args=("expanded_cache",),
)
def relation_hidden_cache_update(
    expanded_cache: Tensor,
    operand_cache: Tensor,
    anchor_index: Tensor,
    partner_index: Tensor,
    anchor_weight: Tensor,
    partner_weight: Tensor,
    product_weight: Tensor,
    first_bias: Tensor,
    position: Tensor,
) -> Tensor:
    """Update the current expanded cache row and compute relation hidden states."""
    batch = operand_cache.size(0)
    k_out = partner_index.size(1)
    relation_dim = operand_cache.size(2)
    hidden_dim = product_weight.size(0)
    output = torch.empty(
        (batch, k_out, hidden_dim), device=operand_cache.device, dtype=torch.float32
    )
    block_n = 64
    torch.library.wrap_triton(_relation_hidden_cache_update_kernel)[
        (batch, triton.cdiv(hidden_dim, block_n))
    ](
        expanded_cache,
        operand_cache,
        anchor_index,
        partner_index,
        anchor_weight,
        partner_weight,
        product_weight,
        first_bias,
        position,
        output,
        MAX_T=operand_cache.size(1),
        RELATION_D=relation_dim,
        HIDDEN_D=hidden_dim,
        K_OUT=k_out,
        BLOCK_M=16,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return output


@triton.jit
def _relation_norm_reduce_kernel(
    RAW,
    SCORES,
    NORM_WEIGHT,
    NORM_BIAS,
    POSITION,
    OUT,
    OUTPUT_D: tl.constexpr,
    K_OUT: tl.constexpr,
    K_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    batch_index = tl.program_id(0)
    rows = tl.arange(0, K_BLOCK)
    columns = tl.arange(0, BLOCK_D)
    valid_row = rows < K_OUT
    valid_column = columns < OUTPUT_D
    raw = tl.load(
        RAW
        + (batch_index * K_OUT + rows[:, None]) * OUTPUT_D
        + columns[None, :],
        mask=valid_row[:, None] & valid_column[None, :],
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(raw, axis=1) / OUTPUT_D
    centered = tl.where(valid_column[None, :], raw - mean[:, None], 0.0)
    variance = tl.sum(centered * centered, axis=1) / OUTPUT_D
    inverse_std = tl.rsqrt(variance + EPS)
    gamma = tl.load(NORM_WEIGHT + columns, mask=valid_column, other=0.0).to(
        tl.float32
    )
    beta = tl.load(NORM_BIAS + columns, mask=valid_column, other=0.0).to(
        tl.float32
    )
    normalized = centered * inverse_std[:, None]
    normalized = normalized * gamma[None, :] + beta[None, :]

    position = tl.load(POSITION).to(tl.float32)
    k_limit = tl.ceil(tl.log2(position + 2.0)).to(tl.int32)
    k_limit = tl.minimum(tl.maximum(k_limit, 1), K_OUT)
    scores = tl.load(
        SCORES + batch_index * K_OUT + rows,
        mask=valid_row,
        other=-float("inf"),
    ).to(tl.float32)
    active = valid_row & (rows < k_limit) & (scores > -1.0e30)
    safe_scores = tl.where(active, scores, -1.0e4)
    maximum = tl.max(safe_scores, axis=0)
    probability = tl.exp(safe_scores - maximum) * active.to(tl.float32)
    probability = probability / tl.maximum(tl.sum(probability, axis=0), 1.0e-9)
    context = tl.sum(normalized * probability[:, None], axis=0)
    tl.store(
        OUT + batch_index * OUTPUT_D + columns,
        context,
        mask=valid_column,
    )


@torch.library.triton_op("relation_lm::relation_norm_reduce", mutates_args=())
def relation_norm_reduce(
    raw: Tensor,
    partner_scores: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    position: Tensor,
    norm_eps: float,
) -> Tensor:
    """Fuse per-partner LayerNorm, active-K softmax, and weighted reduction."""
    batch = raw.size(0)
    k_out = raw.size(1)
    output_dim = raw.size(2)
    context = torch.empty(
        (batch, output_dim), device=raw.device, dtype=torch.float32
    )
    torch.library.wrap_triton(_relation_norm_reduce_kernel)[(batch,)](
        raw,
        partner_scores,
        norm_weight,
        norm_bias,
        position,
        context,
        OUTPUT_D=output_dim,
        K_OUT=k_out,
        K_BLOCK=triton.next_power_of_2(k_out),
        BLOCK_D=triton.next_power_of_2(output_dim),
        EPS=norm_eps,
        num_warps=8,
    )
    return context
