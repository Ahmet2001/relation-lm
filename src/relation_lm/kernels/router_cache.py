from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _router_cache_update_kernel(
    ANCHOR_RAW,
    PARTNER_RAW,
    ANCHOR_RAW_CACHE,
    PARTNER_RAW_CACHE,
    ANCHOR_CONV_CACHE,
    PARTNER_CONV_CACHE,
    ANCHOR_MEAN,
    ANCHOR_STD,
    PARTNER_MEAN,
    PARTNER_STD,
    ANCHOR_CONV_WEIGHT,
    PARTNER_CONV_WEIGHT,
    POSITION,
    OUT,
    MAX_T: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    ROUTER_D: tl.constexpr,
    ANCHOR_RAW_STRIDE: tl.constexpr,
    PARTNER_RAW_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_index = tl.program_id(0)
    columns = tl.arange(0, BLOCK_D)
    valid_column = columns < ROUTER_D
    position = tl.load(POSITION).to(tl.int32)

    anchor_raw = tl.load(
        ANCHOR_RAW + batch_index * ANCHOR_RAW_STRIDE + columns,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)
    partner_raw = tl.load(
        PARTNER_RAW + batch_index * PARTNER_RAW_STRIDE + columns,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)

    previous_1_index = tl.maximum(position - 1, 0)
    previous_2_index = tl.maximum(position - 2, 0)
    anchor_previous_1 = tl.load(
        ANCHOR_RAW_CACHE
        + (batch_index * MAX_T + previous_1_index) * ROUTER_D
        + columns,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)
    anchor_previous_2 = tl.load(
        ANCHOR_RAW_CACHE
        + (batch_index * MAX_T + previous_2_index) * ROUTER_D
        + columns,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)
    partner_previous_1 = tl.load(
        PARTNER_RAW_CACHE
        + (batch_index * MAX_T + previous_1_index) * ROUTER_D
        + columns,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)
    partner_previous_2 = tl.load(
        PARTNER_RAW_CACHE
        + (batch_index * MAX_T + previous_2_index) * ROUTER_D
        + columns,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)

    anchor_weight_0 = tl.load(
        ANCHOR_CONV_WEIGHT + columns * 3,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)
    anchor_weight_1 = tl.load(
        ANCHOR_CONV_WEIGHT + columns * 3 + 1,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)
    anchor_weight_2 = tl.load(
        ANCHOR_CONV_WEIGHT + columns * 3 + 2,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)
    partner_weight_0 = tl.load(
        PARTNER_CONV_WEIGHT + columns * 3,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)
    partner_weight_1 = tl.load(
        PARTNER_CONV_WEIGHT + columns * 3 + 1,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)
    partner_weight_2 = tl.load(
        PARTNER_CONV_WEIGHT + columns * 3 + 2,
        mask=valid_column,
        other=0.0,
    ).to(tl.float32)

    valid_previous_1 = (position >= 1).to(tl.float32)
    valid_previous_2 = (position >= 2).to(tl.float32)
    anchor_conv = anchor_raw * anchor_weight_2
    anchor_conv += anchor_previous_1 * anchor_weight_1 * valid_previous_1
    anchor_conv += anchor_previous_2 * anchor_weight_0 * valid_previous_2
    partner_conv = partner_raw * partner_weight_2
    partner_conv += partner_previous_1 * partner_weight_1 * valid_previous_1
    partner_conv += partner_previous_2 * partner_weight_0 * valid_previous_2

    current_base = (batch_index * MAX_T + position) * ROUTER_D + columns
    tl.store(ANCHOR_RAW_CACHE + current_base, anchor_raw, mask=valid_column)
    tl.store(PARTNER_RAW_CACHE + current_base, partner_raw, mask=valid_column)
    tl.store(ANCHOR_CONV_CACHE + current_base, anchor_conv, mask=valid_column)
    tl.store(PARTNER_CONV_CACHE + current_base, partner_conv, mask=valid_column)

    block_index = position // BLOCK_SIZE
    block_start = block_index * BLOCK_SIZE
    rows = tl.arange(0, BLOCK_SIZE)
    history_indices = block_start + rows
    history_offsets = (
        (batch_index * MAX_T + history_indices[:, None]) * ROUTER_D
        + columns[None, :]
    )
    anchor_block = tl.load(
        ANCHOR_CONV_CACHE + history_offsets,
        mask=valid_column[None, :],
        other=0.0,
    ).to(tl.float32)
    partner_block = tl.load(
        PARTNER_CONV_CACHE + history_offsets,
        mask=valid_column[None, :],
        other=0.0,
    ).to(tl.float32)
    is_current = history_indices[:, None] == position
    anchor_block = tl.where(is_current, anchor_conv[None, :], anchor_block)
    partner_block = tl.where(is_current, partner_conv[None, :], partner_block)

    anchor_mean = tl.sum(anchor_block, axis=0) / BLOCK_SIZE
    partner_mean = tl.sum(partner_block, axis=0) / BLOCK_SIZE
    anchor_centered = anchor_block - anchor_mean[None, :]
    partner_centered = partner_block - partner_mean[None, :]
    anchor_variance = tl.sum(anchor_centered * anchor_centered, axis=0) / BLOCK_SIZE
    partner_variance = tl.sum(partner_centered * partner_centered, axis=0) / BLOCK_SIZE
    anchor_std = tl.sqrt(anchor_variance)
    partner_std = tl.sqrt(partner_variance)

    stats_base = (batch_index * MAX_BLOCKS + block_index) * ROUTER_D + columns
    tl.store(ANCHOR_MEAN + stats_base, anchor_mean, mask=valid_column)
    tl.store(ANCHOR_STD + stats_base, anchor_std, mask=valid_column)
    tl.store(PARTNER_MEAN + stats_base, partner_mean, mask=valid_column)
    tl.store(PARTNER_STD + stats_base, partner_std, mask=valid_column)

    tl.store(
        OUT + (batch_index * 2) * ROUTER_D + columns,
        anchor_conv,
        mask=valid_column,
    )
    tl.store(
        OUT + (batch_index * 2 + 1) * ROUTER_D + columns,
        partner_conv,
        mask=valid_column,
    )


@torch.library.triton_op(
    "relation_lm::router_cache_update",
    mutates_args=(
        "anchor_raw_cache",
        "partner_raw_cache",
        "anchor_conv_cache",
        "partner_conv_cache",
        "anchor_mean",
        "anchor_std",
        "partner_mean",
        "partner_std",
    ),
)
def router_cache_update(
    anchor_raw: Tensor,
    partner_raw: Tensor,
    anchor_raw_cache: Tensor,
    partner_raw_cache: Tensor,
    anchor_conv_cache: Tensor,
    partner_conv_cache: Tensor,
    anchor_mean: Tensor,
    anchor_std: Tensor,
    partner_mean: Tensor,
    partner_std: Tensor,
    anchor_conv_weight: Tensor,
    partner_conv_weight: Tensor,
    position: Tensor,
    block_size: int,
) -> Tensor:
    """Update both router streams, causal conv state, and current-block stats.

    ``anchor_raw`` and ``partner_raw`` may be non-contiguous views from a
    larger packed projection. Row strides are passed explicitly, so callers do
    not need to materialize additional contiguous copies.
    """
    batch = anchor_raw.size(0)
    router_dim = anchor_raw.size(1)
    output = torch.empty(
        (batch, 2, router_dim),
        device=anchor_raw.device,
        dtype=torch.float32,
    )
    torch.library.wrap_triton(_router_cache_update_kernel)[(batch,)](
        anchor_raw,
        partner_raw,
        anchor_raw_cache,
        partner_raw_cache,
        anchor_conv_cache,
        partner_conv_cache,
        anchor_mean,
        anchor_std,
        partner_mean,
        partner_std,
        anchor_conv_weight,
        partner_conv_weight,
        position,
        output,
        MAX_T=anchor_raw_cache.size(1),
        MAX_BLOCKS=anchor_mean.size(1),
        ROUTER_D=router_dim,
        ANCHOR_RAW_STRIDE=anchor_raw.stride(0),
        PARTNER_RAW_STRIDE=partner_raw.stride(0),
        BLOCK_SIZE=block_size,
        BLOCK_D=triton.next_power_of_2(router_dim),
        num_warps=4,
    )
    return output
