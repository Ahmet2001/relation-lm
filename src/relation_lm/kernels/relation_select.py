from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _relation_select_packed_kernel(
    AQ,
    AK,
    ARQ,
    AMEAN,
    ASTD,
    AGAMMA,
    ABIAS,
    PARTNER_BASE,
    PARTNER_ANCHOR_CACHE,
    PK,
    PMEAN,
    PSTD,
    PGAMMA,
    PBIAS,
    POSITION,
    OUT_ANCHOR,
    OUT_ANCHOR_SCORE,
    OUT_PARTNER,
    OUT_PARTNER_SCORE,
    MAX_T: tl.constexpr,
    MAX_NB: tl.constexpr,
    EXACT_D: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_D: tl.constexpr,
    REL_BUCKETS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOCAL: tl.constexpr,
    TOP_BLOCKS: tl.constexpr,
    MAX_TOP: tl.constexpr,
    CANDIDATE_CAPACITY: tl.constexpr,
    K_OUT: tl.constexpr,
    STRICT_VALID: tl.constexpr,
):
    """Fused current-position anchor and partner selection.

    Partner projections are packed into one query-side and one cached anchor-side
    128-channel contribution. This avoids a model-width gather and two 1152x64 matvecs
    between anchor and partner selection.
    """
    b = tl.program_id(0)
    t = tl.load(POSITION).to(tl.int32)
    bo = tl.arange(0, MAX_NB)
    block_end = (bo + 1) * BLOCK_SIZE - 1
    complete = block_end <= t
    local_start = tl.maximum(t - LOCAL + 1, 0)
    if STRICT_VALID:
        remote_eligible = complete & (block_end < local_start)
    else:
        remote_eligible = complete

    distance = tl.maximum(t - block_end, 0)
    bucket = tl.minimum(
        tl.floor(tl.log2(distance.to(tl.float32) + 1.0)).to(tl.int32),
        REL_BUCKETS - 1,
    )
    hd = tl.arange(0, HEAD_D)

    # Anchor router block scores.
    anchor_block_scores = tl.zeros((MAX_NB,), tl.float32)
    for h in tl.static_range(0, HEADS):
        router_q = tl.load(ARQ + b * (HEADS * HEAD_D) + h * HEAD_D + hd).to(
            tl.float32
        )
        mean = tl.load(
            AMEAN
            + (((b * MAX_NB + bo[:, None]) * HEADS + h) * HEAD_D + hd[None, :]),
            mask=bo[:, None] < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        std = tl.load(
            ASTD
            + (((b * MAX_NB + bo[:, None]) * HEADS + h) * HEAD_D + hd[None, :]),
            mask=bo[:, None] < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(AGAMMA + h).to(tl.float32)
        gamma = tl.log(1.0 + tl.exp(gamma))
        score_h = (
            tl.sum(mean * router_q[None, :], axis=1)
            + gamma * tl.sum(std * tl.abs(router_q)[None, :], axis=1)
        ) * (HEAD_D**-0.5)
        score_h += tl.load(
            ABIAS + h * REL_BUCKETS + bucket,
            mask=bo < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        if h == 0:
            anchor_block_scores = score_h
        else:
            maximum = tl.maximum(anchor_block_scores, score_h)
            anchor_block_scores = maximum + tl.log(
                tl.exp(anchor_block_scores - maximum) + tl.exp(score_h - maximum)
            )
    anchor_block_scores = tl.where(
        remote_eligible, anchor_block_scores, -float("inf")
    )

    # Exact anchor selection over local tokens and selected remote blocks.
    exact_lane = tl.arange(0, EXACT_D)
    anchor_query = tl.load(AQ + b * EXACT_D + exact_lane).to(tl.float32)
    local_lane = tl.arange(0, LOCAL)
    local_pos = t - LOCAL + 1 + local_lane
    local_valid = (local_pos >= 0) & (local_pos <= t)
    local_key = tl.load(
        AK + (b * MAX_T + local_pos[:, None]) * EXACT_D + exact_lane[None, :],
        mask=local_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    local_score = tl.sum(local_key * anchor_query[None, :], axis=1) * (
        EXACT_D**-0.5
    )
    local_score = tl.where(local_valid, local_score, -float("inf"))
    anchor_score = tl.max(local_score, axis=0)
    anchor_lane = tl.argmax(local_score, axis=0, tie_break_left=True)
    anchor_pos = t - LOCAL + 1 + anchor_lane

    block_offsets = tl.arange(0, BLOCK_SIZE)
    anchor_work = anchor_block_scores
    for _ in tl.static_range(0, TOP_BLOCKS):
        block_value = tl.max(anchor_work, axis=0)
        block_index = tl.argmax(anchor_work, axis=0, tie_break_left=True)
        block_valid = block_value > -1.0e30
        remote_pos = block_index * BLOCK_SIZE + block_offsets
        remote_valid = (
            block_valid
            & (remote_pos >= 0)
            & (remote_pos <= t)
            & (remote_pos < local_start)
        )
        remote_key = tl.load(
            AK
            + (b * MAX_T + remote_pos[:, None]) * EXACT_D
            + exact_lane[None, :],
            mask=remote_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        remote_score = tl.sum(remote_key * anchor_query[None, :], axis=1) * (
            EXACT_D**-0.5
        )
        remote_score = tl.where(remote_valid, remote_score, -float("inf"))
        candidate_score = tl.max(remote_score, axis=0)
        candidate_lane = tl.argmax(remote_score, axis=0, tie_break_left=True)
        candidate_pos = block_index * BLOCK_SIZE + candidate_lane
        take = candidate_score > anchor_score
        take = take | (
            (candidate_score == anchor_score) & (candidate_pos < anchor_pos)
        )
        anchor_score = tl.where(take, candidate_score, anchor_score)
        anchor_pos = tl.where(take, candidate_pos, anchor_pos)
        anchor_work = tl.where(bo == block_index, -float("inf"), anchor_work)

    tl.store(OUT_ANCHOR + b, anchor_pos.to(tl.int32))
    tl.store(OUT_ANCHOR_SCORE + b, anchor_score)

    # Factorized partner queries: query-side projection plus cached anchor-side
    # projection. Both exact and router partner vectors have EXACT_D elements.
    packed_stride = 2 * EXACT_D
    partner_query = tl.load(
        PARTNER_BASE + b * packed_stride + exact_lane
    ).to(tl.float32)
    partner_query += tl.load(
        PARTNER_ANCHOR_CACHE
        + (b * MAX_T + anchor_pos) * packed_stride
        + exact_lane
    ).to(tl.float32)

    # Partner router block scores.
    partner_block_scores = tl.zeros((MAX_NB,), tl.float32)
    for h in tl.static_range(0, HEADS):
        router_q = tl.load(
            PARTNER_BASE
            + b * packed_stride
            + EXACT_D
            + h * HEAD_D
            + hd
        ).to(tl.float32)
        router_q += tl.load(
            PARTNER_ANCHOR_CACHE
            + (b * MAX_T + anchor_pos) * packed_stride
            + EXACT_D
            + h * HEAD_D
            + hd
        ).to(tl.float32)
        mean = tl.load(
            PMEAN
            + (((b * MAX_NB + bo[:, None]) * HEADS + h) * HEAD_D + hd[None, :]),
            mask=bo[:, None] < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        std = tl.load(
            PSTD
            + (((b * MAX_NB + bo[:, None]) * HEADS + h) * HEAD_D + hd[None, :]),
            mask=bo[:, None] < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(PGAMMA + h).to(tl.float32)
        gamma = tl.log(1.0 + tl.exp(gamma))
        score_h = (
            tl.sum(mean * router_q[None, :], axis=1)
            + gamma * tl.sum(std * tl.abs(router_q)[None, :], axis=1)
        ) * (HEAD_D**-0.5)
        score_h += tl.load(
            PBIAS + h * REL_BUCKETS + bucket,
            mask=bo < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        if h == 0:
            partner_block_scores = score_h
        else:
            maximum = tl.maximum(partner_block_scores, score_h)
            partner_block_scores = maximum + tl.log(
                tl.exp(partner_block_scores - maximum)
                + tl.exp(score_h - maximum)
            )
    partner_block_scores = tl.where(
        remote_eligible, partner_block_scores, -float("inf")
    )

    # Select a tiny number of remote blocks without sorting the full vector.
    selected_lane = tl.arange(0, MAX_TOP)
    selected_blocks = tl.zeros((MAX_TOP,), tl.int32)
    selected_valid = tl.zeros((MAX_TOP,), tl.int1)
    partner_work = partner_block_scores
    for rank in tl.static_range(0, TOP_BLOCKS):
        block_value = tl.max(partner_work, axis=0)
        block_index = tl.argmax(partner_work, axis=0, tie_break_left=True)
        valid = block_value > -1.0e30
        selected_blocks = tl.where(
            selected_lane == rank, block_index, selected_blocks
        )
        selected_valid = tl.where(selected_lane == rank, valid, selected_valid)
        partner_work = tl.where(bo == block_index, -float("inf"), partner_work)

    # Build local + selected-block candidates and perform exact small-K partner
    # selection. Lanes beyond the actual capacity remain invalid.
    candidate_lane = tl.arange(0, CANDIDATE_CAPACITY)
    is_local = candidate_lane < LOCAL
    candidate_local_pos = t - LOCAL + 1 + candidate_lane
    remote_rank = (candidate_lane - LOCAL) // BLOCK_SIZE
    remote_offset = (candidate_lane - LOCAL) - remote_rank * BLOCK_SIZE
    selected_block = tl.sum(
        tl.where(
            selected_lane[None, :] == remote_rank[:, None],
            selected_blocks[None, :],
            0,
        ),
        axis=1,
    )
    selected_ok = (
        tl.sum(
            tl.where(
                selected_lane[None, :] == remote_rank[:, None],
                selected_valid[None, :].to(tl.int32),
                0,
            ),
            axis=1,
        )
        > 0
    )
    candidate_remote_pos = selected_block * BLOCK_SIZE + remote_offset
    candidate_pos = tl.where(is_local, candidate_local_pos, candidate_remote_pos)
    valid = (
        is_local & (candidate_local_pos >= 0) & (candidate_local_pos <= t)
    ) | (
        (~is_local)
        & (remote_rank >= 0)
        & (remote_rank < TOP_BLOCKS)
        & (remote_offset < BLOCK_SIZE)
        & selected_ok
        & (candidate_remote_pos >= 0)
        & (candidate_remote_pos <= t)
        & (candidate_remote_pos < local_start)
    )
    valid = valid & (candidate_pos != anchor_pos)

    partner_key = tl.load(
        PK
        + (b * MAX_T + candidate_pos[:, None]) * EXACT_D
        + exact_lane[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    exact_score = tl.sum(partner_key * partner_query[None, :], axis=1) * (
        EXACT_D**-0.5
    )
    exact_score = tl.where(valid, exact_score, -float("inf"))

    for k in tl.static_range(0, K_OUT):
        best_score = tl.max(exact_score, axis=0)
        best_lane = tl.argmax(exact_score, axis=0, tie_break_left=True)
        best_pos = tl.sum(
            tl.where(candidate_lane == best_lane, candidate_pos, 0), axis=0
        )
        tl.store(OUT_PARTNER + b * K_OUT + k, best_pos.to(tl.int32))
        tl.store(OUT_PARTNER_SCORE + b * K_OUT + k, best_score)
        exact_score = tl.where(
            candidate_lane == best_lane, -float("inf"), exact_score
        )


@torch.library.triton_op("relation_lm::relation_select_packed", mutates_args=())
def relation_select_packed(
    anchor_query: Tensor,
    anchor_keys: Tensor,
    anchor_router_query: Tensor,
    anchor_block_mean: Tensor,
    anchor_block_std: Tensor,
    anchor_gamma: Tensor,
    anchor_bias: Tensor,
    partner_base: Tensor,
    partner_anchor_cache: Tensor,
    partner_keys: Tensor,
    partner_block_mean: Tensor,
    partner_block_std: Tensor,
    partner_gamma: Tensor,
    partner_bias: Tensor,
    position: Tensor,
    local_window: int,
    top_blocks: int,
    k_out: int,
    strict_valid: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch = anchor_query.size(0)
    out_anchor = torch.empty((batch,), device=anchor_query.device, dtype=torch.int32)
    out_anchor_score = torch.empty(
        (batch,), device=anchor_query.device, dtype=torch.float32
    )
    out_partner = torch.empty(
        (batch, k_out), device=anchor_query.device, dtype=torch.int32
    )
    out_partner_score = torch.empty(
        (batch, k_out), device=anchor_query.device, dtype=torch.float32
    )
    block_size = anchor_keys.size(1) // anchor_block_mean.size(1)
    capacity = triton.next_power_of_2(local_window + top_blocks * block_size)
    max_top = triton.next_power_of_2(max(1, top_blocks))
    torch.library.wrap_triton(_relation_select_packed_kernel)[(batch,)](
        anchor_query,
        anchor_keys,
        anchor_router_query,
        anchor_block_mean,
        anchor_block_std,
        anchor_gamma,
        anchor_bias,
        partner_base,
        partner_anchor_cache,
        partner_keys,
        partner_block_mean,
        partner_block_std,
        partner_gamma,
        partner_bias,
        position,
        out_anchor,
        out_anchor_score,
        out_partner,
        out_partner_score,
        MAX_T=anchor_keys.size(1),
        MAX_NB=anchor_block_mean.size(1),
        EXACT_D=anchor_query.size(1),
        HEADS=anchor_block_mean.size(2),
        HEAD_D=anchor_block_mean.size(3),
        REL_BUCKETS=anchor_bias.size(1),
        BLOCK_SIZE=block_size,
        LOCAL=local_window,
        TOP_BLOCKS=top_blocks,
        MAX_TOP=max_top,
        CANDIDATE_CAPACITY=capacity,
        K_OUT=k_out,
        STRICT_VALID=strict_valid,
        num_warps=1,
    )
    return out_anchor, out_anchor_score, out_partner, out_partner_score
