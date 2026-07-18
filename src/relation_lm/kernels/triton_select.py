from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _anchor_select_kernel(
    AQ, AK, RQ, MEAN, STD, GAMMA, BIAS, POSITION, OUT_INDEX, OUT_SCORE,
    MAX_T: tl.constexpr,
    MAX_NB: tl.constexpr,
    EXACT_D: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_D: tl.constexpr,
    REL_BUCKETS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOCAL: tl.constexpr,
    TOP_BLOCKS: tl.constexpr,
    STRICT_VALID: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.load(POSITION).to(tl.int32)
    bo = tl.arange(0, MAX_NB)
    end = (bo + 1) * BLOCK_SIZE - 1
    complete = end <= t
    local_start = tl.maximum(t - LOCAL + 1, 0)
    if STRICT_VALID:
        remote_eligible = complete & (end < local_start)
    else:
        remote_eligible = complete

    distance = tl.maximum(t - end, 0)
    bucket = tl.minimum(
        tl.floor(tl.log2(distance.to(tl.float32) + 1.0)).to(tl.int32),
        REL_BUCKETS - 1,
    )
    hd = tl.arange(0, HEAD_D)
    block_scores = tl.zeros((MAX_NB,), tl.float32)
    for h in tl.static_range(0, HEADS):
        rq = tl.load(RQ + b * (HEADS * HEAD_D) + h * HEAD_D + hd).to(tl.float32)
        mean = tl.load(
            MEAN + (((b * MAX_NB + bo[:, None]) * HEADS + h) * HEAD_D + hd[None, :]),
            mask=bo[:, None] < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        std = tl.load(
            STD + (((b * MAX_NB + bo[:, None]) * HEADS + h) * HEAD_D + hd[None, :]),
            mask=bo[:, None] < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(GAMMA + h).to(tl.float32)
        gamma = tl.log(1.0 + tl.exp(gamma))
        score_h = (
            tl.sum(mean * rq[None, :], axis=1)
            + gamma * tl.sum(std * tl.abs(rq)[None, :], axis=1)
        ) * (HEAD_D ** -0.5)
        score_h += tl.load(
            BIAS + h * REL_BUCKETS + bucket,
            mask=bo < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        if h == 0:
            block_scores = score_h
        else:
            mx = tl.maximum(block_scores, score_h)
            block_scores = mx + tl.log(
                tl.exp(block_scores - mx) + tl.exp(score_h - mx)
            )
    block_scores = tl.where(remote_eligible, block_scores, -float("inf"))

    ed = tl.arange(0, EXACT_D)
    aq = tl.load(AQ + b * EXACT_D + ed).to(tl.float32)
    local_lane = tl.arange(0, LOCAL)
    local_pos = t - LOCAL + 1 + local_lane
    local_valid = (local_pos >= 0) & (local_pos <= t)
    local_key = tl.load(
        AK + (b * MAX_T + local_pos[:, None]) * EXACT_D + ed[None, :],
        mask=local_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    local_score = tl.sum(local_key * aq[None, :], axis=1) * (EXACT_D ** -0.5)
    local_score = tl.where(local_valid, local_score, -float("inf"))
    best_score = tl.max(local_score, axis=0)
    best_lane = tl.argmax(local_score, axis=0, tie_break_left=True)
    best_pos = t - LOCAL + 1 + best_lane

    block_offsets = tl.arange(0, BLOCK_SIZE)
    scores_work = block_scores
    for _ in tl.static_range(0, TOP_BLOCKS):
        block_value = tl.max(scores_work, axis=0)
        block_index = tl.argmax(scores_work, axis=0, tie_break_left=True)
        block_valid = block_value > -1.0e30
        remote_pos = block_index * BLOCK_SIZE + block_offsets
        remote_valid = (
            block_valid
            & (remote_pos >= 0)
            & (remote_pos <= t)
            & (remote_pos < local_start)
        )
        remote_key = tl.load(
            AK + (b * MAX_T + remote_pos[:, None]) * EXACT_D + ed[None, :],
            mask=remote_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        remote_score = tl.sum(remote_key * aq[None, :], axis=1) * (EXACT_D ** -0.5)
        remote_score = tl.where(remote_valid, remote_score, -float("inf"))
        candidate_score = tl.max(remote_score, axis=0)
        candidate_lane = tl.argmax(remote_score, axis=0, tie_break_left=True)
        candidate_pos = block_index * BLOCK_SIZE + candidate_lane
        take = candidate_score > best_score
        tie = candidate_score == best_score
        take = take | (tie & (candidate_pos < best_pos))
        best_score = tl.where(take, candidate_score, best_score)
        best_pos = tl.where(take, candidate_pos, best_pos)
        scores_work = tl.where(bo == block_index, -float("inf"), scores_work)

    tl.store(OUT_INDEX + b, best_pos.to(tl.int32))
    tl.store(OUT_SCORE + b, best_score)


@triton.jit
def _partner_topk_kernel(
    PQ, PK, RQ, MEAN, STD, GAMMA, BIAS, POSITION, ANCHOR, OUT_INDEX, OUT_SCORE,
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
    b = tl.program_id(0)
    t = tl.load(POSITION).to(tl.int32)
    bo = tl.arange(0, MAX_NB)
    end = (bo + 1) * BLOCK_SIZE - 1
    complete = end <= t
    local_start = tl.maximum(t - LOCAL + 1, 0)
    if STRICT_VALID:
        remote_eligible = complete & (end < local_start)
    else:
        remote_eligible = complete

    distance = tl.maximum(t - end, 0)
    bucket = tl.minimum(
        tl.floor(tl.log2(distance.to(tl.float32) + 1.0)).to(tl.int32),
        REL_BUCKETS - 1,
    )
    hd = tl.arange(0, HEAD_D)
    block_scores = tl.zeros((MAX_NB,), tl.float32)
    for h in tl.static_range(0, HEADS):
        rq = tl.load(RQ + b * (HEADS * HEAD_D) + h * HEAD_D + hd).to(tl.float32)
        mean = tl.load(
            MEAN + (((b * MAX_NB + bo[:, None]) * HEADS + h) * HEAD_D + hd[None, :]),
            mask=bo[:, None] < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        std = tl.load(
            STD + (((b * MAX_NB + bo[:, None]) * HEADS + h) * HEAD_D + hd[None, :]),
            mask=bo[:, None] < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(GAMMA + h).to(tl.float32)
        gamma = tl.log(1.0 + tl.exp(gamma))
        score_h = (
            tl.sum(mean * rq[None, :], axis=1)
            + gamma * tl.sum(std * tl.abs(rq)[None, :], axis=1)
        ) * (HEAD_D ** -0.5)
        score_h += tl.load(
            BIAS + h * REL_BUCKETS + bucket,
            mask=bo < MAX_NB,
            other=0.0,
        ).to(tl.float32)
        if h == 0:
            block_scores = score_h
        else:
            mx = tl.maximum(block_scores, score_h)
            block_scores = mx + tl.log(
                tl.exp(block_scores - mx) + tl.exp(score_h - mx)
            )
    block_scores = tl.where(remote_eligible, block_scores, -float("inf"))

    selected_lane = tl.arange(0, MAX_TOP)
    selected_blocks = tl.zeros((MAX_TOP,), tl.int32)
    selected_valid = tl.zeros((MAX_TOP,), tl.int1)
    scores_work = block_scores
    for rank in tl.static_range(0, TOP_BLOCKS):
        block_value = tl.max(scores_work, axis=0)
        block_index = tl.argmax(scores_work, axis=0, tie_break_left=True)
        valid = block_value > -1.0e30
        selected_blocks = tl.where(selected_lane == rank, block_index, selected_blocks)
        selected_valid = tl.where(selected_lane == rank, valid, selected_valid)
        scores_work = tl.where(bo == block_index, -float("inf"), scores_work)

    candidate_lane = tl.arange(0, CANDIDATE_CAPACITY)
    is_local = candidate_lane < LOCAL
    local_pos = t - LOCAL + 1 + candidate_lane
    remote_rank = (candidate_lane - LOCAL) // BLOCK_SIZE
    remote_offset = (candidate_lane - LOCAL) - remote_rank * BLOCK_SIZE
    selected_block = tl.sum(
        tl.where(selected_lane[None, :] == remote_rank[:, None], selected_blocks[None, :], 0),
        axis=1,
    )
    selected_ok = tl.sum(
        tl.where(
            selected_lane[None, :] == remote_rank[:, None],
            selected_valid[None, :].to(tl.int32),
            0,
        ),
        axis=1,
    ) > 0
    remote_pos = selected_block * BLOCK_SIZE + remote_offset
    candidate_pos = tl.where(is_local, local_pos, remote_pos)
    valid = (
        is_local & (local_pos >= 0) & (local_pos <= t)
    ) | (
        (~is_local)
        & (remote_rank >= 0)
        & (remote_rank < TOP_BLOCKS)
        & (remote_offset < BLOCK_SIZE)
        & selected_ok
        & (remote_pos >= 0)
        & (remote_pos <= t)
        & (remote_pos < local_start)
    )
    anchor = tl.load(ANCHOR + b).to(tl.int32)
    valid = valid & (candidate_pos != anchor)

    ed = tl.arange(0, EXACT_D)
    pq = tl.load(PQ + b * EXACT_D + ed).to(tl.float32)
    partner_key = tl.load(
        PK + (b * MAX_T + candidate_pos[:, None]) * EXACT_D + ed[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    exact_score = tl.sum(partner_key * pq[None, :], axis=1) * (EXACT_D ** -0.5)
    exact_score = tl.where(valid, exact_score, -float("inf"))

    for k in tl.static_range(0, K_OUT):
        best_score = tl.max(exact_score, axis=0)
        best_lane = tl.argmax(exact_score, axis=0, tie_break_left=True)
        best_pos = tl.sum(tl.where(candidate_lane == best_lane, candidate_pos, 0), axis=0)
        tl.store(OUT_INDEX + b * K_OUT + k, best_pos.to(tl.int32))
        tl.store(OUT_SCORE + b * K_OUT + k, best_score)
        exact_score = tl.where(candidate_lane == best_lane, -float("inf"), exact_score)


@torch.library.triton_op("relationlex::anchor_select", mutates_args=())
def anchor_select(
    anchor_query: Tensor,
    anchor_keys: Tensor,
    router_query: Tensor,
    block_mean: Tensor,
    block_std: Tensor,
    gamma: Tensor,
    bias: Tensor,
    position: Tensor,
    local_window: int,
    top_blocks: int,
    strict_valid: bool,
) -> tuple[Tensor, Tensor]:
    batch = anchor_query.size(0)
    out_index = torch.empty((batch,), device=anchor_query.device, dtype=torch.int32)
    out_score = torch.empty((batch,), device=anchor_query.device, dtype=torch.float32)
    torch.library.wrap_triton(_anchor_select_kernel)[(batch,)](
        anchor_query,
        anchor_keys,
        router_query,
        block_mean,
        block_std,
        gamma,
        bias,
        position,
        out_index,
        out_score,
        MAX_T=anchor_keys.size(1),
        MAX_NB=block_mean.size(1),
        EXACT_D=anchor_query.size(1),
        HEADS=block_mean.size(2),
        HEAD_D=block_mean.size(3),
        REL_BUCKETS=bias.size(1),
        BLOCK_SIZE=anchor_keys.size(1) // block_mean.size(1),
        LOCAL=local_window,
        TOP_BLOCKS=top_blocks,
        STRICT_VALID=strict_valid,
        num_warps=1,
    )
    return out_index, out_score


@torch.library.triton_op("relationlex::partner_topk", mutates_args=())
def partner_topk(
    partner_query: Tensor,
    partner_keys: Tensor,
    router_query: Tensor,
    block_mean: Tensor,
    block_std: Tensor,
    gamma: Tensor,
    bias: Tensor,
    position: Tensor,
    anchor_index: Tensor,
    local_window: int,
    top_blocks: int,
    k_out: int,
    strict_valid: bool,
) -> tuple[Tensor, Tensor]:
    batch = partner_query.size(0)
    out_index = torch.empty((batch, k_out), device=partner_query.device, dtype=torch.int32)
    out_score = torch.empty((batch, k_out), device=partner_query.device, dtype=torch.float32)
    capacity = triton.next_power_of_2(local_window + top_blocks * (partner_keys.size(1) // block_mean.size(1)))
    max_top = triton.next_power_of_2(max(1, top_blocks))
    torch.library.wrap_triton(_partner_topk_kernel)[(batch,)](
        partner_query,
        partner_keys,
        router_query,
        block_mean,
        block_std,
        gamma,
        bias,
        position,
        anchor_index,
        out_index,
        out_score,
        MAX_T=partner_keys.size(1),
        MAX_NB=block_mean.size(1),
        EXACT_D=partner_query.size(1),
        HEADS=block_mean.size(2),
        HEAD_D=block_mean.size(3),
        REL_BUCKETS=bias.size(1),
        BLOCK_SIZE=partner_keys.size(1) // block_mean.size(1),
        LOCAL=local_window,
        TOP_BLOCKS=top_blocks,
        MAX_TOP=max_top,
        CANDIDATE_CAPACITY=capacity,
        K_OUT=k_out,
        STRICT_VALID=strict_valid,
        num_warps=1,
    )
    return out_index, out_score
