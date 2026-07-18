#!/usr/bin/env python3
"""Microbenchmark packed relation_select against the two-operator path."""
from __future__ import annotations

import argparse
import json

import torch

from relation_lm.kernels.relation_select import relation_select_packed
from relation_lm.kernels.triton_select import anchor_select, partner_topk


def make_inputs(batch: int, context: int, device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    blocks = context // 8
    return {
        "anchor_query": torch.randn(batch, 64, device=device),
        "anchor_keys": torch.randn(batch, context, 64, device=device),
        "anchor_router_query": torch.randn(batch, 64, device=device),
        "anchor_mean": torch.randn(batch, blocks, 4, 16, device=device),
        "anchor_std": torch.rand(batch, blocks, 4, 16, device=device),
        "anchor_gamma": torch.randn(4, device=device),
        "anchor_bias": torch.randn(4, 16, device=device),
        "partner_base": torch.randn(batch, 128, device=device),
        "partner_anchor_cache": torch.randn(batch, context, 128, device=device),
        "partner_keys": torch.randn(batch, context, 64, device=device),
        "partner_mean": torch.randn(batch, blocks, 4, 16, device=device),
        "partner_std": torch.rand(batch, blocks, 4, 16, device=device),
        "partner_gamma": torch.randn(4, device=device),
        "partner_bias": torch.randn(4, 16, device=device),
        "position": torch.tensor(context - 1, device=device, dtype=torch.long),
    }


def packed(inputs: dict[str, torch.Tensor]):
    return relation_select_packed(
        inputs["anchor_query"],
        inputs["anchor_keys"],
        inputs["anchor_router_query"],
        inputs["anchor_mean"],
        inputs["anchor_std"],
        inputs["anchor_gamma"],
        inputs["anchor_bias"],
        inputs["partner_base"],
        inputs["partner_anchor_cache"],
        inputs["partner_keys"],
        inputs["partner_mean"],
        inputs["partner_std"],
        inputs["partner_gamma"],
        inputs["partner_bias"],
        inputs["position"],
        16,
        2,
        8,
        False,
    )


def two_kernel(inputs: dict[str, torch.Tensor]):
    anchor, anchor_score = anchor_select(
        inputs["anchor_query"],
        inputs["anchor_keys"],
        inputs["anchor_router_query"],
        inputs["anchor_mean"],
        inputs["anchor_std"],
        inputs["anchor_gamma"],
        inputs["anchor_bias"],
        inputs["position"],
        16,
        2,
        False,
    )
    batch = torch.arange(anchor.size(0), device=anchor.device)
    anchor_contribution = inputs["partner_anchor_cache"][batch, anchor.long()]
    exact_query = inputs["partner_base"][:, :64] + anchor_contribution[:, :64]
    router_query = inputs["partner_base"][:, 64:] + anchor_contribution[:, 64:]
    partner, partner_score = partner_topk(
        exact_query.contiguous(),
        inputs["partner_keys"],
        router_query.contiguous(),
        inputs["partner_mean"],
        inputs["partner_std"],
        inputs["partner_gamma"],
        inputs["partner_bias"],
        inputs["position"],
        anchor,
        16,
        2,
        8,
        False,
    )
    return anchor, anchor_score, partner, partner_score


def benchmark(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=500)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    inputs = make_inputs(args.batch, args.context, torch.device('cuda'))
    reference = two_kernel(inputs)
    fused = packed(inputs)
    index_equal = torch.equal(reference[0], fused[0]) and torch.equal(reference[2], fused[2])
    score_difference = max(
        float((reference[1] - fused[1]).abs().max()),
        float((reference[3] - fused[3]).abs().max()),
    )
    two_ms = benchmark(lambda: two_kernel(inputs), args.warmup, args.repeats)
    packed_ms = benchmark(lambda: packed(inputs), args.warmup, args.repeats)
    print(json.dumps({
        'batch': args.batch,
        'context': args.context,
        'index_equal': index_equal,
        'max_score_difference': score_difference,
        'two_kernel_ms': two_ms,
        'packed_ms': packed_ms,
        'packed_speedup': two_ms / packed_ms,
    }, indent=2))


if __name__ == '__main__':
    main()
