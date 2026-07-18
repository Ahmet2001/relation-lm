from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from relation_lm.kernels.relation_reduce import (
    factor_relation_first_layer,
    relation_hidden_cache_update,
    relation_hidden_cached,
    relation_norm_reduce,
)


def benchmark_call(function, warmup: int = 50, repeats: int = 400) -> float:
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
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260819 + args.batch)
    device = torch.device("cuda")
    batch = args.batch
    context, relation_dim, hidden_dim, output_dim, k_out = 512, 128, 2304, 576, 8
    position = torch.tensor(context - 1, device=device, dtype=torch.long)
    operand = torch.randn(batch, context, relation_dim, device=device)
    first_weight = torch.randn(hidden_dim, 4 * relation_dim, device=device) / relation_dim**0.5
    first_bias = torch.randn(hidden_dim, device=device) * 0.01
    factors = factor_relation_first_layer(first_weight, first_bias, relation_dim)
    expanded = F.linear(operand, factors.expanded_weight, factors.expanded_bias)
    anchor = torch.randint(0, context, (batch,), device=device, dtype=torch.int32)
    partner = torch.randint(0, context, (batch, k_out), device=device, dtype=torch.int32)
    second_weight = torch.randn(output_dim, hidden_dim, device=device) / hidden_dim**0.5
    second_bias = torch.randn(output_dim, device=device) * 0.01
    norm_weight = torch.ones(output_dim, device=device)
    norm_bias = torch.zeros(output_dim, device=device)
    scores = torch.randn(batch, k_out, device=device)

    def separate() -> torch.Tensor:
        hidden = relation_hidden_cached(
            expanded,
            operand,
            anchor,
            partner,
            factors.product_weight,
        )
        raw = F.linear(hidden, second_weight, second_bias)
        return relation_norm_reduce(raw, scores, norm_weight, norm_bias, position, 1.0e-5)

    mutable = expanded.clone()

    def fused_update() -> torch.Tensor:
        hidden = relation_hidden_cache_update(
            mutable,
            operand,
            anchor,
            partner,
            factors.anchor_weight,
            factors.partner_weight,
            factors.product_weight,
            factors.first_bias,
            position,
        )
        raw = F.linear(hidden, second_weight, second_bias)
        return relation_norm_reduce(raw, scores, norm_weight, norm_bias, position, 1.0e-5)

    separate_ms = benchmark_call(separate)
    fused_ms = benchmark_call(fused_update)
    print(
        json.dumps(
            {
                "batch": batch,
                "separate_update_ms": separate_ms,
                "fused_update_ms": fused_ms,
                "fused_over_separate": separate_ms / fused_ms,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
