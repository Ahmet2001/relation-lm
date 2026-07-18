from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

from relation_lm.inference import pack_sparse_cache_projection_weights


def benchmark_call(function, repeat: int) -> float:
    for _ in range(40):
        function()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(9):
        start.record()
        for _ in range(repeat):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeat)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=500)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20260822)
    device = torch.device("cuda")
    widths = (64, 64, 128, 128, 64, 64)
    rows = []
    for batch in (1, 8):
        memory = torch.randn(batch, 576, device=device)
        weights = tuple(
            torch.randn(width, 576, device=device) * 0.02 for width in widths
        )
        projection = pack_sparse_cache_projection_weights(weights)

        def separate(
            memory=memory,
            weights=weights,
        ):
            return tuple(F.linear(memory, weight) for weight in weights)

        def packed(
            memory=memory,
            projection=projection,
        ):
            return projection.project(memory)

        expected = separate()
        actual = packed()
        maximum_difference = max(
            float((left - right).abs().max())
            for left, right in zip(expected, actual, strict=True)
        )
        separate_ms = benchmark_call(separate, args.repeat)
        packed_ms = benchmark_call(packed, args.repeat)
        row = {
            "batch": batch,
            "separate_ms": separate_ms,
            "packed_ms": packed_ms,
            "speedup": separate_ms / packed_ms,
            "max_difference": maximum_difference,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
