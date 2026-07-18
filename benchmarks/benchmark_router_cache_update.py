from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from relation_lm.kernels.router_cache import router_cache_update


class ReferenceRouterUpdate(nn.Module):
    def __init__(self, batch: int, max_context: int = 512) -> None:
        super().__init__()
        router_dim = 64
        blocks = max_context // 8
        self.batch = batch
        self.max_context = max_context
        self.block_size = 8
        self.register_buffer(
            "anchor_raw_cache", torch.zeros(batch, max_context, router_dim)
        )
        self.register_buffer(
            "partner_raw_cache", torch.zeros(batch, max_context, router_dim)
        )
        self.register_buffer(
            "anchor_conv_cache", torch.zeros(batch, max_context, router_dim)
        )
        self.register_buffer(
            "partner_conv_cache", torch.zeros(batch, max_context, router_dim)
        )
        shape = (batch, blocks, 4, 16)
        self.register_buffer("anchor_mean", torch.zeros(shape))
        self.register_buffer("anchor_std", torch.zeros(shape))
        self.register_buffer("partner_mean", torch.zeros(shape))
        self.register_buffer("partner_std", torch.zeros(shape))
        self.register_buffer("anchor_weight", torch.randn(router_dim, 3) * 0.1)
        self.register_buffer("partner_weight", torch.randn(router_dim, 3) * 0.1)
        self.register_buffer("block_offsets", torch.arange(self.block_size))

    @staticmethod
    def _write(buffer, position, value):
        buffer.index_copy_(1, position.reshape(1), value.unsqueeze(1))

    def _conv(self, raw, cache, weight, position):
        previous_1 = cache.index_select(
            1, (position - 1).clamp(0, self.max_context - 1).reshape(1)
        ).squeeze(1)
        previous_2 = cache.index_select(
            1, (position - 2).clamp(0, self.max_context - 1).reshape(1)
        ).squeeze(1)
        return (
            raw * weight[:, 2]
            + previous_1 * weight[:, 1] * position.ge(1)
            + previous_2 * weight[:, 0] * position.ge(2)
        )

    def forward(self, anchor_raw, partner_raw, position):
        anchor_conv = self._conv(
            anchor_raw, self.anchor_raw_cache, self.anchor_weight, position
        )
        partner_conv = self._conv(
            partner_raw, self.partner_raw_cache, self.partner_weight, position
        )
        self._write(self.anchor_raw_cache, position, anchor_raw)
        self._write(self.partner_raw_cache, position, partner_raw)
        self._write(self.anchor_conv_cache, position, anchor_conv)
        self._write(self.partner_conv_cache, position, partner_conv)
        block_index = torch.div(position, self.block_size, rounding_mode="floor")
        block_start = block_index * self.block_size
        indices = block_start + self.block_offsets
        anchor_block = self.anchor_conv_cache.index_select(1, indices).reshape(
            self.batch, self.block_size, 4, 16
        )
        partner_block = self.partner_conv_cache.index_select(1, indices).reshape(
            self.batch, self.block_size, 4, 16
        )
        self.anchor_mean.index_copy_(
            1, block_index.reshape(1), anchor_block.mean(1).unsqueeze(1)
        )
        self.anchor_std.index_copy_(
            1,
            block_index.reshape(1),
            anchor_block.std(1, unbiased=False).unsqueeze(1),
        )
        self.partner_mean.index_copy_(
            1, block_index.reshape(1), partner_block.mean(1).unsqueeze(1)
        )
        self.partner_std.index_copy_(
            1,
            block_index.reshape(1),
            partner_block.std(1, unbiased=False).unsqueeze(1),
        )
        return torch.stack((anchor_conv, partner_conv), dim=1)


class FusedRouterUpdate(ReferenceRouterUpdate):
    def forward(self, anchor_raw, partner_raw, position):
        return router_cache_update(
            anchor_raw,
            partner_raw,
            self.anchor_raw_cache,
            self.partner_raw_cache,
            self.anchor_conv_cache,
            self.partner_conv_cache,
            self.anchor_mean,
            self.anchor_std,
            self.partner_mean,
            self.partner_std,
            self.anchor_weight,
            self.partner_weight,
            position,
            self.block_size,
        )


def benchmark_call(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        torch.compiler.cudagraph_mark_step_begin()
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        torch.compiler.cudagraph_mark_step_begin()
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/router_cache_update_microbenchmark.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20260825)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    rows = []
    for batch in (1, 8):
        reference = ReferenceRouterUpdate(batch).to(device).eval()
        fused = FusedRouterUpdate(batch).to(device).eval()
        fused.load_state_dict(reference.state_dict())
        packed = torch.randn(batch, 512, device=device)
        anchor_raw = packed[:, 384:448]
        partner_raw = packed[:, 448:512]
        position = torch.tensor(447, device=device, dtype=torch.long)
        compiled_reference = torch.compile(
            reference, fullgraph=True, mode="reduce-overhead"
        )
        compiled_fused = torch.compile(fused, fullgraph=True, mode="reduce-overhead")
        torch.compiler.cudagraph_mark_step_begin()
        reference_output = compiled_reference(anchor_raw, partner_raw, position).clone()
        torch.compiler.cudagraph_mark_step_begin()
        fused_output = compiled_fused(anchor_raw, partner_raw, position).clone()
        torch.cuda.synchronize()
        maximum_difference = float((reference_output - fused_output).abs().max())
        reference_ms = benchmark_call(
            lambda compiled_reference=compiled_reference, anchor_raw=anchor_raw, partner_raw=partner_raw, position=position: compiled_reference(anchor_raw, partner_raw, position),
            args.warmup,
            args.repeats,
        )
        fused_ms = benchmark_call(
            lambda compiled_fused=compiled_fused, anchor_raw=anchor_raw, partner_raw=partner_raw, position=position: compiled_fused(anchor_raw, partner_raw, position),
            args.warmup,
            args.repeats,
        )
        row = {
            "batch": batch,
            "reference_ms": reference_ms,
            "fused_ms": fused_ms,
            "speedup": reference_ms / fused_ms,
            "max_difference": maximum_difference,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    payload = {
        "protocol": "compiled fullgraph state update; current position 447; non-contiguous packed router-key views",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
