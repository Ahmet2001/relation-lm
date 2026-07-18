from __future__ import annotations

import pytest
import torch
from torch import nn

pytest.importorskip("triton")

from relation_lm.kernels.router_cache import router_cache_update

pytestmark = pytest.mark.gpu


def _make_case(batch: int, position_value: int):
    device = torch.device("cuda")
    max_context = 512
    router_dim = 64
    block_size = 8
    heads = 4
    head_dim = 16
    blocks = max_context // block_size
    torch.manual_seed(20260824 + batch + position_value)

    packed = torch.randn(batch, 512, device=device)
    anchor_raw = packed[:, 384:448]
    partner_raw = packed[:, 448:512]
    assert anchor_raw.stride(0) == 512
    assert partner_raw.stride(0) == 512

    anchor_raw_cache = torch.zeros(batch, max_context, router_dim, device=device)
    partner_raw_cache = torch.zeros_like(anchor_raw_cache)
    anchor_raw_cache[:, :position_value].normal_()
    partner_raw_cache[:, :position_value].normal_()
    anchor_conv_cache = torch.zeros_like(anchor_raw_cache)
    partner_conv_cache = torch.zeros_like(partner_raw_cache)
    anchor_weight = torch.randn(router_dim, 3, device=device) * 0.1
    partner_weight = torch.randn(router_dim, 3, device=device) * 0.1

    def current_conv(raw, cache, weight):
        result = raw * weight[:, 2]
        if position_value >= 1:
            result = result + cache[:, position_value - 1] * weight[:, 1]
        if position_value >= 2:
            result = result + cache[:, position_value - 2] * weight[:, 0]
        return result

    for index in range(position_value):
        anchor_conv_cache[:, index] = current_conv(
            anchor_raw_cache[:, index], anchor_raw_cache, anchor_weight
        ) if index == position_value else (
            anchor_raw_cache[:, index] * anchor_weight[:, 2]
            + (anchor_raw_cache[:, index - 1] * anchor_weight[:, 1] if index >= 1 else 0)
            + (anchor_raw_cache[:, index - 2] * anchor_weight[:, 0] if index >= 2 else 0)
        )
        partner_conv_cache[:, index] = (
            partner_raw_cache[:, index] * partner_weight[:, 2]
            + (partner_raw_cache[:, index - 1] * partner_weight[:, 1] if index >= 1 else 0)
            + (partner_raw_cache[:, index - 2] * partner_weight[:, 0] if index >= 2 else 0)
        )

    shape = (batch, blocks, heads, head_dim)
    anchor_mean = torch.zeros(shape, device=device)
    anchor_std = torch.zeros(shape, device=device)
    partner_mean = torch.zeros(shape, device=device)
    partner_std = torch.zeros(shape, device=device)
    position = torch.tensor(position_value, device=device, dtype=torch.long)
    return {
        "anchor_raw": anchor_raw,
        "partner_raw": partner_raw,
        "anchor_raw_cache": anchor_raw_cache,
        "partner_raw_cache": partner_raw_cache,
        "anchor_conv_cache": anchor_conv_cache,
        "partner_conv_cache": partner_conv_cache,
        "anchor_mean": anchor_mean,
        "anchor_std": anchor_std,
        "partner_mean": partner_mean,
        "partner_std": partner_std,
        "anchor_weight": anchor_weight,
        "partner_weight": partner_weight,
        "position": position,
        "block_size": block_size,
    }


def _reference(case):
    position = int(case["position"].item())
    block_size = int(case["block_size"])
    anchor_raw_cache = case["anchor_raw_cache"].clone()
    partner_raw_cache = case["partner_raw_cache"].clone()
    anchor_conv_cache = case["anchor_conv_cache"].clone()
    partner_conv_cache = case["partner_conv_cache"].clone()
    anchor_mean = case["anchor_mean"].clone()
    anchor_std = case["anchor_std"].clone()
    partner_mean = case["partner_mean"].clone()
    partner_std = case["partner_std"].clone()

    def conv(raw, cache, weight):
        result = raw * weight[:, 2]
        if position >= 1:
            result = result + cache[:, position - 1] * weight[:, 1]
        if position >= 2:
            result = result + cache[:, position - 2] * weight[:, 0]
        return result

    anchor_conv = conv(
        case["anchor_raw"], anchor_raw_cache, case["anchor_weight"]
    )
    partner_conv = conv(
        case["partner_raw"], partner_raw_cache, case["partner_weight"]
    )
    anchor_raw_cache[:, position] = case["anchor_raw"]
    partner_raw_cache[:, position] = case["partner_raw"]
    anchor_conv_cache[:, position] = anchor_conv
    partner_conv_cache[:, position] = partner_conv
    block_index = position // block_size
    block_start = block_index * block_size
    block_end = block_start + block_size
    anchor_block = anchor_conv_cache[:, block_start:block_end].reshape(
        anchor_conv_cache.size(0), block_size, 4, 16
    )
    partner_block = partner_conv_cache[:, block_start:block_end].reshape(
        partner_conv_cache.size(0), block_size, 4, 16
    )
    anchor_mean[:, block_index] = anchor_block.mean(1)
    anchor_std[:, block_index] = anchor_block.std(1, unbiased=False)
    partner_mean[:, block_index] = partner_block.mean(1)
    partner_std[:, block_index] = partner_block.std(1, unbiased=False)
    return {
        "output": torch.stack((anchor_conv, partner_conv), dim=1),
        "anchor_raw_cache": anchor_raw_cache,
        "partner_raw_cache": partner_raw_cache,
        "anchor_conv_cache": anchor_conv_cache,
        "partner_conv_cache": partner_conv_cache,
        "anchor_mean": anchor_mean,
        "anchor_std": anchor_std,
        "partner_mean": partner_mean,
        "partner_std": partner_std,
    }


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("position_value", [95, 96, 447, 448])
def test_router_cache_update_state(batch: int, position_value: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    case = _make_case(batch, position_value)
    expected = _reference(case)
    output = router_cache_update(
        case["anchor_raw"],
        case["partner_raw"],
        case["anchor_raw_cache"],
        case["partner_raw_cache"],
        case["anchor_conv_cache"],
        case["partner_conv_cache"],
        case["anchor_mean"],
        case["anchor_std"],
        case["partner_mean"],
        case["partner_std"],
        case["anchor_weight"],
        case["partner_weight"],
        case["position"],
        case["block_size"],
    )
    torch.cuda.synchronize()
    assert torch.allclose(output, expected["output"], atol=1.0e-6, rtol=1.0e-6)
    for name in (
        "anchor_raw_cache",
        "partner_raw_cache",
        "anchor_conv_cache",
        "partner_conv_cache",
        "anchor_mean",
        "anchor_std",
        "partner_mean",
        "partner_std",
    ):
        assert torch.allclose(case[name], expected[name], atol=1.0e-6, rtol=1.0e-6)


class _RouterUpdateModule(nn.Module):
    def __init__(self, case) -> None:
        super().__init__()
        for name in (
            "anchor_raw_cache",
            "partner_raw_cache",
            "anchor_conv_cache",
            "partner_conv_cache",
            "anchor_mean",
            "anchor_std",
            "partner_mean",
            "partner_std",
            "anchor_weight",
            "partner_weight",
        ):
            self.register_buffer(name, case[name].clone())
        self.block_size = int(case["block_size"])

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


@pytest.mark.parametrize("batch", [1, 8])
def test_router_cache_update_fullgraph(batch: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    case = _make_case(batch, 447)
    eager = _RouterUpdateModule(case).cuda().eval()
    compiled_module = _RouterUpdateModule(case).cuda().eval()
    compiled = torch.compile(compiled_module, fullgraph=True, mode="reduce-overhead")
    eager_output = eager(case["anchor_raw"], case["partner_raw"], case["position"])
    torch.compiler.cudagraph_mark_step_begin()
    compiled_output = compiled(
        case["anchor_raw"], case["partner_raw"], case["position"]
    ).clone()
    torch.cuda.synchronize()
    assert torch.allclose(eager_output, compiled_output, atol=1.0e-6, rtol=1.0e-6)
    for name, eager_buffer in eager.named_buffers():
        compiled_buffer = dict(compiled_module.named_buffers())[name]
        assert torch.allclose(eager_buffer, compiled_buffer, atol=1.0e-6, rtol=1.0e-6)
