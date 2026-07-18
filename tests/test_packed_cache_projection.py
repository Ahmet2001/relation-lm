from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from relation_lm.inference import pack_sparse_cache_projection_weights


@pytest.mark.parametrize("batch", [1, 8])
def test_packed_sparse_cache_projection_matches_separate_linears(batch: int) -> None:
    torch.manual_seed(20260822 + batch)
    memory = torch.randn(batch, 576)
    widths = (64, 64, 128, 128, 64, 64)
    weights = tuple(torch.randn(width, 576) * 0.02 for width in widths)
    projection = pack_sparse_cache_projection_weights(weights)
    expected = tuple(F.linear(memory, weight) for weight in weights)
    actual = projection.project(memory)
    assert projection.input_width == 576
    assert projection.output_width == sum(widths)
    assert len(actual) == len(expected)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.allclose(actual_tensor, expected_tensor, atol=2.0e-5, rtol=2.0e-5)
    as_dict = projection.project_dict(memory)
    assert tuple(as_dict) == projection.names


def test_packed_sparse_cache_projection_validates_shapes() -> None:
    with pytest.raises(ValueError):
        pack_sparse_cache_projection_weights(
            (torch.randn(4, 8), torch.randn(4, 9)),
            names=("a", "b"),
        )
