from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

SPARSE_CACHE_NAMES = (
    "anchor_key",
    "partner_key",
    "operand",
    "partner_anchor",
    "anchor_router_key",
    "partner_router_key",
)


@dataclass(frozen=True)
class PackedSparseCacheProjection:
    """One linear projection that produces all sparse-decode cache features.

    Relation LM stateful decode needs six token-local projections from the same
    memory vector. Concatenating their weights replaces six small GEMMs with one
    larger GEMM while preserving every output exactly up to floating-point
    accumulation order.
    """

    weight: Tensor
    split_sizes: tuple[int, ...]
    names: tuple[str, ...] = SPARSE_CACHE_NAMES

    def __post_init__(self) -> None:
        if self.weight.ndim != 2:
            raise ValueError("weight must be a 2D tensor")
        if len(self.split_sizes) != len(self.names):
            raise ValueError("split_sizes and names must have the same length")
        if any(size <= 0 for size in self.split_sizes):
            raise ValueError("split sizes must be positive")
        if sum(self.split_sizes) != self.weight.size(0):
            raise ValueError("split sizes must sum to the packed output width")

    @property
    def input_width(self) -> int:
        return int(self.weight.size(1))

    @property
    def output_width(self) -> int:
        return int(self.weight.size(0))

    def project(self, memory: Tensor) -> tuple[Tensor, ...]:
        """Project ``memory`` and return tensors in ``names`` order."""
        if memory.size(-1) != self.input_width:
            raise ValueError("memory width does not match packed projection input")
        packed = F.linear(memory, self.weight)
        return packed.split(self.split_sizes, dim=-1)

    def project_dict(self, memory: Tensor) -> dict[str, Tensor]:
        """Project ``memory`` and return a name-to-tensor mapping."""
        return dict(zip(self.names, self.project(memory), strict=True))


def pack_sparse_cache_projection_weights(
    weights: Sequence[Tensor],
    *,
    names: Sequence[str] = SPARSE_CACHE_NAMES,
) -> PackedSparseCacheProjection:
    """Concatenate token-local sparse cache projection weights.

    The verified Relation LM order is anchor key, partner key, operand,
    partner-anchor factor, anchor-router key, and partner-router key.
    """
    tensors = tuple(weights)
    output_names = tuple(names)
    if not tensors:
        raise ValueError("at least one weight is required")
    if len(tensors) != len(output_names):
        raise ValueError("weights and names must have the same length")
    if any(weight.ndim != 2 for weight in tensors):
        raise ValueError("every weight must be a 2D tensor")
    input_width = tensors[0].size(1)
    if any(weight.size(1) != input_width for weight in tensors):
        raise ValueError("all weights must have the same input width")
    device = tensors[0].device
    dtype = tensors[0].dtype
    if any(weight.device != device or weight.dtype != dtype for weight in tensors):
        raise ValueError("all weights must share device and dtype")
    packed = torch.cat(tensors, dim=0).contiguous()
    return PackedSparseCacheProjection(
        weight=packed,
        split_sizes=tuple(int(weight.size(0)) for weight in tensors),
        names=output_names,
    )
