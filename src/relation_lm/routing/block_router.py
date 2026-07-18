"""Low-rank causal block router used by sparse Relation LM."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class BlockRouterConfig:
    d_model: int = 576
    block_size: int = 8
    heads: int = 4
    head_dim: int = 16
    relative_buckets: int = 16


class DistilledBlockRouter(nn.Module):
    def __init__(self, config: BlockRouterConfig) -> None:
        super().__init__()
        self.config = config
        routed_width = config.heads * config.head_dim
        self.anchor_q = nn.Linear(config.d_model, routed_width, bias=False)
        self.anchor_k = nn.Linear(config.d_model, routed_width, bias=False)
        self.partner_q = nn.Linear(2 * config.d_model, routed_width, bias=False)
        self.partner_k = nn.Linear(config.d_model, routed_width, bias=False)
        self.anchor_gamma = nn.Parameter(torch.zeros(config.heads))
        self.partner_gamma = nn.Parameter(torch.zeros(config.heads))
        self.anchor_bias = nn.Parameter(
            torch.zeros(config.heads, config.relative_buckets)
        )
        self.partner_bias = nn.Parameter(
            torch.zeros(config.heads, config.relative_buckets)
        )
        self.anchor_conv = nn.Conv1d(
            routed_width, routed_width, 3, groups=routed_width, bias=False
        )
        self.partner_conv = nn.Conv1d(
            routed_width, routed_width, 3, groups=routed_width, bias=False
        )
        with torch.no_grad():
            self.anchor_conv.weight.zero_()
            self.partner_conv.weight.zero_()
            self.anchor_conv.weight[:, 0, 2] = 1
            self.partner_conv.weight[:, 0, 2] = 1

    @staticmethod
    def _causal_conv(x: Tensor, convolution: nn.Conv1d) -> Tensor:
        return convolution(F.pad(x.transpose(1, 2), (2, 0))).transpose(1, 2)

    def block_statistics(self, key: Tensor) -> tuple[Tensor, Tensor, int]:
        batch, length, _ = key.shape
        block_size = self.config.block_size
        blocks = (length + block_size - 1) // block_size
        padding = blocks * block_size - length
        if padding:
            key = F.pad(key, (0, 0, 0, padding))
        grouped = key.view(
            batch,
            blocks,
            block_size,
            self.config.heads,
            self.config.head_dim,
        )
        return grouped.mean(2), grouped.std(2, unbiased=False), blocks

    def _scores(
        self,
        query: Tensor,
        key: Tensor,
        convolution: nn.Conv1d,
        gamma: Tensor,
        bias: Tensor,
    ) -> Tensor:
        batch, length, _ = query.shape
        query = query.view(
            batch, length, self.config.heads, self.config.head_dim
        )
        key = self._causal_conv(key, convolution)
        mean, std, blocks = self.block_statistics(key)
        content = torch.einsum("bthd,bnhd->bthn", query, mean)
        uncertainty = torch.einsum("bthd,bnhd->bthn", query.abs(), std)
        uncertainty = uncertainty * F.softplus(gamma)[None, None, :, None]

        position = torch.arange(length, device=query.device)[:, None]
        block_end = (
            torch.arange(blocks, device=query.device) + 1
        ) * self.config.block_size - 1
        distance = (position - block_end[None]).clamp_min(0)
        bucket = torch.floor(torch.log2(distance.float() + 1)).long()
        bucket = bucket.clamp_max(self.config.relative_buckets - 1)
        relative = bias[:, bucket].permute(1, 0, 2)[None]
        per_head = (
            (content + uncertainty) * (self.config.head_dim ** -0.5) + relative
        )
        scores = torch.logsumexp(per_head, dim=2) - math.log(self.config.heads)
        complete = block_end[None] <= torch.arange(length, device=query.device)[:, None]
        return scores.masked_fill(~complete[None], -torch.inf)

    def anchor_scores(self, query: Tensor, memory: Tensor) -> Tensor:
        return self._scores(
            self.anchor_q(query),
            self.anchor_k(memory),
            self.anchor_conv,
            self.anchor_gamma,
            self.anchor_bias,
        )

    def partner_scores(self, query: Tensor, anchor: Tensor, memory: Tensor) -> Tensor:
        return self._scores(
            self.partner_q(torch.cat((query, anchor), -1)),
            self.partner_k(memory),
            self.partner_conv,
            self.partner_gamma,
            self.partner_bias,
        )
