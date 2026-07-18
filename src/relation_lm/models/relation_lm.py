"""Dense reference implementation of Relation LM."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sinusoidal_positions(length: int, width: int) -> Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    table = torch.zeros(length, width, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * scale)
    table[:, 1::2] = torch.cos(position * scale[: table[:, 1::2].shape[1]])
    return table


@dataclass
class RelationLMConfig:
    vocab_size: int = 16_000
    boundary_vocab_size: int = 34
    d_model: int = 576
    ff_mult: int = 4
    memory_blocks: int = 8
    query_blocks: int = 2
    query_stem_blocks: int = 2
    query_lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    gate_dim: int = 64
    relation_dim: int = 128
    k_max: int = 8
    max_context: int = 512
    query_residual_init: float = 0.25
    relation_mix_logit_init: float = -1.5


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, multiplier: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, multiplier * width),
            nn.GELU(),
            nn.Linear(multiplier * width, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ff(self.norm(x))


class CausalMultiLagQueryBlock(nn.Module):
    def __init__(
        self, width: int, lags: tuple[int, ...], residual_init: float
    ) -> None:
        super().__init__()
        self.lags = tuple(int(value) for value in lags)
        self.norm = nn.LayerNorm(width)
        prior = torch.tensor([-math.log2(float(lag) + 1.0) for lag in self.lags])
        self.lag_logits = nn.Parameter(prior[:, None].expand(-1, width).clone())
        self.gate_weight = nn.Parameter(torch.ones(width))
        self.gate_bias = nn.Parameter(torch.zeros(width))
        self.residual_scale = nn.Parameter(torch.full((width,), float(residual_init)))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        weights = self.lag_logits.softmax(dim=0).to(h.dtype)
        context = torch.zeros_like(h)
        for index, lag in enumerate(self.lags):
            if lag < h.size(1):
                context = context + F.pad(h[:, :-lag], (0, 0, lag, 0)) * weights[index][None, None]
        gate = torch.sigmoid(h * self.gate_weight + self.gate_bias)
        return x + self.residual_scale * context * gate


class RelationLexLM(nn.Module):
    """Attention-free dense Relation LM reference model."""

    def __init__(self, config: RelationLMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.token_embedding = nn.Embedding(config.vocab_size, d)
        self.boundary_embedding = nn.Embedding(config.boundary_vocab_size, d)
        self.input_scale = math.sqrt(d / 2.0)
        self.register_buffer(
            "position_embedding",
            sinusoidal_positions(config.max_context, d),
            persistent=False,
        )
        self.memory_blocks = nn.ModuleList(
            [ResidualMLPBlock(d, config.ff_mult) for _ in range(config.memory_blocks)]
        )
        self.query_stem = nn.ModuleList(
            [
                CausalMultiLagQueryBlock(d, config.query_lags, config.query_residual_init)
                for _ in range(config.query_stem_blocks)
            ]
        )
        self.query_blocks = nn.ModuleList(
            [ResidualMLPBlock(d, config.ff_mult) for _ in range(config.query_blocks)]
        )
        self.memory_norm = nn.LayerNorm(d)
        self.query_norm = nn.LayerNorm(d)
        self.anchor_query = nn.Linear(d, config.gate_dim, bias=False)
        self.anchor_key = nn.Linear(d, config.gate_dim, bias=False)
        self.partner_query = nn.Linear(2 * d, config.gate_dim, bias=False)
        self.partner_key = nn.Linear(d, config.gate_dim, bias=False)
        self.operand_projection = nn.Linear(d, config.relation_dim, bias=False)
        feature_width = 4 * config.relation_dim
        self.relation_mlp = nn.Sequential(
            nn.Linear(feature_width, config.ff_mult * d),
            nn.GELU(),
            nn.Linear(config.ff_mult * d, d),
            nn.LayerNorm(d),
        )
        self.relation_mix_logit = nn.Parameter(
            torch.tensor(float(config.relation_mix_logit_init))
        )
        self.output_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.boundary_output = nn.Linear(d, config.boundary_vocab_size)

    def streams(self, tokens: Tensor, boundaries: Tensor) -> tuple[Tensor, Tensor]:
        length = tokens.size(1)
        if length > self.config.max_context:
            raise ValueError("context exceeds max_context")
        x = (self.token_embedding(tokens) + self.boundary_embedding(boundaries)) * self.input_scale
        x = x + self.position_embedding[:length].to(x.device)
        memory = x
        for block in self.memory_blocks:
            memory = block(memory)
        query = x
        for block in self.query_stem:
            query = block(query)
        for block in self.query_blocks:
            query = block(query)
        return self.query_norm(query), self.memory_norm(memory)

    @staticmethod
    def _causal_scores(query: Tensor, key: Tensor) -> Tensor:
        length = query.size(1)
        score = torch.einsum("btd,bsd->bts", query, key) * (query.size(-1) ** -0.5)
        future = torch.triu(
            torch.ones(length, length, device=query.device, dtype=torch.bool), diagonal=1
        )
        return score.masked_fill(future[None], -torch.inf)

    @staticmethod
    def _gather(source: Tensor, indices: Tensor) -> Tensor:
        batch = torch.arange(source.size(0), device=source.device)[:, None, None]
        return source[batch, indices]

    def relation_context(self, query: Tensor, memory: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size, length, _ = memory.shape
        anchor_scores = self._causal_scores(self.anchor_query(query), self.anchor_key(memory))
        anchor_idx = anchor_scores.argmax(-1)
        batch = torch.arange(batch_size, device=memory.device)[:, None]
        anchor = memory[batch, anchor_idx]

        partner_query = self.partner_query(torch.cat((query, anchor), -1))
        partner_scores = self._causal_scores(partner_query, self.partner_key(memory))
        partner_scores = partner_scores.scatter(2, anchor_idx.unsqueeze(-1), -torch.inf)
        k = min(self.config.k_max, max(1, length - 1))
        top_scores, partner_idx = partner_scores.topk(k, dim=-1)
        partner = self._gather(memory, partner_idx)
        finite = torch.isfinite(top_scores)

        position = torch.arange(length, device=memory.device)
        k_limit = torch.ceil(torch.log2(position.float() + 2)).long().clamp(1, k)
        ranks = torch.arange(k, device=memory.device)[None, None]
        active = finite & ranks.lt(k_limit[None, :, None])
        safe = top_scores.masked_fill(~active, -1e4)
        weights = safe.softmax(-1) * active
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

        anchor_operand = self.operand_projection(anchor).unsqueeze(2).expand(-1, -1, k, -1)
        partner_operand = self.operand_projection(partner)
        features = torch.cat(
            (
                anchor_operand,
                partner_operand,
                anchor_operand * partner_operand,
                anchor_operand - partner_operand,
            ),
            -1,
        )
        relations = self.relation_mlp(features)
        context = (weights.unsqueeze(-1) * relations).sum(2)
        return context, {
            "anchor_positions": anchor_idx,
            "partner_positions": partner_idx,
            "partner_weights": weights,
            "active_partner_mask": active,
        }

    def forward(
        self, tokens: Tensor, boundaries: Tensor, *, return_diagnostics: bool = False
    ):
        query, memory = self.streams(tokens, boundaries)
        context, diagnostics = self.relation_context(query, memory)
        final = self.output_norm(
            query + torch.sigmoid(self.relation_mix_logit) * context
        )
        outputs = self.output(final), self.boundary_output(final)
        return (outputs, diagnostics) if return_diagnostics else outputs
