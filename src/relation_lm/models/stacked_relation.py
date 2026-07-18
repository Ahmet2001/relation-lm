from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class RelationBlockConfig:
    """Configuration for one anchor-plus-multiple-partners relation block."""

    d_model: int
    gate_dim: int = 64
    relation_dim: int = 128
    hidden_dim: int = 768
    k_max: int = 8
    mix_logit_init: float = -1.5


class RelationBlock(nn.Module):
    """Causal relation layer with one anchor and multiple partners per query.

    ``exclude_anchor_history_from_partners`` excludes every position selected
    as an anchor at the current or an earlier query position. Future anchor
    decisions are never used, so the rule remains causal.
    """

    def __init__(self, cfg: RelationBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        self.input_norm = nn.LayerNorm(d_model)
        self.anchor_query = nn.Linear(d_model, cfg.gate_dim, bias=False)
        self.anchor_key = nn.Linear(d_model, cfg.gate_dim, bias=False)
        self.partner_query = nn.Linear(2 * d_model, cfg.gate_dim, bias=False)
        self.partner_key = nn.Linear(d_model, cfg.gate_dim, bias=False)
        self.operand_projection = nn.Linear(d_model, cfg.relation_dim, bias=False)
        self.relation_mlp = nn.Sequential(
            nn.Linear(4 * cfg.relation_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.mix_logit = nn.Parameter(torch.tensor(float(cfg.mix_logit_init)))

    @staticmethod
    def _gather(source: Tensor, indices: Tensor) -> Tensor:
        batch = torch.arange(source.size(0), device=source.device)[:, None, None]
        return source[batch, indices]

    @staticmethod
    def _causal_scores(query: Tensor, key: Tensor) -> Tensor:
        length = query.size(1)
        scores = torch.einsum("btd,bsd->bts", query, key)
        scores = scores * (query.size(-1) ** -0.5)
        future = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=query.device),
            diagonal=1,
        )
        return scores.masked_fill(future[None], -torch.inf)

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        *,
        exclude_anchor_history_from_partners: bool = False,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        normalized = self.input_norm(hidden)
        batch_size, length, _ = normalized.shape

        anchor_scores = self._causal_scores(
            self.anchor_query(normalized), self.anchor_key(memory)
        )
        anchor_index = anchor_scores.argmax(-1)
        anchor = memory[
            torch.arange(batch_size, device=hidden.device)[:, None], anchor_index
        ]

        partner_scores = self._causal_scores(
            self.partner_query(torch.cat((normalized, anchor), dim=-1)),
            self.partner_key(memory),
        )
        partner_scores = partner_scores.scatter(
            2, anchor_index.unsqueeze(-1), -torch.inf
        )

        anchor_history = None
        if exclude_anchor_history_from_partners:
            anchor_history = F.one_hot(anchor_index, num_classes=length).cumsum(1).bool()
            partner_scores = partner_scores.masked_fill(anchor_history, -torch.inf)

        partner_count = min(self.cfg.k_max, length)
        top_scores, partner_index = partner_scores.topk(partner_count, dim=-1)
        finite = torch.isfinite(top_scores)
        positions = torch.arange(length, device=hidden.device)
        active_limit = torch.ceil(torch.log2(positions.float() + 2.0)).long()
        active_limit = active_limit.clamp(1, partner_count)
        ranks = torch.arange(partner_count, device=hidden.device)[None, None]
        active = finite & (ranks < active_limit[None, :, None])
        safe_scores = top_scores.masked_fill(~active, -1.0e4)
        weights = safe_scores.softmax(-1) * active
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0e-9)

        anchor_operand = self.operand_projection(anchor).unsqueeze(2)
        anchor_operand = anchor_operand.expand(-1, -1, partner_count, -1)
        partner = self._gather(memory, partner_index)
        partner_operand = self.operand_projection(partner)
        features = torch.cat(
            (
                anchor_operand,
                partner_operand,
                anchor_operand * partner_operand,
                anchor_operand - partner_operand,
            ),
            dim=-1,
        )
        relation_vectors = self.relation_mlp(features)
        context = (weights.unsqueeze(-1) * relation_vectors).sum(2)
        output = hidden + torch.sigmoid(self.mix_logit) * context

        if not return_diagnostics:
            return output
        return output, {
            "anchor_index": anchor_index,
            "partner_index": partner_index,
            "partner_scores": top_scores,
            "partner_weights": weights,
            "active_partner_mask": active,
            "relation_vectors": relation_vectors,
            "anchor_history_mask": anchor_history,
        }


class StackedRelationLayers(nn.Module):
    """Attention-block analogue built from repeated ``RelationBlock`` layers."""

    def __init__(self, block_configs: list[RelationBlockConfig]) -> None:
        super().__init__()
        if not block_configs:
            raise ValueError("at least one relation block is required")
        d_model = block_configs[0].d_model
        if any(config.d_model != d_model for config in block_configs):
            raise ValueError("all relation blocks must use the same d_model")
        self.blocks = nn.ModuleList(RelationBlock(config) for config in block_configs)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        *,
        exclude_anchor_history_from_partners: bool = False,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, list[dict[str, Any]]]:
        diagnostics: list[dict[str, Any]] = []
        current = hidden
        for block in self.blocks:
            if return_diagnostics:
                current, information = block(
                    current,
                    memory,
                    exclude_anchor_history_from_partners=(
                        exclude_anchor_history_from_partners
                    ),
                    return_diagnostics=True,
                )
                diagnostics.append(information)
            else:
                current = block(
                    current,
                    memory,
                    exclude_anchor_history_from_partners=(
                        exclude_anchor_history_from_partners
                    ),
                )
        current = self.final_norm(current)
        return (current, diagnostics) if return_diagnostics else current
