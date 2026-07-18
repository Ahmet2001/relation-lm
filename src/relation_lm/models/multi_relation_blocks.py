from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class MultiRelationBlockConfig:
    d_model: int
    gate_dim: int
    relation_dim: int
    ff_mult: int = 4
    anchors_per_block: int = 2
    partners_per_anchor: int = 4
    dropout: float = 0.0


class MultiAnchorRelationBlock(nn.Module):
    """A causal relation layer with several anchor slots per query.

    Active anchor slots are distinct. Partner selection excludes every active
    anchor chosen by the block, so anchor and partner roles cannot overlap for
    the same query/layer. At early positions only ``min(H, t + 1)`` anchor
    slots are active because more distinct causal anchors do not yet exist.
    """

    def __init__(self, cfg: MultiRelationBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.anchors_per_block
        d = cfg.d_model
        g = cfg.gate_dim
        r = cfg.relation_dim
        hidden = cfg.ff_mult * d

        self.input_norm = nn.LayerNorm(d)
        self.anchor_query = nn.Linear(d, h * g, bias=False)
        self.anchor_key = nn.Linear(d, h * g, bias=False)
        self.partner_query = nn.ModuleList(
            nn.Linear(2 * d, g, bias=False) for _ in range(h)
        )
        self.partner_key = nn.Linear(d, h * g, bias=False)
        self.operand = nn.Linear(d, r, bias=False)
        self.relation_mlp = nn.Sequential(
            nn.Linear(4 * r, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
            nn.LayerNorm(d),
        )
        self.slot_mix_logits = nn.Parameter(torch.zeros(h))
        self.residual_gate = nn.Parameter(torch.tensor(-1.5))
        self.dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def _gather(source: Tensor, index: Tensor) -> Tensor:
        shape = (source.size(0),) + (1,) * (index.ndim - 1)
        batch = torch.arange(source.size(0), device=source.device).view(shape)
        return source[batch.expand_as(index), index]

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def _select_distinct_anchors(
        self, hidden: Tensor, memory: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = hidden.shape
        h = self.cfg.anchors_per_block
        g = self.cfg.gate_dim
        q = self.anchor_query(hidden).view(batch, length, h, g)
        k = self.anchor_key(memory).view(batch, length, h, g)
        score = torch.einsum("bthd,bshd->bhts", q, k) * (g**-0.5)
        score = score.masked_fill(
            self._causal_mask(length, hidden.device)[None, None], -torch.inf
        )

        time = torch.arange(length, device=hidden.device)
        slot = torch.arange(h, device=hidden.device)
        active = slot[None, :] <= time[:, None]  # [T,H]
        selected: list[Tensor] = []
        for slot_index in range(h):
            slot_score = score[:, slot_index]
            if selected:
                used = torch.stack(selected, dim=-1)
                used_mask = F.one_hot(used, num_classes=length).any(dim=-2)
                slot_score = slot_score.masked_fill(used_mask, -torch.inf)
            index = slot_score.argmax(dim=-1)
            slot_active = active[:, slot_index][None].expand(batch, -1)
            index = torch.where(slot_active, index, torch.zeros_like(index))
            selected.append(index)
        anchor_index = torch.stack(selected, dim=-1)
        anchor = self._gather(memory, anchor_index)
        anchor = anchor * active[None, :, :, None].to(anchor.dtype)
        return anchor_index, anchor, active[None].expand(batch, -1, -1)

    def forward(
        self, hidden: Tensor, memory: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch, length, _ = hidden.shape
        h = self.cfg.anchors_per_block
        k_count = self.cfg.partners_per_anchor
        g = self.cfg.gate_dim
        if length < k_count:
            raise ValueError("sequence length must be >= partners_per_anchor")

        normalized = self.input_norm(hidden)
        anchor_index, anchor, anchor_active = self._select_distinct_anchors(
            normalized, memory
        )

        query_expanded = normalized[:, :, None].expand(-1, -1, h, -1)
        partner_input = torch.cat((query_expanded, anchor), dim=-1)
        partner_query = torch.stack(
            [
                layer(partner_input[:, :, slot_index])
                for slot_index, layer in enumerate(self.partner_query)
            ],
            dim=2,
        )
        partner_key = self.partner_key(memory).view(batch, length, h, g)
        partner_score = torch.einsum(
            "bthd,bshd->bhts", partner_query, partner_key
        ) * (g**-0.5)
        partner_score = partner_score.masked_fill(
            self._causal_mask(length, hidden.device)[None, None], -torch.inf
        )

        active_anchor_one_hot = F.one_hot(
            anchor_index, num_classes=length
        ) & anchor_active.unsqueeze(-1)
        all_anchor_mask = active_anchor_one_hot.any(dim=-2)  # [B,T,S]
        partner_score = partner_score.masked_fill(
            all_anchor_mask[:, None].expand(-1, h, -1, -1), -torch.inf
        )
        partner_score = partner_score.masked_fill(
            ~anchor_active.permute(0, 2, 1).unsqueeze(-1), -torch.inf
        )

        top_score, partner_index = partner_score.topk(k_count, dim=-1)
        partner_index = partner_index.permute(0, 2, 1, 3)
        top_score = top_score.permute(0, 2, 1, 3)
        partner_active = torch.isfinite(top_score)
        partner = self._gather(memory, partner_index)

        anchor_operand = self.operand(anchor).unsqueeze(3).expand(
            -1, -1, -1, k_count, -1
        )
        partner_operand = self.operand(partner)
        feature = torch.cat(
            (
                anchor_operand,
                partner_operand,
                anchor_operand * partner_operand,
                anchor_operand - partner_operand,
            ),
            dim=-1,
        )
        relation = self.relation_mlp(feature)
        safe_score = top_score.masked_fill(~partner_active, -1.0e4)
        partner_weight = safe_score.softmax(dim=-1) * partner_active
        partner_weight = partner_weight / partner_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-9)
        slot_context = (partner_weight.unsqueeze(-1) * relation).sum(dim=3)
        slot_context = slot_context * anchor_active.unsqueeze(-1).to(slot_context.dtype)

        slot_weight = self.slot_mix_logits.softmax(dim=0)
        active_slot_weight = slot_weight[None, None, :] * anchor_active
        active_slot_weight = active_slot_weight / active_slot_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-9)
        context = (slot_context * active_slot_weight.unsqueeze(-1)).sum(dim=2)
        output = hidden + torch.sigmoid(self.residual_gate) * self.dropout(context)
        return output, {
            "anchor_index": anchor_index,
            "anchor_active": anchor_active,
            "partner_index": partner_index,
            "partner_active": partner_active,
            "partner_weight": partner_weight,
            "slot_weight": active_slot_weight,
            "relation_context": context,
        }


class RelationBlockStack(nn.Module):
    """Sequential relation depth analogous to stacking attention layers."""

    def __init__(self, cfg: MultiRelationBlockConfig, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            MultiAnchorRelationBlock(cfg) for _ in range(int(num_blocks))
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self, hidden: Tensor, memory: Tensor
    ) -> tuple[Tensor, list[dict[str, Tensor]]]:
        diagnostics = []
        for block in self.blocks:
            hidden, block_diagnostics = block(hidden, memory)
            diagnostics.append(block_diagnostics)
        return self.final_norm(hidden), diagnostics
