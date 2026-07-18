"""Distance-aware multi-anchor Relation LM blocks.

Each block selects four distinct anchors. Every anchor selects four partners
under a block-specific near/middle/far quota. All selected anchors are removed
from every partner pool. The module also exposes differentiable cross-anchor
partner-overlap and distance-mass losses for short adaptation experiments.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _safe_normalize(weights: Tensor, valid: Tensor, dim: int) -> Tensor:
    weights = weights * valid.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0e-9)


def _safe_softmax(scores: Tensor, valid: Tensor, dim: int) -> Tensor:
    safe = scores.masked_fill(~valid, -1.0e4)
    return _safe_normalize(safe.softmax(dim=dim), valid, dim)


def _gather_time(source: Tensor, indices: Tensor) -> Tensor:
    batch = torch.arange(source.size(0), device=source.device)
    view = [source.size(0)] + [1] * (indices.ndim - 1)
    batch = batch.view(*view).expand_as(indices)
    return source[batch, indices]


@dataclass(frozen=True)
class QuotaRelationBlockConfig:
    d_model: int
    gate_dim: int
    relation_dim: int
    ff_mult: int = 4
    num_anchors: int = 4
    zone_quotas: tuple[int, int, int] = (2, 1, 1)
    mix_logit_init: float = -1.5

    @property
    def partners_per_anchor(self) -> int:
        return int(sum(self.zone_quotas))


class DistanceQuotaRelationBlock(nn.Module):
    ZONE_NAMES = ("near", "middle", "far")

    def __init__(self, cfg: QuotaRelationBlockConfig) -> None:
        super().__init__()
        if cfg.num_anchors < 1:
            raise ValueError("num_anchors must be positive")
        if len(cfg.zone_quotas) != 3 or min(cfg.zone_quotas) < 0:
            raise ValueError("zone_quotas must be three non-negative integers")
        if cfg.partners_per_anchor < 1:
            raise ValueError("at least one partner slot is required")
        self.cfg = cfg
        self.anchor_query = nn.Linear(cfg.d_model, cfg.gate_dim, bias=False)
        self.anchor_key = nn.Linear(cfg.d_model, cfg.gate_dim, bias=False)
        self.partner_query = nn.Linear(2 * cfg.d_model, cfg.gate_dim, bias=False)
        self.partner_key = nn.Linear(cfg.d_model, cfg.gate_dim, bias=False)
        self.operand_projection = nn.Linear(cfg.d_model, cfg.relation_dim, bias=False)
        feature_dim = 4 * cfg.relation_dim
        self.relation_mlp = nn.Sequential(
            nn.Linear(feature_dim, cfg.ff_mult * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.ff_mult * cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.mix_logit = nn.Parameter(torch.tensor(float(cfg.mix_logit_init)))
        slot_zones: list[int] = []
        for zone, quota in enumerate(cfg.zone_quotas):
            slot_zones.extend([zone] * quota)
        self.register_buffer(
            "slot_zones",
            torch.tensor(slot_zones, dtype=torch.long),
            persistent=False,
        )

    def copy_from_single_anchor(self, model: nn.Module, *, mix_logit: float | None = None) -> None:
        with torch.no_grad():
            self.anchor_query.load_state_dict(model.anchor_query.state_dict())
            self.anchor_key.load_state_dict(model.anchor_key.state_dict())
            self.partner_query.load_state_dict(model.partner_query.state_dict())
            self.partner_key.load_state_dict(model.partner_key.state_dict())
            self.operand_projection.load_state_dict(model.operand_projection.state_dict())
            self.relation_mlp.load_state_dict(model.relation_mlp.state_dict())
            self.output_norm.load_state_dict(model.output_norm.state_dict())
            source_mix = float(model.relation_mix_logit.detach())
            self.mix_logit.fill_(source_mix if mix_logit is None else float(mix_logit))

    @staticmethod
    def zone_masks(length: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        query = torch.arange(length, device=device)[:, None]
        source = torch.arange(length, device=device)[None, :]
        age = query - source
        available = query + 1
        first_cut = torch.div(available + 2, 3, rounding_mode="floor")
        second_cut = torch.div(2 * available + 2, 3, rounding_mode="floor")
        causal = age >= 0
        near = causal & (age < first_cut)
        middle = causal & (age >= first_cut) & (age < second_cut)
        far = causal & (age >= second_cut)
        return near, middle, far

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        *,
        valid_memory: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        batch, length, _ = query.shape
        if memory.shape[:2] != (batch, length):
            raise ValueError("query and memory must share [B,T]")
        if valid_memory is None:
            valid_memory = torch.ones((batch, length), dtype=torch.bool, device=query.device)

        source = torch.arange(length, device=query.device)[None, :]
        target = torch.arange(length, device=query.device)[:, None]
        causal = source <= target

        anchor_scores = torch.einsum(
            "btg,bsg->bts", self.anchor_query(query), self.anchor_key(memory)
        ) * (self.cfg.gate_dim**-0.5)
        anchor_valid_source = causal[None] & valid_memory[:, None, :]
        anchor_scores = anchor_scores.masked_fill(~anchor_valid_source, -torch.inf)
        anchor_count = min(self.cfg.num_anchors, length)
        anchor_top_scores, anchor_positions = anchor_scores.topk(anchor_count, dim=-1)
        anchor_valid = torch.isfinite(anchor_top_scores)
        anchor_weights = _safe_softmax(anchor_top_scores, anchor_valid, dim=-1)
        anchors = _gather_time(memory, anchor_positions)

        expanded_query = query.unsqueeze(2).expand(-1, -1, anchor_count, -1)
        partner_queries = self.partner_query(torch.cat((expanded_query, anchors), dim=-1))
        partner_keys = self.partner_key(memory)
        partner_scores = torch.einsum("btag,bsg->btas", partner_queries, partner_keys)
        partner_scores = partner_scores * (self.cfg.gate_dim**-0.5)
        partner_valid_source = causal[None, :, None, :] & valid_memory[:, None, None, :]
        partner_scores = partner_scores.masked_fill(~partner_valid_source, -torch.inf)

        anchor_exclusion = F.one_hot(anchor_positions, num_classes=length).any(dim=2)
        partner_scores = partner_scores.masked_fill(anchor_exclusion[:, :, None, :], -torch.inf)

        selected_scores: list[Tensor] = []
        selected_positions: list[Tensor] = []
        selected_valid: list[Tensor] = []
        zone_masks = self.zone_masks(length, query.device)
        for quota, zone_mask in zip(self.cfg.zone_quotas, zone_masks, strict=True):
            if quota == 0:
                continue
            zone_scores = partner_scores.masked_fill(~zone_mask[None, :, None, :], -torch.inf)
            values, positions = zone_scores.topk(quota, dim=-1)
            selected_scores.append(values)
            selected_positions.append(positions)
            selected_valid.append(torch.isfinite(values))

        partner_top_scores = torch.cat(selected_scores, dim=-1)
        partner_positions = torch.cat(selected_positions, dim=-1)
        partner_valid = torch.cat(selected_valid, dim=-1)
        partner_weights = _safe_softmax(partner_top_scores, partner_valid, dim=-1)
        partners = _gather_time(memory, partner_positions)

        anchor_operands = self.operand_projection(anchors).unsqueeze(3)
        anchor_operands = anchor_operands.expand(
            -1, -1, -1, self.cfg.partners_per_anchor, -1
        )
        partner_operands = self.operand_projection(partners)
        features = torch.cat(
            (
                anchor_operands,
                partner_operands,
                anchor_operands * partner_operands,
                anchor_operands - partner_operands,
            ),
            dim=-1,
        )
        relations = self.relation_mlp(features)
        per_anchor_context = (partner_weights.unsqueeze(-1) * relations).sum(dim=3)
        relation_context = (anchor_weights.unsqueeze(-1) * per_anchor_context).sum(dim=2)
        output = self.output_norm(
            query + torch.sigmoid(self.mix_logit) * relation_context
        )

        # Differentiable penalty: discourage different anchors from placing their
        # full partner distributions on the same positions.
        full_valid = torch.isfinite(partner_scores)
        full_probs = _safe_softmax(partner_scores, full_valid, dim=-1)
        overlap_terms: list[Tensor] = []
        for left in range(anchor_count):
            for right in range(left + 1, anchor_count):
                overlap_terms.append((full_probs[:, :, left] * full_probs[:, :, right]).sum(-1))
        cross_anchor_overlap = (
            torch.stack(overlap_terms, dim=-1).mean()
            if overlap_terms
            else query.new_zeros(())
        )

        zone_mass = []
        for zone in range(3):
            slots = self.slot_zones == zone
            mass = (
                anchor_weights.unsqueeze(-1)
                * partner_weights
                * slots[None, None, None, :]
            ).sum(dim=(2, 3))
            zone_mass.append(mass)
        zone_mass_tensor = torch.stack(zone_mass, dim=-1)
        target_mass = query.new_tensor(self.cfg.zone_quotas, dtype=torch.float32)
        target_mass = target_mass / target_mass.sum()
        zone_balance_loss = (zone_mass_tensor.mean(dim=(0, 1)) - target_mass).pow(2).mean()

        if not return_diagnostics:
            return output
        diagnostics = {
            "anchor_positions": anchor_positions,
            "anchor_weights": anchor_weights,
            "anchor_valid": anchor_valid,
            "partner_positions": partner_positions,
            "partner_weights": partner_weights,
            "partner_valid": partner_valid,
            "partner_top_scores": partner_top_scores,
            "relations": relations,
            "per_anchor_context": per_anchor_context,
            "relation_context": relation_context,
            "anchor_exclusion": anchor_exclusion,
            "zone_mass": zone_mass_tensor,
            "slot_zones": self.slot_zones,
            "cross_anchor_overlap_loss": cross_anchor_overlap,
            "zone_balance_loss": zone_balance_loss,
            "mix_scale": torch.sigmoid(self.mix_logit),
        }
        return output, diagnostics


class DistanceQuotaRelationStack(nn.Module):
    def __init__(
        self,
        block_configs: Iterable[QuotaRelationBlockConfig],
    ) -> None:
        super().__init__()
        configs = list(block_configs)
        if not configs:
            raise ValueError("at least one block is required")
        self.blocks = nn.ModuleList([DistanceQuotaRelationBlock(cfg) for cfg in configs])

    def copy_from_single_anchor(
        self,
        model: nn.Module,
        *,
        extra_block_mix_logit: float = -3.0,
    ) -> None:
        for index, block in enumerate(self.blocks):
            block.copy_from_single_anchor(
                model,
                mix_logit=None if index == 0 else extra_block_mix_logit,
            )

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        *,
        valid_memory: Tensor | None = None,
        return_diagnostics: bool = False,
        disabled_blocks: set[int] | None = None,
    ) -> Tensor | tuple[Tensor, list[dict[str, Tensor]]]:
        disabled = disabled_blocks or set()
        current = query
        diagnostics: list[dict[str, Tensor]] = []
        for index, block in enumerate(self.blocks):
            if index in disabled:
                continue
            if return_diagnostics:
                current, block_diagnostics = block(
                    current,
                    memory,
                    valid_memory=valid_memory,
                    return_diagnostics=True,
                )
                diagnostics.append(block_diagnostics)
            else:
                current = block(current, memory, valid_memory=valid_memory)
        return (current, diagnostics) if return_diagnostics else current


class RelationLexQuotaStackLM(nn.Module):
    """Reuse a trained RelationLex backbone with independent relation blocks."""

    def __init__(
        self,
        base_model: nn.Module,
        block_configs: Iterable[QuotaRelationBlockConfig],
        *,
        extra_block_mix_logit: float = -3.0,
    ) -> None:
        super().__init__()
        self.backbone = base_model
        self.relation_stack = DistanceQuotaRelationStack(block_configs)
        self.relation_stack.copy_from_single_anchor(
            base_model,
            extra_block_mix_logit=extra_block_mix_logit,
        )
        # Remove the now-replaced single-block parameters from the backbone so
        # parameter counts and optimizers do not include unused modules.
        for name in (
            "anchor_query",
            "anchor_key",
            "partner_query",
            "partner_key",
            "operand_projection",
            "relation_mlp",
            "output_norm",
            "relation_mix_logit",
        ):
            delattr(self.backbone, name)

    def forward(
        self,
        tokens: Tensor,
        boundaries: Tensor,
        *,
        return_aux: bool = False,
        return_diagnostics: bool = False,
        disabled_blocks: set[int] | None = None,
    ):
        query, memory = self.backbone.streams_with_boundaries(tokens, boundaries)
        valid = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
        if return_aux or return_diagnostics:
            hidden, diagnostics = self.relation_stack(
                query,
                memory,
                valid_memory=valid,
                return_diagnostics=True,
                disabled_blocks=disabled_blocks,
            )
        else:
            hidden = self.relation_stack(
                query,
                memory,
                valid_memory=valid,
                disabled_blocks=disabled_blocks,
            )
            diagnostics = []
        outputs = (
            self.backbone.output(hidden),
            self.backbone.boundary_output(hidden),
        )
        if return_diagnostics:
            return outputs, diagnostics
        if return_aux:
            overlap = torch.stack(
                [row["cross_anchor_overlap_loss"] for row in diagnostics]
            ).mean()
            zone_balance = torch.stack(
                [row["zone_balance_loss"] for row in diagnostics]
            ).mean()
            mean_k = torch.stack(
                [row["partner_valid"].float().sum(-1).mean() for row in diagnostics]
            ).mean()
            mix_scales = torch.stack([row["mix_scale"] for row in diagnostics])
            return outputs, {
                "cross_anchor_overlap_loss": overlap,
                "zone_balance_loss": zone_balance,
                "mean_k": mean_k.detach(),
                "mix_scales": mix_scales.detach(),
            }
        return outputs


def clone_base_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)
