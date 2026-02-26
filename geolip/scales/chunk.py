"""
Chunk — 512 × 256 × 64 = 8,388,608 tokens
=============================================

Full-scale geometric structural vocabulary.
512 topological states at attention head alignment.
Needs-based loading for hardware efficiency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple, NamedTuple

from ..core import KSimplexLinear, CantorTopology, CayleyMengerValidator
from .patch import Patchifier
from .state import TopologicalState
from .patchwork import NeedsBasedRouter, CrossStateComposition


class ChunkOutput(NamedTuple):
    composed: Tensor
    active_indices: Tensor
    routing_weights: Tensor
    vol_sq: Tensor
    loss: Optional[Tensor]
    loss_dict: Optional[Dict[str, float]]


class Chunk(nn.Module):
    """
    512 × 256 × 64 = 8,388,608 tokens.

    Full geometric structural vocabulary.
    Needs-based loading: max_active_states controls compute budget.
    """

    NUM_STATES = 512
    CANTOR_LEVELS = 9

    def __init__(
        self,
        in_channels: int = 32,
        in_height: int = 64,
        in_width: int = 64,
        patch_grid: int = 8,
        tokens_per_state: int = 256,
        token_dim: int = 64,
        simplex_k: int = 4,
        deform_alpha: float = 0.05,
        max_active_states: int = 64,
    ):
        super().__init__()
        self.tokens_per_state = tokens_per_state
        self.token_dim = token_dim
        self.max_active_states = max_active_states

        self.topology = CantorTopology(
            num_states=self.NUM_STATES, levels=self.CANTOR_LEVELS
        )

        self.patchifier = Patchifier(
            in_channels, in_height, in_width, patch_grid, simplex_k
        )
        patch_dim = self.patchifier.patch_dim

        self.router = NeedsBasedRouter(patch_dim, self.NUM_STATES, simplex_k)
        self.register_buffer("state_positions", self.topology.positions)
        self.register_buffer("alignment_matrix", self.topology.alignment_matrix)

        self.states = nn.ModuleList()
        for i in range(self.NUM_STATES):
            self.states.append(TopologicalState(
                input_dim=patch_dim,
                tokens_per_state=tokens_per_state,
                token_dim=token_dim,
                simplex_k=simplex_k,
                deform_alpha=deform_alpha,
                state_idx=i,
                branch_path=self.topology.branch_paths[i],
                staircase_val=float(self.topology.staircase_vals[i]),
            ))

        self.cross_compose = CrossStateComposition(
            token_dim, tokens_per_state, simplex_k
        )

        self.cm = CayleyMengerValidator(simplex_k)

    def forward(
        self, latents: Tensor, active_budget: Optional[int] = None
    ) -> ChunkOutput:
        B = latents.shape[0]
        budget = active_budget or self.max_active_states

        patches = self.patchifier(latents)
        pooled = patches.mean(dim=1)

        active_idx, routing_w = self.router(
            patches, self.state_positions, self.training
        )
        # Trim to budget
        if active_idx.shape[1] > budget:
            active_idx = active_idx[:, :budget]
            routing_w = F.softmax(routing_w[:, :budget], dim=-1)

        n_active = active_idx.shape[1]

        # Single CPU transfer for dispatch indices (one sync, not N)
        active_idx_cpu = active_idx[0].tolist()

        all_tokens = []
        all_vol = []
        for j in range(n_active):
            tokens, vsq = self.states[active_idx_cpu[j]](pooled)
            all_tokens.append(tokens)
            all_vol.append(vsq)

        stacked = torch.cat(all_tokens, dim=1)
        vol_sq = torch.stack(all_vol, dim=1)

        composed = self.cross_compose(
            stacked, active_idx, routing_w, self.alignment_matrix
        )

        cm_loss = self.cm.validity_loss(vol_sq)

        return ChunkOutput(
            composed=composed,
            active_indices=active_idx,
            routing_weights=routing_w,
            vol_sq=vol_sq,
            loss=cm_loss,
            loss_dict={"cm_validity": cm_loss.detach()},
        )