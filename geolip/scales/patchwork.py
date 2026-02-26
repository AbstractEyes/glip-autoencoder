"""
Patchwork — 8 × 256 × 64 = 131,072 tokens
=============================================

Prototype-scale vocabulary for validation.
8 topological states, each 256 × 64.
Full training: all states active.
Inference: needs-based routing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple

from ..core import KSimplexLinear, CantorTopology, CayleyMengerValidator
from .patch import Patchifier
from .state import TopologicalState


class PatchworkOutput:
    """
    Output container for Patchwork forward pass.

    Access patterns:
        out.composed          — (B, tokens_per_state, token_dim) main output
        out.shape             — shortcut for out.composed.shape
        out.active_indices    — which states were active
        out.routing_weights   — soft routing weights
        out.vol_sq            — per-state CM volumes squared
        out.loss              — CM validity loss (for training)
        out.loss_dict         — dict of detached loss components

    Tensor-like:
        out.shape, out.dtype, out.device all delegate to composed.
        out[0] indexes into composed. out.detach() detaches composed.
    """

    __slots__ = ("composed", "active_indices", "routing_weights", "vol_sq", "loss", "loss_dict")

    def __init__(
        self,
        composed: Tensor,
        active_indices: Tensor,
        routing_weights: Tensor,
        vol_sq: Tensor,
        loss: Optional[Tensor] = None,
        loss_dict: Optional[Dict[str, float]] = None,
    ):
        self.composed = composed
        self.active_indices = active_indices
        self.routing_weights = routing_weights
        self.vol_sq = vol_sq
        self.loss = loss
        self.loss_dict = loss_dict or {}

    # --- Tensor-like delegation to composed ---

    @property
    def shape(self):
        return self.composed.shape

    @property
    def dtype(self):
        return self.composed.dtype

    @property
    def device(self):
        return self.composed.device

    def size(self, *args):
        return self.composed.size(*args)

    def dim(self):
        return self.composed.dim()

    def __getitem__(self, idx):
        return self.composed[idx]

    def __len__(self):
        return self.composed.shape[0]

    def detach(self):
        return PatchworkOutput(
            composed=self.composed.detach(),
            active_indices=self.active_indices.detach(),
            routing_weights=self.routing_weights.detach(),
            vol_sq=self.vol_sq.detach(),
            loss=self.loss.detach() if self.loss is not None else None,
            loss_dict=self.loss_dict,
        )

    def __repr__(self):
        return (
            f"PatchworkOutput(shape={self.shape}, "
            f"active={self.active_indices.shape[-1]} states, "
            f"loss={self.loss.item():.6f})"
            if self.loss is not None
            else f"PatchworkOutput(shape={self.shape}, "
            f"active={self.active_indices.shape[-1]} states)"
        )


class NeedsBasedRouter(nn.Module):
    """Routes input to relevant topological states."""

    def __init__(self, input_dim: int, num_states: int, simplex_k: int = 4):
        super().__init__()
        self.num_states = num_states
        self.proj = KSimplexLinear(input_dim, 1, k=simplex_k)

    def forward(
        self, patches: Tensor, positions: Tensor, training: bool = True
    ) -> Tuple[Tensor, Tensor]:
        pooled = patches.mean(dim=1)
        raw = self.proj(pooled).squeeze(-1)
        pos = torch.sigmoid(raw)
        dists = (pos.unsqueeze(-1) - positions.unsqueeze(0)).abs()

        if training:
            weights = F.softmax(-dists * 10, dim=-1)
            indices = torch.arange(self.num_states, device=pos.device)
            indices = indices.unsqueeze(0).expand(patches.shape[0], -1)
            return indices, weights
        else:
            topk = torch.topk(-dists, min(4, self.num_states), dim=-1)
            return topk.indices, F.softmax(-topk.values, dim=-1)


class CrossStateComposition(nn.Module):
    """Compose across active states with alignment bias."""

    def __init__(
        self,
        token_dim: int = 64,
        tokens_per_state: int = 256,
        simplex_k: int = 4,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.tokens_per_state = tokens_per_state
        K = simplex_k
        self.q_proj = KSimplexLinear(token_dim, token_dim, k=K)
        self.k_proj = KSimplexLinear(token_dim, token_dim, k=K)
        self.v_proj = KSimplexLinear(token_dim, token_dim, k=K)
        self.out_proj = KSimplexLinear(token_dim, token_dim, k=K)
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        all_tokens: Tensor,
        active_indices: Tensor,
        routing_weights: Tensor,
        alignment_matrix: Tensor,
    ) -> Tensor:
        B, S, D = all_tokens.shape
        n_active = active_indices.shape[1]
        tps = self.tokens_per_state

        q = self.q_proj(all_tokens)
        k = self.k_proj(all_tokens)
        v = self.v_proj(all_tokens)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        # Alignment bias — vectorized
        # Gather the (n_active, n_active) sub-matrix from topology
        idx = active_indices[0]  # (n_active,) — topology is shared across batch
        pair_align = alignment_matrix[idx][:, idx]  # (n_active, n_active)
        # Expand each scalar to (tps, tps) block → (n_active*tps, n_active*tps)
        align_bias = pair_align.repeat_interleave(tps, dim=0).repeat_interleave(tps, dim=1)
        # Broadcast across batch
        scores = scores + align_bias.unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Weighted pool — vectorized: (B, n_active, tps, D) → (B, tps, D)
        out_states = out.view(B, n_active, tps, D)
        w = routing_weights.unsqueeze(-1).unsqueeze(-1)  # (B, n_active, 1, 1)
        composed = (out_states * w).sum(dim=1)            # (B, tps, D)

        return self.norm(self.out_proj(composed))


class Patchwork(nn.Module):
    """
    8 × 256 × 64 = 131,072 tokens.

    Prototype-scale geometric vocabulary.
    """

    NUM_STATES = 8
    CANTOR_LEVELS = 5

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
    ):
        super().__init__()
        self.tokens_per_state = tokens_per_state
        self.token_dim = token_dim

        # Topology (deterministic)
        self.topology = CantorTopology(
            num_states=self.NUM_STATES, levels=self.CANTOR_LEVELS
        )

        # Patchification
        self.patchifier = Patchifier(
            in_channels, in_height, in_width, patch_grid, simplex_k
        )
        patch_dim = self.patchifier.patch_dim

        # Router
        self.router = NeedsBasedRouter(patch_dim, self.NUM_STATES, simplex_k)
        self.register_buffer("state_positions", self.topology.positions)
        self.register_buffer("alignment_matrix", self.topology.alignment_matrix)

        # 8 states (ModuleList for compiler-friendly integer indexing)
        states_list = []
        for i in range(self.NUM_STATES):
            states_list.append(TopologicalState(
                input_dim=patch_dim,
                tokens_per_state=tokens_per_state,
                token_dim=token_dim,
                simplex_k=simplex_k,
                deform_alpha=deform_alpha,
                state_idx=i,
                branch_path=self.topology.branch_paths[i],
                staircase_val=float(self.topology.staircase_vals[i]),
            ))
        self.states = nn.ModuleList(states_list)

        # Cross-state composition
        self.cross_compose = CrossStateComposition(
            token_dim, tokens_per_state, simplex_k
        )

        # CM
        self.cm = CayleyMengerValidator(simplex_k)

    def forward(self, latents: Tensor) -> PatchworkOutput:
        B = latents.shape[0]

        patches = self.patchifier(latents)
        pooled = patches.mean(dim=1)

        active_idx, routing_w = self.router(
            patches, self.state_positions, self.training
        )

        # Run all states — only 8, always cheap
        # Routing weights handle the mixing; no .item() needed
        all_tokens = []
        all_vol = []
        for state in self.states:
            tokens, vsq = state(pooled)
            all_tokens.append(tokens)
            all_vol.append(vsq)

        stacked = torch.cat(all_tokens, dim=1)  # (B, 8*256, 64)
        vol_sq = torch.stack(all_vol, dim=1)     # (B, 8, 256)

        composed = self.cross_compose(
            stacked, active_idx, routing_w, self.alignment_matrix
        )

        cm_loss = self.cm.validity_loss(vol_sq)

        return PatchworkOutput(
            composed=composed,
            active_indices=active_idx,
            routing_weights=routing_w,
            vol_sq=vol_sq,
            loss=cm_loss,
            loss_dict={"cm_validity": cm_loss.detach()},
        )