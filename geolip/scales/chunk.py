"""
Chunk — Dual-Stream Geometric Vocabulary (512 states)
======================================================

Full-scale geometric structural vocabulary.
512 topological states with needs-based loading.

Same dual-stream architecture as Patchwork:
    GEOMETRIC:  KSimplexChannel per patch → d², vol² (11-dim, CM-validated)
    FEATURE:    Multi-head self-attention → learned representations (feat_dim)
    GATING:     Geometry gates features: feat_out = feat × sigmoid(geo) × sigmoid(vol²)

Key difference: max_active_states budget system.
Only active states run forward — the rest stay dormant.
At 512 states, full forward is never needed.

Scale hierarchy:
    Patchwork:  8 states   (patchwork.py)
    Chunk:      512 states (this file)
    Sector:     512 chunks (sector.py, placeholder)

Author: AbstractPhil + Claude
License: Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from ..core import CantorTopology
from .patchwork import (
    KSimplexChannel,
    GeometricState,
    CrossStateComposition,
    NeedsBasedRouter,
    PatchworkOutput,
)


class Chunk(nn.Module):
    """
    512-state dual-stream geometric vocabulary.

    Full geometric structural vocabulary with needs-based loading.
    max_active_states controls compute budget — only active states
    run forward. At inference, router selects top-k states.

    (B, C, H, W) → patchify → lift → N× GeometricState → compose → (B, P, combined)
    where N = min(active_budget, 512)
    """

    NUM_STATES = 512
    CANTOR_LEVELS = 9

    def __init__(
        self,
        in_channels: int = 32,
        in_height: int = 64,
        in_width: int = 64,
        patch_grid: int = 8,
        k: int = 4,
        edim: int = 8,
        feat_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_active_states: int = 64,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.k = k
        self.feat_dim = feat_dim
        self.max_active_states = max_active_states
        num_patches = patch_grid * patch_grid

        pH = in_height // patch_grid
        pW = in_width // patch_grid
        patch_dim = in_channels * pH * pW

        # ── Lift ──
        self.geo_lift = KSimplexChannel(k, patch_dim, edim)
        self.geo_dim = self.geo_lift.out_dim  # 11
        self.feat_lift = nn.Sequential(
            nn.Linear(patch_dim, feat_dim),
            nn.GELU(),
            nn.LayerNorm(feat_dim),
        )
        self.num_patches = num_patches
        self.combined_dim = self.geo_dim + feat_dim

        # ── Topology ──
        self.topology = CantorTopology(
            num_states=self.NUM_STATES, levels=self.CANTOR_LEVELS
        )
        self.register_buffer("state_positions", self.topology.positions)
        self.register_buffer("alignment_matrix", self.topology.alignment_matrix)

        # ── Router ──
        self.router = NeedsBasedRouter(self.geo_dim, self.NUM_STATES)

        # ── 512 states ──
        states = []
        for i in range(self.NUM_STATES):
            states.append(GeometricState(
                geo_dim=self.geo_dim,
                feat_dim=feat_dim,
                num_patches=num_patches,
                k=k,
                edim=edim,
                state_idx=i,
                branch_path=self.topology.branch_paths[i],
                staircase_val=float(self.topology.staircase_vals[i]),
                n_heads=n_heads,
                dropout=dropout,
            ))
        self.states = nn.ModuleList(states)

        # ── Cross-state composition ──
        cross_heads = 5 if self.combined_dim % 5 == 0 else 3
        self.cross_compose = CrossStateComposition(
            combined_dim=self.combined_dim,
            num_patches=num_patches,
            n_heads=cross_heads,
            dropout=dropout,
        )

        self.output_dim = self.combined_dim
        self.patch_feat_dim = self.combined_dim * num_patches

    def _patchify(self, latents: Tensor) -> Tensor:
        """(B, C, H, W) → (B, P, patch_dim)"""
        B, C, H, W = latents.shape
        g = self.patch_grid
        pH, pW = H // g, W // g
        x = latents.reshape(B, C, g, pH, g, pW)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, -1)

    def _lift(self, patches: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """(B, P, patch_dim) → geo (B,P,11), feat (B,P,feat_dim), vol (B,P)"""
        B, P, D = patches.shape
        flat = patches.reshape(B * P, D)
        geo, vol = self.geo_lift(flat)
        geo = geo.reshape(B, P, self.geo_dim)
        vol = vol.reshape(B, P)
        feat = self.feat_lift(flat).reshape(B, P, self.feat_dim)
        return geo, feat, vol

    def forward(
        self, latents: Tensor, active_budget: Optional[int] = None
    ) -> PatchworkOutput:
        """
        Args:
            latents:       (B, C, H, W)
            active_budget: override max_active_states for this call
        Returns:
            PatchworkOutput with composed, geo, feat, vol_sq, loss
        """
        budget = active_budget or self.max_active_states
        P = self.num_patches

        patches = self._patchify(latents)
        geo, feat, lift_vol = self._lift(patches)

        # Route — training uses all states, inference uses top-k
        active_idx, routing_w = self.router(
            geo, self.state_positions, self.training
        )

        # Trim to budget
        if active_idx.shape[1] > budget:
            if self.training:
                # During training, keep top-budget by routing weight
                _, top_idx = routing_w.topk(budget, dim=-1)
                active_idx = torch.gather(active_idx, 1, top_idx)
                routing_w = torch.gather(routing_w, 1, top_idx)
                routing_w = F.softmax(routing_w, dim=-1)
            else:
                active_idx = active_idx[:, :budget]
                routing_w = F.softmax(routing_w[:, :budget], dim=-1)

        n_active = active_idx.shape[1]

        # Single CPU transfer for dispatch indices (one sync, not N)
        active_idx_cpu = active_idx[0].tolist()

        # Run only active states
        all_combined = []
        all_vol = [lift_vol]

        for j in range(n_active):
            state = self.states[active_idx_cpu[j]]
            geo_s, feat_s, vol2 = state(geo, feat)
            combined = torch.cat([geo_s, feat_s], dim=-1)
            all_combined.append(combined)
            all_vol.append(vol2)

        stacked = torch.cat(all_combined, dim=1)      # (B, n_active*P, combined)
        vol_sq = torch.stack(all_vol, dim=1)           # (B, 1+n_active, P)

        composed = self.cross_compose(
            stacked, active_idx, routing_w, self.alignment_matrix
        )

        geo_out = composed[:, :, :self.geo_dim]
        feat_out = composed[:, :, self.geo_dim:]
        cm_loss = F.relu(-vol_sq).mean()

        return PatchworkOutput(
            composed=composed,
            geo=geo_out,
            feat=feat_out,
            active_indices=active_idx,
            routing_weights=routing_w,
            vol_sq=vol_sq,
            loss=cm_loss,
        )