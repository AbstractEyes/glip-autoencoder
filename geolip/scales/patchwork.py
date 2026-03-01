"""
Patchwork — Dual-Stream Geometric Vocabulary (8 states)
========================================================

Prototype-scale vocabulary with KSimplexChannel geometry + transformer features.

Two parallel streams through 8 topological states:
    GEOMETRIC:  KSimplexChannel per patch → d², vol² (11-dim, CM-validated)
    FEATURE:    Multi-head self-attention → learned representations (feat_dim)
    GATING:     Geometry gates features: feat_out = feat × sigmoid(geo) × sigmoid(vol²)

Data flow:
    (B, C, H, W) → Patchify → (B, P, patch_dim)
    → Lift: KSC → geo (B,P,11), Linear → feat (B,P,feat_dim)
    → 8× GeometricState: dual-stream with geometry gating
    → CrossStateComposition with Cantor alignment bias
    → (B, P, combined_dim)

Spatial structure preserved end-to-end. CM intrinsic at every level.
Patches ARE the tokens — no manufactured expansion.

Scale hierarchy:
    Patchwork:  8 states (this file)
    Chunk:      512 states (chunk.py)
    Sector:     512 chunks (sector.py, placeholder)

Author: AbstractPhil + Claude
License: Apache-2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple, NamedTuple
from itertools import combinations

from ..core import CantorTopology
from ..vocabulary.simplex_factory import SimplexFactory


# ══════════════════════════════════════════════════════════════════════════════
# CMValidator — batch-friendly Cayley-Menger
# ══════════════════════════════════════════════════════════════════════════════

class CMValidator(nn.Module):
    """Cayley-Menger volume validator. Operates on arbitrary batch dims."""

    def __init__(self, k: int):
        super().__init__()
        self._k = k
        self._nv = k + 1

        pairs = list(combinations(range(self._nv), 2))
        self._npairs = len(pairs)
        self.register_buffer("_pi", torch.tensor([p[0] for p in pairs], dtype=torch.long))
        self.register_buffer("_pj", torch.tensor([p[1] for p in pairs], dtype=torch.long))

        sign = (-1.0) ** (k + 1)
        fact = math.factorial(k)
        self._prefactor = sign / ((2.0 ** k) * (fact ** 2))

    def forward(self, verts: Tensor) -> Tuple[Tensor, Tensor]:
        """verts: (..., nv, edim) → (d2_pairs: ..., npairs), (vol2: ...,)"""
        gram = torch.einsum("...ve,...we->...vw", verts, verts)
        norms = torch.diagonal(gram, dim1=-2, dim2=-1)
        d2_mat = norms.unsqueeze(-1) + norms.unsqueeze(-2) - 2 * gram
        d2_mat = F.relu(d2_mat)

        d2_pairs = d2_mat[..., self._pi, self._pj]

        shape = d2_mat.shape[:-2]
        V = d2_mat.shape[-1]
        cm = torch.zeros(*shape, V + 1, V + 1, device=d2_mat.device, dtype=d2_mat.dtype)
        cm[..., 0, 1:] = 1.0
        cm[..., 1:, 0] = 1.0
        cm[..., 1:, 1:] = d2_mat
        vol2 = self._prefactor * torch.linalg.det(cm)

        return d2_pairs, vol2


# ══════════════════════════════════════════════════════════════════════════════
# KSimplexChannel — per-position geometric features
# ══════════════════════════════════════════════════════════════════════════════

class KSimplexChannel(nn.Module):
    """
    Per-position simplex encoder: input → template deformation → CM geometry.
    For k=4: 5 vertices, 10 pairwise d² + 1 vol² = 11 output features.
    Spatial structure fully preserved.
    """
    BASE_DEFORM = 0.05

    def __init__(self, k: int, in_dim: int, edim: int):
        super().__init__()
        self._k = k
        self._nv = k + 1
        self._edim = edim
        self._cm = CMValidator(k)
        self._out_dim = self._cm._npairs + 1  # 10 + 1 = 11
        self._to_deform = nn.Linear(in_dim, self._nv * edim)
        self._norm = nn.LayerNorm(self._out_dim)

        factory = SimplexFactory(k=k, embed_dim=edim, method="regular", scale=1.0)
        self.register_buffer("_template", factory.build_torch(dtype=torch.float32))

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """x: (..., in_dim) → geo: (..., 11), vol2: (...,)"""
        deform = self._to_deform(x).unflatten(-1, (self._nv, self._edim))
        verts = self._template + self.BASE_DEFORM * deform
        d2, vol2 = self._cm(verts)
        geo = torch.cat([d2, vol2.unsqueeze(-1)], dim=-1)
        geo = self._norm(geo)
        return geo, vol2


# ══════════════════════════════════════════════════════════════════════════════
# GeometricState — dual-stream: geometry + gated features
# ══════════════════════════════════════════════════════════════════════════════

class GeometricState(nn.Module):
    """
    One of N dual-stream geometric "lenses".

    Geometric stream:  KSimplexChannel → 11-dim per patch (CM-validated)
    Feature stream:    Multi-head self-attention + FF on patches
    Gating:            feat_out = feat × sigmoid(geo_gate) × sigmoid(validity)

    Geometry tells the model WHAT is structurally true.
    Transformer learns WHAT to do with that structure.
    Gate ensures features only activate where geometry supports them.
    """

    def __init__(
        self,
        geo_dim: int = 11,
        feat_dim: int = 64,
        num_patches: int = 16,
        k: int = 4,
        edim: int = 8,
        state_idx: int = 0,
        branch_path: Tensor = None,
        staircase_val: float = 0.0,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.geo_dim = geo_dim
        self.feat_dim = feat_dim
        self.num_patches = num_patches
        self.state_idx = state_idx

        if branch_path is not None:
            self.register_buffer("branch_path", branch_path)
        else:
            self.register_buffer("branch_path", torch.zeros(5, dtype=torch.int32))
        self.staircase_val = staircase_val

        # ── Geometric stream ──
        self.geo_channel = KSimplexChannel(k, geo_dim, edim)
        self.geo_residual_norm = nn.LayerNorm(geo_dim)

        # ── Feature stream: multi-head self-attention ──
        assert feat_dim % n_heads == 0, f"feat_dim {feat_dim} not divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = feat_dim // n_heads

        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.attn_norm = nn.LayerNorm(feat_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(feat_dim)

        # ── Geometry gates features ──
        self.geo_gate = nn.Sequential(
            nn.Linear(geo_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.validity_gate = nn.Sequential(
            nn.Linear(1, feat_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, geo: Tensor, feat: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            geo:  (B, P, geo_dim)  — geometric features per patch
            feat: (B, P, feat_dim) — learned features per patch
        Returns:
            geo_out:  (B, P, geo_dim)
            feat_out: (B, P, feat_dim) — gated by geometry
            vol2:     (B, P)
        """
        B, P, _ = geo.shape

        # ── Geometric stream ──
        geo_flat = geo.reshape(B * P, self.geo_dim)
        geo_new, vol2 = self.geo_channel(geo_flat)
        geo_new = geo_new.reshape(B, P, self.geo_dim)
        vol2 = vol2.reshape(B, P)
        geo_out = self.geo_residual_norm(geo + geo_new)

        # ── Feature stream ──
        q = self.q_proj(feat).view(B, P, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(feat).view(B, P, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(feat).view(B, P, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, P, self.feat_dim)
        feat = self.attn_norm(feat + self.out_proj(attn_out))

        # Feed-forward
        feat = self.ff_norm(feat + self.ff(feat))

        # ── Gating ──
        gate = self.geo_gate(geo_out)
        validity = self.validity_gate(vol2.unsqueeze(-1))
        feat_out = feat * gate * validity

        return geo_out, feat_out, vol2


# ══════════════════════════════════════════════════════════════════════════════
# CrossStateComposition
# ══════════════════════════════════════════════════════════════════════════════

class CrossStateComposition(nn.Module):
    """Cross-state attention on combined [geo ‖ feat] with Cantor alignment bias."""

    def __init__(
        self,
        combined_dim: int,
        num_patches: int = 16,
        n_heads: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert combined_dim % n_heads == 0, (
            f"combined_dim {combined_dim} must divide n_heads {n_heads}")
        self.combined_dim = combined_dim
        self.num_patches = num_patches
        self.n_heads = n_heads
        self.head_dim = combined_dim // n_heads

        self.q_proj = nn.Linear(combined_dim, combined_dim)
        self.k_proj = nn.Linear(combined_dim, combined_dim)
        self.v_proj = nn.Linear(combined_dim, combined_dim)
        self.out_proj = nn.Linear(combined_dim, combined_dim)
        self.norm = nn.LayerNorm(combined_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        all_tokens: Tensor,
        active_indices: Tensor,
        routing_weights: Tensor,
        alignment_matrix: Tensor,
    ) -> Tensor:
        """
        Args:
            all_tokens:       (B, n_active * P, combined_dim)
            active_indices:   (B, n_active)
            routing_weights:  (B, n_active)
            alignment_matrix: (num_states, num_states)
        Returns:
            composed: (B, P, combined_dim)
        """
        B, S, D = all_tokens.shape
        n_active = active_indices.shape[1]
        P = self.num_patches

        q = self.q_proj(all_tokens).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(all_tokens).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(all_tokens).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Cantor alignment bias
        idx = active_indices[0]
        pair_align = alignment_matrix[idx][:, idx]
        align_bias = pair_align.repeat_interleave(P, dim=0).repeat_interleave(P, dim=1)
        scores = scores + align_bias.unsqueeze(0).unsqueeze(0)

        attn = self.drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, D)

        # Weighted pool across states
        out_states = out.view(B, n_active, P, D)
        w = routing_weights.unsqueeze(-1).unsqueeze(-1)
        composed = (out_states * w).sum(dim=1)

        return self.norm(self.out_proj(composed))


# ══════════════════════════════════════════════════════════════════════════════
# NeedsBasedRouter
# ══════════════════════════════════════════════════════════════════════════════

class NeedsBasedRouter(nn.Module):
    """Routes patches to topological states based on geometric features."""

    def __init__(self, geo_dim: int, num_states: int):
        super().__init__()
        self.num_states = num_states
        self.proj = nn.Linear(geo_dim, 1)

    def forward(
        self, geo: Tensor, positions: Tensor, training: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            geo:       (B, P, geo_dim)
            positions: (num_states,) — Cantor positions
        Returns:
            indices: (B, n_active)
            weights: (B, n_active)
        """
        pooled = geo.mean(dim=1)
        raw = self.proj(pooled).squeeze(-1)
        pos = torch.sigmoid(raw)
        dists = (pos.unsqueeze(-1) - positions.unsqueeze(0)).abs()

        if training:
            weights = F.softmax(-dists * 10, dim=-1)
            indices = torch.arange(self.num_states, device=pos.device)
            indices = indices.unsqueeze(0).expand(geo.shape[0], -1)
            return indices, weights
        else:
            topk = torch.topk(-dists, min(4, self.num_states), dim=-1)
            return topk.indices, F.softmax(-topk.values, dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# PatchworkOutput
# ══════════════════════════════════════════════════════════════════════════════

class PatchworkOutput(NamedTuple):
    """Output container for Patchwork/Chunk forward pass."""
    composed: Tensor           # (B, P, combined_dim) — [geo ‖ feat]
    geo: Tensor                # (B, P, geo_dim) — pure geometry (11)
    feat: Tensor               # (B, P, feat_dim) — gated features
    active_indices: Tensor     # (B, n_active)
    routing_weights: Tensor    # (B, n_active)
    vol_sq: Tensor             # (B, n_levels, P) — CM volumes
    loss: Tensor               # CM validity loss


# ══════════════════════════════════════════════════════════════════════════════
# Patchwork — 8 states, prototype scale
# ══════════════════════════════════════════════════════════════════════════════

class Patchwork(nn.Module):
    """
    8-state dual-stream geometric vocabulary. Prototype scale.

    (B, C, H, W) → patchify → lift → 8× GeometricState → compose → (B, P, combined)
    """

    NUM_STATES = 8
    CANTOR_LEVELS = 5

    def __init__(
        self,
        in_channels: int = 16,
        in_height: int = 16,
        in_width: int = 16,
        patch_grid: int = 4,
        k: int = 4,
        edim: int = 8,
        feat_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.k = k
        self.feat_dim = feat_dim
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

        # ── 8 states ──
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

    def forward(self, latents: Tensor) -> PatchworkOutput:
        patches = self._patchify(latents)
        geo, feat, lift_vol = self._lift(patches)

        active_idx, routing_w = self.router(geo, self.state_positions, self.training)

        all_combined = []
        all_vol = [lift_vol]

        for state in self.states:
            geo_s, feat_s, vol2 = state(geo, feat)
            combined = torch.cat([geo_s, feat_s], dim=-1)
            all_combined.append(combined)
            all_vol.append(vol2)

        stacked = torch.cat(all_combined, dim=1)
        vol_sq = torch.stack(all_vol, dim=1)

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