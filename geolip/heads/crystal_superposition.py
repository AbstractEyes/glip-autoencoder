"""
Crystal Superposition Head
===========================

Pentachoron-based multi-label classification head using geometric
attractor basins. Each class is a 5-vertex crystal (pentachoron)
in projection space with role-weighted cosine scoring.

Designed to compose with:
    - Patchwork composed tokens:          (B, T, D)
    - SuperpositionPatchClassifier output: (B, 64, embed_dim) + optional (B, 64, 17) gates

Role weights:
    anchor=1.0, need=-0.5, relation=0.75, purpose=0.75, observer=-0.5

The negative roles create a geometric "lock" — a feature vector must
align with the full pentachoron shape (not just one vertex) to score
highly. This is the superposition property: multiple classes can
activate simultaneously via independent attractor basins.

Loss: Rose (margin-based contrastive) + crystal collapse penalty
      + inter-class centroid separation + optional CM passthrough.

Author: AbstractPhil + Claude
License: Apache-2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

NUM_VERTICES = 5  # pentachoron

# Role weights for the 5 vertices of each class pentachoron.
# anchor:   primary attractor — feature should align here
# need:     repulsive — feature should NOT align here (negative weight)
# relation: attractive — structural relationship axis
# purpose:  attractive — functional meaning axis
# observer: repulsive — anti-correlation axis
ROLE_NAMES = ["anchor", "need", "relation", "purpose", "observer"]
DEFAULT_ROLE_WEIGHTS = [1.0, -0.5, 0.75, 0.75, -0.5]


# ══════════════════════════════════════════════════════════════════════════════
# Crystal Superposition Head
# ══════════════════════════════════════════════════════════════════════════════

class CrystalSuperpositionHead(nn.Module):
    """
    Pentachoron crystal classification head.

    Each class is represented by a 5-vertex crystal in projection space.
    Classification is cosine similarity against all 5 vertices, weighted
    by role. Multi-label behavior falls out naturally: each pentachoron
    is an independent attractor basin, not competing through softmax.

    Args:
        feature_dim:    Input feature dimension (token_dim or embed_dim)
        num_classes:    Number of output classes
        crystal_dim:    Dimension of crystal projection space
        gate_dim:       If > 0, expects gate vectors concatenated with features.
                        Set to 17 when using SuperpositionPatchClassifier gates.
                        Set to 0 when using raw Patchwork tokens.
        temperature:    Initial temperature for cosine scaling (learnable)
        role_weights:   Per-vertex role weights (5 floats). None uses defaults.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        crystal_dim: int = 128,
        gate_dim: int = 0,
        temperature: float = 0.07,
        role_weights: Optional[list] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.crystal_dim = crystal_dim
        self.gate_dim = gate_dim

        # Projection: features (+ optional gates) → crystal space
        proj_input_dim = feature_dim + gate_dim
        self.proj = nn.Sequential(
            nn.Linear(proj_input_dim, crystal_dim),
            nn.GELU(),
            nn.Linear(crystal_dim, crystal_dim),
        )

        # Pentachoron crystals: (C, 5, crystal_dim)
        crystals = self._init_pentachora(num_classes, crystal_dim)
        self.crystals = nn.Parameter(crystals)

        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Role weights (frozen)
        rw = role_weights if role_weights is not None else DEFAULT_ROLE_WEIGHTS
        self.register_buffer("role_weights", torch.tensor(rw, dtype=torch.float32))

    @staticmethod
    def _init_pentachora(num_classes: int, dim: int) -> torch.Tensor:
        """
        Initialize pentachora with class-specific centroids and vertex spread.
        Each class gets 5 vertices forming a roughly regular pentachoron.
        """
        # Random base + class-specific centroid offset for separation
        crystals = torch.randn(num_classes, NUM_VERTICES, dim) * 0.1
        centroids = F.normalize(torch.randn(num_classes, 1, dim), dim=-1) * 0.5
        crystals = crystals + centroids

        # Per-vertex perturbation for internal pentachoron structure
        for v in range(NUM_VERTICES):
            crystals[:, v:v+1, :] += torch.randn(num_classes, 1, dim) * 0.1

        return crystals

    def forward(
        self,
        features: torch.Tensor,
        gate_vectors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features:     (B, T, D) token features — pooled internally
            gate_vectors: (B, T, G) optional gate vectors from SuperpositionPatchClassifier

        Returns:
            rose_scores: (B, C) per-class scores (role-weighted cosine)
            projected:   (B, crystal_dim) projected feature vector
        """
        # Pool over token dimension
        pooled = features.mean(dim=1)  # (B, D)

        # Concatenate gates if provided
        if gate_vectors is not None:
            gate_pooled = gate_vectors.mean(dim=1)  # (B, G)
            pooled = torch.cat([pooled, gate_pooled], dim=-1)  # (B, D+G)
        elif self.gate_dim > 0:
            # Gate dim declared but not provided — zero-fill for robustness
            B = pooled.shape[0]
            zeros = torch.zeros(B, self.gate_dim, device=pooled.device, dtype=pooled.dtype)
            pooled = torch.cat([pooled, zeros], dim=-1)

        # Project to crystal space
        projected = self.proj(pooled)                        # (B, crystal_dim)
        f_hat = F.normalize(projected, dim=-1)               # (B, crystal_dim)

        # Normalize crystal vertices
        c_hat = F.normalize(self.crystals, dim=-1)           # (C, 5, crystal_dim)

        # Cosine similarity to all vertices of all classes
        cos_sim = torch.einsum("bd,cvd->bcv", f_hat, c_hat)  # (B, C, 5)

        # Role-weighted scoring
        rose_scores = (cos_sim * self.role_weights.view(1, 1, 5)).sum(dim=-1)  # (B, C)
        rose_scores = rose_scores / self.temperature.clamp(min=0.01)

        return rose_scores, projected

    def crystal_diagnostics(self) -> Dict[str, float]:
        """Returns crystal geometry stats (call in eval, no_grad)."""
        with torch.no_grad():
            c_hat = F.normalize(self.crystals.data, dim=-1)

            # Intra-class: vertex spread within each pentachoron
            intra_sim = torch.einsum("cvd,cwd->cvw", c_hat, c_hat)
            tri_mask = torch.triu(torch.ones(5, 5, device=c_hat.device), diagonal=1).bool()
            intra_vals = intra_sim[:, tri_mask]

            # Inter-class: centroid separation
            centroids = F.normalize(c_hat.mean(dim=1), dim=-1)
            inter_sim = torch.matmul(centroids, centroids.T)
            inter_mask = torch.triu(
                torch.ones(self.num_classes, self.num_classes, device=c_hat.device),
                diagonal=1,
            ).bool()
            inter_vals = inter_sim[inter_mask]

        return {
            "intra_cos_mean": intra_vals.mean().item(),
            "intra_cos_std": intra_vals.std().item(),
            "collapsed_count": (intra_vals.mean(dim=-1) > 0.95).sum().item(),
            "inter_cos_mean": inter_vals.mean().item(),
            "inter_cos_max": inter_vals.max().item(),
            "too_close_count": (inter_vals > 0.5).sum().item(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Rose Loss
# ══════════════════════════════════════════════════════════════════════════════

class RoseLoss(nn.Module):
    """
    Margin-based contrastive loss for pentachoron crystal classification.

    For each sample, pushes the weakest positive class score above the
    strongest negative class score by a margin. Naturally multi-label:
    operates on per-class scores independently.

    Components:
        1. Rose margin:       min_pos > max_neg + margin
        2. Collapse penalty:  intra-class vertex cosine < collapse_thresh
        3. Separation:        inter-class centroid cosine < sep_thresh
        4. CM passthrough:    optional backbone geometry loss

    Args:
        margin:           Score margin between weakest positive and strongest negative
        cm_weight:        Weight for backbone CM loss passthrough
        sep_weight:       Weight for inter-class separation penalty
        collapse_weight:  Weight for intra-class collapse penalty
        sep_thresh:       Cosine threshold for inter-class separation (default 0.3)
        collapse_thresh:  Cosine threshold for intra-class collapse (default 0.8)
    """

    def __init__(
        self,
        margin: float = 0.3,
        cm_weight: float = 0.1,
        sep_weight: float = 0.1,
        collapse_weight: float = 0.05,
        sep_thresh: float = 0.3,
        collapse_thresh: float = 0.8,
    ):
        super().__init__()
        self.margin = margin
        self.cm_weight = cm_weight
        self.sep_weight = sep_weight
        self.collapse_weight = collapse_weight
        self.sep_thresh = sep_thresh
        self.collapse_thresh = collapse_thresh

    def forward(
        self,
        rose_scores: torch.Tensor,        # (B, C)
        labels: torch.Tensor,             # (B, C) multi-hot
        crystals: torch.Tensor,           # (C, 5, D)
        cm_loss: Optional[torch.Tensor] = None,  # scalar from backbone
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B, C = rose_scores.shape
        device = rose_scores.device

        # ── Rose margin loss ──
        pos_mask = labels.bool()
        neg_mask = ~pos_mask

        pos_scores = rose_scores.clone()
        pos_scores[neg_mask] = float("inf")
        min_pos = pos_scores.min(dim=-1).values

        neg_scores = rose_scores.clone()
        neg_scores[pos_mask] = float("-inf")
        max_neg = neg_scores.max(dim=-1).values

        rose = F.relu(self.margin - (min_pos - max_neg))
        has_pos = pos_mask.any(dim=-1)
        rose_loss = rose[has_pos].mean() if has_pos.any() else torch.tensor(0.0, device=device)

        # ── Crystal collapse penalty ──
        c_hat = F.normalize(crystals, dim=-1)
        intra_sim = torch.einsum("cvd,cwd->cvw", c_hat, c_hat)
        tri_mask = torch.triu(torch.ones(5, 5, device=device), diagonal=1).bool()
        intra_vals = intra_sim[:, tri_mask]
        collapse_loss = F.relu(intra_vals - self.collapse_thresh).pow(2).mean()

        # ── Inter-class separation ──
        centroids = F.normalize(c_hat.mean(dim=1), dim=-1)
        inter_sim = torch.matmul(centroids, centroids.T)
        inter_mask = torch.triu(torch.ones(C, C, device=device), diagonal=1).bool()
        sep_loss = F.relu(inter_sim[inter_mask] - self.sep_thresh).pow(2).mean()

        # ── CM passthrough ──
        cm = cm_loss if cm_loss is not None else torch.tensor(0.0, device=device)

        # ── Total ──
        total = (
            rose_loss
            + self.cm_weight * cm
            + self.sep_weight * sep_loss
            + self.collapse_weight * collapse_loss
        )

        # ── Metrics ──
        with torch.no_grad():
            preds = (rose_scores > 0).float()
            tp = (preds * labels).sum()
            fp = (preds * (1 - labels)).sum()
            fn = ((1 - preds) * labels).sum()
            precision = tp / (tp + fp).clamp(min=1)
            recall = tp / (tp + fn).clamp(min=1)
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

        return total, {
            "total": total.detach(),
            "rose": rose_loss.detach(),
            "cm": cm.detach(),
            "sep": sep_loss.detach(),
            "collapse": collapse_loss.detach(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Self-Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    NUM_CLS = 38
    FEAT_DIM = 64
    CRYSTAL_DIM = 128

    print("=== CrystalSuperpositionHead Self-Test ===\n")

    # ── Test 1: Raw tokens (Patchwork mode, no gates) ──
    print("--- Mode 1: Raw tokens (gate_dim=0) ---")
    head = CrystalSuperpositionHead(
        feature_dim=FEAT_DIM, num_classes=NUM_CLS, crystal_dim=CRYSTAL_DIM, gate_dim=0,
    )
    tokens = torch.randn(4, 256, FEAT_DIM)
    scores, proj = head(tokens)
    print(f"  tokens {tokens.shape} → scores {scores.shape}, proj {proj.shape}")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Params: {sum(p.numel() for p in head.parameters()):,}")

    # ── Test 2: Tokens + gates (SuperpositionPatchClassifier mode) ──
    print("\n--- Mode 2: Tokens + 17-dim gates (gate_dim=17) ---")
    head_gated = CrystalSuperpositionHead(
        feature_dim=128, num_classes=NUM_CLS, crystal_dim=CRYSTAL_DIM, gate_dim=17,
    )
    patch_feats = torch.randn(4, 64, 128)
    gate_vecs = torch.randn(4, 64, 17)
    scores_g, proj_g = head_gated(patch_feats, gate_vecs)
    print(f"  feats {patch_feats.shape} + gates {gate_vecs.shape} → scores {scores_g.shape}")
    print(f"  Score range: [{scores_g.min():.3f}, {scores_g.max():.3f}]")
    print(f"  Params: {sum(p.numel() for p in head_gated.parameters()):,}")

    # ── Test 3: Gates declared but not passed (zero-fill fallback) ──
    print("\n--- Mode 3: Gates declared but not passed ---")
    scores_ng, _ = head_gated(patch_feats, gate_vectors=None)
    print(f"  scores {scores_ng.shape} (zero-filled gates)")

    # ── Test 4: Rose loss ──
    print("\n--- Rose Loss ---")
    criterion = RoseLoss(margin=0.3)
    labels = torch.zeros(4, NUM_CLS)
    for i in range(4):
        pos_idx = torch.randint(0, NUM_CLS, (3,))
        labels[i, pos_idx] = 1.0

    loss, info = criterion(scores, labels, head.crystals)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Rose: {info['rose']:.4f}, Sep: {info['sep']:.4f}, Collapse: {info['collapse']:.4f}")
    print(f"  F1: {info['f1']:.4f}, Prec: {info['precision']:.4f}, Rec: {info['recall']:.4f}")

    # ── Test 5: Gradient flow ──
    print("\n--- Gradient Flow ---")
    loss.backward()
    crystal_grad = head.crystals.grad
    has_grad = crystal_grad is not None and crystal_grad.abs().sum() > 0
    print(f"  Crystal gradient: {'✓' if has_grad else '✗'}")
    print(f"  Crystal grad norm: {crystal_grad.norm():.6f}" if has_grad else "")

    # ── Test 6: Crystal diagnostics ──
    print("\n--- Crystal Diagnostics ---")
    diag = head.crystal_diagnostics()
    for k, v in diag.items():
        print(f"  {k}: {v}")

    print("\n✓ crystal_superposition.py operational")