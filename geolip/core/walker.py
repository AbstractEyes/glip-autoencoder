"""
geolip.core.walker
==================

Topology-conditioned field walking fusion.

All states fire simultaneously — every perspective lives. The walker
defines how perspectives interfere through the topology field. Masks
condition each state's view. Blend modes control interpolation. The
boundary_bias matrix IS the walking distance between perspectives.

Based on the Alucard/FieldWalkerFusion architecture:
    BlendMode  x  Schedule  x  Aggregation
    (how)         (when)       (pool)

Design:
    - All states produce outputs in parallel (no routing, no killing)
    - Each state's output is mask-conditioned (9-dim -> feature gating)
    - Walker blends perspectives pairwise, weighted by boundary_bias
    - Aggregation pools across the walked field

    state_outputs: (B, S, P, D)  -- S simultaneous perspectives
    masks:         (S, M)         -- topology interference patterns
    boundary_bias: (S, S)         -- pairwise alignment / walking distance
    -> walked:     (B, P, D)      -- fused output

Copyright 2025-2026 AbstractPhil
MIT License
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# BLEND MODES -- how to combine two perspectives at a given alpha
# =============================================================================
# All: (a, b, alpha) -> blended
#   a, b:    (*, D)
#   alpha:   scalar or (*,) or (*, 1)  in [0, 1]
#   returns: (*, D)

def blend_lerp(a: Tensor, b: Tensor, alpha: Tensor) -> Tensor:
    """Linear interpolation. Standard baseline."""
    if alpha.dim() < a.dim():
        alpha = alpha.unsqueeze(-1)
    return a + alpha * (b - a)


def blend_slerp(a: Tensor, b: Tensor, alpha: Tensor) -> Tensor:
    """Spherical linear interpolation. Preserves magnitude."""
    if alpha.dim() < a.dim():
        alpha = alpha.unsqueeze(-1)
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    dot = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp(min=1e-7)
    safe = (omega.abs() > 1e-5).float()
    slerp_result = (torch.sin((1 - alpha) * omega) * a + torch.sin(alpha * omega) * b) / sin_omega
    lerp_result = a + alpha * (b - a)
    mag_a = a.norm(dim=-1, keepdim=True)
    mag_b = b.norm(dim=-1, keepdim=True)
    mag_out = mag_a + alpha * (mag_b - mag_a)
    result = safe * slerp_result + (1 - safe) * lerp_result
    return result * mag_out / result.norm(dim=-1, keepdim=True).clamp(min=1e-7)


def blend_shiva(a: Tensor, b: Tensor, alpha: Tensor, decay: float = 3.0) -> Tensor:
    """Exponential decay blend. Fast transition with long tail."""
    if alpha.dim() < a.dim():
        alpha = alpha.unsqueeze(-1)
    w = 1.0 - torch.exp(-decay * alpha)
    return (1 - w) * a + w * b


def blend_gilgamesh(a: Tensor, b: Tensor, alpha: Tensor) -> Tensor:
    """Energy-preserving cosine-squared blend. cos^2 + sin^2 = 1."""
    if alpha.dim() < a.dim():
        alpha = alpha.unsqueeze(-1)
    theta = alpha * (math.pi / 2)
    return torch.cos(theta) ** 2 * a + torch.sin(theta) ** 2 * b


def blend_slip(a: Tensor, b: Tensor, alpha: Tensor) -> Tensor:
    """Entropic phase gating. Content-dependent blend.

    Where perspectives agree: smooth blend.
    Where they disagree: sharp phase boundary.
    """
    if alpha.dim() < a.dim():
        alpha = alpha.unsqueeze(-1)
    delta = b - a
    gate = torch.sigmoid((delta * b).sum(dim=-1, keepdim=True))
    effective_alpha = alpha * gate
    return a + effective_alpha * delta


BLEND_MODES: Dict[str, Callable] = {
    "lerp": blend_lerp,
    "slerp": blend_slerp,
    "shiva": blend_shiva,
    "gilgamesh": blend_gilgamesh,
    "slip": blend_slip,
}


# =============================================================================
# SCHEDULES -- how alpha evolves across the topology
# =============================================================================

def schedule_linear(n: int, device: torch.device = None) -> Tensor:
    """Uniform spacing [0, 1]."""
    return torch.linspace(0, 1, n, device=device)


def schedule_cosine(n: int, device: torch.device = None) -> Tensor:
    """Cosine: slow-fast-slow progression."""
    t = torch.linspace(0, 1, n, device=device)
    return (1 - torch.cos(t * math.pi)) / 2


def schedule_tau(n: int, device: torch.device = None) -> Tensor:
    """Golden ratio (tau) spacing. Fibonacci-like quasi-uniform coverage."""
    tau_val = (1 + math.sqrt(5)) / 2
    indices = torch.arange(n, device=device, dtype=torch.float32)
    raw = (indices / tau_val) % 1.0
    return raw.sort().values


def schedule_from_topology(depth_positions: Tensor) -> Tensor:
    """Use topology depth positions directly as the schedule."""
    return depth_positions


SCHEDULES: Dict[str, Callable] = {
    "linear": schedule_linear,
    "cosine": schedule_cosine,
    "tau": schedule_tau,
}


# =============================================================================
# AGGREGATIONS -- how to pool walked perspectives into final output
# =============================================================================

def agg_mean(walked: Tensor, **kwargs) -> Tensor:
    """Simple mean across states. (B, S, P, D) -> (B, P, D)."""
    return walked.mean(dim=1)


def agg_weighted(walked: Tensor, weights: Tensor = None, **kwargs) -> Tensor:
    """Weighted sum. weights: (S,) or (B, S)."""
    if weights is None:
        return walked.mean(dim=1)
    if weights.dim() == 1:
        w = weights.view(1, -1, 1, 1)
    else:
        w = weights.unsqueeze(-1).unsqueeze(-1)
    w = F.softmax(w, dim=1)
    return (walked * w).sum(dim=1)


def agg_similarity_tree(
    walked: Tensor,
    boundary_bias: Tensor = None,
    **kwargs,
) -> Tensor:
    """Hierarchical merge guided by topology similarity.

    Consensus states (high alignment to many) weigh more.
    Unique states (low alignment) contribute residual.

    walked:        (B, S, P, D)
    boundary_bias: (S, S)
    -> (B, P, D)
    """
    B, S, P, D = walked.shape

    if boundary_bias is None:
        return walked.mean(dim=1)

    off_diag = boundary_bias.clone()
    off_diag.fill_diagonal_(0.0)

    # State importance = total alignment pull from others
    state_importance = off_diag.sum(dim=1)  # (S,)
    merge_weights = F.softmax(state_importance, dim=0)  # (S,)

    # Consensus: weighted sum by importance
    consensus = (walked * merge_weights.view(1, S, 1, 1)).sum(dim=1)  # (B, P, D)

    # Residual: unique perspectives contribute their deviations
    uniqueness = 1.0 - merge_weights
    deviations = walked - consensus.unsqueeze(1)
    unique_residual = (deviations * uniqueness.view(1, S, 1, 1)).sum(dim=1)

    return consensus + unique_residual


AGGREGATIONS: Dict[str, Callable] = {
    "mean": agg_mean,
    "weighted": agg_weighted,
    "similarity_tree": agg_similarity_tree,
}


# =============================================================================
# MASK CONDITIONING -- topology masks gate feature channels
# =============================================================================

class MaskConditioner(nn.Module):
    """Project topology masks into feature-space gates.

    'direct': zero params. Groups mask dims across feature channels.
    'learned': Linear(mask_dim, feature_dim) projection.
    """

    def __init__(self, mask_dim: int, feature_dim: int, mode: str = "direct"):
        super().__init__()
        self.mask_dim = mask_dim
        self.feature_dim = feature_dim
        self.mode = mode

        if mode == "learned":
            self.proj = nn.Linear(mask_dim, feature_dim)
        elif mode == "direct":
            self.register_buffer("group_map", self._build_group_map(mask_dim, feature_dim))
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'direct' or 'learned'.")

    @staticmethod
    def _build_group_map(mask_dim: int, feature_dim: int) -> Tensor:
        """Assign each feature channel to a mask dimension."""
        group_size = feature_dim // mask_dim
        remainder = feature_dim % mask_dim
        mapping = torch.zeros(feature_dim, dtype=torch.long)
        idx = 0
        for m in range(mask_dim):
            size = group_size + (1 if m < remainder else 0)
            mapping[idx:idx + size] = m
            idx += size
        return mapping

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        """Apply mask conditioning.

        features: (B, P, D) or (*, D)
        mask:     (M,) -- single state's topology mask
        returns:  same shape, gated
        """
        if self.mode == "learned":
            gate = torch.sigmoid(self.proj(mask))
        else:
            gate = torch.sigmoid(mask[self.group_map])
        return features * gate


# =============================================================================
# TOPOLOGY WALKER -- the thousand in one
# =============================================================================

class TopologyWalker(nn.Module):
    """Field walking fusion across simultaneous topology perspectives.

    All states fire. No routing. Every perspective lives. The walker
    defines interference through blend mode, schedule, and aggregation,
    all structured by the topology.

    Replaces: NeedsBasedRouter + CrossStateComposition

    Args:
        num_states:       Number of simultaneous perspectives (S)
        feature_dim:      Per-patch feature dimension (D)
        mask_dim:         Topology mask dimension (9 for Mobius)
        blend_mode:       'lerp', 'slerp', 'shiva', 'gilgamesh', 'slip'
        schedule:         'linear', 'cosine', 'tau', 'topology'
        aggregation:      'mean', 'weighted', 'similarity_tree'
        mask_mode:        'direct' (0 params) or 'learned' (small projection)
        learnable_weights: Learn per-state importance weights
    """

    def __init__(
        self,
        num_states: int,
        feature_dim: int,
        mask_dim: int,
        blend_mode: str = "shiva",
        schedule: str = "topology",
        aggregation: str = "similarity_tree",
        mask_mode: str = "direct",
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.mask_dim = mask_dim
        self.blend_name = blend_mode
        self.schedule_name = schedule
        self.aggregation_name = aggregation

        if blend_mode not in BLEND_MODES:
            raise ValueError(f"Unknown blend: {blend_mode}. Options: {list(BLEND_MODES)}")
        self.blend_fn = BLEND_MODES[blend_mode]

        self.conditioner = MaskConditioner(mask_dim, feature_dim, mode=mask_mode)

        if learnable_weights:
            self.state_weights = nn.Parameter(torch.zeros(num_states))
        else:
            self.state_weights = None

        self.output_norm = nn.LayerNorm(feature_dim)

    def _compute_alphas(
        self,
        boundary_bias: Tensor,
        depth_positions: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Compute pairwise blend alphas from topology.

        Returns: (S, S) alpha matrix.
        """
        if self.schedule_name == "topology":
            return boundary_bias
        else:
            sched = SCHEDULES[self.schedule_name](self.num_states, device=device)
            return 1.0 - (sched.unsqueeze(0) - sched.unsqueeze(1)).abs()

    def forward(
        self,
        state_outputs: Tensor,
        masks: Tensor,
        boundary_bias: Tensor,
        depth_positions: Tensor,
    ) -> Tensor:
        """Walk the topology field.

        Args:
            state_outputs:   (B, S, P, D) -- all state outputs
            masks:           (S, M) -- topology masks per state
            boundary_bias:   (S, S) -- pairwise alignment
            depth_positions: (S,) -- topology depth schedule

        Returns:
            (B, P, D) -- fused output
        """
        B, S, P, D = state_outputs.shape

        # -- Step 1: Mask conditioning --
        # Each state's features gated by its topology mask fingerprint.
        conditioned = torch.stack([
            self.conditioner(state_outputs[:, si], masks[si])
            for si in range(S)
        ], dim=1)  # (B, S, P, D)

        # -- Step 2: Pairwise topology-weighted blending --
        alphas = self._compute_alphas(boundary_bias, depth_positions, state_outputs.device)

        # Blend weights: row-normalized off-diagonal alphas
        blend_w = alphas.clone()
        blend_w.fill_diagonal_(0.0)
        blend_w = blend_w / blend_w.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (S, S)

        # Vectorized blending: for each state, blend with all others
        # conditioned[:, si] blended toward conditioned[:, sj] at alpha[si, sj]
        walked = torch.zeros_like(conditioned)
        for si in range(S):
            own = conditioned[:, si]  # (B, P, D)
            contributions = torch.zeros_like(own)
            for sj in range(S):
                if si == sj:
                    continue
                other = conditioned[:, sj]
                blended = self.blend_fn(own, other, alphas[si, sj])
                contributions = contributions + blend_w[si, sj] * blended
            walked[:, si] = own + contributions

        # -- Step 3: Aggregation --
        if self.aggregation_name == "similarity_tree":
            output = agg_similarity_tree(walked, boundary_bias)
        elif self.aggregation_name == "weighted":
            output = agg_weighted(walked, self.state_weights)
        else:
            output = agg_mean(walked)

        return self.output_norm(output)

    def extra_repr(self) -> str:
        return (
            f"states={self.num_states}, features={self.feature_dim}, "
            f"mask_dim={self.mask_dim}, blend={self.blend_name}, "
            f"schedule={self.schedule_name}, agg={self.aggregation_name}"
        )


# =============================================================================
# PRESETS -- named configs matching Alucard conventions
# =============================================================================

WALKER_PRESETS: Dict[str, Dict[str, str]] = {
    "alucard": {
        "blend_mode": "lerp",
        "schedule": "topology",
        "aggregation": "mean",
    },
    "shiva": {
        "blend_mode": "shiva",
        "schedule": "topology",
        "aggregation": "similarity_tree",
    },
    "gilgamesh": {
        "blend_mode": "gilgamesh",
        "schedule": "topology",
        "aggregation": "weighted",
    },
    "slip": {
        "blend_mode": "slip",
        "schedule": "cosine",
        "aggregation": "similarity_tree",
    },
    "slerp": {
        "blend_mode": "slerp",
        "schedule": "linear",
        "aggregation": "weighted",
    },
}


def from_preset(
    preset: str,
    num_states: int,
    feature_dim: int,
    mask_dim: int,
    mask_mode: str = "direct",
    learnable_weights: bool = False,
    **overrides,
) -> TopologyWalker:
    """Create TopologyWalker from named preset.

    # >>> walker = from_preset('shiva', num_states=8, feature_dim=75, mask_dim=9)
    """
    if preset not in WALKER_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(WALKER_PRESETS)}")
    config = WALKER_PRESETS[preset].copy()
    config.update(overrides)
    return TopologyWalker(
        num_states=num_states,
        feature_dim=feature_dim,
        mask_dim=mask_dim,
        mask_mode=mask_mode,
        learnable_weights=learnable_weights,
        **config,
    )


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    S, P, D, M = 8, 16, 75, 9
    B = 4

    # Simulate topology
    masks = torch.rand(S, M, device=device)
    masks_norm = F.normalize(masks, dim=-1)
    boundary_bias = masks_norm @ masks_norm.T
    depth_positions = torch.linspace(0.0625, 0.9375, S, device=device)
    state_outputs = torch.randn(B, S, P, D, device=device)

    print("=" * 60)
    print("BLEND MODES")
    print("=" * 60)
    a = torch.randn(B, P, D, device=device)
    b = torch.randn(B, P, D, device=device)
    alpha = torch.tensor(0.5, device=device)
    for name, fn in BLEND_MODES.items():
        out = fn(a, b, alpha)
        print(f"  {name:12s}: {out.shape}, norm={out.norm(dim=-1).mean():.4f}")

    print(f"\n{'=' * 60}")
    print("SCHEDULES")
    print(f"{'=' * 60}")
    for name, fn in SCHEDULES.items():
        sched = fn(S, device=device)
        print(f"  {name:12s}: [{', '.join(f'{v:.3f}' for v in sched.tolist())}]")
    topo_sched = schedule_from_topology(depth_positions)
    print(f"  {'topology':12s}: [{', '.join(f'{v:.3f}' for v in topo_sched.tolist())}]")

    print(f"\n{'=' * 60}")
    print("MASK CONDITIONER")
    print(f"{'=' * 60}")
    for mode in ["direct", "learned"]:
        cond = MaskConditioner(M, D, mode=mode).to(device)
        n_params = sum(p.numel() for p in cond.parameters())
        gated = cond(a, masks[0])
        print(f"  {mode:10s}: {gated.shape}, params={n_params}")

    print(f"\n{'=' * 60}")
    print("WALKER PRESETS")
    print(f"{'=' * 60}")
    for preset_name in WALKER_PRESETS:
        walker = from_preset(preset_name, S, D, M).to(device)
        n_params = sum(p.numel() for p in walker.parameters())
        out = walker(state_outputs, masks, boundary_bias, depth_positions)
        print(f"  {preset_name:12s}: {out.shape}, params={n_params:,}, "
              f"norm={out.norm(dim=-1).mean():.4f}")

    print(f"\n{'=' * 60}")
    print("GRADIENT CHECK")
    print(f"{'=' * 60}")
    walker = from_preset("shiva", S, D, M, learnable_weights=True).to(device)
    state_outputs_g = state_outputs.detach().requires_grad_(True)
    out = walker(state_outputs_g, masks, boundary_bias, depth_positions)
    loss = out.sum()
    loss.backward()
    print(f"  grad on state_outputs: {state_outputs_g.grad is not None}")
    print(f"  grad shape: {state_outputs_g.grad.shape}")
    if walker.state_weights is not None:
        print(f"  state_weights grad: {walker.state_weights.grad}")

    print(f"\n  walker.py self-test complete")