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

Fully vectorized — no Python loops in forward pass.

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
# BLEND MODES -- vectorized for (B, S, S, P, D) pairwise operation
# =============================================================================
# All: (a, b, alpha) -> blended
#   a, b:    (..., D)
#   alpha:   (...) or (..., 1) in [0, 1]
#   returns: (..., D)

def blend_lerp(a: Tensor, b: Tensor, alpha: Tensor) -> Tensor:
    """Linear interpolation."""
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
    """Entropic phase gating. Content-dependent blend."""
    if alpha.dim() < a.dim():
        alpha = alpha.unsqueeze(-1)
    delta = b - a
    gate = torch.sigmoid((delta * b).sum(dim=-1, keepdim=True))
    return a + (alpha * gate) * delta


BLEND_MODES: Dict[str, Callable] = {
    "lerp": blend_lerp,
    "slerp": blend_slerp,
    "shiva": blend_shiva,
    "gilgamesh": blend_gilgamesh,
    "slip": blend_slip,
}


# =============================================================================
# SCHEDULES
# =============================================================================

def schedule_linear(n: int, device: torch.device = None) -> Tensor:
    return torch.linspace(0, 1, n, device=device)

def schedule_cosine(n: int, device: torch.device = None) -> Tensor:
    t = torch.linspace(0, 1, n, device=device)
    return (1 - torch.cos(t * math.pi)) / 2

def schedule_tau(n: int, device: torch.device = None) -> Tensor:
    tau_val = (1 + math.sqrt(5)) / 2
    indices = torch.arange(n, device=device, dtype=torch.float32)
    return ((indices / tau_val) % 1.0).sort().values

def schedule_from_topology(depth_positions: Tensor) -> Tensor:
    return depth_positions

SCHEDULES: Dict[str, Callable] = {
    "linear": schedule_linear,
    "cosine": schedule_cosine,
    "tau": schedule_tau,
}


# =============================================================================
# AGGREGATIONS
# =============================================================================

def agg_mean(walked: Tensor, **kwargs) -> Tensor:
    """(B, S, P, D) -> (B, P, D)"""
    return walked.mean(dim=1)


def agg_weighted(walked: Tensor, weights: Tensor = None, **kwargs) -> Tensor:
    """(B, S, P, D) x (S,) -> (B, P, D)"""
    if weights is None:
        return walked.mean(dim=1)
    w = F.softmax(weights.view(1, -1, 1, 1), dim=1)
    return (walked * w).sum(dim=1)


def agg_similarity_tree(walked: Tensor, boundary_bias: Tensor = None, **kwargs) -> Tensor:
    """Consensus + unique residual guided by boundary_bias.

    walked:        (B, S, P, D)
    boundary_bias: (S, S)
    -> (B, P, D)
    """
    if boundary_bias is None:
        return walked.mean(dim=1)

    B, S, P, D = walked.shape

    # Off-diagonal importance: how much each state is pulled by others
    off_diag = boundary_bias.clone()
    off_diag.fill_diagonal_(0.0)
    state_importance = off_diag.sum(dim=1)                    # (S,)
    merge_weights = F.softmax(state_importance, dim=0)        # (S,)

    # Consensus: importance-weighted sum
    consensus = (walked * merge_weights.view(1, S, 1, 1)).sum(dim=1)  # (B, P, D)

    # Residual: unique perspectives add their deviations
    uniqueness = 1.0 - merge_weights                          # (S,)
    deviations = walked - consensus.unsqueeze(1)              # (B, S, P, D)
    unique_residual = (deviations * uniqueness.view(1, S, 1, 1)).sum(dim=1)

    return consensus + unique_residual


AGGREGATIONS: Dict[str, Callable] = {
    "mean": agg_mean,
    "weighted": agg_weighted,
    "similarity_tree": agg_similarity_tree,
}


# =============================================================================
# MASK CONDITIONING -- fully vectorized across all states
# =============================================================================

class MaskConditioner(nn.Module):
    """Topology masks -> feature-space gates, applied to all states at once.

    'direct': 0 params. Precomputes (S, D) gate matrix from (S, M) masks.
    'learned': Linear(M, D) shared across states.
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
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _build_group_map(mask_dim: int, feature_dim: int) -> Tensor:
        group_size = feature_dim // mask_dim
        remainder = feature_dim % mask_dim
        mapping = torch.zeros(feature_dim, dtype=torch.long)
        idx = 0
        for m in range(mask_dim):
            size = group_size + (1 if m < remainder else 0)
            mapping[idx:idx + size] = m
            idx += size
        return mapping

    def forward(self, features: Tensor, masks: Tensor) -> Tensor:
        """Apply all mask gates at once.

        features: (B, S, P, D)
        masks:    (S, M)
        returns:  (B, S, P, D) -- gated
        """
        if self.mode == "learned":
            gates = torch.sigmoid(self.proj(masks))       # (S, D)
        else:
            gates = torch.sigmoid(masks[:, self.group_map])  # (S, D)
        # Broadcast: (1, S, 1, D) * (B, S, P, D)
        return features * gates.unsqueeze(0).unsqueeze(2)


# =============================================================================
# TOPOLOGY WALKER -- the thousand in one, fully vectorized
# =============================================================================

class TopologyWalker(nn.Module):
    """Field walking fusion across simultaneous topology perspectives.

    All states fire. No routing. Every perspective lives.
    Fully vectorized -- zero Python loops in forward.

    Replaces: NeedsBasedRouter + CrossStateComposition
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

    def forward(
        self,
        state_outputs: Tensor,
        masks: Tensor,
        boundary_bias: Tensor,
        depth_positions: Tensor,
    ) -> Tensor:
        """Walk the topology field. Fully vectorized.

        Args:
            state_outputs:   (B, S, P, D)
            masks:           (S, M)
            boundary_bias:   (S, S)
            depth_positions: (S,)

        Returns:
            (B, P, D)
        """
        B, S, P, D = state_outputs.shape

        # -- Step 1: Mask conditioning (vectorized over all states) --
        # (B, S, P, D) * (1, S, 1, D) -> (B, S, P, D)
        conditioned = self.conditioner(state_outputs, masks)

        # -- Step 2: Pairwise topology-weighted blending (fully vectorized) --
        # Compute alpha matrix
        if self.schedule_name == "topology":
            alphas = boundary_bias                             # (S, S)
        else:
            sched = SCHEDULES[self.schedule_name](S, device=state_outputs.device)
            alphas = 1.0 - (sched.unsqueeze(0) - sched.unsqueeze(1)).abs()

        # Blend weights: row-normalized off-diagonal
        blend_w = alphas.clone()
        blend_w.fill_diagonal_(0.0)
        blend_w = blend_w / blend_w.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (S, S)

        # Expand for all-pairs blending:
        #   own:   (B, S, 1, P, D) -- each state's conditioned output
        #   other: (B, 1, S, P, D) -- all states as blend targets
        own   = conditioned.unsqueeze(2)                       # (B, S, 1, P, D)
        other = conditioned.unsqueeze(1)                       # (B, 1, S, P, D)

        # Expand own to (B, S, S, P, D) for broadcast with other
        own_expanded = own.expand(B, S, S, P, D)
        other_expanded = other.expand(B, S, S, P, D)

        # Alpha for each (i, j) pair: (S, S) -> (1, S, S, 1, 1)
        alpha_ij = alphas.view(1, S, S, 1, 1)

        # Blend all pairs at once: (B, S, S, P, D)
        blended = self.blend_fn(own_expanded, other_expanded, alpha_ij)

        # Weight by blend_w and sum over j (source states)
        # blend_w: (S, S) -> (1, S, S, 1, 1), zero diagonal already
        w = blend_w.view(1, S, S, 1, 1)
        contributions = (blended * w).sum(dim=2)               # (B, S, P, D)

        # Final walked = own + weighted blend from others
        walked = conditioned + contributions                    # (B, S, P, D)

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
# PRESETS
# =============================================================================

WALKER_PRESETS: Dict[str, Dict[str, str]] = {
    "alucard": {"blend_mode": "lerp", "schedule": "topology", "aggregation": "mean"},
    "shiva": {"blend_mode": "shiva", "schedule": "topology", "aggregation": "similarity_tree"},
    "gilgamesh": {"blend_mode": "gilgamesh", "schedule": "topology", "aggregation": "weighted"},
    "slip": {"blend_mode": "slip", "schedule": "cosine", "aggregation": "similarity_tree"},
    "slerp": {"blend_mode": "slerp", "schedule": "linear", "aggregation": "weighted"},
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
    """Create TopologyWalker from named preset."""
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

    print(f"\n{'=' * 60}")
    print("MASK CONDITIONER (vectorized)")
    print(f"{'=' * 60}")
    for mode in ["direct", "learned"]:
        cond = MaskConditioner(M, D, mode=mode).to(device)
        n_params = sum(p.numel() for p in cond.parameters())
        gated = cond(state_outputs, masks)
        print(f"  {mode:10s}: {state_outputs.shape} -> {gated.shape}, params={n_params}")

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
    print(f"  grad exists: {state_outputs_g.grad is not None}")
    print(f"  grad shape:  {state_outputs_g.grad.shape}")
    if walker.state_weights is not None:
        print(f"  state_weights grad: {walker.state_weights.grad}")

    print(f"\n{'=' * 60}")
    print("MEMORY: (B, S, S, P, D) expansion")
    print(f"{'=' * 60}")
    elem = B * S * S * P * D * 4  # float32
    print(f"  B={B}, S={S}, P={P}, D={D}")
    print(f"  Pairwise tensor: {elem / 1024 / 1024:.1f} MB")
    # At training scale: B=32
    elem32 = 32 * S * S * P * D * 4
    print(f"  B=32 training:   {elem32 / 1024 / 1024:.1f} MB")

    print(f"\n  walker.py self-test complete")