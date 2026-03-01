"""
Topology — Seed-Deterministic Topological Engine
==================================================

Universal topology that accepts arbitrary projection functions as
alpha mask generators. Everything precomputed on init, everything
regenerable from (num_states, num_levels, seed, projection).

No stored tensors needed — rebuild from recipe on any device.

Projections:
    A projection maps positions in [0,1]² to alpha masks.
    MobiusLens: tri-wave interference (horizontal + vertical basis)
    Cantor:     Devil's Staircase branch paths
    Any callable: positions → masks

Precomputed per state:
    - position_2d:    (2,)  — [depth_t, width_t] in continuum
    - scale_window:   (2,)  — [low, high] from scale range
    - deform_scale:   (,)   — deformation budget from deform range
    - twist_angle:    (,)   — depth-derived rotation
    - alpha_mask:     (M,)  — projection output at this position
    - thirds_idx:     (,)   — which third (0,1,2) for channel mask

Precomputed global:
    - alignment_matrix: (N, N) — pairwise topology similarity
    - positions:        (N,)   — 1D positions (compat with router)
    - positions_2d:     (N, 2) — full 2D addresses

Scale hierarchy:
    Patchwork:  8 states, 1 level (prototype)
    Chunk:      512 states, N levels (production)
    Ensemble:   B branches × S states × L levels (full)

Author: AbstractPhil + Claude
License: Apache-2.0
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List, Callable, Protocol
from abc import ABC, abstractmethod


# ══════════════════════════════════════════════════════════════════════════════
# Projection Protocol
# ══════════════════════════════════════════════════════════════════════════════

class TopologyProjection(ABC):
    """
    Base class for topology alpha mask generators.

    A projection maps 2D positions in [0,1]² to alpha masks.
    The masks encode the topological "character" at each position.
    Alignment between states = similarity of their masks.
    """

    @abstractmethod
    def mask_dim(self) -> int:
        """Dimensionality of output alpha masks."""
        ...

    @abstractmethod
    def __call__(
        self, depth_t: Tensor, width_t: Tensor, gen: torch.Generator
    ) -> Tensor:
        """
        Args:
            depth_t: (N,) positions in [0,1] along depth axis
            width_t: (N,) positions in [0,1] along width axis
            gen:     seeded generator for deterministic randomness
        Returns:
            masks: (N, mask_dim) alpha masks per position
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# MobiusProjection — tri-wave interference (horizontal + vertical)
# ══════════════════════════════════════════════════════════════════════════════

class MobiusProjection(TopologyProjection):
    """
    Dual-step tri-wave interference topology.

    Horizontal basis: 3 waves at different orientations → fractal gating
    Vertical basis:   3 waves clustered near π/2 → vertical slicing
    Combined:         depth controls H/V blend, width controls phase offset

    Alpha mask = [H_L, H_M, H_R, V_L, V_M, V_R, H_fused, V_fused, combined]

    From the Feb 27 dual-step artifact + Jan 8 MobiusLens formulas.
    scale_range (0.25, 2.75) — all useful scales.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.25, 2.75),
        omega: float = math.pi,
        alpha_base: float = 1.5,
    ):
        self.scale_range = scale_range
        self.omega = omega
        self.alpha_base = alpha_base

    def mask_dim(self) -> int:
        return 9  # H_L, H_M, H_R, V_L, V_M, V_R, H_fused, V_fused, combined

    def _tri_wave(
        self, t: Tensor, scale: Tensor, phase_offset: Tensor, gen: torch.Generator
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Three waves (L/M/R) at given scale with phase offset."""
        omega = self.omega

        # Deterministic wave parameters from generator
        freqs = torch.tensor([1.0, 1.5, 2.0]) * scale.unsqueeze(-1)
        phases = torch.tensor([0.0, math.pi / 3, 2 * math.pi / 3])
        phases = phases + phase_offset.unsqueeze(-1)

        # Wave computation: gate = exp(-alpha * sin²(freq * omega * t + phase))
        positions = freqs * omega * t.unsqueeze(-1) + phases
        waves = torch.sin(positions)
        alpha = self.alpha_base / scale.unsqueeze(-1).clamp(min=0.1)
        gates = torch.exp(-alpha * waves.pow(2))

        L, M, R = gates[..., 0], gates[..., 1], gates[..., 2]
        return L, M, R

    def _fuse_lmr(self, L: Tensor, M: Tensor, R: Tensor) -> Tensor:
        """Cantor-style XOR/AND fusion of L/M/R waves."""
        xor_comp = (L + R - 2 * L * R).abs()
        and_comp = L * R
        lr = 0.6 * xor_comp + 0.4 * and_comp
        fused = (L + M + R) / 3.0
        fused = fused * (0.5 + 0.5 * lr)
        return fused

    def __call__(
        self, depth_t: Tensor, width_t: Tensor, gen: torch.Generator
    ) -> Tensor:
        N = depth_t.shape[0]
        s_lo, s_hi = self.scale_range
        scale_span = s_hi - s_lo

        # Depth determines scale, width determines phase offset
        scale = s_lo + depth_t * scale_span
        phase_offset = width_t * math.pi  # 0 to π across width

        # Horizontal basis (depth-aligned)
        H_L, H_M, H_R = self._tri_wave(depth_t, scale, phase_offset, gen)
        H_fused = self._fuse_lmr(H_L, H_M, H_R)

        # Vertical basis (width-aligned, rotated π/2)
        V_L, V_M, V_R = self._tri_wave(width_t, scale, phase_offset + math.pi / 2, gen)
        V_fused = self._fuse_lmr(V_L, V_M, V_R)

        # Combined: depth controls H/V blend
        combined = depth_t * H_fused + (1 - depth_t) * V_fused

        masks = torch.stack([
            H_L, H_M, H_R, V_L, V_M, V_R, H_fused, V_fused, combined
        ], dim=-1)

        return masks


# ══════════════════════════════════════════════════════════════════════════════
# CantorProjection — Devil's Staircase (backward compatible)
# ══════════════════════════════════════════════════════════════════════════════

class CantorProjection(TopologyProjection):
    """
    Devil's Staircase branch path projection.
    Backward compatible with CantorTopology.

    Alpha mask = branch path softmax probabilities flattened.
    Width axis ignored (1D topology).
    """

    def __init__(self, levels: int = 5, base: int = 3):
        self.levels = levels
        self.base = base

    def mask_dim(self) -> int:
        return self.levels * self.base  # levels × 3 = 15 for default

    def __call__(
        self, depth_t: Tensor, width_t: Tensor, gen: torch.Generator
    ) -> Tensor:
        x = depth_t.to(torch.float64).clamp(1e-10, 1.0 - 1e-10)
        centers = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        scales = torch.tensor(
            [self.base ** k for k in range(1, self.levels + 1)], dtype=torch.float64
        )
        tau = 0.25

        # (N, levels) → mod base → distance to centers → softmax
        y_all = (x.unsqueeze(-1) * scales) % self.base  # (N, levels)
        d2 = (y_all.unsqueeze(-1) - centers) ** 2       # (N, levels, 3)
        p = F.softmax(-d2 / tau, dim=-1)                 # (N, levels, 3)

        # Flatten to (N, levels * 3)
        return p.float().reshape(x.shape[0], -1)


# ══════════════════════════════════════════════════════════════════════════════
# Topology — the universal engine
# ══════════════════════════════════════════════════════════════════════════════

class Topology:
    """
    Seed-deterministic topological engine.

    Accepts any TopologyProjection, precomputes all constants on init.
    Everything regenerable from (num_states, num_levels, seed, projection).
    No stored state needed beyond the recipe.

    Usage:
        # Create with MobiusLens projection
        topo = Topology(num_states=8, projection=MobiusProjection())

        # Or with Cantor (backward compatible)
        topo = Topology(num_states=8, projection=CantorProjection())

        # Access precomputed constants
        topo.positions_2d      # (N, 2) [depth, width]
        topo.scale_windows     # (N, 2) [low, high]
        topo.deform_scales     # (N,)
        topo.twist_angles      # (N,)
        topo.alpha_masks       # (N, mask_dim)
        topo.alignment_matrix  # (N, N)
        topo.thirds_indices    # (N,) which third (0,1,2)

        # Per-state access
        topo.get_state_constants(state_idx) → dict of buffers

        # Rebuild on different device
        topo2 = Topology.from_recipe(topo.recipe)
    """

    def __init__(
        self,
        num_states: int = 8,
        num_levels: int = 1,
        num_branches: int = 1,
        seed: int = 42,
        projection: Optional[TopologyProjection] = None,
        scale_range: Tuple[float, float] = (0.25, 2.75),
        deform_range: Tuple[float, float] = (0.1, 0.25),
    ):
        self.num_states = num_states
        self.num_levels = num_levels
        self.num_branches = num_branches
        self.seed = seed
        self.projection = projection or MobiusProjection(scale_range=scale_range)
        self.scale_range = scale_range
        self.deform_range = deform_range

        # Store recipe for reconstruction
        self.recipe = {
            "num_states": num_states,
            "num_levels": num_levels,
            "num_branches": num_branches,
            "seed": seed,
            "projection_class": type(self.projection).__name__,
            "scale_range": scale_range,
            "deform_range": deform_range,
        }

        # Precompute everything
        self._build()

    def _build(self):
        """Deterministic precomputation from seed. No randomness outside gen."""
        gen = torch.Generator().manual_seed(self.seed)
        N = self.num_states

        # ── 2D positions in [0,1]² ──
        # Depth: evenly spaced across states
        # Width: derived from state index and num_branches
        depth_t = torch.linspace(0, 1, N + 2)[1:-1]  # exclude endpoints

        if self.num_branches > 1:
            # Grid: states distributed across depth × width
            n_depth = N // self.num_branches
            n_width = self.num_branches
            d = torch.linspace(0, 1, n_depth + 2)[1:-1]
            w = torch.linspace(0, 1, n_width + 2)[1:-1]
            grid_d, grid_w = torch.meshgrid(d, w, indexing="ij")
            depth_t = grid_d.reshape(-1)[:N]
            width_t = grid_w.reshape(-1)[:N]
        else:
            # Single branch: width from deterministic hash of position
            width_t = torch.zeros(N)
            for i in range(N):
                # Deterministic width from seed + state_idx
                state_gen = torch.Generator().manual_seed(self.seed * 7919 + i)
                width_t[i] = torch.rand(1, generator=state_gen).item()

        self.positions_2d = torch.stack([depth_t, width_t], dim=-1)  # (N, 2)
        self.positions = depth_t  # 1D compat for router

        # ── Alpha masks from projection ──
        self.alpha_masks = self.projection(depth_t, width_t, gen)  # (N, mask_dim)
        self.mask_dim = self.projection.mask_dim()

        # ── Scale windows: adjacent continuous ──
        s_lo, s_hi = self.scale_range
        scale_span = s_hi - s_lo
        step = scale_span / max(N - 1, 1)
        scale_low = s_lo + depth_t * scale_span
        scale_high = (scale_low + step).clamp(max=s_hi)
        self.scale_windows = torch.stack([scale_low, scale_high], dim=-1)  # (N, 2)

        # ── Deformation tolerance: shallow=tight, deep=loose ──
        d_lo, d_hi = self.deform_range
        self.deform_scales = d_lo + depth_t * (d_hi - d_lo)  # (N,)

        # ── Twist angles: 0 → π across depth ──
        self.twist_angles = depth_t * math.pi  # (N,)

        # ── Thirds indices: rotating channel mask ──
        self.thirds_indices = torch.arange(N) % 3  # (N,) int

        # ── Branch paths (for backward compat and identity) ──
        # Quantize alpha masks to discrete paths
        M = self.mask_dim
        # Use top-k mask indices as branch path
        if M >= 5:
            _, topk = self.alpha_masks.topk(5, dim=-1)
            self.branch_paths = topk.int()  # (N, 5)
        else:
            self.branch_paths = self.alpha_masks.argsort(dim=-1, descending=True).int()

        # ── Staircase values (for backward compat) ──
        # Weighted sum of alpha mask as scalar identity
        weights = torch.arange(1, M + 1, dtype=torch.float32) / M
        self.staircase_vals = (self.alpha_masks * weights).sum(dim=-1)  # (N,)

        # ── Alignment matrix: cosine similarity of alpha masks ──
        self.alignment_matrix = self._compute_alignment()

        # ── Per-level constants (if multi-level) ──
        if self.num_levels > 1:
            self._build_level_constants()

    def _compute_alignment(self) -> Tensor:
        """Pairwise alignment from alpha mask cosine similarity."""
        masks = self.alpha_masks  # (N, M)
        norms = masks.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = masks / norms
        alignment = torch.mm(normed, normed.t())  # (N, N)
        return alignment

    def _build_level_constants(self):
        """
        Per-level topology for depth progression.

        Level 0 = shallowest (fine structure, tight deform)
        Level L-1 = deepest (coarse structure, loose deform)

        Each level gets its own scale window and deform budget,
        creating a timestep-like progression through topology space.
        """
        L = self.num_levels
        N = self.num_states
        gen = torch.Generator().manual_seed(self.seed + 31337)

        level_t = torch.linspace(0, 1, L)  # (L,)

        # Scale windows per level
        s_lo, s_hi = self.scale_range
        span = s_hi - s_lo
        step = span / max(L, 1)
        level_scale_low = s_lo + level_t * span
        level_scale_high = (level_scale_low + step).clamp(max=s_hi)
        self.level_scale_windows = torch.stack(
            [level_scale_low, level_scale_high], dim=-1)  # (L, 2)

        # Deform budget per level
        d_lo, d_hi = self.deform_range
        self.level_deform_scales = d_lo + level_t * (d_hi - d_lo)  # (L,)

        # Twist per level
        self.level_twist_angles = level_t * math.pi  # (L,)

        # Alpha masks per level (project at level depth positions)
        width_t = torch.full((L,), 0.5)  # center width for level defaults
        self.level_alpha_masks = self.projection(
            level_t, width_t, gen)  # (L, mask_dim)

    # ══════════════════════════════════════════════════════════════════════
    # Access API
    # ══════════════════════════════════════════════════════════════════════

    def get_state_constants(self, state_idx: int) -> Dict[str, Tensor]:
        """All topology constants for a single state, ready to register as buffers."""
        return {
            "topo_position_2d": self.positions_2d[state_idx],       # (2,)
            "topo_scale_window": self.scale_windows[state_idx],     # (2,)
            "topo_deform_scale": self.deform_scales[state_idx],     # scalar
            "topo_twist_angle": self.twist_angles[state_idx],       # scalar
            "topo_alpha_mask": self.alpha_masks[state_idx],         # (mask_dim,)
            "topo_thirds_idx": self.thirds_indices[state_idx],      # int
            "topo_branch_path": self.branch_paths[state_idx],       # (5,)
            "topo_staircase_val": self.staircase_vals[state_idx],   # scalar
        }

    def get_level_constants(self, level_idx: int) -> Dict[str, Tensor]:
        """Topology constants for a depth level."""
        if not hasattr(self, "level_scale_windows"):
            raise ValueError("Multi-level not configured (num_levels=1)")
        return {
            "level_scale_window": self.level_scale_windows[level_idx],
            "level_deform_scale": self.level_deform_scales[level_idx],
            "level_twist_angle": self.level_twist_angles[level_idx],
            "level_alpha_mask": self.level_alpha_masks[level_idx],
        }

    def get_state_identity(self, state_idx: int) -> Dict:
        """Human-readable identity for debugging."""
        return {
            "index": state_idx,
            "position_2d": self.positions_2d[state_idx].tolist(),
            "scale_window": self.scale_windows[state_idx].tolist(),
            "deform_scale": self.deform_scales[state_idx].item(),
            "twist_angle": self.twist_angles[state_idx].item(),
            "thirds_idx": self.thirds_indices[state_idx].item(),
            "staircase_val": self.staircase_vals[state_idx].item(),
            "alpha_mask_norm": self.alpha_masks[state_idx].norm().item(),
        }

    def get_alignment(self, i: int, j: int) -> float:
        """Pairwise alignment between two states."""
        return self.alignment_matrix[i, j].item()

    def get_neighborhood(self, state_idx: int, threshold: float = 0.5) -> List[int]:
        """States with alignment above threshold."""
        row = self.alignment_matrix[state_idx]
        return (row > threshold).nonzero(as_tuple=True)[0].tolist()

    # ══════════════════════════════════════════════════════════════════════
    # Reconstruction from recipe
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def from_recipe(cls, recipe: Dict) -> "Topology":
        """Rebuild topology from stored recipe. No saved tensors needed."""
        proj_class = recipe["projection_class"]
        proj_map = {
            "MobiusProjection": MobiusProjection,
            "CantorProjection": CantorProjection,
        }
        projection = proj_map[proj_class](
            **({"scale_range": recipe["scale_range"]}
               if proj_class == "MobiusProjection"
               else {"levels": 5})  # default Cantor levels
        )
        return cls(
            num_states=recipe["num_states"],
            num_levels=recipe["num_levels"],
            num_branches=recipe["num_branches"],
            seed=recipe["seed"],
            projection=projection,
            scale_range=recipe["scale_range"],
            deform_range=recipe["deform_range"],
        )

    def __repr__(self) -> str:
        proj_name = type(self.projection).__name__
        return (
            f"Topology(states={self.num_states}, levels={self.num_levels}, "
            f"branches={self.num_branches}, seed={self.seed}, "
            f"proj={proj_name}, mask_dim={self.mask_dim})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Backward compat alias
# ══════════════════════════════════════════════════════════════════════════════

def CantorTopology(
    num_states: int = 512, levels: int = 9, base: int = 3
) -> Topology:
    """
    Drop-in replacement for old CantorTopology.

    Returns a Topology with CantorProjection.
    Same interface: .positions, .staircase_vals, .branch_paths, .alignment_matrix
    """
    return Topology(
        num_states=num_states,
        num_levels=1,
        num_branches=1,
        seed=42,
        projection=CantorProjection(levels=levels, base=base),
        scale_range=(0.25, 2.75),
        deform_range=(0.1, 0.25),
    )