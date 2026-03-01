"""
Cayley-Menger Formula + Validator
==================================

Ports the robust, tested CM computations from geovocab2 into the
FormulaBase system. Fully differentiable in PyTorch mode.

Formula (FormulaBase):
    CayleyMengerFormula — stateless computation, returns result dict
        vertices → {volume, volume_sq, is_degenerate, cm_det, distances, edge_lengths, ...}

Module (nn.Module):
    CayleyMengerValidator — training losses (validity + consistency)
        vol_sq → validity_loss, consistency_loss, combined_loss

Standalone functions (backward compatible):
    distance_matrix, cayley_menger_determinant, cayley_menger_from_distances,
    gram_volume_sq, simplex_volume, is_valid_simplex

§4: Structural invariant. If CM fails, geometry is invalid.

geolip.vocabulary.cayley_menger

License: MIT
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, Tuple

try:
    from .formula_base import FormulaBase
except ImportError:
    from formula_base import FormulaBase


# ══════════════════════════════════════════════════════════════════════════════
# Standalone Functions (backward compatible with core.cayley_menger)
# ══════════════════════════════════════════════════════════════════════════════

def distance_matrix(vertices: Tensor) -> Tensor:
    """
    Pairwise squared Euclidean distances.

    Args:
        vertices: (..., K+1, D)

    Returns:
        D_sq: (..., K+1, K+1) symmetric, zeros on diagonal
    """
    diff = vertices.unsqueeze(-2) - vertices.unsqueeze(-3)
    return (diff * diff).sum(dim=-1)


def cayley_menger_determinant(vertices: Tensor) -> Tensor:
    """
    Full Cayley-Menger determinant from vertex positions.

    For a k-simplex with k+1 vertices:
        CM = det of (k+2)×(k+2) bordered distance matrix

    Vol² = (-1)^(k+1) / (2^k * (k!)²) * det(CM)

    Args:
        vertices: (..., K+1, D)

    Returns:
        cm_det: (...)
    """
    n = vertices.shape[-2]
    D_sq = distance_matrix(vertices)

    batch_shape = vertices.shape[:-2]
    size = n + 1
    CM = torch.zeros(*batch_shape, size, size, device=vertices.device, dtype=vertices.dtype)

    CM[..., 0, 1:] = 1.0
    CM[..., 1:, 0] = 1.0
    CM[..., 1:, 1:] = D_sq

    return torch.linalg.det(CM)


def cayley_menger_from_distances(D_sq: Tensor) -> Tensor:
    """
    CM determinant from squared distance matrix.

    Args:
        D_sq: (..., K+1, K+1)

    Returns:
        cm_det: (...)
    """
    n = D_sq.shape[-1]
    batch_shape = D_sq.shape[:-2]
    size = n + 1
    CM = torch.zeros(*batch_shape, size, size, device=D_sq.device, dtype=D_sq.dtype)

    CM[..., 0, 1:] = 1.0
    CM[..., 1:, 0] = 1.0
    CM[..., 1:, 1:] = D_sq

    return torch.linalg.det(CM)


def gram_volume_sq(vertices: Tensor, k: Optional[int] = None) -> Tensor:
    """
    Vol² via Gram determinant. More stable than raw CM for training.

    Vol² = det(G) / (k!)²
    where G = edge_vectors @ edge_vectors.T from vertex 0.

    Args:
        vertices: (..., K+1, D)
        k: simplex dimension (default: inferred)

    Returns:
        vol_sq: (...)
    """
    if k is None:
        k = vertices.shape[-2] - 1
    translated = vertices[..., 1:, :] - vertices[..., :1, :]
    G = torch.matmul(translated, translated.transpose(-2, -1))
    return torch.linalg.det(G) / (math.factorial(k) ** 2)


def simplex_volume(vertices: Tensor, k: Optional[int] = None) -> Tensor:
    """
    Actual simplex volume. Returns 0 for degenerate simplices.

    Args:
        vertices: (..., K+1, D)

    Returns:
        vol: (...) non-negative
    """
    vsq = gram_volume_sq(vertices, k)
    return torch.sqrt(F.relu(vsq))


def is_valid_simplex(vertices: Tensor, k: Optional[int] = None, tol: float = 1e-8) -> Tensor:
    """
    Boolean check: non-degenerate simplex?

    Args:
        vertices: (..., K+1, D)
        tol: minimum Vol² threshold

    Returns:
        valid: (...) boolean tensor
    """
    return gram_volume_sq(vertices, k) > tol


def edge_lengths(vertices: Tensor) -> Tensor:
    """
    All pairwise edge lengths (not squared) as a flat vector.

    Args:
        vertices: (..., K+1, D)

    Returns:
        edges: (..., K*(K+1)/2) — upper triangle of distance matrix
    """
    D_sq = distance_matrix(vertices)
    n = vertices.shape[-2]
    # Extract upper triangle indices
    i_idx, j_idx = torch.triu_indices(n, n, offset=1)
    return torch.sqrt(D_sq[..., i_idx, j_idx].clamp(min=0))


# ══════════════════════════════════════════════════════════════════════════════
# FormulaBase Implementation
# ══════════════════════════════════════════════════════════════════════════════

class CayleyMengerFormula(FormulaBase):
    """
    Cayley-Menger formula for simplex geometry analysis.

    Computes volume, degeneracy, edge structure, and CM determinant
    from vertex positions. Fully differentiable in PyTorch mode.

    Drop-in replacement for geovocab2's CayleyMengerFromSimplex:
        formula = CayleyMengerFormula()
        result = formula(vertices)
        result["volume"], result["is_degenerate"]

    Args:
        eps: Tolerance for degeneracy detection
        validate_input: Check input shape/finiteness before compute
    """

    def __init__(self, eps: float = 1e-8, validate_input: bool = True):
        super().__init__(
            name="cayley_menger",
            uid="formula.cayley_menger.simplex",
        )
        self.eps = eps
        self._validate_input = validate_input

    def compute_numpy(
        self,
        vertices: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute CM properties from vertices using NumPy.

        Args:
            vertices: (K+1, D) or (B, K+1, D)

        Returns:
            Dict with volume, volume_sq, is_degenerate, cm_det, etc.
        """
        vt = torch.from_numpy(np.asarray(vertices, dtype=np.float64))
        result = self.compute_torch(vt, **kwargs)

        # Convert back to numpy
        np_result = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                np_result[k] = v.numpy()
            else:
                np_result[k] = v
        return np_result

    def compute_torch(
        self,
        vertices: Tensor,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compute CM properties from vertices using PyTorch.
        Fully differentiable.

        Args:
            vertices: (..., K+1, D) vertex positions

        Returns:
            Dict:
                k:              int — simplex dimension
                num_vertices:   int — K+1
                distance_sq:    (..., K+1, K+1) pairwise squared distances
                edge_lengths:   (..., num_edges) pairwise Euclidean distances
                gram_det:       (...) Gram determinant
                volume_sq:      (...) squared volume
                volume:         (...) non-negative volume
                cm_det:         (...) raw Cayley-Menger determinant
                is_degenerate:  (...) boolean (Vol² < eps)
                is_valid:       (...) boolean (Vol² >= eps)
                edge_mean:      (...) mean edge length
                edge_std:       (...) std of edge lengths (0 = regular)
                regularity:     (...) 1 - cv of edge lengths (1.0 = perfect regular)
        """
        # Infer simplex dimension
        n = vertices.shape[-2]  # K+1
        k = n - 1

        # ── Core geometry ──
        D_sq = distance_matrix(vertices)
        translated = vertices[..., 1:, :] - vertices[..., :1, :]
        G = torch.matmul(translated, translated.transpose(-2, -1))
        gram_det = torch.linalg.det(G)
        fact_k_sq = math.factorial(k) ** 2
        vol_sq = gram_det / fact_k_sq
        vol = torch.sqrt(F.relu(vol_sq))

        # ── CM determinant (for completeness, less stable than Gram) ──
        size = n + 1
        batch_shape = vertices.shape[:-2]
        CM = torch.zeros(*batch_shape, size, size, device=vertices.device, dtype=vertices.dtype)
        CM[..., 0, 1:] = 1.0
        CM[..., 1:, 0] = 1.0
        CM[..., 1:, 1:] = D_sq
        cm_det = torch.linalg.det(CM)

        # ── Edge analysis ──
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        edges = torch.sqrt(D_sq[..., i_idx, j_idx].clamp(min=0))
        edge_mean = edges.mean(dim=-1)
        edge_std = edges.std(dim=-1)
        # Regularity: coefficient of variation → 1 means perfect
        cv = edge_std / edge_mean.clamp(min=1e-10)
        regularity = 1.0 - cv.clamp(max=1.0)

        # ── Degeneracy ──
        is_degenerate = vol_sq < self.eps
        is_valid = ~is_degenerate

        return {
            "k": k,
            "num_vertices": n,
            "distance_sq": D_sq,
            "edge_lengths": edges,
            "gram_det": gram_det,
            "volume_sq": vol_sq,
            "volume": vol,
            "cm_det": cm_det,
            "is_degenerate": is_degenerate,
            "is_valid": is_valid,
            "edge_mean": edge_mean,
            "edge_std": edge_std,
            "regularity": regularity,
        }

    def validate_input(self, vertices, **kwargs) -> Tuple[bool, str]:
        """Check vertex tensor shape and finiteness."""
        if not self._validate_input:
            return True, ""

        if isinstance(vertices, np.ndarray):
            if not np.all(np.isfinite(vertices)):
                return False, "Vertices contain NaN or Inf"
            if vertices.ndim < 2:
                return False, f"Need at least 2 dims (K+1, D), got {vertices.ndim}"
            n, d = vertices.shape[-2], vertices.shape[-1]
            if n < 2:
                return False, f"Need at least 2 vertices, got {n}"
            if d < n - 1:
                return False, f"embed_dim ({d}) must be >= k ({n - 1})"
        elif isinstance(vertices, Tensor):
            if not torch.all(torch.isfinite(vertices)):
                return False, "Vertices contain NaN or Inf"
            if vertices.ndim < 2:
                return False, f"Need at least 2 dims (K+1, D), got {vertices.ndim}"
            n, d = vertices.shape[-2], vertices.shape[-1]
            if n < 2:
                return False, f"Need at least 2 vertices, got {n}"
            if d < n - 1:
                return False, f"embed_dim ({d}) must be >= k ({n - 1})"
        else:
            return False, f"Expected ndarray or Tensor, got {type(vertices)}"

        return True, ""

    def info(self) -> Dict[str, Any]:
        base = super().info()
        base.update({
            "description": "Cayley-Menger simplex geometry analysis",
            "eps": self.eps,
            "outputs": [
                "volume", "volume_sq", "cm_det", "gram_det",
                "is_degenerate", "is_valid",
                "distance_sq", "edge_lengths", "edge_mean", "edge_std", "regularity",
            ],
        })
        return base


# ══════════════════════════════════════════════════════════════════════════════
# Training Module (nn.Module)
# ══════════════════════════════════════════════════════════════════════════════

class CayleyMengerValidator(nn.Module):
    """
    Training losses from Cayley-Menger geometry.

    Uses Gram-based Vol² (more stable than raw CM det) for:
        validity_loss:     penalize Vol² < 0 (collapsed/inverted simplices)
        consistency_loss:  encourage uniform geometry across batch
        regularity_loss:   penalize irregular edge lengths

    Can operate on:
        - Raw vertices: (..., K+1, D) → computes Vol² internally
        - Pre-computed vol_sq: (...) → skips recomputation

    Args:
        k: Simplex dimension (for factorial normalization)
        eps: Degeneracy threshold for validity checks
    """

    def __init__(self, k: int = 4, eps: float = 1e-8):
        super().__init__()
        self.k = k
        self.eps = eps
        self.factorial_k_sq = math.factorial(k) ** 2
        self._formula = CayleyMengerFormula(eps=eps, validate_input=False)

    def analyze(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Full analysis via CayleyMengerFormula. Returns result dict."""
        return self._formula.compute_torch(vertices)

    def gram_volume_sq(self, vertices: Tensor) -> Tensor:
        """vertices: (..., K+1, D) → (...) volume squared."""
        return gram_volume_sq(vertices, self.k)

    def full_determinant(self, vertices: Tensor) -> Tensor:
        """vertices: (..., K+1, D) → (...) raw CM determinant."""
        return cayley_menger_determinant(vertices)

    def validity_loss(self, vol_sq: Tensor) -> Tensor:
        """Penalize Vol² < 0 (collapsed/inverted simplices)."""
        return F.relu(-vol_sq).mean()

    def consistency_loss(self, vol_sq: Tensor) -> Tensor:
        """Encourage uniform simplex geometry across batch."""
        return vol_sq.var()

    def regularity_loss(self, vertices: Tensor) -> Tensor:
        """
        Penalize irregular edge lengths within simplices.
        Loss = 0 when all edges are equal length (regular simplex).
        """
        edges = edge_lengths(vertices)  # (..., num_edges)
        return edges.std(dim=-1).mean()

    def combined_loss(
        self,
        vol_sq: Tensor,
        validity_weight: float = 1.0,
        consistency_weight: float = 0.1,
    ) -> Tensor:
        """Weighted sum of validity + consistency."""
        return (
            validity_weight * self.validity_loss(vol_sq)
            + consistency_weight * self.consistency_loss(vol_sq)
        )

    def validate(self, vertices: Tensor, tol: Optional[float] = None) -> Tensor:
        """Boolean validity check."""
        tol = tol if tol is not None else self.eps
        return is_valid_simplex(vertices, self.k, tol)


# ══════════════════════════════════════════════════════════════════════════════
# Self-Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Cayley-Menger Self-Test ===\n")

    formula = CayleyMengerFormula(eps=1e-8)

    # ── Test 1: Regular triangle in R² ──
    print("--- Regular triangle (k=2) ---")
    verts_2d = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, math.sqrt(3) / 2],
    ])
    result = formula(verts_2d)
    print(f"  k={result['k']}, vertices={result['num_vertices']}")
    print(f"  Volume: {result['volume'].item():.6f} (expected: {math.sqrt(3)/4:.6f})")
    print(f"  Edge lengths: {result['edge_lengths'].tolist()}")
    print(f"  Regularity: {result['regularity'].item():.6f} (1.0 = perfect)")
    print(f"  Is valid: {result['is_valid'].item()}")

    # ── Test 2: Regular tetrahedron ──
    print("\n--- Regular tetrahedron (k=3) ---")
    s = 1.0
    verts_3d = torch.tensor([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=torch.float32) * s / 2
    result = formula(verts_3d)
    print(f"  Volume: {result['volume'].item():.6f}")
    print(f"  Edge mean: {result['edge_mean'].item():.6f}")
    print(f"  Edge std: {result['edge_std'].item():.8f} (should be ~0)")
    print(f"  Regularity: {result['regularity'].item():.6f}")

    # ── Test 3: Degenerate simplex ──
    print("\n--- Degenerate (collinear) ---")
    verts_degen = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    result = formula(verts_degen)
    print(f"  Volume: {result['volume'].item():.8f}")
    print(f"  Is degenerate: {result['is_degenerate'].item()}")

    # ── Test 4: Batched ──
    print("\n--- Batched (B=4, k=4 pentachora in R^10) ---")
    batch_verts = torch.randn(4, 5, 10)
    result = formula(batch_verts)
    print(f"  Volume shape: {result['volume'].shape}")
    print(f"  Volumes: {result['volume'].tolist()}")
    print(f"  Valid: {result['is_valid'].tolist()}")

    # ── Test 5: Gradient flow ──
    print("\n--- Gradient flow ---")
    verts = torch.randn(5, 10, requires_grad=True)
    result = formula(verts)
    result["volume"].backward()
    has_grad = verts.grad is not None and verts.grad.abs().sum() > 0
    print(f"  Gradients flow through volume: {'✓' if has_grad else '✗'}")

    # ── Test 6: Validator module ──
    print("\n--- CayleyMengerValidator ---")
    validator = CayleyMengerValidator(k=4)
    verts_batch = torch.randn(8, 5, 10)
    vol_sq = validator.gram_volume_sq(verts_batch)
    print(f"  Vol² shape: {vol_sq.shape}")
    print(f"  Validity loss: {validator.validity_loss(vol_sq).item():.6f}")
    print(f"  Consistency loss: {validator.consistency_loss(vol_sq).item():.6f}")
    print(f"  Regularity loss: {validator.regularity_loss(verts_batch).item():.6f}")
    print(f"  Valid count: {validator.validate(verts_batch).sum().item()}/{len(verts_batch)}")

    # ── Test 7: Full analysis ──
    print("\n--- Full analysis via validator ---")
    analysis = validator.analyze(verts_batch[0:1])
    for k, v in analysis.items():
        if isinstance(v, torch.Tensor) and v.numel() <= 1:
            print(f"  {k}: {v.item()}")
        elif isinstance(v, int):
            print(f"  {k}: {v}")

    # ── Test 8: NumPy backend ──
    print("\n--- NumPy backend ---")
    np_verts = np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=np.float32)
    np_result = formula.compute(np_verts, backend="numpy")
    print(f"  Volume: {np_result['volume']}")
    print(f"  Is valid: {np_result['is_valid']}")

    print("\n✓ cayley_menger.py operational")