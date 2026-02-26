"""
Cayley-Menger Validation
=========================

§4: Structural invariant. If CM fails, geometry is invalid.

Provides both standalone functions and the validator module:

Functions:
    distance_matrix         — pairwise squared distances from vertices
    cayley_menger_determinant — full CM determinant from vertices
    cayley_menger_from_distances — CM determinant from distance matrix
    gram_volume_sq          — Vol² via Gram determinant (stable)
    is_valid_simplex        — boolean check (Vol² > 0)
    simplex_volume          — actual volume (sqrt of Vol²)

Module:
    CayleyMengerValidator   — losses for training (validity + consistency)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def distance_matrix(vertices: Tensor) -> Tensor:
    """
    Pairwise squared Euclidean distances.

    Args:
        vertices: (..., K+1, D)

    Returns:
        D_sq: (..., K+1, K+1) symmetric, zeros on diagonal
    """
    diff = vertices.unsqueeze(-2) - vertices.unsqueeze(-3)  # (..., K+1, K+1, D)
    return (diff * diff).sum(dim=-1)


def cayley_menger_determinant(vertices: Tensor) -> Tensor:
    """
    Full Cayley-Menger determinant from vertex positions.

    For a k-simplex with k+1 vertices in D dimensions:
        CM = det of (k+2)×(k+2) bordered distance matrix

    The relationship to volume:
        Vol² = (-1)^(k+1) / (2^k * (k!)²) * det(CM)

    Args:
        vertices: (..., K+1, D) — K+1 vertices of a k-simplex

    Returns:
        cm_det: (...) — raw CM determinant values
    """
    n = vertices.shape[-2]  # K+1
    D_sq = distance_matrix(vertices)

    # Build bordered matrix: (n+1) × (n+1)
    # Row/col 0: [0, 1, 1, ..., 1]
    # Rest: [1, d01², d02², ...]
    batch_shape = vertices.shape[:-2]
    size = n + 1
    CM = torch.zeros(*batch_shape, size, size, device=vertices.device, dtype=vertices.dtype)

    CM[..., 0, 1:] = 1.0
    CM[..., 1:, 0] = 1.0
    CM[..., 1:, 1:] = D_sq

    return torch.linalg.det(CM)


def cayley_menger_from_distances(D_sq: Tensor) -> Tensor:
    """
    CM determinant directly from a squared distance matrix.

    Args:
        D_sq: (..., K+1, K+1) — pairwise squared distances

    Returns:
        cm_det: (...) — raw CM determinant values
    """
    n = D_sq.shape[-1]
    batch_shape = D_sq.shape[:-2]
    size = n + 1
    CM = torch.zeros(*batch_shape, size, size, device=D_sq.device, dtype=D_sq.dtype)

    CM[..., 0, 1:] = 1.0
    CM[..., 1:, 0] = 1.0
    CM[..., 1:, 1:] = D_sq

    return torch.linalg.det(CM)


def gram_volume_sq(vertices: Tensor, k: int = None) -> Tensor:
    """
    Vol² via Gram determinant. More numerically stable than raw CM.

    Vol² = det(G) / (k!)²
    where G is the Gram matrix of edge vectors from vertex 0.

    Args:
        vertices: (..., K+1, D)
        k: simplex dimension (default: K+1 - 1, inferred from vertices)

    Returns:
        vol_sq: (...) — squared volume
    """
    if k is None:
        k = vertices.shape[-2] - 1
    translated = vertices[..., 1:, :] - vertices[..., :1, :]
    G = torch.matmul(translated, translated.transpose(-2, -1))
    return torch.linalg.det(G) / (math.factorial(k) ** 2)


def simplex_volume(vertices: Tensor, k: int = None) -> Tensor:
    """
    Actual simplex volume (not squared). Returns 0 for degenerate simplices.

    Args:
        vertices: (..., K+1, D)
        k: simplex dimension (inferred if None)

    Returns:
        vol: (...) — non-negative volume
    """
    vsq = gram_volume_sq(vertices, k)
    return torch.sqrt(F.relu(vsq))


def is_valid_simplex(vertices: Tensor, k: int = None, tol: float = 1e-8) -> Tensor:
    """
    Boolean check: is this a non-degenerate simplex?

    Args:
        vertices: (..., K+1, D)
        k: simplex dimension (inferred if None)
        tol: minimum Vol² to be considered valid

    Returns:
        valid: (...) — boolean tensor
    """
    return gram_volume_sq(vertices, k) > tol


# ---------------------------------------------------------------------------
# Training module
# ---------------------------------------------------------------------------

class CayleyMengerValidator(nn.Module):
    """
    §4.3: Vol² via Gram determinant.
    §4.4: Validity loss (penalize collapsed simplices).
    §4.5: Consistency loss (uniform structure).
    """

    def __init__(self, k: int = 4):
        super().__init__()
        self.k = k
        self.factorial_k = math.factorial(k)

    def gram_volume_sq(self, vertices: Tensor) -> Tensor:
        """vertices: (..., K+1, D) -> (...) volume squared."""
        return gram_volume_sq(vertices, self.k)

    def full_determinant(self, vertices: Tensor) -> Tensor:
        """vertices: (..., K+1, D) -> (...) raw CM determinant."""
        return cayley_menger_determinant(vertices)

    def validity_loss(self, vol_sq: Tensor) -> Tensor:
        """§4.4: Penalize Vol² < 0 (collapsed/inverted simplices)."""
        return F.relu(-vol_sq).mean()

    def consistency_loss(self, vol_sq: Tensor) -> Tensor:
        """§4.5: Encourage uniform simplex geometry across batch."""
        return vol_sq.var()

    def combined_loss(
        self, vol_sq: Tensor, validity_weight: float = 1.0, consistency_weight: float = 0.1
    ) -> Tensor:
        """Weighted sum of validity + consistency."""
        return validity_weight * self.validity_loss(vol_sq) + consistency_weight * self.consistency_loss(vol_sq)

    def validate(self, vertices: Tensor, tol: float = 1e-8) -> Tensor:
        """Boolean validity check for inference/debugging."""
        return is_valid_simplex(vertices, self.k, tol)