"""
Cayley-Menger Validation
=========================

S4: Structural invariant. If CM fails, geometry is invalid.
Uses Gram determinant (S4.3, more stable than raw CM det).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CayleyMengerValidator(nn.Module):
    """
    S4.3: Vol^2 via Gram determinant.
    S4.4: Validity loss (penalize collapsed simplices).
    S4.5: Consistency loss (uniform structure).
    """

    def __init__(self, k: int = 4):
        super().__init__()
        self.k = k
        self.factorial_k = math.factorial(k)

    def gram_volume_sq(self, vertices: Tensor) -> Tensor:
        """vertices: (..., K+1, D) -> (...) volume squared."""
        translated = vertices[..., 1:, :] - vertices[..., :1, :]
        G = torch.matmul(translated, translated.transpose(-2, -1))
        return torch.linalg.det(G) / (self.factorial_k ** 2)

    def validity_loss(self, vol_sq: Tensor) -> Tensor:
        """S4.4: Penalize Vol^2 < 0."""
        return F.relu(-vol_sq).mean()

    def consistency_loss(self, vol_sq: Tensor) -> Tensor:
        """S4.5: Encourage uniform geometry."""
        return vol_sq.var()