"""
Patch
======

Spatial patchification of input latents.
(B, C, H, W) → (B, grid*grid, patch_dim)
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..core.ksimplex_linear import KSimplexLinear


class Patchifier(nn.Module):
    """Patchify latents and project via KSimplex."""

    def __init__(
        self,
        in_channels: int = 32,
        in_height: int = 64,
        in_width: int = 64,
        patch_grid: int = 8,
        simplex_k: int = 4,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        pH = in_height // patch_grid
        pW = in_width // patch_grid
        self.patch_dim = in_channels * pH * pW

        self.proj = KSimplexLinear(self.patch_dim, self.patch_dim, k=simplex_k)
        self.norm = nn.LayerNorm(self.patch_dim)

    def forward(self, latents: Tensor) -> Tensor:
        """(B, C, H, W) → (B, S, patch_dim)"""
        B, C, H, W = latents.shape
        g = self.patch_grid
        pH, pW = H // g, W // g
        x = latents.reshape(B, C, g, pH, g, pW)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, g * g, C * pH * pW)
        return self.norm(self.proj(x))