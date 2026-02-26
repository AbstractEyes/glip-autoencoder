"""
Topological State
==================

Single state in the vocabulary: 256 tokens × 64 dims = 16,384 dimensions.

Behavior axiomatically determined by position in Cantor topology.
Frozen simplex template + minor deformation (anchored navigation).
KSimplexLinear for all transformations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from ..core import KSimplexLinear, CayleyMengerValidator


class TopologicalState(nn.Module):
    """
    256 × 64 token engine at a deterministic topological position.

    The state doesn't learn WHAT it is — the topology assigns that.
    It learns HOW to navigate within its assigned region.
    """

    def __init__(
        self,
        input_dim: int,
        tokens_per_state: int = 256,
        token_dim: int = 64,
        simplex_k: int = 4,
        deform_alpha: float = 0.05,
        state_idx: int = 0,
        branch_path: Tensor = None,
        staircase_val: float = 0.0,
    ):
        super().__init__()
        self.tokens_per_state = tokens_per_state
        self.token_dim = token_dim
        self.simplex_k = simplex_k
        self.state_idx = state_idx
        self.state_dim = tokens_per_state * token_dim

        K = simplex_k
        K1 = K + 1
        deform_scaled = deform_alpha / math.sqrt(K1)

        if branch_path is not None:
            self.register_buffer("branch_path", branch_path)
        else:
            self.register_buffer("branch_path", torch.zeros(5, dtype=torch.int32))

        self.staircase_val = staircase_val

        # Frozen regular simplex template
        template = self._state_template(state_idx, K, token_dim)
        self.register_buffer("template", template)
        self.deform = KSimplexLinear(token_dim, token_dim * K1, k=K)
        self.deform_scale = deform_scaled

        # Intake: input → full token space
        self.intake = KSimplexLinear(input_dim, self.state_dim, k=K)
        self.intake_norm = nn.LayerNorm(self.state_dim)

        # Internal self-attention across 256 tokens
        self.q_proj = KSimplexLinear(token_dim, token_dim, k=K)
        self.k_proj = KSimplexLinear(token_dim, token_dim, k=K)
        self.v_proj = KSimplexLinear(token_dim, token_dim, k=K)
        self.out_proj = KSimplexLinear(token_dim, token_dim, k=K)
        self.attn_norm = nn.LayerNorm(token_dim)

        self.cm = CayleyMengerValidator(K)

    @staticmethod
    def _state_template(state_idx: int, k: int, dim: int) -> Tensor:
        """Deterministic simplex seeded by state index."""
        gen = torch.Generator()
        gen.manual_seed(state_idx * 7919 + 31337)
        n = k + 1
        verts = torch.randn(n, dim, generator=gen) * 0.1
        verts = verts - verts.mean(0, keepdim=True)
        if n > 1:
            edge = (verts[0] - verts[1]).norm()
            if edge > 1e-10:
                verts = verts / edge
        return verts

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: (B, input_dim)

        Returns:
            tokens: (B, 256, 64)
            vol_sq: (B, 256)
        """
        B = features.shape[0]
        K1 = self.simplex_k + 1

        h = self.intake_norm(self.intake(features))
        tokens = h.view(B, self.tokens_per_state, self.token_dim)

        # Simplex deformation
        delta = self.deform(tokens).view(B, self.tokens_per_state, K1, self.token_dim)
        delta = delta * self.deform_scale
        vertices = self.template.unsqueeze(0).unsqueeze(0) + delta
        vol_sq = self.cm.gram_volume_sq(vertices)

        # Branch-path-determined vertex weighting
        path_signal = self.branch_path.float() / 2.0
        v_weights = torch.zeros(K1, device=tokens.device)
        for i in range(min(len(path_signal), K1)):
            v_weights[i] = path_signal[i]
        v_weights = F.softmax(v_weights, dim=0)
        tokens = torch.einsum("btvd,v->btd", vertices, v_weights)

        # Internal composition
        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.token_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        tokens = self.attn_norm(tokens + self.out_proj(out))

        return tokens, vol_sq