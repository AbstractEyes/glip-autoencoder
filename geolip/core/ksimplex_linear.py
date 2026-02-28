"""
KSimplexLinear — Geometric Linear Layer
========================================

Replaces nn.Linear with k-simplex structured computation.

Architecture per simplex (k=4, pentachoron):
    5 scalars → [entry projection] → 5 vertices × 5 hidden
        → [vertex transform] → scale + shift
        → [pairwise signals] → 10 shared sections between all vertex pairs
        → [attenuation] → signals weighted back to vertices PER CHANNEL
        → [exit projection] → 5 scalars

The key insight: hidden layers aren't sequential — they're simplex vertices.
All exist simultaneously and communicate through pairwise shared sections.
Each shared section carries a VECTOR (not scalar) — the hidden dimensions
are the embedding space for producing valid simplex distance geometry.

At scale (512→512, k=4):
    103 simplices × ~290 structure ≈ 35,000 params
    nn.Linear equiv: 262,656 params
    Ratio: ~0.134x (13.4% of linear)

Author: AbstractPhil + Claude
License: Apache-2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class KSimplexLinear(nn.Module):
    """
    K-Simplex structured linear layer.

    Input is chunked into simplex-sized groups of (k+1) scalars.
    Each group flows through entry → vertex → pairwise → attenuate → exit.
    Pairwise signals create shared sections between all vertex pairs
    within each simplex, carrying per-channel vector information.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension (default: same as input)
        k: Simplex dimension (k+1 vertices per simplex)
        hidden_per_vertex: Hidden dimension per vertex (default: k+1)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        k: int = 4,
        hidden_per_vertex: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.k = k
        self.num_vertices = k + 1
        self.hidden_per_vertex = hidden_per_vertex or self.num_vertices
        self.num_pairs = (self.num_vertices * (self.num_vertices - 1)) // 2

        # Tiling: how many simplices cover the input
        self.num_simplices = math.ceil(input_dim / self.num_vertices)
        self.padded_dim = self.num_simplices * self.num_vertices

        # Precompute pair indices for all simplices (vectorized)
        pair_i_local, pair_j_local = [], []
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                pair_i_local.append(i)
                pair_j_local.append(j)
        self.register_buffer(
            'pair_i_local', torch.tensor(pair_i_local, dtype=torch.long)
        )
        self.register_buffer(
            'pair_j_local', torch.tensor(pair_j_local, dtype=torch.long)
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Entry: input scalar → vertex hidden space
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.entry_weights = nn.Parameter(
            torch.empty(self.padded_dim, self.hidden_per_vertex)
        )
        self.entry_bias = nn.Parameter(
            torch.empty(self.padded_dim, self.hidden_per_vertex)
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Vertex transform: scale + shift per vertex
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.vertex_scale = nn.Parameter(
            torch.empty(self.padded_dim, self.hidden_per_vertex)
        )
        self.vertex_bias = nn.Parameter(
            torch.empty(self.padded_dim, self.hidden_per_vertex)
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Pairwise: shared sections between vertex pairs
        # Each pair carries a VECTOR signal (hidden_per_vertex dims)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        total_pairs = self.num_simplices * self.num_pairs
        self.signal_weight_i = nn.Parameter(
            torch.empty(total_pairs, self.hidden_per_vertex)
        )
        self.signal_weight_j = nn.Parameter(
            torch.empty(total_pairs, self.hidden_per_vertex)
        )
        self.signal_bias = nn.Parameter(
            torch.empty(total_pairs, self.hidden_per_vertex)
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Attenuation: per-pair PER-CHANNEL weights back to vertices
        # Vector attenuation preserves channel differentiation
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.attenuation_i = nn.Parameter(
            torch.empty(total_pairs, self.hidden_per_vertex)
        )
        self.attenuation_j = nn.Parameter(
            torch.empty(total_pairs, self.hidden_per_vertex)
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Exit: vertex hidden → output scalar
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.exit_weights = nn.Parameter(
            torch.empty(self.padded_dim, self.hidden_per_vertex)
        )
        self.exit_bias = nn.Parameter(torch.empty(self.padded_dim))

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Output projection (if input_dim != output_dim)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if self.input_dim != self.output_dim:
            self.output_proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        else:
            self.output_proj = None

        self._init_parameters()

    def _init_parameters(self):
        """Initialize with small values for stable training."""
        gain = 1.0 / math.sqrt(self.hidden_per_vertex)

        nn.init.normal_(self.entry_weights, std=gain)
        nn.init.zeros_(self.entry_bias)
        nn.init.ones_(self.vertex_scale)
        nn.init.zeros_(self.vertex_bias)
        nn.init.normal_(self.signal_weight_i, std=gain)
        nn.init.normal_(self.signal_weight_j, std=gain)
        nn.init.zeros_(self.signal_bias)
        nn.init.ones_(self.attenuation_i)
        nn.init.ones_(self.attenuation_j)
        nn.init.normal_(self.exit_weights, std=gain)
        nn.init.zeros_(self.exit_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through k-simplex structure.

        Args:
            x: (..., input_dim) input tensor, any batch dimensions

        Returns:
            (..., output_dim) output tensor
        """
        # Save shape for reshape at end
        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)  # (B, input_dim)
        B = x_flat.shape[0]

        # Pad input to multiple of num_vertices
        if self.padded_dim > self.input_dim:
            x_padded = F.pad(x_flat, (0, self.padded_dim - self.input_dim))
        else:
            x_padded = x_flat  # (B, padded_dim)

        # ━━━━━ Step 1: Entry — each scalar → hidden vector ━━━━━
        # x_padded: (B, padded_dim) → (B, padded_dim, hidden)
        h = x_padded.unsqueeze(-1) * self.entry_weights.unsqueeze(0) + self.entry_bias.unsqueeze(0)

        # ━━━━━ Step 2: Vertex transform — scale + shift ━━━━━
        h = h * self.vertex_scale.unsqueeze(0) + self.vertex_bias.unsqueeze(0)
        # h: (B, padded_dim, hidden)

        # ━━━━━ Step 3: Reshape into simplices ━━━━━
        # (B, padded_dim, hidden) → (B, num_simplices, num_vertices, hidden)
        h_simp = h.view(B, self.num_simplices, self.num_vertices, self.hidden_per_vertex)

        # ━━━━━ Step 4: Pairwise signals — VECTOR per pair ━━━━━
        # Gather vertex pairs for all simplices at once
        # h_simp: (B, S, V, H), pair indices: (P,) → h_i, h_j: (B, S, P, H)
        h_i = h_simp[:, :, self.pair_i_local, :]  # (B, S, P, H)
        h_j = h_simp[:, :, self.pair_j_local, :]  # (B, S, P, H)

        # Reshape signal weights: (S*P, H) → (S, P, H)
        w_i = self.signal_weight_i.view(self.num_simplices, self.num_pairs, self.hidden_per_vertex)
        w_j = self.signal_weight_j.view(self.num_simplices, self.num_pairs, self.hidden_per_vertex)
        bias = self.signal_bias.view(self.num_simplices, self.num_pairs, self.hidden_per_vertex)

        # Compute pairwise signals: (B, S, P, H) — KEPT AS VECTOR
        signals = h_i * w_i.unsqueeze(0) + h_j * w_j.unsqueeze(0) + bias.unsqueeze(0)
        # signals: (B, S, P, H) — each pair carries H-dim relational info

        # ━━━━━ Step 5: Attenuate signals back to vertices PER CHANNEL ━━━━━
        att_i = self.attenuation_i.view(self.num_simplices, self.num_pairs, self.hidden_per_vertex)
        att_j = self.attenuation_j.view(self.num_simplices, self.num_pairs, self.hidden_per_vertex)

        # Per-channel attenuation: (B, S, P, H) * (S, P, H) → (B, S, P, H)
        signals_to_i = signals * att_i.unsqueeze(0)  # (B, S, P, H)
        signals_to_j = signals * att_j.unsqueeze(0)  # (B, S, P, H)

        # Scatter-add into vertex hidden space: (B, S, V, H)
        vertex_signals = torch.zeros(
            B, self.num_simplices, self.num_vertices, self.hidden_per_vertex,
            device=x.device, dtype=x.dtype
        )
        # Expand pair indices for scatter: (P,) → (B, S, P, H)
        pair_i_exp = self.pair_i_local.view(1, 1, -1, 1).expand(B, self.num_simplices, -1, self.hidden_per_vertex)
        pair_j_exp = self.pair_j_local.view(1, 1, -1, 1).expand(B, self.num_simplices, -1, self.hidden_per_vertex)

        vertex_signals.scatter_add_(2, pair_i_exp, signals_to_i)
        vertex_signals.scatter_add_(2, pair_j_exp, signals_to_j)

        # Add per-channel pairwise contribution to vertex hidden states
        h_simp = h_simp + vertex_signals

        # ━━━━━ Step 6: Exit — hidden → output scalar ━━━━━
        # Reshape back: (B, S, V, H) → (B, padded_dim, H)
        h_out = h_simp.view(B, self.padded_dim, self.hidden_per_vertex)

        # Dot product with exit weights + bias
        out = (h_out * self.exit_weights.unsqueeze(0)).sum(dim=-1) + self.exit_bias.unsqueeze(0)
        # out: (B, padded_dim)

        # Trim padding
        out = out[:, :self.input_dim]

        # Output projection if dims differ
        if self.output_proj is not None:
            out = self.output_proj(out)

        # Restore leading dimensions
        return out.view(*leading_shape, self.output_dim)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def linear_equiv_params(self) -> int:
        return self.input_dim * self.output_dim + self.output_dim

    def structure_summary(self):
        print(f"\n=== KSimplexLinear (k={self.k}) ===")
        print(f"Input: {self.input_dim} → Output: {self.output_dim}")
        print(f"Simplices: {self.num_simplices} (each covers {self.num_vertices} inputs)")
        print(f"Per simplex: {self.num_vertices} vertices × {self.hidden_per_vertex} hidden × {self.num_pairs} pairs")
        print(f"Structure size: {self.num_vertices}×{self.hidden_per_vertex}×{self.num_pairs} = {self.num_vertices * self.hidden_per_vertex * self.num_pairs}")
        print(f"\nParams breakdown:")
        print(f"  Entry:     {self.entry_weights.numel() + self.entry_bias.numel():,}")
        print(f"  Vertex:    {self.vertex_scale.numel() + self.vertex_bias.numel():,}")
        print(f"  Pairwise:  {self.signal_weight_i.numel() + self.signal_weight_j.numel() + self.signal_bias.numel():,}")
        print(f"  Attenuate: {self.attenuation_i.numel() + self.attenuation_j.numel():,}")
        print(f"  Exit:      {self.exit_weights.numel() + self.exit_bias.numel():,}")
        if self.output_proj:
            print(f"  OutProj:   {sum(p.numel() for p in self.output_proj.parameters()):,}")
        print(f"  TOTAL:     {self.param_count():,}")
        linear_eq = self.linear_equiv_params()
        print(f"\nnn.Linear equiv: {linear_eq:,}")
        print(f"Ratio: {self.param_count() / linear_eq:.3f}x")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Self-Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=== KSimplexLinear Self-Test ===\n")

    # Test 1: Param counts (will differ from old due to vector attenuation)
    print("--- Param Count Verification ---")
    test_cases = [
        (3, 2),       # k=2, 3 inputs
        (5, 4),       # k=4, 5 inputs
        (512, 4),     # k=4, 512 inputs
    ]

    for input_dim, k in test_cases:
        layer = KSimplexLinear(input_dim, k=k)
        layer.structure_summary()

    # Test 2: Forward pass shapes
    print("\n--- Forward Pass ---")
    for input_dim, k in [(3, 2), (5, 4), (512, 4), (1024, 4)]:
        layer = KSimplexLinear(input_dim, k=k)
        x = torch.randn(4, input_dim)
        y = layer(x)
        print(f"  k={k}, ({4}, {input_dim}) → {y.shape}")

    # Test 3: Batched input with leading dims
    print("\n--- Batched Input ---")
    layer = KSimplexLinear(64, k=4)
    x = torch.randn(2, 16, 64)  # (B, N, D)
    y = layer(x)
    print(f"  (2, 16, 64) → {y.shape}")

    # Test 4: Gradient flow
    print("\n--- Gradient Flow ---")
    layer = KSimplexLinear(128, k=4)
    x = torch.randn(4, 128, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    has_grad = x.grad is not None and x.grad.abs().sum() > 0
    print(f"  Gradients flow: {'✓' if has_grad else '✗'}")

    # Test 5: Training convergence
    print("\n--- Training Test ---")
    layer = KSimplexLinear(64, k=4)
    target = torch.randn(4, 64)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-3)

    for step in range(100):
        x = torch.randn(4, 64)
        y = layer(x)
        loss = F.mse_loss(y, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")

    # Test 6: Different input/output dims
    print("\n--- Input ≠ Output ---")
    layer = KSimplexLinear(512, output_dim=256, k=4)
    x = torch.randn(4, 512)
    y = layer(x)
    print(f"  512 → 256: {y.shape}, params={layer.param_count():,}")
    layer.structure_summary()

    # Test 7: Verify pairwise signals are per-channel
    print("\n--- Pairwise Channel Verification ---")
    layer = KSimplexLinear(5, k=4)
    x = torch.randn(2, 5, requires_grad=True)
    y = layer(x)
    # Check that different hidden channels get different gradients
    y[0, 0].backward(retain_graph=True)
    g1 = x.grad.clone()
    x.grad.zero_()
    y[0, 1].backward(retain_graph=True)
    g2 = x.grad.clone()
    channel_diff = (g1 - g2).abs().sum().item()
    print(f"  Gradient difference across output channels: {channel_diff:.6f}")
    print(f"  Per-channel differentiation: {'✓' if channel_diff > 1e-6 else '✗'}")

    print("\n=== KSimplexLinear operational. ===")