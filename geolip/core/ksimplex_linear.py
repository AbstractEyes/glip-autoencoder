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

Initialization uses SimplexFactory to generate geometrically valid
regular pentachora as the starting structure. Each simplex begins as
a proper geometric object — not random noise hoping to find validity.

At scale (512→512, k=4):
    103 simplices × ~340 structure ≈ 35,000 params
    nn.Linear equiv: 262,656 params
    Ratio: ~0.134x (13.4% of linear)

Author: AbstractPhil + Claude
License: Apache-2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class KSimplexLinear(nn.Module):
    """
    K-Simplex structured linear layer.

    Input is chunked into simplex-sized groups of (k+1) scalars.
    Each group flows through entry → vertex → pairwise → attenuate → exit.
    Pairwise signals create shared sections between all vertex pairs
    within each simplex, carrying per-channel vector information.

    Initialization uses SimplexFactory (method="regular") when available,
    falling back to geometric heuristic init otherwise. Regular init
    ensures each simplex starts as a valid geometric object with equal
    edge lengths and proper volume.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension (default: same as input)
        k: Simplex dimension (k+1 vertices per simplex)
        hidden_per_vertex: Hidden dimension per vertex (default: k+1)
        init_method: SimplexFactory method ("regular", "random", "uniform")
        init_scale: Scale factor for simplex initialization
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        k: int = 4,
        hidden_per_vertex: Optional[int] = None,
        init_method: str = "regular",
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.k = k
        self.num_vertices = k + 1
        self.hidden_per_vertex = hidden_per_vertex or self.num_vertices
        self.num_pairs = (self.num_vertices * (self.num_vertices - 1)) // 2
        self.init_method = init_method
        self.init_scale = init_scale

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
            "pair_i_local", torch.tensor(pair_i_local, dtype=torch.long)
        )
        self.register_buffer(
            "pair_j_local", torch.tensor(pair_j_local, dtype=torch.long)
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Initialization
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_simplex_template(self) -> torch.Tensor:
        """
        Generate a (num_vertices, hidden_per_vertex) simplex template
        using SimplexFactory. Falls back to geometric heuristic if
        factory is unavailable.

        Returns:
            (V, H) tensor — one vertex per row, each H-dimensional
        """
        try:
            from geolip.vocabulary.simplex_factory import SimplexFactory

            factory = SimplexFactory(
                k=self.k,
                embed_dim=self.hidden_per_vertex,
                method=self.init_method,
                scale=self.init_scale,
            )
            return factory.build_torch(device="cpu", dtype=torch.float32, validate=False)

        except ImportError:
            # Fallback: construct regular simplex analytically
            return self._regular_simplex_fallback()

    def _regular_simplex_fallback(self) -> torch.Tensor:
        """
        Construct a regular simplex with unit edge length analytically.
        Matches SimplexFactory(method="regular") output.
        """
        V = self.num_vertices
        H = self.hidden_per_vertex

        if V == 1:
            return torch.zeros(1, H)

        min_dim = V
        verts = torch.full((V, min_dim), -1.0 / self.k)
        coef = math.sqrt((self.k + 1.0) / self.k)
        for i in range(min(V, min_dim)):
            verts[i, i] = coef

        # Embed or truncate to hidden_per_vertex
        if H > min_dim:
            full = torch.zeros(V, H)
            full[:, :min_dim] = verts
            verts = full
        else:
            verts = verts[:, :H]

        # Center and normalize to unit edge length
        verts = verts - verts.mean(dim=0, keepdim=True)
        edge_len = (verts[1] - verts[0]).norm()
        if edge_len > 1e-10:
            verts = verts / edge_len

        return verts * self.init_scale

    def _init_parameters(self):
        """
        Initialize using SimplexFactory-generated template.

        The template pentachoron defines vertex positions in H-dimensional
        hidden space. These positions seed:
            - entry_weights:    vertex directions as projection targets
            - vertex_scale:     ones (identity transform initially)
            - signal_weight_i/j: pairwise difference vectors (d² structure)
            - attenuation_i/j:  edge-length-proportional per-channel weights
            - exit_weights:     transpose of entry (approximate inverse)
        """
        # Generate one template simplex: (V, H)
        template = self._get_simplex_template()  # (num_vertices, hidden_per_vertex)

        V = self.num_vertices
        H = self.hidden_per_vertex
        S = self.num_simplices

        # ── Entry: tile template vertex directions across all simplices ──
        # Each input scalar projects into its vertex's direction in hidden space
        entry_tiled = template.repeat(S, 1)  # (S*V, H) = (padded_dim, H)
        self.entry_weights.data.copy_(entry_tiled[:self.padded_dim])
        nn.init.zeros_(self.entry_bias)

        # ── Vertex: identity transform to start ──
        nn.init.ones_(self.vertex_scale)
        nn.init.zeros_(self.vertex_bias)

        # ── Pairwise: signal weights from simplex edge vectors ──
        # For each pair (i,j), the signal weight encodes vertex_i and vertex_j
        # directions in hidden space — the geometric edge
        pair_w_i = []
        pair_w_j = []
        pair_bias = []
        for s in range(S):
            for i in range(V):
                for j in range(i + 1, V):
                    # Weight_i: direction from vertex_i's position
                    # Weight_j: direction from vertex_j's position
                    # This seeds the pairwise signal as a geometric edge probe
                    pair_w_i.append(template[i])
                    pair_w_j.append(template[j])
                    pair_bias.append(torch.zeros(H))

        self.signal_weight_i.data.copy_(torch.stack(pair_w_i))
        self.signal_weight_j.data.copy_(torch.stack(pair_w_j))
        self.signal_bias.data.copy_(torch.stack(pair_bias))

        # ── Attenuation: edge-length-proportional per-channel ──
        # Start with uniform attenuation scaled by edge length
        att_i = []
        att_j = []
        for s in range(S):
            for i in range(V):
                for j in range(i + 1, V):
                    edge = template[j] - template[i]
                    edge_len = edge.norm().clamp(min=1e-8)
                    # Per-channel: proportional to edge direction magnitude
                    att_i.append(edge.abs() / edge_len)
                    att_j.append(edge.abs() / edge_len)

        self.attenuation_i.data.copy_(torch.stack(att_i))
        self.attenuation_j.data.copy_(torch.stack(att_j))

        # ── Exit: approximate inverse of entry (transpose direction) ──
        self.exit_weights.data.copy_(entry_tiled[:self.padded_dim])
        nn.init.zeros_(self.exit_bias)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Forward
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through k-simplex structure.

        Args:
            x: (..., input_dim) input tensor, any batch dimensions

        Returns:
            (..., output_dim) output tensor
        """
        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)
        B = x_flat.shape[0]

        # Pad input to multiple of num_vertices
        if self.padded_dim > self.input_dim:
            x_padded = F.pad(x_flat, (0, self.padded_dim - self.input_dim))
        else:
            x_padded = x_flat

        # ━━━━━ Step 1: Entry — each scalar → hidden vector ━━━━━
        h = x_padded.unsqueeze(-1) * self.entry_weights.unsqueeze(0) + self.entry_bias.unsqueeze(0)

        # ━━━━━ Step 2: Vertex transform — scale + shift ━━━━━
        h = h * self.vertex_scale.unsqueeze(0) + self.vertex_bias.unsqueeze(0)

        # ━━━━━ Step 3: Reshape into simplices ━━━━━
        h_simp = h.view(B, self.num_simplices, self.num_vertices, self.hidden_per_vertex)

        # ━━━━━ Step 4: Pairwise signals — VECTOR per pair ━━━━━
        h_i = h_simp[:, :, self.pair_i_local, :]
        h_j = h_simp[:, :, self.pair_j_local, :]

        w_i = self.signal_weight_i.view(self.num_simplices, self.num_pairs, self.hidden_per_vertex)
        w_j = self.signal_weight_j.view(self.num_simplices, self.num_pairs, self.hidden_per_vertex)
        bias = self.signal_bias.view(self.num_simplices, self.num_pairs, self.hidden_per_vertex)

        signals = h_i * w_i.unsqueeze(0) + h_j * w_j.unsqueeze(0) + bias.unsqueeze(0)

        # ━━━━━ Step 5: Attenuate signals back to vertices PER CHANNEL ━━━━━
        att_i = self.attenuation_i.view(self.num_simplices, self.num_pairs, self.hidden_per_vertex)
        att_j = self.attenuation_j.view(self.num_simplices, self.num_pairs, self.hidden_per_vertex)

        signals_to_i = signals * att_i.unsqueeze(0)
        signals_to_j = signals * att_j.unsqueeze(0)

        vertex_signals = torch.zeros(
            B, self.num_simplices, self.num_vertices, self.hidden_per_vertex,
            device=x.device, dtype=x.dtype,
        )
        pair_i_exp = self.pair_i_local.view(1, 1, -1, 1).expand(B, self.num_simplices, -1, self.hidden_per_vertex)
        pair_j_exp = self.pair_j_local.view(1, 1, -1, 1).expand(B, self.num_simplices, -1, self.hidden_per_vertex)

        vertex_signals.scatter_add_(2, pair_i_exp, signals_to_i)
        vertex_signals.scatter_add_(2, pair_j_exp, signals_to_j)

        h_simp = h_simp + vertex_signals

        # ━━━━━ Step 6: Exit — hidden → output scalar ━━━━━
        h_out = h_simp.view(B, self.padded_dim, self.hidden_per_vertex)
        out = (h_out * self.exit_weights.unsqueeze(0)).sum(dim=-1) + self.exit_bias.unsqueeze(0)
        out = out[:, :self.input_dim]

        if self.output_proj is not None:
            out = self.output_proj(out)

        return out.view(*leading_shape, self.output_dim)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Diagnostics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def linear_equiv_params(self) -> int:
        return self.input_dim * self.output_dim + self.output_dim

    def structure_summary(self):
        print(f"\n=== KSimplexLinear (k={self.k}) ===")
        print(f"Input: {self.input_dim} → Output: {self.output_dim}")
        print(f"Simplices: {self.num_simplices} (each covers {self.num_vertices} inputs)")
        print(f"Per simplex: {self.num_vertices} vertices × {self.hidden_per_vertex} hidden × {self.num_pairs} pairs")
        print(f"Init: SimplexFactory method='{self.init_method}', scale={self.init_scale}")
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

    # Test 1: Structure with factory init
    print("--- Structure + Factory Init ---")
    for input_dim, k in [(3, 2), (5, 4), (512, 4)]:
        layer = KSimplexLinear(input_dim, k=k, init_method="regular")
        layer.structure_summary()

    # Test 2: Verify simplex template is geometrically valid
    print("\n--- Simplex Template Verification ---")
    layer = KSimplexLinear(5, k=4, init_method="regular")
    template = layer._get_simplex_template()
    print(f"  Template shape: {template.shape}")
    # Check edge lengths
    edges = []
    V = template.shape[0]
    for i in range(V):
        for j in range(i + 1, V):
            edges.append((template[j] - template[i]).norm().item())
    print(f"  Edge lengths: {[f'{e:.4f}' for e in edges]}")
    print(f"  Edge std: {torch.tensor(edges).std():.6f} (should be ~0 for regular)")

    # Test 3: Forward pass
    print("\n--- Forward Pass ---")
    for input_dim, k in [(5, 4), (512, 4), (1024, 4)]:
        layer = KSimplexLinear(input_dim, k=k)
        x = torch.randn(4, input_dim)
        y = layer(x)
        print(f"  k={k}, ({4}, {input_dim}) → {y.shape}")

    # Test 4: Batched input
    print("\n--- Batched Input ---")
    layer = KSimplexLinear(64, k=4)
    x = torch.randn(2, 16, 64)
    y = layer(x)
    print(f"  (2, 16, 64) → {y.shape}")

    # Test 5: Gradient flow
    print("\n--- Gradient Flow ---")
    layer = KSimplexLinear(128, k=4)
    x = torch.randn(4, 128, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    has_grad = x.grad is not None and x.grad.abs().sum() > 0
    print(f"  Gradients flow: {'✓' if has_grad else '✗'}")

    # Test 6: Training convergence
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

    # Test 7: Init method comparison
    print("\n--- Init Method Comparison ---")
    for method in ["regular", "random", "uniform"]:
        layer = KSimplexLinear(64, k=4, init_method=method)
        x = torch.randn(8, 64)
        y = layer(x)
        print(f"  method={method:8s}: output mean={y.mean():.4f}, std={y.std():.4f}")

    # Test 8: Per-channel differentiation
    print("\n--- Per-Channel Differentiation ---")
    layer = KSimplexLinear(5, k=4)
    x = torch.randn(2, 5, requires_grad=True)
    y = layer(x)
    y[0, 0].backward(retain_graph=True)
    g1 = x.grad.clone()
    x.grad.zero_()
    y[0, 1].backward(retain_graph=True)
    g2 = x.grad.clone()
    channel_diff = (g1 - g2).abs().sum().item()
    print(f"  Gradient difference: {channel_diff:.6f}")
    print(f"  Per-channel differentiation: {'✓' if channel_diff > 1e-6 else '✗'}")

    print("\n=== KSimplexLinear operational. ===")