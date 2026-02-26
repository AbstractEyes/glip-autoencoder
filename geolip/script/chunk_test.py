"""
chunk_test.py — Self-test for the full 512-state CHUNK
=======================================================

Validates topology, parameter counts, forward pass, gradient flow.
Runs with a reduced active budget for memory.

Usage:
    python -m geochunker.script.chunk_test
"""

import torch
from geochunker.core import CantorTopology
from geochunker.scales import Chunk


def main():
    print("\n>>> CHUNK Self-Test\n")
    print("=" * 70)
    print("CHUNK — 512 × 256 × 64 = 8,388,608 tokens")
    print("=" * 70)

    # Topology (no GPU needed)
    print("\nBuilding deterministic Cantor topology (512 states)...")
    topo = CantorTopology(num_states=512, levels=9)
    print(f"  Positions: [{topo.positions[0]:.4f}, ..., {topo.positions[-1]:.4f}]")
    print(f"  Staircase: [{topo.staircase_vals[0]:.4f}, ..., {topo.staircase_vals[-1]:.4f}]")
    print(f"  Branch path [0]:   {topo.branch_paths[0].tolist()}")
    print(f"  Branch path [511]: {topo.branch_paths[511].tolist()}")
    print(f"  Alignment [0,1]:   {topo.get_alignment(0, 1):.4f}")
    print(f"  Alignment [0,256]: {topo.get_alignment(0, 256):.4f}")
    print(f"  Alignment [0,511]: {topo.get_alignment(0, 511):.4f}")
    nbrs = topo.get_neighborhood(0, threshold=0.6)
    print(f"  State 0 neighborhood (>0.6): {len(nbrs)} states")

    # Build with 8 states and small budget for testing
    print("\nBuilding reduced CHUNK (8 states, 4 active) for test...")
    model = Chunk(
        in_channels=32, in_height=64, in_width=64,
        max_active_states=4,
    )

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    per_state = sum(p.numel() for p in model.states[0].parameters())
    print(f"\n  Per state:  {per_state:>12,d}")
    print(f"  Total:      {total:>12,d}")
    print(f"\n  Extrapolated to full 512:")
    print(f"  States:     {per_state * 512:>12,d}")
    print(f"  ~{per_state * 512 / 1e6:.1f}M parameters in states alone")

    # Forward
    print(f"\nForward pass (B=1)...")
    latents = torch.randn(1, 32, 64, 64)
    output = model(latents, active_budget=4)
    print(f"  Composed:  {output.composed.shape}")
    print(f"  Active:    {output.active_indices.shape}")
    print(f"  Routing:   {[f'{w:.3f}' for w in output.routing_weights[0].tolist()[:4]]}")
    print(f"  Vol² valid: {(output.vol_sq > 0).float().mean():.2%}")
    print(f"  CM loss:   {float(output.loss):.6f}")

    # Gradient
    print("\nGradient check...")
    output.loss.backward()
    has_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_params = sum(1 for _ in model.parameters())
    print(f"  {has_grad}/{total_params} parameters have gradients")

    print("\n>>> CHUNK operational.\n")


if __name__ == "__main__":
    main()