# CLAUDE.md — GLIP Project Context

## What Is This

GLIP (Geometric Linear Interpolative Patchwork) is a geometric autoencoder package. Python package name: `geolip`. Repo: `AbstractEyes/glip-autoencoder`.

This is NOT a conventional neural network with geometric names bolted on. Every component implements actual geometric mathematics. The topology assigns behavior deterministically — the network learns navigation within assigned regions, not classification or representation from scratch.

## Critical Design Principles

1. **Tokens ARE the structure.** The CHUNK spec (512 × 256 × 64 = 8,388,608 tokens) IS the vocabulary. Each of 512 topological states is deterministically assigned by Cantor mathematics. Tokens are geometric positions, not learned embeddings.

2. **KSimplexLinear is not decorative.** It parameterizes transformations as k-simplex vertices with barycentric coordinates. 11.8% of dense params. CM validated. If you replace it with nn.Linear, you've broken the geometric contract.

3. **CantorTopology is pure math, no parameters.** Devil's Staircase partitions [0,1] into states. Alignment is NOT distance — coarse matches = routing highways, fine matches = local structure. Adjacent states have ~0.97 alignment, opposite ends ~0.002.

4. **CM validation is a structural invariant.** If Cayley-Menger determinant is negative, the simplex has collapsed and geometry is invalid. This isn't a regularizer — it's a correctness check.

5. **Each TopologicalState is a complete geometric description of ONE thing.** Different model outputs (transformer, conv, etc.) get assigned to different states. Cross-model differential: outputs from different architectures become directly comparable through shared geometric language.

## Architecture Hierarchy

```
Sector    → 512 Chunks          (placeholder)
Chunk     → 512 × 256 × 64     (17.3B params, ~2.2B active via needs-based loading)
Patchwork → 8 × 256 × 64       (270M params, prototype for validation)
State     → 256 × 64            (single topological state)
Patch     → single spatial position from input
```

Patchwork is the current training target. Validate that 8 states differentiate along topological lines before scaling to 512-state Chunk.

## Package Layout

```
geolip/
├── core/           # Mathematical primitives (no architecture opinions)
│   ├── ksimplex_linear.py    §8: 11.8% params
│   ├── topology.py           CantorTopology (pure math)
│   └── cayley_menger.py      Functions + validator module
├── scales/         # Architecture at each scale
│   ├── patch.py              Patchifier
│   ├── state.py              TopologicalState
│   ├── patchwork.py          8-state prototype
│   ├── chunk.py              512-state full
│   └── sector.py             placeholder
├── script/         # Runnable utilities
│   ├── chunk_test.py
│   └── generator.py
└── train/          # Data generation, loading, and training
    ├── geometric_generator.py   Pure numpy SDF factory (38 classes)
    ├── voxel_generator.py       Voxel factory (binary occupancy)
    ├── dataloader.py            Dataset/DataLoader wrappers
    └── train_patchwork.py       Training loop
```

All `__init__.py` files are empty. Use direct imports from the package directory.

## Import Convention

All `__init__.py` files are empty. Use direct imports from modules:
```python
from geolip.core.ksimplex_linear import KSimplexLinear
from geolip.core.topology import CantorTopology
from geolip.core.cayley_menger import CayleyMengerValidator, cayley_menger_determinant
from geolip.scales.patchwork import Patchwork
from geolip.train.geometric_generator import GeometricShapeFactory
from geolip.train.dataloader import PreparedDataset, LazyShapeDataset, make_dataloader
```

## Key Signatures

```python
# Patchwork (the thing you're training)
Patchwork(in_channels=32, in_height=64, in_width=64, patch_grid=8,
          tokens_per_state=256, token_dim=64, simplex_k=4, deform_alpha=0.05)
# Returns PatchworkOutput with .shape, .composed, .loss, .loss_dict, etc.

# Shape factory (pure numpy, no torch)
factory = GeometricShapeFactory(width=64, height=64, seed=42)
latent, labels, masks, geometry = factory(channels=32, n_shapes=3)

# Datasets
PreparedDataset.from_factory(n_samples=10000, channels=32, height=64, width=64)
LazyShapeDataset(n_samples=10000, channels=32, height=64, width=64)
```

## Vectorization Notes

Hot paths are vectorized — no Python loops in forward pass:
- CrossStateComposition alignment bias: gather sub-matrix + repeat_interleave
- Weighted pool: reshape to (B, n_active, tps, D), broadcast multiply, sum
- No .item() in any forward path (graph break + CUDA sync)
- Remaining .item() only in debug/inspection utilities

## Training Strategy

Patchwork runs all 8 states every pass with soft routing (cheap enough). Routing weights handle mixing. This validates topology differentiates states correctly.

Chunk (512 states) uses needs-based loading: router picks top-k active, only those compute. ModuleList with .tolist() on active indices (single sync instead of N per forward).

Per-state params stay large during training (learn navigation within topological region). At inference can compress because navigation already learned.

## Phil's Development Setup

- IDE: PyCharm
- Training: Google Colab
- Install: `pip install "git+https://github.com/AbstractEyes/glip-autoencoder.git"`
- Testing: uninstall + reinstall pattern for clean Colab state

## Common Mistakes to Avoid

- Don't treat geometric components as conventional layers with fancy names
- Don't replace KSimplexLinear with nn.Linear
- Don't assume alignment = distance (it's the opposite)
- Don't put .item() in forward paths
- Don't hardcode spatial dims — factory/datasets accept width/height/channels
- Don't put Dataset classes in the generator file — generator is a pure numpy factory
- Check actual function signatures before suggesting test code