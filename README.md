# GLIP — Geometric Linear Interpolative Patchwork

A deterministic topological token architecture for geometric deep learning.

GLIP replaces learned embeddings with mathematically determined geometric structures. Each token is a position in a Cantor topology — its behavior is assigned by the topology, not learned from data. The network learns *how to navigate* within its assigned geometric region, not *what it is*.

## Architecture

```
Input (e.g. VAE latent)
  → Patchifier (spatial → token sequence via KSimplexLinear)
  → NeedsBasedRouter (assigns topological position)
  → TopologicalStates (each: frozen simplex template + learned deformation)
  → CrossStateComposition (alignment-biased attention across states)
  → Output tokens
```

**Hierarchy** (each scale is a complete vocabulary):

| Scale | States | Tokens/State | Dims | Total Tokens | Params |
|-----------|--------|--------------|------|--------------|--------|
| Patchwork | 8 | 256 | 64 | 131,072 | ~270M |
| Chunk | 512 | 256 | 64 | 8,388,608 | ~17.3B |
| Sector | 512 Chunks | — | — | ~4.3B | — |

Chunk uses needs-based loading: only ~64 of 512 states are active per forward pass (~2.2B active params).

## Core Components

**KSimplexLinear** (§8) — Linear layer parameterized as k-simplex vertices + barycentric coordinates. 11.8% of equivalent dense parameters. Cayley-Menger validated.

**CantorTopology** — Deterministic assignment of states via Devil's Staircase. No parameters. Each state gets a unique branch path determining its geometric behavior. Pairwise alignment is precomputed — coarse matches are routing highways, fine matches are local structure.

**CayleyMengerValidator** (§4) — Structural invariant. If CM determinant fails, the geometry is invalid. Provides validity loss (penalize collapsed simplices) and consistency loss (encourage uniform structure). Standalone functions available for direct use.

**TopologicalState** — Single vocabulary entry: 256 tokens × 64 dims. Frozen regular simplex template with learned deformation anchored by α/√(k+1). Branch path determines vertex weighting. Internal self-attention across tokens.

## Installation

```bash
pip install "git+https://github.com/AbstractEyes/glip-autoencoder.git"
```

## Quick Start

```python
from geolip.scales.patchwork import Patchwork
import torch

model = Patchwork(in_channels=32, in_height=64, in_width=64)
x = torch.randn(1, 32, 64, 64)
out = model(x)

print(out.shape)       # torch.Size([1, 256, 64])
print(out.loss)        # CM validity loss
print(out)             # PatchworkOutput(shape=..., active=8 states, loss=...)
```

### Shape Factory

```python
from geolip.train.geometric_generator import GeometricShapeFactory

factory = GeometricShapeFactory(width=64, height=64, seed=42)
latent, labels, masks, geometry = factory(channels=32, n_shapes=3)
# latent: (32, 64, 64) numpy — simulated VAE latent
# labels: ['sphere', 'torus', 'cube']
```

### Training Data

```python
from geolip.train.dataloader import PreparedDataset, LazyShapeDataset, make_dataloader

# Pre-generate into memory
ds = PreparedDataset.from_factory(n_samples=10000, channels=32)

# Or generate on-the-fly (zero RAM, worker-friendly)
ds = LazyShapeDataset(n_samples=10000, channels=32)

loader = make_dataloader(ds, batch_size=16, num_workers=4)
```

## Package Structure

```
geolip/
├── core/
│   ├── ksimplex_linear.py    KSimplexLinear (§8)
│   ├── topology.py           CantorTopology (pure math, no params)
│   └── cayley_menger.py      CM determinant functions + validator
├── scales/
│   ├── patch.py              Patchifier (spatial → tokens)
│   ├── state.py              TopologicalState (256×64)
│   ├── patchwork.py          8×256×64 prototype
│   ├── chunk.py              512×256×64 full scale
│   └── sector.py             512 Chunks (placeholder)
├── script/
│   ├── chunk_test.py         Self-test
│   └── generator.py          Generation utilities
└── train/
    ├── geometric_generator.py   SDF shape factory (38 classes, numpy only)
    ├── voxel_generator.py       Voxel shape factory (5³ + 8×16×16)
    ├── dataloader.py            Dataset/DataLoader wrappers
    └── train_patchwork.py       Training loop
```

All `__init__.py` files are empty — use direct imports from the package directory.

## 38-Class Shape Vocabulary

Spans the full geometric spectrum: 0D point, 1D rigid/curved (lines, arcs, helix), 2D rigid/curved (triangles, squares, circles, discs), 3D rigid (tetrahedron, cube, octahedron, pentachoron), 3D curved solid/hollow/open (sphere, torus, cylinder, shell, bowl, saddle). Eight curvature types: none, convex, concave, cylindrical, conical, toroidal, hyperbolic, helical.

## License

Apache-2.0

## Author

AbstractPhil