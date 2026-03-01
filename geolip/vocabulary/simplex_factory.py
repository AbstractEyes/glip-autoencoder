"""
SimplexFactory
--------------
Factory for generating k-simplices with formula-based validation.

A k-simplex is the convex hull of k+1 affinely independent points.
Examples:
    - 0-simplex: point
    - 1-simplex: line segment (2 points)
    - 2-simplex: triangle (3 points)
    - 3-simplex: tetrahedron (4 points)
    - 4-simplex: pentachoron (5 points)

This factory generates simplices in d-dimensional embedding space and
validates them using the CayleyMengerFormula from the same vocabulary
package — no external dependency on geovocab2.

geolip.vocabulary.simplex_factory

License: MIT
"""

import math
import numpy as np
from typing import Optional, Tuple, Union

try:
    from .factory_base import FactoryBase, HAS_TORCH
except ImportError:
    from factory_base import FactoryBase, HAS_TORCH

if HAS_TORCH:
    import torch


class SimplexFactory(FactoryBase):
    """
    Generate k-simplices with configurable embedding dimension.

    Args:
        k: Simplex dimension (k+1 vertices)
        embed_dim: Embedding space dimension (must be >= k)
        method: Generation method ('random', 'regular', 'uniform')
        scale: Scaling factor for vertex coordinates
        seed: Optional seed for reproducibility
    """

    def __init__(
        self,
        k: int,
        embed_dim: int,
        method: str = "random",
        scale: float = 1.0,
        seed: Optional[int] = None,
    ):
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if embed_dim < k:
            raise ValueError(f"embed_dim ({embed_dim}) must be >= k ({k})")

        super().__init__(
            name=f"simplex_k{k}_d{embed_dim}",
            uid=f"factory.simplex.k{k}.d{embed_dim}.{method}",
        )
        self.k = k
        self.embed_dim = embed_dim
        self.method = method
        self.scale = scale
        self.num_vertices = k + 1
        self.seed = seed

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NumPy Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_numpy(
        self,
        *,
        dtype=np.float32,
        seed: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Build k-simplex using NumPy.

        Returns:
            Array of shape (k+1, embed_dim) representing simplex vertices
        """
        if self.seed is not None:
            seed = self.seed
        rng = np.random.default_rng(seed)

        if self.method == "random":
            vertices = self._generate_random(rng, dtype)
        elif self.method == "regular":
            vertices = self._generate_regular(dtype)
        elif self.method == "uniform":
            vertices = self._generate_uniform(rng, dtype)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return vertices * self.scale

    def _generate_random(self, rng, dtype) -> np.ndarray:
        """Random simplex via QR decomposition for affine independence."""
        raw = rng.standard_normal((self.num_vertices, self.embed_dim))
        q, r = np.linalg.qr(raw.T)
        vertices = q.T[:self.num_vertices]
        vertices = vertices - vertices.mean(axis=0, keepdims=True)
        return vertices.astype(dtype, copy=False)

    def _generate_regular(self, dtype) -> np.ndarray:
        """
        Regular simplex (all edges equal length = 1.0).

        Vertex i has coordinate i = sqrt((k+1)/k), all others = -1/k.
        Then center and normalize to unit edge length.
        """
        if self.k == 0:
            return np.zeros((1, self.embed_dim), dtype=dtype)

        min_dim = self.k + 1
        vertices_minimal = np.full(
            (self.num_vertices, min_dim), -1.0 / self.k, dtype=dtype
        )

        coef = np.sqrt((self.k + 1.0) / self.k)
        np.fill_diagonal(vertices_minimal, coef)

        if self.embed_dim > min_dim:
            vertices = np.zeros((self.num_vertices, self.embed_dim), dtype=dtype)
            vertices[:, :min_dim] = vertices_minimal
        else:
            vertices = vertices_minimal[:, :self.embed_dim]

        vertices = vertices - vertices.mean(axis=0, keepdims=True)

        edge_length = np.linalg.norm(vertices[1] - vertices[0])
        if edge_length > 1e-10:
            vertices = vertices / edge_length

        return vertices

    def _generate_uniform(self, rng, dtype) -> np.ndarray:
        """Uniform hypercube sampling with perturbation for independence."""
        vertices = rng.uniform(-1.0, 1.0, size=(self.num_vertices, self.embed_dim))

        for i in range(1, self.num_vertices):
            vertices[i] += 0.1 * i * np.eye(self.embed_dim)[i % self.embed_dim]

        return vertices.astype(dtype, copy=False)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PyTorch Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_torch(
        self,
        *,
        device: str = "cpu",
        dtype: Optional["torch.dtype"] = torch.float32,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "torch.Tensor":
        """Build k-simplex using PyTorch (direct on-device generation)."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for build_torch")

        seed = seed if seed is not None else self.seed
        target_dtype = dtype or self._infer_torch_dtype(device)
        dev = torch.device(device)

        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
        else:
            gen = None

        if self.method == "random":
            vertices = self._generate_random_torch(gen, target_dtype)
        elif self.method == "regular":
            vertices = self._generate_regular_torch(target_dtype)
        elif self.method == "uniform":
            vertices = self._generate_uniform_torch(gen, target_dtype)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return (vertices * self.scale).to(dev)

    def _generate_random_torch(self, gen, dtype) -> "torch.Tensor":
        """Random simplex using QR decomposition."""
        raw = torch.randn(
            (self.num_vertices, self.embed_dim), generator=gen, dtype=dtype
        )
        q, r = torch.linalg.qr(raw.T)
        vertices = q.T[:self.num_vertices]
        vertices = vertices - vertices.mean(dim=0, keepdim=True)
        return vertices

    def _generate_regular_torch(self, dtype) -> "torch.Tensor":
        """Regular simplex construction (vectorized, matches numpy)."""
        if self.k == 0:
            return torch.zeros((1, self.embed_dim), dtype=dtype)

        min_dim = self.k + 1
        vertices_minimal = torch.full(
            (self.num_vertices, min_dim), -1.0 / self.k, dtype=dtype
        )

        coef = float(np.sqrt((self.k + 1.0) / self.k))
        diag_size = min(self.num_vertices, min_dim)
        diag_indices = torch.arange(diag_size, dtype=torch.long)
        vertices_minimal[diag_indices, diag_indices] = coef

        if self.embed_dim > min_dim:
            vertices = torch.zeros((self.num_vertices, self.embed_dim), dtype=dtype)
            vertices[:, :min_dim] = vertices_minimal
        else:
            vertices = vertices_minimal[:, :self.embed_dim]

        vertices = vertices - vertices.mean(dim=0, keepdim=True)

        edge_length = torch.linalg.norm(vertices[1] - vertices[0])
        if edge_length > 1e-10:
            vertices = vertices / edge_length

        return vertices

    def _generate_uniform_torch(self, gen, dtype) -> "torch.Tensor":
        """Uniform hypercube sampling (vectorized)."""
        vertices = (
            torch.rand((self.num_vertices, self.embed_dim), generator=gen, dtype=dtype)
            * 2 - 1
        )

        if self.num_vertices > 1:
            eye = torch.eye(self.embed_dim, dtype=dtype)
            indices = torch.arange(1, self.num_vertices, dtype=torch.long)
            perturbations = (
                0.1 * indices.unsqueeze(-1).float() * eye[indices % self.embed_dim]
            )
            vertices[1:] += perturbations

        return vertices

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Validation — uses CayleyMengerFormula from vocabulary
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def validate(
        self,
        output: Union[np.ndarray, "torch.Tensor"],
    ) -> Tuple[bool, str]:
        """
        Validate simplex using CayleyMengerFormula.

        Checks:
            1. Shape is (k+1, embed_dim)
            2. No NaN/Inf values
            3. Non-degenerate (positive volume)
        """
        # Shape check
        expected_shape = (self.num_vertices, self.embed_dim)
        if output.shape != expected_shape:
            return False, f"Expected shape {expected_shape}, got {output.shape}"

        # Finiteness check
        if isinstance(output, np.ndarray):
            if not np.all(np.isfinite(output)):
                return False, "Contains NaN or Inf"
            verts_torch = torch.from_numpy(output.astype(np.float64))
        else:
            if not torch.all(torch.isfinite(output)):
                return False, "Contains NaN or Inf"
            verts_torch = output.detach().float()

        # Volume check via CayleyMengerFormula
        try:
            try:
                from .cayley_menger import CayleyMengerFormula
            except ImportError:
                from cayley_menger import CayleyMengerFormula

            formula = CayleyMengerFormula(eps=1e-10, validate_input=False)
            result = formula.compute_torch(verts_torch)

            is_degenerate = result["is_degenerate"].item()
            volume = result["volume"].item()

            if is_degenerate or volume < 1e-10:
                return False, f"Degenerate simplex (volume={volume:.2e})"

            return True, ""

        except ImportError:
            # Fallback: basic rank check
            if isinstance(output, np.ndarray):
                verts = output
            else:
                verts = output.cpu().numpy()

            translated = verts[1:] - verts[0]
            rank = np.linalg.matrix_rank(translated, tol=1e-6)

            if rank < self.k:
                return False, f"Not affinely independent (rank={rank}, need {self.k})"

            return True, ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Metadata
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def info(self):
        base_info = super().info()
        base_info.update({
            "description": f"k-simplex factory (k={self.k}, embed_dim={self.embed_dim})",
            "simplex_dimension": self.k,
            "num_vertices": self.num_vertices,
            "embedding_dimension": self.embed_dim,
            "generation_method": self.method,
            "scale": self.scale,
            "output_shape": (self.num_vertices, self.embed_dim),
        })
        return base_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Self-Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("SIMPLEX FACTORY SELF-TEST")
    print("=" * 70)

    # ── Test 1: Regular pentachoron (k=4) in R^10 ──
    print("\n[1] Regular pentachoron (k=4) in R^10")
    factory = SimplexFactory(k=4, embed_dim=10, method="regular", scale=1.0)
    penta = factory.build(backend="numpy", validate=True)
    print(f"  Shape: {penta.shape}")
    edges = []
    for i in range(5):
        for j in range(i + 1, 5):
            edges.append(np.linalg.norm(penta[i] - penta[j]))
    print(f"  Edge std: {np.std(edges):.8f} (should be ~0)")
    print(f"  Edge mean: {np.mean(edges):.6f}")

    # ── Test 2: Random triangle in R³ ──
    print("\n[2] Random triangle (k=2) in R^3")
    factory_2d = SimplexFactory(k=2, embed_dim=3, method="random", seed=42)
    tri = factory_2d.build(backend="numpy", validate=True)
    print(f"  Shape: {tri.shape}")
    print(f"  Centered: mean={tri.mean(axis=0)}")

    # ── Test 3: Torch backend ──
    if HAS_TORCH:
        print("\n[3] Torch backend — regular pentachoron")
        penta_t = factory.build(backend="torch", device="cpu", validate=True)
        print(f"  Shape: {penta_t.shape}, dtype: {penta_t.dtype}")

        if torch.cuda.is_available():
            print("\n[4] CUDA backend")
            penta_cuda = factory.build(backend="torch", device="cuda:0", validate=True)
            print(f"  Device: {penta_cuda.device}, dtype: {penta_cuda.dtype}")

    # ── Test 4: Validation failure ──
    print("\n[5] Degenerate validation")
    factory_test = SimplexFactory(k=2, embed_dim=3, method="random")
    bad = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    is_valid, msg = factory_test.validate(bad)
    print(f"  Collinear valid? {is_valid}")
    print(f"  Message: {msg}")

    # ── Test 5: CM integration ──
    print("\n[6] CayleyMengerFormula integration")
    try:
        from cayley_menger import CayleyMengerFormula

        formula = CayleyMengerFormula()
        penta_t = factory.build(backend="torch", device="cpu")
        result = formula(penta_t)
        print(f"  Volume: {result['volume'].item():.6f}")
        print(f"  Regularity: {result['regularity'].item():.6f}")
        print(f"  Is valid: {result['is_valid'].item()}")
    except ImportError:
        print("  (CayleyMengerFormula not available in this context)")

    print("\n" + "=" * 70)
    print("SimplexFactory ready for geolip.vocabulary")
    print("=" * 70)