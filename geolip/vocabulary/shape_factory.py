"""
SimpleShapeFactory
------------------
Factory for generating common geometric shapes as point clouds.

Shapes: cube, sphere, cylinder, pyramid, cone
Backends: NumPy, PyTorch (on-device)
Validation: integrated via shape_formulas (volume, quality, classification)

geolip.vocabulary.shape_factory

License: MIT
"""

import math
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any

try:
    from .factory_base import FactoryBase, HAS_TORCH
except ImportError:
    from factory_base import FactoryBase, HAS_TORCH

if HAS_TORCH:
    import torch

try:
    from .shape_formulas import (
        ShapeVolumeEstimator,
        ShapeSurfaceAreaEstimator,
        ShapeQualityMetrics,
        ShapeValidator,
        ShapeClassifier,
    )
    HAS_FORMULAS = True
except ImportError:
    try:
        from shape_formulas import (
            ShapeVolumeEstimator,
            ShapeSurfaceAreaEstimator,
            ShapeQualityMetrics,
            ShapeValidator,
            ShapeClassifier,
        )
        HAS_FORMULAS = True
    except ImportError:
        HAS_FORMULAS = False

SHAPE_TYPES = ("cube", "sphere", "cylinder", "pyramid", "cone")


class SimpleShapeFactory(FactoryBase):
    """
    Generate geometric shapes as point clouds.

    Args:
        shape_type: "cube", "sphere", "cylinder", "pyramid", "cone"
        embed_dim: Embedding dimension (>= 2, >= 3 for cylinder/pyramid/cone)
        resolution: Approximate number of points to generate
        scale: Scaling factor for shape dimensions
        validate_output: Run shape formulas on output
        quality_threshold: Minimum quality score for validation
    """

    def __init__(
        self,
        shape_type: str,
        embed_dim: int = 3,
        resolution: int = 100,
        scale: float = 1.0,
        validate_output: bool = False,
        quality_threshold: float = 0.7,
    ):
        if shape_type not in SHAPE_TYPES:
            raise ValueError(f"shape_type must be one of {SHAPE_TYPES}, got '{shape_type}'")
        if embed_dim < 2:
            raise ValueError(f"embed_dim must be >= 2, got {embed_dim}")
        if resolution < 4:
            raise ValueError(f"resolution must be >= 4, got {resolution}")

        super().__init__(
            name=f"shape_{shape_type}_d{embed_dim}",
            uid=f"factory.shape.{shape_type}.d{embed_dim}",
        )
        self.shape_type = shape_type
        self.embed_dim = embed_dim
        self.resolution = resolution
        self.scale = scale
        self.validate_output = validate_output
        self.quality_threshold = quality_threshold

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NumPy Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_numpy(self, *, dtype=np.float32, seed: Optional[int] = None, **kw) -> np.ndarray:
        rng = np.random.default_rng(seed)
        gen = getattr(self, f"_gen_{self.shape_type}_np")
        return gen(rng, dtype) * self.scale

    def _gen_cube_np(self, rng, dtype) -> np.ndarray:
        n_corners = 2 ** min(self.embed_dim, 10)
        corners = np.array(
            [[(-1) ** ((i >> j) & 1) for j in range(self.embed_dim)] for i in range(n_corners)],
            dtype=dtype,
        )
        n_face = max(4, self.resolution - n_corners)
        face = rng.uniform(-1, 1, size=(n_face, self.embed_dim)).astype(dtype)
        for i in range(n_face):
            d = rng.integers(0, self.embed_dim)
            face[i, d] = rng.choice([-1.0, 1.0])
        return np.vstack([corners, face])

    def _gen_sphere_np(self, rng, dtype) -> np.ndarray:
        pts = rng.standard_normal((self.resolution, self.embed_dim))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return (pts / (norms + 1e-10)).astype(dtype)

    def _gen_cylinder_np(self, rng, dtype) -> np.ndarray:
        if self.embed_dim < 3:
            raise ValueError("Cylinder requires embed_dim >= 3")
        n_s = int(0.7 * self.resolution)
        n_c = (self.resolution - n_s) // 2
        theta = rng.uniform(0, 2 * np.pi, n_s)
        z = rng.uniform(-1, 1, n_s)
        surface = np.column_stack([np.cos(theta), np.sin(theta), z])
        for cap_z, n in [(1.0, n_c), (-1.0, n_c)]:
            r = np.sqrt(rng.uniform(0, 1, n))
            t = rng.uniform(0, 2 * np.pi, n)
            cap = np.column_stack([r * np.cos(t), r * np.sin(t), np.full(n, cap_z)])
            surface = np.vstack([surface, cap])
        return self._embed(surface, dtype)

    def _gen_pyramid_np(self, rng, dtype) -> np.ndarray:
        if self.embed_dim < 3:
            raise ValueError("Pyramid requires embed_dim >= 3")
        apex = np.array([[0.0, 0.0, 1.0]])
        base_corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]], dtype=np.float64)
        n_face = (self.resolution - 5) // 4
        faces = []
        for i in range(4):
            c1, c2 = base_corners[i], base_corners[(i + 1) % 4]
            for _ in range(n_face):
                u, v = rng.random(2)
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                faces.append((1 - u - v) * apex[0] + u * c1 + v * c2)
        n_base = self.resolution - 5 - len(faces)
        base_xy = rng.uniform(-1, 1, size=(n_base, 2))
        base_pts = np.hstack([base_xy, np.full((n_base, 1), -1.0)])
        pts = np.vstack([apex, base_corners, np.array(faces), base_pts])
        return self._embed(pts, dtype)

    def _gen_cone_np(self, rng, dtype) -> np.ndarray:
        if self.embed_dim < 3:
            raise ValueError("Cone requires embed_dim >= 3")
        n_s = int(0.7 * self.resolution)
        n_b = self.resolution - n_s
        apex = np.array([[0.0, 0.0, 1.0]])
        z = rng.uniform(-1, 1, n_s)
        r = (1 - z) / 2
        theta = rng.uniform(0, 2 * np.pi, n_s)
        surface = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
        r_b = np.sqrt(rng.uniform(0, 1, n_b))
        t_b = rng.uniform(0, 2 * np.pi, n_b)
        base = np.column_stack([r_b * np.cos(t_b), r_b * np.sin(t_b), np.full(n_b, -1.0)])
        pts = np.vstack([apex, surface, base])
        return self._embed(pts, dtype)

    def _embed(self, pts_3d: np.ndarray, dtype) -> np.ndarray:
        if self.embed_dim > 3:
            out = np.zeros((len(pts_3d), self.embed_dim), dtype=dtype)
            out[:, :3] = pts_3d
            return out
        return pts_3d.astype(dtype)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PyTorch Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_torch(self, *, device: str = "cpu", dtype=None, seed: Optional[int] = None, **kw):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for build_torch")
        target_dtype = dtype or self._infer_torch_dtype(device)
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        fn = getattr(self, f"_gen_{self.shape_type}_pt")
        pts = fn(gen, target_dtype)
        return (pts * self.scale).to(device)

    def _gen_cube_pt(self, gen, dtype):
        n_corners = 2 ** min(self.embed_dim, 10)
        idx = torch.arange(n_corners, dtype=torch.long)
        corners = torch.zeros((n_corners, self.embed_dim), dtype=dtype)
        for j in range(self.embed_dim):
            corners[:, j] = (-1.0) ** ((idx >> j) & 1).float()
        n_f = max(4, self.resolution - n_corners)
        face = torch.rand((n_f, self.embed_dim), generator=gen, dtype=dtype) * 2 - 1
        for i in range(n_f):
            d = torch.randint(0, self.embed_dim, (1,), generator=gen).item()
            face[i, d] = torch.randint(0, 2, (1,), generator=gen).item() * 2.0 - 1.0
        return torch.cat([corners, face], dim=0)

    def _gen_sphere_pt(self, gen, dtype):
        pts = torch.randn((self.resolution, self.embed_dim), generator=gen, dtype=dtype)
        return pts / (torch.linalg.norm(pts, dim=1, keepdim=True) + 1e-10)

    def _gen_cylinder_pt(self, gen, dtype):
        if self.embed_dim < 3:
            raise ValueError("Cylinder requires embed_dim >= 3")
        n_s = int(0.7 * self.resolution)
        n_c = (self.resolution - n_s) // 2
        theta = torch.rand(n_s, generator=gen, dtype=dtype) * 2 * math.pi
        z = torch.rand(n_s, generator=gen, dtype=dtype) * 2 - 1
        surface = torch.stack([torch.cos(theta), torch.sin(theta), z], dim=1)
        caps = []
        for cap_z in [1.0, -1.0]:
            r = torch.sqrt(torch.rand(n_c, generator=gen, dtype=dtype))
            t = torch.rand(n_c, generator=gen, dtype=dtype) * 2 * math.pi
            cap = torch.stack([r * torch.cos(t), r * torch.sin(t),
                               torch.full((n_c,), cap_z, dtype=dtype)], dim=1)
            caps.append(cap)
        pts = torch.cat([surface] + caps, dim=0)
        return self._embed_pt(pts, dtype)

    def _gen_pyramid_pt(self, gen, dtype):
        if self.embed_dim < 3:
            raise ValueError("Pyramid requires embed_dim >= 3")
        apex = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
        base = torch.tensor([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]], dtype=dtype)
        n_f = (self.resolution - 5) // 4
        faces = []
        for i in range(4):
            c1, c2 = base[i], base[(i + 1) % 4]
            uv = torch.rand((n_f, 2), generator=gen, dtype=dtype)
            mask = (uv[:, 0] + uv[:, 1]) > 1
            uv[mask] = 1 - uv[mask]
            w = 1 - uv.sum(dim=1, keepdim=True)
            faces.append(w * apex + uv[:, 0:1] * c1 + uv[:, 1:2] * c2)
        n_b = self.resolution - 5 - n_f * 4
        base_xy = torch.rand((n_b, 2), generator=gen, dtype=dtype) * 2 - 1
        base_z = torch.full((n_b, 1), -1.0, dtype=dtype)
        pts = torch.cat([apex, base] + faces + [torch.cat([base_xy, base_z], dim=1)], dim=0)
        return self._embed_pt(pts, dtype)

    def _gen_cone_pt(self, gen, dtype):
        if self.embed_dim < 3:
            raise ValueError("Cone requires embed_dim >= 3")
        n_s = int(0.7 * self.resolution)
        n_b = self.resolution - n_s
        apex = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
        z = torch.rand(n_s, generator=gen, dtype=dtype) * 2 - 1
        r = (1 - z) / 2
        theta = torch.rand(n_s, generator=gen, dtype=dtype) * 2 * math.pi
        surface = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=1)
        r_b = torch.sqrt(torch.rand(n_b, generator=gen, dtype=dtype))
        t_b = torch.rand(n_b, generator=gen, dtype=dtype) * 2 * math.pi
        base = torch.stack([r_b * torch.cos(t_b), r_b * torch.sin(t_b),
                            -torch.ones(n_b, dtype=dtype)], dim=1)
        pts = torch.cat([apex, surface, base], dim=0)
        return self._embed_pt(pts, dtype)

    def _embed_pt(self, pts_3d, dtype):
        if self.embed_dim > 3:
            out = torch.zeros((len(pts_3d), self.embed_dim), dtype=dtype)
            out[:, :3] = pts_3d
            return out
        return pts_3d

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Validation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def compute_validation_metrics(self, output) -> Dict[str, Any]:
        """Full validation metrics via shape formulas."""
        if not HAS_FORMULAS:
            return {"validation_available": False}

        pts = torch.from_numpy(output).float() if isinstance(output, np.ndarray) else output
        results = {}

        vol = ShapeVolumeEstimator(method="analytical").forward(pts, self.shape_type)
        results["volume"] = {
            "value": vol["volume"].item(),
            "confidence": vol["confidence"].item(),
        }

        area = ShapeSurfaceAreaEstimator(method="analytical").forward(pts, self.shape_type)
        results["surface_area"] = {
            "value": area["surface_area"].item(),
            "area_to_volume_ratio": area["area_to_volume_ratio"].item(),
        }

        q = ShapeQualityMetrics().forward(pts)
        results["quality"] = {
            "overall": q["overall_quality"].item(),
            "uniformity": q["uniformity"].item(),
            "coverage": q["coverage"].item(),
            "outlier_fraction": q["outlier_fraction"].item(),
        }

        v = ShapeValidator(self.shape_type).forward(pts)
        results["validation"] = {
            "is_valid": v["is_valid"].item(),
            "score": v["validation_score"].item(),
        }

        c = ShapeClassifier().forward(pts)
        from .shape_formulas import SHAPE_NAMES  # noqa: avoid circular at module level
        predicted = SHAPE_NAMES[c["shape_type"].item()]
        results["classification"] = {
            "predicted_type": predicted,
            "expected_type": self.shape_type,
            "correct": predicted == self.shape_type,
            "confidence": c["confidence"].item(),
        }

        return results

    def validate(self, output) -> Tuple[bool, str]:
        if output.ndim != 2:
            return False, f"Expected 2D, got shape {output.shape}"
        if output.shape[1] != self.embed_dim:
            return False, f"Expected embed_dim={self.embed_dim}, got {output.shape[1]}"
        if output.shape[0] < 4:
            return False, f"Too few points: {output.shape[0]}"

        # Check spread only in spatial dims (3D shapes zero-pad higher dims)
        spatial_dims = min(3, self.embed_dim)
        if isinstance(output, np.ndarray):
            if not np.all(np.isfinite(output)):
                return False, "Contains NaN or Inf"
            if np.any(np.std(output[:, :spatial_dims], axis=0) < 1e-8):
                return False, "Degenerate (zero spread)"
        else:
            if not torch.all(torch.isfinite(output)):
                return False, "Contains NaN or Inf"
            if torch.any(torch.std(output[:, :spatial_dims], dim=0) < 1e-8):
                return False, "Degenerate (zero spread)"

        if self.validate_output and HAS_FORMULAS:
            metrics = self.compute_validation_metrics(output)
            if "quality" in metrics and metrics["quality"]["overall"] < self.quality_threshold:
                return False, f"Quality {metrics['quality']['overall']:.3f} < threshold {self.quality_threshold}"
            if "validation" in metrics and not metrics["validation"]["is_valid"]:
                return False, f"Validation failed (score: {metrics['validation']['score']:.3f})"

        return True, ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Metadata
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def info(self):
        base = super().info()
        base.update({
            "description": f"{self.shape_type} point cloud factory (d={self.embed_dim})",
            "shape_type": self.shape_type,
            "embedding_dimension": self.embed_dim,
            "resolution": self.resolution,
            "scale": self.scale,
            "output_shape": f"(~{self.resolution}, {self.embed_dim})",
            "formulas_available": HAS_FORMULAS,
        })
        return base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Self-Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("SIMPLE SHAPE FACTORY SELF-TEST")
    print("=" * 70)

    for shape in SHAPE_TYPES:
        print(f"\n[{shape.upper()}] 3D, 50 pts")
        factory = SimpleShapeFactory(shape, embed_dim=3, resolution=50)
        pts_np = factory.build(backend="numpy", seed=42, validate=True)
        print(f"  shape={pts_np.shape}, range=[{pts_np.min():.3f}, {pts_np.max():.3f}]")

        if HAS_TORCH:
            pts_t = factory.build(backend="torch", device="cpu", seed=42, validate=True)
            print(f"  torch shape={pts_t.shape}, dtype={pts_t.dtype}")

    # Validation metrics
    if HAS_FORMULAS:
        print("\n[VALIDATION] Sphere with formula checks")
        factory = SimpleShapeFactory("sphere", embed_dim=3, resolution=200, validate_output=True)
        pts = factory.build(backend="numpy", seed=42, validate=True)
        metrics = factory.compute_validation_metrics(pts)
        print(f"  volume: {metrics['volume']['value']:.4f}")
        print(f"  quality: {metrics['quality']['overall']:.4f}")
        print(f"  classified as: {metrics['classification']['predicted_type']}")
        print(f"  correct: {metrics['classification']['correct']}")

    if HAS_TORCH and torch.cuda.is_available():
        print("\n[CUDA] Cylinder")
        cyl = SimpleShapeFactory("cylinder", embed_dim=3, resolution=200)
        pts_cuda = cyl.build(backend="torch", device="cuda:0", validate=True)
        print(f"  device={pts_cuda.device}, shape={pts_cuda.shape}")

    print(f"\n{'=' * 70}")
    print("SimpleShapeFactory ready for geolip.vocabulary")
    print(f"{'=' * 70}")