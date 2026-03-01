"""
SceneBuilder
------------
Multi-shape scene composition in 5D continuous space.

Generates overlapping deformed shapes in [-1,1]^5 with rich metadata
for training geometric deep learning models on shape identity, deformation
invariance, and spatial relationship understanding.

Pipeline:
    1. Pick 1-N shapes from SimpleShapeFactory
    2. Generate base point cloud per shape
    3. Apply random deformation (stretch/twist/taper/noise/shear/bend)
    4. Embed into 5D via random SO(5) rotation (distributes structure)
    5. Scale to structural boundary, place in [-1,1]^5
    6. Merge points, compute overlap, build labels

Output format (per scene):
    points:        (N, 5)     — point cloud in [-1,1]^5
    scene_labels:  (C,)       — multi-hot: which shape types present
    point_labels:  (N, C)     — per-point: which shape type each point belongs to
    overlap_count: (N,)       — how many shapes each point participates in
    deform_params: (N_shapes, 8) — [type_idx, magnitude, axis, scale, pos_x..pos_z]
    rotations:     (N_shapes, 5, 5) — per-shape SO(5) rotation matrices

The 5D space is pentachoron-native: 5 vertices of a 4-simplex span exactly 5
dimensions. Shapes embedded here via random rotation spread geometric structure
across all dims, forcing the model to disentangle via simplex operations.

geolip.vocabulary.scene_builder

License: MIT
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Generator

try:
    from .factory_base import FactoryBase, HAS_TORCH
except ImportError:
    from factory_base import FactoryBase, HAS_TORCH

try:
    from .shape_factory import SimpleShapeFactory, SHAPE_TYPES
except ImportError:
    from shape_factory import SimpleShapeFactory, SHAPE_TYPES

if HAS_TORCH:
    import torch
    from torch import Tensor

# Number of base shape types
NUM_SHAPE_TYPES = len(SHAPE_TYPES)  # 5: cube, sphere, cylinder, pyramid, cone

# Deformation catalog
DEFORMATION_TYPES = ("stretch", "twist", "taper", "noise", "shear", "bend")
NUM_DEFORM_TYPES = len(DEFORMATION_TYPES)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5D ROTATION UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _givens_rotation_np(n: int, i: int, j: int, theta: float) -> np.ndarray:
    """Givens rotation matrix in the (i,j) plane."""
    G = np.eye(n, dtype=np.float32)
    c, s = math.cos(theta), math.sin(theta)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G


def random_rotation_5d_np(rng: np.random.Generator) -> np.ndarray:
    """Random SO(5) rotation via composition of Givens rotations in all 10 planes.

    C(5,2) = 10 planes: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
    Each gets a random angle in [0, 2π). Composed left-to-right.
    """
    R = np.eye(5, dtype=np.float32)
    for i in range(5):
        for j in range(i + 1, 5):
            theta = rng.uniform(0, 2 * math.pi)
            R = R @ _givens_rotation_np(5, i, j, theta)
    return R


def random_rotation_5d_torch(gen: "torch.Generator" = None) -> "Tensor":
    """Random SO(5) rotation as torch tensor."""
    # Generate on CPU, angles via torch
    R = torch.eye(5, dtype=torch.float32)
    for i in range(5):
        for j in range(i + 1, 5):
            theta = torch.rand(1, generator=gen).item() * 2 * math.pi
            c, s = math.cos(theta), math.sin(theta)
            G = torch.eye(5, dtype=torch.float32)
            G[i, i] = c
            G[j, j] = c
            G[i, j] = -s
            G[j, i] = s
            R = R @ G
    return R


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHAPE DEFORMATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ShapeDeformer:
    """Apply geometric deformations to point clouds.

    All methods are static — operate on (N, D) arrays/tensors.
    Deformations work in any dimension D >= 2.
    """

    @staticmethod
    def stretch(points: np.ndarray, axis: int, factor: float) -> np.ndarray:
        """Non-uniform scaling along one axis.

        factor > 1: elongate, factor < 1: compress.
        """
        out = points.copy()
        out[:, axis] *= factor
        return out

    @staticmethod
    def twist(points: np.ndarray, axis: int, angle_per_unit: float) -> np.ndarray:
        """Rotation that increases linearly along axis.

        Points farther along `axis` get rotated more.
        Operates in the two lowest non-axis dimensions.
        """
        out = points.copy()
        D = points.shape[1]
        # Pick two dims to rotate in (not the twist axis)
        dims = [d for d in range(min(D, 5)) if d != axis][:2]
        if len(dims) < 2:
            return out
        d0, d1 = dims[0], dims[1]
        # Angle proportional to position along axis
        t = points[:, axis]  # [-1, 1] typically
        angles = t * angle_per_unit
        c, s = np.cos(angles), np.sin(angles)
        x, y = out[:, d0].copy(), out[:, d1].copy()
        out[:, d0] = c * x - s * y
        out[:, d1] = s * x + c * y
        return out

    @staticmethod
    def taper(points: np.ndarray, axis: int, taper_factor: float) -> np.ndarray:
        """Progressive scaling: points shrink toward one end of axis.

        taper_factor controls how aggressively the shape narrows.
        At axis_min: full scale. At axis_max: scaled by (1 - taper_factor).
        """
        out = points.copy()
        t = points[:, axis]
        # Normalize to [0, 1]
        t_min, t_max = t.min(), t.max()
        t_norm = (t - t_min) / (t_max - t_min + 1e-10)
        scale = 1.0 - taper_factor * t_norm
        scale = np.clip(scale, 0.05, 2.0)
        for d in range(points.shape[1]):
            if d != axis:
                out[:, d] *= scale
        return out

    @staticmethod
    def noise(points: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
        """Gaussian perturbation of all coordinates."""
        return points + rng.normal(0, sigma, size=points.shape).astype(points.dtype)

    @staticmethod
    def shear(points: np.ndarray, src_axis: int, dst_axis: int, factor: float) -> np.ndarray:
        """Off-diagonal deformation: dst += factor * src."""
        out = points.copy()
        out[:, dst_axis] += factor * points[:, src_axis]
        return out

    @staticmethod
    def bend(points: np.ndarray, axis: int, curvature: float) -> np.ndarray:
        """Curvature deformation: points curve away from axis.

        Applies circular bending in the plane of axis and the next dimension.
        """
        out = points.copy()
        D = points.shape[1]
        bend_dim = (axis + 1) % D
        t = points[:, axis]
        # Circular arc: offset = R - R*cos(t*curvature), new_axis = R*sin(t*curvature)
        R = 1.0 / (abs(curvature) + 1e-6)
        angles = t * curvature
        out[:, axis] = R * np.sin(angles)
        out[:, bend_dim] += R * (1.0 - np.cos(angles))
        return out

    @classmethod
    def apply(
        cls,
        points: np.ndarray,
        deform_type: str,
        magnitude: float,
        axis: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Apply a named deformation.

        Args:
            points: (N, D) point cloud
            deform_type: one of DEFORMATION_TYPES
            magnitude: strength of deformation (0 = none, 1 = strong)
            axis: primary axis for axis-dependent deformations
            rng: random generator (needed for noise)

        Returns:
            Deformed (N, D) point cloud.
        """
        if rng is None:
            rng = np.random.default_rng()

        if deform_type == "stretch":
            # factor in [0.5, 2.0] scaled by magnitude
            factor = 1.0 + magnitude * (rng.uniform(-1, 1))
            factor = max(0.3, min(3.0, factor))
            return cls.stretch(points, axis, factor)

        elif deform_type == "twist":
            angle = magnitude * math.pi  # up to π twist
            return cls.twist(points, axis, angle)

        elif deform_type == "taper":
            return cls.taper(points, axis, magnitude * 0.8)

        elif deform_type == "noise":
            sigma = magnitude * 0.1
            return cls.noise(points, sigma, rng)

        elif deform_type == "shear":
            D = points.shape[1]
            dst = (axis + 1) % D
            return cls.shear(points, axis, dst, magnitude * 0.5)

        elif deform_type == "bend":
            curvature = magnitude * 2.0
            return cls.bend(points, axis, curvature)

        else:
            raise ValueError(f"Unknown deformation: {deform_type}")

    @classmethod
    def random_deformation(
        cls,
        points: np.ndarray,
        rng: np.random.Generator,
        magnitude: float = 0.3,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply a random deformation and return metadata.

        Returns:
            (deformed_points, deformation_info)
        """
        deform_type = rng.choice(DEFORMATION_TYPES)
        axis = rng.integers(0, points.shape[1])
        mag = rng.uniform(0.1, magnitude)
        deformed = cls.apply(points, deform_type, mag, axis, rng)
        info = {
            "type": deform_type,
            "type_idx": DEFORMATION_TYPES.index(deform_type),
            "magnitude": float(mag),
            "axis": int(axis),
        }
        return deformed, info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCENE BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SceneBuilder(FactoryBase):
    """Build multi-shape scenes in 5D continuous space.

    Generates scenes of overlapping deformed shapes with rich metadata
    for training classification, deformation regression, and spatial
    relationship understanding.

    The 5D space is pentachoron-native: each shape is generated in 3D,
    then lifted to 5D via random SO(5) rotation. This distributes geometric
    structure across all dimensions, forcing the model to disentangle shape
    identity from embedding orientation — exactly the task KSimplexLinear
    and pentachoron crystals are designed for.

    Structural boundary constraint: each shape is scaled so its bounding
    hypervolume in 5D fits within [-1, 1]^5, proportional to the shape's
    geometric complexity.

    Args:
        embed_dim: Dimension of output space (default 5 for pentachoron-native)
        min_shapes: Minimum shapes per scene
        max_shapes: Maximum shapes per scene
        points_per_scene: Total points per scene (distributed across shapes)
        deform_probability: Probability of deforming each shape [0, 1]
        deform_magnitude: Maximum deformation strength
        overlap_margin: How close shape centers can be (lower = more overlap)
        scale_range: (min, max) scale factor for shapes
        structural_padding: fraction of [-1,1] to reserve as boundary (prevents clipping)
    """

    def __init__(
        self,
        embed_dim: int = 5,
        min_shapes: int = 1,
        max_shapes: int = 5,
        points_per_scene: int = 1024,
        deform_probability: float = 0.5,
        deform_magnitude: float = 0.3,
        overlap_margin: float = 0.3,
        scale_range: Tuple[float, float] = (0.2, 0.6),
        structural_padding: float = 0.1,
    ):
        super().__init__(
            name=f"scene_builder_d{embed_dim}",
            uid=f"factory.scene.d{embed_dim}",
        )
        self.embed_dim = embed_dim
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes
        self.points_per_scene = points_per_scene
        self.deform_probability = deform_probability
        self.deform_magnitude = deform_magnitude
        self.overlap_margin = overlap_margin
        self.scale_range = scale_range
        self.structural_padding = structural_padding

    # ── Core generation (NumPy) ──────────────────────────────────

    def build_numpy(
        self,
        *,
        dtype=np.float32,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build one scene.

        Returns dict with:
            points:        (N, embed_dim)       float32
            scene_labels:  (NUM_SHAPE_TYPES,)   float32 multi-hot
            point_labels:  (N, NUM_SHAPE_TYPES)  float32 per-point multi-hot
            overlap_count: (N,)                  int32 per-point overlap count
            shape_meta:    list of per-shape dicts
            n_shapes:      int
        """
        rng = np.random.default_rng(seed)

        # Decide number of shapes
        n_shapes = rng.integers(self.min_shapes, self.max_shapes + 1)

        # Distribute points across shapes (roughly equal, randomized)
        points_per_shape = self._distribute_points(n_shapes, rng)

        all_points = []       # list of (Ni, embed_dim) arrays
        all_type_indices = []  # which shape type each batch of points belongs to
        shape_meta = []

        for i in range(n_shapes):
            # Pick shape type
            type_idx = rng.integers(0, NUM_SHAPE_TYPES)
            shape_type = SHAPE_TYPES[type_idx]

            # Generate base shape in 3D
            n_pts = points_per_shape[i]
            factory = SimpleShapeFactory(
                shape_type=shape_type,
                embed_dim=3,
                resolution=n_pts,
                scale=1.0,
            )
            pts_3d = factory.build(backend="numpy", seed=int(rng.integers(0, 2**31)))

            # Apply deformation (probabilistic)
            deform_info = None
            if rng.random() < self.deform_probability:
                pts_3d, deform_info = ShapeDeformer.random_deformation(
                    pts_3d, rng, self.deform_magnitude,
                )

            # Embed 3D → 5D (or embed_dim)
            pts_embed = self._embed_to_nd(pts_3d, rng, dtype)

            # Apply random SO(n) rotation to distribute across all dims
            rotation = self._random_rotation(rng)
            pts_embed = pts_embed @ rotation.T

            # Scale to structural boundary
            scale = rng.uniform(*self.scale_range)
            pts_embed = self._apply_structural_boundary(pts_embed, scale)

            # Place in scene (random position within bounds)
            position = self._random_position(pts_embed, scale, rng)
            pts_embed = pts_embed + position

            # Clip to [-1, 1] (hard boundary)
            pts_embed = np.clip(pts_embed, -1.0, 1.0)

            all_points.append(pts_embed.astype(dtype))
            all_type_indices.append(np.full(len(pts_embed), type_idx, dtype=np.int32))

            shape_meta.append({
                "type": shape_type,
                "type_idx": int(type_idx),
                "n_points": len(pts_embed),
                "position": position.copy(),
                "scale": float(scale),
                "rotation": rotation.copy(),
                "deformation": deform_info,
            })

        # Merge all points
        points = np.concatenate(all_points, axis=0)
        type_indices = np.concatenate(all_type_indices, axis=0)

        # Pad or subsample to exact points_per_scene
        points, type_indices = self._fix_point_count(
            points, type_indices, rng, dtype,
        )

        # Build labels
        scene_labels = np.zeros(NUM_SHAPE_TYPES, dtype=dtype)
        point_labels = np.zeros((self.points_per_scene, NUM_SHAPE_TYPES), dtype=dtype)

        for meta in shape_meta:
            scene_labels[meta["type_idx"]] = 1.0

        for idx in range(self.points_per_scene):
            if idx < len(type_indices):
                point_labels[idx, type_indices[idx]] = 1.0

        # Compute overlap: for each point, count how many shapes' bounding
        # regions contain it
        overlap_count = self._compute_overlap(
            points, shape_meta, dtype,
        )

        return {
            "points": points,
            "scene_labels": scene_labels,
            "point_labels": point_labels,
            "overlap_count": overlap_count,
            "shape_meta": shape_meta,
            "n_shapes": n_shapes,
        }

    # ── PyTorch backend ──────────────────────────────────────────

    def build_torch(
        self,
        *,
        device: str = "cpu",
        dtype=None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Build one scene as torch tensors."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for build_torch")

        result_np = self.build_numpy(seed=seed)

        target_dtype = dtype or self._infer_torch_dtype(device)
        dev = torch.device(device)

        return {
            "points": torch.from_numpy(result_np["points"]).to(target_dtype).to(dev),
            "scene_labels": torch.from_numpy(result_np["scene_labels"]).to(target_dtype).to(dev),
            "point_labels": torch.from_numpy(result_np["point_labels"]).to(target_dtype).to(dev),
            "overlap_count": torch.from_numpy(result_np["overlap_count"]).to(dev),
            "shape_meta": result_np["shape_meta"],
            "n_shapes": result_np["n_shapes"],
        }

    # ── Batch / stream interface ─────────────────────────────────

    def stream(
        self,
        n: int,
        seed: Optional[int] = None,
        backend: str = "numpy",
        device: str = "cpu",
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield n scenes one at a time."""
        rng_seeds = np.random.default_rng(seed)
        for _ in range(n):
            s = int(rng_seeds.integers(0, 2**31))
            if backend == "torch":
                yield self.build_torch(device=device, seed=s)
            else:
                yield self.build_numpy(seed=s)

    def batch(
        self,
        n: int,
        seed: Optional[int] = None,
        backend: str = "numpy",
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Build n scenes and stack into batched tensors.

        Returns:
            points:        (B, N, D)
            scene_labels:  (B, C)
            point_labels:  (B, N, C)
            overlap_count: (B, N)
            shape_meta:    list of B lists
            n_shapes:      (B,) array/tensor
        """
        scenes = list(self.stream(n, seed=seed, backend=backend, device=device))

        if backend == "torch" and HAS_TORCH:
            return {
                "points": torch.stack([s["points"] for s in scenes]),
                "scene_labels": torch.stack([s["scene_labels"] for s in scenes]),
                "point_labels": torch.stack([s["point_labels"] for s in scenes]),
                "overlap_count": torch.stack([s["overlap_count"] for s in scenes]),
                "shape_meta": [s["shape_meta"] for s in scenes],
                "n_shapes": torch.tensor([s["n_shapes"] for s in scenes]),
            }
        else:
            return {
                "points": np.stack([s["points"] for s in scenes]),
                "scene_labels": np.stack([s["scene_labels"] for s in scenes]),
                "point_labels": np.stack([s["point_labels"] for s in scenes]),
                "overlap_count": np.stack([s["overlap_count"] for s in scenes]),
                "shape_meta": [s["shape_meta"] for s in scenes],
                "n_shapes": np.array([s["n_shapes"] for s in scenes]),
            }

    # ── Internal helpers ─────────────────────────────────────────

    def _distribute_points(self, n_shapes: int, rng) -> List[int]:
        """Distribute points_per_scene across n_shapes with some randomness."""
        base = self.points_per_scene // n_shapes
        remainder = self.points_per_scene - base * n_shapes
        counts = [base] * n_shapes

        # Randomly allocate remainder
        for i in rng.choice(n_shapes, size=remainder, replace=False):
            counts[i] += 1

        # Add variance (±20%)
        for i in range(n_shapes):
            jitter = rng.integers(-counts[i] // 5, counts[i] // 5 + 1)
            counts[i] = max(10, counts[i] + jitter)

        return counts

    def _embed_to_nd(
        self,
        pts_3d: np.ndarray,
        rng: np.random.Generator,
        dtype,
    ) -> np.ndarray:
        """Embed 3D points into embed_dim with structural features.

        dims 0-2: spatial coordinates
        dim 3:    local curvature estimate (from k-NN angle spread)
        dim 4:    local density (inverse avg neighbor distance)
        dims 5+:  zero (if embed_dim > 5)
        """
        N = len(pts_3d)
        pts = np.zeros((N, self.embed_dim), dtype=dtype)
        pts[:, :3] = pts_3d

        if self.embed_dim < 4:
            return pts

        # Compute k-NN once for both curvature (dim 3) and density (dim 4)
        nn_dists = None
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts_3d)
            k = min(8, N - 1)
            nn_dists, _ = tree.query(pts_3d, k=k + 1)
            nn_dists = nn_dists[:, 1:]  # exclude self
        except (ImportError, Exception):
            pass

        if nn_dists is not None:
            # Dim 3: local curvature (std of neighbor distances / mean)
            curvature = np.std(nn_dists, axis=1) / (np.mean(nn_dists, axis=1) + 1e-10)
            c_max = curvature.max() + 1e-10
            pts[:, 3] = (curvature / c_max) * 2 - 1

            # Dim 4: local density (inverse avg neighbor distance)
            if self.embed_dim >= 5:
                avg_dist = np.mean(nn_dists, axis=1)
                density = 1.0 / (avg_dist + 1e-10)
                d_max = density.max() + 1e-10
                pts[:, 4] = (density / d_max) * 2 - 1
        else:
            # Fallback: small random structural features (better than zeros)
            pts[:, 3] = rng.uniform(-0.3, 0.3, size=N).astype(dtype)
            if self.embed_dim >= 5:
                pts[:, 4] = rng.uniform(-0.3, 0.3, size=N).astype(dtype)

        return pts

    def _random_rotation(self, rng: np.random.Generator) -> np.ndarray:
        """Random SO(embed_dim) rotation."""
        if self.embed_dim == 5:
            return random_rotation_5d_np(rng)
        else:
            # General: QR decomposition of random matrix
            M = rng.standard_normal((self.embed_dim, self.embed_dim)).astype(np.float32)
            Q, R = np.linalg.qr(M)
            # Ensure proper rotation (det = +1)
            Q = Q * np.sign(np.diag(R))
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            return Q

    def _apply_structural_boundary(
        self,
        points: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Scale shape so its bounding hypervolume fits within structural limits.

        The structural boundary is determined by the shape's own extent:
        larger/more complex shapes get scaled down more to fit [-1,1]^D.
        """
        # Center the shape
        center = points.mean(axis=0)
        pts = points - center

        # Find max extent in any dimension
        max_extent = np.abs(pts).max()
        if max_extent < 1e-10:
            return pts

        # Scale so max extent = scale * (1 - padding)
        target = scale * (1.0 - self.structural_padding)
        pts = pts * (target / max_extent)

        return pts

    def _random_position(
        self,
        points: np.ndarray,
        scale: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Random position within [-1,1]^D, accounting for shape extent."""
        # Shape extent after scaling
        max_extent = np.abs(points).max(axis=0)

        # Position range: ensure shape stays within [-1, 1]
        pos_range = 1.0 - max_extent - self.structural_padding
        pos_range = np.clip(pos_range, 0.0, 1.0)

        position = rng.uniform(-pos_range, pos_range).astype(np.float32)
        return position

    def _fix_point_count(
        self,
        points: np.ndarray,
        type_indices: np.ndarray,
        rng: np.random.Generator,
        dtype,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad or subsample to exact points_per_scene."""
        N = len(points)
        target = self.points_per_scene

        if N == target:
            return points, type_indices
        elif N > target:
            # Subsample
            idx = rng.choice(N, size=target, replace=False)
            return points[idx], type_indices[idx]
        else:
            # Pad by duplicating random existing points (with small jitter)
            n_pad = target - N
            pad_idx = rng.choice(N, size=n_pad, replace=True)
            pad_pts = points[pad_idx] + rng.normal(0, 1e-4, size=(n_pad, self.embed_dim)).astype(dtype)
            pad_types = type_indices[pad_idx]
            return (
                np.concatenate([points, pad_pts], axis=0),
                np.concatenate([type_indices, pad_types], axis=0),
            )

    def _compute_overlap(
        self,
        points: np.ndarray,
        shape_meta: List[Dict],
        dtype,
    ) -> np.ndarray:
        """Compute per-point overlap count.

        For each point, count how many shapes' bounding hyperspheres contain it.
        """
        N = len(points)
        overlap = np.zeros(N, dtype=np.int32)

        for meta in shape_meta:
            pos = meta["position"]
            scale = meta["scale"]

            # Bounding radius: scale * sqrt(embed_dim) covers the hypercube diagonal
            radius = scale * math.sqrt(self.embed_dim) * (1.0 + self.structural_padding)

            # Distance from each point to shape center
            dists = np.linalg.norm(points - pos, axis=1)
            overlap += (dists < radius).astype(np.int32)

        return overlap

    # ── Validation ───────────────────────────────────────────────

    def validate(self, output) -> Tuple[bool, str]:
        """Validate scene output."""
        if not isinstance(output, dict):
            return False, f"Expected dict, got {type(output)}"

        pts = output.get("points")
        if pts is None:
            return False, "Missing 'points' key"

        if isinstance(pts, np.ndarray):
            shape = pts.shape
        elif HAS_TORCH and isinstance(pts, torch.Tensor):
            shape = tuple(pts.shape)
        else:
            return False, f"Unknown points type: {type(pts)}"

        if len(shape) != 2 or shape[0] != self.points_per_scene or shape[1] != self.embed_dim:
            return False, f"Expected shape ({self.points_per_scene}, {self.embed_dim}), got {shape}"

        return True, ""

    def info(self) -> Dict[str, Any]:
        base = super().info()
        base.update({
            "description": f"Scene builder: {self.min_shapes}-{self.max_shapes} shapes in {self.embed_dim}D",
            "embed_dim": self.embed_dim,
            "min_shapes": self.min_shapes,
            "max_shapes": self.max_shapes,
            "points_per_scene": self.points_per_scene,
            "shape_types": list(SHAPE_TYPES),
            "deformation_types": list(DEFORMATION_TYPES),
            "deform_probability": self.deform_probability,
            "output_keys": ["points", "scene_labels", "point_labels", "overlap_count", "shape_meta"],
        })
        return base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCENE DATASET (for training)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if HAS_TORCH:
    class SceneDataset(torch.utils.data.Dataset):
        """Lazy on-the-fly scene generation for DataLoader.

        Each __getitem__ generates a fresh scene with a deterministic seed
        derived from the index, so epochs are reproducible.

        Args:
            builder: SceneBuilder instance
            length: Dataset length (scenes per epoch)
            base_seed: Seed for reproducibility (scene i gets seed base_seed + i)
            device: Target device for tensors
        """

        def __init__(
            self,
            builder: SceneBuilder,
            length: int = 10000,
            base_seed: int = 0,
            device: str = "cpu",
        ):
            self.builder = builder
            self.length = length
            self.base_seed = base_seed
            self.device = device

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, idx: int) -> Dict[str, "Tensor"]:
            scene = self.builder.build_torch(
                device=self.device,
                seed=self.base_seed + idx,
            )
            # Return only stackable tensors for DataLoader collation
            return {
                "points": scene["points"],
                "scene_labels": scene["scene_labels"],
                "point_labels": scene["point_labels"],
                "overlap_count": scene["overlap_count"],
            }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SELF-TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("SCENE BUILDER SELF-TEST")
    print("=" * 70)

    # --- Deformer ---
    print("\n── ShapeDeformer ──")
    pts = np.random.randn(50, 5).astype(np.float32) * 0.5
    rng = np.random.default_rng(42)

    for dt in DEFORMATION_TYPES:
        deformed = ShapeDeformer.apply(pts, dt, magnitude=0.3, axis=0, rng=rng)
        diff = np.abs(deformed - pts).mean()
        print(f"  {dt:8s}: mean_diff={diff:.4f}, shape={deformed.shape}")

    deformed, info = ShapeDeformer.random_deformation(pts, rng, magnitude=0.5)
    print(f"  random:  type={info['type']}, mag={info['magnitude']:.3f}, axis={info['axis']}")

    # --- SO(5) rotation ---
    print("\n── SO(5) Rotation ──")
    R = random_rotation_5d_np(rng)
    det = np.linalg.det(R)
    ortho_err = np.abs(R @ R.T - np.eye(5)).max()
    print(f"  det={det:.6f} (should be 1.0), ortho_err={ortho_err:.2e}")

    # --- SceneBuilder single scene ---
    print("\n── SceneBuilder (single) ──")
    builder = SceneBuilder(
        embed_dim=5,
        min_shapes=2,
        max_shapes=4,
        points_per_scene=512,
        deform_probability=0.5,
        deform_magnitude=0.3,
    )

    scene = builder.build(backend="numpy", seed=42)
    print(f"  points:        {scene['points'].shape}")
    print(f"  scene_labels:  {scene['scene_labels']} (sum={scene['scene_labels'].sum():.0f})")
    print(f"  point_labels:  {scene['point_labels'].shape}")
    print(f"  overlap_count: min={scene['overlap_count'].min()}, max={scene['overlap_count'].max()}")
    print(f"  n_shapes:      {scene['n_shapes']}")
    print(f"  point range:   [{scene['points'].min():.3f}, {scene['points'].max():.3f}]")

    for i, meta in enumerate(scene["shape_meta"]):
        deform_str = meta["deformation"]["type"] if meta["deformation"] else "none"
        print(f"    shape {i}: {meta['type']:10s} scale={meta['scale']:.2f} deform={deform_str}")

    valid, msg = builder.validate(scene)
    print(f"  valid: {valid} {msg}")

    # --- Batch ---
    print("\n── SceneBuilder (batch) ──")
    batch = builder.batch(8, seed=42, backend="numpy")
    print(f"  batch points:  {batch['points'].shape}")
    print(f"  batch labels:  {batch['scene_labels'].shape}")
    print(f"  batch overlap: {batch['overlap_count'].shape}")
    print(f"  n_shapes:      {batch['n_shapes']}")

    # --- Torch ---
    if HAS_TORCH:
        print("\n── SceneBuilder (torch) ──")
        scene_t = builder.build(backend="torch", seed=42)
        print(f"  points:  {scene_t['points'].shape} dtype={scene_t['points'].dtype}")
        print(f"  labels:  {scene_t['scene_labels'].shape}")

        batch_t = builder.batch(4, seed=42, backend="torch")
        print(f"  batch:   {batch_t['points'].shape}")

        # Dataset
        print("\n── SceneDataset ──")
        ds = SceneDataset(builder, length=100, base_seed=0)
        sample = ds[0]
        print(f"  sample keys: {list(sample.keys())}")
        print(f"  points: {sample['points'].shape}")

        # DataLoader
        loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=0)
        batch_dl = next(iter(loader))
        print(f"  DataLoader batch: points={batch_dl['points'].shape}")

        if torch.cuda.is_available():
            scene_gpu = builder.build(backend="torch", device="cuda:0", seed=42)
            print(f"\n  CUDA: {scene_gpu['points'].device}")

    print(f"\n{'=' * 70}")
    print("SceneBuilder ready for 5D training pipeline")
    print(f"{'=' * 70}")