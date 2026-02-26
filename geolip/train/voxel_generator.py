"""
Voxel Shape Generator
======================

38 geometric primitives on 5×5×5 binary occupancy grids.
Multi-shape superposition on 8×16×16 grids for PatchMaker-style training.

Shapes span the full geometric spectrum:
    0D: point
    1D rigid: line_x, line_y, line_z, line_diag, cross, l_shape, collinear
    1D curved: arc, helix
    2D rigid: triangle_xy, triangle_xz, triangle_3d, square_xy, square_xz,
              rectangle, coplanar, plane
    2D curved: circle, ellipse, disc
    3D rigid: tetrahedron, pyramid, pentachoron, cube, cuboid,
              triangular_prism, octahedron
    3D curved solid: sphere, hemisphere, cylinder, cone, capsule, torus
    3D curved hollow: shell, tube
    3D curved open: bowl, saddle

8 curvature types: none, convex, concave, cylindrical, conical,
                   toroidal, hyperbolic, helical

Author: AbstractPhil + Claude
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Shape Catalog
# =============================================================================

SHAPE_CATALOG = {
    "point":            {"dim": 0, "curved": False, "curvature": "none"},
    "line_x":           {"dim": 1, "curved": False, "curvature": "none"},
    "line_y":           {"dim": 1, "curved": False, "curvature": "none"},
    "line_z":           {"dim": 1, "curved": False, "curvature": "none"},
    "line_diag":        {"dim": 1, "curved": False, "curvature": "none"},
    "cross":            {"dim": 1, "curved": False, "curvature": "none"},
    "l_shape":          {"dim": 1, "curved": False, "curvature": "none"},
    "collinear":        {"dim": 1, "curved": False, "curvature": "none"},
    "triangle_xy":      {"dim": 2, "curved": False, "curvature": "none"},
    "triangle_xz":      {"dim": 2, "curved": False, "curvature": "none"},
    "triangle_3d":      {"dim": 2, "curved": False, "curvature": "none"},
    "square_xy":        {"dim": 2, "curved": False, "curvature": "none"},
    "square_xz":        {"dim": 2, "curved": False, "curvature": "none"},
    "rectangle":        {"dim": 2, "curved": False, "curvature": "none"},
    "coplanar":         {"dim": 2, "curved": False, "curvature": "none"},
    "plane":            {"dim": 2, "curved": False, "curvature": "none"},
    "tetrahedron":      {"dim": 3, "curved": False, "curvature": "none"},
    "pyramid":          {"dim": 3, "curved": False, "curvature": "none"},
    "pentachoron":      {"dim": 3, "curved": False, "curvature": "none"},
    "cube":             {"dim": 3, "curved": False, "curvature": "none"},
    "cuboid":           {"dim": 3, "curved": False, "curvature": "none"},
    "triangular_prism": {"dim": 3, "curved": False, "curvature": "none"},
    "octahedron":       {"dim": 3, "curved": False, "curvature": "none"},
    "arc":              {"dim": 1, "curved": True, "curvature": "convex"},
    "helix":            {"dim": 1, "curved": True, "curvature": "helical"},
    "circle":           {"dim": 2, "curved": True, "curvature": "convex"},
    "ellipse":          {"dim": 2, "curved": True, "curvature": "convex"},
    "disc":             {"dim": 2, "curved": True, "curvature": "convex"},
    "sphere":           {"dim": 3, "curved": True, "curvature": "convex"},
    "hemisphere":       {"dim": 3, "curved": True, "curvature": "convex"},
    "cylinder":         {"dim": 3, "curved": True, "curvature": "cylindrical"},
    "cone":             {"dim": 3, "curved": True, "curvature": "conical"},
    "capsule":          {"dim": 3, "curved": True, "curvature": "convex"},
    "torus":            {"dim": 3, "curved": True, "curvature": "toroidal"},
    "shell":            {"dim": 3, "curved": True, "curvature": "convex"},
    "tube":             {"dim": 3, "curved": True, "curvature": "cylindrical"},
    "bowl":             {"dim": 3, "curved": True, "curvature": "concave"},
    "saddle":           {"dim": 3, "curved": True, "curvature": "hyperbolic"},
}

CLASS_NAMES = list(SHAPE_CATALOG.keys())
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

CURVATURE_TYPES = [
    "none", "convex", "concave", "cylindrical",
    "conical", "toroidal", "hyperbolic", "helical",
]
CURV_TO_IDX = {c: i for i, c in enumerate(CURVATURE_TYPES)}


# =============================================================================
# Single-Shape Voxel Generator (5×5×5)
# =============================================================================

class VoxelShapeGenerator:
    """Generates single shapes on a 5×5×5 binary grid."""

    GS = 5

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(self, name: str) -> np.ndarray:
        """Returns (5, 5, 5) binary occupancy."""
        gs = self.GS
        grid = np.zeros((gs, gs, gs), dtype=np.float32)
        fn = getattr(self, f"_make_{name}", None)
        if fn is None:
            fn = self._make_generic
        fn(grid, name)
        return grid

    def generate_all(self, n_per_class: int = 10) -> List[Tuple[np.ndarray, str]]:
        samples = []
        for name in CLASS_NAMES:
            for _ in range(n_per_class):
                grid = self.generate(name)
                samples.append((grid, name))
        return samples

    # --- 0D ---
    def _make_point(self, g, name):
        c = self.rng.randint(1, 4, size=3)
        g[c[0], c[1], c[2]] = 1.0

    # --- 1D rigid ---
    def _make_line_x(self, g, name):
        y, z = self.rng.randint(0, 5, size=2)
        g[:, y, z] = 1.0

    def _make_line_y(self, g, name):
        x, z = self.rng.randint(0, 5, size=2)
        g[x, :, z] = 1.0

    def _make_line_z(self, g, name):
        x, y = self.rng.randint(0, 5, size=2)
        g[x, y, :] = 1.0

    def _make_line_diag(self, g, name):
        for i in range(5):
            g[i, i, self.rng.randint(0, 5)] = 1.0

    def _make_cross(self, g, name):
        z = self.rng.randint(0, 5)
        g[2, :, z] = 1.0
        g[:, 2, z] = 1.0

    def _make_l_shape(self, g, name):
        z = self.rng.randint(0, 5)
        g[:, 0, z] = 1.0
        g[4, :, z] = 1.0

    def _make_collinear(self, g, name):
        y, z = self.rng.randint(0, 5, size=2)
        g[0, y, z] = 1.0
        g[2, y, z] = 1.0
        g[4, y, z] = 1.0

    # --- 1D curved ---
    def _make_arc(self, g, name):
        z = self.rng.randint(0, 5)
        for t in np.linspace(0, np.pi, 20):
            x = int(np.clip(2 + 1.8 * np.cos(t), 0, 4))
            y = int(np.clip(2 + 1.8 * np.sin(t), 0, 4))
            g[x, y, z] = 1.0

    def _make_helix(self, g, name):
        for t in np.linspace(0, 2 * np.pi, 30):
            x = int(np.clip(2 + 1.5 * np.cos(t), 0, 4))
            y = int(np.clip(2 + 1.5 * np.sin(t), 0, 4))
            z = int(np.clip(t / (2 * np.pi) * 4, 0, 4))
            g[x, y, z] = 1.0

    # --- 2D rigid ---
    def _make_triangle_xy(self, g, name):
        z = self.rng.randint(0, 5)
        g[0, 0, z] = g[4, 0, z] = g[2, 4, z] = 1.0
        for t in np.linspace(0, 1, 10):
            g[int(t * 4), 0, z] = 1.0
            g[int(t * 2), int(t * 4), z] = 1.0
            g[int(4 - t * 2), int(t * 4), z] = 1.0

    def _make_triangle_xz(self, g, name):
        y = self.rng.randint(0, 5)
        g[0, y, 0] = g[4, y, 0] = g[2, y, 4] = 1.0
        for t in np.linspace(0, 1, 10):
            g[int(t * 4), y, 0] = 1.0
            g[int(t * 2), y, int(t * 4)] = 1.0

    def _make_triangle_3d(self, g, name):
        g[0, 0, 0] = g[4, 4, 0] = g[2, 2, 4] = 1.0
        for t in np.linspace(0, 1, 10):
            x, y, z = int(t * 4), int(t * 4), 0
            g[np.clip(x, 0, 4), np.clip(y, 0, 4), z] = 1.0

    def _make_square_xy(self, g, name):
        z = self.rng.randint(0, 5)
        g[0, 0, z] = g[0, 4, z] = g[4, 0, z] = g[4, 4, z] = 1.0
        g[0, :, z] = g[4, :, z] = g[:, 0, z] = g[:, 4, z] = 1.0

    def _make_square_xz(self, g, name):
        y = self.rng.randint(0, 5)
        g[0, y, :] = g[4, y, :] = 1.0
        g[:, y, 0] = g[:, y, 4] = 1.0

    def _make_rectangle(self, g, name):
        z = self.rng.randint(0, 5)
        g[0:2, :, z] = 1.0
        g[:, 0, z] = g[:, 4, z] = 1.0

    def _make_coplanar(self, g, name):
        z = self.rng.randint(0, 5)
        for _ in range(6):
            x, y = self.rng.randint(0, 5, size=2)
            g[x, y, z] = 1.0

    def _make_plane(self, g, name):
        z = self.rng.randint(0, 5)
        g[:, :, z] = 1.0

    # --- 2D curved ---
    def _make_circle(self, g, name):
        z = self.rng.randint(0, 5)
        for t in np.linspace(0, 2 * np.pi, 30):
            x = int(np.clip(2 + 1.8 * np.cos(t), 0, 4))
            y = int(np.clip(2 + 1.8 * np.sin(t), 0, 4))
            g[x, y, z] = 1.0

    def _make_ellipse(self, g, name):
        z = self.rng.randint(0, 5)
        for t in np.linspace(0, 2 * np.pi, 30):
            x = int(np.clip(2 + 1.8 * np.cos(t), 0, 4))
            y = int(np.clip(2 + 1.2 * np.sin(t), 0, 4))
            g[x, y, z] = 1.0

    def _make_disc(self, g, name):
        z = self.rng.randint(0, 5)
        for x in range(5):
            for y in range(5):
                if (x - 2) ** 2 + (y - 2) ** 2 <= 4:
                    g[x, y, z] = 1.0

    # --- 3D rigid ---
    def _make_tetrahedron(self, g, name):
        verts = [(0, 0, 0), (4, 0, 0), (2, 4, 0), (2, 2, 4)]
        for v in verts:
            g[v] = 1.0
        for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            for t in np.linspace(0, 1, 8):
                p = tuple(int(np.clip(verts[a][i] + t * (verts[b][i] - verts[a][i]), 0, 4)) for i in range(3))
                g[p] = 1.0

    def _make_pyramid(self, g, name):
        g[:, :, 0] = 1.0  # base
        g[2, 2, 4] = 1.0  # apex
        for x in [0, 4]:
            for y in [0, 4]:
                for t in np.linspace(0, 1, 8):
                    p = tuple(int(np.clip(x + t * (2 - x), 0, 4)) for x_ in [x, y])
                    g[int(np.clip(x + t * (2 - x), 0, 4)),
                      int(np.clip(y + t * (2 - y), 0, 4)),
                      int(t * 4)] = 1.0

    def _make_pentachoron(self, g, name):
        # 5 vertices of a pentachoron projected to 3D
        verts = [(0, 0, 0), (4, 0, 0), (2, 4, 0), (2, 2, 4), (1, 1, 2)]
        for v in verts:
            g[v] = 1.0

    def _make_cube(self, g, name):
        g[0, :, :] = g[4, :, :] = 1.0
        g[:, 0, :] = g[:, 4, :] = 1.0
        g[:, :, 0] = g[:, :, 4] = 1.0

    def _make_cuboid(self, g, name):
        g[0, :, :] = g[3, :, :] = 1.0
        g[:, 0, :] = g[:, 4, :] = 1.0
        g[:, :, 0] = g[:, :, 2] = 1.0

    def _make_triangular_prism(self, g, name):
        for z in range(5):
            g[0, 0, z] = g[4, 0, z] = g[2, 4, z] = 1.0
            g[:, 0, z] = 1.0

    def _make_octahedron(self, g, name):
        g[2, 2, 0] = g[2, 2, 4] = 1.0
        g[0, 2, 2] = g[4, 2, 2] = 1.0
        g[2, 0, 2] = g[2, 4, 2] = 1.0

    # --- 3D curved ---
    def _make_sphere(self, g, name):
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if (x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2 <= 5:
                        g[x, y, z] = 1.0

    def _make_hemisphere(self, g, name):
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if (x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2 <= 5 and z >= 2:
                        g[x, y, z] = 1.0

    def _make_cylinder(self, g, name):
        for x in range(5):
            for y in range(5):
                if (x - 2) ** 2 + (y - 2) ** 2 <= 4:
                    g[x, y, :] = 1.0

    def _make_cone(self, g, name):
        for z in range(5):
            r = 2.0 * (1 - z / 4.0)
            for x in range(5):
                for y in range(5):
                    if (x - 2) ** 2 + (y - 2) ** 2 <= r ** 2:
                        g[x, y, z] = 1.0

    def _make_capsule(self, g, name):
        self._make_cylinder(g, name)
        self._make_sphere(g, name)

    def _make_torus(self, g, name):
        R, r = 1.8, 0.8
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    dx, dy, dz = x - 2, y - 2, z - 2
                    q = math.sqrt(dx ** 2 + dy ** 2) - R
                    if q ** 2 + dz ** 2 <= r ** 2:
                        g[x, y, z] = 1.0

    def _make_shell(self, g, name):
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    d = (x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2
                    if 3 <= d <= 5:
                        g[x, y, z] = 1.0

    def _make_tube(self, g, name):
        for x in range(5):
            for y in range(5):
                d = (x - 2) ** 2 + (y - 2) ** 2
                if 2 <= d <= 4:
                    g[x, y, :] = 1.0

    def _make_bowl(self, g, name):
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    d = (x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2
                    if 3 <= d <= 5 and z <= 2:
                        g[x, y, z] = 1.0

    def _make_saddle(self, g, name):
        for x in range(5):
            for y in range(5):
                z_val = (x - 2) ** 2 - (y - 2) ** 2
                z = int(np.clip(2 + z_val * 0.5, 0, 4))
                g[x, y, z] = 1.0

    def _make_generic(self, g, name):
        # Fallback: random sparse fill
        n = self.rng.randint(3, 12)
        for _ in range(n):
            p = tuple(self.rng.randint(0, 5, size=3))
            g[p] = 1.0


# =============================================================================
# Multi-Shape Superposition (8×16×16)
# =============================================================================

class MultiShapeVoxelGenerator:
    """
    Generates multi-shape superpositions on 8×16×16 grids.
    Matches PatchMaker training format: (8, 16, 16) occupancy grids
    with 2-4 shapes randomly placed.
    """

    GZ, GY, GX = 8, 16, 16

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.single_gen = VoxelShapeGenerator(seed=seed)

    def generate_sample(
        self, n_shapes: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Returns:
            grid: (8, 16, 16) float32 occupancy
            labels: list of shape names
            membership: (num_shapes, 8, 16, 16) per-shape masks
        """
        if n_shapes is None:
            n_shapes = self.rng.randint(2, 5)

        grid = np.zeros((self.GZ, self.GY, self.GX), dtype=np.float32)
        choices = self.rng.choice(NUM_CLASSES, n_shapes, replace=False)
        labels = []
        masks = []

        for idx in choices:
            name = CLASS_NAMES[idx]
            labels.append(name)

            # Generate 5×5×5, place at random offset in 8×16×16
            small = self.single_gen.generate(name)
            mask = np.zeros_like(grid)

            oz = self.rng.randint(0, self.GZ - 4)
            oy = self.rng.randint(0, self.GY - 4)
            ox = self.rng.randint(0, self.GX - 4)
            mask[oz:oz + 5, oy:oy + 5, ox:ox + 5] = small
            masks.append(mask)
            grid = np.maximum(grid, mask)

        membership = np.stack(masks, axis=0) if masks else np.zeros((0, *grid.shape))
        return grid, labels, membership


# =============================================================================
# Datasets
# =============================================================================

class VoxelSingleShapeDataset(Dataset):
    """Single shapes on 5×5×5 grid. Returns (grid, class_idx, geometry)."""

    def __init__(self, n_per_class: int = 50, seed: int = 42):
        gen = VoxelShapeGenerator(seed=seed)
        self.samples = gen.generate_all(n_per_class)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        grid, name = self.samples[idx]
        props = SHAPE_CATALOG[name]
        return (
            torch.from_numpy(grid),
            CLASS_TO_IDX[name],
            props["dim"],
            1 if props["curved"] else 0,
            CURV_TO_IDX[props["curvature"]],
        )


class VoxelMultiShapeDataset(Dataset):
    """Multi-shape superpositions on 8×16×16. Returns (grid, label_vec)."""

    def __init__(self, n_samples: int = 10000, seed: int = 42):
        self.data = []
        gen = MultiShapeVoxelGenerator(seed=seed)
        for i in range(n_samples):
            if i % 2000 == 0:
                gen = MultiShapeVoxelGenerator(seed=seed + i)
            grid, labels, _ = gen.generate_sample()
            label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            for name in labels:
                label_vec[CLASS_TO_IDX[name]] = 1.0
            self.data.append((
                torch.from_numpy(grid).unsqueeze(0),  # (1, 8, 16, 16)
                torch.from_numpy(label_vec),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]