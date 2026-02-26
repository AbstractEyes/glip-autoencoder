"""
Geometric Shape Factory
========================

Pure generation utilities for SDF-based geometric shapes.
No torch dependency — returns numpy arrays only.

38-class shape vocabulary rendered as signed distance fields.
Multi-shape composition with configurable spatial resolution
and channel count for simulated VAE latent structure.

Usage:
    factory = GeometricShapeFactory(width=64, height=64, seed=42)

    # Single sample
    latent, labels, masks, geometry = factory(channels=32, n_shapes=3)

    # Iterator
    for latent, labels, masks, geometry in factory.stream(n=1000, channels=32):
        ...

Author: AbstractPhil + Claude
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Iterator


# ---------------------------------------------------------------------------
# Shape catalog (imported from voxel_generator to stay canonical)
# ---------------------------------------------------------------------------

from .voxel_generator import (
    SHAPE_CATALOG, CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX,
    CURVATURE_TYPES, CURV_TO_IDX,
)


# ---------------------------------------------------------------------------
# SDF Primitives
# ---------------------------------------------------------------------------

class SDFRenderer:
    """Renders shapes as continuous signed distance fields on H×W grids."""

    def __init__(self, width: int = 64, height: int = 64):
        y, x = np.mgrid[:height, :width].astype(np.float32)
        self._yy = y
        self._xx = x
        self.width = width
        self.height = height

    def sphere(self, cx, cy, r):
        d = np.sqrt((self._xx - cx) ** 2 + (self._yy - cy) ** 2) - r
        return np.clip(1.0 - d / 2.0, 0, 1)

    def cube(self, cx, cy, half):
        dx = np.abs(self._xx - cx) - half
        dy = np.abs(self._yy - cy) - half
        d = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
        return np.clip(1.0 - d / 2.0, 0, 1)

    def cylinder(self, cx, cy, r, half_h):
        dr = np.abs(np.sqrt((self._xx - cx) ** 2) - r)
        dh = np.abs(self._yy - cy) - half_h
        d = np.sqrt(np.maximum(dr - r, 0) ** 2 + np.maximum(dh, 0) ** 2)
        return np.clip(1.0 - d / 2.0, 0, 1)

    def torus(self, cx, cy, R, r):
        q = np.sqrt((self._xx - cx) ** 2) - R
        d = np.sqrt(q ** 2 + (self._yy - cy) ** 2) - r
        return np.clip(1.0 - d / 2.0, 0, 1)

    def line(self, x0, y0, x1, y1, thickness=1.5):
        dx, dy = float(x1 - x0), float(y1 - y0)
        length = max(math.sqrt(dx ** 2 + dy ** 2), 1e-6)
        t = np.clip(
            ((self._xx - x0) * dx + (self._yy - y0) * dy) / (length ** 2), 0, 1
        )
        px, py = x0 + t * dx, y0 + t * dy
        d = np.sqrt((self._xx - px) ** 2 + (self._yy - py) ** 2) - thickness
        return np.clip(1.0 - d / 2.0, 0, 1)

    def disc(self, cx, cy, r):
        return self.sphere(cx, cy, r)

    def ring(self, cx, cy, R, r):
        return self.torus(cx, cy, R, r)

    def cross(self, cx, cy, arm_len, thickness=1.5):
        h = self.line(cx - arm_len, cy, cx + arm_len, cy, thickness)
        v = self.line(cx, cy - arm_len, cx, cy + arm_len, thickness)
        return np.maximum(h, v)

    def triangle(self, cx, cy, size):
        p0 = (cx, cy - size * 0.577)
        p1 = (cx - size * 0.5, cy + size * 0.289)
        p2 = (cx + size * 0.5, cy + size * 0.289)

        def edge_d(ax, ay, bx, by):
            dx, dy = bx - ax, by - ay
            n = math.sqrt(dx ** 2 + dy ** 2) + 1e-8
            return ((self._xx - ax) * dy - (self._yy - ay) * dx) / n

        d0 = edge_d(*p0, *p1)
        d1 = edge_d(*p1, *p2)
        d2 = edge_d(*p2, *p0)
        inside = np.minimum(np.minimum(d0, d1), d2)
        return np.clip(inside / 2.0, 0, 1)

    def plane(self, y_pos, thickness=2.0):
        d = np.abs(self._yy - y_pos) - thickness
        return np.clip(1.0 - d / 2.0, 0, 1)

    def helix(self, cx, cy, r, pitch, thickness=1.0):
        t = self._yy / max(self.height, 1) * 4.0 * np.pi
        hx = cx + r * np.cos(t + pitch)
        d = np.abs(np.sqrt((self._xx - hx) ** 2)) - thickness
        return np.clip(1.0 - np.abs(d) / 2.0, 0, 1)

    def ray(self, x0, y0, angle, length, thickness=1.0):
        x1 = x0 + length * math.cos(angle)
        y1 = y0 + length * math.sin(angle)
        return self.line(x0, y0, x1, y1, thickness)

    def cone(self, cx, cy, r_base, height):
        t = np.clip((self._yy - (cy - height / 2)) / max(height, 1e-6), 0, 1)
        r = r_base * (1.0 - t)
        d = np.sqrt((self._xx - cx) ** 2) - r
        d_h = np.maximum(-(self._yy - (cy - height / 2)), self._yy - (cy + height / 2))
        d_total = np.maximum(d, d_h)
        return np.clip(1.0 - d_total / 2.0, 0, 1)

    def capsule(self, cx, cy, r, half_h):
        top = self.sphere(cx, cy - half_h, r)
        bot = self.sphere(cx, cy + half_h, r)
        mid = self.cube(cx, cy, max(r, half_h))
        return np.maximum(np.maximum(top, bot), mid * 0.5)

    def shell(self, cx, cy, R, thickness=2.0):
        outer = self.sphere(cx, cy, R)
        inner = self.sphere(cx, cy, R - thickness)
        return np.clip(outer - inner, 0, 1)

    def hemisphere(self, cx, cy, r):
        full = self.sphere(cx, cy, r)
        mask = np.where(self._yy <= cy, 1.0, 0.0)
        return full * mask

    def bowl(self, cx, cy, R, thickness=2.0):
        s = self.shell(cx, cy, R, thickness)
        mask = np.where(self._yy >= cy, 1.0, 0.0)
        return s * mask

    def saddle(self, cx, cy, scale=8.0):
        dx = (self._xx - cx) / scale
        dy = (self._yy - cy) / scale
        z = dx ** 2 - dy ** 2
        return np.clip(0.5 + 0.5 * np.exp(-z ** 2), 0, 1) * np.clip(
            1.0 - (dx ** 2 + dy ** 2) / 4.0, 0, 1
        )

    def ellipse(self, cx, cy, rx, ry, thickness=1.5):
        d = np.sqrt(((self._xx - cx) / rx) ** 2 + ((self._yy - cy) / ry) ** 2) - 1.0
        return np.clip(1.0 - np.abs(d) * min(rx, ry) / 2.0, 0, 1)

    def rectangle(self, cx, cy, hx, hy):
        return self.cube(cx, cy, max(hx, hy))

    def point(self, cx, cy, size=2.0):
        return self.sphere(cx, cy, size)

    def octahedron(self, cx, cy, size):
        dx = np.abs(self._xx - cx) / size
        dy = np.abs(self._yy - cy) / size
        d = dx + dy - 1.0
        return np.clip(1.0 - d * size / 2.0, 0, 1)

    def tube(self, cx, cy, R, r):
        return self.ring(cx, cy, R, r)


# ---------------------------------------------------------------------------
# Shape → SDF dispatch
# ---------------------------------------------------------------------------

_SHAPE_RENDER_MAP = {
    "point":            ("point",      {"size": 2.0}),
    "line_x":           ("line",       {"axis": "h"}),
    "line_y":           ("line",       {"axis": "v"}),
    "line_z":           ("line",       {"axis": "d"}),
    "line_diag":        ("line",       {"axis": "diag"}),
    "cross":            ("cross",      {"arm_len": 12}),
    "l_shape":          ("cross",      {"arm_len": 10}),
    "collinear":        ("line",       {"axis": "h"}),
    "triangle_xy":      ("triangle",   {"size": 12}),
    "triangle_xz":      ("triangle",   {"size": 10}),
    "triangle_3d":      ("triangle",   {"size": 11}),
    "square_xy":        ("cube",       {"half": 10}),
    "square_xz":        ("cube",       {"half": 8}),
    "rectangle":        ("rectangle",  {"hx": 12, "hy": 6}),
    "coplanar":         ("plane",      {"thickness": 1.5}),
    "plane":            ("plane",      {"thickness": 3.0}),
    "tetrahedron":      ("triangle",   {"size": 14}),
    "pyramid":          ("triangle",   {"size": 16}),
    "pentachoron":      ("triangle",   {"size": 12}),
    "cube":             ("cube",       {"half": 10}),
    "cuboid":           ("cube",       {"half": 8}),
    "triangular_prism": ("triangle",   {"size": 10}),
    "octahedron":       ("octahedron", {"size": 10}),
    "arc":              ("ray",        {"length": 20}),
    "helix":            ("helix",      {"r": 10, "pitch": 0.5}),
    "circle":           ("ring",       {"R": 12, "r": 1.5}),
    "ellipse":          ("ellipse",    {"rx": 14, "ry": 8}),
    "disc":             ("disc",       {"r": 12}),
    "sphere":           ("sphere",     {"r": 12}),
    "hemisphere":       ("hemisphere", {"r": 12}),
    "cylinder":         ("cylinder",   {"r": 8, "half_h": 12}),
    "cone":             ("cone",       {"r_base": 10, "height": 18}),
    "capsule":          ("capsule",    {"r": 6, "half_h": 10}),
    "torus":            ("torus",      {"R": 12, "r": 4}),
    "shell":            ("shell",      {"R": 14, "thickness": 3}),
    "tube":             ("tube",       {"R": 10, "r": 3}),
    "bowl":             ("bowl",       {"R": 14, "thickness": 3}),
    "saddle":           ("saddle",     {"scale": 8}),
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class GeometricShapeFactory:
    """
    Pure numpy factory for geometric shape generation.

    Produces (C, H, W) simulated VAE latents from SDF compositions.
    No torch dependency. Stateful RNG for reproducibility.

    Usage:
        factory = GeometricShapeFactory(width=64, height=64, seed=42)
        latent, labels, masks, geometry = factory(channels=32, n_shapes=3)
    """

    def __init__(self, width: int = 64, height: int = 64, seed: int = 42):
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.sdf = SDFRenderer(width=width, height=height)

    def reset(self, seed: int = None):
        """Reset RNG state. Uses original seed if none given."""
        self.rng = np.random.RandomState(seed if seed is not None else self.seed)

    def render_shape(
        self, name: str, cx: float, cy: float, scale: float, angle: float
    ) -> np.ndarray:
        """Render a single named shape at given position/scale/angle."""
        method_name, defaults = _SHAPE_RENDER_MAP.get(name, ("sphere", {"r": 10}))
        method = getattr(self.sdf, method_name)

        if method_name == "line":
            axis = defaults.get("axis", "h")
            length = 20 * scale
            if axis == "h":
                return self.sdf.line(cx - length, cy, cx + length, cy)
            elif axis == "v":
                return self.sdf.line(cx, cy - length, cx, cy + length)
            elif axis == "diag":
                return self.sdf.line(cx - length, cy - length, cx + length, cy + length)
            else:
                x1 = cx + length * math.cos(angle)
                y1 = cy + length * math.sin(angle)
                return self.sdf.line(cx, cy, x1, y1)

        elif method_name == "ray":
            return self.sdf.ray(cx, cy, angle, defaults["length"] * scale)
        elif method_name in ("sphere", "disc", "point"):
            r = defaults.get("r", defaults.get("size", 10)) * scale
            return method(cx, cy, r)
        elif method_name == "cube":
            return method(cx, cy, defaults["half"] * scale)
        elif method_name == "cross":
            return method(cx, cy, defaults["arm_len"] * scale)
        elif method_name == "triangle":
            return method(cx, cy, defaults["size"] * scale)
        elif method_name == "torus":
            return method(cx, cy, defaults["R"] * scale, defaults["r"] * scale)
        elif method_name == "ring":
            return method(cx, cy, defaults["R"] * scale, defaults.get("r", 2) * scale)
        elif method_name == "cylinder":
            return method(cx, cy, defaults["r"] * scale, defaults["half_h"] * scale)
        elif method_name == "cone":
            return method(cx, cy, defaults["r_base"] * scale, defaults["height"] * scale)
        elif method_name == "capsule":
            return method(cx, cy, defaults["r"] * scale, defaults["half_h"] * scale)
        elif method_name == "shell":
            return method(cx, cy, defaults["R"] * scale, defaults.get("thickness", 3) * scale)
        elif method_name == "hemisphere":
            return method(cx, cy, defaults["r"] * scale)
        elif method_name == "bowl":
            return method(cx, cy, defaults["R"] * scale, defaults.get("thickness", 3) * scale)
        elif method_name == "saddle":
            return method(cx, cy, defaults["scale"] * scale)
        elif method_name == "helix":
            return method(cx, cy, defaults["r"] * scale, defaults.get("pitch", 0.5))
        elif method_name == "ellipse":
            return method(cx, cy, defaults["rx"] * scale, defaults["ry"] * scale)
        elif method_name == "plane":
            return method(cy, defaults["thickness"] * scale)
        elif method_name == "rectangle":
            return method(cx, cy, defaults["hx"] * scale, defaults["hy"] * scale)
        elif method_name == "octahedron":
            return method(cx, cy, defaults["size"] * scale)
        elif method_name == "tube":
            return method(cx, cy, defaults["R"] * scale, defaults["r"] * scale)

        return self.sdf.sphere(cx, cy, 10 * scale)

    def composite_to_latent(self, composite: np.ndarray, channels: int = 32) -> np.ndarray:
        """
        (H, W) composite density → (C, H, W) simulated VAE latent.

        Each channel gets deterministic frequency modulation simulating
        how a real VAE distributes information across channels.
        """
        H, W = composite.shape
        latent = np.zeros((channels, H, W), dtype=np.float32)
        for c in range(channels):
            freq = 1.0 + c * 0.3
            phase = c * np.pi / 16
            modulation = 0.5 + 0.5 * np.sin(freq * composite * np.pi + phase)
            latent[c] = composite * modulation
            latent[c] += self.rng.randn(H, W).astype(np.float32) * 0.05 * (1 + 0.1 * c)
        return latent

    def __call__(
        self,
        channels: int = 32,
        n_shapes: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray], Dict[str, Tuple]]:
        """
        Generate one sample.

        Returns:
            latent:   (C, H, W) float32 — simulated VAE latent
            labels:   list of shape names present
            masks:    {name: (H, W)} per-shape SDF density
            geometry: {name: (dim, curved, curvature_str)}
        """
        H, W = self.height, self.width
        if n_shapes is None:
            n_shapes = self.rng.randint(2, 6)

        choices = self.rng.choice(NUM_CLASSES, n_shapes, replace=False)
        composite = np.zeros((H, W), dtype=np.float32)
        labels = []
        masks = {}

        margin = max(4, min(H, W) // 5)
        hi_x = max(margin + 1, W - margin)
        hi_y = max(margin + 1, H - margin)

        for idx in choices:
            name = CLASS_NAMES[idx]
            labels.append(name)
            cx = self.rng.randint(margin, hi_x)
            cy = self.rng.randint(margin, hi_y)
            scale = self.rng.uniform(0.6, 1.4)
            angle = self.rng.uniform(0, 2 * np.pi)
            mask = self.render_shape(name, float(cx), float(cy), scale, angle)
            masks[name] = mask
            composite = np.maximum(composite, mask)

        latent = self.composite_to_latent(composite, channels=channels)
        geometry = {
            name: (
                SHAPE_CATALOG[name]["dim"],
                SHAPE_CATALOG[name]["curved"],
                SHAPE_CATALOG[name]["curvature"],
            )
            for name in labels
        }
        return latent, labels, masks, geometry

    def stream(
        self,
        n: int,
        channels: int = 32,
        n_shapes: Optional[int] = None,
    ) -> Iterator[Tuple[np.ndarray, List[str], Dict[str, np.ndarray], Dict[str, Tuple]]]:
        """Yield n samples sequentially. Respects RNG state continuity."""
        for _ in range(n):
            yield self(channels=channels, n_shapes=n_shapes)

    def batch(
        self,
        n: int,
        channels: int = 32,
        n_shapes: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Generate n samples, stack latents into a single array.

        Returns:
            latents: (N, C, H, W) float32
            all_labels: list of N label lists
        """
        latents = []
        all_labels = []
        for latent, labels, _, _ in self.stream(n, channels, n_shapes):
            latents.append(latent)
            all_labels.append(labels)
        return np.stack(latents, axis=0), all_labels