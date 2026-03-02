"""
Topology — Builder + Frozen Container (numpy/torch backend)
=============================================================

TopoBuilder: configures and constructs topology.
Topology:    frozen precomputed result. Arrays only. No logic.

Backend modes:
    'numpy': pure numpy, no torch dependency.
    'torch': torch tensors with device + dtype control.

Backend is a tensor operations layer — creation, math, shape, linalg,
random, cast. No activations, no neural net ops.

Author: AbstractPhil + Claude
License: Apache-2.0
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from abc import ABC, abstractmethod

ArrayLike = Any  # np.ndarray or torch.Tensor


# ══════════════════════════════════════════════════════════════════════════════
# Backend — pure tensor ops, no activations
# ══════════════════════════════════════════════════════════════════════════════

class Backend:
    """
    Tensor operations backend. numpy or torch. Never both.

    This is a creation/math/shape/linalg/random/cast layer.
    No activations (relu, sigmoid, etc). No neural net ops.
    """

    def __init__(self, mode: str = "numpy", device: str = "cpu", dtype: str = "float32"):
        assert mode in ("numpy", "torch"), f"mode must be 'numpy' or 'torch', got '{mode}'"
        self.mode = mode
        self.device = device
        self.dtype_str = dtype

        if mode == "torch":
            import torch as _torch
            self._t = _torch
            self._dtype = getattr(_torch, dtype)
            self._int_dtype = _torch.int32
            self._device = _torch.device(device)
        else:
            self._t = None
            self._dtype = getattr(np, dtype)
            self._int_dtype = np.int32
            self._device = None

    # ── Creation ──

    def zeros(self, *shape):
        if self._t:
            return self._t.zeros(*shape, dtype=self._dtype, device=self._device)
        return np.zeros(shape, dtype=self._dtype)

    def ones(self, *shape):
        if self._t:
            return self._t.ones(*shape, dtype=self._dtype, device=self._device)
        return np.ones(shape, dtype=self._dtype)

    def arange(self, n):
        if self._t:
            return self._t.arange(n, device=self._device)
        return np.arange(n)

    def arange_int(self, n):
        if self._t:
            return self._t.arange(n, dtype=self._int_dtype, device=self._device)
        return np.arange(n, dtype=self._int_dtype)

    def linspace(self, start, end, n):
        if self._t:
            return self._t.linspace(start, end, n, dtype=self._dtype, device=self._device)
        return np.linspace(start, end, n, dtype=self._dtype)

    def full(self, shape, value):
        if self._t:
            return self._t.full(shape, value, dtype=self._dtype, device=self._device)
        return np.full(shape, value, dtype=self._dtype)

    def tensor(self, data):
        if self._t:
            return self._t.tensor(data, dtype=self._dtype, device=self._device)
        return np.asarray(data, dtype=self._dtype)

    # ── Math ──

    def sin(self, x):
        return self._t.sin(x) if self._t else np.sin(x)

    def cos(self, x):
        return self._t.cos(x) if self._t else np.cos(x)

    def exp(self, x):
        return self._t.exp(x) if self._t else np.exp(x)

    def log(self, x):
        return self._t.log(x) if self._t else np.log(x)

    def abs(self, x):
        return self._t.abs(x) if self._t else np.abs(x)

    def sqrt(self, x):
        return self._t.sqrt(x) if self._t else np.sqrt(x)

    def pow(self, x, p):
        if self._t:
            return x.pow(p)
        return np.power(x, p)

    def clamp(self, x, lo=None, hi=None):
        if self._t:
            return self._t.clamp(x, min=lo, max=hi)
        return np.clip(x, lo, hi)

    def mod(self, x, divisor):
        if self._t:
            return x % divisor
        return np.mod(x, divisor)

    # ── Reduction ──

    def sum(self, x, axis=-1):
        if self._t:
            return x.sum(dim=axis)
        return np.sum(x, axis=axis)

    def max(self, x, axis=-1, keepdims=False):
        if self._t:
            return x.max(dim=axis, keepdim=keepdims).values
        return np.max(x, axis=axis, keepdims=keepdims)

    def norm(self, x, axis=-1, keepdims=False):
        if self._t:
            return x.norm(dim=axis, keepdim=keepdims)
        return np.linalg.norm(x, axis=axis, keepdims=keepdims)

    # ── Shape ──

    def stack(self, arrays, axis=0):
        if self._t:
            return self._t.stack(arrays, dim=axis)
        return np.stack(arrays, axis=axis)

    def cat(self, arrays, axis=0):
        if self._t:
            return self._t.cat(arrays, dim=axis)
        return np.concatenate(arrays, axis=axis)

    def reshape(self, x, shape):
        return x.reshape(shape)

    def unsqueeze(self, x, axis):
        if self._t:
            return x.unsqueeze(axis)
        return np.expand_dims(x, axis=axis)

    def meshgrid(self, *arrays, indexing="ij"):
        if self._t:
            return self._t.meshgrid(*arrays, indexing=indexing)
        return np.meshgrid(*arrays, indexing=indexing)

    def slice_to(self, x, n):
        """First n elements."""
        return x[:n]

    # ── Sort / Select ──

    def topk(self, x, k, axis=-1):
        """Returns (values, indices) for top-k along axis."""
        if self._t:
            return x.topk(k, dim=axis)
        idx = np.argsort(x, axis=axis)[..., -k:]
        idx = np.flip(idx, axis=axis).copy()
        vals = np.take_along_axis(x, idx, axis=axis)
        return vals, idx

    def argsort(self, x, axis=-1, descending=False):
        if self._t:
            return x.argsort(dim=axis, descending=descending)
        idx = np.argsort(x, axis=axis)
        if descending:
            idx = np.flip(idx, axis=axis).copy()
        return idx

    # ── Linear algebra ──

    def mm(self, a, b):
        if self._t:
            return self._t.mm(a, b)
        return np.dot(a, b)

    def transpose(self, x):
        if self._t:
            return x.t()
        return x.T

    def rotation_2d(self, angles: ArrayLike) -> ArrayLike:
        """
        Batch 2D rotation matrices from angles.
        Args: angles (N,) in radians
        Returns: (N, 2, 2) rotation matrices
        """
        c = self.cos(angles)
        s = self.sin(angles)
        if self._t:
            R = self._t.zeros(angles.shape[0], 2, 2, dtype=self._dtype, device=self._device)
        else:
            R = np.zeros((angles.shape[0], 2, 2), dtype=self._dtype)
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        return R

    def givens(self, angles: ArrayLike, dim: int, plane: tuple) -> ArrayLike:
        """
        Batch Givens rotation matrices.
        Args:
            angles: (N,) in radians
            dim:    matrix dimension
            plane:  (i, j) indices of the rotation plane
        Returns: (N, dim, dim) rotation matrices
        """
        N = angles.shape[0]
        i, j = plane
        c = self.cos(angles)
        s = self.sin(angles)
        if self._t:
            R = self._t.eye(dim, dtype=self._dtype, device=self._device).unsqueeze(0).expand(N, -1, -1).clone()
        else:
            R = np.tile(np.eye(dim, dtype=self._dtype), (N, 1, 1))
        R[:, i, i] = c
        R[:, i, j] = -s
        R[:, j, i] = s
        R[:, j, j] = c
        return R

    # ── Random (seeded) ──

    def make_rng(self, seed: int):
        if self._t:
            g = self._t.Generator(device=self._device)
            g.manual_seed(seed)
            return g
        return np.random.RandomState(seed)

    def rand(self, shape, rng):
        if self._t:
            return self._t.rand(*shape, generator=rng, dtype=self._dtype, device=self._device)
        return rng.rand(*shape).astype(self._dtype)

    # ── Cast ──

    def to_int(self, x):
        if self._t:
            return x.to(self._int_dtype)
        return x.astype(self._int_dtype)

    def to_float(self, x):
        if self._t:
            return x.to(self._dtype)
        return x.astype(self._dtype)

    def item(self, x):
        if self._t:
            return x.item()
        return float(x)

    def tolist(self, x):
        if self._t:
            return x.tolist()
        return x.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Projection Protocol
# ══════════════════════════════════════════════════════════════════════════════

class TopologyProjection(ABC):
    """Mask generator. Maps N-dimensional positions to topology masks."""

    @abstractmethod
    def mask_dim(self) -> int:
        ...

    @abstractmethod
    def __call__(
        self, positions: ArrayLike, B: Backend
    ) -> ArrayLike:
        """
        Args:
            positions: (N, n_axes) coordinates in [0,1]
            B:         backend for all tensor ops
        Returns:
            masks: (N, mask_dim)
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# MobiusProjection — fully configurable
# ══════════════════════════════════════════════════════════════════════════════

class MobiusProjection(TopologyProjection):
    """
    Dual-step tri-wave interference topology.

    All wave parameters configurable:
        n_waves:     number of waves per basis
        frequencies: per-wave frequency multipliers
        phases:      per-wave phase offsets
        omega:       base angular frequency
        alpha_base:  gate sharpness

    mask_dim = 2 * n_waves + 3
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.25, 2.75),
        n_waves: int = 3,
        frequencies: Optional[List[float]] = None,
        phases: Optional[List[float]] = None,
        omega: float = math.pi,
        alpha_base: float = 1.5,
    ):
        self.scale_range = scale_range
        self.n_waves = n_waves
        self.omega = omega
        self.alpha_base = alpha_base

        self.frequencies = list(frequencies) if frequencies is not None else [
            1.0 + i * 0.5 for i in range(n_waves)
        ]
        self.phases = list(phases) if phases is not None else [
            i * math.pi / n_waves for i in range(n_waves)
        ]
        assert len(self.frequencies) == n_waves
        assert len(self.phases) == n_waves

    def mask_dim(self) -> int:
        return 2 * self.n_waves + 3

    def _waves(
        self, t: ArrayLike, scale: ArrayLike, phase_offset: ArrayLike, B: Backend
    ) -> List[ArrayLike]:
        """N waves at given scale with phase offset."""
        freqs = B.tensor(self.frequencies)
        base_phases = B.tensor(self.phases)

        f = B.unsqueeze(scale, -1) * B.unsqueeze(freqs, 0)
        p = B.unsqueeze(base_phases, 0) + B.unsqueeze(phase_offset, -1)
        positions = f * self.omega * B.unsqueeze(t, -1) + p
        waves = B.sin(positions)
        alpha = self.alpha_base / B.clamp(B.unsqueeze(scale, -1), lo=0.1)
        gates = B.exp(-alpha * B.pow(waves, 2))
        return [gates[..., i] for i in range(self.n_waves)]

    def _fuse(self, wave_list: List[ArrayLike], B: Backend) -> ArrayLike:
        """Cantor-style XOR/AND fusion."""
        n = len(wave_list)
        if n < 2:
            return wave_list[0]
        L, R = wave_list[0], wave_list[-1]
        xor_comp = B.abs(L + R - 2 * L * R)
        and_comp = L * R
        lr = 0.6 * xor_comp + 0.4 * and_comp
        total = wave_list[0]
        for w in wave_list[1:]:
            total = total + w
        fused = total / n
        fused = fused * (0.5 + 0.5 * lr)
        return fused

    def __call__(
        self, positions: ArrayLike, B: Backend
    ) -> ArrayLike:
        depth_t = positions[..., 0]
        width_t = positions[..., 1] if positions.shape[-1] > 1 else B.zeros(positions.shape[0])
        step_t = positions[..., 2] if positions.shape[-1] > 2 else B.full((positions.shape[0],), 0.5)

        s_lo, s_hi = self.scale_range
        scale = s_lo + depth_t * (s_hi - s_lo)
        phase_offset = width_t * math.pi + step_t * math.pi * 2

        H_waves = self._waves(depth_t, scale, phase_offset, B)
        H_fused = self._fuse(H_waves, B)

        V_waves = self._waves(width_t, scale, phase_offset + math.pi / 2, B)
        V_fused = self._fuse(V_waves, B)

        combined = depth_t * H_fused + (1 - depth_t) * V_fused
        return B.stack(H_waves + V_waves + [H_fused, V_fused, combined], axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# CantorProjection
# ══════════════════════════════════════════════════════════════════════════════

class CantorProjection(TopologyProjection):
    """Devil's Staircase branch path projection."""

    def __init__(self, levels: int = 5, base: int = 3):
        self.levels = levels
        self.base = base

    def mask_dim(self) -> int:
        return self.levels * self.base

    def _softmax(self, x: ArrayLike, B: Backend, axis: int = -1) -> ArrayLike:
        """Numerically stable softmax — math op, lives in projection."""
        m = B.max(x, axis=axis, keepdims=True)
        e = B.exp(x - m)
        s = B.sum(e, axis=axis)
        return e / B.unsqueeze(s, axis)

    def __call__(
        self, positions: ArrayLike, B: Backend
    ) -> ArrayLike:
        depth_t = positions[..., 0]
        step_t = positions[..., 2] if positions.shape[-1] > 2 else B.zeros(positions.shape[0])

        x = B.clamp(depth_t + step_t * 0.5, lo=1e-10, hi=1.0 - 1e-10)
        centers = B.tensor([0.5, 1.5, 2.5])
        scales = B.tensor([self.base ** k for k in range(1, self.levels + 1)])
        tau = 0.25

        y_all = B.mod(B.unsqueeze(x, -1) * B.unsqueeze(scales, 0), self.base)
        d2 = B.pow(B.unsqueeze(y_all, -1) - B.unsqueeze(B.unsqueeze(centers, 0), 0), 2)
        p = self._softmax(-d2 / tau, B, axis=-1)

        N = positions.shape[0]
        return B.reshape(p, (N, -1))


# ══════════════════════════════════════════════════════════════════════════════
# Topology — frozen container
# ══════════════════════════════════════════════════════════════════════════════

class Topology:
    """
    Frozen precomputed topology. Arrays only. No computation.
    Created by TopoBuilder. Holds numpy or torch arrays.
    """

    __slots__ = (
        "num_states", "n_depth", "n_width", "n_step", "divisor_h", "divisor_w",
        "top_k", "mask_dim", "backend_mode",
        "positions_3d", "positions_2d", "positions",
        "depth_positions", "width_positions", "step_positions",
        "alpha_masks", "scale_windows", "deform_scales", "twist_angles",
        "divisor_indices_h", "divisor_indices_w", "branch_paths",
        "staircase_vals", "alignment_matrix",
        "per_projection_alignments", "boundary_bias",
        "rotation_matrices",
        "level_scale_windows", "level_deform_scales", "level_twist_angles",
        "level_alpha_masks", "level_rotation_matrices", "num_levels",
        "level_width_center",
        "recipe",
    )

    def get_state_constants(self, state_idx: int) -> Dict[str, ArrayLike]:
        """All topology constants for one state."""
        return {
            "topo_position_3d": self.positions_3d[state_idx],
            "topo_position_2d": self.positions_2d[state_idx],
            "topo_step": self.step_positions[state_idx],
            "topo_scale_window": self.scale_windows[state_idx],
            "topo_deform_scale": self.deform_scales[state_idx],
            "topo_twist_angle": self.twist_angles[state_idx],
            "topo_alpha_mask": self.alpha_masks[state_idx],
            "topo_divisor_h": self.divisor_indices_h[state_idx],
            "topo_divisor_w": self.divisor_indices_w[state_idx],
            "topo_branch_path": self.branch_paths[state_idx],
            "topo_staircase_val": self.staircase_vals[state_idx],
            "topo_rotation": self.rotation_matrices[state_idx],
        }

    def get_level_constants(self, level_idx: int) -> Dict[str, ArrayLike]:
        if self.level_scale_windows is None:
            raise ValueError("No multi-level constants (num_levels=1)")
        return {
            "level_scale_window": self.level_scale_windows[level_idx],
            "level_deform_scale": self.level_deform_scales[level_idx],
            "level_twist_angle": self.level_twist_angles[level_idx],
            "level_alpha_mask": self.level_alpha_masks[level_idx],
            "level_rotation": self.level_rotation_matrices[level_idx],
        }

    def get_state_identity(self, state_idx: int) -> Dict:
        p3d = self.positions_3d[state_idx]
        p2d = self.positions_2d[state_idx]
        sw = self.scale_windows[state_idx]
        return {
            "index": state_idx,
            "position_3d": p3d.tolist() if hasattr(p3d, 'tolist') else list(p3d),
            "position_2d": p2d.tolist() if hasattr(p2d, 'tolist') else list(p2d),
            "scale_window": sw.tolist() if hasattr(sw, 'tolist') else list(sw),
            "deform_scale": float(self.deform_scales[state_idx]),
            "twist_angle": float(self.twist_angles[state_idx]),
            "divisor_h": int(self.divisor_indices_h[state_idx]),
            "divisor_w": int(self.divisor_indices_w[state_idx]),
            "staircase_val": float(self.staircase_vals[state_idx]),
        }

    def get_alignment(self, i: int, j: int) -> float:
        return float(self.alignment_matrix[i, j])

    def get_neighborhood(self, state_idx: int, threshold: float = 0.5) -> List[int]:
        row = self.alignment_matrix[state_idx]
        return [j for j in range(self.num_states) if float(row[j]) > threshold]

    def to_torch(self, device: str = "cpu", dtype: str = "float32"):
        """Convert all arrays to torch tensors. Returns self."""
        import torch
        dt = getattr(torch, dtype)
        dev = torch.device(device)
        for attr in self.__slots__:
            val = getattr(self, attr, None)
            if val is None or isinstance(val, (int, str, float, dict)):
                continue
            if isinstance(val, np.ndarray):
                t = torch.from_numpy(val)
                if np.issubdtype(val.dtype, np.integer):
                    setattr(self, attr, t.to(dev))
                else:
                    setattr(self, attr, t.to(dtype=dt, device=dev))
        self.backend_mode = "torch"
        return self

    def to_numpy(self):
        """Convert all arrays to numpy. Returns self."""
        for attr in self.__slots__:
            val = getattr(self, attr, None)
            if val is None or isinstance(val, (int, str, float, dict, np.ndarray)):
                continue
            if hasattr(val, 'cpu'):
                setattr(self, attr, val.detach().cpu().numpy())
        self.backend_mode = "numpy"
        return self

    def __repr__(self) -> str:
        return (
            f"Topology(states={self.num_states}, depth={self.n_depth}, "
            f"width={self.n_width}, step={self.n_step}, levels={self.num_levels}, "
            f"mask_dim={self.mask_dim}, div_h={self.divisor_h}, "
            f"div_w={self.divisor_w}, top_k={self.top_k}, "
            f"mode={self.backend_mode})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Lens Operations — combine (N, N) alignment matrices
# ══════════════════════════════════════════════════════════════════════════════

def lens_additive(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Sum. Reinforces where either topology has alignment."""
    return a + b


def lens_multiplicative(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Product. Only passes where BOTH topologies agree."""
    return a * b


def lens_xor(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Symmetric difference. High where topologies DISAGREE."""
    return a + b - 2.0 * a * b


def lens_screen(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Screen blend. Like additive but saturates at 1."""
    return a + b - a * b


LENS_OPS = {
    "additive": lens_additive,
    "multiplicative": lens_multiplicative,
    "xor": lens_xor,
    "screen": lens_screen,
}


def combine_alignments(
    matrices: List[ArrayLike],
    ops: Optional[List[str]] = None,
) -> ArrayLike:
    """
    Combine multiple (N, N) alignment matrices with lens operations.

    Args:
        matrices: list of (N, N) alignment matrices
        ops: list of lens op names, len = len(matrices) - 1
             default: all "multiplicative"
    Returns:
        (N, N) combined boundary bias
    """
    if len(matrices) == 1:
        return matrices[0]
    if ops is None:
        ops = ["multiplicative"] * (len(matrices) - 1)
    assert len(ops) == len(matrices) - 1

    result = matrices[0]
    for i, op_name in enumerate(ops):
        result = LENS_OPS[op_name](result, matrices[i + 1])
    return result


def alignment_from_masks(masks: ArrayLike, B: Backend) -> ArrayLike:
    """(N, D) masks → (N, N) cosine similarity matrix."""
    norms = B.clamp(B.norm(masks, axis=-1, keepdims=True), lo=1e-8)
    normed = masks / norms
    return B.mm(normed, B.transpose(normed))


# ══════════════════════════════════════════════════════════════════════════════
# TopoBuilder
# ══════════════════════════════════════════════════════════════════════════════

class TopoBuilder:
    """
    Builds a frozen Topology from configuration + projection.

    Usage:
        topo = TopoBuilder(num_states=8).build()

        topo = (TopoBuilder(num_states=32, mode="torch", device="cuda")
            .set_grid(n_depth=8, n_width=4, n_step=4)
            .set_step(min_step=0.0, max_step=1.0, start_step=0.0, end_step=0.25)
            .set_projection(MobiusProjection(
                n_waves=5,
                frequencies=[0.5, 1.0, 1.5, 2.0, 3.0],
                phases=[0.0, 0.3, 0.6, 0.9, 1.2],
            ))
            .set_scales(scale_range=(0.25, 2.75), scale_span=1.5, scale_center=1.0)
            .set_deform(deform_range=(0.1, 0.25))
            .set_twist(twist_range=(0.0, math.pi / 2))
            .set_mask(divisor_h=5, divisor_w=3, top_k=7)
            .set_levels(num_levels=4, level_width_center=0.5)
            .set_seed(42)
            .build())

        topo2 = TopoBuilder.from_recipe(topo.recipe).build()
    """

    def __init__(
        self,
        num_states: int = 8,
        mode: str = "numpy",
        device: str = "cpu",
        dtype: str = "float32",
    ):
        self._num_states = num_states
        self._mode = mode
        self._device = device
        self._dtype = dtype

        # Grid
        self._n_depth: Optional[int] = None
        self._n_width: int = 1
        self._n_step: int = 1

        # Step spectrum (3rd topology axis)
        self._min_step: float = 0.0
        self._max_step: float = 1.0
        self._start_step: float = 0.0
        self._end_step: float = 1.0

        # Projection
        self._projection: Optional[TopologyProjection] = None
        self._extra_projections: List[TopologyProjection] = []
        self._lens_ops: List[str] = []

        # Scale
        self._scale_range: Tuple[float, float] = (0.25, 2.75)
        self._scale_span: float = 2.5
        self._scale_center: float = 1.5
        self._scale_window_width: float = 0.25

        # Deformation
        self._deform_range: Tuple[float, float] = (0.1, 0.25)

        # Twist + rotation
        self._twist_range: Tuple[float, float] = (0.0, math.pi)
        self._rotation_dim: int = 2
        self._rotation_plane: Tuple[int, int] = (0, 1)

        # Mask
        self._divisor_h: int = 3
        self._divisor_w: int = 3
        self._top_k: int = 5

        # Levels
        self._num_levels: int = 1
        self._level_width_center: float = 0.5

        # Seed
        self._seed: int = 42

    # ── Fluent setters ──

    def set_grid(self, n_depth: Optional[int] = None, n_width: int = 1, n_step: int = 1) -> "TopoBuilder":
        self._n_depth = n_depth
        self._n_width = n_width
        self._n_step = n_step
        return self

    def set_projection(self, projection: TopologyProjection) -> "TopoBuilder":
        self._projection = projection
        return self

    def add_projection(self, projection: TopologyProjection, lens_op: str = "multiplicative") -> "TopoBuilder":
        """Add an additional projection. Its alignment combines with prior via lens_op."""
        assert lens_op in LENS_OPS, f"Unknown lens op: {lens_op}"
        self._extra_projections.append(projection)
        self._lens_ops.append(lens_op)
        return self

    def set_step(
        self,
        min_step: float = 0.0,
        max_step: float = 1.0,
        start_step: float = 0.0,
        end_step: float = 1.0,
    ) -> "TopoBuilder":
        """
        Configure 3rd topology axis (step spectrum).

        min_step/max_step: full range of the step spectrum.
        start_step/end_step: sector within that range for this topology.
            Like diffusion timestep windowing — a Sector might own
            steps 0.0-0.25, another 0.25-0.5, etc.
        """
        self._min_step = min_step
        self._max_step = max_step
        self._start_step = start_step
        self._end_step = end_step
        return self

    def set_scales(
        self,
        scale_range: Tuple[float, float] = (0.25, 2.75),
        scale_span: float = 2.5,
        scale_center: float = 1.5,
        scale_window_width: float = 0.25,
    ) -> "TopoBuilder":
        self._scale_range = scale_range
        self._scale_span = scale_span
        self._scale_center = scale_center
        self._scale_window_width = scale_window_width
        return self

    def set_deform(self, deform_range: Tuple[float, float]) -> "TopoBuilder":
        self._deform_range = deform_range
        return self

    def set_twist(
        self,
        twist_range: Tuple[float, float] = (0.0, math.pi),
        rotation_dim: int = 2,
        rotation_plane: Tuple[int, int] = (0, 1),
    ) -> "TopoBuilder":
        self._twist_range = twist_range
        self._rotation_dim = rotation_dim
        self._rotation_plane = rotation_plane
        return self

    def set_mask(
        self,
        divisor_h: int = 3,
        divisor_w: int = 3,
        top_k: int = 5,
    ) -> "TopoBuilder":
        self._divisor_h = divisor_h
        self._divisor_w = divisor_w
        self._top_k = top_k
        return self

    def set_levels(self, num_levels: int, level_width_center: float = 0.5) -> "TopoBuilder":
        self._num_levels = num_levels
        self._level_width_center = level_width_center
        return self

    def set_seed(self, seed: int) -> "TopoBuilder":
        self._seed = seed
        return self

    # ── Build ──

    def build(self) -> Topology:
        B = Backend(mode=self._mode, device=self._device, dtype=self._dtype)
        topo = Topology()
        N = self._num_states

        # Resolve grid
        n_width = self._n_width
        n_depth = self._n_depth if self._n_depth is not None else N
        if n_width > 1 and self._n_depth is None:
            n_depth = math.ceil(N / n_width)

        projection = self._projection or MobiusProjection(scale_range=self._scale_range)

        # ── Scalars ──
        topo.num_states = N
        topo.n_depth = n_depth
        topo.n_width = n_width
        topo.n_step = self._n_step
        topo.divisor_h = self._divisor_h
        topo.divisor_w = self._divisor_w
        topo.top_k = self._top_k
        topo.mask_dim = projection.mask_dim()
        topo.num_levels = self._num_levels
        topo.level_width_center = self._level_width_center
        topo.backend_mode = self._mode

        # ── 2D positions ──
        if n_width > 1:
            d = B.linspace(0, 1, n_depth + 2)[1:-1]
            w = B.linspace(0, 1, n_width + 2)[1:-1]
            grid_d, grid_w = B.meshgrid(d, w, indexing="ij")
            depth_t = B.slice_to(B.reshape(grid_d, (-1,)), N)
            width_t = B.slice_to(B.reshape(grid_w, (-1,)), N)
        else:
            depth_t = B.linspace(0, 1, N + 2)[1:-1]
            width_t = B.zeros(N)
            for i in range(N):
                rng = B.make_rng(self._seed * 7919 + i)
                width_t[i] = B.rand((1,), rng)[0]

        topo.positions_2d = B.stack([depth_t, width_t], axis=-1)
        topo.positions = depth_t
        topo.depth_positions = depth_t
        topo.width_positions = width_t

        # ── Step positions (3rd axis, sector within min/max) ──
        step_range = self._end_step - self._start_step
        if self._n_step > 1:
            step_t = B.linspace(self._start_step, self._end_step, self._n_step + 2)[1:-1]
            # Tile to cover all N states if n_step < N
            if self._n_step < N:
                repeats = math.ceil(N / self._n_step)
                step_t = B.slice_to(B.cat([step_t] * repeats), N)
        else:
            # Single step: center of sector
            sector_center = (self._start_step + self._end_step) / 2.0
            step_t = B.full((N,), sector_center)

        topo.step_positions = step_t
        topo.positions_3d = B.stack([depth_t, width_t, step_t], axis=-1)

        # ── Alpha masks ──
        topo.alpha_masks = projection(topo.positions_3d, B)

        # ── Scale windows ──
        span = self._scale_span
        center = self._scale_center
        s_lo, s_hi = self._scale_range
        base = center - span / 2.0
        win_lo = B.clamp(base + depth_t * span, lo=s_lo, hi=s_hi)
        win_hi = B.clamp(win_lo + self._scale_window_width, lo=s_lo, hi=s_hi)
        topo.scale_windows = B.stack([win_lo, win_hi], axis=-1)

        # ── Deformation tolerance ──
        d_lo, d_hi = self._deform_range
        topo.deform_scales = d_lo + depth_t * (d_hi - d_lo)

        # ── Twist angles ──
        tw_lo, tw_hi = self._twist_range
        topo.twist_angles = tw_lo + depth_t * (tw_hi - tw_lo)

        # ── Rotation matrices from twist angles ──
        if self._rotation_dim == 2:
            topo.rotation_matrices = B.rotation_2d(topo.twist_angles)
        else:
            topo.rotation_matrices = B.givens(
                topo.twist_angles, self._rotation_dim, self._rotation_plane)

        # ── Divisor indices ──
        idx = B.arange_int(N)
        topo.divisor_indices_h = B.to_int(B.mod(idx, self._divisor_h))
        topo.divisor_indices_w = B.to_int(B.mod(idx, self._divisor_w))

        # ── Branch paths ──
        k = min(self._top_k, topo.mask_dim)
        _, topk_idx = B.topk(topo.alpha_masks, k, axis=-1)
        topo.branch_paths = B.to_int(topk_idx)

        # ── Staircase values ──
        M = topo.mask_dim
        weights = B.linspace(1.0 / M, 1.0, M)
        topo.staircase_vals = B.sum(topo.alpha_masks * B.unsqueeze(weights, 0), axis=-1)

        # ── Alignment matrix + boundary bias ──
        alignments = [alignment_from_masks(topo.alpha_masks, B)]

        for extra_proj in self._extra_projections:
            extra_masks = extra_proj(topo.positions_3d, B)
            alignments.append(alignment_from_masks(extra_masks, B))

        topo.per_projection_alignments = alignments
        topo.alignment_matrix = alignments[0]
        topo.boundary_bias = combine_alignments(alignments, self._lens_ops or None)

        # ── Multi-level ──
        if self._num_levels > 1:
            self._build_levels(topo, projection, B)
        else:
            topo.level_scale_windows = None
            topo.level_deform_scales = None
            topo.level_twist_angles = None
            topo.level_alpha_masks = None
            topo.level_rotation_matrices = None

        # ── Recipe ──
        topo.recipe = self._make_recipe(projection)

        return topo

    def _build_levels(self, topo: Topology, projection: TopologyProjection, B: Backend):
        """Per-level topology for depth progression."""
        L = self._num_levels
        level_t = B.linspace(0, 1, L)

        # Scale
        s_lo, s_hi = self._scale_range
        span = self._scale_span
        center = self._scale_center
        base = center - span / 2.0
        lev_lo = B.clamp(base + level_t * span, lo=s_lo, hi=s_hi)
        lev_hi = B.clamp(lev_lo + self._scale_window_width, lo=s_lo, hi=s_hi)
        topo.level_scale_windows = B.stack([lev_lo, lev_hi], axis=-1)

        # Deform
        d_lo, d_hi = self._deform_range
        topo.level_deform_scales = d_lo + level_t * (d_hi - d_lo)

        # Twist
        tw_lo, tw_hi = self._twist_range
        topo.level_twist_angles = tw_lo + level_t * (tw_hi - tw_lo)

        # Level rotations
        if self._rotation_dim == 2:
            topo.level_rotation_matrices = B.rotation_2d(topo.level_twist_angles)
        else:
            topo.level_rotation_matrices = B.givens(
                topo.level_twist_angles, self._rotation_dim, self._rotation_plane)

        # Level masks — width from config, not hardcoded
        level_width = B.full((L,), self._level_width_center)
        level_step = B.full((L,), (self._start_step + self._end_step) / 2.0)
        level_positions = B.stack([level_t, level_width, level_step], axis=-1)
        topo.level_alpha_masks = projection(level_positions, B)

    def _make_recipe(self, projection: TopologyProjection) -> Dict[str, Any]:
        """Serializable recipe for reconstruction."""
        proj_config: Dict[str, Any] = {}
        if isinstance(projection, MobiusProjection):
            proj_config = {
                "scale_range": projection.scale_range,
                "n_waves": projection.n_waves,
                "frequencies": projection.frequencies,
                "phases": projection.phases,
                "omega": projection.omega,
                "alpha_base": projection.alpha_base,
            }
        elif isinstance(projection, CantorProjection):
            proj_config = {
                "levels": projection.levels,
                "base": projection.base,
            }

        return {
            "num_states": self._num_states,
            "n_depth": self._n_depth,
            "n_width": self._n_width,
            "n_step": self._n_step,
            "min_step": self._min_step,
            "max_step": self._max_step,
            "start_step": self._start_step,
            "end_step": self._end_step,
            "seed": self._seed,
            "mode": self._mode,
            "device": self._device,
            "dtype": self._dtype,
            "projection_class": type(projection).__name__,
            "projection_config": proj_config,
            "scale_range": self._scale_range,
            "scale_span": self._scale_span,
            "scale_center": self._scale_center,
            "scale_window_width": self._scale_window_width,
            "deform_range": self._deform_range,
            "twist_range": self._twist_range,
            "rotation_dim": self._rotation_dim,
            "rotation_plane": self._rotation_plane,
            "divisor_h": self._divisor_h,
            "divisor_w": self._divisor_w,
            "top_k": self._top_k,
            "num_levels": self._num_levels,
            "level_width_center": self._level_width_center,
        }

    # ── Recipe reconstruction ──

    @classmethod
    def from_recipe(cls, recipe: Dict[str, Any]) -> "TopoBuilder":
        builder = cls(
            num_states=recipe["num_states"],
            mode=recipe.get("mode", "numpy"),
            device=recipe.get("device", "cpu"),
            dtype=recipe.get("dtype", "float32"),
        )
        builder.set_grid(
            n_depth=recipe.get("n_depth"),
            n_width=recipe.get("n_width", 1),
            n_step=recipe.get("n_step", 1),
        )
        builder.set_step(
            min_step=recipe.get("min_step", 0.0),
            max_step=recipe.get("max_step", 1.0),
            start_step=recipe.get("start_step", 0.0),
            end_step=recipe.get("end_step", 1.0),
        )
        builder.set_scales(
            scale_range=recipe.get("scale_range", (0.25, 2.75)),
            scale_span=recipe.get("scale_span", 2.5),
            scale_center=recipe.get("scale_center", 1.5),
            scale_window_width=recipe.get("scale_window_width", 0.25),
        )
        builder.set_deform(recipe.get("deform_range", (0.1, 0.25)))
        builder.set_twist(
            recipe.get("twist_range", (0.0, math.pi)),
            rotation_dim=recipe.get("rotation_dim", 2),
            rotation_plane=tuple(recipe.get("rotation_plane", (0, 1))),
        )
        builder.set_mask(
            divisor_h=recipe.get("divisor_h", 3),
            divisor_w=recipe.get("divisor_w", 3),
            top_k=recipe.get("top_k", 5),
        )
        builder.set_levels(
            num_levels=recipe.get("num_levels", 1),
            level_width_center=recipe.get("level_width_center", 0.5),
        )
        builder.set_seed(recipe.get("seed", 42))

        proj_config = recipe.get("projection_config", {})
        proj_name = recipe.get("projection_class", "MobiusProjection")
        if proj_name == "MobiusProjection":
            builder.set_projection(MobiusProjection(**proj_config))
        elif proj_name == "CantorProjection":
            builder.set_projection(CantorProjection(**proj_config))

        return builder


# ══════════════════════════════════════════════════════════════════════════════
# Backward compat
# ══════════════════════════════════════════════════════════════════════════════

def CantorTopology(
    num_states: int = 512, levels: int = 9, base: int = 3
) -> Topology:
    """Drop-in replacement for old CantorTopology class."""
    return (TopoBuilder(num_states=num_states)
        .set_projection(CantorProjection(levels=levels, base=base))
        .set_seed(42)
        .build())