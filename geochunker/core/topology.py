"""
Deterministic Cantor Topology
==============================

Pure mathematics. No learnable parameters.
Partitions [0,1] via Devil's Staircase into unique adjacent positions.
Each position has a branch path that IS its identity.

S5.1: Staircase
S5.3: Alignment (NOT distance)
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple


class CantorTopology:
    """
    Deterministic topological assignment for N states.

    Branch paths determine:
        - Which topological state this is
        - How it composes with other states (alignment)
        - Its fractal depth and hierarchical role
    """

    def __init__(self, num_states: int = 512, levels: int = 9, base: int = 3):
        self.num_states = num_states
        self.levels = levels
        self.base = base

        positions = torch.linspace(0, 1, num_states + 2)[1:-1]
        staircase_vals, branch_paths = self._compute_all(positions)

        self.positions = positions
        self.staircase_vals = staircase_vals
        self.branch_paths = branch_paths
        self.alignment_matrix = self._compute_alignment_matrix()

    def _staircase(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.to(torch.float64).clamp(1e-10, 1.0 - 1e-10)
        centers = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        weights = torch.tensor(
            [0.5 ** k for k in range(1, self.levels + 1)], dtype=torch.float64
        )
        scales = torch.tensor(
            [self.base ** k for k in range(1, self.levels + 1)], dtype=torch.float64
        )
        tau = 0.25
        alpha = 0.5  # triadic equilibrium

        y_all = (x.unsqueeze(-1) * scales) % self.base
        d2 = (y_all.unsqueeze(-1) - centers) ** 2
        p = F.softmax(-d2 / tau, dim=-1)
        bits = p[..., 2] + alpha * p[..., 1]
        cantor = (bits * weights).sum(dim=-1)
        paths = p.argmax(dim=-1)
        return cantor.float(), paths.int()

    def _compute_all(self, positions):
        return self._staircase(positions)

    def _compute_alignment_matrix(self) -> Tensor:
        """S5.3: Pairwise hierarchical alignment."""
        weights = torch.tensor(
            [0.5 ** k for k in range(1, self.levels + 1)], dtype=torch.float32
        )
        paths = self.branch_paths
        match = (paths.unsqueeze(1) == paths.unsqueeze(0)).float()
        return (match * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

    def get_state_identity(self, state_idx: int) -> Dict:
        return {
            "index": state_idx,
            "position": self.positions[state_idx].item(),
            "staircase": self.staircase_vals[state_idx].item(),
            "branch_path": self.branch_paths[state_idx].tolist(),
        }

    def get_alignment(self, i: int, j: int) -> float:
        return self.alignment_matrix[i, j].item()

    def get_neighborhood(self, state_idx: int, threshold: float = 0.5) -> List[int]:
        row = self.alignment_matrix[state_idx]
        return (row > threshold).nonzero(as_tuple=True)[0].tolist()