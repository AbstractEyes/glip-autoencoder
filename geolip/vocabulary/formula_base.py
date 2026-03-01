"""
FormulaBase
-----------
Abstract base class for stateless mathematical computations.

Design Philosophy:
    Factories CREATE objects; Formulas COMPUTE properties.

    - Formulas are stateless computations (same input → same output)
    - Support both NumPy and PyTorch backends
    - Return structured result dicts for introspection
    - Validation hooks for input checking

    Similar to FactoryBase but focused on computation rather than construction.

geolip.vocabulary.formula_base

License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class FormulaBase(ABC):
    """
    Abstract base for mathematical formula computations.

    Subclasses implement:
        - compute_numpy(): NumPy backend computation
        - compute_torch(): PyTorch backend computation (optional)
        - validate_input(): Optional input validation
        - info(): Formula metadata

    Results are always returned as dicts for structured access.

    Attributes:
        name: Human-readable formula name
        uid: Unique identifier for registry systems
    """

    def __init__(self, name: str, uid: str):
        """
        Initialize formula with identifying metadata.

        Args:
            name: Human-readable name (e.g., "cayley_menger")
            uid: Unique identifier (e.g., "formula.cayley_menger.volume")
        """
        self.name = name
        self.uid = uid

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Abstract computation methods (subclasses MUST implement)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @abstractmethod
    def compute_numpy(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Compute using NumPy backend.

        Returns:
            Dict of named results (arrays, scalars, etc.)
        """
        pass

    def compute_torch(self, *args, **kwargs) -> Dict[str, "torch.Tensor"]:
        """
        Compute using PyTorch backend (optional override).

        Default implementation: convert to numpy, compute, convert back.
        Subclasses SHOULD override for differentiable computation.

        Returns:
            Dict of named results (tensors)
        """
        if not HAS_TORCH:
            raise RuntimeError(f"PyTorch required for compute_torch() in {self.name}")

        # Default: convert to numpy and back (NOT differentiable)
        np_args = []
        for a in args:
            if isinstance(a, torch.Tensor):
                np_args.append(a.detach().cpu().numpy())
            else:
                np_args.append(a)

        np_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                np_kwargs[k] = v.detach().cpu().numpy()
            else:
                np_kwargs[k] = v

        result = self.compute_numpy(*np_args, **np_kwargs)

        # Convert results back to tensors
        torch_result = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                torch_result[k] = torch.from_numpy(v)
            elif isinstance(v, (bool, np.bool_)):
                torch_result[k] = torch.tensor(v)
            elif isinstance(v, (int, float)):
                torch_result[k] = torch.tensor(v)
            else:
                torch_result[k] = v

        return torch_result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Unified compute interface
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def compute(
        self,
        *args,
        backend: Optional[str] = None,
        validate: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Unified compute interface with automatic backend selection.

        If backend is None, infers from the first tensor argument:
            - torch.Tensor → "torch"
            - np.ndarray or other → "numpy"

        Args:
            *args: Formula-specific inputs
            backend: "numpy" or "torch" (auto-detected if None)
            validate: Run input validation before compute
            **kwargs: Formula-specific keyword arguments

        Returns:
            Dict of named results
        """
        # Auto-detect backend
        if backend is None:
            backend = self._detect_backend(*args, **kwargs)

        # Optional input validation
        if validate:
            is_valid, error_msg = self.validate_input(*args, **kwargs)
            if not is_valid:
                raise ValueError(f"Formula {self.name} input invalid: {error_msg}")

        if backend == "numpy":
            return self.compute_numpy(*args, **kwargs)
        elif backend == "torch":
            if not HAS_TORCH:
                raise RuntimeError(f"PyTorch required for backend='torch' in {self.name}")
            return self.compute_torch(*args, **kwargs)
        else:
            raise ValueError(f"Invalid backend '{backend}' (allowed: 'numpy', 'torch')")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Forward alias (nn.Module compatibility)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for compute() — matches nn.Module convention."""
        return self.compute(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """Make formula callable."""
        return self.compute(*args, **kwargs)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Optional validation and metadata
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def validate_input(self, *args, **kwargs) -> tuple:
        """
        Validate inputs before computation.

        Returns:
            (is_valid, error_message)
        """
        return True, ""

    def info(self) -> Dict[str, Any]:
        """Formula metadata for introspection."""
        return {
            "name": self.name,
            "uid": self.uid,
            "description": "No description provided",
            "backend_support": {
                "numpy": True,
                "torch": HAS_TORCH,
            },
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Utility
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _detect_backend(*args, **kwargs) -> str:
        """Detect backend from input types."""
        if HAS_TORCH:
            for a in args:
                if isinstance(a, torch.Tensor):
                    return "torch"
            for v in kwargs.values():
                if isinstance(v, torch.Tensor):
                    return "torch"
        return "numpy"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', uid='{self.uid}')"

    def __str__(self) -> str:
        return f"Formula[{self.name}] ({self.__class__.__name__})"


if __name__ == "__main__":
    print("FormulaBase — abstract base class for mathematical computations")
    print("Subclass and implement compute_numpy() / compute_torch()")
    print(f"PyTorch available: {HAS_TORCH}")