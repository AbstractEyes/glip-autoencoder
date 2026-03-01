"""
geolip.vocabulary
=================

Mathematical vocabulary for geometric deep learning.

Factories CREATE objects; Formulas COMPUTE properties.

Factories:
    SimplexFactory — generate k-simplices (random, regular, uniform)
    FactoryBase    — abstract base for tensor/object factories

Formulas:
    CayleyMengerFormula    — simplex geometry analysis (volume, degeneracy, regularity)
    CayleyMengerValidator  — training losses (validity, consistency, regularity)
    FormulaBase            — abstract base for mathematical computations

Standalone functions (backward compatible):
    distance_matrix, cayley_menger_determinant, cayley_menger_from_distances,
    gram_volume_sq, simplex_volume, is_valid_simplex, edge_lengths

License: MIT
"""

# Bases
from .factory_base import FactoryBase
from .formula_base import FormulaBase

# Factories
from .simplex_factory import SimplexFactory

# Formulas
from .cayley_menger import (
    CayleyMengerFormula,
    CayleyMengerValidator,
    # Standalone functions
    distance_matrix,
    cayley_menger_determinant,
    cayley_menger_from_distances,
    gram_volume_sq,
    simplex_volume,
    is_valid_simplex,
    edge_lengths,
)

__all__ = [
    # Bases
    "FactoryBase",
    "FormulaBase",
    # Factories
    "SimplexFactory",
    # Formulas
    "CayleyMengerFormula",
    "CayleyMengerValidator",
    # Standalone functions
    "distance_matrix",
    "cayley_menger_determinant",
    "cayley_menger_from_distances",
    "gram_volume_sq",
    "simplex_volume",
    "is_valid_simplex",
    "edge_lengths",
]