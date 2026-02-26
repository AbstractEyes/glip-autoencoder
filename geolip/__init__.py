"""
geolip — Geometric linear interpolative patchwork Vocabulary
=============================================

512 × 256 × 64 = 8,388,608 tokens.
Deterministic topological token architecture.

Hierarchy:
    Sector    → 512 Chunks
    Chunk     → 512 × 256 × 64
    Patchwork → 8 × 256 × 64 (prototype)
    State     → 256 × 64 (single topological state)
    Patch     → single spatial position from input
"""

__version__ = "0.1.0"
__author__ = "AbstractPhil"
__email__ = "abstractpowered@outlook"