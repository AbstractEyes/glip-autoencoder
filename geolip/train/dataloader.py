"""
Data Loading & Preparation
============================

Dataset and DataLoader utilities for GLIP training.
Consumes GeometricShapeFactory (and VoxelShapeGenerator)
to produce torch-ready batches.

Prep strategies:
    - LazyShapeDataset:   generates on __getitem__, no RAM, worker-friendly
    - PreparedDataset:    pre-generates N samples to disk or memory
    - prepare_to_disk:    write N samples as .pt files for fast loading
    - make_dataloader:    convenience constructor with sane defaults

Author: AbstractPhil + Claude
"""

import os
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from pathlib import Path

from .geometric_generator import GeometricShapeFactory
from .voxel_generator import NUM_CLASSES, CLASS_TO_IDX


# ---------------------------------------------------------------------------
# Lazy dataset — generate per-worker, no prep step
# ---------------------------------------------------------------------------

class LazyShapeDataset(Dataset):
    """
    On-the-fly generation using GeometricShapeFactory.

    Each worker gets a deterministic factory seeded by (base_seed + idx),
    so results are reproducible and workers don't collide.

    Args:
        n_samples: virtual dataset size
        channels:  latent channel count
        height:    spatial height
        width:     spatial width
        n_shapes:  fixed shape count per sample, or None for random 2-5
        seed:      base seed for reproducibility
    """

    def __init__(
        self,
        n_samples: int = 10000,
        channels: int = 32,
        height: int = 64,
        width: int = 64,
        n_shapes: Optional[int] = None,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.channels = channels
        self.height = height
        self.width = width
        self.n_shapes = n_shapes
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        factory = GeometricShapeFactory(
            width=self.width, height=self.height, seed=self.seed + idx
        )
        latent_np, labels, _, _ = factory(
            channels=self.channels, n_shapes=self.n_shapes
        )
        label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        for name in labels:
            label_vec[CLASS_TO_IDX[name]] = 1.0
        return torch.from_numpy(latent_np), torch.from_numpy(label_vec)


# ---------------------------------------------------------------------------
# Prepared dataset — pre-generate into memory
# ---------------------------------------------------------------------------

class PreparedDataset(Dataset):
    """
    Pre-generated samples held in memory as tensors.

    Use when dataset fits in RAM and you want fast epochs.
    Construct via PreparedDataset.from_factory() or
    PreparedDataset.from_disk().

    Args:
        latents: (N, C, H, W) tensor
        labels:  (N, NUM_CLASSES) tensor
    """

    def __init__(self, latents: Tensor, labels: Tensor):
        assert latents.shape[0] == labels.shape[0]
        self.latents = latents
        self.labels = labels

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.latents[idx], self.labels[idx]

    @classmethod
    def from_factory(
        cls,
        n_samples: int,
        channels: int = 32,
        height: int = 64,
        width: int = 64,
        n_shapes: Optional[int] = None,
        seed: int = 42,
    ) -> "PreparedDataset":
        """Pre-generate n_samples into memory."""
        factory = GeometricShapeFactory(width=width, height=height, seed=seed)
        latents = []
        labels = []
        for latent_np, sample_labels, _, _ in factory.stream(n_samples, channels, n_shapes):
            latents.append(latent_np)
            label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            for name in sample_labels:
                label_vec[CLASS_TO_IDX[name]] = 1.0
            labels.append(label_vec)
        return cls(
            latents=torch.from_numpy(np.stack(latents)),
            labels=torch.from_numpy(np.stack(labels)),
        )

    @classmethod
    def from_disk(cls, path: str) -> "PreparedDataset":
        """Load from a .pt file written by prepare_to_disk."""
        data = torch.load(path, weights_only=True)
        return cls(latents=data["latents"], labels=data["labels"])

    def save(self, path: str):
        """Save to disk as .pt file."""
        torch.save({"latents": self.latents, "labels": self.labels}, path)


# ---------------------------------------------------------------------------
# Disk preparation
# ---------------------------------------------------------------------------

def prepare_to_disk(
    path: str,
    n_samples: int,
    channels: int = 32,
    height: int = 64,
    width: int = 64,
    n_shapes: Optional[int] = None,
    seed: int = 42,
) -> str:
    """
    Generate n_samples and save as a single .pt file.

    Returns the path written to.
    """
    ds = PreparedDataset.from_factory(
        n_samples=n_samples,
        channels=channels,
        height=height,
        width=width,
        n_shapes=n_shapes,
        seed=seed,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.save(path)
    return path


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Convenience DataLoader constructor with training defaults.

    For LazyShapeDataset: use num_workers > 0 for parallel generation.
    For PreparedDataset: num_workers=0 is fine since data is in memory.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )