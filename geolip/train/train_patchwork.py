"""
train_patchwork.py — Train the 8-state Patchwork on geometric shapes
=====================================================================

8 × 256 × 64 = 131,072 tokens.
Multi-label classification: which shapes present in each sample.
Crystal similarity scoring (no CE on geometric features).

Usage:
    python -m geolip.script.train_patchwork [--epochs 50] [--device cuda]
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from geolip.core import KSimplexLinear, CantorTopology
from geolip.scales.patchwork import Patchwork
from geolip.train.dataloader import LazyShapeDataset, PreparedDataset, make_dataloader
from geolip.train.voxel_generator import NUM_CLASSES, CLASS_NAMES


# =============================================================================
# Classification Head
# =============================================================================

class GeometricClassificationHead(nn.Module):
    """
    Crystal similarity for multi-label classification.
    No CE on geometric features.
    """

    def __init__(self, state_dim: int, num_classes: int, simplex_k: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.crystals = nn.Parameter(torch.randn(num_classes, state_dim) * 0.02)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, composed):
        """composed: (B, 256, 64) → (B, num_classes) logits."""
        B = composed.shape[0]
        flat = composed.reshape(B, -1)
        f_hat = F.normalize(flat, dim=-1)
        c_hat = F.normalize(self.crystals, dim=-1)
        return torch.matmul(f_hat, c_hat.T) / self.temperature


# =============================================================================
# Loss
# =============================================================================

class PatchworkLoss(nn.Module):
    """Multi-label shape + CM validity + crystal separation."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels, cm_loss, crystals):
        shape_loss = F.binary_cross_entropy_with_logits(logits, labels)

        c_hat = F.normalize(crystals, dim=-1)
        cd = torch.cdist(c_hat, c_hat)
        C = self.num_classes
        mask = torch.triu(torch.ones(C, C, device=cd.device), diagonal=1).bool()
        sep_loss = F.relu(1.0 - cd[mask]).pow(2).mean()

        total = shape_loss + 0.1 * cm_loss + 0.1 * sep_loss

        with torch.no_grad():
            preds = (logits > 0).float()
            acc = (preds == labels).float().mean()

        return total, {
            "shape": shape_loss.detach(),
            "cm": cm_loss.detach(),
            "sep": sep_loss.detach(),
            "total": total.detach(),
            "acc": acc,
        }


# =============================================================================
# Training
# =============================================================================

def train(
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    n_train: int = 8000,
    n_val: int = 1600,
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("=" * 70)
    print("Patchwork Training — 8 × 256 × 64 = 131,072 tokens")
    print("=" * 70)

    # Topology
    topo = CantorTopology(num_states=8, levels=5)
    print("\nTopology (8 states):")
    for i in range(8):
        bp = topo.branch_paths[i].tolist()
        sv = float(topo.staircase_vals[i])
        print(f"  State {i}: path={bp} staircase={sv:.4f}")

    # Data
    print(f"\nGenerating {n_train} train + {n_val} val samples...")
    train_ds = PreparedDataset.from_factory(n_samples=n_train, seed=42)
    val_ds = PreparedDataset.from_factory(n_samples=n_val, seed=999)
    train_loader = make_dataloader(train_ds, batch_size=batch_size, num_workers=0)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = Patchwork(
        in_channels=32, in_height=64, in_width=64,
        patch_grid=8, tokens_per_state=256, token_dim=64,
    ).to(device)

    state_dim = 256 * 64
    cls_head = GeometricClassificationHead(state_dim, NUM_CLASSES).to(device)
    criterion = PatchworkLoss(NUM_CLASSES)

    all_params = list(model.parameters()) + list(cls_head.parameters())
    total = sum(p.numel() for p in all_params)
    print(f"\nTotal parameters: {total:,}")

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train
    print(f"\nTraining for {epochs} epochs...\n")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        cls_head.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for latents, labels in train_loader:
            latents = latents.to(device)
            labels = labels.to(device)

            output = model(latents)
            logits = cls_head(output.composed)
            loss, info = criterion(logits, labels, output.loss, cls_head.crystals)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            epoch_loss += float(info["total"])
            epoch_acc += float(info["acc"])
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        cls_head.eval()
        val_acc = 0.0
        val_vol = 0.0
        vn = 0

        with torch.no_grad():
            for latents, labels in val_loader:
                latents = latents.to(device)
                labels = labels.to(device)
                output = model(latents)
                logits = cls_head(output.composed)
                _, info = criterion(logits, labels, output.loss, cls_head.crystals)
                val_acc += float(info["acc"])
                val_vol += float((output.vol_sq > 0).float().mean())
                vn += 1

        ta = epoch_loss / n_batches
        train_acc = epoch_acc / n_batches
        va = val_acc / vn
        vv = val_vol / vn

        if va > best_acc:
            best_acc = va

        print(
            f"  Epoch {epoch:3d}: loss={ta:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={va:.4f}  "
            f"vol_valid={vv:.2%}  best={best_acc:.4f}"
        )

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    return model, cls_head


def main():
    parser = argparse.ArgumentParser(description="Train Patchwork")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_train", type=int, default=8000)
    parser.add_argument("--n_val", type=int, default=1600)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()