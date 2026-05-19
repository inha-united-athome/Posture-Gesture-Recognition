"""
Train TCN on 6-class skeleton dataset.

Classes:
    0: idle, 1: waving, 2: hands_up_single, 3: hands_up_both, 4: pointing, 5: stop

Usage:
    python src/train_tcn.py
    python src/train_tcn.py --pkl wholebody_6class.pkl --split xsub --epochs 150
    python src/train_tcn.py --split xset --max_frames 100
"""

import sys
import os
import argparse

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.ntu_dataset import NTUSkeletonDataset
from models.tcn import TCN


def parse_args():
    parser = argparse.ArgumentParser(description="Train TCN on NTU skeleton data")
    parser.add_argument("--pkl", type=str, default="wholebody_6class.pkl",
                        help="Path to wholebody skeleton pkl file")
    parser.add_argument("--split", type=str, default="xsub",
                        choices=["xsub", "xview", "xset"],
                        help="Split protocol: xsub (cross-subject), xview (cross-view, NTU-60), xset (cross-setup, NTU-120)")
    parser.add_argument("--max_frames", type=int, default=120,
                        help="Max sequence length (pad/crop)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_dims", type=int, nargs="+",
                        default=[64, 128, 128, 256],
                        help="TCN hidden dimensions")
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=None,
                        help="Model save path (default: models/best_tcn_{split}.pth)")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for features, mask, labels in pbar:
        features = features.to(device)   # (B, C, T)
        mask = mask.to(device)           # (B, T)
        labels = labels.to(device)       # (B,)

        optimizer.zero_grad()
        logits = model(features, mask)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += features.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        logits = model(features, mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += features.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def print_confusion_matrix(preds, labels, num_classes, class_names=None):
    """Print confusion matrix and per-class metrics."""
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1

    print("\n  ┌─ Confusion Matrix ─────────────────────")
    header = "  │ " + "True\\Pred".ljust(14) + "".join(class_names[i].ljust(14) for i in range(num_classes))
    print(header)
    print("  │ " + "─" * (14 + 14 * num_classes))
    for i in range(num_classes):
        row = "  │ " + class_names[i].ljust(14) + "".join(str(cm[i][j]).ljust(14) for j in range(num_classes))
        print(row)
    print("  └─────────────────────────────────────────")

    print("\n  Per-Class Metrics:")
    for i in range(num_classes):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"    {class_names[i]:14s}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")


def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    pkl_path = os.path.join(ROOT_DIR, args.pkl) if not os.path.isabs(args.pkl) else args.pkl
    save_path = args.save_path or os.path.join(ROOT_DIR, "models", f"best_tcn_{args.split}.pth")

    train_split = f"{args.split}_train"
    val_split = f"{args.split}_val"

    # ===== Datasets =====
    print(f"\n{'='*60}")
    print(f"  Loading data: {args.pkl}")
    print(f"  Split: {args.split} | Max frames: {args.max_frames}")
    print(f"{'='*60}\n")

    train_dataset = NTUSkeletonDataset(
        pkl_path, split=train_split,
        max_frames=args.max_frames, augment=True,
    )
    val_dataset = NTUSkeletonDataset(
        pkl_path, split=val_split,
        max_frames=args.max_frames, augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # ===== Model =====
    num_classes = max(train_dataset.num_classes, val_dataset.num_classes)
    num_joints = train_dataset.num_joints
    input_dim = num_joints * 2  # V joints × 2 (x, y)

    model = TCN(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=args.hidden_dims,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    print(f"\nModel: TCN")
    print(f"  Num joints: {num_joints}")
    print(f"  Input dim: {input_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Kernel size: {args.kernel_size}")
    print(f"  Parameters: {model.count_parameters():,}")

    # ===== Class weights =====
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"  Class weights: {class_weights.tolist()}")

    # ===== Loss & Optimizer =====
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ===== Training loop =====
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Starting training ({args.epochs} epochs)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:3d}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  │  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  │  "
            f"LR: {lr:.6f}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "input_dim": input_dim,
                "num_joints": num_joints,
                "num_classes": num_classes,
                "hidden_dims": args.hidden_dims,
                "kernel_size": args.kernel_size,
                "dropout": args.dropout,
                "max_frames": args.max_frames,
                "split": args.split,
            }, save_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ✗ Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # ===== Final evaluation =====
    print(f"\n{'='*60}")
    print(f"  Final Evaluation on Validation Set")
    print(f"{'='*60}")

    # Load best model
    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded best model from epoch {checkpoint['epoch']} "
          f"(Val Acc: {checkpoint['val_acc']:.4f})")

    val_loss, val_acc, val_preds, val_labels = evaluate(
        model, val_loader, criterion, device
    )
    print(f"\n  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

    class_names = [train_dataset.get_action_name(i) for i in range(num_classes)]
    print_confusion_matrix(val_preds, val_labels, num_classes, class_names)

    print(f"\n✓ Training complete! Model saved to: {save_path}")


if __name__ == "__main__":
    main()
