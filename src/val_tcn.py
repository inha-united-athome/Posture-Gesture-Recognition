"""
Evaluate trained TCN model on NTU skeleton dataset.

Usage:
    python src/val_tcn.py
    python src/val_tcn.py --model models/best_tcn_xsub.pth --split xsub_val
    python src/val_tcn.py --split xview_val --model models/best_tcn_xview.pth
"""

import sys
import os
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
import numpy as np
from torch.utils.data import DataLoader

from data.ntu_dataset import NTUSkeletonDataset
from models.tcn import TCN


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TCN on NTU skeleton data")
    parser.add_argument("--pkl", type=str, default="wholebody_5class.pkl")
    parser.add_argument("--split", type=str, default="xsub_val",
                        choices=["xsub_train", "xsub_val", "xview_train", "xview_val",
                                 "xset_train", "xset_val"])
    parser.add_argument("--model", type=str, default=None,
                        help="Model checkpoint path (default: auto from split name)")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        logits = model(features, mask)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += features.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    return accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pkl_path = os.path.join(ROOT_DIR, args.pkl) if not os.path.isabs(args.pkl) else args.pkl

    # Auto model path
    if args.model is None:
        split_base = args.split.replace("_train", "").replace("_val", "")
        model_path = os.path.join(ROOT_DIR, "models", f"best_tcn_{split_base}.pth")
    else:
        model_path = args.model

    # Load checkpoint
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    input_dim = checkpoint.get("input_dim", 130)
    num_classes = checkpoint.get("num_classes", 5)
    hidden_dims = checkpoint.get("hidden_dims", [64, 128, 128, 256])
    kernel_size = checkpoint.get("kernel_size", 5)
    dropout = checkpoint.get("dropout", 0.3)
    max_frames = checkpoint.get("max_frames", 120)
    num_joints = checkpoint.get("num_joints", input_dim // 2)

    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Val Acc: {checkpoint.get('val_acc', '?'):.4f}")
    print(f"  Config: joints={num_joints}, dim={input_dim}, classes={num_classes}, "
          f"hidden={hidden_dims}, kernel={kernel_size}")

    # Model
    model = TCN(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = NTUSkeletonDataset(
        pkl_path, split=args.split,
        max_frames=max_frames, augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Evaluate
    accuracy, preds, labels, probs = evaluate(model, loader, device)

    # Results
    print(f"\n{'='*60}")
    print(f"  {args.split} Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy:.4f} ({int(accuracy * len(labels))}/{len(labels)})")

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1

    class_names = [dataset.get_action_name(i) for i in range(num_classes)]

    print(f"\n  Confusion Matrix:")
    header = "  " + "True\\Pred".ljust(16) + "".join(n.ljust(16) for n in class_names)
    print(header)
    for i in range(num_classes):
        row = "  " + class_names[i].ljust(16) + "".join(str(cm[i][j]).ljust(16) for j in range(num_classes))
        print(row)

    print(f"\n  Per-Class Metrics:")
    for i in range(num_classes):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"    {class_names[i]:16s}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

    # Top-k confidence analysis
    print(f"\n  Confidence Analysis:")
    for i in range(num_classes):
        mask_cls = labels == i
        if mask_cls.sum() > 0:
            cls_probs = probs[mask_cls, i]
            print(f"    {class_names[i]:16s}  "
                  f"mean_conf={cls_probs.mean():.4f}  "
                  f"min={cls_probs.min():.4f}  "
                  f"max={cls_probs.max():.4f}")

    print()


if __name__ == "__main__":
    main()
