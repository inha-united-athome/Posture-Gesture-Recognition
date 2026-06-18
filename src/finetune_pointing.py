"""
pointing 클래스 파인튜닝 스크립트.

- Base: best_tcn_xsub.pth (wholebody_6class.pkl로 학습된 96.94% 모델)
- 추가 데이터: wholebody_8class.pkl의 left_pointing(label5) + right_pointing(label6)
              → pointing(label4)으로 remapping해서 6class pkl에 합산
- 전략: 낮은 lr로 전체 모델 파인튜닝 (pointing 데이터 증가)

Usage:
    python src/finetune_pointing.py
    python src/finetune_pointing.py --base_pkl wholebody_6class.pkl --extra_pkl wholebody_8class.pkl
    python src/finetune_pointing.py --epochs 30 --lr 2e-4
"""

import sys, os, argparse, pickle, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from collections import Counter
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from data.ntu_dataset import NTUSkeletonDataset
from models.tcn import TCN


POINTING_LABEL = 4   # 6-class에서 pointing의 label index

# 8-class → 6-class label remap (pointing만 필요하지만 전체 정의)
REMAP_8TO6 = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5}

CLASS_NAMES = {
    0: "idle",
    1: "waving",
    2: "hands_up_single",
    3: "hands_up_both",
    4: "pointing",
    5: "stop",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_pkl",  type=str, default="wholebody_6class.pkl")
    p.add_argument("--extra_pkl", type=str, default="wholebody_8class.pkl",
                   help="8-class pkl — left/right pointing 추출용")
    p.add_argument("--base_model", type=str, default="models/best_tcn_xsub.pth")
    p.add_argument("--save_path",  type=str, default="models/best_tcn_6class_finetuned_pointing.pth")
    p.add_argument("--split",   type=str, default="xsub")
    p.add_argument("--epochs",  type=int, default=30)
    p.add_argument("--lr",      type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--patience", type=int, default=10)
    return p.parse_args()


def build_augmented_pkl(base_pkl_path, extra_pkl_path, split):
    """base(6class) pkl에 extra(8class)의 pointing 샘플을 추가한 임시 pkl 반환."""
    with open(base_pkl_path, "rb") as f:
        base = pickle.load(f)
    with open(extra_pkl_path, "rb") as f:
        extra = pickle.load(f)

    base_ann  = base["annotations"]
    extra_ann = extra["annotations"]
    n_base    = len(base_ann)

    # 8class에서 pointing 샘플만 추출 (label 5, 6 → 4)
    pointing_ann = []
    for ann in extra_ann:
        if ann["label"] in (5, 6):
            new = copy.copy(ann)
            new["label"] = POINTING_LABEL
            pointing_ann.append(new)

    merged_ann = base_ann + pointing_ann
    n_extra = len(pointing_ann)

    # split 인덱스: base 그대로 + pointing 새 인덱스 추가
    base_split  = base["split"]
    extra_split = extra["split"]

    merged_split = {}
    for split_key in base_split:
        base_indices  = list(base_split[split_key])
        # extra split에서 pointing(5,6)인 것만
        extra_src = split_key.replace(split + "_", "") if split in split_key else split_key
        extra_key = f"{split}_{extra_src}" if not split_key.startswith(split) else split_key
        extra_indices_raw = extra_split.get(extra_key, extra_split.get(split_key, []))

        added = []
        for ei in extra_indices_raw:
            if ei < len(extra_ann) and extra_ann[ei]["label"] in (5, 6):
                # pointing_ann 내 위치를 찾아서 merged 기준 인덱스로 변환
                added.append(n_base + sum(
                    1 for a in extra_ann[:ei] if a["label"] in (5, 6)
                ))
        # 중복 제거
        merged_split[split_key] = base_indices + list(dict.fromkeys(added))

    print(f"  Base pointing samples   : {sum(1 for a in base_ann if a['label']==POINTING_LABEL)}")
    print(f"  Extra pointing added    : {n_extra}")

    for sk, idxs in merged_split.items():
        cnts = Counter(merged_ann[i]["label"] for i in idxs if i < len(merged_ann))
        print(f"  {sk}: {dict(cnts)}")

    return {"annotations": merged_ann, "split": merged_split}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for feats, mask, labels in pbar:
        feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(feats, mask), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
        preds = model(feats, mask).argmax(1)
        correct += (preds == labels).sum().item()
        total   += feats.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for feats, mask, labels in loader:
        feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
        logits = model(feats, mask)
        total_loss += criterion(logits, labels).item() * feats.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += feats.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def print_metrics(preds, labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    print("\n  Per-Class Metrics:")
    for i in range(num_classes):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
        print(f"    {CLASS_NAMES.get(i,'cls'+str(i)):16s}  P={pr:.4f}  R={rc:.4f}  F1={f1:.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_pkl_path  = os.path.join(ROOT_DIR, args.base_pkl)  if not os.path.isabs(args.base_pkl)  else args.base_pkl
    extra_pkl_path = os.path.join(ROOT_DIR, args.extra_pkl) if not os.path.isabs(args.extra_pkl) else args.extra_pkl
    model_path     = os.path.join(ROOT_DIR, args.base_model) if not os.path.isabs(args.base_model) else args.base_model
    save_path      = os.path.join(ROOT_DIR, args.save_path)  if not os.path.isabs(args.save_path)  else args.save_path

    # ── 임시 증강 pkl 생성 ──────────────────────────────────
    print("\n[1] Building augmented pkl with extra pointing data...")
    aug_data = build_augmented_pkl(base_pkl_path, extra_pkl_path, args.split)

    tmp_pkl = "/tmp/aug_pointing_finetune.pkl"
    with open(tmp_pkl, "wb") as f:
        pickle.dump(aug_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── 데이터셋 ────────────────────────────────────────────
    train_ds = NTUSkeletonDataset(tmp_pkl, split=f"{args.split}_train", max_frames=120, augment=True)
    val_ds   = NTUSkeletonDataset(tmp_pkl, split=f"{args.split}_val",   max_frames=120, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── 모델 로드 ────────────────────────────────────────────
    print(f"\n[2] Loading base model: {model_path}")
    ck = torch.load(model_path, map_location=device, weights_only=True)
    model = TCN(
        input_dim   = ck.get("input_dim",    130),
        num_classes = ck.get("num_classes",    6),
        hidden_dims = ck.get("hidden_dims", [64, 128, 128, 256]),
        kernel_size = ck.get("kernel_size",    5),
        dropout     = ck.get("dropout",      0.3),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    print(f"  Loaded epoch {ck.get('epoch','?')}  val_acc={ck.get('val_acc',0):.4f}")

    # ── 학습 설정 ────────────────────────────────────────────
    class_weights = train_ds.get_class_weights().to(device)
    print(f"  Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── 파인튜닝 루프 ────────────────────────────────────────
    print(f"\n[3] Fine-tuning for {args.epochs} epochs (lr={args.lr})")
    # val set이 달라지므로 0부터 시작 (기존 96.94%와 직접 비교 불가)
    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:3d}/{args.epochs}]  "
              f"Train {train_acc:.4f}  │  Val {val_acc:.4f}  │  LR {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                **ck,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "finetuned_from": model_path,
            }, save_path)
            print(f"  ✓ Best saved  val_acc={val_acc:.4f}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # ── 최종 평가 ────────────────────────────────────────────
    print(f"\n[4] Final evaluation on validation set")
    best_ck = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ck["model_state_dict"])
    val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
    print(f"  Val Acc: {val_acc:.4f} ({int(val_acc*len(labels))}/{len(labels)})")
    print_metrics(preds, labels, 6)
    print(f"\n✓ Saved: {save_path}")


if __name__ == "__main__":
    main()
