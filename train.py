"""
train.py
========
Trains two models:
  1. Identity model  – who is the person?   (113-class classifier)
  2. Expression model – what is the expression? (5-class classifier)

Both models share a MobileNetV2 backbone fine-tuned on the THERMAL images only.
RGB images are used solely during pre-training feature alignment (optional toggle).

Dataset folder layout expected after extraction:
  data/
    thermal-face-128x128/   <- 1-TD-A-0.jpg ... 113-TD-E-5.jpg
    RGB-faces-128x128/      <- same naming

Run:
    python train.py --data_dir data --epochs 30 --batch_size 32
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json

from model import DualHeadFaceNet, EXPR_MAP, IMG_SIZE, NUM_EXPRESSIONS

# ─────────────────────────────── CONFIG ──────────────────────────────────────


# ─────────────────────────────── DATASET ─────────────────────────────────────

def parse_filename(fname: str):
    """
    Returns (person_id:int, mode:'A'|'E', index:int) or None if not parseable.
    Example:  '7-TD-A-3.jpg'  ->  (7, 'A', 3)
              '23-TD-E-2.jpg' ->  (23, 'E', 2)
    """
    m = re.match(r"^(\d+)-TD-([AE])-(\d+)\.jpg$", fname, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), m.group(2).upper(), int(m.group(3))


class ThermalFaceDataset(Dataset):
    """
    Returns (image_tensor, person_label, expression_label).

    person_label    : 0-based index (0 … 112)
    expression_label: 0-based index (0 … 4)  — valid ONLY for 'E' images
                      For 'A' (angle) images expression_label = -1 (ignored
                      in the expression loss).
    """

    def __init__(self, folder: str, transform=None, min_person_id=None, max_person_id=None):
        self.transform = transform
        self.samples = []          # list of (path, person_0idx, expr_0idx_or_-1)
        self.person_ids = []       # sorted list of unique integer person ids

        folder = Path(folder)
        all_ids = set()
        raw = []
        for img_path in sorted(folder.glob("*.jpg")):
            parsed = parse_filename(img_path.name)
            if parsed is None:
                continue
            pid, mode, idx = parsed
            if min_person_id is not None and pid < min_person_id:
                continue
            if max_person_id is not None and pid > max_person_id:
                continue
            all_ids.add(pid)
            raw.append((img_path, pid, mode, idx))

        self.person_ids = sorted(all_ids)
        self.pid_to_label = {pid: i for i, pid in enumerate(self.person_ids)}

        for img_path, pid, mode, idx in raw:
            person_lbl = self.pid_to_label[pid]
            expr_lbl = (idx - 1) if mode == "E" else -1   # 0-based; -1 means N/A
            self.samples.append((img_path, person_lbl, expr_lbl))

        print(f"  Loaded {len(self.samples)} images | "
              f"{len(self.person_ids)} persons | folder: {folder.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, person_lbl, expr_lbl = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, person_lbl, expr_lbl


# ─────────────────────────────── TRAINING ────────────────────────────────────

def compute_loss(id_logits, expr_logits, id_labels, expr_labels,
                 id_criterion, expr_criterion, lambda_expr=0.5):
    """
    Combined loss. Expression loss is computed only on images that have
    expression labels (expr_label >= 0).
    """
    id_loss = id_criterion(id_logits, id_labels)

    mask = expr_labels >= 0
    if mask.sum() > 0:
        expr_loss = expr_criterion(expr_logits[mask], expr_labels[mask])
    else:
        expr_loss = torch.tensor(0.0, device=id_logits.device)

    return id_loss + lambda_expr * expr_loss, id_loss.item(), expr_loss.item()


def accuracy(logits, labels, mask=None):
    preds = logits.argmax(dim=1)
    if mask is not None:
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == labels[mask]).float().mean().item()
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, optimizer, scheduler, device, lambda_expr=0.5):
    model.train()
    total_loss = id_acc_sum = expr_acc_sum = n = 0

    for imgs, id_lbl, expr_lbl in loader:
        imgs      = imgs.to(device)
        id_lbl    = id_lbl.to(device)
        expr_lbl  = expr_lbl.to(device)

        id_logits, expr_logits = model(imgs)
        loss, _, _ = compute_loss(
            id_logits, expr_logits, id_lbl, expr_lbl,
            nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), lambda_expr=lambda_expr
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mask = expr_lbl >= 0
        total_loss  += loss.item()
        id_acc_sum  += accuracy(id_logits, id_lbl)
        expr_acc_sum += accuracy(expr_logits, expr_lbl, mask)
        n += 1

    scheduler.step()
    return total_loss / n, id_acc_sum / n, expr_acc_sum / n


@torch.no_grad()
def evaluate(model, loader, device, lambda_expr=0.5):
    model.eval()
    total_loss = id_acc_sum = expr_acc_sum = n = 0

    all_id_preds, all_id_true = [], []
    all_ex_preds, all_ex_true = [], []

    for imgs, id_lbl, expr_lbl in loader:
        imgs     = imgs.to(device)
        id_lbl   = id_lbl.to(device)
        expr_lbl = expr_lbl.to(device)

        id_logits, expr_logits = model(imgs)
        loss, _, _ = compute_loss(
            id_logits, expr_logits, id_lbl, expr_lbl,
            nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), lambda_expr=lambda_expr
        )

        mask = expr_lbl >= 0
        total_loss  += loss.item()
        id_acc_sum  += accuracy(id_logits, id_lbl)
        expr_acc_sum += accuracy(expr_logits, expr_lbl, mask)
        n += 1

        all_id_preds.extend(id_logits.argmax(1).cpu().numpy())
        all_id_true.extend(id_lbl.cpu().numpy())
        if mask.sum() > 0:
            all_ex_preds.extend(expr_logits[mask].argmax(1).cpu().numpy())
            all_ex_true.extend(expr_lbl[mask].cpu().numpy())

    return (total_loss / n, id_acc_sum / n, expr_acc_sum / n,
            all_id_preds, all_id_true, all_ex_preds, all_ex_true)


def plot_history(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["train_id_acc"], label="Train")
    axes[1].plot(history["val_id_acc"],   label="Val")
    axes[1].set_title("Identity Accuracy"); axes[1].legend()

    axes[2].plot(history["train_expr_acc"], label="Train")
    axes[2].plot(history["val_expr_acc"],   label="Val")
    axes[2].set_title("Expression Accuracy"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Training curves saved → {save_path}")


# ─────────────────────────────── MAIN ────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Thermal Face Recognition – Training")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Transforms ──────────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ── Dataset ──────────────────────────────────────────────────────────────
    thermal_dir = os.path.join(args.data_dir, "thermal-face-128x128")
    full_ds = ThermalFaceDataset(thermal_dir, transform=train_tf,
                                 min_person_id=args.min_person_id,
                                 max_person_id=args.max_person_id)

    lambda_expr = 0.0 if args.identity_only else 0.5

    n_total = len(full_ds)
    n_val   = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # Apply val transform to val split
    val_ds.dataset.transform = val_tf   # shared dataset object — set at eval time

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # Save label map so inference.py can use it
    label_map = {"pid_to_label": full_ds.pid_to_label,
                 "person_ids":   full_ds.person_ids,
                 "expr_map":     EXPR_MAP}
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # ── Model ────────────────────────────────────────────────────────────────
    model = DualHeadFaceNet(
        num_persons=len(full_ds.person_ids),
        num_expressions=NUM_EXPRESSIONS,
        dropout=args.dropout,
        pretrained=True,
    ).to(device)

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    # Freeze backbone for first warm-up epochs, then unfreeze
    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {k: [] for k in
               ["train_loss","val_loss","train_id_acc","val_id_acc",
                "train_expr_acc","val_expr_acc"]}

    best_val_id_acc = 0.0
    unfreeze_done = False

    print(f"  Train samples : {n_train}")
    print(f"  Val   samples : {n_val}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}\n")

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after warm-up
        if epoch == args.unfreeze_epoch and not unfreeze_done:
            print(f"\n  [Epoch {epoch}] Unfreezing backbone …")
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(),
                                    lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch)
            unfreeze_done = True

        t0 = time.time()
        tr_loss, tr_id, tr_ex = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, lambda_expr=lambda_expr)
        vl_loss, vl_id, vl_ex, \
        id_preds, id_true, ex_preds, ex_true = evaluate(model, val_loader, device, lambda_expr=lambda_expr)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_id_acc"].append(tr_id)
        history["val_id_acc"].append(vl_id)
        history["train_expr_acc"].append(tr_ex)
        history["val_expr_acc"].append(vl_ex)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Loss {tr_loss:.3f}/{vl_loss:.3f} | "
              f"ID-Acc {tr_id:.3f}/{vl_id:.3f} | "
              f"Expr-Acc {tr_ex:.3f}/{vl_ex:.3f} | "
              f"{elapsed:.1f}s")

        if vl_id > best_val_id_acc:
            best_val_id_acc = vl_id
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "num_persons": len(full_ds.person_ids),
                "num_expressions": NUM_EXPRESSIONS,
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  ✓ Saved best model (val ID acc={vl_id:.4f})")

    # ── Final report ─────────────────────────────────────────────────────────
    if not args.identity_only:
        print("\n── Expression Classification Report (val set) ──")
        expr_names = [EXPR_MAP[str(i+1)] for i in range(NUM_EXPRESSIONS)]
        if ex_true:
            print(classification_report(ex_true, ex_preds, target_names=expr_names))

    plot_history(history, os.path.join(args.output_dir, "training_curves.png"))
    print(f"\n  Best Val Identity Accuracy : {best_val_id_acc:.4f}")
    print(f"  Model saved to            : {args.output_dir}/best_model.pth")
    print("  Done.\n")


# ─────────────────────────────── CLI ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Thermal Face Recognition")
    parser.add_argument("--data_dir",      default="data",       help="Root data directory")
    parser.add_argument("--output_dir",    default="checkpoints",help="Where to save models/plots")
    parser.add_argument("--epochs",        type=int, default=40)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--val_split",     type=float, default=0.15)
    parser.add_argument("--dropout",       type=float, default=0.4)
    parser.add_argument("--unfreeze_epoch",type=int, default=10,
                        help="Epoch at which to unfreeze the backbone")
    parser.add_argument("--workers",       type=int, default=4)
    parser.add_argument("--min_person_id", type=int, default=None,
                        help="Only train on person IDs >= this value")
    parser.add_argument("--max_person_id", type=int, default=None,
                        help="Only train on person IDs <= this value")
    parser.add_argument("--identity_only", action="store_true",
                        help="Skip the expression loss/report and train the identity head only")
    args = parser.parse_args()
    main(args)
