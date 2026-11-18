"""
train.py
========
Trains the identity model – who is the person? (num_persons-class classifier).

Expression ("what is the expression?") training is disabled for now (not a
current focus -- revisit next month); search for "expression disabled" in
this file for the commented-out pieces. The model still has an
expression_head (see model.py) so existing checkpoints keep loading, it's
just not part of the loss/reporting anymore.

The backbone is a MobileNetV2 fine-tuned on the THERMAL images only.
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
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
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

    def __init__(self, folder: str, transform=None, exclude=None):
        self.transform = transform
        self.samples = []          # list of (path, person_0idx, expr_0idx_or_-1)
        self.frame_index = []      # per sample: the frame number from the filename (capture order)
        self.frame_mode = []       # per sample: 'A' (angle/video frame) or 'E' (expression still)
        self.person_ids = []       # sorted list of unique integer person ids

        folder = Path(folder)
        all_ids = set()
        raw = []
        for img_path in sorted(folder.glob("*.jpg")):
            parsed = parse_filename(img_path.name)
            if parsed is None or (exclude and img_path.name in exclude):
                continue
            pid, mode, idx = parsed
            all_ids.add(pid)
            raw.append((img_path, pid, mode, idx))

        self.person_ids = sorted(all_ids)
        self.pid_to_label = {pid: i for i, pid in enumerate(self.person_ids)}

        for img_path, pid, mode, idx in raw:
            person_lbl = self.pid_to_label[pid]
            expr_lbl = (idx - 1) if mode == "E" else -1   # 0-based; -1 means N/A
            self.samples.append((img_path, person_lbl, expr_lbl))
            self.frame_index.append(idx)
            self.frame_mode.append(mode)

        print(f"  Loaded {len(self.samples)} images | "
              f"{len(self.person_ids)} persons | folder: {folder.name}"
              + (f" | {len(exclude)} names on the exclude list" if exclude else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, person_lbl, expr_lbl = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, person_lbl, expr_lbl


class RandomEyeBand:
    """With probability p, draw a dark horizontal band across the eye region (a stand-in for
    sunglasses, which are opaque in thermal). In the training data only one person (Asher)
    wears sunglasses, so without this the model learns 'dark band across the eyes = Asher'
    and calls anyone wearing sunglasses Asher (seen live on 2026-11-02). Applied to everyone
    in training so the band stops being a clue to identity."""

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        from PIL import ImageDraw
        w, h = img.size
        top = h * random.uniform(0.30, 0.58)   # eyes sit anywhere from ~40% to ~65% down these crops
        bottom = top + h * random.uniform(0.10, 0.18)
        left, right = w * random.uniform(0.12, 0.25), w * random.uniform(0.75, 0.88)
        shade = random.randint(20, 70)
        img = img.copy()
        ImageDraw.Draw(img).rectangle([left, top, right, bottom], fill=(shade, shade, shade))
        return img


def load_exclusions(path):
    """Image file names (one per line, '#' starts a comment) to leave out of training,
    e.g. blank-background crops that were saved under a person's id. The files stay on
    disk; this only skips them."""
    if not path or not os.path.exists(path):
        return set()
    with open(path) as f:
        return {ln.split("#")[0].strip() for ln in f if ln.split("#")[0].strip()}


def load_sessions(path):
    """Capture sessions logged by collect_data.py (one JSON object per line), grouped by
    person id. Returns {} if there is no log."""
    sessions = {}
    if not path or not os.path.exists(path):
        return sessions
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                sessions.setdefault(int(entry["person_id"]), []).append(entry)
    return sessions


def _session_of(sessions_for_person, mode, frame):
    """(started, session_id) of the logged session that contains this frame; frames that
    no logged session covers (everything recorded before the log existed, and expression
    stills) all count as one 'legacy' session, treated as the oldest."""
    if mode == "A":
        for s in sessions_for_person:
            if "frame_list" in s and frame in s["frame_list"]:
                return s["started"], s["session_id"]
            if "first_frame" in s and s["first_frame"] <= frame <= s["last_frame"]:
                return s["started"], s["session_id"]
    return "", "legacy"


def split_indices(ds, val_frac, mode, seed=42, sessions=None):
    """Returns (train_idx, val_idx) into `ds.samples`.

    'session': for each person, the most recent logged capture session (see collect_data.py
    and load_sessions) is held out whole. That tests the model on a recording it has never
    seen frames from, which is what the live camera is. A person with only one session
    (e.g. only legacy data) falls back to 'block' for that person, with a printed warning.

    'block': for each person, the LAST val_frac of their frames (by capture order, i.e. the
    number in the file name) are held out. Frames are captured ~4 per second, so a random
    split puts near-identical neighbours of every validation frame in the training set and
    the validation accuracy says nothing (it hit 100% at epoch 4). Holding out a whole
    stretch removes that leak. It still can't tell recording sessions apart, so it is a
    better estimate, not a guarantee.
    'random': the old behaviour."""
    n = len(ds.samples)
    if mode == "random":
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
        n_val = int(n * val_frac)
        return perm[n_val:], perm[:n_val]
    by_person = {}
    for i, (_, person_lbl, _) in enumerate(ds.samples):
        by_person.setdefault(person_lbl, []).append(i)
    train_idx, val_idx = [], []
    for label, idxs in by_person.items():
        idxs.sort(key=lambda i: ds.frame_index[i])
        if mode == "session":
            pid = ds.person_ids[label]
            mine = (sessions or {}).get(pid, [])
            groups = {}
            for i in idxs:
                started, sid = _session_of(mine, ds.frame_mode[i], ds.frame_index[i])
                groups.setdefault((started, sid), []).append(i)
            if len(groups) >= 2:
                latest = max(groups)                      # newest 'started' (legacy sorts first)
                for key, members in groups.items():
                    (val_idx if key == latest else train_idx).extend(members)
                continue
            print(f"  WARNING: person {pid} has only one capture session; "
                  f"falling back to a block split for them.")
        n_val = max(1, int(round(len(idxs) * val_frac)))
        train_idx += idxs[:-n_val]
        val_idx += idxs[-n_val:]
    return train_idx, val_idx


# ─────────────────────────────── TRAINING ────────────────────────────────────

def compute_loss(id_logits, expr_logits, id_labels, expr_labels,
                 id_criterion, expr_criterion, lambda_expr=0.5):
    """
    Identity loss only for now -- expression is not being trained/used
    (commented out below; revisit next month). expression_head still exists
    on the model (see model.py) so existing checkpoints keep loading, it's
    just not part of the loss anymore.
    """
    id_loss = id_criterion(id_logits.float(), id_labels)   # float: the class weights are fp32

    # -- Expression loss disabled --------------------------------------
    # mask = expr_labels >= 0
    # if mask.sum() > 0:
    #     expr_loss = expr_criterion(expr_logits[mask], expr_labels[mask])
    # else:
    #     expr_loss = torch.tensor(0.0, device=id_logits.device)
    expr_loss = torch.tensor(0.0, device=id_logits.device)

    # return id_loss + lambda_expr * expr_loss, id_loss.item(), expr_loss.item()
    return id_loss, id_loss.item(), expr_loss.item()


def accuracy(logits, labels, mask=None):
    preds = logits.argmax(dim=1)
    if mask is not None:
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == labels[mask]).float().mean().item()
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, id_criterion=None):
    id_criterion = id_criterion or nn.CrossEntropyLoss()
    model.train()
    total_loss = id_acc_sum = expr_acc_sum = n = 0
    use_amp = device.type == "cuda"

    for imgs, id_lbl, expr_lbl in loader:
        imgs      = imgs.to(device, non_blocking=True)
        id_lbl    = id_lbl.to(device, non_blocking=True)
        expr_lbl  = expr_lbl.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # Mixed precision on GPU: this backbone's convs/linears run in fp16/bf16 at
        # roughly 2x the throughput on tensor-core hardware, with GradScaler guarding
        # against fp16 underflow in the backward pass. No-ops cleanly on CPU (scaler
        # is constructed with enabled=False there) so this isn't a GPU-only code path.
        with torch.autocast(device_type=device.type, enabled=use_amp):
            id_logits, expr_logits = model(imgs)
            loss, _, _ = compute_loss(
                id_logits, expr_logits, id_lbl, expr_lbl,
                id_criterion, nn.CrossEntropyLoss()
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # mask = expr_lbl >= 0
        total_loss  += loss.item()
        id_acc_sum  += accuracy(id_logits, id_lbl)
        # expr_acc_sum += accuracy(expr_logits, expr_lbl, mask)  # expression disabled
        n += 1

    scheduler.step()
    return total_loss / n, id_acc_sum / n, expr_acc_sum / n


@torch.no_grad()
def evaluate(model, loader, device, id_criterion=None):
    id_criterion = id_criterion or nn.CrossEntropyLoss()
    model.eval()
    total_loss = id_acc_sum = expr_acc_sum = n = 0
    use_amp = device.type == "cuda"

    all_id_preds, all_id_true = [], []
    all_ex_preds, all_ex_true = [], []

    for imgs, id_lbl, expr_lbl in loader:
        imgs     = imgs.to(device, non_blocking=True)
        id_lbl   = id_lbl.to(device, non_blocking=True)
        expr_lbl = expr_lbl.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            id_logits, expr_logits = model(imgs)
        loss, _, _ = compute_loss(
            id_logits, expr_logits, id_lbl, expr_lbl,
            id_criterion, nn.CrossEntropyLoss()
        )

        # mask = expr_lbl >= 0
        total_loss  += loss.item()
        id_acc_sum  += accuracy(id_logits, id_lbl)
        # expr_acc_sum += accuracy(expr_logits, expr_lbl, mask)  # expression disabled
        n += 1

        all_id_preds.extend(id_logits.argmax(1).cpu().numpy())
        all_id_true.extend(id_lbl.cpu().numpy())
        # if mask.sum() > 0:
        #     all_ex_preds.extend(expr_logits[mask].argmax(1).cpu().numpy())
        #     all_ex_true.extend(expr_lbl[mask].cpu().numpy())

    return (total_loss / n, id_acc_sum / n, expr_acc_sum / n,
            all_id_preds, all_id_true, all_ex_preds, all_ex_true)


def plot_history(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["train_id_acc"], label="Train")
    axes[1].plot(history["val_id_acc"],   label="Val")
    axes[1].set_title("Identity Accuracy"); axes[1].legend()

    # -- Expression accuracy plot disabled --------------------------------
    # axes[2].plot(history["train_expr_acc"], label="Train")
    # axes[2].plot(history["val_expr_acc"],   label="Val")
    # axes[2].set_title("Expression Accuracy"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Training curves saved → {save_path}")


# ─────────────────────────────── MAIN ────────────────────────────────────────

def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Autotunes cuDNN's conv algorithm selection for this model's fixed 128x128
        # input shape -- pays for itself after the first few batches since the shape
        # never changes here, and costs nothing to leave on.
        torch.backends.cudnn.benchmark = True
    print(f"\n{'='*60}")
    print(f"  Thermal Face Recognition – Training")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Transforms ──────────────────────────────────────────────────────────
    # Scale / shift augmentation matters here: live crops come out zoomed in or out, off
    # centre and tilted compared with the saved training crops, and a model trained without
    # it (as this script did until 2026-11-02, see the dataset section below) breaks on that.
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        RandomEyeBand(p=args.eye_band_prob),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.12, 0.12), scale=(0.75, 1.3)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
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
    exclude = load_exclusions(args.exclude_file)
    # Two dataset objects, one per transform. (Before 2026-11-02 this was one shared object
    # with `val_ds.dataset.transform = val_tf`, which also switched the TRAINING subset to
    # the no-augmentation transform, so training never used augmentation.)
    full_ds = ThermalFaceDataset(thermal_dir, transform=train_tf, exclude=exclude)
    val_full_ds = ThermalFaceDataset(thermal_dir, transform=val_tf, exclude=exclude)

    train_idx, val_idx = split_indices(full_ds, args.val_split, args.split_mode,
                                       sessions=load_sessions(args.sessions_file))
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(val_full_ds, val_idx)
    n_train, n_val = len(train_ds), len(val_ds)

    # Class weights: inverse frequency in the TRAIN split (people have 61-231 images), so a
    # person with few images isn't drowned out by one with many.
    train_labels = torch.tensor([full_ds.samples[i][1] for i in train_idx])
    class_counts = torch.bincount(train_labels, minlength=len(full_ds.person_ids)).float()
    class_weights = (class_counts.sum() / (len(class_counts) * class_counts.clamp(min=1))
                     if args.class_weights else None)
    print(f"  Train images per person : {class_counts.int().tolist()}")
    print(f"  Held-out per person     : "
          f"{torch.bincount(torch.tensor([val_full_ds.samples[i][1] for i in val_idx]), minlength=len(class_counts)).tolist()}")
    id_criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device),
        label_smoothing=args.label_smoothing)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=(args.workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=(args.workers > 0))

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
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    history = {k: [] for k in
               ["train_loss","val_loss","train_id_acc","val_id_acc"]}
               # "train_expr_acc","val_expr_acc"  -- expression disabled

    best_val_id_acc, best_val_loss = 0.0, float("inf")
    best_preds, best_true = [], []
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
            model, train_loader, optimizer, scheduler, device, scaler, id_criterion)
        vl_loss, vl_id, vl_ex, \
        id_preds, id_true, ex_preds, ex_true = evaluate(model, val_loader, device, id_criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_id_acc"].append(tr_id)
        history["val_id_acc"].append(vl_id)
        # history["train_expr_acc"].append(tr_ex)  -- expression disabled
        # history["val_expr_acc"].append(vl_ex)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Loss {tr_loss:.3f}/{vl_loss:.3f} | "
              f"ID-Acc {tr_id:.3f}/{vl_id:.3f} | "
              # f"Expr-Acc {tr_ex:.3f}/{vl_ex:.3f} | "  -- expression disabled
              f"{elapsed:.1f}s")

        # Best = highest held-out accuracy, ties broken by lower held-out loss (so a later,
        # better-trained epoch replaces an earlier one that merely reached the same accuracy).
        if (vl_id, -vl_loss) > (best_val_id_acc, -best_val_loss):
            best_val_id_acc, best_val_loss = vl_id, vl_loss
            best_preds, best_true = list(id_preds), list(id_true)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "num_persons": len(full_ds.person_ids),
                "num_expressions": NUM_EXPRESSIONS,
                "val_id_acc": vl_id,
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  ✓ Saved best model (held-out ID acc={vl_id:.4f}, loss={vl_loss:.3f})")

    # ── Final report ─────────────────────────────────────────────────────────
    # -- Expression report disabled ---------------------------------------
    # print("\n── Expression Classification Report (val set) ──")
    # expr_names = [EXPR_MAP[str(i+1)] for i in range(NUM_EXPRESSIONS)]
    # if ex_true:
    #     print(classification_report(ex_true, ex_preds, target_names=expr_names))

    if best_true:
        print("\n── Held-out identity report (best epoch) ──")
        print(classification_report(best_true, best_preds, labels=list(range(len(full_ds.person_ids))),
                                    target_names=[str(p) for p in full_ds.person_ids], zero_division=0))
        print("  Confusion matrix (rows = true, cols = predicted):")
        print(confusion_matrix(best_true, best_preds, labels=list(range(len(full_ds.person_ids)))))

    plot_history(history, os.path.join(args.output_dir, "training_curves.png"))
    print(f"\n  Best Held-out Identity Accuracy : {best_val_id_acc:.4f}")
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
    parser.add_argument("--split_mode",    choices=["block", "random", "session"], default="block",
                        help="block = hold out the last frames of each person (default, no "
                             "near-duplicate leak); session = hold out each person's newest "
                             "logged capture session (needs data/sessions.jsonl from "
                             "collect_data.py, the most honest check); random = the old "
                             "random-frame split")
    parser.add_argument("--sessions_file", default=os.path.join("data", "sessions.jsonl"),
                        help="Capture-session log written by collect_data.py")
    parser.add_argument("--exclude_file",  default=os.path.join("data", "excluded_images.txt"),
                        help="Text file of image names to skip (not deleted from disk)")
    parser.add_argument("--no_class_weights", dest="class_weights", action="store_false",
                        help="Turn off inverse-frequency class weighting of the loss")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the random number generators (default: unseeded, so every "
                             "run differs; live results varied a lot between unseeded runs)")
    parser.add_argument("--eye_band_prob", type=float, default=0.3,
                        help="Chance of drawing a fake sunglasses band on a training image "
                             "(0 = off); see RandomEyeBand")
    parser.add_argument("--dropout",       type=float, default=0.4)
    parser.add_argument("--unfreeze_epoch",type=int, default=10,
                        help="Epoch at which to unfreeze the backbone")
    parser.add_argument("--workers",       type=int, default=(0 if os.name == "nt" else 4),
                        help="DataLoader worker processes. Defaults to 0 on Windows: "
                             "Windows' spawn-based multiprocessing deadlocked here with "
                             "workers>0 (confirmed directly -- GPU/CPU both went idle "
                             "mid-run); 0 loads data on the main process instead, which "
                             "is slower per-batch but has never hung. Override if your "
                             "setup doesn't hit that.")
    args = parser.parse_args()
    main(args)
