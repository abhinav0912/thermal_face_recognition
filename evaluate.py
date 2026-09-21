"""
evaluate.py
===========
Runs a full evaluation of the trained model on the test split (or any folder)
and produces a per-person classification report + confusion matrix PNG.

Expression evaluation/reporting is commented out for now (not a current
focus -- revisit next month); search for "expression disabled" in this file.

Usage:
    python evaluate.py --checkpoint_dir checkpoints --data_dir data
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

# Re-import dataset & model (import from shared location in real project)
from train import ThermalFaceDataset, DualHeadFaceNet, NUM_EXPRESSIONS
# EXPR_MAP unused -- expression disabled


TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def plot_confusion_matrix(cm, labels, title, save_path, max_labels=30):
    """Plot and save confusion matrix; truncates to max_labels for readability."""
    if len(labels) > max_labels:
        labels = labels[:max_labels]
        cm = cm[:max_labels, :max_labels]

    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=90)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load label map ───────────────────────────────────────────────────────
    lmap_path = os.path.join(args.checkpoint_dir, "label_map.json")
    with open(lmap_path) as f:
        lmap = json.load(f)
    label_to_pid = {v: int(k) for k, v in lmap["pid_to_label"].items()}
    num_persons  = len(label_to_pid)

    # ── Dataset ──────────────────────────────────────────────────────────────
    thermal_dir = os.path.join(args.data_dir, "thermal-face-128x128")
    dataset = ThermalFaceDataset(thermal_dir, transform=TRANSFORM)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # ── Model ────────────────────────────────────────────────────────────────
    ckpt = torch.load(
        os.path.join(args.checkpoint_dir, "best_model.pth"),
        map_location=device
    )
    model = DualHeadFaceNet(
        num_persons=ckpt.get("num_persons", num_persons),
        num_expressions=NUM_EXPRESSIONS,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Run inference ────────────────────────────────────────────────────────
    all_id_preds, all_id_true = [], []
    # all_ex_preds, all_ex_true = [], []  -- expression disabled

    with torch.no_grad():
        for imgs, id_lbl, expr_lbl in loader:
            imgs     = imgs.to(device)
            id_lbl   = id_lbl.to(device)
            expr_lbl = expr_lbl.to(device)

            id_logits, expr_logits = model(imgs)  # expr_logits unused -- expression disabled

            all_id_preds.extend(id_logits.argmax(1).cpu().numpy())
            all_id_true.extend(id_lbl.cpu().numpy())

            # mask = expr_lbl >= 0
            # if mask.sum() > 0:
            #     all_ex_preds.extend(expr_logits[mask].argmax(1).cpu().numpy())
            #     all_ex_true.extend(expr_lbl[mask].cpu().numpy())

    # ── Reports ──────────────────────────────────────────────────────────────
    print("\n── Identity Classification Report ──")
    person_label_names = [f"P{label_to_pid[i]}" for i in range(num_persons)]
    print(classification_report(
        all_id_true, all_id_preds, target_names=person_label_names, zero_division=0
    ))

    # -- Expression report disabled ---------------------------------------
    # print("\n── Expression Classification Report ──")
    # expr_names = [EXPR_MAP[str(i+1)] for i in range(NUM_EXPRESSIONS)]
    # print(classification_report(
    #     all_ex_true, all_ex_preds, target_names=expr_names, zero_division=0
    # ))

    # ── Confusion matrices ────────────────────────────────────────────────────
    cm_id = confusion_matrix(all_id_true, all_id_preds)
    plot_confusion_matrix(
        cm_id, person_label_names,
        "Identity Confusion Matrix (first 30 persons)",
        os.path.join(args.output_dir, "cm_identity.png")
    )

    # cm_ex = confusion_matrix(all_ex_true, all_ex_preds)
    # plot_confusion_matrix(
    #     cm_ex, expr_names,
    #     "Expression Confusion Matrix",
    #     os.path.join(args.output_dir, "cm_expression.png")
    # )

    # Summary
    id_acc = np.mean(np.array(all_id_preds) == np.array(all_id_true))
    # ex_acc = np.mean(np.array(all_ex_preds) == np.array(all_ex_true))
    print(f"\n  Overall Identity   Accuracy : {id_acc*100:.2f}%")
    # print(f"  Overall Expression Accuracy : {ex_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_dir",       default="data")
    parser.add_argument("--output_dir",     default="checkpoints")
    main(parser.parse_args())
