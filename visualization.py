"""
visualization.py
================
Advanced visualization suite for thermal face recognition.

Generates:
  • Confusion matrices (identity & expression)
  • Per-person accuracy heatmap
  • Confidence distribution plots
  • Loss curves with annotations
  • Comparison plots (actual vs predicted)

Usage:
    python visualization.py --checkpoint_dir checkpoints --data_dir data
"""

import argparse
import json
import os
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
import seaborn as sns

EXPR_MAP = {
    "1": "Angry",
    "2": "Happy",
    "3": "Neutral",
    "4": "Sad",
    "5": "Surprised",
}

def parse_filename(fname: str):
    m = re.match(r"^(\d+)-TD-([AE])-(\d+)\.jpg$", fname, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), m.group(2).upper(), int(m.group(3))


class ThermalFaceDataset:
    def __init__(self, folder: str, transform=None):
        self.transform = transform
        self.samples = []
        self.person_ids = []
        
        folder = Path(folder)
        all_ids = set()
        raw = []
        for img_path in sorted(folder.glob("*.jpg")):
            parsed = parse_filename(img_path.name)
            if parsed is None:
                continue
            pid, mode, idx = parsed
            all_ids.add(pid)
            raw.append((img_path, pid, mode, idx))
        
        self.person_ids = sorted(all_ids)
        self.pid_to_label = {pid: i for i, pid in enumerate(self.person_ids)}
        
        for img_path, pid, mode, idx in raw:
            person_lbl = self.pid_to_label[pid]
            expr_lbl = (idx - 1) if mode == "E" else -1
            self.samples.append((img_path, person_lbl, expr_lbl))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, person_lbl, expr_lbl = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, person_lbl, expr_lbl


class DualHeadFaceNet(nn.Module):
    def __init__(self, num_persons: int, num_expressions: int, dropout: float = 0.4):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.identity_head = nn.Linear(512, num_persons)
        self.expression_head = nn.Linear(512, num_expressions)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        feat = self.shared_fc(x)
        return self.identity_head(feat), self.expression_head(feat)


class Visualizer:
    def __init__(self, checkpoint_dir: str, data_dir: str, output_dir: str = "visualizations"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model = DualHeadFaceNet(
            num_persons=ckpt.get("num_persons", 113),
            num_expressions=ckpt.get("num_expressions", 5),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        
        # Dataset
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        
        thermal_dir = os.path.join(data_dir, "thermal-face-128x128")
        self.dataset = ThermalFaceDataset(thermal_dir, transform=transform)
        self.loader = DataLoader(self.dataset, batch_size=32, shuffle=False, num_workers=2)

    def get_predictions(self):
        """Run inference and collect all predictions."""
        all_id_preds = []
        all_id_true = []
        all_id_conf = []
        all_ex_preds = []
        all_ex_true = []
        all_ex_conf = []
        
        with torch.no_grad():
            for imgs, id_lbl, expr_lbl in self.loader:
                imgs = imgs.to(self.device)
                id_lbl = id_lbl.to(self.device)
                expr_lbl = expr_lbl.to(self.device)
                
                id_logits, expr_logits = self.model(imgs)
                
                id_probs = F.softmax(id_logits, dim=1)
                id_preds = id_logits.argmax(1)
                id_confs = id_probs.max(1)[0]
                
                expr_probs = F.softmax(expr_logits, dim=1)
                expr_preds = expr_logits.argmax(1)
                expr_confs = expr_probs.max(1)[0]
                
                all_id_preds.extend(id_preds.cpu().numpy())
                all_id_true.extend(id_lbl.cpu().numpy())
                all_id_conf.extend(id_confs.cpu().numpy())
                
                mask = expr_lbl >= 0
                if mask.sum() > 0:
                    all_ex_preds.extend(expr_preds[mask].cpu().numpy())
                    all_ex_true.extend(expr_lbl[mask].cpu().numpy())
                    all_ex_conf.extend(expr_confs[mask].cpu().numpy())
        
        return {
            "id_preds": np.array(all_id_preds),
            "id_true": np.array(all_id_true),
            "id_conf": np.array(all_id_conf),
            "ex_preds": np.array(all_ex_preds),
            "ex_true": np.array(all_ex_true),
            "ex_conf": np.array(all_ex_conf),
        }

    def plot_identity_confusion_matrix(self, data, top_n=30):
        """Plot identity confusion matrix (top N persons)."""
        cm = confusion_matrix(data["id_true"], data["id_preds"])
        
        # Show top N
        cm = cm[:top_n, :top_n]
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm, cmap="Blues", ax=ax, cbar_kws={"label": "Count"},
                    xticklabels=[str(self.dataset.person_ids[i]) for i in range(top_n)],
                    yticklabels=[str(self.dataset.person_ids[i]) for i in range(top_n)])
        ax.set_xlabel("Predicted Person ID", fontsize=12)
        ax.set_ylabel("True Person ID", fontsize=12)
        ax.set_title("Identity Confusion Matrix (Top 30 Persons)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, "confusion_identity.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")

    def plot_expression_confusion_matrix(self, data):
        """Plot expression confusion matrix."""
        cm = confusion_matrix(data["ex_true"], data["ex_preds"])
        
        expr_names = [EXPR_MAP[str(i+1)] for i in range(5)]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, cmap="RdYlGn", ax=ax, annot=True, fmt="d",
                    xticklabels=expr_names, yticklabels=expr_names,
                    cbar_kws={"label": "Count"})
        ax.set_xlabel("Predicted Expression", fontsize=11)
        ax.set_ylabel("True Expression", fontsize=11)
        ax.set_title("Expression Confusion Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, "confusion_expression.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")

    def plot_confidence_distribution(self, data):
        """Plot confidence score distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Identity confidence
        correct_id = data["id_conf"][data["id_preds"] == data["id_true"]]
        wrong_id = data["id_conf"][data["id_preds"] != data["id_true"]]
        
        axes[0].hist(correct_id, bins=30, alpha=0.6, label="Correct", color="green")
        axes[0].hist(wrong_id, bins=30, alpha=0.6, label="Wrong", color="red")
        axes[0].set_xlabel("Confidence Score", fontsize=11)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].set_title("Identity Prediction Confidence", fontsize=12, fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Expression confidence
        correct_ex = data["ex_conf"][data["ex_preds"] == data["ex_true"]]
        wrong_ex = data["ex_conf"][data["ex_preds"] != data["ex_true"]]
        
        axes[1].hist(correct_ex, bins=30, alpha=0.6, label="Correct", color="green")
        axes[1].hist(wrong_ex, bins=30, alpha=0.6, label="Wrong", color="red")
        axes[1].set_xlabel("Confidence Score", fontsize=11)
        axes[1].set_ylabel("Count", fontsize=11)
        axes[1].set_title("Expression Prediction Confidence", fontsize=12, fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "confidence_distributions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")

    def plot_per_person_accuracy(self, data):
        """Plot accuracy per person (bar chart, top performers)."""
        person_accs = {}
        for label in range(len(self.dataset.person_ids)):
            mask = data["id_true"] == label
            if mask.sum() == 0:
                continue
            acc = (data["id_preds"][mask] == data["id_true"][mask]).mean()
            pid = self.dataset.person_ids[label]
            person_accs[pid] = acc
        
        # Sort and plot top 20 and bottom 20
        sorted_items = sorted(person_accs.items(), key=lambda x: x[1])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Worst performers
        worst_pids = [str(p) for p, _ in sorted_items[:15]]
        worst_accs = [a * 100 for _, a in sorted_items[:15]]
        axes[0].barh(worst_pids, worst_accs, color="tomato")
        axes[0].set_xlabel("Accuracy (%)", fontsize=11)
        axes[0].set_title("Hardest Identities (Lowest Accuracy)", fontsize=12, fontweight="bold")
        axes[0].set_xlim(0, 100)
        for i, v in enumerate(worst_accs):
            axes[0].text(v + 2, i, f"{v:.1f}%", va="center", fontsize=9)
        
        # Best performers
        best_pids = [str(p) for p, _ in sorted_items[-15:]]
        best_accs = [a * 100 for _, a in sorted_items[-15:]]
        axes[1].barh(best_pids, best_accs, color="lightgreen")
        axes[1].set_xlabel("Accuracy (%)", fontsize=11)
        axes[1].set_title("Easiest Identities (Highest Accuracy)", fontsize=12, fontweight="bold")
        axes[1].set_xlim(0, 100)
        for i, v in enumerate(best_accs):
            axes[1].text(v + 2, i, f"{v:.1f}%", va="center", fontsize=9)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "per_person_accuracy.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")

    def plot_accuracy_metrics(self, data):
        """Plot overall accuracy metrics."""
        id_acc = (data["id_preds"] == data["id_true"]).mean() * 100
        ex_acc = (data["ex_preds"] == data["ex_true"]).mean() * 100
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        metrics = ["Identity\nAccuracy", "Expression\nAccuracy"]
        accuracies = [id_acc, ex_acc]
        colors = ["#2ecc71", "#3498db"]
        
        bars = ax.bar(metrics, accuracies, color=colors, width=0.6, edgecolor="black", linewidth=2)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2f}%',
                   ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "accuracy_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")

    def run_all(self):
        print("\n" + "="*70)
        print("  GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        
        print("  Running inference...", end=" ", flush=True)
        data = self.get_predictions()
        print("Done.")
        
        print("\n  Generating plots:")
        self.plot_accuracy_metrics(data)
        self.plot_identity_confusion_matrix(data)
        self.plot_expression_confusion_matrix(data)
        self.plot_confidence_distribution(data)
        self.plot_per_person_accuracy(data)
        
        print("\n" + "="*70)
        print(f"  All visualizations saved to: {os.path.abspath(self.output_dir)}")
        print("="*70 + "\n")


def main(args):
    viz = Visualizer(args.checkpoint_dir, args.data_dir, args.output_dir)
    viz.run_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermal Face Visualization")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="visualizations")
    args = parser.parse_args()
    main(args)
