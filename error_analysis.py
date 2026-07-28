"""
error_analysis.py
=================
Detailed error analysis for thermal face recognition.

Identifies:
  • Hardest identities (lowest accuracy per person)
  • Hardest expressions (per emotion)
  • Top confusion pairs
  • Confidence calibration
  • False positive/negative analysis

Usage:
    python error_analysis.py --checkpoint_dir checkpoints --data_dir data
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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
import re

EXPR_MAP = {
    "1": "Angry",
    "2": "Happy",
    "3": "Neutral",
    "4": "Sad",
    "5": "Surprised",
}

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


def parse_filename(fname: str):
    """Parse filename to get person_id, mode, index."""
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
            self.samples.append((img_path, person_lbl, expr_lbl, pid))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, person_lbl, expr_lbl, pid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, person_lbl, expr_lbl, pid, str(path.name)


class ErrorAnalyzer:
    def __init__(self, checkpoint_dir: str, data_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
        self.results = {}

    def run_inference(self):
        """Run model on all data and collect predictions + confidences."""
        all_id_preds = []
        all_id_true = []
        all_id_conf = []
        all_ex_preds = []
        all_ex_true = []
        all_ex_conf = []
        all_pids = []
        all_filenames = []
        
        with torch.no_grad():
            for imgs, id_lbl, expr_lbl, pid, fname in self.loader:
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
                
                all_pids.extend(pid)
                all_filenames.extend(fname)
        
        return {
            "id_preds": np.array(all_id_preds),
            "id_true": np.array(all_id_true),
            "id_conf": np.array(all_id_conf),
            "ex_preds": np.array(all_ex_preds),
            "ex_true": np.array(all_ex_true),
            "ex_conf": np.array(all_ex_conf),
            "pids": all_pids,
            "filenames": all_filenames,
        }

    def analyze_per_person_accuracy(self, data):
        """Accuracy breakdown by person."""
        preds = data["id_preds"]
        trues = data["id_true"]
        pids = data["pids"]
        
        person_accs = {}
        for person_label in range(len(self.dataset.person_ids)):
            mask = trues == person_label
            if mask.sum() == 0:
                continue
            acc = (preds[mask] == trues[mask]).mean()
            pid = self.dataset.person_ids[person_label]
            person_accs[pid] = float(acc)
        
        # Sort by accuracy
        sorted_accs = sorted(person_accs.items(), key=lambda x: x[1])
        
        print("\n── HARDEST IDENTITIES (Lowest Accuracy) ──")
        for pid, acc in sorted_accs[:10]:
            print(f"  Person {pid:3d} : {acc*100:.1f}%")
        
        print("\n── EASIEST IDENTITIES (Highest Accuracy) ──")
        for pid, acc in sorted_accs[-10:][::-1]:
            print(f"  Person {pid:3d} : {acc*100:.1f}%")
        
        self.results["per_person_accuracy"] = person_accs
        return person_accs

    def analyze_per_expression_accuracy(self, data):
        """Accuracy breakdown by expression."""
        preds = data["ex_preds"]
        trues = data["ex_true"]
        
        expr_accs = {}
        for expr_idx in range(5):
            mask = trues == expr_idx
            if mask.sum() == 0:
                continue
            acc = (preds[mask] == trues[mask]).mean()
            expr_name = EXPR_MAP[str(expr_idx + 1)]
            expr_accs[expr_name] = float(acc)
        
        print("\n── EXPRESSION ACCURACY ──")
        for expr, acc in sorted(expr_accs.items(), key=lambda x: x[1]):
            print(f"  {expr:12s} : {acc*100:.1f}%")
        
        self.results["per_expression_accuracy"] = expr_accs
        return expr_accs

    def analyze_top_confusions(self, data, top_k=10):
        """Find most common confusion pairs."""
        preds = data["id_preds"]
        trues = data["id_true"]
        
        # Only look at misclassifications
        wrong = preds != trues
        
        confusions = {}
        for true_label, pred_label in zip(trues[wrong], preds[wrong]):
            true_pid = self.dataset.person_ids[true_label]
            pred_pid = self.dataset.person_ids[pred_label]
            key = (true_pid, pred_pid)
            confusions[key] = confusions.get(key, 0) + 1
        
        # Sort by frequency
        sorted_conf = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n── TOP {top_k} CONFUSION PAIRS ──")
        print(f"(Format: Actual Person → Misclassified As [count])")
        for (true_pid, pred_pid), count in sorted_conf[:top_k]:
            print(f"  Person {true_pid:3d} → Person {pred_pid:3d} [{count}x]")
        
        self.results["top_confusions"] = [
            {"true_person": int(t), "pred_person": int(p), "count": int(c)}
            for (t, p), c in sorted_conf[:top_k]
        ]
        return sorted_conf

    def analyze_confidence_calibration(self, data):
        """Check if confidence scores are well-calibrated."""
        preds = data["id_preds"]
        trues = data["id_true"]
        confs = data["id_conf"]
        
        correct = preds == trues
        
        print("\n── CONFIDENCE CALIBRATION ──")
        print(f"  Mean confidence (correct)   : {confs[correct].mean()*100:.1f}%")
        print(f"  Mean confidence (incorrect) : {confs[~correct].mean()*100:.1f}%")
        print(f"  Min confidence              : {confs.min()*100:.1f}%")
        print(f"  Max confidence              : {confs.max()*100:.1f}%")
        
        # Bins
        for threshold in [0.5, 0.7, 0.9, 0.95]:
            high_conf = confs >= threshold
            if high_conf.sum() > 0:
                acc_high = (preds[high_conf] == trues[high_conf]).mean()
                print(f"  Accuracy when conf >= {threshold:.2f} : {acc_high*100:.1f}% (n={high_conf.sum()})")
        
        self.results["confidence_calibration"] = {
            "mean_conf_correct": float(confs[correct].mean()),
            "mean_conf_incorrect": float(confs[~correct].mean()),
        }

    def save_results(self, output_path: str = "error_analysis.json"):
        """Save analysis results."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n  Error analysis saved → {output_path}")

    def run_all(self):
        """Run complete analysis."""
        print("\n" + "="*70)
        print("  ERROR ANALYSIS – THERMAL FACE RECOGNITION")
        print("="*70)
        
        data = self.run_inference()
        self.analyze_per_person_accuracy(data)
        self.analyze_per_expression_accuracy(data)
        self.analyze_top_confusions(data, top_k=15)
        self.analyze_confidence_calibration(data)
        
        print("\n" + "="*70 + "\n")


def main(args):
    analyzer = ErrorAnalyzer(args.checkpoint_dir, args.data_dir)
    analyzer.run_all()
    analyzer.save_results(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermal Face Error Analysis")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output", default="error_analysis.json")
    args = parser.parse_args()
    main(args)
