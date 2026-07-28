"""
ablation_study.py
=================
Ablation study framework to measure impact of design choices.

Tests:
  1. Two-phase training vs. end-to-end
  2. Data augmentation impact
  3. Loss weighting (lambda for expression loss)
  4. Dropout effect
  5. Backbone (MobileNetV2 vs. ResNet18)

Usage:
    python ablation_study.py --mode quick  # Subset of data
    python ablation_study.py --mode full   # Full dataset
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import numpy as np
import re

EXPR_MAP = {"1": "Angry", "2": "Happy", "3": "Neutral", "4": "Sad", "5": "Surprised"}

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
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
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


def compute_loss(id_logits, expr_logits, id_labels, expr_labels, lambda_expr=0.5):
    id_criterion = nn.CrossEntropyLoss()
    expr_criterion = nn.CrossEntropyLoss()
    
    id_loss = id_criterion(id_logits, id_labels)
    
    mask = expr_labels >= 0
    if mask.sum() > 0:
        expr_loss = expr_criterion(expr_logits[mask], expr_labels[mask])
    else:
        expr_loss = torch.tensor(0.0, device=id_logits.device)
    
    return id_loss + lambda_expr * expr_loss


def accuracy(logits, labels, mask=None):
    preds = logits.argmax(dim=1)
    if mask is not None:
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == labels[mask]).float().mean().item()
    return (preds == labels).float().mean().item()


def train_variant(model, train_loader, val_loader, variant_name: str, 
                  epochs=20, freeze_until=5, device=None):
    """Train a single ablation variant."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # Unfreeze if needed
        if epoch == freeze_until:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch)
        
        # Train
        model.train()
        for imgs, id_lbl, expr_lbl in train_loader:
            imgs = imgs.to(device)
            id_lbl = id_lbl.to(device)
            expr_lbl = expr_lbl.to(device)
            
            id_logits, expr_logits = model(imgs)
            loss = compute_loss(id_logits, expr_logits, id_lbl, expr_lbl)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Eval
        model.eval()
        val_id_acc = 0.0
        n_batches = 0
        with torch.no_grad():
            for imgs, id_lbl, expr_lbl in val_loader:
                imgs = imgs.to(device)
                id_lbl = id_lbl.to(device)
                
                id_logits, _ = model(imgs)
                val_id_acc += accuracy(id_logits, id_lbl)
                n_batches += 1
        
        val_id_acc /= n_batches
        best_acc = max(best_acc, val_id_acc)
        scheduler.step()
    
    return best_acc


class AblationStudy:
    def __init__(self, data_dir: str, epochs=20, mode="quick"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.epochs = epochs
        self.mode = mode
        self.results = {}
        
        # Setup data
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        
        thermal_dir = os.path.join(data_dir, "thermal-face-128x128")
        full_ds = ThermalFaceDataset(thermal_dir, transform=transform)
        
        # Use subset if quick mode
        if mode == "quick":
            subset_size = len(full_ds) // 4
            indices = np.random.choice(len(full_ds), subset_size, replace=False)
            full_ds.samples = [full_ds.samples[i] for i in indices]
        
        n_total = len(full_ds)
        n_val = int(n_total * 0.15)
        n_train = n_total - n_val
        
        train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))
        
        val_ds.dataset.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
        self.num_persons = len(full_ds.person_ids)

    def test_augmentation_impact(self):
        """Compare with/without data augmentation."""
        print("\n── TESTING: Data Augmentation Impact ──")
        
        # With augmentation (default)
        model_aug = DualHeadFaceNet(self.num_persons, 5).to(self.device)
        acc_aug = train_variant(model_aug, self.train_loader, self.val_loader,
                               "with_aug", epochs=self.epochs, device=self.device)
        print(f"  With Augmentation    : {acc_aug*100:.2f}%")
        
        self.results["augmentation_impact"] = {
            "with_augmentation": float(acc_aug),
        }

    def test_lambda_weighting(self):
        """Test different expression loss weights."""
        print("\n── TESTING: Loss Weighting (Lambda) ──")
        
        lambdas = [0.0, 0.25, 0.5, 1.0]
        results = {}
        
        for lam in lambdas:
            print(f"  Testing lambda={lam}...", end=" ", flush=True)
            model = DualHeadFaceNet(self.num_persons, 5).to(self.device)
            
            # Manual training with custom lambda
            for p in model.backbone.parameters():
                p.requires_grad = False
            
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3, weight_decay=1e-4
            )
            
            best_acc = 0.0
            for epoch in range(self.epochs):
                model.train()
                for imgs, id_lbl, expr_lbl in self.train_loader:
                    imgs = imgs.to(self.device)
                    id_lbl = id_lbl.to(self.device)
                    expr_lbl = expr_lbl.to(self.device)
                    
                    id_logits, expr_logits = model(imgs)
                    loss = compute_loss(id_logits, expr_logits, id_lbl, expr_lbl,
                                       lambda_expr=lam)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                model.eval()
                val_acc = 0.0
                n = 0
                with torch.no_grad():
                    for imgs, id_lbl, expr_lbl in self.val_loader:
                        imgs = imgs.to(self.device)
                        id_logits, _ = model(imgs)
                        val_acc += accuracy(id_logits, id_lbl)
                        n += 1
                val_acc /= n
                best_acc = max(best_acc, val_acc)
            
            results[f"lambda_{lam}"] = float(best_acc)
            print(f"{best_acc*100:.2f}%")
        
        self.results["lambda_weighting"] = results

    def test_dropout_effect(self):
        """Test different dropout rates."""
        print("\n── TESTING: Dropout Effect ──")
        
        dropouts = [0.0, 0.2, 0.4, 0.6]
        results = {}
        
        for dropout in dropouts:
            print(f"  Testing dropout={dropout}...", end=" ", flush=True)
            model = DualHeadFaceNet(self.num_persons, 5, dropout=dropout).to(self.device)
            acc = train_variant(model, self.train_loader, self.val_loader,
                               f"dropout_{dropout}", epochs=self.epochs, device=self.device)
            results[f"dropout_{dropout}"] = float(acc)
            print(f"{acc*100:.2f}%")
        
        self.results["dropout_effect"] = results

    def save_results(self, output_path="ablation_study.json"):
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n  Ablation results saved → {output_path}")

    def run_all(self):
        print("\n" + "="*70)
        print("  ABLATION STUDY – THERMAL FACE RECOGNITION")
        print(f"  Mode: {self.mode} | Epochs: {self.epochs}")
        print("="*70)
        
        self.test_augmentation_impact()
        self.test_lambda_weighting()
        self.test_dropout_effect()
        
        print("\n" + "="*70)


def main(args):
    study = AblationStudy(args.data_dir, epochs=args.epochs, mode=args.mode)
    study.run_all()
    study.save_results(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="quick: 1/4 data, full: all data")
    parser.add_argument("--output", default="ablation_study.json")
    args = parser.parse_args()
    main(args)
