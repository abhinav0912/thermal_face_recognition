"""
inference.py
============
Given a thermal face image (path or directory), predicts:
  • Person identity  (person ID 1–113)
  • Facial expression (Angry / Happy / Neutral / Sad / Surprised)

Usage:
    # Single image
    python inference.py --image path/to/thermal.jpg

    # Whole folder
    python inference.py --folder path/to/thermal_folder/

    # Interactive demo (opens a file-picker loop)
    python inference.py --demo
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Reproduce model class (keep in sync with train.py or import from shared module)
# ──────────────────────────────────────────────────────────────────────────────

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
        self.identity_head   = nn.Linear(512, num_persons)
        self.expression_head = nn.Linear(512, num_expressions)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        feat = self.shared_fc(x)
        return self.identity_head(feat), self.expression_head(feat)


# ──────────────────────────────────────────────────────────────────────────────
#  Inference engine
# ──────────────────────────────────────────────────────────────────────────────

EXPR_NAMES = ["Neutral", "Smile", "Eyes Closed", "Shocked", "Wearing Sunglasses"]

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class FaceRecognizer:
    """Load a trained checkpoint and run inference on thermal images."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Load label map ───────────────────────────────────────────────────
        lmap_path = os.path.join(checkpoint_dir, "label_map.json")
        if not os.path.exists(lmap_path):
            raise FileNotFoundError(
                f"label_map.json not found in '{checkpoint_dir}'.\n"
                "Please run train.py first."
            )
        with open(lmap_path) as f:
            lmap = json.load(f)

        # label_map.json stores pid_to_label as str->int (JSON keys are strings)
        self.label_to_pid = {v: int(k) for k, v in lmap["pid_to_label"].items()}
        num_persons = len(self.label_to_pid)

        # ── Load model ───────────────────────────────────────────────────────
        ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"best_model.pth not found in '{checkpoint_dir}'.\n"
                "Please run train.py first."
            )
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model = DualHeadFaceNet(
            num_persons=ckpt.get("num_persons", num_persons),
            num_expressions=ckpt.get("num_expressions", 5),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        print(f"  Model loaded  : {ckpt_path}  (epoch {ckpt.get('epoch','?')})")
        print(f"  Persons known : {num_persons}")
        print(f"  Device        : {self.device}\n")

    @torch.no_grad()
    def predict(self, image_path: str, top_k: int = 3) -> dict:
        """
        Parameters
        ----------
        image_path : str  path to thermal .jpg/.png
        top_k      : int  number of top-person candidates to return

        Returns
        -------
        dict with keys:
            person_id   : int   – predicted person ID (1-based)
            person_conf : float – confidence (0-1) for that person
            top_persons : list  – [(person_id, confidence), ...]
            expression  : str   – predicted expression name
            expr_conf   : float – confidence (0-1) for that expression
            top_exprs   : list  – [(expr_name, confidence), ...]
        """
        img = Image.open(image_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)

        id_logits, expr_logits = self.model(tensor)

        id_probs   = F.softmax(id_logits,   dim=1)[0].cpu().numpy()
        expr_probs = F.softmax(expr_logits, dim=1)[0].cpu().numpy()

        # Person
        top_k_ids  = np.argsort(id_probs)[::-1][:top_k]
        person_id  = self.label_to_pid[int(top_k_ids[0])]
        person_conf = float(id_probs[top_k_ids[0]])
        top_persons = [(self.label_to_pid[int(i)], float(id_probs[i]))
                       for i in top_k_ids]

        # Expression
        top_expr_ids = np.argsort(expr_probs)[::-1]
        expr_idx  = int(top_expr_ids[0])
        expression = EXPR_NAMES[expr_idx]
        expr_conf  = float(expr_probs[expr_idx])
        top_exprs  = [(EXPR_NAMES[i], float(expr_probs[i]))
                      for i in top_expr_ids]

        return {
            "person_id":   person_id,
            "person_conf": person_conf,
            "top_persons": top_persons,
            "expression":  expression,
            "expr_conf":   expr_conf,
            "top_exprs":   top_exprs,
        }


def print_result(image_path: str, result: dict):
    """Pretty-print a single prediction result."""
    bar = "─" * 50
    print(f"\n{bar}")
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Person ID  : {result['person_id']}  "
          f"(confidence: {result['person_conf']*100:.1f}%)")
    print(f"  Expression : {result['expression']}  "
          f"(confidence: {result['expr_conf']*100:.1f}%)")
    print(f"\n  Top-3 persons:")
    for pid, conf in result["top_persons"]:
        print(f"    Person {pid:>3d}  →  {conf*100:.1f}%")
    print(f"\n  All expressions:")
    for expr, conf in result["top_exprs"]:
        print(f"    {expr:<12s}  →  {conf*100:.1f}%")
    print(bar)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    recognizer = FaceRecognizer(args.checkpoint_dir)

    if args.image:
        result = recognizer.predict(args.image, top_k=3)
        print_result(args.image, result)

    elif args.folder:
        folder = Path(args.folder)
        images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        if not images:
            print(f"No images found in {folder}")
            return
        print(f"Running inference on {len(images)} images …\n")
        for img_path in images:
            result = recognizer.predict(str(img_path), top_k=3)
            print_result(str(img_path), result)

    elif args.demo:
        print("=== Interactive Demo ===")
        print("Enter the path to a thermal image (or 'q' to quit):\n")
        while True:
            path = input("  Image path: ").strip()
            if path.lower() in ("q", "quit", "exit"):
                break
            if not os.path.exists(path):
                print(f"  File not found: {path}")
                continue
            try:
                result = recognizer.predict(path, top_k=3)
                print_result(path, result)
            except Exception as e:
                print(f"  Error: {e}")

    else:
        print("Please specify --image, --folder, or --demo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermal Face Inference")
    parser.add_argument("--image",          default=None,
                        help="Path to a single thermal image")
    parser.add_argument("--folder",         default=None,
                        help="Path to a folder of thermal images")
    parser.add_argument("--demo",           action="store_true",
                        help="Interactive CLI demo mode")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="Directory containing best_model.pth & label_map.json")
    args = parser.parse_args()
    main(args)
