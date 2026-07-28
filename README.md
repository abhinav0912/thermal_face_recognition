# Thermal Face Recognition

Recognize **who** a person is and **what expression** they are making from a **thermal face image**.

- **113 people**, 9 angle shots + 5 expression shots each  
- **Input**: single thermal `.jpg` image (128 × 128)  
- **Output**: Person ID (1–113) + Expression (Angry / Happy / Neutral / Sad / Surprised)

---

## Project Structure

```
thermal_face_recognition/
├── prepare_data.py      # Extract zip files into data/
├── train.py             # Train the dual-head MobileNetV2 model
├── evaluate.py          # Full evaluation + confusion matrices
├── inference.py         # Predict on new thermal images
├── requirements.txt
├── .vscode/
│   ├── launch.json      # One-click run configs for VS Code
│   └── settings.json
└── README.md
```

After running, these folders are created automatically:
```
data/
├── thermal-face-128x128/   ← thermal images used for training
└── RGB-faces-128x128/      ← RGB images (available for future use)

checkpoints/
├── best_model.pth          ← saved model weights
├── label_map.json          ← person ID mapping
├── training_curves.png     ← loss/accuracy plots
├── cm_identity.png         ← identity confusion matrix
└── cm_expression.png       ← expression confusion matrix
```

---

## Quickstart

### 1 – Set up environment

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 2 – Place zip files

Copy both zip files into the project root:
```
thermal_face_recognition/
├── RGB-faces-128x128.zip
├── thermal-face-128x128.zip
├── train.py
└── ...
```

### 3 – Prepare data

```bash
python prepare_data.py
```

This extracts both zips into `data/`.

### 4 – Train

```bash
python train.py --epochs 40 --batch_size 32
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data` | Root data directory |
| `--output_dir` | `checkpoints` | Where to save model & plots |
| `--epochs` | `40` | Total training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--val_split` | `0.15` | Fraction of data for validation |
| `--unfreeze_epoch` | `10` | Epoch to unfreeze backbone |
| `--workers` | `4` | DataLoader workers |

Training uses a **two-phase** strategy:
- **Epochs 1–9**: Only the classification heads are trained (backbone frozen)  
- **Epoch 10+**: Full fine-tuning with a lower learning rate

### 5 – Evaluate

```bash
python evaluate.py
```

Outputs per-class precision/recall/F1 for both tasks + confusion matrix PNGs.

### 6 – Inference

```bash
# Single image
python inference.py --image data/thermal-face-128x128/7-TD-E-3.jpg

# All images in a folder
python inference.py --folder data/thermal-face-128x128/

# Interactive mode (type paths one by one)
python inference.py --demo
```

Example output:
```
──────────────────────────────────────────────────
  Image      : 7-TD-E-3.jpg
  Person ID  : 7     (confidence: 98.4%)
  Expression : Neutral  (confidence: 87.2%)

  Top-3 persons:
    Person   7  →  98.4%
    Person  71  →   0.8%
    Person  17  →   0.4%

  All expressions:
    Neutral       →  87.2%
    Happy         →   7.1%
    Sad           →   3.1%
    Angry         →   1.8%
    Surprised     →   0.8%
──────────────────────────────────────────────────
```

---

## Naming Convention

| Filename Pattern | Meaning |
|---|---|
| `{id}-TD-A-{0..8}.jpg` | Person `id`, 9 angle shots |
| `{id}-TD-E-{1..5}.jpg` | Person `id`, expression 1–5 |

Expression index mapping:

| Index | Expression |
|---|---|
| 1 | Angry |
| 2 | Happy |
| 3 | Neutral |
| 4 | Sad |
| 5 | Surprised |

---

## Architecture

```
Input (128×128 thermal image)
        │
  MobileNetV2 Backbone  (pretrained ImageNet, fine-tuned)
        │
  AdaptiveAvgPool → Flatten
        │
  FC(1280→512) + BN + ReLU + Dropout
       / \
      /   \
Identity  Expression
Head      Head
(113)     (5)
```

- **Loss**: Cross-entropy for identity + 0.5 × cross-entropy for expression  
- **Expression loss** is only computed on images tagged `TD-E` (not angle shots)
- **Optimizer**: AdamW + Cosine Annealing LR
- **Augmentation**: Horizontal flip, color jitter, random rotation

---

## VS Code Usage

Open the project folder in VS Code, then use the **Run and Debug** panel (Ctrl+Shift+D) to select and run any of the pre-configured launch targets:

1. **Prepare Data** – extract zips
2. **Train Model** – full training run
3. **Evaluate Model** – metrics + plots
4. **Inference – Single Image** – predict one image
5. **Inference – Interactive Demo** – type image paths interactively
6. **Inference – Whole Folder** – batch predict

---

## GPU vs CPU

The code automatically uses a GPU if one is available (`cuda`). On CPU, training will be slower but works correctly. Reduce `--workers` to `0` on Windows if you encounter DataLoader issues.

---

## Expected Performance

With 40 epochs of training:

| Metric | Expected Range |
|---|---|
| Identity Accuracy | 90–98% |
| Expression Accuracy | 80–92% |

Results depend on hardware, random seed, and exact split.
