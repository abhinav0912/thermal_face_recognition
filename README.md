# Thermal Face Recognition

Recognize **who** a person is from a **thermal face image**. There's no
pre-existing dataset behind this — the roster is whoever's been captured live
with `collect_data.py` (see below); it grows only as more people get
collected, and the trained model only knows whoever's in `data/` at the time
`train.py` last ran.

- **Input**: single thermal `.jpg` image (128 × 128)
- **Output**: Person ID + name (whoever's currently in `data/thermal-face-128x128/`)

The model architecture also has an expression head (a leftover from an
earlier direction), but expression prediction is disabled throughout the
codebase right now — not a current focus. See the Architecture section.

---

## Project Structure

```
thermal_face_recognition/
├── prepare_data.py      # Optional: extract a zip-based dataset into data/, if you ever have one
├── model.py             # Shared model definition (DualHeadFaceNet) + label constants
├── train.py             # Train the dual-head MobileNetV2 model
├── evaluate.py          # Full evaluation + confusion matrices
├── inference.py         # Predict on saved thermal images
├── camera.py            # Frame source for the live pipeline (RTSP, e.g. FLIR A50)
├── face_detector.py     # Locates/crops/tracks faces in a raw live frame
├── live_inference.py    # Real-time recognition from the live camera feed (CLI/OpenCV window)
├── app.py               # Streamlit frontend for the same live pipeline (browser UI)
├── collect_data.py      # Capture new labeled thermal faces (video or stills) from the live feed
├── gallery.py           # Instant, no-retrain enrollment (data/gallery.json)
├── person_names.py      # person_id <-> name registry (data/person_names.json)
├── test_camera_connection.py  # One-off probe to find how a camera streams
├── pyproject.toml       # Dependencies + the GPU-index pin (see Quickstart, GPU vs CPU)
├── requirements.txt     # Same dependency list, for plain pip -- pyproject.toml/uv is the tested path
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

Uses [`uv`](https://docs.astral.sh/uv/) rather than raw `pip` -- mainly because
`pip install -r requirements.txt` silently resolves `torch` to its **CPU-only**
build (PyPI's default), and nothing about that command tells you it happened.
`pyproject.toml` pins `torch`/`torchvision` to PyTorch's CUDA 12.8 index instead,
so the one command below is enough to get a GPU-enabled install on a machine with
an NVIDIA GPU -- see the **GPU vs CPU** section for what to do on a CPU-only machine.

```bash
uv sync
```

That creates `.venv/` and installs everything, GPU build included. Run any
script from here on as `uv run python train.py ...` (or activate the venv
first with `.venv\Scripts\Activate.ps1` / `source .venv/bin/activate` and
just run `python train.py ...`).

### 2 – Collect data

There's no bundled dataset. Every person the model can recognize has to be
captured live first:

```bash
uv run python collect_data.py
```

Prompts for a person ID (new or existing) and a name, then records a short
video while you move your head/expression naturally -- frames get
auto-sampled, face-cropped, and saved into `data/thermal-face-128x128/`. See
**Live Recognition** below for the full walkthrough (video vs. angle-shot
capture, the naming convention, etc.).

(`prepare_data.py` still exists for extracting a zip-based dataset into
`data/`, if you ever have one in the original `{id}-TD-A-{n}.jpg` /
`{id}-TD-E-{n}.jpg` layout -- not needed for the live-collection path above.)

### 3 – Train

```bash
uv run python train.py --epochs 40 --batch_size 32
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
| `--workers` | `0` on Windows, `4` elsewhere | DataLoader workers -- Windows' spawn-based multiprocessing has deadlocked here with workers>0; override with `--workers N` if your setup doesn't hit that |

Training uses a **two-phase** strategy:
- **Epochs 1–9**: Only the classification heads are trained (backbone frozen)  
- **Epoch 10+**: Full fine-tuning with a lower learning rate

Label map (`checkpoints/label_map.json`) is rebuilt from whatever's in
`data/thermal-face-128x128/` every run -- adding a new person is just:
collect their data, re-run this.

### 4 – Evaluate

```bash
uv run python evaluate.py
```

Outputs per-class precision/recall/F1 + a confusion matrix PNG.

### 5 – Inference

```bash
# Single image
uv run python inference.py --image data/thermal-face-128x128/<id>-TD-A-0.jpg

# All images in a folder
uv run python inference.py --folder data/thermal-face-128x128/

# Interactive mode (type paths one by one)
uv run python inference.py --demo
```

Example output:
```
──────────────────────────────────────────────────
  Image      : 120-TD-A-0.jpg
  Person     : Abhinav  (ID 120)  (confidence: 97.3%)

  Top-3 persons:
    Person 120  →  97.3%
    Person 121  →   2.2%
    Person 124  →   0.3%
──────────────────────────────────────────────────
```

---

## Live Recognition (FLIR A50)

The A50 streams over RTSP. Confirmed working URL for this setup:
`rtsp://169.254.0.82:554/avc` (find yours with `test_camera_connection.py --ip <ip>`
if the camera's IP is different).

```bash
# Real-time recognition from the live feed (default source is the URL above)
python live_inference.py

# Or point at a different camera / URL
python live_inference.py --source rtsp://<camera-ip>:554/avc

# Test the pipeline with a regular webcam before the thermal camera is available
python live_inference.py --source 0
```

Each frame is: face-detected (OpenCV Haar cascade + CLAHE contrast boost) →
cropped → resized to 128×128 → run through the same `DualHeadFaceNet` used
for static images. A bounding box with the predicted person's **name** is
drawn over the video window; press `q` to quit. Any face whose
top match falls below `--unknown_threshold` (default 50%) is labeled
`Unknown` instead of being forced onto the closest known identity — the
model is a closed-set classifier over the trained people, so this threshold
matters for anyone not in the training set. Names come from
`data/person_names.json` (see below); a person with no registered name falls
back to `Person {id}`.

### Collecting new training data live

```bash
python collect_data.py
```

Prompts for a person ID (existing or new — new IDs are asked for a name,
which live_inference.py then displays instead of the numeric ID) and a
capture mode:

- **`v` — video** (recommended): records a short clip (15s by default,
  `--duration` to change) while the person naturally moves their head and
  expression. Frames are auto-sampled a few times a second, face-cropped,
  and saved as training images — this is the fastest way to build up a
  person's data. The raw clip is also kept in `data/videos/` for reference
  or re-processing later.
- **`a` — angle shots**: 9 manually-posed stills, SPACE to capture each.

(Expression-shot capture is currently disabled — not a focus for now, see below.)

Either way, images are saved into `data/thermal-face-128x128/` using the
`{id}-TD-A-{n}.jpg` / `{id}-TD-E-{1..5}.jpg` naming convention (see below),
so adding a new person is just: collect their data, then re-run
`python train.py` — `label_map.json` is rebuilt from whatever's on disk
each run.

---

## Streamlit Frontend

A browser-based UI over the same live pipeline (`app.py`) — no OpenCV window,
no keyboard-focus juggling, just buttons.

```bash
streamlit run app.py
```

- **Live feed** (left column): toggle "Run live feed" on to start recognizing
  everyone currently in frame (multi-person, same classifier → gallery →
  Unknown logic as `live_inference.py`, batched into one model call per frame
  regardless of how many faces are in it). Toggle off to pause.
- **Settings + Registered people** (right column): camera source, checkpoint
  dir, device, both confidence thresholds, and a live roster of everyone
  registered (trained + gallery-enrolled), with counts.
- **Enroll a new person** (bottom, always visible): type a name and click
  "Capture & enroll" (disabled while the feed is running -- click "Pause feed
  first" or toggle it off yourself) to capture a few frames of whoever's
  largest/closest to the camera and add them to the live gallery, no
  retraining. Same trade-off as `live_inference.py`'s `n` key: instant, but
  weaker than a fully trained identity.

Data collection (video/angle capture for actual training) is still done via
`collect_data.py` — the browser UI is for the live demo/enrollment side, not
for building the training dataset itself.

---

## Naming Convention

| Filename Pattern | Meaning |
|---|---|
| `{id}-TD-A-{0..8}.jpg` | Person `id`, 9 angle shots |
| `{id}-TD-E-{1..5}.jpg` | Person `id`, expression 1–5 (capture currently disabled -- see below) |

Expression index mapping (inactive while expression capture/training is disabled, kept for when it's revisited):

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
(N)       (5)
```
`N` = however many people are currently in `data/thermal-face-128x128/` --
set dynamically by `train.py` from `label_map.json`, not fixed.

- **Loss**: identity cross-entropy only right now -- the expression head still
  exists (so old checkpoints keep loading) but isn't part of the loss or
  reported metrics; expression training/capture is disabled throughout the
  codebase (not a current focus, revisit later)
- **Optimizer**: AdamW + Cosine Annealing LR
- **Augmentation**: Horizontal flip, color jitter, random rotation

---

## VS Code Usage

Open the project folder in VS Code, then use the **Run and Debug** panel (Ctrl+Shift+D) to select and run any of the pre-configured launch targets:

1. **Collect Data** – capture a new person live from the camera
2. **Train Model** – full training run
3. **Evaluate Model** – metrics + plots
4. **Inference – Single Image** – predict one image
5. **Inference – Interactive Demo** – type image paths interactively
6. **Inference – Whole Folder** – batch predict

---

## GPU vs CPU

The code auto-detects and uses a GPU (`cuda`) whenever one's available, in
every script (training, evaluation, inference, live recognition). On CPU it
still works, just slower.

**Getting the GPU build installed in the first place is the part that
silently goes wrong**: `pip install -r requirements.txt` resolves `torch`
from PyPI, whose default wheel is **CPU-only** -- nothing about that command
warns you, `torch.cuda.is_available()` will just quietly return `False`. This
project's `pyproject.toml` fixes that for anyone using `uv sync` (see
Quickstart step 1) by pinning `torch`/`torchvision` to PyTorch's CUDA 12.8
index instead. If you're on a CPU-only machine, delete the `[[tool.uv.index]]`
/ `[tool.uv.sources]` blocks at the bottom of `pyproject.toml` and `uv sync`
will fall back to the normal CPU build.

If you do end up on plain `pip` instead of `uv` for some reason, get the CUDA
build explicitly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
(pick the CUDA version matching your driver -- `nvidia-smi` shows it)

`--workers` (train.py) defaults to `0` on Windows, `4` elsewhere -- Windows'
spawn-based multiprocessing has deadlocked here with `workers>0` (both
CPU and GPU usage go flat, no error, no progress); override with
`--workers N` if your setup doesn't hit that.

---

## Expected Performance

No fixed number to quote here -- there's no benchmark dataset behind this,
just whoever's been collected live (see Quickstart), so accuracy depends
entirely on how many people are registered and how varied their captured
data is (angles, lighting, expressions, with/without glasses). A handful of
people with a few hundred well-varied frames each can validate near 100% --
that's a sign the model has cleanly learned to tell *that specific roster*
apart, not a general accuracy guarantee, and it won't by itself generalize to
conditions the training data didn't cover (e.g. sunglasses, if none of the
captured data included them). Run `python evaluate.py` after training to see
real numbers for your current roster.
