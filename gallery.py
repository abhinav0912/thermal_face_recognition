"""
gallery.py
==========
A small "enroll instantly, no retraining" fallback for live_inference.py.

The trained classifier (checkpoints/best_model.pth) only knows the people it
was trained on, and adding someone new means resizing its output layer and
retraining — never instant. This gallery sidesteps that for live demo
enrollment: it stores an averaged 512-dim feature vector (see
DualHeadFaceNet.get_embedding) per enrolled name, and matches new faces by
cosine similarity against it.

This is meant as a FALLBACK only, used in live_inference.py when the trained
classifier isn't confident about a face — someone the classifier already
knows should never need this path. Match quality here is inherently weaker
than the trained classifier, since these embeddings come from a network
trained for classification, not metric learning, and are usually averaged
from just a handful of enrollment frames rather than a full training set.
"""

import json
import os

import numpy as np

DEFAULT_GALLERY_PATH = os.path.join("data", "gallery.json")


def load_gallery(path: str = DEFAULT_GALLERY_PATH) -> dict:
    """Returns {name: embedding (list of float)}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_gallery(gallery: dict, path: str = DEFAULT_GALLERY_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(gallery, f, indent=2)


def enroll(name: str, embeddings: list, path: str = DEFAULT_GALLERY_PATH) -> dict:
    """
    Average one or more embedding vectors (e.g. from several captured
    frames) and store them under `name`. Re-enrolling an existing name
    overwrites their entry rather than blending with the old one.
    """
    avg = np.mean(np.stack(embeddings), axis=0)
    gallery = load_gallery(path)
    gallery[name] = avg.tolist()
    save_gallery(gallery, path)
    return gallery


def match(embedding, gallery: dict, threshold: float = 0.6):
    """
    Cosine-similarity match against every enrolled entry.
    Returns (name, similarity) for the best match at or above `threshold`,
    or (None, best_similarity) if nothing clears the bar (best_similarity
    is 0.0 if the gallery is empty).
    """
    if not gallery:
        return None, 0.0

    query = np.asarray(embedding, dtype=np.float64)
    query_norm = query / (np.linalg.norm(query) + 1e-8)

    best_name, best_sim = None, -1.0
    for name, vec in gallery.items():
        vec = np.asarray(vec, dtype=np.float64)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        sim = float(np.dot(query_norm, vec_norm))
        if sim > best_sim:
            best_name, best_sim = name, sim

    if best_sim >= threshold:
        return best_name, best_sim
    return None, best_sim
