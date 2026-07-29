"""
person_names.py
================
Simple person_id -> display name registry.

Stored separately from checkpoints/label_map.json because that file gets
fully rebuilt every time train.py runs (it only knows numeric IDs derived
from filenames); names are entered once by whoever runs collect_data.py and
need to survive retraining.
"""

import json
import os

DEFAULT_NAMES_PATH = os.path.join("data", "person_names.json")


def load_names(path: str = DEFAULT_NAMES_PATH) -> dict:
    """Returns {person_id (int): name (str)}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_names(names: dict, path: str = DEFAULT_NAMES_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in names.items()}, f, indent=2)


def set_name(person_id: int, name: str, path: str = DEFAULT_NAMES_PATH) -> dict:
    names = load_names(path)
    names[int(person_id)] = name
    save_names(names, path)
    return names
