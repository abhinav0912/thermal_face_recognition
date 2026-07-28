"""
prepare_data.py
===============
Extracts the two zip files into a clean  data/  folder.

Usage:
    python prepare_data.py \
        --rgb_zip   "RGB-faces-128x128.zip" \
        --thermal_zip "thermal-face-128x128.zip" \
        --out_dir   data
"""

import argparse
import zipfile
import os
from pathlib import Path


def extract(zip_path: str, out_dir: str):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    print(f"  Extracting {zip_path.name} → {out_dir}/")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)

    # Count extracted images
    out = Path(out_dir)
    total = sum(1 for p in out.rglob("*.jpg"))
    print(f"  Done – {total} .jpg files in {out_dir}/")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    extract(args.rgb_zip,     args.out_dir)
    extract(args.thermal_zip, args.out_dir)
    print("\n  Data ready. Expected structure:")
    print(f"    {args.out_dir}/")
    print(f"      RGB-faces-128x128/")
    print(f"        1-TD-A-0.jpg  …  113-TD-E-5.jpg")
    print(f"      thermal-face-128x128/")
    print(f"        1-TD-A-0.jpg  …  113-TD-E-5.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_zip",     default="RGB-faces-128x128.zip")
    parser.add_argument("--thermal_zip", default="thermal-face-128x128.zip")
    parser.add_argument("--out_dir",     default="data")
    main(parser.parse_args())
