"""
evaluate_recording.py
=====================
Replays a recording through the live pipeline (detector -> tracker -> classifier ->
gallery fallback -> optional name smoothing) and reports how well it names people.

This is the check that matters for the demo: the held-out score printed by train.py
comes from the same recording sessions as the training frames and read 99-100% for
models that named the right person in only 0-78% of live frames (2026-11-02).

A recording is a folder of images (sorted by name = time order) or a video file. Record
one with the camera, e.g. frames saved from camera.py, or collect_data.py's video clips.

    python evaluate_recording.py --frames PATH --expect Abhinav

--expect NAME: the true name. It assumes EVERY face in the recording is that person, so
use a recording with one person (or only that person). Without it the script still prints
the counts per name and how often labels flip.

Numbers are per face per frame ("face-frames") after the tracker has confirmed the face.
"flips" is how often the name shown for the same tracked face changes between frames,
per 100 face-frames: it is what looks like flicker on screen.
"""

import argparse
import glob
import os
from collections import Counter

import cv2
from PIL import Image

import gallery
from face_detector import ThermalFaceDetector, MultiFaceTracker
from inference import FaceRecognizer
from label_smoother import LabelSmoother, decide


def read_frames(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg")))
        for fp in files:
            frame = cv2.imread(fp)
            if frame is not None:
                yield frame
    else:
        cap = cv2.VideoCapture(path)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()


def summarize(title, shown, flips, face_frames, expect):
    """shown: Counter of displayed names (None = Unknown)."""
    parts = [f"{(name or 'Unknown')}: {n} ({100 * n / face_frames:.0f}%)"
             for name, n in shown.most_common()]
    print(f"  {title:<9} " + " | ".join(parts))
    if expect:
        right = shown.get(expect, 0)
        unknown = shown.get(None, 0)
        wrong = face_frames - right - unknown
        print(f"            correct {100 * right / face_frames:.0f}%  wrong name {100 * wrong / face_frames:.0f}%  "
              f"Unknown {100 * unknown / face_frames:.0f}%  |  label flips {100 * flips / face_frames:.1f} per 100 face-frames")
    else:
        print(f"            label flips {100 * flips / face_frames:.1f} per 100 face-frames")


def main(args):
    recognizer = FaceRecognizer(args.checkpoint_dir, device=args.device)
    detector = ThermalFaceDetector(detect_scale=1.0 if args.full_res else 0.5)
    tracker = MultiFaceTracker()
    smoother = LabelSmoother()
    gallery_data = gallery.load_gallery(args.gallery_path)

    raw_shown, smooth_shown = Counter(), Counter()
    raw_prev, smooth_prev = {}, {}
    raw_flips = smooth_flips = face_frames = frames = frames_with_face = 0

    for frame in read_frames(args.frames):
        frames += 1
        tracks = tracker.update(detector.detect(frame))
        ids, crops = [], []
        for tid, box in tracks.items():
            crop = detector.crop(frame, box)
            if crop.size:
                ids.append(tid)
                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        if not ids:
            continue
        frames_with_face += 1
        raw = [decide(r, e, gallery_data, args.unknown_threshold, args.gallery_threshold)
               for r, e in recognizer.predict_batch(crops, top_k=1)]
        smooth = smoother.update(ids, raw)
        for tid, r, s in zip(ids, raw, smooth):
            face_frames += 1
            raw_shown[r.name] += 1
            smooth_shown[s.name] += 1
            if tid in raw_prev and raw_prev[tid] != r.name:
                raw_flips += 1
            if tid in smooth_prev and smooth_prev[tid] != s.name:
                smooth_flips += 1
            raw_prev[tid], smooth_prev[tid] = r.name, s.name

    print(f"\n{args.frames}\n  {frames} frames, a confirmed face in {frames_with_face}, "
          f"{face_frames} face-frames, "
          f"{'full' if args.full_res else 'half'}-resolution detection, checkpoint {args.checkpoint_dir}")
    if not face_frames:
        print("  No face was confirmed in this recording.")
        return
    summarize("raw", raw_shown, raw_flips, face_frames, args.expect)
    summarize("smoothed", smooth_shown, smooth_flips, face_frames, args.expect)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Score the live pipeline on a recording")
    p.add_argument("--frames", required=True, help="Folder of images, or a video file")
    p.add_argument("--expect", default=None, help="True name of the (only) person in the recording")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--gallery_path", default=gallery.DEFAULT_GALLERY_PATH)
    p.add_argument("--unknown_threshold", type=float, default=0.5)
    p.add_argument("--gallery_threshold", type=float, default=0.6)
    p.add_argument("--full_res", action="store_true", help="Full-resolution detection (default: half)")
    p.add_argument("--device", default=None)
    main(p.parse_args())
