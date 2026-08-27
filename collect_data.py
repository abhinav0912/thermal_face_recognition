"""
collect_data.py
================
Capture live thermal face data and save it using the exact naming
convention train.py expects, so new people/expressions can be folded into
the dataset just by re-running train.py — no separate merge step needed.

Usage:
    python collect_data.py
    python collect_data.py --source rtsp://169.254.0.82:554/avc

Flow:
    1. Enter a person ID (reuse an existing one to add more data to them, or
       a new number to add a new person — you'll be asked for their name the
       first time, so the live feed can display it instead of a numeric ID).
    2. Choose a capture mode:
         [v]ideo      - record a short clip while the person moves their
                        head/expression naturally; frames are auto-sampled
                        and face-cropped into training images. This is the
                        fastest way to build up a dataset for someone.
         [a]ngle shots - 9 manually-posed stills (SPACE to capture each)
         [e]xpression  - 5 manually-posed stills, one per expression
    3. Repeat for more people/sessions, or 'q' at the person-ID prompt to exit.

Saved images land in data/thermal-face-128x128/ as {id}-TD-A-{n}.jpg or
{id}-TD-E-{1..5}.jpg, matching what train.py's ThermalFaceDataset parses.
Raw videos are also kept in data/videos/ for reference/re-processing.
Person names are stored in data/person_names.json (see person_names.py).
"""

import argparse
import os
import re
import time

import cv2

from camera import ThermalCamera, DEFAULT_RTSP_URL
from face_detector import ThermalFaceDetector
from model import EXPR_NAMES
from person_names import load_names, set_name, DEFAULT_NAMES_PATH

IMG_SIZE = 128


def capture_one(camera, detector, window_name):
    """
    Show a live preview until the user presses SPACE (capture) or 'q' (skip).
    Returns the cropped+resized face image (BGR np.ndarray), or None if skipped.
    """
    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            continue

        boxes = detector.detect(frame)
        display = frame.copy()
        crop = None
        if boxes:
            box = boxes[0]
            x, y, w, h = box
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop = detector.crop(frame, box)
        else:
            cv2.putText(display, "No face detected", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display, "SPACE = capture   q = skip",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and crop is not None and crop.size > 0:
            return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        if key == ord("q"):
            return None


def save_image(img, out_dir, person_id, mode, idx):
    fname = f"{person_id}-TD-{mode}-{idx}.jpg"
    path = os.path.join(out_dir, fname)
    if os.path.exists(path):
        resp = input(f"  {fname} already exists. Overwrite? [y/N]: ").strip().lower()
        if resp != "y":
            print("  Skipped.")
            return
    cv2.imwrite(path, img)
    print(f"  Saved {fname}")


def run_angle_session(camera, detector, out_dir, person_id):
    print("\n-- Angle shots (9 poses) -- vary head angle slightly between captures --")
    for idx in range(9):
        print(f"\nPose {idx + 1}/9")
        img = capture_one(camera, detector, "Collect: Angle Shots")
        if img is None:
            print("  Stopped early.")
            break
        save_image(img, out_dir, person_id, "A", idx)


def run_expression_session(camera, detector, out_dir, person_id):
    print("\n-- Expression shots (5 expressions) --")
    for i, expr in enumerate(EXPR_NAMES, start=1):
        print(f"\nExpression {i}/5: make a '{expr}' face")
        img = capture_one(camera, detector, "Collect: Expression Shots")
        if img is None:
            print("  Stopped early.")
            break
        save_image(img, out_dir, person_id, "E", i)


def _next_angle_index(out_dir, person_id):
    """Continue numbering from whatever angle-shot frames already exist for this person."""
    pattern = re.compile(rf"^{re.escape(str(person_id))}-TD-A-(\d+)\.jpg$")
    max_idx = -1
    if os.path.isdir(out_dir):
        for fname in os.listdir(out_dir):
            m = pattern.match(fname)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def wait_for_start(camera, detector, window_name, timeout_sec: float = 20.0):
    """
    Live preview so the person can be positioned in frame before recording
    starts. Returns True on SPACE (start), False on 'q' (cancel) or timeout.
    """
    last_frame_time = time.time()
    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            if time.time() - last_frame_time > timeout_sec:
                print(f"  No frame received after {timeout_sec:.0f}s — aborting. "
                      "Check the camera/RTSP connection (see any decode errors above).")
                return False
            continue
        last_frame_time = time.time()

        boxes = detector.detect(frame)
        display = frame.copy()
        if boxes:
            x, y, w, h = boxes[0]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(display, "No face detected", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display, "Position the face in frame. SPACE = start recording, q = cancel",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            return True
        if key == ord("q"):
            print("  Cancelled.")
            return False


def _smooth_box(prev, new, alpha: float = 0.3):
    """Exponential moving average between two (x, y, w, h) boxes."""
    if prev is None:
        return new
    return tuple(int(alpha * n + (1 - alpha) * p) for p, n in zip(prev, new))


def run_video_session(camera, detector, out_dir, person_id, video_dir,
                       duration_sec: float = 15.0, sample_fps: float = 4.0):
    """
    Shows a live preview to position the person, then (on SPACE) records a
    short video (saved to video_dir for reference) while sampling frames
    from it at ~sample_fps, face-cropping each one, and adding them to the
    training set as new angle-shot images.

    The crop box is smoothed across sampled frames (see _smooth_box) rather
    than trusting each frame's raw Haar-cascade detection in isolation —
    without that, normal frame-to-frame detection jitter makes consecutive
    saved frames look inconsistently "zoomed" even when the person barely
    moved, since each one gets cropped to whatever box that frame's
    detection happened to find and then resized to a fixed 128x128.
    """
    print(f"\n-- Video capture ({duration_sec:.0f}s) -- "
          f"slowly turn your head left/right/up/down, and vary expression --")

    window_name = "Collect: Video"
    if not wait_for_start(camera, detector, window_name):
        return

    ok, frame = camera.read()
    while not ok or frame is None:
        ok, frame = camera.read()
    h, w = frame.shape[:2]

    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{person_id}_{int(time.time())}.mp4")
    record_fps = 15.0
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), record_fps, (w, h))

    start_idx = _next_angle_index(out_dir, person_id)
    saved = 0
    frame_interval = max(1, round(record_fps / sample_fps))
    frame_i = 0
    smoothed_box = None
    t0 = time.time()

    while time.time() - t0 < duration_sec:
        ok, frame = camera.read()
        if not ok or frame is None:
            continue
        writer.write(frame)

        remaining = duration_sec - (time.time() - t0)
        display = frame.copy()
        cv2.putText(display, f"Recording... {remaining:0.1f}s left (q = stop early)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display, f"Frames saved: {saved}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if frame_i % frame_interval == 0:
            boxes = detector.detect(frame)
            if boxes:
                smoothed_box = _smooth_box(smoothed_box, boxes[0])
            if smoothed_box is not None:
                crop = detector.crop(frame, smoothed_box)
                if crop.size > 0:
                    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                    fname = f"{person_id}-TD-A-{start_idx + saved}.jpg"
                    cv2.imwrite(os.path.join(out_dir, fname), resized)
                    saved += 1
        frame_i += 1

    writer.release()
    cv2.destroyWindow(window_name)
    print(f"  Saved video: {video_path}")
    print(f"  Extracted {saved} training frames "
          f"({person_id}-TD-A-{start_idx}.jpg .. {person_id}-TD-A-{start_idx + max(saved - 1, 0)}.jpg)")


def main(args):
    os.makedirs(args.data_dir, exist_ok=True)
    camera = ThermalCamera(source=args.source)
    detector = ThermalFaceDetector()
    names = load_names(args.names_path)

    print("=== Thermal Data Collection ===")
    print(f"Saving images into: {args.data_dir}")
    print(f"Saving videos into: {args.video_dir}\n")

    try:
        while True:
            person_id = input("Person ID (number, existing or new; 'q' to quit): ").strip()
            if person_id.lower() == "q":
                break
            if not person_id.isdigit():
                print("  Please enter a numeric ID.")
                continue

            pid_int = int(person_id)
            if pid_int in names:
                print(f"  Adding data for existing person: {names[pid_int]}")
            else:
                name = input(f"  New person. Enter a name for ID {person_id}: ").strip()
                if name:
                    names = set_name(pid_int, name, args.names_path)
                    print(f"  Registered '{name}' as person {person_id}.")

            mode = input("Capture [v]ideo, [a]ngle shots, or [e]xpression shots?: ").strip().lower()
            if mode.startswith("v"):
                run_video_session(camera, detector, args.data_dir, person_id, args.video_dir,
                                   duration_sec=args.duration)
            elif mode.startswith("a"):
                run_angle_session(camera, detector, args.data_dir, person_id)
            elif mode.startswith("e"):
                run_expression_session(camera, detector, args.data_dir, person_id)
            else:
                print("  Please enter 'v', 'a', or 'e'.")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\nDone. Re-run train.py to fold new data into the model:")
    print("  python train.py --epochs 40 --batch_size 32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect live thermal face data")
    parser.add_argument("--source", default=DEFAULT_RTSP_URL)
    parser.add_argument("--data_dir", default="data/thermal-face-128x128")
    parser.add_argument("--video_dir", default="data/videos",
                        help="Where raw recorded videos are kept")
    parser.add_argument("--names_path", default=DEFAULT_NAMES_PATH,
                        help="Where the person_id -> name registry is stored")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Seconds to record per video capture session")
    args = parser.parse_args()
    main(args)
