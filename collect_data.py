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
                        head naturally; frames are auto-sampled and
                        face-cropped into training images. This is the
                        fastest way to build up a dataset for someone.
         [a]ngle shots - 9 manually-posed stills (SPACE to capture each)
         (expression-shot capture is disabled for now -- not a current
         focus, revisit next month; search "expression disabled" below)
    3. Repeat for more people/sessions, or 'q' at the person-ID prompt to exit.

Each capture is logged as a session in data/sessions.jsonl (who, when, which frame
numbers, sunglasses yes/no, a note), so train.py --split_mode session can hold out a
whole recording as a test. Record several separate sessions per person (different
days, distances, with/without sunglasses) -- one session cannot be held out.

Saved images land in data/thermal-face-128x128/ as {id}-TD-A-{n}.jpg or
{id}-TD-E-{1..5}.jpg, matching what train.py's ThermalFaceDataset parses.
Raw videos are also kept in data/videos/ for reference/re-processing.
Person names are stored in data/person_names.json (see person_names.py).
"""

import argparse
import json
import os
import re
import time
from datetime import datetime

import cv2

from camera import ThermalCamera, DEFAULT_RTSP_URL
from face_detector import ThermalFaceDetector, FaceTracker
# from model import EXPR_NAMES  -- expression disabled
from person_names import load_names, set_name, DEFAULT_NAMES_PATH

IMG_SIZE = 128
DISPLAY_SCALE = 1.6  # upscale factor for preview windows only -- captured/saved crops are unaffected
DEFAULT_SESSIONS_PATH = os.path.join("data", "sessions.jsonl")


def log_session(path, entry):
    """Append one capture session (a JSON line) to the session log. train.py can then hold
    out a whole session for validation; see train.py's --split_mode session."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def ask_session_meta():
    """Conditions that change how a face looks to the camera, stored with the session."""
    glasses = input("  Wearing sunglasses/glasses this session? [y/N]: ").strip().lower().startswith("y")
    note = input("  Note (distance, lighting, anything unusual; Enter to skip): ").strip()
    return {"sunglasses": glasses, "note": note}


def _show(window_name, frame):
    display = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                          interpolation=cv2.INTER_LINEAR)
    cv2.imshow(window_name, display)


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
        _show(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and crop is not None and crop.size > 0:
            return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        if key == ord("q"):
            return None


def save_image(img, out_dir, person_id, mode, idx):
    """Returns True if the image was written."""
    fname = f"{person_id}-TD-{mode}-{idx}.jpg"
    path = os.path.join(out_dir, fname)
    if os.path.exists(path):
        resp = input(f"  {fname} already exists. Overwrite? [y/N]: ").strip().lower()
        if resp != "y":
            print("  Skipped.")
            return False
    cv2.imwrite(path, img)
    print(f"  Saved {fname}")
    return True


def run_angle_session(camera, detector, out_dir, person_id):
    """Returns the list of frame numbers saved."""
    print("\n-- Angle shots (9 poses) -- vary head angle slightly between captures --")
    saved = []
    for idx in range(9):
        print(f"\nPose {idx + 1}/9")
        img = capture_one(camera, detector, "Collect: Angle Shots")
        if img is None:
            print("  Stopped early.")
            break
        if save_image(img, out_dir, person_id, "A", idx):
            saved.append(idx)
    return saved


# -- Expression-shot capture disabled (not a current focus) -----------------
# def run_expression_session(camera, detector, out_dir, person_id):
#     print("\n-- Expression shots (5 expressions) --")
#     for i, expr in enumerate(EXPR_NAMES, start=1):
#         print(f"\nExpression {i}/5: make a '{expr}' face")
#         img = capture_one(camera, detector, "Collect: Expression Shots")
#         if img is None:
#             print("  Stopped early.")
#             break
#         save_image(img, out_dir, person_id, "E", i)


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


def wait_for_start(camera, detector, tracker, window_name, timeout_sec: float = 20.0):
    """
    Live preview so the person can be positioned in frame before recording
    starts, feeding detections into `tracker` the whole time. Returns True
    on SPACE (start), False on 'q' (cancel) or timeout. Recording then
    reuses the same tracker, so it starts already locked onto whatever
    position the person could see looked correct in the preview, instead
    of starting cold.
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

        box = tracker.update(detector.detect(frame))
        display = frame.copy()
        if box:
            x, y, w, h = box
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(display, "No face detected", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display, "Position the face in frame. SPACE = start recording, q = cancel",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        _show(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            return True
        if key == ord("q"):
            print("  Cancelled.")
            return False


def run_video_session(camera, detector, out_dir, person_id, video_dir,
                       duration_sec: float = 15.0, sample_fps: float = 4.0):
    """
    Shows a live preview to position the person, then (on SPACE) records a
    short video (saved to video_dir for reference) while sampling frames
    from it at ~sample_fps, face-cropping each one, and adding them to the
    training set as new angle-shot images.

    Uses a FaceTracker (see face_detector.py) rather than trusting each
    frame's raw Haar-cascade detection in isolation — without that, normal
    frame-to-frame detection jitter makes consecutive saved frames look
    inconsistently "zoomed", and occasional wildly wrong detections
    (hairline, chin, background mistaken for a face) drag the crop away
    from the actual face entirely.
    """
    print(f"\n-- Video capture ({duration_sec:.0f}s) -- "
          f"slowly turn your head left/right/up/down, and vary expression --")

    window_name = "Collect: Video"
    tracker = FaceTracker()
    if not wait_for_start(camera, detector, tracker, window_name):
        return None

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
        _show(window_name, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if frame_i % frame_interval == 0:
            boxes = detector.detect(frame)
            box = tracker.update(boxes)
            # Only save when a face was actually found in THIS frame. The tracker keeps
            # returning its last box after the face leaves the frame, and saving that box
            # put crops of empty background into the training set (e.g. 120-TD-A-215..218).
            if boxes and box is not None:
                crop = detector.crop(frame, box)
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
    if not saved:
        return None
    return {"video": os.path.basename(video_path),
            "first_frame": start_idx, "last_frame": start_idx + saved - 1}


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

            mode = input("Capture [v]ideo or [a]ngle shots?: ").strip().lower()
            if not (mode.startswith("v") or mode.startswith("a")):   # expression capture disabled
                print("  Please enter 'v' or 'a'.")
                continue
            meta = ask_session_meta()
            started = datetime.now().isoformat(timespec="seconds")
            if mode.startswith("v"):
                result = run_video_session(camera, detector, args.data_dir, person_id, args.video_dir,
                                            duration_sec=args.duration)
            else:
                frames = run_angle_session(camera, detector, args.data_dir, person_id)
                result = {"frame_list": frames} if frames else None
            # elif mode.startswith("e"):  -- expression capture disabled
            #     run_expression_session(camera, detector, args.data_dir, person_id)
            if result:
                entry = {"session_id": f"{person_id}_{started}", "person_id": int(person_id),
                         "mode": "video" if mode.startswith("v") else "angle",
                         "started": started, **meta, **result}
                log_session(args.sessions_path, entry)
                print(f"  Logged session {entry['session_id']} in {args.sessions_path}")
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
    parser.add_argument("--sessions_path", default=DEFAULT_SESSIONS_PATH,
                        help="Log of capture sessions (who, when, which frames, sunglasses)")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Seconds to record per video capture session")
    args = parser.parse_args()
    main(args)
