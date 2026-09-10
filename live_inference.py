"""
live_inference.py
==================
Real-time thermal face recognition from a live camera feed.

Tracks one primary face at a time (via FaceTracker, see face_detector.py)
rather than classifying whatever the Haar cascade finds on every single
frame — that stops one glitchy frame from producing a confidently-labeled
box on something that isn't actually a face. If multiple people are ever in
frame simultaneously, only the primary tracked face gets recognized.

Identification is a two-stage fallback:
  1. The trained classifier (checkpoints/best_model.pth) — used whenever
     it's confident. Full accuracy, but only knows people it was trained on.
  2. gallery.py's live-enrolled gallery — checked only when the classifier
     ISN'T confident. Press 'n' during the live feed to enroll whoever is
     currently in frame with no retraining and no restart: a few frames are
     captured on the spot, averaged into an embedding, and they're
     recognizable within seconds. Weaker than a trained classifier entry,
     but far better than "not recognized until the next retrain".

Usage:
    # Default: FLIR A50 over RTSP
    python live_inference.py

    # Explicit source
    python live_inference.py --source rtsp://169.254.0.82:554/avc

    # Test the pipeline with a regular webcam instead of the thermal camera
    python live_inference.py --source 0

Press 'n' to enroll the person currently in frame, 'q' to quit.
"""

import argparse
import time

import cv2
from PIL import Image

import gallery
from camera import ThermalCamera, DEFAULT_RTSP_URL
from face_detector import ThermalFaceDetector, FaceTracker
from inference import FaceRecognizer

WINDOW_NAME = "Thermal Face Recognition"


def annotate(frame, box, person_label, expr_label):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, person_label, (x, max(20, y - 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, expr_label, (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)


def enroll_new_person(camera, detector, tracker, recognizer, gallery_path,
                       num_frames: int = 8):
    """
    Captures a few frames of whoever the tracker currently has locked onto,
    averages their embeddings, and saves them to the gallery under a name
    typed into the console. Returns the updated gallery dict, or None if
    enrollment was cancelled/failed.
    """
    name = input("\n  Enter name for new person (blank to cancel): ").strip()
    if not name:
        print("  Enrollment cancelled.")
        return None

    print(f"  Enrolling '{name}' — stay facing the camera ...")
    embeddings = []
    attempts = 0
    max_attempts = num_frames * 6

    while len(embeddings) < num_frames and attempts < max_attempts:
        ok, frame = camera.read()
        attempts += 1
        cv2.waitKey(1)  # keep the window responsive while this runs
        if not ok or frame is None:
            continue

        box = tracker.update(detector.detect(frame))
        if box is None:
            continue
        crop = detector.crop(frame, box)
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        embeddings.append(recognizer.embed_image(pil_img))
        print(f"    Captured {len(embeddings)}/{num_frames}")

    if len(embeddings) < 3:
        print("  Not enough usable frames — make sure a face is clearly "
              "visible and try again ('n').")
        return None

    updated = gallery.enroll(name, embeddings, gallery_path)
    print(f"  Enrolled '{name}' from {len(embeddings)} frames. "
          f"They should be recognized live now.\n")
    return updated


def run(args):
    print("Loading model ...")
    recognizer = FaceRecognizer(args.checkpoint_dir, device=args.device)
    detector = ThermalFaceDetector()
    tracker = FaceTracker()
    camera = ThermalCamera(source=args.source)
    gallery_data = gallery.load_gallery(args.gallery_path)

    print("Starting live feed. Press 'n' to enroll a new person, 'q' to quit.\n")
    fps_t0 = time.time()
    frame_count = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("  No frame received, retrying ...")
                time.sleep(0.5)
                continue

            # Tracked, not raw per-frame detection: a single glitchy frame
            # (thermal texture on skin/clothing that coincidentally matches
            # the Haar cascade's pattern) would otherwise produce a
            # confidently-labeled box on something that isn't a face at all.
            box = tracker.update(detector.detect(frame))
            if box is not None:
                crop = detector.crop(frame, box)
                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(crop_rgb)
                    result = recognizer.predict_image(pil_img, top_k=1)

                    if result["person_conf"] >= args.unknown_threshold:
                        person_label = f"{result['person_name']} ({result['person_conf']*100:.0f}%)"
                    else:
                        # The trained classifier isn't confident -- fall back
                        # to the live-enrolled gallery before giving up.
                        embedding = recognizer.embed_image(pil_img)
                        gallery_name, sim = gallery.match(
                            embedding, gallery_data, threshold=args.gallery_threshold)
                        if gallery_name:
                            person_label = f"{gallery_name} ({sim*100:.0f}% live-match)"
                        else:
                            person_label = "Unknown"
                    expr_label = f"{result['expression']} ({result['expr_conf']*100:.0f}%)"

                    annotate(frame, box, person_label, expr_label)

            cv2.putText(frame, "n = enroll new person   q = quit", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - fps_t0)
                cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} - {fps:.1f} FPS")

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                updated = enroll_new_person(camera, detector, tracker, recognizer,
                                             args.gallery_path)
                if updated is not None:
                    gallery_data = updated
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live thermal face recognition")
    parser.add_argument("--source", default=DEFAULT_RTSP_URL,
                        help="RTSP URL, or a webcam index (e.g. 0) for testing")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--unknown_threshold", type=float, default=0.5,
                        help="Below this person-confidence, label as 'Unknown' "
                             "instead of forcing a match to a known identity")
    parser.add_argument("--gallery_path", default=gallery.DEFAULT_GALLERY_PATH,
                        help="Where live-enrolled people are stored")
    parser.add_argument("--gallery_threshold", type=float, default=0.6,
                        help="Minimum cosine similarity to accept a gallery match")
    parser.add_argument("--device", default=None,
                        help="Force 'cpu' or 'cuda'. Defaults to cuda if available. "
                             "Use --device cpu to work around a GPU that PyTorch "
                             "doesn't have kernels for yet (e.g. very new hardware).")
    args = parser.parse_args()
    run(args)
