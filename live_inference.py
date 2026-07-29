"""
live_inference.py
==================
Real-time thermal face recognition from a live camera feed.

Usage:
    # Default: FLIR A50 over RTSP
    python live_inference.py

    # Explicit source
    python live_inference.py --source rtsp://169.254.0.82:554/avc

    # Test the pipeline with a regular webcam instead of the thermal camera
    python live_inference.py --source 0

Press 'q' in the video window to quit.
"""

import argparse
import time

import cv2
from PIL import Image

from camera import ThermalCamera, DEFAULT_RTSP_URL
from face_detector import ThermalFaceDetector
from inference import FaceRecognizer

WINDOW_NAME = "Thermal Face Recognition"


def annotate(frame, box, person_label, expr_label):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, person_label, (x, max(20, y - 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, expr_label, (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)


def run(args):
    print("Loading model ...")
    recognizer = FaceRecognizer(args.checkpoint_dir, device=args.device)
    detector = ThermalFaceDetector()
    camera = ThermalCamera(source=args.source)

    print("Starting live feed. Press 'q' to quit.\n")
    fps_t0 = time.time()
    frame_count = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("  No frame received, retrying ...")
                time.sleep(0.5)
                continue

            boxes = detector.detect(frame)
            for box in boxes[:args.max_faces]:
                crop = detector.crop(frame, box)
                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                result = recognizer.predict_image(pil_img, top_k=1)

                if result["person_conf"] < args.unknown_threshold:
                    person_label = "Unknown"
                else:
                    person_label = f"{result['person_name']} ({result['person_conf']*100:.0f}%)"
                expr_label = f"{result['expression']} ({result['expr_conf']*100:.0f}%)"

                annotate(frame, box, person_label, expr_label)

            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - fps_t0)
                cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} - {fps:.1f} FPS")

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live thermal face recognition")
    parser.add_argument("--source", default=DEFAULT_RTSP_URL,
                        help="RTSP URL, or a webcam index (e.g. 0) for testing")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--max_faces", type=int, default=3,
                        help="Max number of faces to recognize per frame")
    parser.add_argument("--unknown_threshold", type=float, default=0.5,
                        help="Below this person-confidence, label as 'Unknown' "
                             "instead of forcing a match to a known identity")
    parser.add_argument("--device", default=None,
                        help="Force 'cpu' or 'cuda'. Defaults to cuda if available. "
                             "Use --device cpu to work around a GPU that PyTorch "
                             "doesn't have kernels for yet (e.g. very new hardware).")
    args = parser.parse_args()
    run(args)
