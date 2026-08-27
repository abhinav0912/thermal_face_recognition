"""
face_detector.py
=================
Locates a face in a raw camera frame and crops it, bridging the gap between
a live frame (whole scene) and what DualHeadFaceNet expects (a pre-cropped
face image, matching the training data).

Uses OpenCV's bundled Haar cascade on a CLAHE-contrast-enhanced frame.
Thermal frames tend to be low-contrast, which hurts the Haar cascade's
gradient-based features; CLAHE (adaptive histogram equalization) makes
detection noticeably more reliable than running the cascade raw.

This is a pragmatic starting point, not a thermal-specific detector. If face
detection is unreliable on real footage from the A50, the fix is collecting
labeled thermal frames and training a dedicated detector — not tuning the
cascade parameters further.
"""

import cv2


class ThermalFaceDetector:
    def __init__(self, min_size=(60, 60), scale_factor: float = 1.1, min_neighbors: int = 8):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, frame_bgr):
        """Return a list of (x, y, w, h) boxes, largest (closest) face first."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        faces = self.cascade.detectMultiScale(
            enhanced,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        return [tuple(map(int, b)) for b in faces]

    @staticmethod
    def crop(frame_bgr, box, margin: float = 0.2):
        """Crop a box out of the frame with extra margin, clipped to frame bounds."""
        h_img, w_img = frame_bgr.shape[:2]
        x, y, w, h = box
        mx, my = int(w * margin), int(h * margin)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(w_img, x + w + mx)
        y1 = min(h_img, y + h + my)
        return frame_bgr[y0:y1, x0:x1]
