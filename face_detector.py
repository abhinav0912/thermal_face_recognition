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
    def crop(frame_bgr, box, margin: float = 0.2, bottom_margin: float = 0.45):
        """
        Crop a box out of the frame with extra margin, clipped to frame bounds.

        The bottom gets a bigger margin than the other three sides by
        default: haarcascade_frontalface_default.xml systematically stops
        short of the chin/jawline (it was trained on face crops annotated
        that way), so an equal margin on all sides still cuts the chin off
        — padding the bottom specifically compensates without unnecessarily
        widening the sides/top too.
        """
        h_img, w_img = frame_bgr.shape[:2]
        x, y, w, h = box
        mx = int(w * margin)
        my_top = int(h * margin)
        my_bottom = int(h * bottom_margin)
        x0 = max(0, x - mx)
        y0 = max(0, y - my_top)
        x1 = min(w_img, x + w + mx)
        y1 = min(h_img, y + h + my_bottom)
        return frame_bgr[y0:y1, x0:x1]


class FaceTracker:
    """
    Temporally stabilizes a ThermalFaceDetector's raw per-frame detections
    for one primary face, instead of trusting each frame's raw detection in
    isolation. Two problems this fixes:

      1. Normal jitter: even on a genuine face, the cascade's box wobbles a
         bit frame to frame (thermal contrast noise, tiny pose shifts) —
         smoothed via an exponential moving average.
      2. Outright false positives: the cascade occasionally locks onto a
         region that isn't a face at all (a warm patch of skin/clothing
         happens to match the cascade's pattern), which is not "jitter" —
         it's a different target entirely, and blending it into the average
         would drag the tracked box away from the real face. Such jumps are
         rejected outright, unless several land in a row (self.reject_streak
         reaches reject_streak_limit), in which case the tracker assumes its
         own lock is the one that's wrong and snaps to the new detection.

    Used by both collect_data.py (so saved training frames stay consistently
    framed) and live_inference.py (so a single bad frame doesn't produce a
    confident identity guess on a garbage crop — see run_video_session /
    live_inference.py's run() for how each wires this in).
    """

    def __init__(self, alpha: float = 0.3, max_center_frac: float = 0.6,
                 reject_streak_limit: int = 3):
        self.alpha = alpha
        self.max_center_frac = max_center_frac
        self.reject_streak_limit = reject_streak_limit
        self.box = None
        self.reject_streak = 0

    def update(self, boxes):
        """
        boxes: this frame's detector.detect() output (largest first), or [].
        Returns the current tracked box (x, y, w, h), or None if no face has
        ever been locked onto yet.
        """
        if not boxes:
            return self.box

        candidate = boxes[0]
        if self.box is None or self._is_plausible(candidate):
            self.box = self._smooth(self.box, candidate)
            self.reject_streak = 0
        else:
            self.reject_streak += 1
            if self.reject_streak >= self.reject_streak_limit:
                self.box = candidate
                self.reject_streak = 0
        return self.box

    def reset(self):
        """Start tracking fresh (e.g. at the beginning of a new session)."""
        self.box = None
        self.reject_streak = 0

    def _smooth(self, prev, new):
        if prev is None:
            return new
        return tuple(int(self.alpha * n + (1 - self.alpha) * p) for p, n in zip(prev, new))

    def _is_plausible(self, candidate):
        scale = max(self.box[2], self.box[3])
        return self._center_distance(self.box, candidate) <= self.max_center_frac * scale

    @staticmethod
    def _center_distance(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        acx, acy = ax + aw / 2, ay + ah / 2
        bcx, bcy = bx + bw / 2, by + bh / 2
        return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


class MultiFaceTracker:
    """
    Tracks several faces at once, each as its own independent FaceTracker
    (smoothed box, outlier rejection, stuck-lock recovery — see FaceTracker
    above for why that matters). Every frame, each detected box is assigned
    to whichever existing track is closest to it (within max_center_frac of
    that track's own scale); anything left over starts a new track, and any
    track that goes unmatched for `max_missed_frames` in a row is dropped
    (the person left frame, or stopped being detected).

    Used by live_inference.py so multiple people in frame each get their
    own stable box and identity, rather than one tracker only following a
    single person. collect_data.py still uses the plain single-target
    FaceTracker directly — enrollment/capture is inherently one person at a
    time, so there's nothing for multi-target tracking to do there.
    """

    def __init__(self, max_missed_frames: int = 10, max_center_frac: float = 0.6,
                 alpha: float = 0.3, reject_streak_limit: int = 3):
        self.max_missed_frames = max_missed_frames
        self.max_center_frac = max_center_frac
        self._tracker_kwargs = dict(alpha=alpha, max_center_frac=max_center_frac,
                                     reject_streak_limit=reject_streak_limit)
        self._next_id = 0
        self.tracks = {}   # track_id -> {"tracker": FaceTracker, "missed": int}

    def update(self, boxes):
        """
        boxes: this frame's detector.detect() output (any order), or [].
        Returns {track_id: box} for every currently active track (boxes
        that weren't matched this frame keep showing their last known
        position until max_missed_frames is exceeded).
        """
        unmatched = list(boxes)
        matched_ids = set()

        for track_id, track in self.tracks.items():
            current_box = track["tracker"].box
            if current_box is None or not unmatched:
                continue
            scale = max(current_box[2], current_box[3])
            best_idx, best_dist = None, None
            for i, cand in enumerate(unmatched):
                dist = FaceTracker._center_distance(current_box, cand)
                if dist <= self.max_center_frac * scale and (best_dist is None or dist < best_dist):
                    best_idx, best_dist = i, dist
            if best_idx is not None:
                matched_box = unmatched.pop(best_idx)
                track["tracker"].update([matched_box])
                track["missed"] = 0
                matched_ids.add(track_id)

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_ids:
                self.tracks[track_id]["missed"] += 1
                if self.tracks[track_id]["missed"] > self.max_missed_frames:
                    del self.tracks[track_id]

        for cand in unmatched:
            tracker = FaceTracker(**self._tracker_kwargs)
            tracker.update([cand])
            self.tracks[self._next_id] = {"tracker": tracker, "missed": 0}
            self._next_id += 1

        return {tid: t["tracker"].box for tid, t in self.tracks.items()
                if t["tracker"].box is not None}
