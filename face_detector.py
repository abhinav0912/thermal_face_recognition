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
detection is unreliable on real footage from the A50, the eventual fix is
collecting labeled thermal frames and training a dedicated detector.

Detection runs in two passes rather than one fixed-sensitivity pass. The
cascade's default settings (min_neighbors=8) want a near-perfectly frontal,
well-contrasted face -- confirmed by direct testing: a person clearly on
camera but turned slightly produced zero detections at these settings, while
a relaxed pass on the same frame found them immediately. In a multi-person
frame this bites unevenly -- whoever is more frontal/better-contrasted still
gets a clean strict-pass box, while anyone at a worse angle or with a
different thermal signature can be missed entirely rather than just
detected-with-lower-confidence (Haar cascades don't expose a confidence
score to threshold on -- min_neighbors IS the sensitivity knob, and it's
frame-wide, not adjustable per-face). So: run the strict pass first, then a
second, more permissive pass, and only add relaxed-pass boxes that don't
already overlap a strict-pass box -- a face the strict pass found keeps its
clean box as-is, and only a face it missed entirely gets pulled in via the
fallback, instead of loosening detection for every face all the time.

The fallback pass alone isn't enough, though: tested directly against 12
frames captured a second apart from real footage, min_neighbors is a noisy
knob on this camera's footage -- no single value reliably caught the one
real (but off-angle) face in every frame without ALSO starting to flag
featureless patches of background. min_neighbors and relaxed-pass boxes are
kept fairly permissive as a result, and a second, more reliable signal is
used to separate real faces from background noise: local grey-value std
within the candidate box. Across that same test set, every real face had
std >= 40 (thermal edges/features -- hairline, glasses, nose bridge) while
every background false positive had std <= 11 (flat, textureless regions
that only coincidentally satisfied the cascade's gradient pattern). A
threshold of 20 sits with wide margin on both sides of that observed split.
This filter is applied only to the fallback pass's *extra* boxes, not the
strict pass -- strict produced zero false positives in that same test, so
there's nothing to gain by filtering it, and doing so would risk rejecting
a genuine strict detection on an unusually uniform-temperature face.

One more duplicate source needed fixing after the above shipped: caught live
against real footage, MultiFaceTracker would occasionally show 2-3 boxes on
what was clearly one person, all carrying the same predicted name. Traced to
detect() itself -- OpenCV's detectMultiScale can return more than one box for
a single real face within ONE call (it fires at nearby scales/positions that
don't always fully collapse via minNeighbors), and the original dedup only
ever compared relaxed-pass boxes against strict-pass boxes, never checked for
duplicates *within* a single pass's own output. Once detect() hands the
tracker two boxes for one face in the same frame, MultiFaceTracker has no way
to know they're the same person -- it spawns a second, independent track that
never gets reconciled with the first.

Fixed with a final dedup pass over ALL candidate boxes together (both passes,
not just relaxed-vs-strict), using overlap-coefficient (intersection / the
SMALLER box's area) instead of IoU. IoU understates exactly the case that
came up here: a 155x155 box and a 55x55 box both centered on the same face,
one almost fully nested inside the other, scored IoU=0.13 (reads as "probably
different") but overlap-coefficient=1.0 (correctly "same face, one just
tighter than the other") -- IoU's union term is dominated by the big box, so
it heavily discounts a smaller box even when that smaller box is entirely
contained in the bigger one.

One more thing surfaced by extended live running: a single shared
cv2.CascadeClassifier, called every frame with two DIFFERENT parameter sets
(strict then relaxed), can crash with an internal OpenCV assertion --
"0 <= scaleIdx && scaleIdx < (int)scaleData->size()" -- deep in its C++
feature evaluator. That evaluator caches a scale pyramid keyed to whatever
scaleFactor/minSize a call used; a classifier is normally called with fixed
parameters for its whole lifetime, and flip-flopping between two different
parameter sets on the SAME instance, every frame, isn't a pattern OpenCV's
cascade code seems to expect. Fixed two ways: each pass now gets its own
classifier instance (so neither ever sees a parameter change between calls),
and detectMultiScale calls are wrapped in try/except regardless -- detection
is inherently best-effort already (a frame with zero boxes is a normal,
already-handled outcome), so a caught error is treated the same way instead
of taking down the whole live feed.
"""

import cv2


class ThermalFaceDetector:
    def __init__(self, min_size=(60, 60), scale_factor: float = 1.1, min_neighbors: int = 8,
                 relaxed_min_size=(50, 50), relaxed_scale_factor: float = 1.05,
                 relaxed_min_neighbors: int = 4, relaxed_min_std: float = 20.0,
                 duplicate_overlap_threshold: float = 0.5):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        # Separate instances per pass -- see module docstring: one shared classifier
        # called with two different parameter sets every frame could crash OpenCV's
        # internal scale-pyramid cache.
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.relaxed_cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty() or self.relaxed_cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        # Fallback pass: only fills in faces the strict pass missed entirely (see module docstring).
        self.relaxed_min_size = relaxed_min_size
        self.relaxed_scale_factor = relaxed_scale_factor
        self.relaxed_min_neighbors = relaxed_min_neighbors
        self.relaxed_min_std = relaxed_min_std
        self.duplicate_overlap_threshold = duplicate_overlap_threshold

    def detect(self, frame_bgr):
        """Return a list of (x, y, w, h) boxes, one per face, largest (closest) first."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)

        strict = self._raw_detect(self.cascade, enhanced,
                                   self.scale_factor, self.min_neighbors, self.min_size)
        relaxed = self._raw_detect(self.relaxed_cascade, enhanced, self.relaxed_scale_factor,
                                    self.relaxed_min_neighbors, self.relaxed_min_size)
        relaxed = [b for b in relaxed if self._looks_like_a_face(gray, b)]

        # Largest first, so the dedup pass below keeps the bigger/cleaner box of any pair.
        candidates = sorted(strict + relaxed, key=lambda b: b[2] * b[3], reverse=True)
        faces = self._suppress_duplicates(candidates)
        return [tuple(map(int, b)) for b in faces]

    def _looks_like_a_face(self, gray, box):
        """Reject flat/featureless fallback-pass boxes (background) -- see module docstring
        for the std >= 40 (real) vs std <= 11 (background) split this threshold is based on."""
        x, y, w, h = box
        roi = gray[y:y + h, x:x + w]
        return roi.size > 0 and roi.std() >= self.relaxed_min_std

    @staticmethod
    def _raw_detect(cascade, enhanced, scale_factor, min_neighbors, min_size):
        try:
            faces = cascade.detectMultiScale(
                enhanced,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
            )
        except cv2.error:
            # Transient OpenCV-internal failure (see module docstring) -- treat this pass as
            # having found nothing this frame rather than taking down the whole live feed.
            # MultiFaceTracker already handles "no boxes this frame" as a normal outcome.
            return []
        return [tuple(map(int, b)) for b in faces]

    def _suppress_duplicates(self, boxes):
        """Greedy dedup over every candidate box from both passes -- see module docstring for
        why this runs on the combined list instead of just relaxed-vs-strict, and why it uses
        overlap-coefficient instead of IoU."""
        kept = []
        for box in boxes:
            if all(self._overlap_coefficient(box, k) < self.duplicate_overlap_threshold
                   for k in kept):
                kept.append(box)
        return kept

    @staticmethod
    def _overlap_coefficient(a, b):
        """Intersection over the SMALLER box's area (Szymkiewicz-Simpson coefficient) --
        1.0 when one box is fully nested inside the other, regardless of the size gap
        between them. See module docstring for why plain IoU misses nested duplicates."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        return inter / min(aw * ah, bw * bh)

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
