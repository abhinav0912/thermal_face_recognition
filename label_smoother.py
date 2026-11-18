"""
label_smoother.py
=================
Turns each frame's raw model output into a name decision, and steadies that
decision over time.

Why: live, the classifier's answer for one person changes from frame to frame
(measured 2026-11-02: one person was named Chandhini 160, Lakshmi 85, Asher 40
and the right name 20 times in 305 frames), so the label on screen flickered
between people. Showing the answer for each frame on its own is also the worst
case for a closed-set classifier: it always names somebody, however unsure it
is. LabelSmoother keeps the last few answers per tracked face and only shows a
name when enough of them agree; otherwise it shows "Unknown".

What this can and cannot do: it removes flicker and turns "split between two
people" into "Unknown". It cannot fix a model that is consistently wrong about
someone: if the model says Asher 80% of the time, the smoothed label is a
steady "Asher".

Standard library plus gallery.py (numpy) only, so it can be tested on its own.
"""

from collections import Counter, deque
from typing import NamedTuple, Optional

import gallery


class Decision(NamedTuple):
    name: Optional[str]      # None = Unknown
    score: float             # classifier confidence, or gallery cosine similarity (0-1)
    source: Optional[str]    # "classifier", "gallery" or None


UNKNOWN = Decision(None, 0.0, None)


def decide(result, embedding, gallery_data, unknown_threshold, gallery_threshold) -> Decision:
    """One face, one frame: trained classifier if it is confident, else the live-enrolled
    gallery, else Unknown. `result` and `embedding` come from FaceRecognizer.predict_batch."""
    if result["person_conf"] >= unknown_threshold:
        return Decision(result["person_name"], result["person_conf"], "classifier")
    gallery_name, sim = gallery.match(embedding, gallery_data, threshold=gallery_threshold)
    if gallery_name:
        return Decision(gallery_name, sim, "gallery")
    return UNKNOWN


def format_label(decision: Decision):
    """(text to draw, known?) -- the same strings the live views used before."""
    if decision.name is None:
        return "Unknown", False
    if decision.source == "gallery":
        return f"{decision.name} ({decision.score * 100:.0f}% live-match)", True
    return f"{decision.name} ({decision.score * 100:.0f}%)", True


class LabelSmoother:
    """
    Per tracked face, remembers the last `window` raw decisions and returns a steadier one.

    - A name is shown once at least `min_frames` decisions exist and that name has at least
      `min_share` of the window. Until then, or when no name has enough votes, the result
      is Unknown.
    - Once a name is shown it is kept while it still has at least `keep_share` of the
      window, unless a different name reaches `switch_share` (hysteresis, so two close
      names don't swap every other frame).
    - The score shown is the mean score of that name's votes in the window.

    Defaults are first guesses, not tuned values: window=15 is about 0.5-1 s at the
    15-30 fps this pipeline runs at.
    """

    def __init__(self, window: int = 15, min_frames: int = 6, min_share: float = 0.6,
                 keep_share: float = 0.5, switch_share: float = 0.75):
        self.window = window
        self.min_frames = min_frames
        self.min_share = min_share
        self.keep_share = keep_share
        self.switch_share = switch_share
        self._history = {}   # track_id -> deque[Decision]
        self._current = {}   # track_id -> name currently shown (or None)

    def reset(self):
        self._history.clear()
        self._current.clear()

    def update(self, track_ids, decisions):
        """track_ids and decisions are parallel lists for this frame; returns the
        smoothed decisions in the same order. State for tracks that are gone is dropped."""
        live = set(track_ids)
        for tid in [t for t in self._history if t not in live]:
            del self._history[tid]
            self._current.pop(tid, None)
        return [self._update_one(tid, d) for tid, d in zip(track_ids, decisions)]

    def _update_one(self, tid, decision):
        hist = self._history.setdefault(tid, deque(maxlen=self.window))
        hist.append(decision)
        if len(hist) < self.min_frames:
            return UNKNOWN

        n = len(hist)
        counts = Counter(d.name for d in hist if d.name is not None)
        current = self._current.get(tid)
        winner, winner_count = counts.most_common(1)[0] if counts else (None, 0)

        shown = None
        if current is not None and counts.get(current, 0) / n >= self.keep_share:
            shown = current
            if winner != current and winner_count / n >= self.switch_share:
                shown = winner
        elif winner is not None and winner_count / n >= self.min_share:
            shown = winner
        self._current[tid] = shown
        if shown is None:
            return UNKNOWN

        votes = [d for d in hist if d.name == shown]
        latest = votes[-1]
        return Decision(shown, sum(d.score for d in votes) / len(votes), latest.source)
