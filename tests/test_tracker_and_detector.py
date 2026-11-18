import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_detector import MultiFaceTracker, ThermalFaceDetector

FACE = (200, 100, 150, 150)


class TrackerTest(unittest.TestCase):
    def test_new_face_is_shown_on_the_fifth_matching_frame(self):
        t = MultiFaceTracker()
        shown = [bool(t.update([FACE])) for _ in range(8)]
        self.assertEqual(shown, [False] * 4 + [True] * 4)

    def test_min_hits_1_is_the_old_behaviour(self):
        self.assertTrue(MultiFaceTracker(min_hits=1).update([FACE]))

    def test_flickering_false_box_is_never_shown(self):
        t = MultiFaceTracker()
        shown = [bool(t.update([(300, 300, 64, 64)] if i % 12 == 0 else [])) for i in range(240)]
        self.assertFalse(any(shown))

    def test_face_detected_in_most_frames_is_confirmed(self):
        pattern = [True, True, False, True, True, True, False, True, True, True]   # 70%
        t = MultiFaceTracker()
        shown = [bool(t.update([FACE] if hit else [])) for hit in pattern * 3]
        self.assertTrue(shown[-1])

    def test_confirmed_face_survives_short_gaps_then_drops(self):
        t = MultiFaceTracker()
        for _ in range(10):
            t.update([FACE])
        counts = [len(t.update([])) for _ in range(12)]
        self.assertEqual(counts[:10], [1] * 10)
        self.assertEqual(counts[10:], [0, 0])


class FaceCheckTest(unittest.TestCase):
    def setUp(self):
        self.det = ThermalFaceDetector()
        rng = np.random.default_rng(0)
        self.flat = np.full((200, 200), 110, np.uint8)                 # blank wall
        self.busy = rng.integers(0, 255, (200, 200), dtype=np.uint8)   # lots of contrast
        self.box = (50, 50, 80, 80)

    def check(self, gray, **kw):
        return self.det._looks_like_a_face(gray, self.box, float(gray.mean()), **kw)

    def test_flat_box_is_rejected_and_textured_box_accepted(self):
        self.assertFalse(self.check(self.flat, min_std=20.0))
        self.assertTrue(self.check(self.busy, min_std=20.0))

    def test_box_much_darker_than_the_frame_is_rejected(self):
        gray = self.busy.copy()
        gray[:, :] = np.clip(gray, 150, 255)                           # bright frame
        gray[50:130, 50:130] = np.random.default_rng(1).integers(0, 90, (80, 80), dtype=np.uint8)
        self.assertFalse(self.check(gray, min_std=20.0))               # dark, but textured
        self.det.min_rel_brightness = None                             # rule off
        self.assertTrue(self.check(gray, min_std=20.0))

    def test_box_partly_outside_the_frame_does_not_crash(self):
        self.assertFalse(self.det._looks_like_a_face(self.flat, (-20, -20, 60, 60), 110.0, 20.0))


if __name__ == "__main__":
    unittest.main()
