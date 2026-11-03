"""RecognitionStreamer end to end with a fake camera, detector and identifier (no hardware,
no model): checks the frame loop produces JPEGs, applies the smoother, and stays alive."""
import itertools
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_detector import MultiFaceTracker
from label_smoother import Decision
from stream_server import RecognitionStreamer


class FakeCamera:
    def __init__(self):
        self.frame_id = 0
        self._frame = np.zeros((120, 160, 3), np.uint8)

    def read(self):
        self.frame_id += 1
        return True, self._frame.copy()


class FakeDetector:
    def detect(self, frame):
        return [(40, 30, 60, 60)]

    @staticmethod
    def crop(frame, box):
        x, y, w, h = box
        return frame[y:y + h, x:x + w]


class StreamerTest(unittest.TestCase):
    def run_streamer(self, smoothing, names):
        drawn = []
        feed = itertools.cycle(names)

        def identify(recognizer, gallery_data, crops, ut, gt):
            n = next(feed)
            return [Decision(n, 0.9, "classifier") for _ in crops]

        def annotate(frame, box, label, known):
            drawn.append(label)

        s = RecognitionStreamer(FakeDetector(), MultiFaceTracker(), identify, annotate,
                                lambda: {}, "no-such-gallery.json")
        s.smoothing_enabled = smoothing
        s.configure(FakeCamera(), recognizer=None, unknown_threshold=0.5, gallery_threshold=0.6)
        s.set_active(True)
        deadline = time.time() + 5
        while len(drawn) < 40 and time.time() < deadline:
            time.sleep(0.02)
        s.set_active(False)
        return s, drawn

    def test_loop_runs_encodes_frames_and_reports_no_error(self):
        s, drawn = self.run_streamer(True, ["Ann"] * 50)
        self.assertGreaterEqual(len(drawn), 20)
        self.assertEqual(s.stats()["error"], "")
        jpeg, _ = s.wait_for_frame(-1, 0)
        self.assertTrue(jpeg and jpeg[:2] == b"\xff\xd8")

    def test_smoothing_replaces_flicker_with_unknown(self):
        _, drawn = self.run_streamer(True, ["Ann", "Bob"] * 30)
        self.assertEqual(set(drawn), {"Unknown"})

    def test_smoothing_off_shows_each_raw_answer(self):
        _, drawn = self.run_streamer(False, ["Ann", "Bob"] * 30)
        self.assertTrue(any(label.startswith("Ann") for label in drawn))
        self.assertTrue(any(label.startswith("Bob") for label in drawn))


if __name__ == "__main__":
    unittest.main()
