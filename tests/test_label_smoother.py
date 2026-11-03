import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_smoother import Decision, LabelSmoother, UNKNOWN, decide, format_label


def D(name, score=0.9, source="classifier"):
    return Decision(name, score, source)


def run(smoother, per_frame_names, tid=0):
    """Feed one track a sequence of raw names (None = Unknown); return the shown names."""
    return [smoother.update([tid], [D(n) if n else UNKNOWN])[0].name for n in per_frame_names]


class LabelSmootherTest(unittest.TestCase):
    def test_steady_name_is_shown_after_min_frames(self):
        shown = run(LabelSmoother(min_frames=5), ["Ann"] * 10)
        self.assertEqual(shown[:4], [None] * 4)          # not enough frames yet
        self.assertEqual(shown[4:], ["Ann"] * 6)

    def test_flicker_between_two_people_is_unknown_not_a_swap(self):
        shown = run(LabelSmoother(), ["Ann", "Bob"] * 10)
        self.assertTrue(all(n is None for n in shown))   # 50/50 never reaches min_share

    def test_single_wrong_frame_does_not_change_the_label(self):
        shown = run(LabelSmoother(), ["Ann"] * 10 + ["Bob"] + ["Ann"] * 5)
        self.assertEqual(set(shown[5:]), {"Ann"})

    def test_hysteresis_keeps_name_until_other_name_dominates(self):
        s = LabelSmoother(window=10, min_frames=5, min_share=0.6, keep_share=0.4, switch_share=0.75)
        # (keep_share lowered here on purpose, to exercise the hysteresis path with a 6/4 split)
        run(s, ["Ann"] * 10)
        shown = run(s, ["Bob"] * 4)                      # window now 6 Ann / 4 Bob
        self.assertEqual(shown[-1], "Ann")
        shown = run(s, ["Bob"] * 4)                      # 2 Ann / 8 Bob -> switch
        self.assertEqual(shown[-1], "Bob")

    def test_tracks_are_independent_and_cleaned_up(self):
        s = LabelSmoother(min_frames=3)
        for _ in range(5):
            out = s.update([1, 2], [D("Ann"), D("Bob")])
        self.assertEqual([d.name for d in out], ["Ann", "Bob"])
        s.update([2], [D("Bob")])                        # track 1 is gone
        self.assertNotIn(1, s._history)

    def test_score_is_mean_of_the_winning_votes(self):
        s = LabelSmoother(min_frames=3)
        for sc in (0.6, 0.8, 1.0):
            out = s.update([0], [D("Ann", sc)])[0]
        self.assertAlmostEqual(out.score, 0.8)

    def test_reset_forgets_everything(self):
        s = LabelSmoother()
        run(s, ["Ann"] * 10)
        s.reset()
        self.assertIsNone(run(s, ["Ann"])[0])


class DecideAndFormatTest(unittest.TestCase):
    def test_confident_classifier_wins(self):
        d = decide({"person_conf": 0.9, "person_name": "Ann"}, None, {}, 0.5, 0.6)
        self.assertEqual(d, Decision("Ann", 0.9, "classifier"))

    def test_low_confidence_without_gallery_is_unknown(self):
        d = decide({"person_conf": 0.2, "person_name": "Ann"}, [0.0] * 4, {}, 0.5, 0.6)
        self.assertEqual(d, UNKNOWN)

    def test_format_matches_the_old_label_text(self):
        self.assertEqual(format_label(D("Ann", 0.953)), ("Ann (95%)", True))
        self.assertEqual(format_label(D("Bob", 0.71, "gallery")), ("Bob (71% live-match)", True))
        self.assertEqual(format_label(UNKNOWN), ("Unknown", False))


if __name__ == "__main__":
    unittest.main()
