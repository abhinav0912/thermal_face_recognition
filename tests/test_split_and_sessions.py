import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train
from collect_data import log_session


def fake_dataset(per_person):
    """per_person: {pid: [(mode, frame), ...]} -> object shaped like ThermalFaceDataset."""
    pids = sorted(per_person)
    ds = types.SimpleNamespace(samples=[], frame_index=[], frame_mode=[], person_ids=pids)
    for label, pid in enumerate(pids):
        for mode, frame in per_person[pid]:
            ds.samples.append((f"{pid}-TD-{mode}-{frame}.jpg", label, -1))
            ds.frame_index.append(frame)
            ds.frame_mode.append(mode)
    return ds


def frames_of(ds, idxs, pid):
    label = ds.person_ids.index(pid)
    return sorted(ds.frame_index[i] for i in idxs if ds.samples[i][1] == label)


class SessionSplitTest(unittest.TestCase):
    def setUp(self):
        self.ds = fake_dataset({
            120: [("A", f) for f in range(20)],      # legacy 0-9, session B 10-14, session C 15-19
            121: [("A", f) for f in range(10)],      # only legacy data
        })
        self.sessions = {120: [
            {"session_id": "120_b", "person_id": 120, "started": "2026-11-05T10:00:00",
             "first_frame": 10, "last_frame": 14},
            {"session_id": "120_c", "person_id": 120, "started": "2026-11-09T10:00:00",
             "frame_list": [15, 16, 17, 18, 19]},
        ]}

    def test_newest_session_is_held_out_whole(self):
        tr, va = train.split_indices(self.ds, 0.2, "session", sessions=self.sessions)
        self.assertEqual(frames_of(self.ds, va, 120), [15, 16, 17, 18, 19])
        self.assertEqual(frames_of(self.ds, tr, 120), list(range(15)))
        self.assertFalse(set(tr) & set(va))

    def test_person_with_one_session_falls_back_to_block_split(self):
        tr, va = train.split_indices(self.ds, 0.2, "session", sessions=self.sessions)
        self.assertEqual(frames_of(self.ds, va, 121), [8, 9])   # last 20% of 10 frames

    def test_no_log_means_block_split_for_everyone(self):
        tr, va = train.split_indices(self.ds, 0.25, "session", sessions={})
        self.assertEqual(frames_of(self.ds, va, 120), list(range(15, 20)))

    def test_expression_stills_are_never_mistaken_for_a_logged_session(self):
        mine = [{"session_id": "s", "person_id": 120, "started": "2026-11-05T10:00:00",
                 "first_frame": 10, "last_frame": 14}]
        self.assertEqual(train._session_of(mine, "A", 12), ("2026-11-05T10:00:00", "s"))
        self.assertEqual(train._session_of(mine, "E", 12), ("", "legacy"))   # same number, but an expression still
        self.assertEqual(train._session_of(mine, "A", 3), ("", "legacy"))    # before the log existed

    def test_block_and_random_splits_do_not_overlap(self):
        for mode in ("block", "random"):
            tr, va = train.split_indices(self.ds, 0.2, mode)
            self.assertFalse(set(tr) & set(va))
            self.assertEqual(len(tr) + len(va), len(self.ds.samples))


class SessionLogTest(unittest.TestCase):
    def test_log_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "sessions.jsonl")
            log_session(path, {"session_id": "120_a", "person_id": 120, "started": "2026-11-05T10:00:00",
                               "first_frame": 0, "last_frame": 4, "sunglasses": True})
            log_session(path, {"session_id": "121_a", "person_id": 121, "started": "2026-11-05T11:00:00",
                               "frame_list": [0, 1]})
            loaded = train.load_sessions(path)
            self.assertEqual(sorted(loaded), [120, 121])
            self.assertTrue(loaded[120][0]["sunglasses"])

    def test_missing_log_is_empty(self):
        self.assertEqual(train.load_sessions("no/such/file.jsonl"), {})


if __name__ == "__main__":
    unittest.main()
