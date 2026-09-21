"""Tracking: identity persistence, which is what every event is keyed to."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from backend.inference.base import Detection
from backend.tracking.bytetrack import ByteTracker, TrackState, _hungarian
from backend.tracking.kalman import xyah_to_xyxy, xyxy_to_xyah


def box(x: float, y: float, w: float = 60, h: float = 160) -> Detection:
    return Detection(bbox=(x, y, x + w, y + h), confidence=0.9, class_id=0, label="person")


# ── assignment ────────────────────────────────────────────────────────────
def test_hungarian_matches_brute_force_optimum():
    """The solver must be exact, not merely greedy."""
    rng = np.random.default_rng(3)
    for _ in range(120):
        rows, cols = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        cost = rng.random((rows, cols))
        r, c = _hungarian(cost)
        got = float(cost[r, c].sum())
        k = min(rows, cols)
        if rows <= cols:
            best = min(
                sum(cost[i, p[i]] for i in range(k))
                for p in itertools.permutations(range(cols), k)
            )
        else:
            best = min(
                sum(cost[p[i], i] for i in range(k))
                for p in itertools.permutations(range(rows), k)
            )
        assert got == pytest.approx(float(best), abs=1e-9)
        assert len(r) == k


def test_hungarian_handles_empty():
    r, c = _hungarian(np.zeros((0, 3)))
    assert len(r) == 0 and len(c) == 0


# ── kalman round trip ─────────────────────────────────────────────────────
def test_xyah_round_trip():
    original = (100.0, 50.0, 160.0, 210.0)
    assert xyah_to_xyxy(xyxy_to_xyah(original)) == pytest.approx(original, abs=1e-3)


# ── identity ──────────────────────────────────────────────────────────────
def test_ids_are_stable_across_many_frames(mock_detector, blank_frame):
    """The requirement from §5: no new ID every frame."""
    tracker = ByteTracker(min_hits=3)
    seen: set[int] = set()
    for index in range(150):
        result = mock_detector.infer(blank_frame)
        seen.update(t.track_id for t in tracker.update(result.detections, timestamp=index / 12))
    assert len(seen) == 3, f"expected 3 stable identities, got {sorted(seen)}"


def test_identity_survives_occlusion(mock_detector, blank_frame):
    """A person who reappears must keep their ID, or their event history breaks."""
    tracker = ByteTracker(min_hits=2, max_age=30)
    for index in range(20):
        tracker.update(mock_detector.infer(blank_frame).detections, timestamp=index / 12)
    before = sorted(t.track_id for t in tracker.active_tracks)

    for index in range(20, 35):  # fully occluded
        tracker.update([], timestamp=index / 12)
    for index in range(35, 45):
        tracker.update(mock_detector.infer(blank_frame).detections, timestamp=index / 12)

    assert sorted(t.track_id for t in tracker.active_tracks) == before


def test_track_retires_after_max_age():
    tracker = ByteTracker(min_hits=1, max_age=5)
    tracker.update([box(100, 100)], timestamp=0.0)
    assert len(tracker.active_tracks) == 1
    for i in range(1, 10):
        tracker.update([], timestamp=i * 0.1)
    assert tracker.active_tracks == []
    assert all(t.state is not TrackState.TRACKED for t in tracker.tracks)


def test_min_hits_delays_confirmation():
    tracker = ByteTracker(min_hits=3)
    for i in range(2):
        assert tracker.update([box(100 + i * 3, 100)], timestamp=i * 0.1) == []
    assert len(tracker.update([box(106, 100)], timestamp=0.2)) == 1


def test_two_people_get_two_ids():
    tracker = ByteTracker(min_hits=1)
    tracks = tracker.update([box(100, 100), box(600, 100)], timestamp=0.0)
    assert len({t.track_id for t in tracks}) == 2


def test_non_person_labels_ignored():
    tracker = ByteTracker(min_hits=1, target_labels=("person",))
    car = Detection(bbox=(0, 0, 50, 50), confidence=0.95, class_id=2, label="car")
    assert tracker.update([car], timestamp=0.0) == []


def test_history_is_bounded():
    """Unbounded history would leak memory on a long-lived track."""
    tracker = ByteTracker(min_hits=1)
    for i in range(400):
        tracker.update([box(100 + i * 0.4, 100)], timestamp=i * 0.05)
    for track in tracker.tracks:
        assert len(track.history) <= track.max_history


def test_reset_clears_state():
    tracker = ByteTracker(min_hits=1)
    tracker.update([box(10, 10)], timestamp=0.0)
    tracker.reset()
    assert tracker.tracks == [] and tracker.frame_index == 0
    assert tracker.update([box(10, 10)], timestamp=0.0)[0].track_id == 1
