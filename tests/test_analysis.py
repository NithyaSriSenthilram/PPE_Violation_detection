"""Behaviour rules: zones, PPE, loitering, movement, falls, crowd."""

from __future__ import annotations

import numpy as np
import pytest

from backend.analysis.base import FrameContext
from backend.analysis.crowd import CrowdAnalyser
from backend.analysis.fall import FallAnalyser
from backend.analysis.intrusion import IntrusionAnalyser
from backend.analysis.loitering import LoiteringAnalyser
from backend.analysis.movement import MovementAnalyser, measure_speed
from backend.analysis.ppe_rules import PPEAnalyser
from backend.analysis.zones import (
    ResolvedZone,
    contains_points,
    validate_polygon,
    zones_for_points,
)
from backend.events.types import EventType
from backend.inference.ppe import HeuristicPPEEstimator, PPEAssessment
from backend.tracking.bytetrack import Track, TrackState

WIDTH, HEIGHT = 960, 540


def make_zone(zone_type: str = "restricted", **kwargs) -> ResolvedZone:
    return ResolvedZone(
        zone_id=kwargs.pop("zone_id", "z1"),
        name=kwargs.pop("name", "Zone A"),
        zone_type=zone_type,
        polygon_norm=kwargs.pop("polygon", [[0.4, 0.4], [0.9, 0.4], [0.9, 0.9], [0.4, 0.9]]),
        **kwargs,
    )


def make_track(track_id: int, bbox, timestamp: float, history=None) -> Track:
    track = Track(
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        state=TrackState.TRACKED,
    )
    for sample in history or []:
        track.history.append(sample)
    track.record(timestamp)
    return track


def context(tracks, zones, index=0, timestamp=0.0, ppe=None) -> FrameContext:
    return FrameContext(
        camera_id="cam",
        frame_index=index,
        timestamp=timestamp,
        wall_time=timestamp,
        frame_width=WIDTH,
        frame_height=HEIGHT,
        tracks=tracks,
        zones=zones,
        ppe=ppe or {},
    )


# ══════════════════════════════════════════════════════════════════════════
#  Zone geometry
# ══════════════════════════════════════════════════════════════════════════
class TestZones:
    def test_point_inside_and_outside(self):
        zone = make_zone()
        assert zone.contains(700, 400, WIDTH, HEIGHT)
        assert not zone.contains(50, 50, WIDTH, HEIGHT)

    def test_boundary_and_area(self):
        zone = make_zone(polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert zone.area_fraction() == pytest.approx(1.0)
        assert zone.contains(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT)

    def test_concave_polygon(self):
        """Ray casting must handle a non-convex zone, e.g. an L-shaped bay."""
        zone = make_zone(
            polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.4], [0.4, 0.4], [0.4, 0.9], [0.1, 0.9]]
        )
        assert zone.contains(0.2 * WIDTH, 0.8 * HEIGHT, WIDTH, HEIGHT)
        assert not zone.contains(0.8 * WIDTH, 0.8 * HEIGHT, WIDTH, HEIGHT)

    def test_batched_matches_individual(self):
        zone = make_zone()
        polygon = zone.pixel_polygon(WIDTH, HEIGHT)
        points = np.array([[700, 400], [50, 50], [500, 500]], dtype=np.float32)
        batched = contains_points(points, polygon)
        for index, point in enumerate(points):
            assert bool(batched[index]) == zone.contains(*point, WIDTH, HEIGHT)

    def test_zones_for_points_multi_membership(self):
        a = make_zone(zone_id="a", polygon=[[0.0, 0.0], [0.6, 0.0], [0.6, 1.0], [0.0, 1.0]])
        b = make_zone(zone_id="b", polygon=[[0.4, 0.0], [1.0, 0.0], [1.0, 1.0], [0.4, 1.0]])
        points = np.array([[0.5 * WIDTH, 0.5 * HEIGHT]], dtype=np.float32)
        assert set(zones_for_points(points, [a, b], WIDTH, HEIGHT)[0]) == {"a", "b"}

    def test_disabled_zone_ignored(self):
        zone = make_zone(enabled=False)
        points = np.array([[700, 400]], dtype=np.float32)
        assert zones_for_points(points, [zone], WIDTH, HEIGHT) == [[]]

    @pytest.mark.parametrize(
        "polygon,ok",
        [
            ([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], True),
            ([[0.1, 0.1], [0.9, 0.1]], False),                     # too few points
            ([[0.1, 0.1], [0.2, 0.1], [0.3, 0.1]], False),         # collinear
            ([[0.1, 0.1], [1.5, 0.1], [0.5, 0.9]], False),         # not normalised
        ],
    )
    def test_validate_polygon(self, polygon, ok):
        assert validate_polygon(polygon)[0] is ok


# ══════════════════════════════════════════════════════════════════════════
#  Intrusion
# ══════════════════════════════════════════════════════════════════════════
class TestIntrusion:
    def test_fires_once_on_entry(self):
        analyser = IntrusionAnalyser()
        zone = make_zone()
        events = []
        for i in range(15):
            x = 100 + i * 55
            track = make_track(1, (x, 300, x + 60, 480), i * 0.1)
            events += analyser.analyse(context([track], [zone], i, i * 0.1))
        assert len(events) == 1
        assert events[0].event_type == EventType.RESTRICTED_AREA
        assert events[0].person_id == 1
        assert events[0].zone_id == "z1"

    def test_uses_foot_point_not_centroid(self):
        """Zone membership is about where someone is *standing*.

        A zone covering the upper half of frame is the discriminating case: a
        tall person's centroid falls inside it while their feet are below it.
        A centroid test would fire; a foot-point test correctly does not.
        """
        analyser = IntrusionAnalyser(confirm_frames=1)
        zone = make_zone(polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]])

        # centroid y = 250 (inside the band), feet y = 400 (below it)
        standing_below = make_track(1, (400, 100, 460, 400), 0.0)
        assert analyser.analyse(context([standing_below], [zone], 0, 0.0)) == []

        # feet y = 260, genuinely inside the band
        standing_inside = make_track(2, (400, 60, 460, 260), 0.0)
        assert len(analyser.analyse(context([standing_inside], [zone], 1, 0.1))) == 1

    def test_re_entry_fires_again(self):
        analyser = IntrusionAnalyser(confirm_frames=1)
        zone = make_zone()
        inside = make_track(1, (700, 300, 760, 480), 0.0)
        outside = make_track(1, (50, 300, 110, 480), 0.0)
        assert len(analyser.analyse(context([inside], [zone], 0, 0.0))) == 1
        assert analyser.analyse(context([outside], [zone], 1, 0.1)) == []
        assert len(analyser.analyse(context([inside], [zone], 2, 0.2))) == 1

    def test_non_restricted_zone_type_ignored(self):
        analyser = IntrusionAnalyser(confirm_frames=1)
        zone = make_zone(zone_type="monitored")
        track = make_track(1, (700, 300, 760, 480), 0.0)
        assert analyser.analyse(context([track], [zone], 0, 0.0)) == []


# ══════════════════════════════════════════════════════════════════════════
#  Loitering
# ══════════════════════════════════════════════════════════════════════════
class TestLoitering:
    def test_fires_once_at_threshold(self):
        analyser = LoiteringAnalyser(default_threshold=3.0)
        zone = make_zone()
        events = []
        for i in range(80):
            t = i * 0.1
            track = make_track(1, (700, 300, 760, 480), t)
            events += analyser.analyse(context([track], [zone], i, t))
        assert len(events) == 1
        assert events[0].event_type == EventType.LOITERING
        assert events[0].metadata["dwell_seconds"] >= 3.0

    def test_zone_threshold_overrides_global(self):
        analyser = LoiteringAnalyser(default_threshold=60.0)
        zone = make_zone(loitering_threshold=1.0)
        events = []
        for i in range(30):
            t = i * 0.1
            events += analyser.analyse(
                context([make_track(1, (700, 300, 760, 480), t)], [zone], i, t)
            )
        assert len(events) == 1

    def test_dwell_measured_in_seconds_not_frames(self):
        """The threshold must mean the same at any frame rate."""
        zone = make_zone()
        for fps in (5.0, 25.0):
            analyser = LoiteringAnalyser(default_threshold=2.0)
            events = []
            for i in range(int(fps * 4)):
                t = i / fps
                events += analyser.analyse(
                    context([make_track(1, (700, 300, 760, 480), t)], [zone], i, t)
                )
            assert len(events) == 1, f"failed at {fps} fps"

    def test_leaving_resets_dwell(self):
        analyser = LoiteringAnalyser(default_threshold=2.0)
        zone = make_zone()
        for i in range(10):  # 1.0s inside
            t = i * 0.1
            analyser.analyse(context([make_track(1, (700, 300, 760, 480), t)], [zone], i, t))
        # Away long enough to clear the grace period.
        for i in range(10, 60):
            t = i * 0.1
            analyser.analyse(context([make_track(1, (50, 300, 110, 480), t)], [zone], i, t))
        assert analyser.dwell_seconds(1, "z1") == 0.0


# ══════════════════════════════════════════════════════════════════════════
#  Movement
# ══════════════════════════════════════════════════════════════════════════
class TestMovement:
    @staticmethod
    def _run(pixels_per_frame: float, threshold: float = 2.2, frames: int = 20):
        analyser = MovementAnalyser(threshold=threshold)
        events, history = [], []
        for i in range(frames):
            t = i * 0.1
            x = 100 + i * pixels_per_frame
            track = Track(
                track_id=1, bbox=(x, 300, x + 72, 480), confidence=0.9,
                state=TrackState.TRACKED,
            )
            track.history = list(history)
            track.record(t)
            history = track.history
            events += analyser.analyse(context([track], [], i, t))
        return events

    def test_fast_movement_alerts(self):
        events = self._run(90)  # ~5 body-heights/s
        assert len(events) == 1
        assert events[0].event_type == EventType.ABNORMAL_MOVEMENT
        assert events[0].metadata["speed_body_heights_per_second"] > 2.2

    def test_walking_does_not_alert(self):
        assert self._run(12) == []

    def test_stationary_does_not_alert(self):
        assert self._run(0) == []

    def test_speed_normalised_by_body_height(self):
        """The same real-world speed must read the same near and far from the lens."""
        near = Track(track_id=1, bbox=(0, 0, 100, 400), confidence=0.9)
        far = Track(track_id=2, bbox=(0, 0, 25, 100), confidence=0.9)
        # Each moves one body-height per second.
        near.history = [(0.0, 0.0, 200.0, 400.0), (1.0, 400.0, 200.0, 400.0)]
        far.history = [(0.0, 0.0, 50.0, 100.0), (1.0, 100.0, 50.0, 100.0)]
        assert measure_speed(near, window=1.0) == pytest.approx(
            measure_speed(far, window=1.0), rel=0.01
        )

    def test_insufficient_history_is_zero(self):
        assert measure_speed(Track(track_id=1, bbox=(0, 0, 10, 10), confidence=0.9)) == 0.0


# ══════════════════════════════════════════════════════════════════════════
#  Fall
# ══════════════════════════════════════════════════════════════════════════
class TestFall:
    @staticmethod
    def _sequence(analyser, steps):
        events, history = [], []
        for i, bbox in enumerate(steps):
            t = i * 0.1
            track = Track(track_id=1, bbox=bbox, confidence=0.9, state=TrackState.TRACKED)
            track.history = list(history)
            track.record(t)
            history = track.history
            events += analyser.analyse(context([track], [], i, t))
        return events

    def test_full_fall_sequence_detected(self):
        analyser = FallAnalyser(aspect_threshold=1.35, drop_fraction=0.28, confirm_seconds=0.5)
        steps = (
            [(400, 250, 472, 430)] * 8                                    # upright
            + [(400, 250 + k * 28, 472, 430 + k * 28) for k in range(4)]  # drop
            + [(380, 400, 560, 460)] * 13                                 # horizontal
        )
        events = self._sequence(analyser, steps)
        assert len(events) == 1
        assert events[0].event_type == EventType.POSSIBLE_FALL
        assert "advisory" in events[0].metadata

    def test_crouching_is_not_a_fall(self):
        """Getting shorter without becoming horizontal must not fire."""
        analyser = FallAnalyser(confirm_seconds=0.3)
        steps = [(400, 250, 472, 430)] * 8 + [(400, 340, 472, 430)] * 20
        assert self._sequence(analyser, steps) == []

    def test_horizontal_without_drop_is_not_a_fall(self):
        """Someone already lying down when first seen is not a detected fall."""
        analyser = FallAnalyser(confirm_seconds=0.3)
        assert self._sequence(analyser, [(380, 400, 560, 460)] * 25) == []

    def test_brief_horizontal_does_not_confirm(self):
        analyser = FallAnalyser(confirm_seconds=2.0)
        steps = (
            [(400, 250, 472, 430)] * 8
            + [(400, 250 + k * 28, 472, 430 + k * 28) for k in range(4)]
            + [(380, 400, 560, 460)] * 3   # only 0.3s horizontal
            + [(400, 250, 472, 430)] * 5   # stands back up
        )
        assert self._sequence(analyser, steps) == []


# ══════════════════════════════════════════════════════════════════════════
#  Crowd
# ══════════════════════════════════════════════════════════════════════════
class TestCrowd:
    def test_absolute_threshold(self):
        analyser = CrowdAnalyser(default_threshold=15)
        events = []
        for i in range(30):
            count = 5 if i < 20 else 25
            tracks = [
                make_track(j, (50 + j * 30, 300, 80 + j * 30, 450), i * 0.2)
                for j in range(count)
            ]
            events += analyser.analyse(context(tracks, [], i, i * 0.2))
        assert len(events) == 1
        assert events[0].event_type == EventType.CROWD_ANOMALY
        assert events[0].metadata["people_count"] == 25
        assert events[0].metadata["trigger"] == "absolute_threshold"

    def test_below_threshold_is_quiet(self):
        analyser = CrowdAnalyser(default_threshold=15)
        for i in range(40):
            tracks = [make_track(j, (50 + j * 30, 300, 80 + j * 30, 450), i * 0.2) for j in range(4)]
            assert analyser.analyse(context(tracks, [], i, i * 0.2)) == []

    def test_crowd_zone_scopes_the_count(self):
        """Only people inside the crowd zone should count toward it."""
        analyser = CrowdAnalyser(default_threshold=3)
        zone = make_zone(zone_type="crowd", polygon=[[0.0, 0.0], [0.3, 0.0], [0.3, 1.0], [0.0, 1.0]])
        # Two inside the left band, four outside it.
        tracks = [make_track(j, (20 + j * 30, 300, 50 + j * 30, 450), 0.0) for j in range(2)]
        tracks += [make_track(10 + j, (700 + j * 20, 300, 730 + j * 20, 450), 0.0) for j in range(4)]
        assert analyser.analyse(context(tracks, [zone], 0, 0.0)) == []


# ══════════════════════════════════════════════════════════════════════════
#  PPE
# ══════════════════════════════════════════════════════════════════════════
class TestPPERules:
    @staticmethod
    def _feed(analyser, assessment, frames=10):
        events = []
        for i in range(frames):
            track = make_track(1, (400, 100, 480, 420), i * 0.1)
            events += analyser.analyse(
                context([track], [], i, i * 0.1, ppe={1: assessment})
            )
        return events

    def test_undetermined_is_not_a_violation(self):
        """`None` means the detector could not tell — it must never alert."""
        analyser = PPEAnalyser(min_consecutive=2)
        assessment = PPEAssessment(person_id=1, helmet=None, vest=None, method="heuristic")
        assert self._feed(analyser, assessment) == []

    def test_missing_helmet_fires_once(self):
        analyser = PPEAnalyser(min_consecutive=3)
        assessment = PPEAssessment(
            person_id=1, helmet=False, vest=True,
            helmet_confidence=0.6, method="model",
        )
        events = self._feed(analyser, assessment)
        assert len(events) == 1
        assert events[0].event_type == EventType.MISSING_HELMET

    def test_both_missing_raises_one_aggregate_event(self):
        """One situation, one alert — not two."""
        analyser = PPEAnalyser(min_consecutive=2)
        assessment = PPEAssessment(
            person_id=1, helmet=False, vest=False,
            helmet_confidence=0.6, vest_confidence=0.6, method="model",
        )
        events = self._feed(analyser, assessment)
        assert len(events) == 1
        assert events[0].event_type == EventType.PPE_VIOLATION
        assert set(events[0].metadata["missing"]) == {"helmet", "vest"}

    def test_transient_finding_is_suppressed(self):
        """A single bad frame must not raise an alert."""
        analyser = PPEAnalyser(min_consecutive=5)
        bad = PPEAssessment(person_id=1, helmet=False, vest=True, helmet_confidence=0.6)
        good = PPEAssessment(person_id=1, helmet=True, vest=True, helmet_confidence=0.6)
        events = []
        for i, assessment in enumerate([good, good, bad, good, good, good]):
            track = make_track(1, (400, 100, 480, 420), i * 0.1)
            events += analyser.analyse(context([track], [], i, i * 0.1, ppe={1: assessment}))
        assert events == []

    def test_method_is_propagated(self):
        """The UI must be able to mark a heuristic reading as advisory."""
        analyser = PPEAnalyser(min_consecutive=1)
        assessment = PPEAssessment(
            person_id=1, helmet=False, vest=True, helmet_confidence=0.55, method="heuristic"
        )
        events = self._feed(analyser, assessment, frames=1)
        assert events[0].metadata["ppe_method"] == "heuristic"


class TestHeuristicPPE:
    """The colour estimator itself. It is a fallback, but it must be correct."""

    @staticmethod
    def _scene(head_bgr, torso_bgr):
        import cv2

        frame = np.full((540, 960, 3), 60, np.uint8)
        cv2.rectangle(frame, (315, 105), (385, 175), head_bgr, -1)
        cv2.rectangle(frame, (305, 190), (395, 300), torso_bgr, -1)
        track = Track(track_id=1, bbox=(300, 100, 400, 420), confidence=0.9)
        return HeuristicPPEEstimator().assess(frame, [track], [])[1]

    def test_detects_yellow_helmet_and_orange_vest(self):
        assessment = self._scene((0, 210, 240), (20, 120, 255))
        assert assessment.helmet is True
        assert assessment.vest is True
        assert not assessment.is_violation

    def test_detects_bare_head_and_plain_shirt(self):
        assessment = self._scene((90, 110, 140), (70, 70, 75))
        assert assessment.helmet is False
        assert assessment.vest is False
        assert set(assessment.violations) == {"helmet", "vest"}

    def test_confidence_is_capped(self):
        """A colour test must never present itself as certain."""
        from backend.inference.ppe import HEURISTIC_CONFIDENCE_CEILING

        assessment = self._scene((90, 110, 140), (70, 70, 75))
        assert assessment.confidence <= HEURISTIC_CONFIDENCE_CEILING

    def test_small_person_is_undetermined(self):
        """Too few pixels to judge — say so rather than guess."""
        frame = np.full((540, 960, 3), 60, np.uint8)
        tiny = Track(track_id=1, bbox=(100, 100, 120, 160), confidence=0.9)
        assessment = HeuristicPPEEstimator().assess(frame, [tiny], [])[1]
        assert assessment.helmet is None and assessment.vest is None
        assert "skipped" in assessment.detail

    def test_method_always_labelled(self):
        assert self._scene((0, 210, 240), (20, 120, 255)).method == "heuristic"
