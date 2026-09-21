"""PPE detection: model loading, class mapping, association, rules, evidence.

These tests never touch real model weights. Everything below the ONNX session
is exercised with mock detections — which is the point of the split between
:class:`ModelPPEDetector` (association, class interpretation, absence
inference) and the :class:`Detector` that produces the boxes: the interesting
logic is testable without a 38 MB file, and the tests stay fast and offline.

One test does load the real model, and is skipped when it is not installed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from backend.analysis.base import FrameContext
from backend.analysis.ppe_rules import PPEAnalyser
from backend.config import settings
from backend.events.engine import CooldownRegistry, EventEngine
from backend.events.types import EventType
from backend.inference.base import Detection, InferenceResult
from backend.inference.ppe import (
    HEURISTIC_CONFIDENCE_CEILING,
    INFERRED_ABSENCE_CONFIDENCE,
    HeuristicPPEEstimator,
    ModelPPEDetector,
    NullPPEDetector,
    PPEAssessment,
    _threshold_for,
    associate_items,
    ppe_diagnostics,
    reset_ppe_detector,
    resolve_ppe_detector,
)
from backend.inference.ppe_taxonomy import (
    ClassMapError,
    ClassRole,
    PPEClassMap,
    normalise,
    parse_overrides,
)
from backend.tracking import Track, TrackState

WIDTH, HEIGHT = 960, 540


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════
def make_track(track_id: int, bbox=(100, 100, 200, 400)) -> Track:
    track = Track(
        track_id=track_id, bbox=bbox, confidence=0.9, state=TrackState.TRACKED
    )
    track.record(0.0)
    return track


def detection(label: str, bbox, confidence: float = 0.9) -> Detection:
    return Detection(bbox=bbox, confidence=confidence, class_id=0, label=label)


class FakeDetector:
    """Stands in for a loaded PPE model, returning scripted detections.

    Mirrors only what :meth:`ModelPPEDetector.assess` uses, which is the whole
    :class:`Detector` surface it depends on: `infer`, `name`, `is_loaded`.
    """

    name = "fake"

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.is_loaded = True
        self.calls = 0

    def infer(self, frame):
        self.calls += 1
        return InferenceResult(detections=list(self.detections), backend=self.name)

    def stats(self):
        return {"backend": self.name, "loaded": True}

    def close(self):
        self.is_loaded = False


def model_detector(
    detections: list[Detection], labels: list[str], overrides: str = ""
) -> ModelPPEDetector:
    """A ModelPPEDetector wired to scripted detections and a class list."""
    detector = ModelPPEDetector(detector=FakeDetector(detections))
    detector.class_map = PPEClassMap.build(labels, overrides)
    return detector


def context(tracks, ppe, index: int = 0, timestamp: float = 0.0) -> FrameContext:
    return FrameContext(
        camera_id="cam",
        frame_index=index,
        timestamp=timestamp,
        wall_time=timestamp,
        frame_width=WIDTH,
        frame_height=HEIGHT,
        tracks=tracks,
        zones=[],
        ppe=ppe,
    )


FULL_LABELS = ["helmet", "no-helmet", "no-vest", "person", "vest"]


# ══════════════════════════════════════════════════════════════════════════
#  Class mapping
# ══════════════════════════════════════════════════════════════════════════
class TestClassMapping:
    def test_maps_the_shipped_model_classes(self):
        mapping = PPEClassMap.build(FULL_LABELS)
        assert mapping.covered_items() == {"helmet", "vest"}
        assert mapping.role_of("no-helmet") == ("ppe", "helmet", False)
        assert mapping.role_of("vest") == ("ppe", "vest", True)
        assert mapping.is_person("person")
        assert mapping.unmapped == []

    @pytest.mark.parametrize(
        ("label", "item", "present"),
        [
            ("Hardhat", "helmet", True),
            ("hard_hat", "helmet", True),
            ("NO-Hardhat", "helmet", False),
            ("head", "helmet", False),
            ("safety-vest", "vest", True),
            ("Safety Vest", "vest", True),
            ("no_vest", "vest", False),
        ],
    )
    def test_naming_conventions_in_the_wild(self, label, item, present):
        """Third-party models name the same thing five different ways."""
        role = PPEClassMap.build([label]).role_of(label)
        assert role == ("ppe", item, present)

    def test_normalise_is_separator_and_case_insensitive(self):
        assert normalise("NO-Hard_Hat") == normalise("no hard hat") == "no hard hat"

    def test_unrecognised_classes_are_reported_not_guessed(self):
        mapping = PPEClassMap.build(["helmet", "circular_saw", "fire_extinguisher"])
        assert mapping.unmapped == ["circular_saw", "fire_extinguisher"]
        assert mapping.covered_items() == {"helmet"}

    def test_overrides_extend_the_vocabulary(self):
        mapping = PPEClassMap.build(
            ["hat", "bare_head", "worker"],
            "hat=helmet:present,bare_head=helmet:absent,worker=person",
        )
        assert mapping.role_of("hat") == ("ppe", "helmet", True)
        assert mapping.role_of("bare_head") == ("ppe", "helmet", False)
        assert mapping.is_person("worker")
        assert mapping.unmapped == []

    def test_future_items_are_already_supported(self):
        """gloves / boots / goggles / mask / harness need no code change."""
        labels = ["glove", "no-glove", "boots", "goggles", "mask", "harness"]
        mapping = PPEClassMap.build(labels)
        assert mapping.covered_items() == {
            "gloves", "boots", "goggles", "mask", "harness",
        }
        assert mapping.has_absence_class("gloves")
        assert not mapping.has_absence_class("boots")

    def test_malformed_override_raises_rather_than_silently_dropping(self):
        with pytest.raises(ClassMapError, match="unknown item"):
            parse_overrides("hat=sunglasses:present")
        with pytest.raises(ClassMapError, match="present"):
            parse_overrides("hat=helmet:maybe")
        with pytest.raises(ClassMapError, match="label=role"):
            parse_overrides("nonsense")

    def test_required_ppe_config_parses_and_validates(self, monkeypatch):
        monkeypatch.setattr(settings, "required_ppe", "helmet, vest ,gloves")
        assert settings.required_ppe_items == ["helmet", "vest", "gloves"]
        monkeypatch.setattr(settings, "required_ppe", "helmet,umbrella,helmet")
        assert settings.required_ppe_items == ["helmet"]


# ══════════════════════════════════════════════════════════════════════════
#  Association
# ══════════════════════════════════════════════════════════════════════════
class TestAssociation:
    #: One person, standing: head at the top, torso in the middle.
    PERSON = np.array([[100, 100, 200, 400]], dtype=np.float32)

    def test_helmet_on_the_head_belongs_to_that_person(self):
        helmet = np.array([[120, 105, 180, 150]], dtype=np.float32)
        owner, score = associate_items(self.PERSON, helmet, ["helmet"])
        assert owner[0] == 0
        assert score[0] > 0.9

    def test_helmet_elsewhere_in_the_frame_belongs_to_nobody(self):
        """A helmet on a bench is not a helmet being worn."""
        bench = np.array([[600, 420, 660, 460]], dtype=np.float32)
        owner, _ = associate_items(self.PERSON, bench, ["helmet"])
        assert owner[0] == -1

    def test_helmet_at_waist_height_is_rejected(self):
        """Inside the box, but nowhere a worn helmet can be."""
        carried = np.array([[120, 300, 170, 340]], dtype=np.float32)
        owner, _ = associate_items(self.PERSON, carried, ["helmet"])
        assert owner[0] == -1

    def test_vest_on_the_torso_is_accepted(self):
        vest = np.array([[105, 160, 195, 280]], dtype=np.float32)
        owner, _ = associate_items(self.PERSON, vest, ["vest"])
        assert owner[0] == 0

    def test_the_nearer_person_wins_when_two_overlap(self):
        """Two people side by side must not share one helmet."""
        people = np.array(
            [[100, 100, 200, 400], [190, 100, 290, 400]], dtype=np.float32
        )
        helmet = np.array([[215, 105, 265, 150]], dtype=np.float32)
        owner, _ = associate_items(people, helmet, ["helmet"])
        assert owner[0] == 1

    def test_threshold_is_configurable(self):
        """A half-overlapping box passes a loose threshold, fails a strict one."""
        half_out = np.array([[160, 105, 260, 150]], dtype=np.float32)
        assert associate_items(self.PERSON, half_out, ["helmet"], 0.3)[0][0] == 0
        assert associate_items(self.PERSON, half_out, ["helmet"], 0.9)[0][0] == -1

    def test_empty_inputs_are_safe(self):
        empty = np.zeros((0, 4), dtype=np.float32)
        assert len(associate_items(empty, empty, [])[0]) == 0
        assert len(associate_items(self.PERSON, empty, [])[0]) == 0
        owner, _ = associate_items(empty, self.PERSON, ["helmet"])
        assert owner.tolist() == [-1]


# ══════════════════════════════════════════════════════════════════════════
#  Model detector
# ══════════════════════════════════════════════════════════════════════════
class TestModelPPEDetector:
    FRAME = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    def test_missing_helmet_is_detected_and_attributed(self):
        track = make_track(1)
        detector = model_detector(
            [
                detection("person", (100, 100, 200, 400)),
                detection("no-helmet", (120, 105, 180, 150), 0.88),
                detection("vest", (105, 160, 195, 280), 0.91),
            ],
            FULL_LABELS,
        )
        result = detector.assess(self.FRAME, [track], [])[1]
        assert result.helmet is False
        assert result.vest is True
        assert result.helmet_confidence == pytest.approx(0.88)
        assert result.method == "model"
        assert result.status("helmet").source == "detected"
        assert result.missing(["helmet", "vest"]) == ["helmet"]

    def test_missing_vest_is_detected(self):
        track = make_track(1)
        detector = model_detector(
            [
                detection("person", (100, 100, 200, 400)),
                detection("helmet", (120, 105, 180, 150), 0.94),
                detection("no-vest", (105, 160, 195, 280), 0.77),
            ],
            FULL_LABELS,
        )
        result = detector.assess(self.FRAME, [track], [])[1]
        assert result.helmet is True
        assert result.vest is False
        assert result.vest_confidence == pytest.approx(0.77)

    def test_fully_compliant_person_raises_nothing(self):
        track = make_track(1)
        detector = model_detector(
            [
                detection("person", (100, 100, 200, 400)),
                detection("helmet", (120, 105, 180, 150), 0.94),
                detection("vest", (105, 160, 195, 280), 0.9),
            ],
            FULL_LABELS,
        )
        result = detector.assess(self.FRAME, [track], [])[1]
        assert result.helmet is True and result.vest is True
        assert result.violations == []
        assert result.is_violation is False

    def test_ppe_on_another_person_is_not_credited(self):
        """The core failure mode: one helmet must not clear the whole frame."""
        wearing = make_track(1, (100, 100, 200, 400))
        bare = make_track(2, (600, 100, 700, 400))
        detector = model_detector(
            [
                detection("person", (100, 100, 200, 400)),
                detection("person", (600, 100, 700, 400)),
                detection("helmet", (120, 105, 180, 150), 0.94),
                detection("no-helmet", (620, 105, 680, 150), 0.86),
            ],
            FULL_LABELS,
        )
        results = detector.assess(self.FRAME, [wearing, bare], [])
        assert results[1].helmet is True
        assert results[2].helmet is False

    def test_two_people_get_independent_verdicts(self):
        compliant = make_track(1, (100, 100, 200, 400))
        violator = make_track(2, (600, 100, 700, 400))
        detector = model_detector(
            [
                detection("person", (100, 100, 200, 400)),
                detection("person", (600, 100, 700, 400)),
                detection("helmet", (120, 105, 180, 150)),
                detection("vest", (105, 160, 195, 280)),
                detection("no-helmet", (620, 105, 680, 150)),
                detection("no-vest", (605, 160, 695, 280)),
            ],
            FULL_LABELS,
        )
        results = detector.assess(self.FRAME, [compliant, violator], [])
        assert results[1].violations == []
        assert set(results[2].missing(["helmet", "vest"])) == {"helmet", "vest"}

    def test_unseen_item_stays_undetermined_not_absent(self):
        """Silence from a model with absence classes means 'unknown'."""
        track = make_track(1)
        detector = model_detector(
            [detection("person", (100, 100, 200, 400))], FULL_LABELS
        )
        result = detector.assess(self.FRAME, [track], [])[1]
        assert result.helmet is None
        assert result.vest is None
        assert result.is_violation is False

    def test_positive_only_model_infers_absence_at_lower_confidence(self):
        """A model with no `no_helmet` class can only imply the absence."""
        track = make_track(1)
        detector = model_detector(
            [detection("person", (100, 100, 200, 400))], ["person", "helmet", "vest"]
        )
        result = detector.assess(self.FRAME, [track], [])[1]
        assert result.helmet is False
        assert result.helmet_confidence == pytest.approx(INFERRED_ABSENCE_CONFIDENCE)
        assert result.status("helmet").source == "inferred"

    def test_absence_is_not_inferred_for_a_person_the_model_never_saw(self):
        """No detection at all is not evidence of a bare head."""
        track = make_track(1, (600, 100, 700, 400))
        detector = model_detector(
            [detection("person", (100, 100, 200, 400))], ["person", "helmet", "vest"]
        )
        result = detector.assess(self.FRAME, [track], [])[1]
        assert result.helmet is None
        assert result.detail["model_saw_person"] is False

    def test_small_person_is_not_assessed(self):
        tiny = make_track(1, (100, 100, 130, 140))
        detector = model_detector(
            [detection("person", (100, 100, 130, 140))], ["person", "helmet", "vest"]
        )
        result = detector.assess(self.FRAME, [tiny], [])[1]
        assert result.helmet is None
        assert result.detail["skipped"] == "person too small to assess"

    def test_conflicting_detections_resolve_to_the_stronger_one(self):
        track = make_track(1)
        detector = model_detector(
            [
                detection("person", (100, 100, 200, 400)),
                detection("no-helmet", (120, 105, 180, 150), 0.55),
                detection("helmet", (122, 106, 178, 148), 0.93),
            ],
            FULL_LABELS,
        )
        result = detector.assess(self.FRAME, [track], [])[1]
        assert result.helmet is True
        assert result.helmet_confidence == pytest.approx(0.93)

    def test_unmapped_classes_are_ignored(self):
        track = make_track(1)
        detector = model_detector(
            [
                detection("person", (100, 100, 200, 400)),
                detection("circular_saw", (120, 105, 180, 150), 0.99),
            ],
            ["person", "helmet", "circular_saw"],
        )
        result = detector.assess(self.FRAME, [track], [])[1]
        assert result.items.get("circular_saw") is None

    def test_no_tracks_means_no_inference_call(self):
        """PPE inference on a frame with nobody in it is wasted work."""
        fake = FakeDetector([])
        detector = ModelPPEDetector(detector=fake)
        detector.class_map = PPEClassMap.build(FULL_LABELS)
        assert detector.assess(self.FRAME, [], []) == {}
        assert fake.calls == 0


# ══════════════════════════════════════════════════════════════════════════
#  Confidence threshold
# ══════════════════════════════════════════════════════════════════════════
class TestPerRoleThresholds:
    """Presence and absence classes are judged against different bars.

    A PPE model does not score the two symmetrically. On the reference footage
    `no-helmet` never exceeds 0.54 while `vest` reaches 0.86, so a single
    shared bar cannot serve both: set it for presence and the absence class is
    switched off; set it for absence and weak presence detections get through.
    These tests pin the split, because the failure it prevents is silent — the
    class simply stops firing and every person reads as undetermined.
    """

    def test_absence_class_uses_the_absence_bar(self, monkeypatch):
        monkeypatch.setattr(settings, "ppe_confidence_threshold", 0.35)
        monkeypatch.setattr(settings, "ppe_absence_confidence_threshold", 0.25)
        absent = ClassRole(kind="ppe", item="helmet", present=False)
        present = ClassRole(kind="ppe", item="helmet", present=True)
        assert _threshold_for(absent) == 0.25
        assert _threshold_for(present) == 0.35

    def test_person_class_uses_the_presence_bar(self, monkeypatch):
        monkeypatch.setattr(settings, "ppe_confidence_threshold", 0.4)
        assert _threshold_for(ClassRole(kind="person")) == 0.4

    def test_absence_detection_between_the_two_bars_is_kept(self, monkeypatch):
        """The exact regression: a 0.39 no-helmet under a 0.50 presence bar.

        This is what made a bare-headed person in the foreground report as
        undetermined rather than as missing a helmet.
        """
        monkeypatch.setattr(settings, "ppe_confidence_threshold", 0.50)
        monkeypatch.setattr(settings, "ppe_absence_confidence_threshold", 0.25)
        detector = model_detector(
            [
                detection("person", (100, 100, 200, 400), 0.9),
                detection("no-helmet", (110, 110, 190, 160), 0.39),
            ],
            FULL_LABELS,
        )
        track = make_track(1)
        result = detector.assess(np.zeros((HEIGHT, WIDTH, 3), np.uint8), [track], [])
        status = result[1].status("helmet")
        assert status.present is False, "a 0.39 no-helmet must survive the presence bar"

    def test_presence_detection_below_its_bar_is_dropped(self, monkeypatch):
        monkeypatch.setattr(settings, "ppe_confidence_threshold", 0.50)
        monkeypatch.setattr(settings, "ppe_absence_confidence_threshold", 0.25)
        detector = model_detector(
            [
                detection("person", (100, 100, 200, 400), 0.9),
                detection("vest", (110, 180, 190, 300), 0.30),
            ],
            FULL_LABELS,
        )
        track = make_track(1)
        result = detector.assess(np.zeros((HEIGHT, WIDTH, 3), np.uint8), [track], [])
        assert result[1].status("vest").present is not True

    def test_intake_floor_sits_below_both_bars(self):
        """Otherwise the model filters absence detections before anything sees them.

        The intake floor is applied inside the decode path, which does not know
        what a class means; the semantic bars are applied after. An intake floor
        above either bar would quietly cap it.
        """
        assert settings.ppe_intake_confidence_threshold <= settings.ppe_confidence_threshold
        assert (
            settings.ppe_intake_confidence_threshold
            <= settings.ppe_absence_confidence_threshold
        )


class TestConfidenceThreshold:
    def test_ppe_threshold_is_independent_of_the_person_detector(self, monkeypatch):
        """Changing one must not move the other — different nets, different scores."""
        monkeypatch.setattr(settings, "ppe_confidence_threshold", 0.8)
        monkeypatch.setattr(settings, "confidence_threshold", 0.2)
        assert settings.ppe_confidence_threshold == 0.8
        assert settings.confidence_threshold == 0.2

    def test_threshold_is_passed_to_the_backend_not_reimplemented(self, monkeypatch):
        """Filtering happens once, inside the shared decode path."""
        from backend.inference.backends.onnx_backend import OnnxDetector

        detector = OnnxDetector(conf_threshold=0.5, iou_threshold=0.4)
        assert detector.conf_threshold == 0.5
        assert detector.iou_threshold == 0.4
        # With no override the global settings still apply.
        assert OnnxDetector().conf_threshold == settings.confidence_threshold

    def test_low_confidence_event_is_dropped_by_the_engine(self):
        engine = EventEngine(persist=False, broadcast=False, capture_evidence=False)
        analyser = PPEAnalyser(min_consecutive=1, required=["helmet"])
        track = make_track(1)
        weak = PPEAssessment(person_id=1, helmet=False, helmet_confidence=0.2,
                             method="model")
        candidates = analyser.analyse(context([track], {1: weak}))
        assert candidates and candidates[0].confidence == pytest.approx(0.2)
        assert engine.submit(candidates, camera_id="cam") == []


# ══════════════════════════════════════════════════════════════════════════
#  Heuristic fallback
# ══════════════════════════════════════════════════════════════════════════
class TestHeuristicFallback:
    def test_heuristic_is_used_when_no_model_and_permitted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "ppe_model_path", tmp_path / "absent.onnx")
        monkeypatch.setattr(settings, "ppe_allow_heuristic", True)
        reset_ppe_detector()
        try:
            detector = resolve_ppe_detector()
            assert isinstance(detector, HeuristicPPEEstimator)
            assert detector.method == "heuristic"
        finally:
            reset_ppe_detector()

    def test_heuristic_off_means_disabled_not_guessed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "ppe_model_path", tmp_path / "absent.onnx")
        monkeypatch.setattr(settings, "ppe_allow_heuristic", False)
        reset_ppe_detector()
        try:
            detector = resolve_ppe_detector()
            assert isinstance(detector, NullPPEDetector)
            result = detector.assess(
                np.zeros((HEIGHT, WIDTH, 3), np.uint8), [make_track(1)], []
            )[1]
            assert result.helmet is None and result.vest is None
            assert result.is_violation is False
        finally:
            reset_ppe_detector()

    def test_heuristic_confidence_never_reaches_model_levels(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        frame[:, :] = (40, 220, 250)  # saturated high-vis everywhere
        track = make_track(1, (100, 100, 260, 460))
        result = HeuristicPPEEstimator().assess(frame, [track], [])[1]
        assert result.helmet_confidence <= HEURISTIC_CONFIDENCE_CEILING
        assert result.vest_confidence <= HEURISTIC_CONFIDENCE_CEILING
        assert result.method == "heuristic"

    def test_heuristic_results_are_labelled_as_such_end_to_end(self):
        analyser = PPEAnalyser(min_consecutive=1, required=["helmet"])
        track = make_track(1)
        assessment = PPEAssessment(
            person_id=1, helmet=False, helmet_confidence=0.6, method="heuristic"
        )
        events = analyser.analyse(context([track], {1: assessment}))
        assert events[0].metadata["ppe_method"] == "heuristic"

    def test_heuristic_cannot_speak_to_items_it_has_no_signal_for(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        result = HeuristicPPEEstimator().assess(frame, [make_track(1, (100, 100, 260, 460))], [])[1]
        for item in ("gloves", "boots", "goggles", "mask", "harness"):
            assert result.present(item) is None


# ══════════════════════════════════════════════════════════════════════════
#  Rules: events, debounce, duplicates
# ══════════════════════════════════════════════════════════════════════════
class TestPPERules:
    def _run(self, analyser, assessment, frames: int, track=None):
        track = track or make_track(1)
        events = []
        for i in range(frames):
            events += analyser.analyse(
                context([track], {1: assessment}, i, i * 0.1)
            )
        return events

    def test_missing_helmet_raises_missing_helmet(self):
        analyser = PPEAnalyser(min_consecutive=3, required=["helmet", "vest"])
        assessment = PPEAssessment(
            person_id=1, helmet=False, vest=True, helmet_confidence=0.9, method="model"
        )
        events = self._run(analyser, assessment, 6)
        assert [e.event_type for e in events] == [EventType.MISSING_HELMET]
        assert events[0].metadata["missing"] == ["helmet"]

    def test_missing_vest_raises_missing_vest(self):
        analyser = PPEAnalyser(min_consecutive=3, required=["helmet", "vest"])
        assessment = PPEAssessment(
            person_id=1, helmet=True, vest=False, vest_confidence=0.8, method="model"
        )
        events = self._run(analyser, assessment, 6)
        assert [e.event_type for e in events] == [EventType.MISSING_VEST]

    def test_both_missing_is_one_aggregate_violation(self):
        analyser = PPEAnalyser(min_consecutive=2, required=["helmet", "vest"])
        assessment = PPEAssessment(
            person_id=1, helmet=False, vest=False,
            helmet_confidence=0.9, vest_confidence=0.7, method="model",
        )
        events = self._run(analyser, assessment, 8)
        assert len(events) == 1
        assert events[0].event_type == EventType.PPE_VIOLATION
        assert set(events[0].metadata["missing"]) == {"helmet", "vest"}
        assert events[0].confidence == pytest.approx(0.9)

    def test_debounce_holds_a_finding_until_it_persists(self):
        analyser = PPEAnalyser(min_consecutive=5, required=["helmet"])
        assessment = PPEAssessment(person_id=1, helmet=False, helmet_confidence=0.9)
        assert self._run(analyser, assessment, 4) == []
        assert len(self._run(analyser, assessment, 1)) == 1

    def test_a_single_bad_frame_never_fires(self):
        analyser = PPEAnalyser(min_consecutive=3, required=["helmet"])
        track = make_track(1)
        good = PPEAssessment(person_id=1, helmet=True, helmet_confidence=0.9)
        bad = PPEAssessment(person_id=1, helmet=False, helmet_confidence=0.9)
        events = []
        for i, assessment in enumerate([good, bad, good, good, bad, good]):
            events += analyser.analyse(context([track], {1: assessment}, i, i * 0.1))
        assert events == []

    def test_continuous_violation_does_not_duplicate(self):
        """300 frames of one bare head is one alert, not 300."""
        analyser = PPEAnalyser(min_consecutive=3, required=["helmet"])
        assessment = PPEAssessment(person_id=1, helmet=False, helmet_confidence=0.9)
        assert len(self._run(analyser, assessment, 300)) == 1

    def test_it_rearms_once_the_person_becomes_compliant(self):
        analyser = PPEAnalyser(min_consecutive=2, required=["helmet"])
        track = make_track(1)
        bad = PPEAssessment(person_id=1, helmet=False, helmet_confidence=0.9)
        good = PPEAssessment(person_id=1, helmet=True, helmet_confidence=0.9)
        events = []
        for i, assessment in enumerate([bad] * 3 + [good] * 3 + [bad] * 3):
            events += analyser.analyse(context([track], {1: assessment}, i, i * 0.1))
        assert len(events) == 2

    def test_undetermined_never_becomes_a_violation(self):
        analyser = PPEAnalyser(min_consecutive=1, required=["helmet", "vest"])
        assessment = PPEAssessment(person_id=1, helmet=None, vest=None, method="model")
        assert self._run(analyser, assessment, 10) == []

    def test_required_ppe_is_configurable(self):
        """A site that does not require vests gets no vest alerts."""
        assessment = PPEAssessment(
            person_id=1, helmet=True, vest=False, vest_confidence=0.9, method="model"
        )
        assert self._run(PPEAnalyser(min_consecutive=1, required=["helmet"]),
                         assessment, 3) == []
        assert len(self._run(PPEAnalyser(min_consecutive=1, required=["vest"]),
                             assessment, 3)) == 1

    def test_a_future_item_raises_the_aggregate_event_type(self):
        """gloves has no dedicated event type, so it lands on PPE_VIOLATION."""
        analyser = PPEAnalyser(min_consecutive=1, required=["gloves"])
        assessment = PPEAssessment(person_id=1, method="model")
        assessment.set("gloves", False, 0.8, source="detected")
        events = self._run(analyser, assessment, 2)
        assert events[0].event_type == EventType.PPE_VIOLATION
        assert events[0].metadata["missing"] == ["gloves"]

    def test_state_is_dropped_when_a_track_disappears(self):
        analyser = PPEAnalyser(min_consecutive=2, required=["helmet"])
        bad = PPEAssessment(person_id=1, helmet=False, helmet_confidence=0.9)
        analyser.analyse(context([make_track(1)], {1: bad}))
        analyser.analyse(context([], {}))
        assert analyser._state == {}

    def test_two_people_are_debounced_independently(self):
        analyser = PPEAnalyser(min_consecutive=2, required=["helmet"])
        a, b = make_track(1), make_track(2, (600, 100, 700, 400))
        bad = PPEAssessment(person_id=1, helmet=False, helmet_confidence=0.9)
        good = PPEAssessment(person_id=2, helmet=True, helmet_confidence=0.9)
        events = []
        for i in range(4):
            events += analyser.analyse(
                context([a, b], {1: bad, 2: good}, i, i * 0.1)
            )
        assert len(events) == 1 and events[0].person_id == 1


# ══════════════════════════════════════════════════════════════════════════
#  Event engine: cooldown, burst cap, evidence
# ══════════════════════════════════════════════════════════════════════════
class TestPPEEventsAndEvidence:
    def _candidates(self, person_id: int = 1):
        analyser = PPEAnalyser(min_consecutive=1, required=["helmet"])
        assessment = PPEAssessment(
            person_id=person_id, helmet=False, helmet_confidence=0.9, method="model"
        )
        return analyser.analyse(
            context([make_track(person_id)], {person_id: assessment})
        )

    def test_cooldown_suppresses_a_repeat_for_the_same_person(self):
        engine = EventEngine(
            cooldown=CooldownRegistry(default_seconds=30.0, max_per_minute=0),
            persist=False, broadcast=False, capture_evidence=False,
        )
        first = engine.submit(self._candidates(), camera_id="cam", monotonic_time=0.0)
        second = engine.submit(self._candidates(), camera_id="cam", monotonic_time=5.0)
        third = engine.submit(self._candidates(), camera_id="cam", monotonic_time=40.0)
        assert len(first) == 1 and second == [] and len(third) == 1

    def test_burst_cap_holds_back_track_id_churn(self):
        """New track IDs must not become a way around the debounce."""
        engine = EventEngine(
            cooldown=CooldownRegistry(default_seconds=30.0, max_per_minute=3),
            persist=False, broadcast=False, capture_evidence=False,
        )
        accepted = 0
        for person_id in range(1, 20):
            accepted += len(
                engine.submit(
                    self._candidates(person_id), camera_id="cam", monotonic_time=1.0
                )
            )
        assert accepted == 3
        assert engine.stats()["rejected_burst_cap"] == 16

    def test_evidence_records_the_full_provenance(self, tmp_path):
        """Snapshot, camera, person, time, confidence, method and event id."""
        engine = EventEngine(
            cooldown=CooldownRegistry(default_seconds=0.0, max_per_minute=0),
            persist=False, broadcast=False, capture_evidence=True,
        )
        frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
        before = datetime.now(UTC)
        records = engine.submit(
            self._candidates(), camera_id="cam-7", frame=frame, monotonic_time=0.0
        )
        assert len(records) == 1
        record = records[0]

        from backend.events.evidence import resolve_evidence_path

        assert record.snapshot_path
        stored = resolve_evidence_path(record.snapshot_path)
        assert stored is not None and stored.is_file()
        assert stored.stat().st_size > 0
        assert record.event_id in stored.name
        assert record.camera_id == "cam-7"
        assert record.person_id == 1
        assert record.confidence == pytest.approx(0.9)
        assert record.detection_metadata["ppe_method"] == "model"
        assert record.timestamp >= before
        assert record.event_type == EventType.MISSING_HELMET

    def test_snapshot_survives_a_custom_evidence_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "evidence_dir", tmp_path / "elsewhere")
        settings.ensure_directories()
        engine = EventEngine(
            cooldown=CooldownRegistry(default_seconds=0.0, max_per_minute=0),
            persist=False, broadcast=False, capture_evidence=True,
        )
        frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
        record = engine.submit(
            self._candidates(), camera_id="cam", frame=frame, monotonic_time=0.0
        )[0]

        from backend.events.evidence import resolve_evidence_path

        assert resolve_evidence_path(record.snapshot_path) is not None


# ══════════════════════════════════════════════════════════════════════════
#  Diagnostics
# ══════════════════════════════════════════════════════════════════════════
class TestPPEDiagnostics:
    REQUIRED_KEYS = (
        "method", "method_label", "model_path", "model_present", "model_format",
        "model_loaded", "backend", "classes", "confidence_threshold",
        "required_ppe", "heuristic_allowed", "warning",
    )

    def test_reports_every_required_field(self):
        report = ppe_diagnostics()
        for key in self.REQUIRED_KEYS:
            assert key in report, key

    def test_missing_model_warns_actionably_and_does_not_raise(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(settings, "ppe_model_path", tmp_path / "absent.onnx")
        monkeypatch.setattr(settings, "ppe_allow_heuristic", False)
        reset_ppe_detector()
        try:
            report = ppe_diagnostics()
            assert report["model_present"] is False
            assert report["model_loaded"] is False
            assert report["method"] == "disabled"
            assert "fetch_models.py --ppe" in report["warning"]
            assert "DISABLED" in report["warning"]
        finally:
            reset_ppe_detector()

    def test_heuristic_fallback_is_named_as_such(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "ppe_model_path", tmp_path / "absent.onnx")
        monkeypatch.setattr(settings, "ppe_allow_heuristic", True)
        reset_ppe_detector()
        try:
            report = ppe_diagnostics()
            assert report["method_label"] == "HEURISTIC FALLBACK"
            assert "HEURISTIC FALLBACK" in report["warning"]
        finally:
            reset_ppe_detector()

    def test_probe_never_raises_on_a_missing_model(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "ppe_model_path", tmp_path / "absent.onnx")
        ok, reason, detail = ModelPPEDetector.probe()
        assert ok is False
        assert "fetch_models.py --ppe" in reason
        assert detail["model_present"] is False


# ══════════════════════════════════════════════════════════════════════════
#  Real weights (skipped when not installed)
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(
    not settings.ppe_model_file.exists(), reason="no PPE model installed"
)
class TestRealPPEModel:
    def test_the_installed_model_loads_and_maps_its_classes(self):
        detector = ModelPPEDetector()
        detector.load()
        try:
            assert detector.detector is not None
            assert detector.detector.is_loaded
            assert detector.class_map is not None
            assert detector.class_map.covered_items() >= {"helmet", "vest"}
            assert detector.class_map.unmapped == []
        finally:
            detector.close()

    def test_it_finds_ppe_in_a_real_photograph(self):
        """End-to-end on real pixels: a bare head is found and attributed."""
        import cv2

        from backend.tracking import ByteTracker

        image = cv2.imread("tests/fixtures/zidane.jpg")
        assert image is not None

        from backend.inference.backends.onnx_backend import OnnxDetector

        person_detector = OnnxDetector()
        person_detector.load()
        ppe = ModelPPEDetector()
        ppe.load()
        try:
            people = [
                d for d in person_detector.infer(image).detections if d.label == "person"
            ]
            assert people, "no people detected in the fixture"
            tracker = ByteTracker()
            tracks = []
            for i in range(5):
                tracks = tracker.update(people, timestamp=i * 0.1)
            results = ppe.assess(image, tracks, people)
            assert results
            assert any(a.helmet is False for a in results.values())
        finally:
            person_detector.close()
            ppe.close()
