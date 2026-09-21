"""Hard-hat validation: only an industrial hard hat counts as a helmet.

The bug these tests exist for is specific and measured. The PPE model calls a
baseball cap a ``helmet`` — on 45 head crops from the reference site footage it
did so for 27 of the 28 caps, at confidences up to 0.80 — and a system that
takes that at face value reports a worker in a cap as PPE-compliant. Confidence
thresholds cannot fix it, because the detector is confident and wrong.

So the tests below hold two lines:

* **a cap must never become a valid helmet**, by any path — high detector
  confidence, a missing validator, a broken validator, a crop too small to
  classify. Every one of those has to end at ``NO HELMET``;
* **the correction has to reach the event engine**, or the render is honest
  while the incident record is not.

Most run against stub validators, so the logic is tested without a 350 MB
download. The ones that load the real validator are marked and skipped when it
is not installed, and they run over real crops in ``tests/fixtures/headwear``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.analysis.base import FrameContext
from backend.analysis.ppe_rules import PPEAnalyser
from backend.config import settings
from backend.events.types import EventType
from backend.inference.hardhat import (
    ClipHardHatValidator,
    HardHatResult,
    HardHatValidator,
    HeuristicHardHatValidator,
    TrackedHardHatValidator,
    UnavailableHardHatValidator,
    crop_head,
    hardhat_diagnostics,
    reset_hardhat_validator,
    resolve_hardhat_validator,
)
from backend.inference.ppe import (
    UNVERIFIED_HELMET_CONFIDENCE,
    PPEAssessment,
    validate_helmets,
)
from backend.tracking import Track, TrackState

FIXTURES = Path(__file__).parent / "fixtures" / "headwear"
WIDTH, HEIGHT = 960, 540

#: A person, and the helmet box the PPE detector found on their head.
PERSON = (200.0, 100.0, 300.0, 400.0)
HELMET_BOX = (218.0, 102.0, 282.0, 140.0)


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════
def make_track(track_id: int = 1, bbox: tuple = PERSON) -> Track:
    track = Track(
        track_id=track_id, bbox=bbox, confidence=0.9, state=TrackState.TRACKED
    )
    track.record(0.0)
    return track


def frame() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)


def detected_helmet(
    confidence: float = 0.76, person_id: int = 1, box: tuple | None = HELMET_BOX
) -> PPEAssessment:
    """What the PPE detector hands over: a helmet it is confident about."""
    assessment = PPEAssessment(person_id=person_id, method="model")
    assessment.set("helmet", True, confidence, source="detected", bbox=box)
    assessment.set("vest", True, 0.84, source="detected", bbox=(205.0, 160.0, 295.0, 270.0))
    return assessment


class StubValidator(HardHatValidator):
    """A validator with a scripted answer, for testing everything around it."""

    def __init__(self, verdict: str, confidence: float = 0.9, method: str = "model"):
        self.verdict, self.confidence, self.method = verdict, confidence, method  # type: ignore[assignment]
        self.description = f"stub validator ({verdict})"
        self.calls: list[tuple[int, int]] = []

    def classify(self, crop: np.ndarray) -> HardHatResult:
        self.calls.append(crop.shape[:2])
        probability = self.confidence if self.verdict == "hard_hat" else 1 - self.confidence
        return HardHatResult(
            self.verdict, self.confidence, probability, self.method  # type: ignore[arg-type]
        )


class ExplodingValidator(HardHatValidator):
    """A validator whose model is broken. Must not become false compliance."""

    method = "model"  # type: ignore[assignment]

    def classify(self, crop: np.ndarray) -> HardHatResult:
        raise RuntimeError("onnxruntime session died")


def tracked(validator: HardHatValidator) -> TrackedHardHatValidator:
    return TrackedHardHatValidator(validator)


def assess_with(validator: HardHatValidator, assessment: PPEAssessment, track: Track):
    results = {track.track_id: assessment}
    validate_helmets(frame(), results, [track], tracked(validator))
    return assessment.status("helmet")


# ══════════════════════════════════════════════════════════════════════════
#  The rule: a cap is not a helmet
# ══════════════════════════════════════════════════════════════════════════
class TestCapIsNotAHelmet:
    def test_baseball_cap_is_not_valid_helmet(self):
        """The detector says helmet 0.76; the validator says cap."""
        status = assess_with(
            StubValidator("not_hard_hat", 0.82), detected_helmet(0.76), make_track()
        )
        assert status.present is False, "a cap was accepted as a helmet"
        assert status.source == "rejected"
        assert status.validation == "not_hard_hat"

    def test_sports_cap_is_not_valid_helmet(self):
        status = assess_with(
            StubValidator("not_hard_hat", 0.91), detected_helmet(0.88), make_track()
        )
        assert status.present is False

    def test_a_very_confident_detection_does_not_override_the_validator(self):
        """Detector confidence is not evidence about what the object *is*."""
        status = assess_with(
            StubValidator("not_hard_hat", 0.6), detected_helmet(0.99), make_track()
        )
        assert status.present is False

    def test_industrial_hardhat_is_valid_helmet(self):
        status = assess_with(
            StubValidator("hard_hat", 0.94), detected_helmet(0.88), make_track()
        )
        assert status.present is True
        assert status.validation == "hard_hat"

    def test_unknown_headwear_defaults_to_no_helmet(self):
        status = assess_with(
            StubValidator("unknown", 0.0), detected_helmet(0.76), make_track()
        )
        assert status.present is False, "unknown must never mean compliant"
        assert status.source == "unverified"

    def test_a_validator_that_crashes_does_not_create_false_compliance(self):
        status = assess_with(ExplodingValidator(), detected_helmet(0.93), make_track())
        assert status.present is False
        assert status.source == "unverified"

    def test_validator_failure_does_not_create_false_compliance(self):
        """No validator installed at all — the same conservative answer."""
        status = assess_with(
            UnavailableHardHatValidator("no model"), detected_helmet(0.93), make_track()
        )
        assert status.present is False
        assert status.validation == "unknown"
        assert status.confidence == pytest.approx(UNVERIFIED_HELMET_CONFIDENCE)

    def test_a_head_too_small_to_classify_is_not_compliant(self):
        tiny = (210.0, 100.0, 218.0, 108.0)
        status = assess_with(
            StubValidator("hard_hat", 0.99), detected_helmet(box=tiny), make_track()
        )
        assert status.present is False
        assert status.source == "unverified"

    def test_an_absent_helmet_is_left_alone(self):
        """Nothing was claimed, so there is nothing to overrule."""
        assessment = PPEAssessment(person_id=1, method="model")
        assessment.set("helmet", False, 0.71, source="detected", bbox=HELMET_BOX)
        stub = StubValidator("hard_hat", 0.99)
        status = assess_with(stub, assessment, make_track())
        assert status.present is False
        assert status.source == "detected"
        assert not stub.calls, "a missing helmet does not need validating"


# ══════════════════════════════════════════════════════════════════════════
#  Confidences stay separate
# ══════════════════════════════════════════════════════════════════════════
class TestConfidence:
    def test_the_detector_confidence_is_kept_but_not_reused(self):
        status = assess_with(
            StubValidator("not_hard_hat", 0.82), detected_helmet(0.76), make_track()
        )
        assert status.detector_confidence == pytest.approx(0.76)
        assert status.validation_confidence == pytest.approx(0.82)
        # The reported confidence is the validator's, never the detector's.
        assert status.confidence == pytest.approx(0.82)

    def test_a_confirmed_helmet_reports_the_validation_confidence(self):
        status = assess_with(
            StubValidator("hard_hat", 0.91), detected_helmet(0.76), make_track()
        )
        assert status.detector_confidence == pytest.approx(0.76)
        assert status.confidence == pytest.approx(0.91)

    def test_the_validation_is_visible_in_the_serialised_assessment(self):
        assessment = detected_helmet(0.76)
        assess_with(StubValidator("not_hard_hat", 0.82), assessment, make_track())
        item = assessment.as_dict()["items"]["helmet"]
        assert item["present"] is False
        assert item["validation"] == "not_hard_hat"
        assert item["detector_confidence"] == pytest.approx(0.76)


# ══════════════════════════════════════════════════════════════════════════
#  Vest is untouched
# ══════════════════════════════════════════════════════════════════════════
class TestVestUnchanged:
    def test_the_vest_verdict_is_not_validated_or_altered(self):
        assessment = detected_helmet(0.76)
        assess_with(StubValidator("not_hard_hat", 0.9), assessment, make_track())
        vest = assessment.status("vest")
        assert vest.present is True
        assert vest.confidence == pytest.approx(0.84)
        assert vest.validation == ""

    def test_only_the_helmet_is_sent_to_the_validator(self):
        stub = StubValidator("hard_hat", 0.9)
        assess_with(stub, detected_helmet(0.76), make_track())
        assert len(stub.calls) == 1


# ══════════════════════════════════════════════════════════════════════════
#  Association: the right head, for the right person
# ══════════════════════════════════════════════════════════════════════════
class TestAssociation:
    def test_multiple_people_hardhat_validation_is_associated_correctly(self):
        """Each verdict lands on the person whose head was classified."""
        left, right = (200.0, 100.0, 300.0, 400.0), (600.0, 100.0, 700.0, 400.0)
        tracks = [make_track(1, left), make_track(2, right)]
        compliant = PPEAssessment(person_id=1, method="model")
        compliant.set("helmet", True, 0.8, source="detected", bbox=(218.0, 102.0, 282.0, 140.0))
        capped = PPEAssessment(person_id=2, method="model")
        capped.set("helmet", True, 0.8, source="detected", bbox=(618.0, 102.0, 682.0, 140.0))

        class ByPosition(HardHatValidator):
            method = "model"  # type: ignore[assignment]

            def validate(self, image, box):
                verdict = "hard_hat" if box[0] < 400 else "not_hard_hat"
                return HardHatResult(verdict, 0.9, 0.9 if box[0] < 400 else 0.1, "model")

            def classify(self, crop):  # pragma: no cover - validate is overridden
                raise AssertionError

        validate_helmets(
            frame(), {1: compliant, 2: capped}, tracks, tracked(ByPosition())
        )
        assert compliant.present("helmet") is True
        assert capped.present("helmet") is False

    def test_the_crop_comes_from_the_detector_box_not_the_whole_person(self):
        canvas = frame()
        crop = crop_head(canvas, HELMET_BOX)
        assert crop is not None
        box_h = HELMET_BOX[3] - HELMET_BOX[1]
        person_h = PERSON[3] - PERSON[1]
        assert crop.shape[0] < person_h * 0.25
        assert crop.shape[0] >= box_h

    def test_a_helmet_with_no_box_falls_back_to_the_head_region(self):
        """The heuristic estimator names no box; the head region is used."""
        assessment = PPEAssessment(person_id=1, method="heuristic")
        assessment.set("helmet", True, 0.6, source="heuristic")
        seen: list[tuple] = []

        class Recording(HardHatValidator):
            method = "model"  # type: ignore[assignment]

            def validate(self, image, box):
                seen.append(box)
                return HardHatResult("not_hard_hat", 0.8, 0.2, "model")

            def classify(self, crop):  # pragma: no cover
                raise AssertionError

        validate_helmets(frame(), {1: assessment}, [make_track()], tracked(Recording()))
        assert seen, "no validation was attempted"
        top, bottom = seen[0][1], seen[0][3]
        assert top >= PERSON[1]
        assert bottom <= PERSON[1] + 0.35 * (PERSON[3] - PERSON[1])


# ══════════════════════════════════════════════════════════════════════════
#  Temporal behaviour
# ══════════════════════════════════════════════════════════════════════════
class TestTemporalStability:
    def test_a_verdict_is_held_between_revalidations(self, monkeypatch):
        monkeypatch.setattr(settings, "hardhat_validate_every_n_detections", 6)
        stub = StubValidator("hard_hat", 0.9)
        validator = tracked(stub)
        canvas, track = frame(), make_track()
        for _ in range(5):
            validator.begin_frame()
            validator.validate_track(canvas, HELMET_BOX, track.track_id)
        assert len(stub.calls) == 1, "the head was re-classified every frame"

    def test_it_revalidates_once_the_interval_passes(self, monkeypatch):
        monkeypatch.setattr(settings, "hardhat_validate_every_n_detections", 3)
        stub = StubValidator("hard_hat", 0.9)
        validator = tracked(stub)
        canvas, track = frame(), make_track()
        for _ in range(9):
            validator.begin_frame()
            validator.validate_track(canvas, HELMET_BOX, track.track_id)
        assert len(stub.calls) == 3

    def test_one_bad_look_does_not_flip_the_verdict(self, monkeypatch):
        """Two hard-hat readings then one cap reading stays a hard hat."""
        monkeypatch.setattr(settings, "hardhat_validate_every_n_detections", 1)
        monkeypatch.setattr(settings, "hardhat_smoothing_window", 3)

        class Scripted(HardHatValidator):
            method = "model"  # type: ignore[assignment]
            probabilities = [0.95, 0.90, 0.05, 0.92]

            def __init__(self):
                self.index = 0

            def validate(self, image, box):
                p = self.probabilities[min(self.index, len(self.probabilities) - 1)]
                self.index += 1
                return HardHatResult(
                    "hard_hat" if p >= 0.5 else "not_hard_hat", p, p, "model"
                )

            def classify(self, crop):  # pragma: no cover
                raise AssertionError

        validator = tracked(Scripted())
        canvas, verdicts = frame(), []
        for _ in range(4):
            validator.begin_frame()
            verdicts.append(validator.validate_track(canvas, HELMET_BOX, 1).verdict)
        assert verdicts == ["hard_hat"] * 4

    def test_track_ids_do_not_collide_across_scopes(self, monkeypatch):
        """A camera and an upload job both count from 1."""
        monkeypatch.setattr(settings, "hardhat_validate_every_n_detections", 100)

        class ByScope(HardHatValidator):
            method = "model"  # type: ignore[assignment]

            def __init__(self):
                self.next = "hard_hat"

            def validate(self, image, box):
                verdict, self.next = self.next, "not_hard_hat"
                return HardHatResult(verdict, 0.9, 0.9 if verdict == "hard_hat" else 0.1, "model")

            def classify(self, crop):  # pragma: no cover
                raise AssertionError

        validator = tracked(ByScope())
        canvas = frame()
        validator.begin_frame()
        first = validator.validate_track(canvas, HELMET_BOX, 1, scope="camera-a")
        second = validator.validate_track(canvas, HELMET_BOX, 1, scope="upload-b")
        assert first.verdict == "hard_hat"
        assert second.verdict == "not_hard_hat", "one scope read the other's verdict"

    def test_departed_people_are_forgotten(self):
        validator = tracked(StubValidator("hard_hat", 0.9))
        canvas = frame()
        validator.begin_frame()
        validator.validate_track(canvas, HELMET_BOX, 1, scope="cam")
        validator.validate_track(canvas, HELMET_BOX, 2, scope="cam")
        assert validator.info()["tracked_people"] == 2
        validator.forget("cam", keep={1})
        assert validator.info()["tracked_people"] == 1


# ══════════════════════════════════════════════════════════════════════════
#  Through to the event engine
# ══════════════════════════════════════════════════════════════════════════
def run_analyser(assessment: PPEAssessment, frames: int = 6):
    analyser = PPEAnalyser(min_consecutive=5, required=["helmet", "vest"])
    track = make_track()
    candidates = []
    for index in range(frames):
        context = FrameContext(
            camera_id="cam", frame_index=index, timestamp=float(index),
            wall_time=datetime.now(UTC).timestamp(),
            frame_width=WIDTH, frame_height=HEIGHT,
            tracks=[track], zones=[], ppe={1: assessment},
        )
        candidates.extend(analyser.analyse(context))
    return candidates


class TestEventEngine:
    def test_detector_helmet_plus_nonhardhat_validator_causes_violation(self):
        assessment = detected_helmet(0.76)
        assess_with(StubValidator("not_hard_hat", 0.82), assessment, make_track())
        events = run_analyser(assessment)
        assert [e.event_type for e in events] == [EventType.MISSING_HELMET]
        assert events[0].confidence == pytest.approx(0.82)

    def test_detector_helmet_plus_hardhat_validator_is_compliant(self):
        assessment = detected_helmet(0.88)
        assess_with(StubValidator("hard_hat", 0.94), assessment, make_track())
        assert run_analyser(assessment) == []

    def test_the_event_records_why_the_helmet_was_rejected(self):
        assessment = detected_helmet(0.76)
        assess_with(StubValidator("not_hard_hat", 0.82), assessment, make_track())
        events = run_analyser(assessment)
        helmet = events[0].metadata["ppe_items"]["helmet"]
        assert helmet["validation"] == "not_hard_hat"
        assert helmet["detector_confidence"] == pytest.approx(0.76)
        assert helmet["source"] == "rejected"
        assert events[0].metadata["ppe_helmet_validation"]["verdict"] == "not_hard_hat"

    def test_an_unvalidated_helmet_still_fires_above_the_event_floor(self):
        """A violation nobody could verify must not be silently dropped."""
        assessment = detected_helmet(0.76)
        assess_with(UnavailableHardHatValidator("none"), assessment, make_track())
        events = run_analyser(assessment)
        assert events and events[0].confidence >= settings.event_min_confidence


# ══════════════════════════════════════════════════════════════════════════
#  Wiring: the stage is reached from the detector, not just callable
# ══════════════════════════════════════════════════════════════════════════
class TestDetectorIntegration:
    """`ModelPPEDetector.assess` must run the validator itself.

    Every consumer — the live pipeline, the upload job, evidence, the render —
    goes through `assess`. If the stage were wired in anywhere else, one of
    them would still be reporting caps as helmets.
    """

    def build(self, monkeypatch, verdict: str, confidence: float = 0.9):
        from backend.inference import ppe as ppe_module
        from backend.inference.base import Detection, InferenceResult
        from backend.inference.ppe import ModelPPEDetector
        from backend.inference.ppe_taxonomy import PPEClassMap

        stub = StubValidator(verdict, confidence)
        monkeypatch.setattr(
            ppe_module, "resolve_hardhat_validator", lambda: tracked(stub)
        )

        class FakeDetector:
            name = "mock"

            def infer(self, image):
                return InferenceResult(
                    detections=[
                        Detection(bbox=HELMET_BOX, confidence=0.76, class_id=0, label="helmet"),
                        Detection(bbox=PERSON, confidence=0.9, class_id=3, label="person"),
                    ],
                    inference_ms=1.0, backend="mock",
                )

            def close(self):
                pass

        detector = ModelPPEDetector(detector=FakeDetector())
        detector.class_map = PPEClassMap.build(
            ["helmet", "no-helmet", "no-vest", "person", "vest"]
        )
        return detector, stub

    def test_a_cap_detected_as_a_helmet_comes_back_as_a_violation(self, monkeypatch):
        detector, stub = self.build(monkeypatch, "not_hard_hat", 0.82)
        track = make_track(1, PERSON)
        result = detector.assess(frame(), [track], [])[1]
        assert stub.calls, "assess() never consulted the hard-hat validator"
        assert result.helmet is False
        assert result.is_violation
        assert "helmet" in result.violations

    def test_a_real_hard_hat_comes_back_compliant(self, monkeypatch):
        detector, _ = self.build(monkeypatch, "hard_hat", 0.94)
        result = detector.assess(frame(), [make_track(1, PERSON)], [])[1]
        assert result.helmet is True
        assert result.status("helmet").validation == "hard_hat"

    def test_the_scope_reaches_the_validator(self, monkeypatch):
        detector, stub = self.build(monkeypatch, "hard_hat", 0.9)
        detector.assess(frame(), [make_track(1, PERSON)], [], scope="upload-abc")
        assert stub.calls


# ══════════════════════════════════════════════════════════════════════════
#  Configuration and reporting
# ══════════════════════════════════════════════════════════════════════════
class TestConfiguration:
    def test_the_conservative_defaults_are_the_defaults(self):
        """Asserted on the field defaults: the suite turns validation off.

        See the note in conftest — synthetic frames are not photographs of
        heads, so the validator would reject every fixture. What ships must
        still default to validating, and to refusing the heuristic.
        """
        from backend.config import Settings

        fields = Settings.model_fields
        assert fields["hardhat_validation_enabled"].default is True
        assert fields["hardhat_allow_heuristic"].default is False
        assert fields["hardhat_confidence_threshold"].default == 0.50

    def test_disabling_validation_is_reported_as_a_risk(self, monkeypatch):
        monkeypatch.setattr(settings, "hardhat_validation_enabled", False)
        reset_hardhat_validator()
        try:
            report = hardhat_diagnostics()
            assert report["method"] == "disabled"
            assert report["compliance_ready"] is False
            assert "cap" in report["warning"].lower()
        finally:
            reset_hardhat_validator()

    def test_a_missing_validator_is_never_reported_as_compliance_ready(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(settings, "hardhat_validation_enabled", True)
        monkeypatch.setattr(settings, "hardhat_model_path", tmp_path / "absent.onnx")
        monkeypatch.setattr(settings, "hardhat_allow_heuristic", False)
        reset_hardhat_validator()
        try:
            report = hardhat_diagnostics()
            assert report["method"] == "unavailable"
            assert report["compliance_ready"] is False
            assert report["model_available"] is False
        finally:
            reset_hardhat_validator()

    def test_the_heuristic_is_only_used_when_permitted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "hardhat_validation_enabled", True)
        monkeypatch.setattr(settings, "hardhat_model_path", tmp_path / "absent.onnx")
        monkeypatch.setattr(settings, "hardhat_allow_heuristic", True)
        reset_hardhat_validator()
        try:
            validator = resolve_hardhat_validator()
            assert validator.method == "heuristic"
            assert hardhat_diagnostics()["method_label"] == "HEURISTIC FALLBACK"
        finally:
            reset_hardhat_validator()

    def test_disabling_validation_leaves_the_detection_alone(self, monkeypatch):
        """Off means the stage does not run — not that it rejects everything.

        The distinction matters: `disabled` is an operator decision to trust
        the detector, `unavailable` is the platform being unable to check. One
        keeps the helmet, the other refuses to certify it.
        """
        assessment = detected_helmet(0.76)
        disabled = UnavailableHardHatValidator("switched off", method="disabled")
        assess_with(disabled, assessment, make_track())
        assert assessment.helmet is True
        assert assessment.status("helmet").validation == ""

    def test_the_threshold_moves_the_decision(self, monkeypatch):
        from backend.inference.hardhat import _decide

        monkeypatch.setattr(settings, "hardhat_confidence_threshold", 0.9)
        assert _decide(0.8, "model").verdict == "not_hard_hat"
        monkeypatch.setattr(settings, "hardhat_confidence_threshold", 0.5)
        assert _decide(0.8, "model").verdict == "hard_hat"


class TestHeuristicFallback:
    def test_it_never_sounds_as_certain_as_the_model(self):
        validator = HeuristicHardHatValidator()
        for name in sorted(FIXTURES.glob("*.png")):
            result = validator.classify(cv2.imread(str(name)))
            assert result.method == "heuristic"
            assert result.hard_hat_probability <= 0.88

    def test_it_explains_which_signals_it_used(self):
        result = HeuristicHardHatValidator().classify(
            cv2.imread(str(next(FIXTURES.glob("hardhat_*.png"))))
        )
        for signal in ("smoothness", "curvature", "symmetry", "aspect"):
            assert signal in result.reason

    def test_it_declines_on_a_crop_too_small_to_measure(self):
        result = HeuristicHardHatValidator().classify(np.zeros((8, 8, 3), np.uint8))
        assert result.verdict == "unknown"


# ══════════════════════════════════════════════════════════════════════════
#  The real validator, over real headwear
# ══════════════════════════════════════════════════════════════════════════
def _validator_installed() -> bool:
    return ClipHardHatValidator.probe()[0]


@pytest.mark.skipif(
    not _validator_installed(),
    reason="hard-hat validator not installed (scripts/fetch_models.py --hardhat)",
)
class TestRealValidator:
    """Measured behaviour on real head crops from the reference footage.

    The PPE detector called every one of these a helmet. Whether it was right
    is exactly what these assert.
    """

    @pytest.fixture(scope="class")
    def validator(self) -> ClipHardHatValidator:
        instance = ClipHardHatValidator()
        instance.load()
        return instance

    @pytest.fixture(scope="class")
    def manifest(self) -> dict[str, str]:
        return json.loads((FIXTURES / "manifest.json").read_text())

    def test_no_cap_is_accepted_as_a_hard_hat(self, validator, manifest):
        """The regression that matters: cap false-positive rate must be zero."""
        wrong = []
        for name, truth in sorted(manifest.items()):
            if truth == "hardhat":
                continue
            result = validator.classify(cv2.imread(str(FIXTURES / name)))
            if result.is_hard_hat:
                wrong.append((name, round(result.hard_hat_probability, 3)))
        assert not wrong, f"ordinary headwear accepted as a hard hat: {wrong}"

    def test_industrial_hard_hats_are_recognised(self, validator, manifest):
        hats = [n for n, truth in manifest.items() if truth == "hardhat"]
        accepted = sum(
            validator.classify(cv2.imread(str(FIXTURES / name))).is_hard_hat
            for name in hats
        )
        # Measured at 75% on the full 45-crop set. The floor is set below that
        # so the test tracks a real regression rather than encoding one run,
        # while still failing if recognition collapses.
        assert accepted / len(hats) >= 0.6, (
            f"only {accepted}/{len(hats)} hard hats recognised"
        )

    def test_a_bare_head_is_not_a_hard_hat(self, validator):
        result = validator.classify(cv2.imread(str(FIXTURES / "bare_022.png")))
        assert result.verdict == "not_hard_hat"

    def test_it_reports_itself_as_a_model_not_a_heuristic(self, validator):
        info = validator.info()
        assert info["method"] == "model"
        assert info["concepts"] > 2
        assert "hard hat" in " ".join(info["hardhat_concepts"])

    def test_the_prompt_bank_and_model_must_agree(self, validator, tmp_path):
        """A bank from a different model is refused, not silently misread."""
        from backend.inference.base import BackendUnavailable

        bogus = tmp_path / "prompts.npz"
        np.savez(
            bogus,
            text_embeddings=np.zeros((2, 7), np.float32),
            is_hardhat=np.array([True, False]),
            concepts=np.array(["a", "b"]),
            image_size=np.int32(224),
            image_mean=np.zeros(3, np.float32),
            image_std=np.ones(3, np.float32),
            normalise=np.bool_(True),
            logit_scale=np.float32(100.0),
        )
        mismatched = ClipHardHatValidator(prompts_path=bogus)
        with pytest.raises(BackendUnavailable, match="different models"):
            mismatched.load()
