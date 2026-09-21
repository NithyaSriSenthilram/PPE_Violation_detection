"""PPE assessment: which protective equipment each tracked person is wearing.

Two implementations behind one interface, and the difference between them is
never hidden from the operator.

**ModelPPEDetector** — a trained PPE object-detection network loaded through
the ordinary :mod:`backend.inference.registry` backend chain (MAX → ONNX
Runtime → Ultralytics), so the PPE model gets the same hardware acceleration,
the same Mojo-accelerated decode/NMS and the same honest backend reporting as
the person detector. It reads its class vocabulary from the model's own
``.names`` file and resolves each class to a meaning through
:mod:`backend.inference.ppe_taxonomy`, so a third-party model that calls a hard
hat ``Hardhat`` needs configuration, not code. This is the production path.

**HeuristicPPEEstimator** — a colour/geometry estimator over the head and torso
regions of each person, used only when no PPE model is installed *and*
``PPE_ALLOW_HEURISTIC=true``. It is a genuine measurement (HSV statistics over
defined regions) rather than a placeholder, but it is materially less reliable
than a trained model: a yellow shirt, a bright background or unusual lighting
will fool it. Its confidence is therefore capped well below certainty.

Every result carries `method`, which is propagated through the event metadata,
the API and the dashboard and rendered there as **AI MODEL** or **HEURISTIC
FALLBACK**. With ``PPE_ALLOW_HEURISTIC=false`` and no model, PPE status is
reported as *undetermined* rather than guessed, and no violation can fire.

Helmets get a third stage. A PPE model is trained to find head *coverings*, and
it calls a baseball cap a helmet with real confidence — so every helmet it
proposes is checked by :mod:`backend.inference.hardhat`, which asks the
different question *is this an industrial hard hat?*. Only that answer decides
compliance, and when it cannot be given the helmet is rejected rather than
assumed. See :func:`validate_helmets`.

Association is the other half of the problem. Detecting a helmet somewhere in
the frame says nothing about whether the person in the restricted area is
wearing one, so every PPE box is attributed to a specific person by how much of
it falls inside that person's box *and* whether it sits where the item belongs
on a body — see :func:`associate_items`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from backend.config import settings
from backend.inference.base import BackendUnavailable, Detection, Detector
from backend.inference.hardhat import (
    HardHatResult,
    TrackedHardHatValidator,
    resolve_hardhat_validator,
)
from backend.inference.ppe_taxonomy import (
    ITEM_SPECS,
    ClassMapError,
    ClassRole,
    PPEClassMap,
    body_region_box,
    item_label,
)
from backend.logging_conf import get_logger
from backend.tracking import Track

logger = get_logger(__name__)

PPEMethod = Literal["model", "heuristic", "disabled"]

#: How each method is named to an operator. The dashboard shows these
#: verbatim so a colour estimate is never mistaken for a trained model.
METHOD_LABELS: dict[str, str] = {
    "model": "AI MODEL",
    "heuristic": "HEURISTIC FALLBACK",
    "disabled": "DISABLED",
}

#: Ceiling on heuristic confidence. A colour test must never present itself as
#: being as sure as a trained detector.
HEURISTIC_CONFIDENCE_CEILING = 0.62

#: Confidence attached to an absence the model *implied* rather than named: it
#: found no helmet on a person it clearly saw, but has no `no_helmet` class to
#: say so directly. Deliberately modest — it is a weaker signal than a positive
#: detection of a bare head.
INFERRED_ABSENCE_CONFIDENCE = 0.55

#: Items whose detection is not taken at face value. Only the helmet: "is this
#: a hard hat or a cap?" is a question with a wrong answer that reports false
#: compliance. A vest is a vest.
VALIDATED_ITEMS: tuple[str, ...] = ("helmet",)

#: Confidence attached to a helmet the validator could not rule on — no
#: validator installed, or a head crop too small to classify. The item is
#: absent because it is *unverified*, not because anything was seen to be
#: wrong, so the finding is reported at the same modest confidence as an
#: absence inferred from omission rather than at the detector's confidence,
#: which measured the wrong thing entirely.
UNVERIFIED_HELMET_CONFIDENCE = 0.55

#: How far outside its expected body band an item may sit before the match is
#: rejected, as a fraction of person height.
BAND_TOLERANCE = 0.18

#: IoU above which a PPE-model `person` box is taken to be the same person as a
#: track. Used only to establish that the model actually looked at this person.
PERSON_MATCH_IOU = 0.45


def _min_height() -> int:
    """Person boxes shorter than this are not assessable (configurable)."""
    return settings.ppe_min_person_height


def _threshold_for(role: ClassRole) -> float:
    """The confidence bar this class has to clear, by what the class *means*.

    Presence and absence are not equally easy to see, and the model scores them
    accordingly — on the reference footage `no-helmet` peaks at 0.54 while
    `vest` reaches 0.86. Judging both against one number either discards most
    of the absence class or admits weak presence detections; which one you get
    depends only on where the single number is set. So the bar is chosen here,
    in the one layer that knows whether a class asserts an item is *on* someone
    or *missing* from them.
    """
    if role.kind == "ppe" and not role.present:
        return settings.ppe_absence_confidence_threshold
    return settings.ppe_confidence_threshold


@dataclass(slots=True)
class PPEItemStatus:
    """Verdict for one item on one person."""

    present: bool | None = None
    confidence: float = 0.0
    #: ``detected`` — the model named it; ``inferred`` — absence deduced from
    #: the item never being found on a person the model saw; ``heuristic`` —
    #: colour estimate; ``rejected`` — detected, but the second stage found it
    #: was not the real article; ``unverified`` — detected, and the second
    #: stage could not say.
    source: str = ""
    bbox: tuple[float, float, float, float] | None = None
    #: What the *detector* scored, kept separately from :attr:`confidence`
    #: once a second stage has spoken. The detector was sure it saw a helmet
    #: and it was looking at a cap; presenting that number as the confidence in
    #: the final verdict would be presenting it as evidence for something it
    #: never measured.
    detector_confidence: float = 0.0
    #: ``hard_hat`` | ``not_hard_hat`` | ``unknown``, or empty when the item
    #: has no second stage.
    validation: str = ""
    validation_confidence: float = 0.0
    #: ``model`` | ``heuristic`` | ``unavailable`` | ``disabled``.
    validation_method: str = ""


class PPEAssessment:
    """PPE status for one person, across every item the detector covers.

    ``None`` means *undetermined* — genuinely different from ``False``
    (definitely absent), and the rule engine treats it as such: an undetermined
    helmet never raises a violation.

    ``helmet``/``vest`` remain first-class attributes because they are the two
    items every deployment uses, but they are views onto :attr:`items`, which
    is what carries gloves, boots, goggles, masks and harnesses.
    """

    __slots__ = ("person_id", "method", "detail", "items")

    def __init__(
        self,
        person_id: int | None = None,
        helmet: bool | None = None,
        vest: bool | None = None,
        helmet_confidence: float = 0.0,
        vest_confidence: float = 0.0,
        method: PPEMethod = "disabled",
        detail: dict[str, Any] | None = None,
        items: dict[str, PPEItemStatus] | None = None,
    ) -> None:
        self.person_id = person_id
        self.method: PPEMethod = method
        self.detail: dict[str, Any] = detail if detail is not None else {}
        self.items: dict[str, PPEItemStatus] = items if items is not None else {}
        if helmet is not None or helmet_confidence:
            self.set("helmet", helmet, helmet_confidence, source=method)
        if vest is not None or vest_confidence:
            self.set("vest", vest, vest_confidence, source=method)

    # ── item access ──────────────────────────────────────────────────────
    def status(self, item: str) -> PPEItemStatus:
        return self.items.get(item) or PPEItemStatus()

    def set(
        self,
        item: str,
        present: bool | None,
        confidence: float = 0.0,
        *,
        source: str = "",
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Record a verdict, keeping the strongest evidence for the item.

        One frame can yield several readings for the same item on the same
        person — two overlapping `helmet` boxes, or a `helmet` and a
        `no_helmet` box that disagree. The strongest determination wins, and a
        determination always beats *undetermined*.
        """
        existing = self.items.get(item)
        if existing is not None:
            if present is None and existing.present is not None:
                return
            if existing.present is not None and confidence <= existing.confidence:
                return
        self.items[item] = PPEItemStatus(present, confidence, source, bbox)

    def _overwrite(self, item: str, present: bool | None, confidence: float) -> None:
        """Unconditional write, backing the legacy helmet/vest attributes."""
        current = self.items.get(item)
        self.items[item] = PPEItemStatus(
            present, confidence, self.method, current.bbox if current else None
        )

    def apply_validation(self, item: str, result: HardHatResult) -> None:
        """Let a second-stage verdict overrule a detection.

        The detector's own confidence is preserved in
        :attr:`PPEItemStatus.detector_confidence` and never reused as the
        confidence in the outcome — the two measure different things, and the
        whole point of this stage is that the first one can be confidently
        wrong.
        """
        status = self.items.get(item)
        if status is None or status.present is not True:
            return  # nothing was claimed; there is nothing to overrule

        detector_confidence = status.detector_confidence or status.confidence
        status.detector_confidence = detector_confidence
        status.validation = result.verdict
        status.validation_confidence = result.confidence
        status.validation_method = result.method

        if result.verdict == "hard_hat":
            # Confirmed. The confidence that matters now is the validator's:
            # it is what says this is PPE rather than headwear.
            status.confidence = result.confidence
            return

        status.present = False
        if result.verdict == "not_hard_hat":
            status.source = "rejected"
            status.confidence = result.confidence
        else:
            # Unknown. Absent because unverified — never silently compliant.
            status.source = "unverified"
            status.confidence = UNVERIFIED_HELMET_CONFIDENCE

    def present(self, item: str) -> bool | None:
        return self.status(item).present

    def confidence_for(self, item: str) -> float:
        return self.status(item).confidence

    # ── helmet / vest views ──────────────────────────────────────────────
    @property
    def helmet(self) -> bool | None:
        return self.present("helmet")

    @helmet.setter
    def helmet(self, value: bool | None) -> None:
        self._overwrite("helmet", value, self.confidence_for("helmet"))

    @property
    def vest(self) -> bool | None:
        return self.present("vest")

    @vest.setter
    def vest(self, value: bool | None) -> None:
        self._overwrite("vest", value, self.confidence_for("vest"))

    @property
    def helmet_confidence(self) -> float:
        return self.confidence_for("helmet")

    @helmet_confidence.setter
    def helmet_confidence(self, value: float) -> None:
        self._overwrite("helmet", self.present("helmet"), value)

    @property
    def vest_confidence(self) -> float:
        return self.confidence_for("vest")

    @vest_confidence.setter
    def vest_confidence(self, value: float) -> None:
        self._overwrite("vest", self.present("vest"), value)

    # ── violations ───────────────────────────────────────────────────────
    def missing(self, required: Sequence[str] | None = None) -> list[str]:
        """Required items that are *known* to be absent."""
        wanted = list(required) if required is not None else settings.required_ppe_items
        return [item for item in wanted if self.present(item) is False]

    @property
    def violations(self) -> list[str]:
        return self.missing()

    @property
    def is_violation(self) -> bool:
        return bool(self.violations)

    @property
    def confidence(self) -> float:
        """Confidence in the violation finding."""
        scores = [self.confidence_for(item) for item in self.violations]
        return max(scores) if scores else 0.0

    def summary(self, required: Sequence[str] | None = None) -> str:
        """Human-readable line, e.g. ``Helmet: YES | Safety Vest: NO``."""
        wanted = list(required) if required is not None else settings.required_ppe_items
        # Items the detector spoke about but the site does not require still
        # belong in the summary — an operator reading an incident wants the
        # whole picture, not just the rule that fired.
        for item in self.items:
            if item not in wanted:
                wanted.append(item)
        if not wanted:
            return "No PPE items assessed"

        def render(value: bool | None) -> str:
            return "YES" if value else ("NO" if value is False else "UNKNOWN")

        return " | ".join(
            f"{item_label(item)}: {render(self.present(item))}" for item in wanted
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "method": self.method,
            "items": {
                item: {
                    "present": s.present,
                    "confidence": round(s.confidence, 3),
                    "source": s.source,
                    **(
                        {
                            "validation": s.validation,
                            "validation_confidence": round(s.validation_confidence, 3),
                            "validation_method": s.validation_method,
                            "detector_confidence": round(s.detector_confidence, 3),
                        }
                        if s.validation
                        else {}
                    ),
                }
                for item, s in self.items.items()
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PPEAssessment(person_id={self.person_id}, method={self.method!r}, "
            f"{self.summary()})"
        )


class PPEDetector(ABC):
    """Interface for PPE assessment."""

    method: PPEMethod = "disabled"
    description: str = ""

    @abstractmethod
    def assess(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        detections: list[Detection],
        scope: str = "",
    ) -> dict[int, PPEAssessment]:
        """Assess every track. Returns ``{track_id: assessment}``.

        `scope` namespaces the per-person hard-hat cache. Track IDs are per
        tracker, so a camera and an upload job both counting from 1 would read
        each other's verdicts without it.
        """

    def info(self) -> dict[str, Any]:
        return {"method": self.method, "description": self.description}


# ══════════════════════════════════════════════════════════════════════════
#  Hard-hat validation
# ══════════════════════════════════════════════════════════════════════════
def validate_helmets(
    frame: np.ndarray,
    results: Mapping[int, PPEAssessment],
    tracks: Sequence[Track],
    validator: TrackedHardHatValidator | None = None,
    scope: str = "",
) -> None:
    """Second-stage check on every helmet a detector claimed to see.

    Runs *after* association, on the already-attributed item, so the crop it
    classifies is the headwear this person was found to be wearing — nothing
    about tracking or association changes here. Each assessment is updated in
    place, so the corrected verdict is what the rules, the events, the evidence
    and the render all see: there is no second version of the truth.

    The box preferred is the detector's own helmet box. Where there is not one
    — a heuristic estimate, or an absence inferred from omission — the head
    region of the person box is used instead, which is the same region the
    heuristic estimator measured.
    """
    if not results or not tracks:
        return
    validator = validator or resolve_hardhat_validator()
    if validator.method == "disabled":
        # Switched off by configuration, which means the stage does not run —
        # not that it runs and refuses everything. The detector's verdict
        # stands, and `resolve_hardhat_validator` has already logged what that
        # costs. `unavailable` is the other thing entirely: the stage should
        # have run, could not, and therefore certifies nothing.
        return
    validator.begin_frame()

    by_id = {track.track_id: track for track in tracks}
    for track_id, assessment in results.items():
        track = by_id.get(track_id)
        if track is None:
            continue
        for item in VALIDATED_ITEMS:
            status = assessment.items.get(item)
            if status is None or status.present is not True:
                continue
            box = status.bbox or body_region_box(track.bbox, item)
            result = validator.validate_track(frame, box, track_id, scope=scope)
            assessment.apply_validation(item, result)
            assessment.detail[f"{item}_validation"] = result.as_dict()

    validator.forget(scope, keep=set(by_id))


# ══════════════════════════════════════════════════════════════════════════
#  Association
# ══════════════════════════════════════════════════════════════════════════
def associate_items(
    person_boxes: np.ndarray,
    item_boxes: np.ndarray,
    item_names: Sequence[str],
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Attribute each PPE box to the person wearing it.

    Returns ``(owner, score)``: for every item box, the index of its owning
    person (``-1`` when unattributed) and the match score.

    Two signals decide it, and both are necessary:

    * **Containment** — the fraction of the *item* box inside the person box.
      IoU is the wrong measure here: a helmet is a tiny fraction of a body, so
      its IoU with a whole person is small no matter how certainly it is being
      worn. Containment stays near 1.0 for a worn item and collapses for a
      helmet sitting on a bench two metres away.
    * **Body band** — where the item sits vertically within the person box,
      compared with where that item belongs (:data:`ITEM_SPECS`). Without it, a
      helmet held at waist height, or a vest box that happens to overlap the
      person standing behind its wearer, would both be accepted.
    """
    limit = settings.ppe_association_threshold if threshold is None else threshold
    n, m = len(person_boxes), len(item_boxes)
    owner = np.full(m, -1, dtype=np.int32)
    score = np.zeros(m, dtype=np.float32)
    if n == 0 or m == 0:
        return owner, score

    persons = np.asarray(person_boxes, dtype=np.float32).reshape(-1, 4)
    items = np.asarray(item_boxes, dtype=np.float32).reshape(-1, 4)

    px1, py1, px2, py2 = (persons[:, i][:, None] for i in range(4))
    ix1, iy1, ix2, iy2 = (items[:, i][None, :] for i in range(4))

    inter_w = np.clip(np.minimum(px2, ix2) - np.maximum(px1, ix1), 0.0, None)
    inter_h = np.clip(np.minimum(py2, iy2) - np.maximum(py1, iy1), 0.0, None)
    item_area = np.maximum((ix2 - ix1) * (iy2 - iy1), 1e-6)
    containment = (inter_w * inter_h) / item_area           # (N, M)

    person_height = np.maximum(py2 - py1, 1e-6)
    item_centre_y = (iy1 + iy2) / 2.0
    relative = (item_centre_y - py1) / person_height         # (N, M)

    band_low = np.array(
        [ITEM_SPECS[name].band[0] if name in ITEM_SPECS else 0.0 for name in item_names],
        dtype=np.float32,
    )[None, :]
    band_high = np.array(
        [ITEM_SPECS[name].band[1] if name in ITEM_SPECS else 1.0 for name in item_names],
        dtype=np.float32,
    )[None, :]

    outside = np.maximum(band_low - relative, relative - band_high)
    affinity = np.clip(1.0 - np.maximum(outside, 0.0) / BAND_TOLERANCE, 0.0, 1.0)

    combined = containment * affinity
    best = np.argmax(combined, axis=0)
    rows = np.arange(m)
    best_containment = containment[best, rows]
    best_score = combined[best, rows]

    accepted = (best_containment >= limit) & (best_score > 0.0)
    owner[accepted] = best[accepted].astype(np.int32)
    score[accepted] = best_score[accepted]
    return owner, score


# ══════════════════════════════════════════════════════════════════════════
#  Disabled
# ══════════════════════════════════════════════════════════════════════════
class NullPPEDetector(PPEDetector):
    """Reports everything as undetermined. Used when PPE checks are off."""

    method = "disabled"
    description = "PPE assessment disabled (no model, heuristic not permitted)"

    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    def assess(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        detections: list[Detection],
        scope: str = "",
    ) -> dict[int, PPEAssessment]:
        return {t.track_id: PPEAssessment(person_id=t.track_id) for t in tracks}

    def info(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "description": self.description,
            "reason": self.reason,
        }


# ══════════════════════════════════════════════════════════════════════════
#  Model-based
# ══════════════════════════════════════════════════════════════════════════
class ModelPPEDetector(PPEDetector):
    """PPE detection from a dedicated trained model.

    The model runs on whichever inference backend can take it, resolved by the
    shared registry — this class contains no engine-specific code and no
    duplicate decode/NMS logic.
    """

    method = "model"

    def __init__(
        self,
        model_path: Path | None = None,
        labels_path: Path | None = None,
        detector: Detector | None = None,
    ) -> None:
        self.model_path = model_path or settings.ppe_model_file
        self.labels_path = labels_path or settings.ppe_labels_file
        self.detector: Detector | None = detector
        self.class_map: PPEClassMap | None = None
        self.attempts: list[tuple[str, str]] = []
        self.description = f"Trained PPE model ({self.model_path.name})"

    # ── lifecycle ────────────────────────────────────────────────────────
    @classmethod
    def probe(cls, model_path: Path | None = None) -> tuple[bool, str, dict[str, Any]]:
        """Report whether a PPE model is installed and interpretable."""
        model = model_path or settings.ppe_model_file
        detail: dict[str, Any] = {
            "model_path": str(model),
            "model_present": model.exists(),
            "model_format": model.suffix.lstrip(".") or "unknown",
        }
        if not model.exists():
            return (
                False,
                f"no PPE model at {model} — run 'python scripts/fetch_models.py "
                f"--ppe' to download and export one, or set PPE_MODEL_PATH",
                detail,
            )
        detail["model_size_mb"] = round(model.stat().st_size / 1e6, 1)
        labels = settings.ppe_labels_file
        detail["labels_path"] = str(labels)
        if not labels.exists():
            return (
                False,
                f"PPE model {model.name} has no label file at {labels}; class "
                f"indices cannot be interpreted",
                detail,
            )
        return True, "", detail

    def load(self) -> None:
        from backend.inference.postprocess import load_labels
        from backend.inference.registry import build_detector

        available, reason, _ = self.probe(self.model_path)
        if not available:
            raise BackendUnavailable(reason)

        labels = load_labels(self.labels_path)
        if not labels:
            raise BackendUnavailable(
                f"PPE label file {self.labels_path} is empty; class indices "
                f"cannot be interpreted"
            )
        try:
            self.class_map = PPEClassMap.build(labels, settings.ppe_class_map)
        except ClassMapError as exc:
            raise BackendUnavailable(f"invalid PPE_CLASS_MAP: {exc}") from exc

        if not self.class_map.covered_items():
            raise BackendUnavailable(
                f"PPE model {self.model_path.name} exposes no recognisable PPE "
                f"classes (got {labels}); map them with PPE_CLASS_MAP"
            )

        self.detector, self.attempts = build_detector(
            self.model_path,
            self.labels_path,
            conf_threshold=settings.ppe_intake_confidence_threshold,
            iou_threshold=settings.nms_iou_threshold,
            purpose="PPE",
        )
        self.description = (
            f"Trained PPE model ({self.model_path.name}) on "
            f"{self.detector.name} backend"
        )
        if self.class_map.unmapped:
            logger.warning(
                "PPE model classes not recognised and therefore ignored: %s. "
                "Map them with PPE_CLASS_MAP if they matter.",
                ", ".join(self.class_map.unmapped),
            )
        missing_coverage = [
            item
            for item in settings.required_ppe_items
            if item not in self.class_map.covered_items()
        ]
        if missing_coverage:
            logger.warning(
                "REQUIRED_PPE asks for %s but the PPE model has no class for "
                "%s — those items stay undetermined and cannot raise a "
                "violation.",
                ", ".join(settings.required_ppe_items), ", ".join(missing_coverage),
            )
        logger.info(
            "PPE model ready: %s (%d classes: %s) via %s backend "
            "@ present>=%.2f absent>=%.2f (intake %.2f)",
            self.model_path.name, len(labels), ", ".join(labels),
            self.detector.name, settings.ppe_confidence_threshold,
            settings.ppe_absence_confidence_threshold,
            settings.ppe_intake_confidence_threshold,
        )

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()

    # ── assessment ───────────────────────────────────────────────────────
    def assess(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        detections: list[Detection],
        scope: str = "",
    ) -> dict[int, PPEAssessment]:
        results = {
            t.track_id: PPEAssessment(person_id=t.track_id, method="model")
            for t in tracks
        }
        if not tracks or self.detector is None or self.class_map is None:
            return results

        result = self.detector.infer(frame)

        # Split the model's output into PPE items and its own person boxes.
        item_boxes: list[tuple[float, float, float, float]] = []
        item_names: list[str] = []
        item_present: list[bool] = []
        item_scores: list[float] = []
        model_persons: list[tuple[float, float, float, float]] = []

        for detection in result.detections:
            role = self.class_map.role_of(detection.label)
            if role is None:
                continue
            if detection.confidence < _threshold_for(role):
                continue
            if role.kind == "person":
                model_persons.append(detection.bbox)
                continue
            if role.item is None:
                continue
            item_boxes.append(detection.bbox)
            item_names.append(role.item)
            item_present.append(role.present)
            item_scores.append(detection.confidence)

        person_boxes = np.array([t.bbox for t in tracks], dtype=np.float32)
        owner, score = associate_items(
            person_boxes, np.array(item_boxes, dtype=np.float32).reshape(-1, 4), item_names
        )

        for index, person_index in enumerate(owner):
            if person_index < 0:
                continue
            assessment = results[tracks[int(person_index)].track_id]
            assessment.set(
                item_names[index],
                item_present[index],
                item_scores[index],
                source="detected",
                bbox=item_boxes[index],
            )
            assessment.detail.setdefault("association", {})[item_names[index]] = round(
                float(score[index]), 3
            )

        # Which people did the PPE model itself see? Only for those can the
        # *absence* of a positive class be read as the item being missing.
        seen = self._people_seen_by_model(person_boxes, model_persons)

        for position, track in enumerate(tracks):
            assessment = results[track.track_id]
            assessment.detail["height_px"] = round(track.height, 1)
            assessment.detail["model_saw_person"] = bool(seen[position])
            if track.height < _min_height():
                assessment.detail["skipped"] = "person too small to assess"
                continue
            self._infer_absences(assessment, saw_person=bool(seen[position]))

        # A detected helmet is a detected head covering. Whether it is PPE is a
        # separate question, asked last so it overrules everything above it.
        validate_helmets(frame, results, tracks, scope=scope)

        return results

    def _people_seen_by_model(
        self, person_boxes: np.ndarray, model_persons: Sequence[Any]
    ) -> np.ndarray:
        """Per-track flag: did the PPE model detect a person here too?

        Uses the Mojo-accelerated IoU kernel — box-to-box overlap between two
        sets of person boxes is exactly what it is for. When the PPE model has
        no person class at all, every track counts as seen, because the model
        was never in a position to say otherwise.
        """
        if not model_persons:
            has_person_class = any(
                r.kind == "person" for r in (self.class_map.roles.values() if self.class_map else [])
            )
            return np.full(len(person_boxes), not has_person_class, dtype=bool)

        from backend.inference.mojo.bridge import get_bridge

        overlap = get_bridge().iou_matrix(
            person_boxes, np.array(model_persons, dtype=np.float32).reshape(-1, 4)
        )
        return overlap.max(axis=1) >= PERSON_MATCH_IOU

    def _infer_absences(self, assessment: PPEAssessment, saw_person: bool) -> None:
        """Fill in items the model can only report by omission.

        A model with an explicit ``no_helmet`` class says absence directly and
        nothing is inferred. A model that only labels helmets has to be read the
        other way round: no helmet box on a person it clearly detected means the
        helmet is missing. That inference is recorded at a lower confidence and
        marked ``source="inferred"`` so it is never confused with a positive
        detection.
        """
        if self.class_map is None or not saw_person:
            return
        for item in self.class_map.covered_items():
            if assessment.present(item) is not None:
                continue
            if self.class_map.has_absence_class(item):
                # The model would have said so. Silence is genuinely unknown.
                continue
            if not self.class_map.has_presence_class(item):
                continue
            assessment.set(
                item, False, INFERRED_ABSENCE_CONFIDENCE, source="inferred"
            )

    # ── introspection ────────────────────────────────────────────────────
    def info(self) -> dict[str, Any]:
        detector = self.detector
        _, reason, probe_detail = self.probe(self.model_path)
        info: dict[str, Any] = {
            "method": self.method,
            "description": self.description,
            "model": self.model_path.name,
            "model_path": str(self.model_path),
            "model_present": self.model_path.exists(),
            "model_format": self.model_path.suffix.lstrip(".") or "unknown",
            "model_loaded": bool(detector and detector.is_loaded),
            "labels_path": str(self.labels_path),
            "backend": detector.name if detector else "none",
            "backend_detail": detector.stats() if detector else {},
            "confidence_threshold": settings.ppe_confidence_threshold,
            "absence_confidence_threshold": settings.ppe_absence_confidence_threshold,
            "intake_confidence_threshold": settings.ppe_intake_confidence_threshold,
            "association_threshold": settings.ppe_association_threshold,
            "min_person_height_px": settings.ppe_min_person_height,
            "required_ppe": settings.required_ppe_items,
            "fallback_chain": [{"backend": n, "reason": r} for n, r in self.attempts],
            **probe_detail,
        }
        if reason:
            info["warning"] = reason
        if self.class_map is not None:
            info.update(self.class_map.describe())
            info["unsupported_required_items"] = [
                item
                for item in settings.required_ppe_items
                if item not in self.class_map.covered_items()
            ]
        return info


# ══════════════════════════════════════════════════════════════════════════
#  Heuristic
# ══════════════════════════════════════════════════════════════════════════
class HeuristicPPEEstimator(PPEDetector):
    """Colour/region PPE estimator used when no PPE model is installed.

    Head region  = top 24% of the person box, central 64% horizontally.
    Torso region = 26%–62% of box height, central 76% horizontally.

    A helmet reads as a *uniform, saturated or very bright* cap of colour over
    the head region; high-visibility vests read as saturated yellow-green or
    orange over the torso. Both tests also require the region to be
    substantially covered, which rejects small bright background patches.

    It can only ever speak to helmet and vest — the two items with a distinctive
    colour signature at surveillance resolution. Gloves, boots, goggles, masks
    and harnesses need a trained model, and are left undetermined here rather
    than guessed at.
    """

    method = "heuristic"
    description = (
        "HSV colour/region heuristic (no PPE model installed) — advisory only"
    )

    #: The only items a colour test can speak to.
    SUPPORTED_ITEMS = ("helmet", "vest")

    # OpenCV HSV: H 0-179, S 0-255, V 0-255.
    HELMET_HUE_BANDS = (
        (0, 10),     # red
        (10, 30),    # orange / yellow
        (30, 45),    # yellow-green
        (100, 135),  # blue
        (170, 180),  # red wrap-around
    )
    # Measured, not guessed: saturated safety orange lands at OpenCV H~13
    # (BGR 20,120,255), so a band starting at 15 misses it entirely. Skin tone
    # also falls in H 5-20, which is why the saturation floor below is high —
    # high-vis fabric sits at S>150 while skin sits nearer S 40-120.
    VEST_HUE_BANDS = (
        (5, 22),    # safety orange
        (22, 45),   # high-vis yellow-green
        (45, 70),   # high-vis green
    )

    def __init__(
        self,
        helmet_coverage: float = 0.30,
        vest_coverage: float = 0.22,
        reason: str = "",
    ) -> None:
        self.helmet_coverage = helmet_coverage
        self.vest_coverage = vest_coverage
        self.reason = reason

    # ── region helpers ───────────────────────────────────────────────────
    @staticmethod
    def _crop(frame: np.ndarray, box: tuple[float, float, float, float],
              y0: float, y1: float, x_inset: float) -> np.ndarray | None:
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = box
        bw, bh = bx2 - bx1, by2 - by1
        if bw <= 1 or bh <= 1:
            return None
        inset = bw * x_inset / 2.0
        x1 = int(max(0, min(w - 1, bx1 + inset)))
        x2 = int(max(0, min(w, bx2 - inset)))
        ry1 = int(max(0, min(h - 1, by1 + bh * y0)))
        ry2 = int(max(0, min(h, by1 + bh * y1)))
        if x2 - x1 < 4 or ry2 - ry1 < 4:
            return None
        return frame[ry1:ry2, x1:x2]

    @classmethod
    def _band_mask(cls, hsv: np.ndarray, bands, s_min: int, v_min: int) -> np.ndarray:
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        mask = np.zeros(h.shape, dtype=bool)
        for lo, hi in bands:
            mask |= (h >= lo) & (h < hi)
        return mask & (s >= s_min) & (v >= v_min)

    # ── tests ────────────────────────────────────────────────────────────
    def _helmet(self, region: np.ndarray) -> tuple[bool | None, float, dict[str, Any]]:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        s, v = hsv[..., 1], hsv[..., 2]

        coloured = self._band_mask(hsv, self.HELMET_HUE_BANDS, s_min=95, v_min=90)
        # A white helmet is the opposite signature: bright and desaturated.
        white = (s < 45) & (v > 185)
        candidate = coloured | white
        coverage = float(candidate.mean())

        # Uniformity separates a helmet from patterned hair/background.
        if candidate.sum() >= 12:
            hue_spread = float(np.std(hsv[..., 0][candidate]))
        else:
            hue_spread = 180.0

        detail = {
            "head_coverage": round(coverage, 3),
            "hue_spread": round(hue_spread, 2),
            "white_fraction": round(float(white.mean()), 3),
        }

        if coverage >= self.helmet_coverage and hue_spread < 26:
            confidence = min(
                HEURISTIC_CONFIDENCE_CEILING, 0.34 + coverage * 0.55
            )
            return True, confidence, detail
        if coverage < self.helmet_coverage * 0.45:
            # Convincingly no helmet-like colour over the head.
            confidence = min(HEURISTIC_CONFIDENCE_CEILING, 0.36 + (1 - coverage) * 0.22)
            return False, confidence, detail
        # In between: say so rather than guessing.
        return None, 0.0, detail

    def _vest(self, region: np.ndarray) -> tuple[bool | None, float, dict[str, Any]]:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        high_vis = self._band_mask(hsv, self.VEST_HUE_BANDS, s_min=130, v_min=110)
        coverage = float(high_vis.mean())
        detail = {"torso_coverage": round(coverage, 3)}

        if coverage >= self.vest_coverage:
            confidence = min(HEURISTIC_CONFIDENCE_CEILING, 0.36 + coverage * 0.5)
            return True, confidence, detail
        if coverage < self.vest_coverage * 0.4:
            confidence = min(HEURISTIC_CONFIDENCE_CEILING, 0.38 + (1 - coverage) * 0.2)
            return False, confidence, detail
        return None, 0.0, detail

    # ── interface ────────────────────────────────────────────────────────
    def assess(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        detections: list[Detection],
        scope: str = "",
    ) -> dict[int, PPEAssessment]:
        results: dict[int, PPEAssessment] = {}
        for track in tracks:
            assessment = PPEAssessment(person_id=track.track_id, method="heuristic")
            if track.height < _min_height():
                assessment.detail = {
                    "skipped": "person too small to assess",
                    "height_px": round(track.height, 1),
                }
                results[track.track_id] = assessment
                continue

            head = self._crop(frame, track.bbox, 0.0, 0.24, 0.36)
            torso = self._crop(frame, track.bbox, 0.26, 0.62, 0.24)
            detail: dict[str, Any] = {"height_px": round(track.height, 1)}

            if head is not None:
                present, confidence, info = self._helmet(head)
                assessment.set("helmet", present, confidence, source="heuristic")
                detail.update(info)
            if torso is not None:
                present, confidence, info = self._vest(torso)
                assessment.set("vest", present, confidence, source="heuristic")
                detail.update(info)

            assessment.detail = detail
            results[track.track_id] = assessment

        # A colour estimate is even less able to tell a hard hat from a cap
        # than the trained detector is, so the same second stage applies.
        validate_helmets(frame, results, tracks, scope=scope)
        return results

    def info(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "description": self.description,
            "model_present": settings.ppe_model_file.exists(),
            "model_path": str(settings.ppe_model_file),
            "model_loaded": False,
            "backend": "opencv-hsv",
            "confidence_threshold": settings.ppe_confidence_threshold,
            "confidence_ceiling": HEURISTIC_CONFIDENCE_CEILING,
            "covered_items": list(self.SUPPORTED_ITEMS),
            "required_ppe": settings.required_ppe_items,
            "unsupported_required_items": [
                item
                for item in settings.required_ppe_items
                if item not in self.SUPPORTED_ITEMS
            ],
            "reason": self.reason,
        }


# ══════════════════════════════════════════════════════════════════════════
#  Resolution
# ══════════════════════════════════════════════════════════════════════════
_detector: PPEDetector | None = None


def ppe_diagnostics() -> dict[str, Any]:
    """The full, honest PPE picture for `GET /api/diagnostics` and `make
    diagnostics`.

    The same keys are present whichever detector is active, so a missing model
    reads as an actionable warning rather than a hole in the report. Never
    raises: diagnostics has to work precisely when something is broken.
    """
    base: dict[str, Any] = {
        "method": "unknown",
        "method_label": "UNKNOWN",
        "model_path": str(settings.ppe_model_file),
        "model_present": False,
        "model_format": settings.ppe_model_file.suffix.lstrip(".") or "unknown",
        "model_loaded": False,
        "labels_path": str(settings.ppe_labels_file),
        "labels_present": False,
        "backend": "none",
        "classes": [],
        "confidence_threshold": settings.ppe_confidence_threshold,
        "absence_confidence_threshold": settings.ppe_absence_confidence_threshold,
        "intake_confidence_threshold": settings.ppe_intake_confidence_threshold,
        "association_threshold": settings.ppe_association_threshold,
        "min_consecutive_frames": settings.ppe_min_consecutive_frames,
        "required_ppe": settings.required_ppe_items,
        "heuristic_allowed": settings.ppe_allow_heuristic,
        "warning": "",
    }
    try:
        base["model_present"] = settings.ppe_model_file.exists()
        base["labels_present"] = settings.ppe_labels_file.exists()
        detector = resolve_ppe_detector()
        base.update(detector.info())
        base["method"] = detector.method
    except Exception as exc:  # diagnostics must never be the thing that fails
        base["warning"] = f"PPE diagnostics failed: {exc}"

    base["method_label"] = METHOD_LABELS.get(base["method"], "UNKNOWN")
    if not base.get("warning"):
        base["warning"] = _ppe_warning(base)

    # The helmet verdict is only as good as the stage that validates it, so the
    # PPE report carries that stage's state rather than making an operator go
    # looking for it somewhere else.
    try:
        from backend.inference.hardhat import hardhat_diagnostics

        base["hardhat"] = hardhat_diagnostics()
    except Exception as exc:
        base["hardhat"] = {"method": "unknown", "warning": f"failed: {exc}"}
    return base


def _ppe_warning(info: dict[str, Any]) -> str:
    """One actionable sentence when PPE is not running on a trained model."""
    if info["method"] == "model" and info.get("model_loaded"):
        unsupported = info.get("unsupported_required_items") or []
        if unsupported:
            return (
                f"REQUIRED_PPE includes {', '.join(unsupported)} but the model "
                f"has no class for {'them' if len(unsupported) > 1 else 'it'}; "
                f"those items can never raise a violation."
            )
        return ""
    if not info["model_present"]:
        action = (
            f"No PPE model at {info['model_path']}. Run "
            f"'python scripts/fetch_models.py --ppe' to download and export "
            f"one, or point PPE_MODEL_PATH at an existing model."
        )
    elif not info["labels_present"]:
        action = (
            f"PPE model is present but its label file {info['labels_path']} is "
            f"missing, so class indices cannot be interpreted."
        )
    else:
        action = f"PPE model present but not loaded: {info.get('reason') or 'unknown reason'}"

    if info["method"] == "heuristic":
        return (
            f"{action} Running the HSV colour HEURISTIC FALLBACK instead — "
            f"results are advisory and capped at "
            f"{HEURISTIC_CONFIDENCE_CEILING:.2f} confidence."
        )
    return f"{action} PPE assessment is DISABLED; no PPE violations will fire."


def resolve_ppe_detector() -> PPEDetector:
    """Process-wide PPE detector, resolved once.

    Every camera thread, every uploaded-video job and the diagnostics endpoint
    share one instance. The PPE model is tens of megabytes and its ONNX Runtime
    session is safe to call concurrently, so loading a second copy per camera
    would cost memory and load time for nothing. Assessment itself holds no
    per-camera state.
    """
    global _detector
    if _detector is None:
        _detector = _build_ppe_detector()
    return _detector


def reset_ppe_detector() -> None:
    """Drop the singleton so the next call re-resolves (tests, config reload)."""
    global _detector
    if _detector is not None and hasattr(_detector, "close"):
        _detector.close()
    _detector = None


def _build_ppe_detector() -> PPEDetector:
    """Pick the best available PPE detector, always preferring a real model."""
    try:
        detector = ModelPPEDetector()
        detector.load()
        return detector
    except BackendUnavailable as exc:
        reason = str(exc)
    except Exception as exc:  # a bad model must not stop the pipeline
        reason = f"PPE model failed to load: {exc}"
        logger.warning("%s", reason)

    if settings.ppe_allow_heuristic:
        logger.warning(
            "PPE: %s — falling back to the colour heuristic. Results are "
            "advisory, capped at %.2f confidence and flagged "
            "method=heuristic everywhere they surface.",
            reason, HEURISTIC_CONFIDENCE_CEILING,
        )
        return HeuristicPPEEstimator(reason=reason)

    logger.warning(
        "PPE: %s — PPE assessment is DISABLED (PPE_ALLOW_HEURISTIC=false). "
        "No PPE violations will be raised.", reason,
    )
    return NullPPEDetector(reason=reason)
