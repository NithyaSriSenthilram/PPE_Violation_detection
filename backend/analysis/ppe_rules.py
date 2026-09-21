"""PPE violation rules.

Turns per-frame :class:`PPEAssessment` results into events. Which items count
is configuration, not code: ``REQUIRED_PPE=helmet,vest`` is the default, and a
site that also mandates gloves and boots sets
``REQUIRED_PPE=helmet,vest,gloves,boots`` and supplies a model that emits those
classes. An item the detector cannot speak to stays undetermined and therefore
silent — it never produces a violation by default.

Three guards keep the alert stream meaningful:

* **Undetermined is not a violation.** ``helmet is None`` means the detector
  could not tell; only an explicit ``False`` counts.
* **Violations must persist.** A finding has to hold for
  ``PPE_MIN_CONSECUTIVE_FRAMES`` before it fires, so one bad frame — a head
  turned away, a moment of glare — does not raise an alert.
* **One situation, one alert.** A person missing several required items raises
  a single aggregate ``PPE_VIOLATION`` rather than one event per item: it is
  one thing for the operator to act on, and splitting it multiplies the noise
  for exactly the worst case. Firing state is per person, so a violation that
  continues does not re-fire until the person becomes compliant again — the
  event engine's cooldown and burst cap then sit behind this as further
  backstops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.analysis.base import Analyser, EventCandidate, FrameContext
from backend.config import settings
from backend.events.types import EventType
from backend.inference.ppe import PPEAssessment
from backend.inference.ppe_taxonomy import ITEM_SPECS, item_label

#: Consecutive frames a finding must hold before it becomes an event, when the
#: caller does not specify and no configured value applies.
DEFAULT_MIN_CONSECUTIVE = 5


def _event_type_for(item: str) -> str:
    """Dedicated event type for an item, or the aggregate PPE_VIOLATION."""
    spec = ITEM_SPECS.get(item)
    return (spec.event_type if spec and spec.event_type else None) or (
        EventType.PPE_VIOLATION
    )


@dataclass
class _PersonPPEState:
    """Per-person debounce and firing state."""

    #: item -> consecutive frames it has been observed missing
    missing_frames: dict[str, int] = field(default_factory=dict)
    #: keys already fired: an item name, or "a|b" for an aggregate
    fired: set[str] = field(default_factory=set)


class PPEAnalyser(Analyser):
    """Raises MISSING_HELMET / MISSING_VEST / PPE_VIOLATION."""

    name = "ppe"

    def __init__(
        self,
        min_consecutive: int | None = None,
        required: list[str] | None = None,
    ) -> None:
        self.min_consecutive = max(
            1, min_consecutive or settings.ppe_min_consecutive_frames
        )
        # Captured at construction so a running camera keeps one consistent
        # rule set; restarting the pipeline picks up a changed REQUIRED_PPE.
        self.required = list(required) if required is not None else (
            settings.required_ppe_items
        )
        self._state: dict[int, _PersonPPEState] = {}

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def reset(self) -> None:
        self._state.clear()

    # ── rule ─────────────────────────────────────────────────────────────
    def analyse(self, context: FrameContext) -> list[EventCandidate]:
        # Retire state for departed tracks first, and unconditionally: a camera
        # watching an empty scene reports no PPE at all, and returning early
        # before this point would leave every past person's debounce counters
        # in memory until someone next walked into frame.
        live_ids = {t.track_id for t in context.tracks}
        for stale in [tid for tid in self._state if tid not in live_ids]:
            del self._state[stale]

        if not context.ppe or not self.required:
            return []

        candidates: list[EventCandidate] = []

        for track in context.tracks:
            assessment: PPEAssessment | None = context.ppe.get(track.track_id)
            if assessment is None:
                continue

            state = self._state.setdefault(track.track_id, _PersonPPEState())
            confirmed: list[str] = []

            for item in self.required:
                is_missing = assessment.present(item) is False
                count = state.missing_frames.get(item, 0) + 1 if is_missing else 0
                state.missing_frames[item] = count
                if count >= self.min_consecutive:
                    confirmed.append(item)
                if not is_missing:
                    # Re-arm every finding that involved this item once the
                    # person is compliant again.
                    state.fired = {k for k in state.fired if item not in k.split("|")}

            if not confirmed:
                continue

            key = "|".join(sorted(confirmed))
            if key in state.fired:
                continue
            state.fired.add(key)
            candidates.append(
                self._candidate(track, assessment, confirmed, state)
            )

        return candidates

    # ── candidate construction ───────────────────────────────────────────
    def _candidate(
        self,
        track,
        assessment: PPEAssessment,
        confirmed: list[str],
        state: _PersonPPEState,
    ) -> EventCandidate:
        confidence = max(assessment.confidence_for(item) for item in confirmed)
        metadata = {
            "ppe_method": assessment.method,
            "ppe_required": list(self.required),
            "ppe_items": {
                item: {
                    "present": assessment.present(item),
                    "confidence": round(assessment.confidence_for(item), 3),
                    "source": assessment.status(item).source,
                    # Present only for items with a second stage. A helmet
                    # rejected as a cap has to be able to say so in the record,
                    # or the event reads as though nothing was on the head.
                    **(
                        {
                            "validation": status.validation,
                            "validation_confidence": round(
                                status.validation_confidence, 3
                            ),
                            "validation_method": status.validation_method,
                            "detector_confidence": round(
                                status.detector_confidence, 3
                            ),
                        }
                        if (status := assessment.status(item)).validation
                        else {}
                    ),
                }
                for item in dict.fromkeys([*self.required, *assessment.items])
            },
            # Retained for the dashboard, the API contract and stored events
            # written before PPE became multi-item.
            "helmet": assessment.helmet,
            "vest": assessment.vest,
            "helmet_confidence": round(assessment.helmet_confidence, 3),
            "vest_confidence": round(assessment.vest_confidence, 3),
            "summary": assessment.summary(self.required),
            "missing": list(confirmed),
            "frames_observed": max(
                state.missing_frames.get(item, 0) for item in confirmed
            ),
            **{f"ppe_{k}": v for k, v in assessment.detail.items()},
        }

        if len(confirmed) == 1:
            item = confirmed[0]
            description = (
                f"Person #{track.track_id:03d} has no "
                f"{item_label(item).lower()} ({assessment.summary(self.required)})"
            )
            event_type = _event_type_for(item)
        else:
            names = ", ".join(item_label(i).lower() for i in confirmed[:-1])
            description = (
                f"Person #{track.track_id:03d} is missing "
                f"{names} and {item_label(confirmed[-1]).lower()} "
                f"({assessment.summary(self.required)})"
            )
            event_type = EventType.PPE_VIOLATION

        return EventCandidate(
            event_type=event_type,
            confidence=confidence,
            description=description,
            person_id=track.track_id,
            bbox=list(track.bbox),
            metadata=metadata,
        )
