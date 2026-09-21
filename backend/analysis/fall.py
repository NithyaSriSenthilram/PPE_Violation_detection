"""Possible-fall detection.

A three-stage temporal test, run per tracked person:

1. **Upright baseline** — the box has been taller than wide.
2. **Rapid vertical drop** — the box centre falls by more than
   ``FALL_VERTICAL_DROP`` of the person's own height inside
   :data:`DROP_WINDOW_SECONDS`.
3. **Horizontal persistence** — the box aspect ratio exceeds
   ``FALL_ASPECT_RATIO`` and *stays* there for ``FALL_CONFIRM_SECONDS``.

Requiring all three in order is what separates a fall from someone crouching
(no drop), sitting down (no horizontal box), or a detector glitch (does not
persist).

This is an inference from box geometry, and it is labelled that way
everywhere: the event type is ``POSSIBLE_FALL``, it is in
``ADVISORY_EVENT_TYPES``, and the API attaches an advisory notice. It is not a
medical determination and must not be presented as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from backend.analysis.base import Analyser, EventCandidate, FrameContext
from backend.config import settings
from backend.events.types import ADVISORY_NOTICE, EventType
from backend.tracking import Track

#: Window in which the vertical drop must occur.
DROP_WINDOW_SECONDS = 1.2

#: Aspect ratio below which a person counts as upright for stage 1.
UPRIGHT_ASPECT_MAX = 0.85

#: How long the state machine waits for stage 3 before giving up on a drop.
DROP_EXPIRY_SECONDS = 3.0


class _Stage(Enum):
    UPRIGHT = auto()
    DROPPED = auto()   # stage 2 satisfied, waiting for horizontal persistence
    CONFIRMED = auto()


@dataclass
class _FallState:
    stage: _Stage = _Stage.UPRIGHT
    upright_samples: int = 0
    drop_at: float = 0.0
    drop_magnitude: float = 0.0
    horizontal_since: float | None = None
    reported: bool = False
    detail: dict[str, float] = field(default_factory=dict)


class FallAnalyser(Analyser):
    """Raises POSSIBLE_FALL. Advisory: an AI detection, not a diagnosis."""

    name = "fall"

    def __init__(
        self,
        aspect_threshold: float | None = None,
        drop_fraction: float | None = None,
        confirm_seconds: float | None = None,
    ) -> None:
        self.aspect_threshold = (
            settings.fall_aspect_ratio if aspect_threshold is None else aspect_threshold
        )
        self.drop_fraction = (
            settings.fall_vertical_drop if drop_fraction is None else drop_fraction
        )
        self.confirm_seconds = (
            settings.fall_confirm_seconds if confirm_seconds is None else confirm_seconds
        )
        self._state: dict[int, _FallState] = {}

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def reset(self) -> None:
        self._state.clear()

    # ── stage helpers ────────────────────────────────────────────────────
    def _vertical_drop(self, track: Track, now: float) -> tuple[float, float]:
        """Largest downward centre movement within the drop window.

        Returns ``(drop_as_fraction_of_height, reference_height)``.
        """
        history = track.history
        if len(history) < 3:
            return 0.0, track.height

        recent = [s for s in history if now - s[0] <= DROP_WINDOW_SECONDS]
        if len(recent) < 3:
            return 0.0, track.height

        # Highest point (smallest y) in the window vs the latest y.
        highest = min(recent, key=lambda s: s[2])
        latest = recent[-1]
        if latest[0] <= highest[0]:
            return 0.0, track.height  # the peak is the newest sample

        drop_px = latest[2] - highest[2]
        reference = max(1e-3, max(s[3] for s in recent))
        return max(0.0, drop_px / reference), reference

    def analyse(self, context: FrameContext) -> list[EventCandidate]:
        live = {t.track_id for t in context.tracks}
        for stale in [tid for tid in self._state if tid not in live]:
            del self._state[stale]

        now = context.timestamp
        candidates: list[EventCandidate] = []

        for track in context.tracks:
            state = self._state.setdefault(track.track_id, _FallState())
            aspect = track.aspect_ratio
            is_horizontal = aspect >= self.aspect_threshold

            # Stage 1 — accumulate an upright baseline.
            if aspect <= UPRIGHT_ASPECT_MAX:
                state.upright_samples += 1

            # Stage 2 — look for a rapid drop, only once upright was seen.
            if state.stage is _Stage.UPRIGHT and state.upright_samples >= 3:
                drop, reference = self._vertical_drop(track, now)
                if drop >= self.drop_fraction:
                    state.stage = _Stage.DROPPED
                    state.drop_at = now
                    state.drop_magnitude = drop
                    state.detail = {
                        "drop_fraction": round(drop, 3),
                        "reference_height_px": round(reference, 1),
                    }

            # A drop that never becomes horizontal is not a fall.
            if (
                state.stage is _Stage.DROPPED
                and now - state.drop_at > DROP_EXPIRY_SECONDS
            ):
                state.stage = _Stage.UPRIGHT
                state.upright_samples = 0
                state.horizontal_since = None

            # Stage 3 — horizontal, and staying that way.
            if state.stage is _Stage.DROPPED:
                if is_horizontal:
                    if state.horizontal_since is None:
                        state.horizontal_since = now
                    elapsed = now - state.horizontal_since
                    if elapsed >= self.confirm_seconds and not state.reported:
                        state.stage = _Stage.CONFIRMED
                        state.reported = True
                        candidates.append(
                            EventCandidate(
                                event_type=EventType.POSSIBLE_FALL,
                                confidence=min(
                                    0.9,
                                    0.45
                                    + state.drop_magnitude * 0.6
                                    + min(0.2, elapsed * 0.1),
                                ),
                                description=(
                                    f"Person #{track.track_id:03d} may have fallen — "
                                    f"rapid drop of "
                                    f"{state.drop_magnitude * 100:.0f}% of body height "
                                    f"followed by a horizontal posture held for "
                                    f"{elapsed:.1f}s"
                                ),
                                person_id=track.track_id,
                                bbox=list(track.bbox),
                                metadata={
                                    **state.detail,
                                    "aspect_ratio": round(aspect, 2),
                                    "aspect_threshold": self.aspect_threshold,
                                    "horizontal_seconds": round(elapsed, 2),
                                    "advisory": ADVISORY_NOTICE,
                                    "detection_method": "temporal_geometry",
                                },
                            )
                        )
                else:
                    state.horizontal_since = None

            # Recovery — standing back up re-arms the detector.
            if state.stage is _Stage.CONFIRMED and aspect <= UPRIGHT_ASPECT_MAX:
                state.stage = _Stage.UPRIGHT
                state.upright_samples = 0
                state.horizontal_since = None
                state.reported = False

        return candidates
