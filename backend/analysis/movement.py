"""Abnormal / high-speed movement detection.

Speed is measured in **body heights per second**, not pixels per second. That
normalisation is what makes one threshold work across cameras and across the
frame: a person near the lens covers many more pixels per step than someone at
the far end of a corridor, but both cover roughly the same number of their own
body heights. Pixel thresholds have to be retuned per camera and still
misfire with depth.

Terminology is deliberately neutral — ``ABNORMAL_MOVEMENT``. Moving quickly is
not evidence of wrongdoing; it is a cue for a human to look.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from backend.analysis.base import Analyser, EventCandidate, FrameContext
from backend.config import settings
from backend.events.types import EventType
from backend.tracking import Track

#: Window over which displacement is measured. Long enough to be robust to
#: box jitter, short enough not to average a sprint away.
SPEED_WINDOW_SECONDS = 0.6

#: Consecutive qualifying samples required before an event is raised.
SUSTAIN_SAMPLES = 3


@dataclass
class _MovementState:
    fast_samples: int = 0
    alerted: bool = False
    peak_speed: float = 0.0


def measure_speed(track: Track, window: float = SPEED_WINDOW_SECONDS) -> float:
    """Speed in body heights per second over the last `window` seconds.

    Returns 0.0 when there is not yet enough history to measure.
    """
    history = track.history
    if len(history) < 2:
        return 0.0

    latest_t, latest_x, latest_y, latest_h = history[-1]
    # Walk back to the first sample at least `window` old.
    oldest = history[0]
    for sample in reversed(history):
        if latest_t - sample[0] >= window:
            oldest = sample
            break
    else:
        oldest = history[0]

    dt = latest_t - oldest[0]
    if dt <= 1e-3:
        return 0.0

    distance = math.hypot(latest_x - oldest[1], latest_y - oldest[2])
    # Average the two heights: perspective changes as someone approaches.
    reference_height = max(1e-3, (latest_h + oldest[3]) / 2.0)
    return (distance / reference_height) / dt


class MovementAnalyser(Analyser):
    """Raises ABNORMAL_MOVEMENT for sustained high-speed motion."""

    name = "movement"

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = (
            settings.running_threshold if threshold is None else threshold
        )
        self._state: dict[int, _MovementState] = {}

    def forget(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def reset(self) -> None:
        self._state.clear()

    def speed_for(self, track: Track) -> float:
        """Current speed for a track — also used for the live overlay."""
        return measure_speed(track)

    def analyse(self, context: FrameContext) -> list[EventCandidate]:
        live = {t.track_id for t in context.tracks}
        for stale in [tid for tid in self._state if tid not in live]:
            del self._state[stale]

        candidates: list[EventCandidate] = []
        for track in context.tracks:
            speed = measure_speed(track)
            state = self._state.setdefault(track.track_id, _MovementState())
            state.peak_speed = max(state.peak_speed, speed)

            if speed >= self.threshold:
                state.fast_samples += 1
            else:
                state.fast_samples = 0
                # Re-arm once they slow down, so a later sprint alerts again.
                if speed < self.threshold * 0.6:
                    state.alerted = False

            if state.fast_samples < SUSTAIN_SAMPLES or state.alerted:
                continue

            state.alerted = True
            over = speed / self.threshold
            candidates.append(
                EventCandidate(
                    event_type=EventType.ABNORMAL_MOVEMENT,
                    confidence=min(0.95, 0.5 + (over - 1.0) * 0.4),
                    description=(
                        f"Person #{track.track_id:03d} is moving at "
                        f"{speed:.1f} body-heights/s "
                        f"(threshold {self.threshold:.1f}) — high-speed movement"
                    ),
                    person_id=track.track_id,
                    bbox=list(track.bbox),
                    metadata={
                        "speed_body_heights_per_second": round(speed, 2),
                        "threshold": self.threshold,
                        "peak_speed": round(state.peak_speed, 2),
                        "window_seconds": SPEED_WINDOW_SECONDS,
                        "measure": "body_heights_per_second",
                    },
                )
            )
        return candidates
