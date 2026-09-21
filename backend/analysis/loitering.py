"""Loitering detection.

Accumulates, per (person, zone), the wall-clock time the person's foot point
has been inside the zone and raises one event when the configured dwell
threshold is crossed. Dwell is measured from frame timestamps rather than
frame counts so the threshold means the same thing at 5 fps and 25 fps, and on
a video analysed faster than real time.

The alert fires **once** per continuous stay. Leaving the zone for longer than
:data:`RESET_GRACE_SECONDS` clears the accumulator so a later return is
treated as a new stay; briefer gaps are tolerated, since a tracker blink
should not reset a 90-second dwell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.analysis.base import Analyser, EventCandidate, FrameContext
from backend.analysis.zones import zones_for_points
from backend.config import settings
from backend.events.types import EventType

#: Zone types that participate in loitering. `crowd` zones measure density.
LOITERING_ZONE_TYPES = frozenset({"loitering", "monitored", "restricted"})

#: How long a person may be absent from a zone before dwell resets.
RESET_GRACE_SECONDS = 3.0


@dataclass
class _Dwell:
    seconds: float = 0.0
    last_seen: float = 0.0
    alerted: bool = False


class LoiteringAnalyser(Analyser):
    """Raises LOITERING when dwell time in a zone exceeds its threshold."""

    name = "loitering"

    def __init__(self, default_threshold: float | None = None) -> None:
        self.default_threshold = (
            settings.loitering_threshold if default_threshold is None else default_threshold
        )
        self._dwell: dict[tuple[int, str], _Dwell] = {}

    def forget(self, track_id: int) -> None:
        for key in [k for k in self._dwell if k[0] == track_id]:
            del self._dwell[key]

    def reset(self) -> None:
        self._dwell.clear()

    def dwell_seconds(self, track_id: int, zone_id: str) -> float:
        """Current accumulated dwell — surfaced to the live dashboard."""
        entry = self._dwell.get((track_id, zone_id))
        return entry.seconds if entry else 0.0

    def analyse(self, context: FrameContext) -> list[EventCandidate]:
        zones = [
            z
            for z in context.zones
            if z.enabled and z.is_valid and z.zone_type in LOITERING_ZONE_TYPES
        ]
        now = context.timestamp

        # Expire accumulators for anyone who has left, then bail if idle.
        for key, entry in list(self._dwell.items()):
            if now - entry.last_seen > RESET_GRACE_SECONDS:
                del self._dwell[key]

        if not zones or not context.tracks:
            return []

        points = np.array([t.foot_point for t in context.tracks], dtype=np.float32)
        memberships = zones_for_points(
            points, zones, context.frame_width, context.frame_height
        )

        candidates: list[EventCandidate] = []
        for track, zone_ids in zip(context.tracks, memberships, strict=False):
            for zone_id in zone_ids:
                key = (track.track_id, zone_id)
                entry = self._dwell.get(key)
                if entry is None:
                    self._dwell[key] = _Dwell(seconds=0.0, last_seen=now)
                    continue

                # Only credit the elapsed interval since the last sighting, so
                # a tracker gap does not inflate dwell.
                delta = max(0.0, now - entry.last_seen)
                if delta <= RESET_GRACE_SECONDS:
                    entry.seconds += delta
                entry.last_seen = now

                zone = context.zone_by_id(zone_id)
                threshold = (
                    zone.loitering_threshold
                    if zone and zone.loitering_threshold
                    else self.default_threshold
                )
                if entry.seconds < threshold or entry.alerted:
                    continue

                entry.alerted = True
                zone_name = zone.name if zone else zone_id
                candidates.append(
                    EventCandidate(
                        event_type=EventType.LOITERING,
                        # Confidence grows with how far past the threshold the
                        # dwell has gone, capped — this is a timing fact, so it
                        # should read as high certainty once well past.
                        confidence=min(0.98, 0.6 + (entry.seconds / threshold - 1) * 0.3),
                        description=(
                            f"Person #{track.track_id:03d} has remained in "
                            f"'{zone_name}' for {entry.seconds:.0f}s "
                            f"(threshold {threshold:.0f}s)"
                        ),
                        person_id=track.track_id,
                        zone_id=zone_id,
                        zone_name=zone_name,
                        bbox=list(track.bbox),
                        metadata={
                            "dwell_seconds": round(entry.seconds, 1),
                            "threshold_seconds": threshold,
                            "zone_type": zone.zone_type if zone else "monitored",
                        },
                    )
                )
        return candidates
