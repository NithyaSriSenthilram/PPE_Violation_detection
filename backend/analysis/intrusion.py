"""Restricted-area intrusion.

Fires on the *transition* into a restricted zone, not on every frame spent
inside it. The zone's polygon is tested against the person's foot point, so a
person standing outside a floor-marked boundary does not trigger it by leaning
across.
"""

from __future__ import annotations

import numpy as np

from backend.analysis.base import Analyser, EventCandidate, FrameContext
from backend.analysis.zones import zones_for_points
from backend.events.types import EventType

#: Zone types that constitute an intrusion when entered.
INTRUSION_ZONE_TYPES = frozenset({"restricted"})

#: Frames a person must be inside before entry is confirmed. Guards against a
#: single jittery box on the boundary.
CONFIRM_FRAMES = 2


class IntrusionAnalyser(Analyser):
    """Raises RESTRICTED_AREA when a tracked person enters a restricted zone."""

    name = "intrusion"

    def __init__(self, confirm_frames: int = CONFIRM_FRAMES) -> None:
        self.confirm_frames = max(1, confirm_frames)
        # (track_id, zone_id) -> consecutive frames inside
        self._inside: dict[tuple[int, str], int] = {}
        self._announced: set[tuple[int, str]] = set()

    def forget(self, track_id: int) -> None:
        for key in [k for k in self._inside if k[0] == track_id]:
            del self._inside[key]
        self._announced = {k for k in self._announced if k[0] != track_id}

    def reset(self) -> None:
        self._inside.clear()
        self._announced.clear()

    def analyse(self, context: FrameContext) -> list[EventCandidate]:
        restricted = [
            z
            for z in context.zones
            if z.enabled and z.is_valid and z.zone_type in INTRUSION_ZONE_TYPES
        ]
        if not restricted or not context.tracks:
            return []

        points = np.array([t.foot_point for t in context.tracks], dtype=np.float32)
        memberships = zones_for_points(
            points, restricted, context.frame_width, context.frame_height
        )

        candidates: list[EventCandidate] = []
        seen_keys: set[tuple[int, str]] = set()

        for track, zone_ids in zip(context.tracks, memberships, strict=False):
            for zone_id in zone_ids:
                key = (track.track_id, zone_id)
                seen_keys.add(key)
                count = self._inside.get(key, 0) + 1
                self._inside[key] = count

                if count < self.confirm_frames or key in self._announced:
                    continue
                self._announced.add(key)

                zone = context.zone_by_id(zone_id)
                zone_name = zone.name if zone else zone_id
                candidates.append(
                    EventCandidate(
                        event_type=EventType.RESTRICTED_AREA,
                        confidence=track.confidence,
                        description=(
                            f"Person #{track.track_id:03d} entered restricted zone "
                            f"'{zone_name}'"
                        ),
                        person_id=track.track_id,
                        zone_id=zone_id,
                        zone_name=zone_name,
                        bbox=list(track.bbox),
                        metadata={
                            "zone_type": zone.zone_type if zone else "restricted",
                            "foot_point": [round(v, 1) for v in track.foot_point],
                            "frames_inside": count,
                            "detection_confidence": round(track.confidence, 3),
                        },
                    )
                )

        # Clear state for pairs that are no longer inside, so a re-entry later
        # raises a fresh event.
        for key in [k for k in self._inside if k not in seen_keys]:
            del self._inside[key]
            self._announced.discard(key)

        return candidates
