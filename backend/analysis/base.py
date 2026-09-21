"""Shared types for behaviour analysis.

Analysers are pure: they read tracks plus their own rolling state and return
:class:`EventCandidate` objects. They never touch the database, write files or
emit alerts — the event engine owns deduplication, severity, persistence and
evidence. That separation is what makes each rule unit-testable in isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from backend.analysis.zones import ResolvedZone
from backend.tracking import Track


@dataclass(slots=True)
class EventCandidate:
    """A rule firing. Becomes an Event only after the engine accepts it."""

    event_type: str
    confidence: float
    description: str
    person_id: int | None = None
    zone_id: str | None = None
    zone_name: str | None = None
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    #: Distinguishes concurrent firings of one rule for one person, so the
    #: engine's cooldown keys stay precise (e.g. loitering in two zones).
    @property
    def dedupe_key(self) -> tuple[Any, ...]:
        return (self.event_type, self.person_id, self.zone_id)


@dataclass(slots=True)
class FrameContext:
    """Everything an analyser may look at for one frame."""

    camera_id: str
    frame_index: int
    timestamp: float          # monotonic seconds, for durations
    wall_time: float          # epoch seconds, for records
    frame_width: int
    frame_height: int
    tracks: list[Track]
    zones: list[ResolvedZone]
    #: track_id -> PPE assessment, when a PPE detector ran on this frame.
    ppe: dict[int, Any] = field(default_factory=dict)

    def zone_by_id(self, zone_id: str) -> ResolvedZone | None:
        return next((z for z in self.zones if z.zone_id == zone_id), None)


class Analyser(ABC):
    """Base class for a behaviour rule."""

    #: Stable identifier used in logs and diagnostics.
    name: str = "analyser"

    @abstractmethod
    def analyse(self, context: FrameContext) -> list[EventCandidate]:
        """Evaluate the rule for one frame."""

    def forget(self, track_id: int) -> None:  # noqa: B027 - optional hook
        """Drop per-track state when a track retires.

        Deliberately concrete and empty: most analysers hold no per-track
        state, and forcing them all to implement a no-op would be noise.
        """

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Clear all state (camera restart, tests). Optional, like `forget`."""
