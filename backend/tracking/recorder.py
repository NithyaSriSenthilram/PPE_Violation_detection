"""Durable person-track summaries.

The tracker keeps per-frame state in memory and throws it away when a track
retires. That is right for tracking but useless for reporting: "how many
people passed through today" and "show me everything about person #14" both
need a record that outlives the frame.

:class:`TrackRecorder` accumulates one summary per tracked person and writes
it to `person_tracks` when the track retires (or periodically, so a long stay
is not lost if the process stops). Writes are batched — a row per frame would
dominate the SQLite write path for no gain.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from backend.db.base import session_scope
from backend.db.repository import upsert_person_track
from backend.inference.ppe import PPEAssessment
from backend.logging_conf import get_logger, throttled
from backend.tracking.bytetrack import Track

logger = get_logger(__name__)

#: Long-lived tracks are checkpointed this often (seconds).
CHECKPOINT_INTERVAL = 30.0

#: Tracks confirmed for fewer frames than this are not persisted — they are
#: almost always detector noise, and writing them would inflate every count.
MIN_FRAMES_TO_PERSIST = 5


@dataclass
class TrackSummary:
    """Accumulated observations for one person."""

    camera_id: str
    track_id: int
    first_seen: datetime
    last_seen: datetime
    job_id: str | None = None
    frames: int = 0
    confidence_sum: float = 0.0
    max_speed: float = 0.0
    dwell_seconds: float = 0.0
    zones: set[str] = field(default_factory=set)
    ppe_counts: dict[str, int] = field(default_factory=dict)
    events: int = 0
    first_monotonic: float = 0.0
    last_monotonic: float = 0.0
    persisted: bool = False
    last_checkpoint: float = 0.0

    @property
    def average_confidence(self) -> float:
        return self.confidence_sum / self.frames if self.frames else 0.0

    def ppe_summary(self) -> dict[str, Any]:
        total = self.ppe_counts.get("observations", 0)
        if not total:
            return {}
        return {
            "observations": total,
            "helmet_seen": self.ppe_counts.get("helmet_yes", 0),
            "helmet_missing": self.ppe_counts.get("helmet_no", 0),
            "vest_seen": self.ppe_counts.get("vest_yes", 0),
            "vest_missing": self.ppe_counts.get("vest_no", 0),
            "method": self.ppe_counts.get("method", "unknown"),
            "compliance_rate": round(
                1.0 - (self.ppe_counts.get("violations", 0) / total), 3
            ),
        }


class TrackRecorder:
    """Accumulates and persists person-track summaries for one source."""

    def __init__(self, camera_id: str, job_id: str | None = None) -> None:
        self.camera_id = camera_id
        self.job_id = job_id
        self._summaries: dict[int, TrackSummary] = {}
        self._written = 0

    # ── observation ──────────────────────────────────────────────────────
    def observe(
        self,
        tracks: Sequence[Track],
        *,
        timestamp: float,
        speeds: Mapping[int, float] | None = None,
        zone_map: Mapping[int, list[str]] | None = None,
        ppe: Mapping[int, PPEAssessment] | None = None,
    ) -> None:
        """Fold one frame's tracks into the running summaries."""
        now = datetime.now(UTC)
        speeds = speeds or {}
        zone_map = zone_map or {}
        ppe = ppe or {}

        for track in tracks:
            summary = self._summaries.get(track.track_id)
            if summary is None:
                summary = TrackSummary(
                    camera_id=self.camera_id,
                    track_id=track.track_id,
                    first_seen=now,
                    last_seen=now,
                    job_id=self.job_id,
                    first_monotonic=timestamp,
                    last_checkpoint=timestamp,
                )
                self._summaries[track.track_id] = summary

            summary.frames += 1
            summary.last_seen = now
            summary.last_monotonic = timestamp
            summary.confidence_sum += track.confidence
            summary.max_speed = max(summary.max_speed, speeds.get(track.track_id, 0.0))
            summary.dwell_seconds = max(0.0, timestamp - summary.first_monotonic)
            summary.zones.update(zone_map.get(track.track_id, []))

            assessment = ppe.get(track.track_id)
            if assessment is not None and assessment.method != "disabled":
                counts = summary.ppe_counts
                counts["observations"] = counts.get("observations", 0) + 1
                counts["method"] = assessment.method
                if assessment.helmet is True:
                    counts["helmet_yes"] = counts.get("helmet_yes", 0) + 1
                elif assessment.helmet is False:
                    counts["helmet_no"] = counts.get("helmet_no", 0) + 1
                if assessment.vest is True:
                    counts["vest_yes"] = counts.get("vest_yes", 0) + 1
                elif assessment.vest is False:
                    counts["vest_no"] = counts.get("vest_no", 0) + 1
                if assessment.is_violation:
                    counts["violations"] = counts.get("violations", 0) + 1

    def note_events(self, person_ids: Sequence[int | None]) -> None:
        """Attribute raised events to their person summaries."""
        for person_id in person_ids:
            if person_id is None:
                continue
            summary = self._summaries.get(person_id)
            if summary is not None:
                summary.events += 1

    # ── persistence ──────────────────────────────────────────────────────
    def flush(
        self, active_track_ids: set[int] | None = None, *, force: bool = False
    ) -> int:
        """Write summaries for retired (and checkpoint-due) tracks.

        Returns the number of rows written. Never raises — losing a reporting
        row must not disturb detection.
        """
        now = time.monotonic()
        to_write: list[TrackSummary] = []
        retired: list[int] = []

        for track_id, summary in self._summaries.items():
            gone = active_track_ids is not None and track_id not in active_track_ids
            due = (now - summary.last_checkpoint) >= CHECKPOINT_INTERVAL
            if force or gone or due:
                if summary.frames >= MIN_FRAMES_TO_PERSIST:
                    to_write.append(summary)
                summary.last_checkpoint = now
                if force or gone:
                    retired.append(track_id)

        for track_id in retired:
            self._summaries.pop(track_id, None)

        if not to_write:
            return 0

        try:
            with session_scope() as session:
                for summary in to_write:
                    upsert_person_track(
                        session,
                        camera_id=summary.camera_id,
                        track_id=summary.track_id,
                        first_seen=summary.first_seen,
                        last_seen=summary.last_seen,
                        frames_seen=summary.frames,
                        max_speed=round(summary.max_speed, 3),
                        avg_confidence=round(summary.average_confidence, 4),
                        dwell_seconds=round(summary.dwell_seconds, 2),
                        zones_entered=sorted(summary.zones),
                        ppe_summary=summary.ppe_summary(),
                        event_count=summary.events,
                        job_id=summary.job_id,
                    )
                    summary.persisted = True
            self._written += len(to_write)
            return len(to_write)
        except Exception as exc:
            throttled(
                logger,
                f"track-persist-{self.camera_id}",
                f"Could not persist person tracks for {self.camera_id}: {exc}",
                level=30,
                interval=60.0,
            )
            return 0

    @property
    def written(self) -> int:
        return self._written

    @property
    def tracked(self) -> int:
        return len(self._summaries)
