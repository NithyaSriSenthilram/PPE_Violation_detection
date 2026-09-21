"""The event engine.

Single funnel for every detection rule. Responsibilities, in order:

1. **Confidence gate** — drop candidates below `EVENT_MIN_CONFIDENCE`.
2. **Severity** — resolve from the configurable override map.
3. **Cooldown / debounce** — the heart of it. One person standing in a
   restricted zone for two minutes at 12 fps produces ~1,400 rule firings; the
   operator needs *one* alert. Suppression is keyed on
   ``(camera, event_type, person, zone)`` so two people, or one person in two
   zones, remain distinct events.
4. **Evidence** — snapshot now, clip queued around the event.
5. **Persistence** — one row per accepted event.
6. **Broadcast** — push to the dashboard over the event bus.

Analysers stay pure and testable because all of this lives here.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from backend.analysis.base import EventCandidate
from backend.config import settings
from backend.db.base import session_scope
from backend.db.models import Event
from backend.events.bus import get_bus
from backend.events.evidence import get_evidence_writer
from backend.events.types import (
    ADVISORY_NOTICE,
    EventStatus,
    Severity,
    default_severity,
    is_advisory,
    label_for,
)
from backend.logging_conf import ALERT_LEVEL, get_logger, throttled
from backend.video.ring_buffer import FrameRingBuffer

logger = get_logger(__name__)


@dataclass(slots=True)
class EventRecord:
    """A persisted event, in the shape the bus and API want."""

    event_id: str
    event_type: str
    camera_id: str | None
    person_id: int | None
    zone_id: str | None
    severity: str
    confidence: float
    status: str
    description: str
    bbox: list[float] | None
    snapshot_path: str | None
    clip_path: str | None
    detection_metadata: dict[str, Any]
    timestamp: datetime
    job_id: str | None = None
    video_timestamp: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "label": label_for(self.event_type),
            "camera_id": self.camera_id,
            "person_id": self.person_id,
            "zone_id": self.zone_id,
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "description": self.description,
            "bbox": self.bbox,
            "snapshot_path": self.snapshot_path,
            "clip_path": self.clip_path,
            "detection_metadata": self.detection_metadata,
            "job_id": self.job_id,
            "video_timestamp": self.video_timestamp,
            "timestamp": self.timestamp.isoformat(),
            "advisory": is_advisory(self.event_type),
        }


class CooldownRegistry:
    """Two-layer suppression.

    **Per-key debounce** stops one continuous violation from re-firing: the key
    includes the person, so two people in the same zone stay distinct.

    **Burst cap** is the backstop for when person identity itself is unstable.
    A crowded doorway, heavy occlusion or a scene cut all churn track IDs, and
    a new ID means a fresh debounce key — which is exactly how an operator ends
    up with hundreds of near-identical alerts. The cap limits events per
    (camera, type, zone) per minute regardless of who the tracker thinks it is
    looking at, and reports how many it held back so nothing is silently lost.
    """

    def __init__(
        self,
        default_seconds: float | None = None,
        max_per_minute: int | None = None,
    ) -> None:
        self.default_seconds = (
            settings.event_cooldown_seconds if default_seconds is None else default_seconds
        )
        self.max_per_minute = (
            settings.event_max_per_minute if max_per_minute is None else max_per_minute
        )
        self._last: dict[tuple[Any, ...], float] = {}
        self._suppressed: dict[tuple[Any, ...], int] = {}
        #: (scope, event_type, zone) -> recent accept timestamps
        self._recent: dict[tuple[Any, ...], deque[float]] = {}
        self._rate_suppressed: dict[tuple[Any, ...], int] = {}
        self._lock = threading.Lock()

    def _burst_key(self, key: tuple[Any, ...]) -> tuple[Any, ...]:
        """(scope, event_type, zone) — deliberately excludes the person."""
        # key is (scope, event_type, person_id, zone_id)
        return (key[0], key[1], key[3]) if len(key) >= 4 else key

    def within_burst_cap(self, key: tuple[Any, ...], now: float) -> bool:
        """True if the burst cap still has room. Records the accept."""
        if self.max_per_minute <= 0:
            return True
        burst_key = self._burst_key(key)
        window = self._recent.setdefault(burst_key, deque())
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self.max_per_minute:
            self._rate_suppressed[burst_key] = (
                self._rate_suppressed.get(burst_key, 0) + 1
            )
            return False
        window.append(now)
        return True

    def rate_suppressed_count(self, key: tuple[Any, ...]) -> int:
        return self._rate_suppressed.get(self._burst_key(key), 0)

    def allow(self, key: tuple[Any, ...], now: float, seconds: float | None = None) -> bool:
        """True if `key` may fire now; records the firing when it does."""
        window = self.default_seconds if seconds is None else seconds
        with self._lock:
            previous = self._last.get(key)
            if previous is not None and (now - previous) < window:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                return False
            self._last[key] = now
            return True

    def suppressed_count(self, key: tuple[Any, ...]) -> int:
        with self._lock:
            return self._suppressed.get(key, 0)

    def forget_person(self, camera_id: str, person_id: int) -> None:
        with self._lock:
            for key in [
                k for k in self._last if len(k) > 2 and k[0] == camera_id and k[2] == person_id
            ]:
                del self._last[key]
                self._suppressed.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._last.clear()
            self._suppressed.clear()
            self._recent.clear()
            self._rate_suppressed.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "tracked_keys": len(self._last),
                "total_suppressed": sum(self._suppressed.values()),
                "burst_suppressed": sum(self._rate_suppressed.values()),
                "max_per_minute": self.max_per_minute,
            }


class EventEngine:
    """Accepts candidates, produces persisted, deduplicated events."""

    def __init__(
        self,
        cooldown: CooldownRegistry | None = None,
        persist: bool = True,
        broadcast: bool = True,
        capture_evidence: bool = True,
    ) -> None:
        self.cooldown = cooldown or CooldownRegistry()
        self.persist = persist
        self.broadcast = broadcast
        self.capture_evidence = capture_evidence
        self._severity_overrides: dict[str, str] = {}
        self._accepted = 0
        self._rejected_confidence = 0
        self._rate_limited = 0
        self._counts: dict[str, int] = {}

    # ── configuration ────────────────────────────────────────────────────
    def set_severity_overrides(self, overrides: dict[str, str]) -> None:
        """Replace the severity map, e.g. from the SystemSetting table."""
        valid = {s.value for s in Severity}
        self._severity_overrides = {
            key: value for key, value in overrides.items() if value in valid
        }

    def severity_for(self, event_type: str) -> str:
        return self._severity_overrides.get(event_type) or default_severity(event_type)

    # ── main entry point ─────────────────────────────────────────────────
    def submit(
        self,
        candidates: Iterable[EventCandidate],
        *,
        camera_id: str | None,
        scope: str | None = None,
        frame: np.ndarray | None = None,
        annotated_frame: np.ndarray | None = None,
        buffer: FrameRingBuffer | None = None,
        fps: float = 12.0,
        monotonic_time: float | None = None,
        job_id: str | None = None,
        video_timestamp: float | None = None,
    ) -> list[EventRecord]:
        """Process a frame's candidates. Returns the accepted events.

        `scope` namespaces cooldown keys and evidence filenames. Live cameras
        leave it unset and are scoped by `camera_id`; uploaded-video analysis
        must pass its job scope, because `camera_id` is NULL for those events
        and two concurrent jobs would otherwise share one cooldown namespace
        and silently suppress each other's incidents.
        """
        now = time.monotonic() if monotonic_time is None else monotonic_time
        wall_time = datetime.now(UTC)
        accepted: list[EventRecord] = []

        for candidate in candidates:
            if candidate.confidence < settings.event_min_confidence:
                self._rejected_confidence += 1
                continue

            key = (candidate.event_type, candidate.person_id, candidate.zone_id)
            full_key = (scope or camera_id, *key)
            if not self.cooldown.allow(full_key, now):
                continue
            # Backstop for unstable person identity — see CooldownRegistry.
            if not self.cooldown.within_burst_cap(full_key, now):
                self._rate_limited += 1
                throttled(
                    logger,
                    f"burst-{full_key[0]}-{candidate.event_type}",
                    f"Burst cap reached for {candidate.event_type} on "
                    f"{full_key[0]}: holding further alerts this minute "
                    f"(cap {self.cooldown.max_per_minute}/min)",
                    level=30,
                    interval=60.0,
                )
                continue

            record = self._materialise(
                candidate,
                camera_id=camera_id,
                scope=scope or camera_id or "unscoped",
                wall_time=wall_time,
                monotonic_time=now,
                frame=annotated_frame if annotated_frame is not None else frame,
                buffer=buffer,
                fps=fps,
                job_id=job_id,
                video_timestamp=video_timestamp,
                suppressed=self.cooldown.suppressed_count(full_key),
            )
            accepted.append(record)
            self._accepted += 1
            self._counts[record.event_type] = self._counts.get(record.event_type, 0) + 1

            logger.log(
                ALERT_LEVEL,
                "%s — %s",
                label_for(record.event_type).upper(),
                record.description,
                extra={
                    "camera": camera_id or scope,
                    "event": record.event_type,
                    "person": record.person_id,
                },
            )

            if self.broadcast:
                get_bus().publish("event", record.to_payload(), camera_id=camera_id)

        return accepted

    # ── construction ─────────────────────────────────────────────────────
    def _materialise(
        self,
        candidate: EventCandidate,
        *,
        camera_id: str | None,
        scope: str,
        wall_time: datetime,
        monotonic_time: float,
        frame: np.ndarray | None,
        buffer: FrameRingBuffer | None,
        fps: float,
        job_id: str | None,
        video_timestamp: float | None,
        suppressed: int,
    ) -> EventRecord:
        import uuid

        event_id = str(uuid.uuid4())
        # Coerce the StrEnum to a plain str so persisted rows, JSON payloads
        # and stats dicts all carry the bare value rather than an enum repr.
        event_type = str(candidate.event_type)
        severity = str(self.severity_for(event_type))

        metadata: dict[str, Any] = {
            **candidate.metadata,
            "zone_name": candidate.zone_name,
            "duplicates_suppressed": suppressed,
            "burst_suppressed": self.cooldown.rate_suppressed_count(
                (scope, str(candidate.event_type), candidate.person_id, candidate.zone_id)
            ),
        }
        if is_advisory(event_type):
            metadata.setdefault("advisory", ADVISORY_NOTICE)

        snapshot_path: str | None = None
        clip_path: str | None = None
        if self.capture_evidence and frame is not None:
            writer = get_evidence_writer()
            snapshot_path = writer.write_snapshot(
                frame, scope, event_type, event_id, wall_time
            )
            if buffer is not None:
                clip_path = writer.schedule_clip(
                    buffer,
                    scope,
                    event_type,
                    event_id,
                    monotonic_time,
                    fps,
                    wall_time,
                )

        record = EventRecord(
            event_id=event_id,
            event_type=event_type,
            camera_id=camera_id,
            person_id=candidate.person_id,
            zone_id=candidate.zone_id,
            severity=severity,
            confidence=candidate.confidence,
            status=EventStatus.OPEN.value,
            description=candidate.description,
            bbox=candidate.bbox,
            snapshot_path=snapshot_path,
            clip_path=clip_path,
            detection_metadata=metadata,
            timestamp=wall_time,
            job_id=job_id,
            video_timestamp=video_timestamp,
        )

        if self.persist:
            self._save(record)
        return record

    @staticmethod
    def _save(record: EventRecord) -> None:
        """Insert the event. A database failure must not stop detection."""
        try:
            with session_scope() as session:
                session.add(
                    Event(
                        event_id=record.event_id,
                        event_type=record.event_type,
                        camera_id=record.camera_id,
                        person_id=record.person_id,
                        zone_id=record.zone_id,
                        severity=record.severity,
                        confidence=record.confidence,
                        status=record.status,
                        description=record.description,
                        bbox=record.bbox,
                        snapshot_path=record.snapshot_path,
                        clip_path=record.clip_path,
                        detection_metadata=record.detection_metadata,
                        job_id=record.job_id,
                        video_timestamp=record.video_timestamp,
                        timestamp=record.timestamp,
                    )
                )
        except Exception as exc:
            logger.error(
                "Failed to persist event %s (%s): %s",
                record.event_id, record.event_type, exc,
            )

    # ── introspection ────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        return {
            "accepted": self._accepted,
            "rejected_low_confidence": self._rejected_confidence,
            "rejected_burst_cap": self._rate_limited,
            "by_type": dict(self._counts),
            "cooldown": self.cooldown.stats(),
            "cooldown_seconds": self.cooldown.default_seconds,
            "min_confidence": settings.event_min_confidence,
            "severity_overrides": dict(self._severity_overrides),
        }


_engine: EventEngine | None = None


def get_event_engine() -> EventEngine:
    """Process-wide event engine."""
    global _engine
    if _engine is None:
        _engine = EventEngine()
    return _engine


def reset_event_engine() -> None:
    """Drop the singleton (tests)."""
    global _engine
    _engine = None
