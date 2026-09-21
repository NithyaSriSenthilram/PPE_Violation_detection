"""ORM models.

Enumerations are stored as plain strings rather than DB enums so new event
types, severities and source types can be added without a migration — the
prompt requires the taxonomy to stay open-ended.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from backend.db.base import Base, UTCDateTime, utcnow


def _uuid() -> str:
    return str(uuid.uuid4())


class Camera(Base):
    """A video source: uploaded file, local webcam or RTSP/HTTP stream."""

    __tablename__ = "cameras"

    camera_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    location: Mapped[str] = mapped_column(String(200), default="")
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)  # rtsp|webcam|file|http
    stream_url: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="offline", index=True)

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    analytics_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # Per-camera threshold overrides; empty dict means "inherit global config".
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Runtime telemetry, refreshed by the pipeline worker.
    last_frame_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    people_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    zones: Mapped[list[Zone]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", lazy="selectin"
    )
    events: Mapped[list[Event]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Camera {self.camera_id[:8]} {self.name!r} {self.status}>"


class Zone(Base):
    """A polygon region of interest drawn over a camera's frame.

    Coordinates are stored **normalised** (0..1) so a zone stays correct when
    the stream resolution changes or the UI renders at any size.
    """

    __tablename__ = "zones"

    zone_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    camera_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cameras.camera_id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    zone_type: Mapped[str] = mapped_column(String(30), default="restricted")
    # [[x, y], ...] normalised, >= 3 points (validated in the Pydantic layer)
    polygon: Mapped[list[list[float]]] = mapped_column(JSON, nullable=False)

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    loitering_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    crowd_threshold: Mapped[int | None] = mapped_column(Integer, nullable=True)
    colour: Mapped[str] = mapped_column(String(16), default="#f43f5e")

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    camera: Mapped[Camera] = relationship(back_populates="zones")

    __table_args__ = (Index("ix_zones_camera_enabled", "camera_id", "enabled"),)


class Event(Base):
    """A detected incident. The canonical record behind alerts and evidence."""

    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    camera_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("cameras.camera_id", ondelete="CASCADE"), index=True
    )
    person_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    zone_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    severity: Mapped[str] = mapped_column(String(10), default="MEDIUM", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="OPEN", index=True)

    description: Mapped[str] = mapped_column(Text, default="")
    # [x1, y1, x2, y2] in source-frame pixels
    bbox: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    clip_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Detector/analyser provenance: backend used, method (model|heuristic),
    # measured speed, dwell time, crowd counts, etc.
    detection_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Set when the event came from an uploaded-video analysis run.
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    video_timestamp: Mapped[float | None] = mapped_column(Float, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    camera: Mapped[Camera | None] = relationship(back_populates="events")

    __table_args__ = (
        Index("ix_events_ts_type", "timestamp", "event_type"),
        Index("ix_events_camera_ts", "camera_id", "timestamp"),
        Index("ix_events_status_sev", "status", "severity"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Event {self.event_type} cam={self.camera_id} sev={self.severity}>"


class PersonTrack(Base):
    """Persisted summary of one tracked person's lifetime on a camera.

    The live tracker keeps per-frame state in memory; this table is the
    durable roll-up used for historical search and analytics.
    """

    __tablename__ = "person_tracks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    camera_id: Mapped[str] = mapped_column(String(36), index=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    first_seen: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    last_seen: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    frames_seen: Mapped[int] = mapped_column(Integer, default=0)

    max_speed: Mapped[float] = mapped_column(Float, default=0.0)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    dwell_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    zones_entered: Mapped[list[str]] = mapped_column(JSON, default=list)
    ppe_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    event_count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("camera_id", "track_id", "first_seen", name="uq_track_identity"),
        Index("ix_tracks_camera_seen", "camera_id", "last_seen"),
    )


class VideoJob(Base):
    """An uploaded-video analysis run, processed off the request thread."""

    __tablename__ = "video_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    profile: Mapped[str] = mapped_column(String(40), default="standard")

    status: Mapped[str] = mapped_column(String(20), default="QUEUED", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")

    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    processed_frames: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    resolution: Mapped[str] = mapped_column(String(20), default="")

    events_created: Mapped[int] = mapped_column(Integer, default=0)
    people_detected: Mapped[int] = mapped_column(Integer, default=0)
    #: Project-relative path to the full annotated render — the primary output
    #: of an analysis, distinct from the per-event evidence clips.
    annotated_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: What the encoder actually produced: codec, frames written, resolution,
    #: whether audio survived, and any warnings. Kept as one JSON blob rather
    #: than six columns because it is read as a unit and never queried on.
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    #: Optional camera whose zones apply to this footage. Uploaded video has no
    #: camera of its own, so without this no zone geometry exists and the
    #: zone-dependent rules can never fire.
    zone_camera_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class SystemSetting(Base):
    """Runtime-editable key/value settings that override `.env` defaults."""

    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
