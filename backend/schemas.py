"""Pydantic request/response models.

These are the API contract. ORM objects are never returned directly — every
endpoint serialises through a model here, which keeps internal columns
(stored_path, last_error internals) from leaking and gives the frontend a
stable shape.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from backend.events.types import EventStatus, EventType, Severity, is_advisory, label_for

SourceType = Literal["rtsp", "webcam", "file", "http"]
ZoneType = Literal["restricted", "monitored", "crowd", "loitering"]

# Schemes we will hand to OpenCV. Anything else (file://, ftp://, ...) is
# rejected so a camera record cannot be used to read arbitrary local paths.
_ALLOWED_STREAM_SCHEMES = {"rtsp", "rtsps", "http", "https"}
_NAME_RE = re.compile(r"^[\w\s\-\.\(\)/#&,]{1,120}$", re.UNICODE)


class ORMModel(BaseModel):
    """Base for models populated from SQLAlchemy instances."""

    model_config = ConfigDict(from_attributes=True)


# ══════════════════════════════════════════════════════════════════════════
#  Cameras
# ══════════════════════════════════════════════════════════════════════════
class CameraBase(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    location: Annotated[str, Field(max_length=200)] = ""
    source_type: SourceType
    stream_url: Annotated[str, Field(min_length=1, max_length=2000)]
    enabled: bool = True
    analytics_enabled: bool = True
    settings_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        v = v.strip()
        if not _NAME_RE.match(v):
            raise ValueError("name contains unsupported characters")
        return v

    @model_validator(mode="after")
    def _validate_source(self) -> CameraBase:
        url = self.stream_url.strip()

        if self.source_type == "webcam":
            # OpenCV takes an integer device index for local capture devices.
            if not url.isdigit():
                raise ValueError("webcam stream_url must be a device index, e.g. '0'")
        elif self.source_type == "file":
            # Validated against UPLOAD_DIR at the service layer, where the
            # filesystem is actually consulted; here we only block traversal.
            if ".." in url or url.startswith("/"):
                raise ValueError(
                    "file stream_url must be a relative name inside the upload directory"
                )
        else:
            parsed = urlparse(url)
            if parsed.scheme.lower() not in _ALLOWED_STREAM_SCHEMES:
                raise ValueError(
                    f"stream_url scheme must be one of {sorted(_ALLOWED_STREAM_SCHEMES)}"
                )
            if not parsed.netloc:
                raise ValueError("stream_url is missing a host")

        object.__setattr__(self, "stream_url", url)
        return self


class CameraCreate(CameraBase):
    """Payload for POST /api/cameras."""


class CameraUpdate(BaseModel):
    """Partial update — every field optional (PUT /api/cameras/{id})."""

    name: Annotated[str, Field(min_length=1, max_length=120)] | None = None
    location: Annotated[str, Field(max_length=200)] | None = None
    source_type: SourceType | None = None
    stream_url: Annotated[str, Field(min_length=1, max_length=2000)] | None = None
    enabled: bool | None = None
    analytics_enabled: bool | None = None
    settings_json: dict[str, Any] | None = None


class CameraOut(ORMModel):
    camera_id: str
    name: str
    location: str
    source_type: str
    stream_url: str
    status: str
    enabled: bool
    analytics_enabled: bool
    settings_json: dict[str, Any] = Field(default_factory=dict)
    last_frame_at: datetime | None = None
    last_error: str | None = None
    fps: float = 0.0
    people_count: int = 0
    created_at: datetime
    updated_at: datetime
    zone_count: int = 0


class CameraRuntime(BaseModel):
    """Live pipeline telemetry, merged into the camera list by the API."""

    camera_id: str
    status: str
    fps: float = 0.0
    people_count: int = 0
    active_tracks: int = 0
    inference_ms: float = 0.0
    processing_ms: float = 0.0
    frames_processed: int = 0
    frames_dropped: int = 0
    backend: str | None = None
    last_error: str | None = None
    uptime_seconds: float = 0.0


# ══════════════════════════════════════════════════════════════════════════
#  Zones
# ══════════════════════════════════════════════════════════════════════════
Point = Annotated[list[float], Field(min_length=2, max_length=2)]


class ZoneBase(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    zone_type: ZoneType = "restricted"
    polygon: Annotated[list[Point], Field(min_length=3, max_length=64)]
    enabled: bool = True
    loitering_threshold: Annotated[float, Field(ge=1.0, le=86400.0)] | None = None
    crowd_threshold: Annotated[int, Field(ge=1, le=10000)] | None = None
    colour: Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")] = "#f43f5e"

    @field_validator("polygon")
    @classmethod
    def _normalised(cls, v: list[list[float]]) -> list[list[float]]:
        for x, y in v:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    "polygon points must be normalised to 0..1 "
                    "(divide pixel coords by frame width/height)"
                )
        return [[float(x), float(y)] for x, y in v]


class ZoneCreate(ZoneBase):
    camera_id: str


class ZoneUpdate(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=120)] | None = None
    zone_type: ZoneType | None = None
    polygon: Annotated[list[Point], Field(min_length=3, max_length=64)] | None = None
    enabled: bool | None = None
    loitering_threshold: Annotated[float, Field(ge=1.0, le=86400.0)] | None = None
    crowd_threshold: Annotated[int, Field(ge=1, le=10000)] | None = None
    colour: Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")] | None = None


class ZoneOut(ORMModel):
    zone_id: str
    camera_id: str
    name: str
    zone_type: str
    polygon: list[list[float]]
    enabled: bool
    loitering_threshold: float | None = None
    crowd_threshold: int | None = None
    colour: str
    created_at: datetime
    updated_at: datetime


# ══════════════════════════════════════════════════════════════════════════
#  Events
# ══════════════════════════════════════════════════════════════════════════
class EventOut(ORMModel):
    event_id: str
    event_type: str
    camera_id: str | None = None
    person_id: int | None = None
    zone_id: str | None = None
    severity: str
    confidence: float
    status: str
    description: str
    bbox: list[float] | None = None
    snapshot_path: str | None = None
    clip_path: str | None = None
    detection_metadata: dict[str, Any] = Field(default_factory=dict)
    job_id: str | None = None
    video_timestamp: float | None = None
    timestamp: datetime
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    notes: str | None = None

    # Denormalised for display; filled by the API layer.
    camera_name: str | None = None
    zone_name: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        return label_for(self.event_type)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def advisory(self) -> bool:
        """True when this result must be presented as an AI inference."""
        return is_advisory(self.event_type)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_snapshot(self) -> bool:
        return bool(self.snapshot_path)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_clip(self) -> bool:
        return bool(self.clip_path)


class EventPage(BaseModel):
    """Paginated event listing."""

    items: list[EventOut]
    total: int
    limit: int
    offset: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total


class EventStatusUpdate(BaseModel):
    status: EventStatus
    notes: Annotated[str, Field(max_length=2000)] | None = None


class EventFilters(BaseModel):
    """Query parameters shared by /api/events and /api/alerts."""

    camera_id: str | None = None
    event_type: EventType | None = None
    severity: Severity | None = None
    status: EventStatus | None = None
    person_id: int | None = None
    job_id: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    search: Annotated[str, Field(max_length=200)] | None = None
    limit: Annotated[int, Field(ge=1, le=500)] = 50
    offset: Annotated[int, Field(ge=0)] = 0


# ══════════════════════════════════════════════════════════════════════════
#  Statistics / analytics
# ══════════════════════════════════════════════════════════════════════════
class KpiSummary(BaseModel):
    people_detected: int = 0
    active_cameras: int = 0
    total_cameras: int = 0
    ppe_violations: int = 0
    intrusions: int = 0
    suspicious_events: int = 0
    open_alerts: int = 0
    critical_alerts: int = 0
    events_today: int = 0


class TimeBucket(BaseModel):
    bucket: str
    total: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    by_severity: dict[str, int] = Field(default_factory=dict)


class CameraActivity(BaseModel):
    camera_id: str
    camera_name: str
    event_count: int = 0
    people_detected: int = 0
    status: str = "offline"


class StatisticsOut(BaseModel):
    kpis: KpiSummary
    by_type: dict[str, int] = Field(default_factory=dict)
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)
    timeline: list[TimeBucket] = Field(default_factory=list)
    camera_activity: list[CameraActivity] = Field(default_factory=list)
    range_start: datetime
    range_end: datetime
    granularity: str


# ══════════════════════════════════════════════════════════════════════════
#  Video analysis jobs
# ══════════════════════════════════════════════════════════════════════════
DetectionProfile = Literal["fast", "standard", "thorough", "ppe_only", "security_only"]


class VideoJobOut(ORMModel):
    job_id: str
    filename: str
    profile: str
    status: str
    progress: float
    message: str
    total_frames: int
    processed_frames: int
    duration_seconds: float
    fps: float
    resolution: str
    events_created: int
    people_detected: int
    annotated_path: str | None = None
    output: dict[str, Any] | None = None
    zone_camera_id: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TimelineEntry(BaseModel):
    """One incident, positioned on the source timeline.

    `video_timestamp` is the seek target: clicking the entry moves the player
    to the moment that produced the event.
    """

    event_id: str
    event_type: str
    label: str
    severity: str
    person_id: int | None = None
    confidence: float
    video_timestamp: float | None = None
    snapshot_available: bool = False


class VideoJobResult(BaseModel):
    """The finished analysis, as the result screen consumes it.

    Carries API URLs only. A filesystem path would leak the host layout and
    would break the moment storage moved.
    """

    job_id: str
    filename: str
    status: str
    progress: float
    message: str

    annotated_ready: bool
    video_url: str | None = None
    download_url: str | None = None
    original_url: str | None = None

    duration_seconds: float = 0.0
    fps: float = 0.0
    frames: int = 0
    resolution: str = ""
    size_bytes: int = 0
    codec: str = ""
    has_audio: bool = False
    browser_compatible: bool = False
    #: Anything given up during encoding — missing audio, a downscale, a
    #: fallback codec. Surfaced rather than hidden.
    warnings: list[str] = Field(default_factory=list)

    summary: dict[str, Any] = Field(default_factory=dict)
    events_created: int = 0
    people_detected: int = 0
    timeline: list[TimelineEntry] = Field(default_factory=list)


class VideoUploadResponse(BaseModel):
    job_id: str
    filename: str
    size_bytes: int
    profile: str
    status: str
    message: str


# ══════════════════════════════════════════════════════════════════════════
#  Diagnostics / health
# ══════════════════════════════════════════════════════════════════════════
class BackendInfo(BaseModel):
    """Truthful report of one inference backend's state."""

    name: str
    available: bool
    active: bool = False
    reason: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    app: str
    version: str
    environment: str
    uptime_seconds: float
    database: bool
    inference_backend: str
    model_loaded: bool
    cameras_online: int
    cameras_total: int
    timestamp: datetime


class DiagnosticsOut(BaseModel):
    app: str
    version: str
    environment: str
    uptime_seconds: float
    platform: dict[str, Any]
    python_version: str
    database: dict[str, Any]
    inference: dict[str, Any]
    backends: list[BackendInfo]
    mojo: dict[str, Any]
    ppe: dict[str, Any]
    video_output: dict[str, Any] = Field(default_factory=dict)
    pipeline: dict[str, Any]
    config: dict[str, Any]
    timestamp: datetime


# ══════════════════════════════════════════════════════════════════════════
#  WebSocket envelopes
# ══════════════════════════════════════════════════════════════════════════
WsMessageType = Literal[
    "hello",
    "detections",
    "event",
    "camera_status",
    "job_progress",
    "stats",
    "pong",
    "error",
]


class BoxOut(BaseModel):
    """A single detection pushed to the dashboard for overlay rendering.

    Coordinates are normalised 0..1 so the browser can draw them over a video
    element of any size without knowing the source resolution.
    """

    track_id: int | None = None
    label: str = "person"
    confidence: float = 0.0
    bbox: list[float]
    helmet: bool | None = None
    vest: bool | None = None
    ppe_method: str | None = None
    violation: bool = False
    speed: float = 0.0
    state: str = "tracked"
    zones: list[str] = Field(default_factory=list)
    dwell_seconds: float = 0.0


class DetectionFrame(BaseModel):
    camera_id: str
    frame_index: int
    timestamp: datetime
    people_count: int
    boxes: list[BoxOut] = Field(default_factory=list)
    fps: float = 0.0
    inference_ms: float = 0.0
    processing_ms: float = 0.0
    backend: str = ""


class WsEnvelope(BaseModel):
    type: WsMessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: datetime


class ErrorResponse(BaseModel):
    """Uniform error body. Never contains a stack trace in production."""

    error: str
    detail: str | None = None
    request_id: str | None = None
