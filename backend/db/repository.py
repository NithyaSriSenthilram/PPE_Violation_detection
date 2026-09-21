"""Data access helpers.

Query logic lives here rather than in route handlers, so the API layer stays
thin and the same queries are reusable from the pipeline and the job runner.
Everything is expressed in SQLAlchemy Core/ORM constructs — no dialect-specific
SQL — so switching `DATABASE_URL` to PostgreSQL needs no changes here.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from backend.analysis.zones import ResolvedZone
from backend.db.base import session_scope
from backend.db.models import Camera, Event, PersonTrack, SystemSetting, VideoJob, Zone
from backend.events.types import EventType, Severity
from backend.schemas import EventFilters


# ══════════════════════════════════════════════════════════════════════════
#  Zones
# ══════════════════════════════════════════════════════════════════════════
def to_resolved_zone(zone: Zone) -> ResolvedZone:
    """ORM row → the pipeline's hit-testable representation."""
    return ResolvedZone(
        zone_id=zone.zone_id,
        name=zone.name,
        zone_type=zone.zone_type,
        polygon_norm=list(zone.polygon or []),
        enabled=zone.enabled,
        loitering_threshold=zone.loitering_threshold,
        crowd_threshold=zone.crowd_threshold,
        colour=zone.colour,
    )


def load_zones_for_camera(camera_id: str) -> list[ResolvedZone]:
    """Enabled zones for a camera, ready for the pipeline."""
    with session_scope() as session:
        rows = (
            session.execute(
                select(Zone).where(Zone.camera_id == camera_id, Zone.enabled.is_(True))
            )
            .scalars()
            .all()
        )
        return [to_resolved_zone(z) for z in rows]


# ══════════════════════════════════════════════════════════════════════════
#  Cameras
# ══════════════════════════════════════════════════════════════════════════
def list_cameras(session: Session, *, enabled_only: bool = False) -> list[Camera]:
    statement = select(Camera).order_by(Camera.created_at)
    if enabled_only:
        statement = statement.where(Camera.enabled.is_(True))
    return list(session.execute(statement).scalars().all())


def get_camera(session: Session, camera_id: str) -> Camera | None:
    return session.get(Camera, camera_id)


def zone_counts(session: Session) -> dict[str, int]:
    """camera_id → number of zones, for the camera listing."""
    rows = session.execute(
        select(Zone.camera_id, func.count(Zone.zone_id)).group_by(Zone.camera_id)
    ).all()
    return dict(rows)


# ══════════════════════════════════════════════════════════════════════════
#  Events
# ══════════════════════════════════════════════════════════════════════════
def _apply_event_filters(statement: Select, filters: EventFilters) -> Select:
    if filters.camera_id:
        statement = statement.where(Event.camera_id == filters.camera_id)
    if filters.event_type:
        statement = statement.where(Event.event_type == filters.event_type.value)
    if filters.severity:
        statement = statement.where(Event.severity == filters.severity.value)
    if filters.status:
        statement = statement.where(Event.status == filters.status.value)
    if filters.person_id is not None:
        statement = statement.where(Event.person_id == filters.person_id)
    if filters.job_id:
        statement = statement.where(Event.job_id == filters.job_id)
    if filters.start:
        statement = statement.where(Event.timestamp >= filters.start)
    if filters.end:
        statement = statement.where(Event.timestamp <= filters.end)
    if filters.search:
        pattern = f"%{filters.search.strip()}%"
        statement = statement.where(
            or_(Event.description.ilike(pattern), Event.event_type.ilike(pattern))
        )
    return statement


def query_events(
    session: Session, filters: EventFilters
) -> tuple[list[Event], int]:
    """Filtered, paginated events plus the total matching count."""
    base = _apply_event_filters(select(Event), filters)
    total = session.execute(
        _apply_event_filters(select(func.count(Event.event_id)), filters)
    ).scalar_one()
    rows = (
        session.execute(
            base.order_by(Event.timestamp.desc())
            .limit(filters.limit)
            .offset(filters.offset)
        )
        .scalars()
        .all()
    )
    return list(rows), int(total)


def get_event(session: Session, event_id: str) -> Event | None:
    return session.get(Event, event_id)


def camera_names(session: Session, camera_ids: Sequence[str]) -> dict[str, str]:
    """camera_id → name, for denormalising event listings."""
    ids = [c for c in set(camera_ids) if c]
    if not ids:
        return {}
    rows = session.execute(
        select(Camera.camera_id, Camera.name).where(Camera.camera_id.in_(ids))
    ).all()
    return dict(rows)


def zone_names(session: Session, zone_ids: Sequence[str]) -> dict[str, str]:
    ids = [z for z in set(zone_ids) if z]
    if not ids:
        return {}
    rows = session.execute(
        select(Zone.zone_id, Zone.name).where(Zone.zone_id.in_(ids))
    ).all()
    return dict(rows)


# ══════════════════════════════════════════════════════════════════════════
#  Statistics
# ══════════════════════════════════════════════════════════════════════════
#: Event types that roll up into each KPI card.
PPE_TYPES = (
    EventType.PPE_VIOLATION.value,
    EventType.MISSING_HELMET.value,
    EventType.MISSING_VEST.value,
)
SUSPICIOUS_TYPES = (
    EventType.LOITERING.value,
    EventType.ABNORMAL_MOVEMENT.value,
    EventType.POSSIBLE_FALL.value,
    EventType.CROWD_ANOMALY.value,
)


def count_events(
    session: Session,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    types: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    severities: Sequence[str] | None = None,
) -> int:
    statement = select(func.count(Event.event_id))
    if since:
        statement = statement.where(Event.timestamp >= since)
    if until:
        statement = statement.where(Event.timestamp <= until)
    if types:
        statement = statement.where(Event.event_type.in_(list(types)))
    if statuses:
        statement = statement.where(Event.status.in_(list(statuses)))
    if severities:
        statement = statement.where(Event.severity.in_(list(severities)))
    return int(session.execute(statement).scalar_one())


def group_events(
    session: Session,
    column,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, int]:
    """Counts grouped by one column over an optional time range."""
    statement = select(column, func.count(Event.event_id)).group_by(column)
    if since:
        statement = statement.where(Event.timestamp >= since)
    if until:
        statement = statement.where(Event.timestamp <= until)
    return {str(key): int(count) for key, count in session.execute(statement).all()}


def event_timeline(
    session: Session, since: datetime, until: datetime, granularity: str
) -> list[dict[str, Any]]:
    """Bucketed event counts.

    Bucketing is done in Python rather than with a dialect-specific date
    function (SQLite's `strftime` vs PostgreSQL's `date_trunc`), which keeps
    this query portable — the row count over a reporting window is small.
    """
    rows = session.execute(
        select(Event.timestamp, Event.event_type, Event.severity)
        .where(Event.timestamp >= since, Event.timestamp <= until)
        .order_by(Event.timestamp)
    ).all()

    if granularity == "hour":
        fmt, step = "%Y-%m-%dT%H:00", timedelta(hours=1)
    elif granularity == "day":
        fmt, step = "%Y-%m-%d", timedelta(days=1)
    else:
        fmt, step = "%Y-%m-%dT%H:%M", timedelta(minutes=1)

    buckets: dict[str, dict[str, Any]] = {}
    cursor = since
    while cursor <= until:
        buckets[cursor.strftime(fmt)] = {
            "bucket": cursor.strftime(fmt), "total": 0, "by_type": {}, "by_severity": {}
        }
        cursor += step

    for timestamp, event_type, severity in rows:
        key = timestamp.strftime(fmt)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets.setdefault(
                key, {"bucket": key, "total": 0, "by_type": {}, "by_severity": {}}
            )
        bucket["total"] += 1
        bucket["by_type"][event_type] = bucket["by_type"].get(event_type, 0) + 1
        bucket["by_severity"][severity] = bucket["by_severity"].get(severity, 0) + 1

    return [buckets[key] for key in sorted(buckets)]


def camera_activity(
    session: Session, since: datetime, until: datetime
) -> list[dict[str, Any]]:
    """Per-camera event counts and people totals for the analytics page."""
    cameras = list_cameras(session)
    counts = dict(
        session.execute(
            select(Event.camera_id, func.count(Event.event_id))
            .where(Event.timestamp >= since, Event.timestamp <= until)
            .group_by(Event.camera_id)
        ).all()
    )
    people = dict(
        session.execute(
            select(PersonTrack.camera_id, func.count(PersonTrack.id))
            .where(PersonTrack.first_seen >= since)
            .group_by(PersonTrack.camera_id)
        ).all()
    )
    return [
        {
            "camera_id": camera.camera_id,
            "camera_name": camera.name,
            "event_count": int(counts.get(camera.camera_id, 0)),
            "people_detected": int(people.get(camera.camera_id, 0)),
            "status": camera.status,
        }
        for camera in cameras
    ]


def distinct_people_count(
    session: Session, since: datetime | None = None
) -> int:
    """Distinct tracked people, used for the People Detected KPI."""
    statement = select(func.count(PersonTrack.id))
    if since:
        statement = statement.where(PersonTrack.first_seen >= since)
    return int(session.execute(statement).scalar_one())


# ══════════════════════════════════════════════════════════════════════════
#  Person tracks
# ══════════════════════════════════════════════════════════════════════════
def upsert_person_track(
    session: Session,
    *,
    camera_id: str,
    track_id: int,
    first_seen: datetime,
    last_seen: datetime,
    frames_seen: int,
    max_speed: float = 0.0,
    avg_confidence: float = 0.0,
    dwell_seconds: float = 0.0,
    zones_entered: list[str] | None = None,
    ppe_summary: dict[str, Any] | None = None,
    event_count: int = 0,
    job_id: str | None = None,
) -> PersonTrack:
    """Insert or update the durable summary for one tracked person."""
    existing = session.execute(
        select(PersonTrack).where(
            PersonTrack.camera_id == camera_id,
            PersonTrack.track_id == track_id,
            PersonTrack.first_seen == first_seen,
        )
    ).scalar_one_or_none()

    if existing is None:
        existing = PersonTrack(
            camera_id=camera_id, track_id=track_id, first_seen=first_seen, job_id=job_id
        )
        session.add(existing)

    existing.last_seen = last_seen
    existing.frames_seen = frames_seen
    existing.max_speed = max_speed
    existing.avg_confidence = avg_confidence
    existing.dwell_seconds = dwell_seconds
    existing.zones_entered = zones_entered or []
    existing.ppe_summary = ppe_summary or {}
    existing.event_count = event_count
    return existing


# ══════════════════════════════════════════════════════════════════════════
#  Video jobs
# ══════════════════════════════════════════════════════════════════════════
def get_job(session: Session, job_id: str) -> VideoJob | None:
    return session.get(VideoJob, job_id)


def list_events_for_job(session: Session, job_id: str) -> list[Event]:
    """Events raised by one analysis, in source-timeline order.

    Ordered by `video_timestamp`, not by when the row was written: the result
    screen presents these as a timeline over the footage, and insertion order
    only matches that by accident.
    """
    return list(
        session.execute(
            select(Event)
            .where(Event.job_id == job_id)
            .order_by(Event.video_timestamp.asc().nulls_last(), Event.timestamp.asc())
        )
        .scalars()
        .all()
    )


def list_jobs(session: Session, limit: int = 50) -> list[VideoJob]:
    return list(
        session.execute(
            select(VideoJob).order_by(VideoJob.created_at.desc()).limit(limit)
        )
        .scalars()
        .all()
    )


# ══════════════════════════════════════════════════════════════════════════
#  System settings
# ══════════════════════════════════════════════════════════════════════════
def get_setting(session: Session, key: str, default: Any = None) -> Any:
    row = session.get(SystemSetting, key)
    return default if row is None else row.value


def set_setting(
    session: Session, key: str, value: Any, description: str = ""
) -> SystemSetting:
    row = session.get(SystemSetting, key)
    if row is None:
        row = SystemSetting(key=key, value=value, description=description)
        session.add(row)
    else:
        row.value = value
        if description:
            row.description = description
    return row


def load_severity_overrides() -> dict[str, str]:
    """Severity overrides from the settings table, if an operator set any."""
    with session_scope() as session:
        value = get_setting(session, "severity_overrides", {}) or {}
    valid = {s.value for s in Severity}
    return {
        str(k): str(v) for k, v in value.items() if str(v) in valid
    } if isinstance(value, dict) else {}
