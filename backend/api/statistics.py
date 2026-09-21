"""Statistics and analytics.

`/api/statistics` serves both the dashboard KPI cards and the analytics page.
The `range` parameter selects a preset window and picks a sensible bucket
size, so the timeline never returns thousands of near-empty buckets:

    today   -> hourly
    7d      -> daily
    30d     -> daily
    custom  -> hourly under 3 days, daily beyond
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status

from backend.api.deps import DbSession, Manager
from backend.db.models import Event
from backend.db.repository import (
    PPE_TYPES,
    SUSPICIOUS_TYPES,
    camera_activity,
    count_events,
    distinct_people_count,
    event_timeline,
    group_events,
    list_cameras,
)
from backend.events.types import EventStatus, EventType, Severity
from backend.schemas import (
    CameraActivity,
    KpiSummary,
    StatisticsOut,
    TimeBucket,
)

router = APIRouter(tags=["analytics"])

RangePreset = Literal["today", "24h", "7d", "30d", "custom"]


def _resolve_range(
    preset: RangePreset, start: str | None, end: str | None
) -> tuple[datetime, datetime, str]:
    """Turn a preset (or explicit bounds) into ``(start, end, granularity)``."""
    now = datetime.now(UTC)

    if preset == "custom":
        if not start or not end:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "range=custom requires both `start` and `end` (ISO-8601)",
            )
        try:
            begin = datetime.fromisoformat(start)
            finish = datetime.fromisoformat(end)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Invalid ISO-8601 timestamp: {exc}",
            ) from exc
        if begin.tzinfo is None:
            begin = begin.replace(tzinfo=UTC)
        if finish.tzinfo is None:
            finish = finish.replace(tzinfo=UTC)
        if finish <= begin:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "`end` must be after `start`"
            )
        granularity = "hour" if (finish - begin) <= timedelta(days=3) else "day"
        return begin, finish, granularity

    if preset == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0), now, "hour"
    if preset == "24h":
        return now - timedelta(hours=24), now, "hour"
    if preset == "7d":
        return now - timedelta(days=7), now, "day"
    return now - timedelta(days=30), now, "day"


@router.get(
    "/statistics",
    response_model=StatisticsOut,
    summary="KPIs, breakdowns, timeline and per-camera activity",
)
def statistics(
    session: DbSession,
    manager: Manager,
    range: RangePreset = Query("7d", description="Preset reporting window"),
    start: str | None = Query(None, description="ISO-8601 start (range=custom)"),
    end: str | None = Query(None, description="ISO-8601 end (range=custom)"),
) -> StatisticsOut:
    begin, finish, granularity = _resolve_range(range, start, end)
    today = datetime.now(UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    cameras = list_cameras(session)
    runtimes = manager.runtimes()
    online = sum(
        1
        for c in cameras
        if runtimes.get(c.camera_id, {}).get("status", c.status) == "online"
    )

    kpis = KpiSummary(
        people_detected=distinct_people_count(session, since=begin),
        active_cameras=online,
        total_cameras=len(cameras),
        ppe_violations=count_events(session, since=begin, until=finish, types=PPE_TYPES),
        intrusions=count_events(
            session, since=begin, until=finish, types=(EventType.RESTRICTED_AREA.value,)
        ),
        suspicious_events=count_events(
            session, since=begin, until=finish, types=SUSPICIOUS_TYPES
        ),
        open_alerts=count_events(
            session,
            statuses=(EventStatus.OPEN.value, EventStatus.ACKNOWLEDGED.value),
        ),
        critical_alerts=count_events(
            session,
            statuses=(EventStatus.OPEN.value,),
            severities=(Severity.CRITICAL.value, Severity.HIGH.value),
        ),
        events_today=count_events(session, since=today),
    )

    return StatisticsOut(
        kpis=kpis,
        by_type=group_events(session, Event.event_type, since=begin, until=finish),
        by_severity=group_events(session, Event.severity, since=begin, until=finish),
        by_status=group_events(session, Event.status, since=begin, until=finish),
        timeline=[
            TimeBucket(**bucket)
            for bucket in event_timeline(session, begin, finish, granularity)
        ],
        camera_activity=[
            CameraActivity(**row) for row in camera_activity(session, begin, finish)
        ],
        range_start=begin,
        range_end=finish,
        granularity=granularity,
    )
