"""Event and alert endpoints.

`/api/events` is the historical record; `/api/alerts` is the same data scoped
to what needs attention now (open/acknowledged, most severe first). They share
one query builder so filters behave identically.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, status

from backend.api.deps import Authorised, DbSession, EventQuery
from backend.db.models import Event
from backend.db.repository import camera_names, get_event, query_events, zone_names
from backend.events.bus import get_bus
from backend.events.types import EventStatus
from backend.logging_conf import get_logger
from backend.schemas import EventOut, EventPage, EventStatusUpdate

logger = get_logger(__name__)
router = APIRouter(tags=["events"])


def _serialise(session, events: list[Event]) -> list[EventOut]:
    """Attach camera and zone names so the UI needs no extra round trips."""
    cameras = camera_names(session, [e.camera_id for e in events if e.camera_id])
    zones = zone_names(session, [e.zone_id for e in events if e.zone_id])
    out: list[EventOut] = []
    for event in events:
        model = EventOut.model_validate(event)
        model.camera_name = cameras.get(event.camera_id or "")
        model.zone_name = zones.get(event.zone_id or "") or (
            event.detection_metadata or {}
        ).get("zone_name")
        out.append(model)
    return out


@router.get("/events", response_model=EventPage, summary="Search events")
def list_events(filters: EventQuery, session: DbSession) -> EventPage:
    events, total = query_events(session, filters)
    return EventPage(
        items=_serialise(session, events),
        total=total,
        limit=filters.limit,
        offset=filters.offset,
    )


@router.get("/events/{event_id}", response_model=EventOut, summary="Get one event")
def retrieve_event(event_id: str, session: DbSession) -> EventOut:
    event = get_event(session, event_id)
    if event is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Event not found")
    return _serialise(session, [event])[0]


@router.patch(
    "/events/{event_id}/status",
    response_model=EventOut,
    summary="Acknowledge, resolve or dismiss an event",
    dependencies=[Authorised],
)
def update_status(
    event_id: str, payload: EventStatusUpdate, session: DbSession
) -> EventOut:
    event = get_event(session, event_id)
    if event is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Event not found")

    now = datetime.now(UTC)
    event.status = payload.status.value
    if payload.notes is not None:
        event.notes = payload.notes
    if payload.status is EventStatus.ACKNOWLEDGED and event.acknowledged_at is None:
        event.acknowledged_at = now
    if payload.status in (EventStatus.RESOLVED, EventStatus.DISMISSED):
        event.resolved_at = now
        # Resolving implies it was seen, even if Acknowledge was skipped.
        event.acknowledged_at = event.acknowledged_at or now
    session.commit()
    session.refresh(event)

    result = _serialise(session, [event])[0]
    get_bus().publish(
        "event", {**result.model_dump(mode="json"), "updated": True},
        camera_id=event.camera_id,
    )
    logger.info(
        "Event %s -> %s", event_id[:8], event.status,
        extra={"camera": event.camera_id, "event": event.event_type},
    )
    return result


@router.get(
    "/alerts",
    response_model=EventPage,
    summary="Active alerts, most severe and most recent first",
)
def list_alerts(
    filters: EventQuery,
    session: DbSession,
    include_resolved: bool = Query(
        False, description="Include RESOLVED and DISMISSED events"
    ),
) -> EventPage:
    from sqlalchemy import case, func, select

    from backend.db.repository import _apply_event_filters
    from backend.events.types import Severity

    statement = _apply_event_filters(select(Event), filters)
    count_statement = _apply_event_filters(select(func.count(Event.event_id)), filters)
    if not include_resolved and filters.status is None:
        active = [EventStatus.OPEN.value, EventStatus.ACKNOWLEDGED.value]
        statement = statement.where(Event.status.in_(active))
        count_statement = count_statement.where(Event.status.in_(active))

    # Order by severity rank then recency. Expressed as a CASE so it works on
    # SQLite and PostgreSQL alike without a lookup table.
    severity_rank = case(
        {
            Severity.CRITICAL.value: 0,
            Severity.HIGH.value: 1,
            Severity.MEDIUM.value: 2,
            Severity.LOW.value: 3,
        },
        value=Event.severity,
        else_=4,
    )
    total = int(session.execute(count_statement).scalar_one())
    events = list(
        session.execute(
            statement.order_by(severity_rank, Event.timestamp.desc())
            .limit(filters.limit)
            .offset(filters.offset)
        )
        .scalars()
        .all()
    )
    return EventPage(
        items=_serialise(session, events),
        total=total,
        limit=filters.limit,
        offset=filters.offset,
    )
