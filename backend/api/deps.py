"""Shared FastAPI dependencies."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.config import settings
from backend.db.base import get_session
from backend.schemas import EventFilters
from backend.video.manager import CameraManager, get_manager

#: Process start time, for uptime reporting.
STARTED_AT = time.monotonic()


def uptime_seconds() -> float:
    return time.monotonic() - STARTED_AT


async def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Shared-secret auth.

    Disabled when `API_KEY` is empty, which is the sane default for a LAN dev
    box. Set it for anything reachable beyond localhost. Comparison is
    constant-time so a wrong key cannot be discovered by timing.
    """
    if not settings.api_key:
        return
    import hmac

    if not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key header is required",
        )


DbSession = Annotated[Session, Depends(get_session)]
Manager = Annotated[CameraManager, Depends(get_manager)]
Authorised = Depends(require_api_key)


def event_filters(
    camera_id: str | None = Query(None, description="Filter by camera"),
    event_type: str | None = Query(None, description="Filter by event type"),
    severity: str | None = Query(None, description="LOW | MEDIUM | HIGH | CRITICAL"),
    status_: str | None = Query(
        None, alias="status", description="OPEN | ACKNOWLEDGED | RESOLVED | DISMISSED"
    ),
    person_id: int | None = Query(None, description="Filter by tracked person ID"),
    job_id: str | None = Query(None, description="Filter by video-analysis job"),
    start: str | None = Query(None, description="ISO-8601 lower bound (inclusive)"),
    end: str | None = Query(None, description="ISO-8601 upper bound (inclusive)"),
    search: str | None = Query(None, description="Substring match on description"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> EventFilters:
    """Parse and validate the shared event query parameters.

    Values arrive as strings so an invalid enum or timestamp becomes a 422 with
    a useful message rather than a 500 from deeper in the stack.
    """
    try:
        return EventFilters(
            camera_id=camera_id,
            event_type=event_type,
            severity=severity,
            status=status_,
            person_id=person_id,
            job_id=job_id,
            start=start,
            end=end,
            search=search,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid filter: {exc}",
        ) from exc


EventQuery = Annotated[EventFilters, Depends(event_filters)]
