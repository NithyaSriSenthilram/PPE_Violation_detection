"""Evidence retrieval.

Files are served by **event id**, never by a client-supplied path. The path
comes from the database and is then re-validated with
:func:`resolve_evidence_path`, which refuses anything resolving outside
`EVIDENCE_DIR`. Two independent barriers: the client cannot name a file, and
a tampered database value still cannot escape the evidence directory.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse

from backend.api.deps import DbSession
from backend.db.repository import camera_names, get_event
from backend.events.evidence import resolve_evidence_path
from backend.logging_conf import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/evidence", tags=["evidence"])

EvidenceKind = str


@router.get("/{event_id}", summary="Evidence manifest for an event")
def manifest(event_id: str, session: DbSession) -> dict[str, object]:
    """What evidence exists for this event, and whether it is ready yet."""
    event = get_event(session, event_id)
    if event is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Event not found")

    snapshot = resolve_evidence_path(event.snapshot_path or "")
    clip = resolve_evidence_path(event.clip_path or "")

    return {
        "event_id": event_id,
        "event_type": event.event_type,
        "timestamp": event.timestamp.isoformat(),
        "snapshot": {
            "available": snapshot is not None,
            "url": f"/api/evidence/{event_id}/snapshot" if snapshot else None,
            "size_bytes": snapshot.stat().st_size if snapshot else 0,
        },
        "clip": {
            # A recorded path with no file yet means the post-roll is still
            # being written — that is "pending", not "missing".
            "available": clip is not None,
            "pending": bool(event.clip_path) and clip is None,
            "url": f"/api/evidence/{event_id}/clip" if clip else None,
            "size_bytes": clip.stat().st_size if clip else 0,
        },
        "export_url": f"/api/evidence/{event_id}/export",
    }


@router.get("/{event_id}/snapshot", summary="Download the event snapshot")
def snapshot(event_id: str, session: DbSession) -> FileResponse:
    event = get_event(session, event_id)
    if event is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Event not found")
    path = resolve_evidence_path(event.snapshot_path or "")
    if path is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "No snapshot is available for this event"
        )
    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=path.name,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/{event_id}/clip", summary="Stream the event video clip")
def clip(event_id: str, session: DbSession) -> FileResponse:
    event = get_event(session, event_id)
    if event is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Event not found")
    path = resolve_evidence_path(event.clip_path or "")
    if path is None:
        detail = (
            "The clip for this event is still being written — retry shortly"
            if event.clip_path
            else "No clip is available for this event"
        )
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail)
    # FileResponse handles Range requests, which is what lets the browser seek.
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.get("/{event_id}/export", summary="Download a zip of all event evidence")
def export(
    event_id: str,
    session: DbSession,
    include_clip: bool = Query(True, description="Include the video clip"),
) -> StreamingResponse:
    """Bundle the snapshot, clip and a metadata JSON into one archive."""
    event = get_event(session, event_id)
    if event is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Event not found")

    names = camera_names(session, [event.camera_id] if event.camera_id else [])
    metadata = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "severity": event.severity,
        "status": event.status,
        "confidence": event.confidence,
        "description": event.description,
        "camera_id": event.camera_id,
        "camera_name": names.get(event.camera_id or ""),
        "person_id": event.person_id,
        "zone_id": event.zone_id,
        "bbox": event.bbox,
        "detection_metadata": event.detection_metadata,
        "timestamp": event.timestamp.isoformat(),
        "acknowledged_at": (
            event.acknowledged_at.isoformat() if event.acknowledged_at else None
        ),
        "resolved_at": event.resolved_at.isoformat() if event.resolved_at else None,
        "notes": event.notes,
        "exported_at": datetime.now(UTC).isoformat(),
        "exported_by": "SentinelVision AI",
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("event.json", json.dumps(metadata, indent=2, default=str))
        snapshot_path = resolve_evidence_path(event.snapshot_path or "")
        if snapshot_path:
            archive.write(snapshot_path, f"snapshot{snapshot_path.suffix}")
        if include_clip:
            clip_path = resolve_evidence_path(event.clip_path or "")
            if clip_path:
                archive.write(clip_path, f"clip{clip_path.suffix}")
    buffer.seek(0)

    stem = f"evidence_{event.event_type}_{event_id[:8]}"
    logger.info(
        "Evidence exported for event %s", event_id[:8],
        extra={"camera": event.camera_id, "event": event.event_type},
    )
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{stem}.zip"'},
    )
