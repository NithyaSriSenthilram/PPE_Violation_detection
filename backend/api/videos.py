"""Video upload and analysis.

Upload safety (§28) is enforced here, in this order:

1. Extension must be in `ALLOWED_VIDEO_EXTENSIONS`.
2. The stored filename is generated, never taken from the client — the
   original name is kept only as a display label. This is what defeats path
   traversal and any attempt at writing outside `UPLOAD_DIR`.
3. The body is streamed to disk in chunks with a running size check, so an
   oversized upload is aborted mid-stream instead of being buffered into
   memory first.
4. The saved file is probed with the decoder. Anything that does not yield a
   frame is deleted and rejected — extension checks alone prove nothing.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from backend.api.deps import Authorised, DbSession
from backend.config import PROJECT_ROOT, settings
from backend.db.models import VideoJob
from backend.db.repository import get_job, list_events_for_job, list_jobs
from backend.events.types import label_for
from backend.jobs.video_jobs import JobStatus, get_job_runner
from backend.logging_conf import get_logger
from backend.schemas import (
    DetectionProfile,
    TimelineEntry,
    VideoJobOut,
    VideoJobResult,
    VideoUploadResponse,
)
from backend.video.source import probe_video_file

logger = get_logger(__name__)
router = APIRouter(prefix="/videos", tags=["video analysis"])

#: Streaming chunk size for the upload write loop.
CHUNK_SIZE = 1024 * 1024


@router.post(
    "/upload",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a video and queue it for analysis",
    dependencies=[Authorised],
)
async def upload(
    session: DbSession,
    file: UploadFile = File(..., description="Surveillance video file"),
    profile: DetectionProfile = Form(
        "standard", description="Detection profile to run"
    ),
    start: bool = Form(True, description="Begin analysis immediately"),
    camera_id: str = Form(
        "",
        description=(
            "Optional camera whose zones apply to this footage. Without it no "
            "zone geometry exists, so zone-dependent rules cannot fire."
        ),
    ),
) -> VideoUploadResponse:
    original = Path(file.filename or "upload").name
    suffix = Path(original).suffix.lower()

    if suffix not in settings.allowed_extension_set:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{suffix or '(none)'}'. Allowed: "
            f"{', '.join(sorted(settings.allowed_extension_set))}",
        )

    settings.ensure_directories()
    job_id = str(uuid.uuid4())
    # Generated name — the client never influences the path we write to.
    stored = settings.upload_root / f"{job_id}{suffix}"

    written = 0
    try:
        with stored.open("wb") as handle:
            while chunk := await file.read(CHUNK_SIZE):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    handle.close()
                    stored.unlink(missing_ok=True)
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"File exceeds the {settings.max_upload_mb} MB limit",
                    )
                handle.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        stored.unlink(missing_ok=True)
        logger.error("Upload write failed: %s", exc)
        raise HTTPException(
            status.HTTP_507_INSUFFICIENT_STORAGE, "Could not store the uploaded file"
        ) from exc
    finally:
        await file.close()

    if written == 0:
        stored.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty")

    # Content validation — an extension is not evidence of a decodable video.
    probe = probe_video_file(stored)
    if not probe.get("ok"):
        stored.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"This file is not a readable video: {probe.get('error')}",
        )

    job = VideoJob(
        job_id=job_id,
        filename=original,
        stored_path=str(stored),
        profile=profile,
        status=JobStatus.QUEUED if start else "PENDING",
        total_frames=probe.get("total_frames", 0),
        fps=probe.get("fps", 0.0),
        resolution=f"{probe.get('width')}x{probe.get('height')}",
        duration_seconds=probe.get("duration_seconds", 0.0),
        zone_camera_id=camera_id.strip() or None,
        message="Queued for analysis" if start else "Awaiting start",
    )
    session.add(job)
    session.commit()

    logger.info(
        "Video uploaded: %s (%.1f MB, %s, %.1fs) profile=%s",
        original, written / 1e6, job.resolution, job.duration_seconds, profile,
        extra={"job": job_id[:8]},
    )

    if start:
        get_job_runner().submit(job_id, profile)

    return VideoUploadResponse(
        job_id=job_id,
        filename=original,
        size_bytes=written,
        profile=profile,
        status=job.status,
        message=(
            f"Analysis queued — {probe.get('total_frames', 0)} frames "
            f"at {probe.get('fps', 0)} fps"
            if start
            else "Uploaded; call /start to begin analysis"
        ),
    )


@router.get("/jobs", response_model=list[VideoJobOut], summary="List analysis jobs")
def list_all(session: DbSession, limit: int = Query(50, ge=1, le=200)) -> list[VideoJobOut]:
    return [VideoJobOut.model_validate(j) for j in list_jobs(session, limit)]


@router.get(
    "/jobs/{job_id}", response_model=VideoJobOut, summary="Analysis job progress"
)
def retrieve(job_id: str, session: DbSession) -> VideoJobOut:
    job = get_job(session, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return VideoJobOut.model_validate(job)


@router.post(
    "/jobs/{job_id}/start",
    response_model=VideoJobOut,
    summary="Start (or re-run) analysis",
    dependencies=[Authorised],
)
def start_job(job_id: str, session: DbSession) -> VideoJobOut:
    job = get_job(session, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status.HTTP_409_CONFLICT, "This job is already running")
    if not Path(job.stored_path).is_file():
        raise HTTPException(
            status.HTTP_410_GONE, "The uploaded file is no longer available"
        )

    job.status = JobStatus.QUEUED
    job.progress = 0.0
    job.message = "Queued for analysis"
    job.finished_at = None
    session.commit()
    session.refresh(job)
    get_job_runner().submit(job_id, job.profile)
    return VideoJobOut.model_validate(job)


@router.post(
    "/jobs/{job_id}/cancel", summary="Cancel a running job", dependencies=[Authorised]
)
def cancel_job(job_id: str, session: DbSession) -> dict[str, object]:
    job = get_job(session, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    cancelled = get_job_runner().cancel(job_id)
    if not cancelled:
        raise HTTPException(status.HTTP_409_CONFLICT, "This job is not running")
    return {"job_id": job_id, "cancelled": True}


@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
    summary="Delete a job and its uploaded file",
    dependencies=[Authorised],
)
def delete_job(job_id: str, session: DbSession) -> None:
    job = get_job(session, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    get_job_runner().cancel(job_id)

    # Only ever unlink inside UPLOAD_DIR, whatever the stored value claims.
    try:
        path = Path(job.stored_path).resolve()
        if path.is_relative_to(settings.upload_root.resolve()):
            path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete upload for job %s: %s", job_id[:8], exc)

    session.delete(job)
    session.commit()


def _resolve_render(job: VideoJob) -> Path | None:
    """Absolute path of a job's annotated render, or None.

    Containment is the point: the stored value is confined to PROCESSED_DIR
    before it is ever opened, so a crafted or stale database row cannot be used
    to read an arbitrary file off the host. Clients only ever see job ids.
    """
    if not job.annotated_path:
        return None
    root = settings.processed_root.resolve()
    candidate = Path(job.annotated_path)
    absolute = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    try:
        resolved = absolute.resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(root):
        logger.warning(
            "Rejected annotated path outside PROCESSED_DIR for job %s", job.job_id[:8]
        )
        return None
    return resolved if resolved.is_file() else None


def _resolve_upload(job: VideoJob) -> Path | None:
    """Absolute path of the original upload, confined to UPLOAD_DIR."""
    root = settings.upload_root.resolve()
    try:
        resolved = Path(job.stored_path).resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(root):
        return None
    return resolved if resolved.is_file() else None


def _download_stem(job: VideoJob) -> str:
    from backend.events.evidence import sanitise

    return sanitise(Path(job.filename).stem, 60) or "video"


@router.get(
    "/jobs/{job_id}/result",
    response_model=VideoJobResult,
    summary="Annotated video URL, playback metadata and the event timeline",
)
def job_result(job_id: str, session: DbSession) -> VideoJobResult:
    """Everything the result screen needs, in one call.

    Returns API URLs, never filesystem paths — the browser must not be handed
    a location on the host, and the paths on disk are free to change.
    """
    job = get_job(session, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")

    render = _resolve_render(job)
    output = dict(job.output or {})
    summary = dict(job.summary or {})

    events = list_events_for_job(session, job_id)
    timeline = [
        TimelineEntry(
            event_id=event.event_id,
            event_type=event.event_type,
            label=label_for(event.event_type),
            severity=event.severity,
            person_id=event.person_id,
            confidence=round(event.confidence, 3),
            # The seek target. Events carry the source timeline position, so
            # clicking one lands on the frame that produced it.
            video_timestamp=event.video_timestamp,
            snapshot_available=bool(event.snapshot_path),
        )
        for event in events
        if event.video_timestamp is not None
    ]

    ready = job.status == JobStatus.COMPLETED and render is not None
    if job.status == JobStatus.COMPLETED and render is None:
        # Completed but the file is gone — say so rather than handing the
        # player a URL that 404s.
        output.setdefault(
            "warnings", []
        ).append("The annotated video is no longer on disk")

    return VideoJobResult(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status,
        progress=job.progress,
        message=job.message,
        annotated_ready=ready,
        video_url=f"/api/videos/jobs/{job.job_id}/video" if ready else None,
        download_url=f"/api/videos/jobs/{job.job_id}/download" if ready else None,
        original_url=(
            f"/api/videos/jobs/{job.job_id}/original"
            if _resolve_upload(job) is not None
            else None
        ),
        duration_seconds=(
            float(output.get("duration_seconds") or job.duration_seconds or 0.0)
        ),
        fps=float(output.get("fps") or job.fps or 0.0),
        frames=int(output.get("frames_written") or job.processed_frames or 0),
        resolution=str(output.get("resolution") or job.resolution or ""),
        size_bytes=int(output.get("size_bytes") or 0),
        codec=str(output.get("codec") or ""),
        has_audio=bool(output.get("has_audio")),
        browser_compatible=bool(output.get("browser_compatible")),
        warnings=list(output.get("warnings") or []),
        summary=summary,
        events_created=job.events_created,
        people_detected=job.people_detected,
        timeline=timeline,
    )


@router.get(
    "/jobs/{job_id}/video",
    summary="Stream the annotated video (supports range requests for seeking)",
)
def annotated_video(job_id: str, session: DbSession) -> FileResponse:
    """Serve the render for playback.

    `FileResponse` answers Range requests with 206 Partial Content, which is
    what lets the browser start playing before the download finishes and seek
    without refetching. The disposition is `inline` so the player renders it
    rather than the browser offering to save it.
    """
    job = get_job(session, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    path = _resolve_render(job)
    if path is None:
        detail = (
            "This analysis is still running — the annotated video is not ready"
            if job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            else "No annotated video exists for this job"
        )
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail)
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename="{_download_stem(job)}_annotated.mp4"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.get("/jobs/{job_id}/download", summary="Download the annotated video")
def download_annotated(job_id: str, session: DbSession) -> FileResponse:
    job = get_job(session, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    path = _resolve_render(job)
    if path is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "No annotated video exists for this job"
        )
    return FileResponse(
        path, media_type="video/mp4",
        filename=f"{_download_stem(job)}_annotated.mp4",
    )


@router.get("/jobs/{job_id}/original", summary="Download the original upload")
def download_original(job_id: str, session: DbSession) -> FileResponse:
    job = get_job(session, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    path = _resolve_upload(job)
    if path is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "The original upload is no longer on disk"
        )
    return FileResponse(
        path, media_type="video/mp4",
        filename=f"{_download_stem(job)}{path.suffix}",
    )


@router.get(
    "/profiles", summary="Available detection profiles and what each one runs"
)
def profiles() -> dict[str, object]:
    from backend.analysis import PROFILE_ANALYSERS

    descriptions = {
        "fast": "Zone intrusion and crowd only — quickest pass",
        "standard": "PPE, intrusion, loitering, movement, crowd",
        "thorough": "Everything, including fall detection",
        "ppe_only": "PPE compliance only",
        "security_only": "Security behaviours without PPE checks",
    }
    return {
        "profiles": [
            {
                "name": name,
                "analysers": list(analysers),
                "description": descriptions.get(name, ""),
            }
            for name, analysers in PROFILE_ANALYSERS.items()
        ],
        "default": "standard",
    }
