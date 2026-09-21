"""Camera CRUD and pipeline control."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse

from backend.api.deps import Authorised, DbSession, Manager
from backend.db.models import Camera
from backend.db.repository import get_camera, list_cameras, zone_counts
from backend.logging_conf import get_logger
from backend.schemas import CameraCreate, CameraOut, CameraRuntime, CameraUpdate

logger = get_logger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])


def _to_out(camera: Camera, zones: int = 0, runtime: dict | None = None) -> CameraOut:
    """Serialise a camera, preferring live pipeline telemetry when present."""
    out = CameraOut.model_validate(camera)
    out.zone_count = zones
    if runtime:
        # The DB copy is flushed every few seconds; the pipeline is authoritative.
        out.status = runtime.get("status", out.status)
        out.fps = runtime.get("fps", out.fps)
        out.people_count = runtime.get("people_count", out.people_count)
        out.last_error = runtime.get("last_error") or out.last_error
    return out


@router.get("", response_model=list[CameraOut], summary="List cameras")
def list_all(session: DbSession, manager: Manager) -> list[CameraOut]:
    counts = zone_counts(session)
    runtimes = manager.runtimes()
    return [
        _to_out(c, counts.get(c.camera_id, 0), runtimes.get(c.camera_id))
        for c in list_cameras(session)
    ]


@router.post(
    "",
    response_model=CameraOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add a camera",
    dependencies=[Authorised],
)
def create(payload: CameraCreate, session: DbSession, manager: Manager) -> CameraOut:
    camera = Camera(**payload.model_dump())
    session.add(camera)
    session.commit()
    session.refresh(camera)
    logger.info(
        "Camera created: %s (%s)", camera.name, camera.source_type,
        extra={"camera": camera.camera_id},
    )
    if camera.enabled:
        # Start eagerly; a source that cannot connect reports `error` status
        # rather than failing the create.
        manager.start_camera(camera.camera_id)
    return _to_out(camera)


@router.get("/{camera_id}", response_model=CameraOut, summary="Get one camera")
def retrieve(camera_id: str, session: DbSession, manager: Manager) -> CameraOut:
    camera = get_camera(session, camera_id)
    if camera is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found")
    runtimes = manager.runtimes()
    return _to_out(camera, len(camera.zones), runtimes.get(camera_id))


@router.put(
    "/{camera_id}",
    response_model=CameraOut,
    summary="Update a camera",
    dependencies=[Authorised],
)
def update(
    camera_id: str, payload: CameraUpdate, session: DbSession, manager: Manager
) -> CameraOut:
    camera = get_camera(session, camera_id)
    if camera is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found")

    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        return _to_out(camera, len(camera.zones))

    # Re-validate the source as a whole: source_type and stream_url constrain
    # each other, so a partial update must be checked against the merged value.
    if {"source_type", "stream_url"} & changes.keys():
        merged = CameraCreate(
            name=changes.get("name", camera.name),
            location=changes.get("location", camera.location),
            source_type=changes.get("source_type", camera.source_type),
            stream_url=changes.get("stream_url", camera.stream_url),
        )
        changes["stream_url"] = merged.stream_url

    restart_needed = bool(
        {"source_type", "stream_url", "enabled", "analytics_enabled", "settings_json"}
        & changes.keys()
    )
    for field, value in changes.items():
        setattr(camera, field, value)
    session.commit()
    session.refresh(camera)

    if restart_needed:
        manager.stop_camera(camera_id)
        if camera.enabled:
            manager.start_camera(camera_id)
        else:
            camera.status = "offline"
            session.commit()

    return _to_out(camera, len(camera.zones), manager.runtimes().get(camera_id))


@router.delete(
    "/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
    summary="Delete a camera",
    dependencies=[Authorised],
)
def delete(camera_id: str, session: DbSession, manager: Manager) -> None:
    camera = get_camera(session, camera_id)
    if camera is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found")
    manager.stop_camera(camera_id)
    session.delete(camera)  # cascades to zones and events
    session.commit()
    logger.info("Camera deleted: %s", camera_id, extra={"camera": camera_id})


@router.post(
    "/{camera_id}/start",
    summary="Start a camera pipeline",
    dependencies=[Authorised],
)
def start(camera_id: str, manager: Manager) -> dict[str, object]:
    started, message = manager.start_camera(camera_id)
    if not started and message == "camera not found":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found")
    return {"camera_id": camera_id, "started": started, "message": message}


@router.post(
    "/{camera_id}/stop", summary="Stop a camera pipeline", dependencies=[Authorised]
)
def stop(camera_id: str, session: DbSession, manager: Manager) -> dict[str, object]:
    stopped = manager.stop_camera(camera_id)
    camera = get_camera(session, camera_id)
    if camera is not None:
        camera.status = "offline"
        camera.fps = 0.0
        camera.people_count = 0
        session.commit()
    return {"camera_id": camera_id, "stopped": stopped}


@router.get(
    "/{camera_id}/runtime",
    response_model=CameraRuntime,
    summary="Live pipeline telemetry",
)
def runtime(camera_id: str, manager: Manager) -> CameraRuntime:
    pipeline = manager.get(camera_id)
    if pipeline is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "No running pipeline for this camera"
        )
    data = pipeline.runtime()
    return CameraRuntime(
        camera_id=camera_id,
        status=data["status"],
        fps=data["fps"],
        people_count=data["people_count"],
        active_tracks=data["active_tracks"],
        inference_ms=data["inference_ms"],
        processing_ms=data["processing_ms"],
        frames_processed=data["frames_processed"],
        frames_dropped=data["frames_dropped"],
        backend=data.get("backend"),
        last_error=data.get("last_error") or None,
        uptime_seconds=data["uptime_seconds"],
    )


@router.get(
    "/{camera_id}/stream",
    summary="MJPEG preview of the camera's current frames",
    response_class=StreamingResponse,
)
def stream(
    camera_id: str,
    manager: Manager,
    fps: int = Query(8, ge=1, le=25, description="Preview frame rate"),
    width: int = Query(640, ge=160, le=1920, description="Preview width"),
) -> StreamingResponse:
    """Serve the raw camera frames as MJPEG.

    Detections are *not* burned in — the dashboard overlays them in the
    browser from the WebSocket feed, which keeps this endpoint cheap and lets
    the operator toggle layers without re-encoding.
    """
    pipeline = manager.get(camera_id)
    if pipeline is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "No running pipeline for this camera"
        )

    import time

    import cv2

    from backend.inference.preprocess import downscale_to_width

    boundary = "frame"

    def generate():
        interval = 1.0 / fps
        while True:
            started = time.monotonic()
            pipe = manager.get(camera_id)
            if pipe is None or not pipe.is_running:
                break
            frame = pipe.latest_frame()
            if frame is not None:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    downscale_to_width(frame, width),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 72],
                )
                if ok:
                    payload = encoded.tobytes()
                    yield (
                        b"--" + boundary.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
                        + payload + b"\r\n"
                    )
            elapsed = time.monotonic() - started
            if elapsed < interval:
                time.sleep(interval - elapsed)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-store, no-cache", "Pragma": "no-cache"},
    )
