"""Zone (restricted-area) management.

Polygons are stored normalised (0..1). The frontend's zone editor draws over
the video element and divides by its rendered size, so a zone drawn on a
laptop stays correct on a wall display and after a stream resolution change.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from backend.analysis.zones import validate_polygon
from backend.api.deps import Authorised, DbSession, Manager
from backend.db.models import Camera, Zone
from backend.logging_conf import get_logger
from backend.schemas import ZoneCreate, ZoneOut, ZoneUpdate

logger = get_logger(__name__)
router = APIRouter(prefix="/zones", tags=["zones"])


@router.post(
    "",
    response_model=ZoneOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a zone",
    dependencies=[Authorised],
)
def create(payload: ZoneCreate, session: DbSession, manager: Manager) -> ZoneOut:
    if session.get(Camera, payload.camera_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found")

    ok, reason = validate_polygon(payload.polygon)
    if not ok:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, reason)

    zone = Zone(**payload.model_dump())
    session.add(zone)
    session.commit()
    session.refresh(zone)
    # Push the change to the running pipeline without a restart.
    manager.notify_zones_changed(payload.camera_id)
    logger.info(
        "Zone created: %s (%s) on camera %s",
        zone.name, zone.zone_type, payload.camera_id,
        extra={"camera": payload.camera_id},
    )
    return ZoneOut.model_validate(zone)


@router.get(
    "/{camera_id}", response_model=list[ZoneOut], summary="List a camera's zones"
)
def list_for_camera(camera_id: str, session: DbSession) -> list[ZoneOut]:
    if session.get(Camera, camera_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found")
    return [ZoneOut.model_validate(z) for z in session.get(Camera, camera_id).zones]


@router.put(
    "/detail/{zone_id}",
    response_model=ZoneOut,
    summary="Update a zone",
    dependencies=[Authorised],
)
def update(
    zone_id: str, payload: ZoneUpdate, session: DbSession, manager: Manager
) -> ZoneOut:
    zone = session.get(Zone, zone_id)
    if zone is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Zone not found")

    changes = payload.model_dump(exclude_unset=True)
    if "polygon" in changes:
        ok, reason = validate_polygon(changes["polygon"])
        if not ok:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, reason)

    for field, value in changes.items():
        setattr(zone, field, value)
    session.commit()
    session.refresh(zone)
    manager.notify_zones_changed(zone.camera_id)
    return ZoneOut.model_validate(zone)


@router.delete(
    "/detail/{zone_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
    summary="Delete a zone",
    dependencies=[Authorised],
)
def delete(zone_id: str, session: DbSession, manager: Manager) -> None:
    zone = session.get(Zone, zone_id)
    if zone is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Zone not found")
    camera_id = zone.camera_id
    session.delete(zone)
    session.commit()
    manager.notify_zones_changed(camera_id)
    logger.info("Zone deleted: %s", zone_id, extra={"camera": camera_id})
