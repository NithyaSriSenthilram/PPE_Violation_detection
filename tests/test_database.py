"""Persistence layer: models, timestamps, cascades, and portability."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import inspect, select

from backend.db.base import engine, utcnow
from backend.db.models import (
    Camera,
    Event,
    PersonTrack,
    SystemSetting,
    VideoJob,
    Zone,
)
from backend.db.repository import (
    count_events,
    event_timeline,
    get_setting,
    group_events,
    list_cameras,
    load_zones_for_camera,
    set_setting,
    to_resolved_zone,
    upsert_person_track,
    zone_counts,
)
from backend.schemas import EventFilters


def make_camera(session, **kwargs) -> Camera:
    camera = Camera(
        camera_id=str(uuid.uuid4()),
        name=kwargs.pop("name", "Cam"),
        source_type=kwargs.pop("source_type", "file"),
        stream_url=kwargs.pop("stream_url", "x.mp4"),
        **kwargs,
    )
    session.add(camera)
    session.commit()
    session.refresh(camera)
    return camera


class TestSchema:
    def test_all_tables_created(self):
        tables = set(inspect(engine).get_table_names())
        assert {
            "cameras", "zones", "events", "person_tracks", "video_jobs",
            "system_settings",
        } <= tables

    def test_indexes_exist_on_hot_columns(self):
        """Event search filters on these; without indexes it degrades badly."""
        names = {i["name"] for i in inspect(engine).get_indexes("events")}
        assert any("ts" in n or "timestamp" in n for n in names)
        assert any("camera" in n for n in names)


class TestTimestamps:
    def test_timestamps_round_trip_as_utc_aware(self, session):
        """SQLite has no tz type; a naive round trip breaks every comparison."""
        camera = make_camera(session)
        assert camera.created_at.tzinfo is not None
        assert camera.created_at.utcoffset() == timedelta(0)

    def test_aware_input_is_normalised(self, session):
        moment = datetime(2026, 8, 29, 12, 30, tzinfo=UTC)
        event = Event(
            event_id=str(uuid.uuid4()), event_type="LOITERING",
            severity="MEDIUM", confidence=0.8, timestamp=moment,
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        assert event.timestamp == moment

    def test_comparisons_against_utcnow_work(self, session):
        """The bug this guards: naive/aware mixing raises at query time."""
        event = Event(
            event_id=str(uuid.uuid4()), event_type="LOITERING",
            severity="MEDIUM", confidence=0.8,
            timestamp=utcnow() - timedelta(hours=1),
        )
        session.add(event)
        session.commit()
        found = session.execute(
            select(Event).where(Event.timestamp >= utcnow() - timedelta(hours=2))
        ).scalars().all()
        assert len(found) == 1


class TestRelationships:
    def test_deleting_camera_cascades(self, session):
        camera = make_camera(session)
        session.add(
            Zone(zone_id=str(uuid.uuid4()), camera_id=camera.camera_id,
                 name="Z", polygon=[[0, 0], [1, 0], [1, 1]])
        )
        session.add(
            Event(event_id=str(uuid.uuid4()), event_type="LOITERING",
                  camera_id=camera.camera_id, severity="MEDIUM", confidence=0.8)
        )
        session.commit()

        session.delete(camera)
        session.commit()
        assert session.execute(select(Zone)).scalars().all() == []
        assert session.execute(select(Event)).scalars().all() == []

    def test_json_columns_round_trip(self, session):
        camera = make_camera(session, settings_json={"profile": "thorough", "loop": True})
        session.refresh(camera)
        assert camera.settings_json["profile"] == "thorough"
        assert camera.settings_json["loop"] is True

    def test_polygon_survives_round_trip(self, session):
        camera = make_camera(session)
        polygon = [[0.1, 0.2], [0.8, 0.25], [0.75, 0.9]]
        zone = Zone(
            zone_id=str(uuid.uuid4()), camera_id=camera.camera_id,
            name="Z", polygon=polygon,
        )
        session.add(zone)
        session.commit()
        session.refresh(zone)
        assert zone.polygon == polygon
        assert to_resolved_zone(zone).polygon_norm == polygon


class TestRepository:
    def test_zone_counts_and_loading(self, session):
        camera = make_camera(session)
        for i in range(3):
            session.add(
                Zone(zone_id=str(uuid.uuid4()), camera_id=camera.camera_id,
                     name=f"Z{i}", polygon=[[0, 0], [1, 0], [1, 1]],
                     enabled=i != 2)
            )
        session.commit()
        assert zone_counts(session)[camera.camera_id] == 3
        # Only enabled zones reach the pipeline.
        assert len(load_zones_for_camera(camera.camera_id)) == 2

    def test_enabled_only_camera_listing(self, session):
        make_camera(session, name="On", enabled=True)
        make_camera(session, name="Off", enabled=False)
        assert len(list_cameras(session)) == 2
        assert len(list_cameras(session, enabled_only=True)) == 1

    def test_count_and_group(self, session):
        for i in range(6):
            session.add(
                Event(
                    event_id=str(uuid.uuid4()),
                    event_type="MISSING_HELMET" if i % 2 else "LOITERING",
                    severity="HIGH" if i % 2 else "LOW",
                    confidence=0.8,
                    timestamp=utcnow() - timedelta(minutes=i),
                )
            )
        session.commit()
        assert count_events(session) == 6
        assert count_events(session, types=("LOITERING",)) == 3
        assert group_events(session, Event.severity) == {"HIGH": 3, "LOW": 3}

    def test_timeline_buckets_are_contiguous(self, session):
        """Empty buckets must still appear, or the chart shows a false gap."""
        session.add(
            Event(event_id=str(uuid.uuid4()), event_type="LOITERING",
                  severity="LOW", confidence=0.8, timestamp=utcnow())
        )
        session.commit()
        start = utcnow() - timedelta(hours=6)
        buckets = event_timeline(session, start, utcnow(), "hour")
        assert len(buckets) >= 6
        assert sum(b["total"] for b in buckets) == 1
        assert [b["bucket"] for b in buckets] == sorted(b["bucket"] for b in buckets)

    def test_person_track_upsert_is_idempotent(self, session):
        first_seen = utcnow()
        for frames in (10, 40):
            upsert_person_track(
                session, camera_id="cam", track_id=7,
                first_seen=first_seen, last_seen=utcnow(),
                frames_seen=frames, max_speed=1.5,
                zones_entered=["zA"], event_count=1,
            )
            session.commit()
        rows = session.execute(select(PersonTrack)).scalars().all()
        assert len(rows) == 1, "upsert created a duplicate"
        assert rows[0].frames_seen == 40

    def test_settings_round_trip(self, session):
        set_setting(session, "severity_overrides", {"MISSING_VEST": "HIGH"})
        session.commit()
        assert get_setting(session, "severity_overrides")["MISSING_VEST"] == "HIGH"
        assert get_setting(session, "absent", "fallback") == "fallback"

    def test_filters_translate_to_sql(self, session):
        from backend.db.repository import query_events

        camera = make_camera(session)
        session.add(
            Event(event_id=str(uuid.uuid4()), event_type="MISSING_VEST",
                  camera_id=camera.camera_id, person_id=3, severity="MEDIUM",
                  confidence=0.7, description="vest missing", timestamp=utcnow())
        )
        session.commit()
        rows, total = query_events(session, EventFilters(camera_id=camera.camera_id))
        assert total == 1
        rows, total = query_events(session, EventFilters(search="vest"))
        assert total == 1
        rows, total = query_events(session, EventFilters(person_id=99))
        assert total == 0


class TestVideoJobModel:
    def test_defaults(self, session):
        job = VideoJob(job_id=str(uuid.uuid4()), filename="a.mp4", stored_path="/tmp/a.mp4")
        session.add(job)
        session.commit()
        session.refresh(job)
        assert job.status == "QUEUED"
        assert job.progress == 0.0
        assert job.summary == {}


class TestInterruptedJobs:
    def test_queued_and_running_jobs_fail_on_restart(self, session):
        """A restart kills the in-process pool; stale jobs must not stay RUNNING."""
        from backend.jobs.video_jobs import reconcile_interrupted_jobs

        ids = {}
        for status in ("QUEUED", "RUNNING", "COMPLETED", "FAILED"):
            job = VideoJob(
                job_id=str(uuid.uuid4()), filename="a.mp4", stored_path="/tmp/a.mp4",
                status=status,
            )
            session.add(job)
            ids[status] = job.job_id
        session.commit()

        assert reconcile_interrupted_jobs() == 2

        session.expire_all()
        assert session.get(VideoJob, ids["QUEUED"]).status == "FAILED"
        running = session.get(VideoJob, ids["RUNNING"])
        assert running.status == "FAILED"
        assert "restart" in running.message
        assert running.finished_at is not None
        assert session.get(VideoJob, ids["COMPLETED"]).status == "COMPLETED"
        assert reconcile_interrupted_jobs() == 0


class TestPortability:
    def test_no_sqlite_specific_column_types(self):
        """Everything must work on PostgreSQL without a rewrite."""
        from backend.db.base import Base

        for table in Base.metadata.sorted_tables:
            for column in table.columns:
                name = type(column.type).__name__
                assert "SQLITE" not in name.upper(), f"{table.name}.{column.name} is {name}"

    def test_settings_key_is_the_primary_key(self, session):
        """Upserting a setting must not create duplicates."""
        set_setting(session, "k", 1)
        session.commit()
        set_setting(session, "k", 2)
        session.commit()
        assert len(session.execute(select(SystemSetting)).scalars().all()) == 1


class TestStoredRenderPath:
    def test_inside_project_is_relative(self):
        from backend.config import PROJECT_ROOT
        from backend.jobs.video_jobs import _stored_path

        assert _stored_path(PROJECT_ROOT / "processed" / "x.mp4") == "processed/x.mp4"

    def test_outside_project_is_absolute(self, tmp_path):
        """PROCESSED_DIR on a mounted disk is not under the project root."""
        from backend.jobs.video_jobs import _stored_path

        target = tmp_path / "processed" / "x.mp4"
        stored = _stored_path(target)
        assert stored.startswith("/") and stored.endswith("processed/x.mp4")
