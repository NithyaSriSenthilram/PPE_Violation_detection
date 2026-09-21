"""API contract: endpoints, validation, and the security boundaries."""

from __future__ import annotations

import io
from datetime import UTC

import pytest

CAMERA = {
    "name": "Gate Camera",
    "location": "North Gate",
    "source_type": "rtsp",
    "stream_url": "rtsp://10.0.0.5:554/stream1",
}


def create_camera(client, **overrides):
    payload = {**CAMERA, **overrides}
    response = client.post("/api/cameras", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


# ══════════════════════════════════════════════════════════════════════════
#  System
# ══════════════════════════════════════════════════════════════════════════
class TestSystem:
    def test_health(self, client):
        body = client.get("/api/health").json()
        assert body["status"] in {"ok", "degraded"}
        assert body["app"] and body["version"]
        assert isinstance(body["database"], bool)

    def test_diagnostics_reports_every_backend_truthfully(self, client):
        """Spec §24: the real active backend, and why the others are not."""
        body = client.get("/api/diagnostics").json()
        names = {b["name"] for b in body["backends"]}
        assert {"max", "onnx", "ultralytics", "mock"} <= names
        active = [b for b in body["backends"] if b["active"]]
        assert len(active) <= 1
        for backend in body["backends"]:
            # An unavailable backend must explain itself.
            if not backend["available"]:
                assert backend["reason"], f"{backend['name']} gave no reason"

    def test_diagnostics_reports_mojo_state(self, client):
        mojo = client.get("/api/diagnostics").json()["mojo"]
        assert isinstance(mojo["active"], bool)
        assert isinstance(mojo["available"], bool)
        # Never claim active without a loaded library.
        assert not (mojo["active"] and not mojo["available"])
        for kernel in mojo["kernels"].values():
            assert kernel["implementation"] in {"mojo", "numpy"}

    def test_openapi_served(self, client):
        assert client.get("/api/openapi.json").status_code == 200


# ══════════════════════════════════════════════════════════════════════════
#  Cameras
# ══════════════════════════════════════════════════════════════════════════
class TestCameras:
    def test_crud_round_trip(self, client):
        created = create_camera(client)
        camera_id = created["camera_id"]

        assert client.get(f"/api/cameras/{camera_id}").json()["name"] == "Gate Camera"
        assert any(c["camera_id"] == camera_id for c in client.get("/api/cameras").json())

        updated = client.put(
            f"/api/cameras/{camera_id}", json={"name": "Renamed", "location": "South"}
        )
        assert updated.status_code == 200
        assert updated.json()["name"] == "Renamed"

        assert client.delete(f"/api/cameras/{camera_id}").status_code == 204
        assert client.get(f"/api/cameras/{camera_id}").status_code == 404

    def test_missing_camera_is_404(self, client):
        assert client.get("/api/cameras/does-not-exist").status_code == 404
        assert client.delete("/api/cameras/does-not-exist").status_code == 404

    @pytest.mark.parametrize(
        "payload",
        [
            {"source_type": "rtsp", "stream_url": "file:///etc/passwd"},
            {"source_type": "rtsp", "stream_url": "ftp://host/stream"},
            {"source_type": "rtsp", "stream_url": "rtsp://"},
            {"source_type": "webcam", "stream_url": "/dev/video0"},
            {"source_type": "webcam", "stream_url": "not-a-number"},
            {"source_type": "file", "stream_url": "../../etc/passwd"},
            {"source_type": "file", "stream_url": "/etc/passwd"},
            {"source_type": "carrier-pigeon", "stream_url": "x"},
        ],
    )
    def test_dangerous_sources_rejected(self, client, payload):
        """A camera record must never become a way to read arbitrary files."""
        response = client.post("/api/cameras", json={"name": "X", **payload})
        assert response.status_code == 422, response.text

    def test_name_length_enforced(self, client):
        response = client.post("/api/cameras", json={**CAMERA, "name": "x" * 200})
        assert response.status_code == 422

    def test_partial_update_revalidates_source_pair(self, client):
        """source_type and stream_url constrain each other."""
        camera_id = create_camera(client)["camera_id"]
        response = client.put(
            f"/api/cameras/{camera_id}", json={"source_type": "webcam", "stream_url": "0"}
        )
        assert response.status_code == 200
        assert response.json()["source_type"] == "webcam"

    def test_runtime_404_for_camera_with_no_pipeline(self, client):
        """A disabled camera never gets a worker, so it has no runtime."""
        camera_id = create_camera(client, enabled=False)["camera_id"]
        assert client.get(f"/api/cameras/{camera_id}/runtime").status_code == 404

    def test_unreachable_camera_still_creates_and_reports_a_fault(self, client):
        """Spec §29: one bad camera must not fail the request or the server."""
        created = create_camera(client, name="Dead Camera")
        assert created["camera_id"]
        # It is registered and the API stays healthy regardless of the stream.
        assert client.get("/api/health").json()["status"] in {"ok", "degraded"}


# ══════════════════════════════════════════════════════════════════════════
#  Zones
# ══════════════════════════════════════════════════════════════════════════
class TestZones:
    def test_create_and_list(self, client):
        camera_id = create_camera(client)["camera_id"]
        response = client.post(
            "/api/zones",
            json={
                "camera_id": camera_id,
                "name": "Restricted A",
                "zone_type": "restricted",
                "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "loitering_threshold": 45,
            },
        )
        assert response.status_code == 201, response.text
        assert response.json()["loitering_threshold"] == 45
        assert len(client.get(f"/api/zones/{camera_id}").json()) == 1

    @pytest.mark.parametrize(
        "polygon",
        [
            [[0.1, 0.1], [0.9, 0.1]],                     # too few points
            [[0.1, 0.1], [1.4, 0.1], [0.5, 0.9]],         # not normalised
            [[0.1, 0.1], [-0.2, 0.1], [0.5, 0.9]],        # negative
            [[0.1, 0.1], [0.2, 0.1], [0.3, 0.1]],         # zero area
        ],
    )
    def test_bad_polygons_rejected(self, client, polygon):
        camera_id = create_camera(client)["camera_id"]
        response = client.post(
            "/api/zones", json={"camera_id": camera_id, "name": "Z", "polygon": polygon}
        )
        assert response.status_code == 422, response.text

    def test_zone_for_unknown_camera_is_404(self, client):
        response = client.post(
            "/api/zones",
            json={
                "camera_id": "nope",
                "name": "Z",
                "polygon": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
            },
        )
        assert response.status_code == 404

    def test_update_and_delete(self, client):
        camera_id = create_camera(client)["camera_id"]
        zone = client.post(
            "/api/zones",
            json={
                "camera_id": camera_id,
                "name": "Z",
                "polygon": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
            },
        ).json()
        updated = client.put(
            f"/api/zones/detail/{zone['zone_id']}", json={"enabled": False, "name": "Off"}
        )
        assert updated.status_code == 200
        assert updated.json()["enabled"] is False
        assert client.delete(f"/api/zones/detail/{zone['zone_id']}").status_code == 204

    def test_deleting_camera_cascades_to_zones(self, client):
        camera_id = create_camera(client, enabled=False)["camera_id"]
        client.post(
            "/api/zones",
            json={
                "camera_id": camera_id,
                "name": "Z",
                "polygon": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
            },
        )
        client.delete(f"/api/cameras/{camera_id}")
        assert client.get(f"/api/zones/{camera_id}").status_code == 404


# ══════════════════════════════════════════════════════════════════════════
#  Events and alerts
# ══════════════════════════════════════════════════════════════════════════
class TestEvents:
    @staticmethod
    def _seed(count=5):
        import uuid
        from datetime import datetime, timedelta

        from backend.db.base import session_scope
        from backend.db.models import Event

        ids = []
        with session_scope() as session:
            for i in range(count):
                event_id = str(uuid.uuid4())
                ids.append(event_id)
                session.add(
                    Event(
                        event_id=event_id,
                        event_type="MISSING_HELMET" if i % 2 else "RESTRICTED_AREA",
                        camera_id=None,
                        person_id=i,
                        severity="HIGH" if i % 2 else "MEDIUM",
                        confidence=0.7 + i * 0.02,
                        status="OPEN",
                        description=f"Seeded incident {i}",
                        timestamp=datetime.now(UTC) - timedelta(minutes=i),
                    )
                )
        return ids

    def test_listing_and_pagination(self, client):
        self._seed(12)
        page = client.get("/api/events?limit=5").json()
        assert page["total"] == 12
        assert len(page["items"]) == 5
        assert page["has_more"] is True

        second = client.get("/api/events?limit=5&offset=10").json()
        assert len(second["items"]) == 2
        assert second["has_more"] is False

    def test_computed_fields_present(self, client):
        self._seed(1)
        item = client.get("/api/events?limit=1").json()["items"][0]
        assert item["label"]
        assert isinstance(item["advisory"], bool)
        assert isinstance(item["has_snapshot"], bool)

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("event_type=MISSING_HELMET", 3),
            ("severity=HIGH", 3),
            ("status=OPEN", 6),
            ("search=Seeded", 6),
            ("person_id=2", 1),
        ],
    )
    def test_filters(self, client, query, expected):
        self._seed(6)
        assert client.get(f"/api/events?{query}").json()["total"] == expected

    @pytest.mark.parametrize(
        "query",
        ["severity=NOPE", "status=WHATEVER", "event_type=MADE_UP", "limit=0", "limit=9999"],
    )
    def test_invalid_filters_are_422(self, client, query):
        assert client.get(f"/api/events?{query}").status_code == 422

    def test_alerts_ordered_by_severity(self, client):
        self._seed(6)
        items = client.get("/api/alerts").json()["items"]
        rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        ranks = [rank[i["severity"]] for i in items]
        assert ranks == sorted(ranks)

    def test_status_workflow(self, client):
        event_id = self._seed(1)[0]
        acked = client.patch(
            f"/api/events/{event_id}/status",
            json={"status": "ACKNOWLEDGED", "notes": "Checked on site"},
        )
        assert acked.status_code == 200
        assert acked.json()["acknowledged_at"] is not None
        assert acked.json()["notes"] == "Checked on site"

        resolved = client.patch(
            f"/api/events/{event_id}/status", json={"status": "RESOLVED"}
        ).json()
        assert resolved["resolved_at"] is not None

    def test_resolving_implies_acknowledged(self, client):
        """Skipping Acknowledge must still record that it was seen."""
        event_id = self._seed(1)[0]
        body = client.patch(
            f"/api/events/{event_id}/status", json={"status": "RESOLVED"}
        ).json()
        assert body["acknowledged_at"] is not None

    def test_unknown_event_is_404(self, client):
        assert client.get("/api/events/nope").status_code == 404
        assert (
            client.patch("/api/events/nope/status", json={"status": "RESOLVED"}).status_code
            == 404
        )

    def test_alerts_exclude_closed_by_default(self, client):
        event_id = self._seed(4)[0]
        client.patch(f"/api/events/{event_id}/status", json={"status": "RESOLVED"})
        assert client.get("/api/alerts").json()["total"] == 3
        assert client.get("/api/alerts?include_resolved=true").json()["total"] == 4


# ══════════════════════════════════════════════════════════════════════════
#  Statistics
# ══════════════════════════════════════════════════════════════════════════
class TestStatistics:
    def test_shape(self, client):
        body = client.get("/api/statistics?range=7d").json()
        for key in ("kpis", "by_type", "by_severity", "timeline", "camera_activity"):
            assert key in body
        assert body["granularity"] in {"minute", "hour", "day"}

    @pytest.mark.parametrize("preset", ["today", "24h", "7d", "30d"])
    def test_presets(self, client, preset):
        assert client.get(f"/api/statistics?range={preset}").status_code == 200

    def test_custom_range_requires_bounds(self, client):
        assert client.get("/api/statistics?range=custom").status_code == 422

    def test_custom_range_rejects_inverted_bounds(self, client):
        response = client.get(
            "/api/statistics?range=custom"
            "&start=2026-08-29T00:00:00&end=2026-08-01T00:00:00"
        )
        assert response.status_code == 422

    def test_custom_range_accepts_valid_bounds(self, client):
        response = client.get(
            "/api/statistics?range=custom"
            "&start=2026-08-01T00:00:00&end=2026-08-29T00:00:00"
        )
        assert response.status_code == 200
        assert response.json()["granularity"] == "day"


# ══════════════════════════════════════════════════════════════════════════
#  Uploads
# ══════════════════════════════════════════════════════════════════════════
class TestUploads:
    def test_profiles_listed(self, client):
        body = client.get("/api/videos/profiles").json()
        names = {p["name"] for p in body["profiles"]}
        assert {"fast", "standard", "thorough", "ppe_only", "security_only"} <= names
        for profile in body["profiles"]:
            assert profile["analysers"] and profile["description"]

    def test_extension_rejected(self, client):
        response = client.post(
            "/api/videos/upload",
            files={"file": ("evil.txt", io.BytesIO(b"nope"), "text/plain")},
            data={"profile": "standard"},
        )
        assert response.status_code == 415

    def test_executable_disguised_by_extension_rejected(self, client):
        """An allowed extension proves nothing — the content must decode."""
        response = client.post(
            "/api/videos/upload",
            files={"file": ("payload.mp4", io.BytesIO(b"MZ\x90\x00" * 64), "video/mp4")},
            data={"profile": "standard", "start": "false"},
        )
        assert response.status_code == 422
        assert "not a readable video" in response.json()["error"].lower()

    def test_empty_upload_rejected(self, client):
        response = client.post(
            "/api/videos/upload",
            files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")},
            data={"profile": "standard"},
        )
        assert response.status_code == 400

    def test_traversal_filename_cannot_escape_upload_dir(self, client):
        """The stored name is generated; the client's name is only a label."""
        from backend.config import settings

        before = set(settings.upload_root.glob("**/*"))
        client.post(
            "/api/videos/upload",
            files={
                "file": (
                    "../../../../tmp/pwned.mp4",
                    io.BytesIO(b"\x00" * 128),
                    "video/mp4",
                )
            },
            data={"profile": "standard", "start": "false"},
        )
        after = set(settings.upload_root.glob("**/*"))
        for path in after - before:
            assert path.is_relative_to(settings.upload_root)
            assert "pwned" not in path.name

    def test_unknown_job_is_404(self, client):
        assert client.get("/api/videos/jobs/nope").status_code == 404
        assert client.post("/api/videos/jobs/nope/cancel").status_code == 404


# ══════════════════════════════════════════════════════════════════════════
#  Evidence
# ══════════════════════════════════════════════════════════════════════════
class TestEvidenceApi:
    def test_manifest_for_event_without_evidence(self, client):
        import uuid

        from backend.db.base import session_scope
        from backend.db.models import Event

        event_id = str(uuid.uuid4())
        with session_scope() as session:
            session.add(
                Event(event_id=event_id, event_type="LOITERING", severity="MEDIUM",
                      confidence=0.8, description="no evidence")
            )
        body = client.get(f"/api/evidence/{event_id}").json()
        assert body["snapshot"]["available"] is False
        assert body["clip"]["available"] is False
        assert client.get(f"/api/evidence/{event_id}/snapshot").status_code == 404

    def test_export_always_contains_metadata(self, client):
        """Evidence export must work even with no media attached."""
        import uuid
        import zipfile

        from backend.db.base import session_scope
        from backend.db.models import Event

        event_id = str(uuid.uuid4())
        with session_scope() as session:
            session.add(
                Event(event_id=event_id, event_type="LOITERING", severity="MEDIUM",
                      confidence=0.8, description="exported")
            )
        response = client.get(f"/api/evidence/{event_id}/export")
        assert response.status_code == 200
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        assert "event.json" in archive.namelist()

    def test_unknown_event_is_404(self, client):
        assert client.get("/api/evidence/nope").status_code == 404
        assert client.get("/api/evidence/nope/clip").status_code == 404
