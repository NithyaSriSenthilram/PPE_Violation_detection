"""Event engine: severity, cooldown, burst cap, persistence, evidence."""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pytest

from backend.analysis.base import EventCandidate
from backend.db.base import session_scope
from backend.db.models import Event
from backend.events.engine import CooldownRegistry, EventEngine
from backend.events.evidence import (
    EvidenceWriter,
    evidence_stem,
    resolve_evidence_path,
    sanitise,
)
from backend.events.types import (
    DEFAULT_SEVERITY,
    EventStatus,
    EventType,
    Severity,
    default_severity,
    is_advisory,
    label_for,
)


def candidate(
    event_type=EventType.RESTRICTED_AREA,
    person_id=1,
    zone_id="zA",
    confidence=0.9,
) -> EventCandidate:
    return EventCandidate(
        event_type=event_type,
        confidence=confidence,
        description=f"Person #{person_id} triggered {event_type}",
        person_id=person_id,
        zone_id=zone_id,
    )


def engine(**kwargs) -> EventEngine:
    defaults = {"persist": False, "broadcast": False, "capture_evidence": False}
    defaults.update(kwargs)
    return EventEngine(**defaults)


# ══════════════════════════════════════════════════════════════════════════
#  Taxonomy
# ══════════════════════════════════════════════════════════════════════════
class TestTaxonomy:
    def test_every_type_has_a_severity_and_label(self):
        for event_type in EventType:
            assert default_severity(event_type) in {s.value for s in Severity}
            assert label_for(event_type) and label_for(event_type) != event_type.value

    def test_unknown_type_degrades_gracefully(self):
        """A type added server-side must not break older consumers."""
        assert default_severity("SOME_NEW_TYPE") == Severity.MEDIUM
        assert label_for("SOME_NEW_TYPE") == "Some New Type"

    def test_inference_types_are_advisory(self):
        assert is_advisory(EventType.POSSIBLE_FALL)
        assert is_advisory(EventType.ABNORMAL_MOVEMENT)
        assert not is_advisory(EventType.RESTRICTED_AREA)

    def test_documented_severities(self):
        """Spec §14: missing vest MEDIUM, restricted area HIGH, fall HIGH."""
        assert DEFAULT_SEVERITY[EventType.MISSING_VEST] == Severity.MEDIUM
        assert DEFAULT_SEVERITY[EventType.RESTRICTED_AREA] == Severity.HIGH
        assert DEFAULT_SEVERITY[EventType.POSSIBLE_FALL] == Severity.HIGH


# ══════════════════════════════════════════════════════════════════════════
#  Cooldown
# ══════════════════════════════════════════════════════════════════════════
class TestCooldown:
    def test_continuous_violation_collapses_to_few_alerts(self):
        """Spec §14: one continuous violation must not produce hundreds."""
        eng = engine(cooldown=CooldownRegistry(default_seconds=30, max_per_minute=0))
        accepted = 0
        for i in range(1400):  # ~2 min at 12 fps
            accepted += len(eng.submit([candidate()], camera_id="cam", monotonic_time=i / 12))
        assert 1 <= accepted <= 6, f"expected a handful of alerts, got {accepted}"

    def test_distinct_people_and_zones_stay_distinct(self):
        eng = engine(cooldown=CooldownRegistry(default_seconds=30, max_per_minute=0))
        accepted = eng.submit(
            [
                candidate(person_id=1, zone_id="zA"),
                candidate(person_id=2, zone_id="zA"),
                candidate(person_id=1, zone_id="zB"),
            ],
            camera_id="cam",
            monotonic_time=0.0,
        )
        assert len(accepted) == 3

    def test_different_cameras_are_independent(self):
        eng = engine(cooldown=CooldownRegistry(default_seconds=30, max_per_minute=0))
        assert len(eng.submit([candidate()], camera_id="cam-1", monotonic_time=0.0)) == 1
        assert len(eng.submit([candidate()], camera_id="cam-2", monotonic_time=0.0)) == 1

    def test_fires_again_after_the_window(self):
        eng = engine(cooldown=CooldownRegistry(default_seconds=10, max_per_minute=0))
        assert len(eng.submit([candidate()], camera_id="cam", monotonic_time=0.0)) == 1
        assert len(eng.submit([candidate()], camera_id="cam", monotonic_time=5.0)) == 0
        assert len(eng.submit([candidate()], camera_id="cam", monotonic_time=11.0)) == 1

    def test_burst_cap_survives_person_id_churn(self):
        """The per-person debounce cannot bite when every ID is new.

        This is the crowded-doorway / heavy-occlusion case, and it is how a
        system without a burst cap floods the operator.
        """
        eng = engine(cooldown=CooldownRegistry(default_seconds=30, max_per_minute=10))
        accepted = 0
        for i in range(400):
            accepted += len(
                eng.submit([candidate(person_id=i)], camera_id="cam", monotonic_time=i * 0.1)
            )
        assert accepted <= 20, f"burst cap did not hold: {accepted}"
        assert eng.stats()["rejected_burst_cap"] > 300

    def test_burst_cap_disabled_by_zero(self):
        eng = engine(cooldown=CooldownRegistry(default_seconds=0, max_per_minute=0))
        accepted = sum(
            len(eng.submit([candidate(person_id=i)], camera_id="cam", monotonic_time=i * 0.1))
            for i in range(50)
        )
        assert accepted == 50


# ══════════════════════════════════════════════════════════════════════════
#  Filtering and severity
# ══════════════════════════════════════════════════════════════════════════
class TestEngine:
    def test_low_confidence_rejected(self):
        eng = engine()
        assert eng.submit([candidate(confidence=0.05)], camera_id="cam", monotonic_time=0.0) == []
        assert eng.stats()["rejected_low_confidence"] == 1

    def test_severity_override_applied(self):
        eng = engine()
        eng.set_severity_overrides({EventType.MISSING_VEST.value: Severity.CRITICAL.value})
        accepted = eng.submit(
            [candidate(event_type=EventType.MISSING_VEST)], camera_id="cam", monotonic_time=0.0
        )
        assert accepted[0].severity == Severity.CRITICAL

    def test_invalid_override_ignored(self):
        eng = engine()
        eng.set_severity_overrides({EventType.MISSING_VEST.value: "CATASTROPHIC"})
        accepted = eng.submit(
            [candidate(event_type=EventType.MISSING_VEST)], camera_id="cam", monotonic_time=0.0
        )
        assert accepted[0].severity == Severity.MEDIUM

    def test_advisory_notice_attached(self):
        eng = engine()
        accepted = eng.submit(
            [candidate(event_type=EventType.POSSIBLE_FALL)], camera_id="cam", monotonic_time=0.0
        )
        assert "advisory" in accepted[0].detection_metadata
        assert accepted[0].to_payload()["advisory"] is True

    def test_event_type_serialises_as_plain_string(self):
        eng = engine()
        accepted = eng.submit([candidate()], camera_id="cam", monotonic_time=0.0)
        assert type(accepted[0].event_type) is str

    def test_payload_shape_matches_documented_schema(self):
        eng = engine()
        payload = eng.submit([candidate()], camera_id="cam", monotonic_time=0.0)[0].to_payload()
        for key in (
            "event_id", "event_type", "camera_id", "person_id", "severity",
            "confidence", "timestamp", "description", "snapshot_path",
            "clip_path", "status",
        ):
            assert key in payload, f"missing {key}"
        assert payload["status"] == EventStatus.OPEN.value

    def test_upload_scope_isolates_cooldowns(self):
        """Two concurrent jobs must not suppress each other's incidents."""
        eng = engine(cooldown=CooldownRegistry(default_seconds=60, max_per_minute=0))
        a = eng.submit([candidate()], camera_id=None, scope="upload-a", monotonic_time=0.0)
        b = eng.submit([candidate()], camera_id=None, scope="upload-b", monotonic_time=0.0)
        assert len(a) == 1 and len(b) == 1

    def test_persisted_row_matches_record(self):
        eng = engine(persist=True)
        accepted = eng.submit([candidate()], camera_id=None, monotonic_time=0.0)
        with session_scope() as session:
            row = session.get(Event, accepted[0].event_id)
            assert row is not None
            assert row.event_type == EventType.RESTRICTED_AREA.value
            assert row.status == EventStatus.OPEN.value
            assert row.severity == Severity.HIGH.value

    def test_database_failure_does_not_break_detection(self, monkeypatch):
        """A DB outage must not stop the pipeline raising events.

        The failure is injected at `session_scope` — the layer that actually
        breaks when the database is unreachable — so this exercises the real
        resilience path rather than a stubbed-out method.
        """
        import backend.events.engine as engine_module

        def broken_scope():
            raise RuntimeError("database is gone")

        monkeypatch.setattr(engine_module, "session_scope", broken_scope)
        eng = engine(persist=True)
        accepted = eng.submit([candidate()], camera_id=None, monotonic_time=0.0)
        assert len(accepted) == 1  # detection continues; the event is in-process


# ══════════════════════════════════════════════════════════════════════════
#  Evidence
# ══════════════════════════════════════════════════════════════════════════
class TestEvidence:
    def test_stem_carries_provenance(self):
        from datetime import datetime

        stem = evidence_stem(
            "cam-01", "PPE_VIOLATION", "abc-123", datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
        )
        assert "cam-01" in stem
        assert "PPE_VIOLATION" in stem
        assert "abc-123" in stem
        assert "20260829T120000" in stem

    @pytest.mark.parametrize(
        "raw",
        ["../../etc/passwd", "a/b/c", "name with spaces", "sem;icolon", ""],
    )
    def test_sanitise_removes_path_characters(self, raw):
        cleaned = sanitise(raw)
        assert "/" not in cleaned and ".." not in cleaned and " " not in cleaned
        assert cleaned

    def test_snapshot_written_and_readable(self):
        import cv2

        writer = EvidenceWriter()
        frame = np.full((120, 200, 3), 90, np.uint8)
        relative = writer.write_snapshot(frame, "cam", "TEST_EVENT", "evt-1")
        assert relative
        resolved = resolve_evidence_path(relative)
        assert resolved is not None and resolved.is_file()
        assert cv2.imread(str(resolved)) is not None

    def test_write_and_resolve_agree_on_the_base_path(self):
        """Regression: writing and resolving must share one base directory.

        They previously disagreed (project root vs `evidence_root.parent`),
        which worked only while EVIDENCE_DIR sat directly under the project
        root and broke every evidence download for any other configuration.
        """
        writer = EvidenceWriter()
        frame = np.zeros((60, 80, 3), np.uint8)
        for event_id in ("a1", "b2", "c3"):
            relative = writer.write_snapshot(frame, "cam", "T", event_id)
            assert relative, "snapshot was not written"
            assert resolve_evidence_path(relative) is not None, (
                f"stored path {relative!r} did not resolve back to a file"
            )

    @pytest.mark.parametrize(
        "path",
        ["../../../etc/passwd", "/etc/passwd", "evidence/../../secret", ""],
    )
    def test_paths_outside_evidence_dir_are_refused(self, path):
        """Two barriers: clients cannot name files, and a bad stored value
        still cannot escape EVIDENCE_DIR."""
        assert resolve_evidence_path(path) is None
