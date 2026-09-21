"""Shared test fixtures.

Every test runs against an in-memory SQLite database and the mock detector, so
the suite needs no model file, no GPU and no network — which is the point of
the mock backend existing (§30's "test mode using mock detections").
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Point the settings singleton at a throwaway environment *before* any backend
# module imports it. Settings are cached, so this has to happen first.
_TEST_ROOT = Path(__file__).resolve().parent
os.environ["SENTINEL_ENV_FILE"] = str(_TEST_ROOT / "test.env")
os.environ.update(
    {
        "APP_ENV": "development",
        # A temp *file*, not ":memory:". An in-memory SQLite database is
        # per-connection, so it needs StaticPool — which makes every session
        # share one connection. Camera pipelines write telemetry from their own
        # threads while a request holds a transaction, and on a shared
        # connection that collides ("cannot commit - no transaction is active").
        # A file database gives each thread its own connection, exactly as in
        # production, so the tests exercise the real concurrency behaviour.
        "DATABASE_URL": f"sqlite:///{_TEST_ROOT / '_artifacts' / 'test.db'}",
        "INFERENCE_BACKEND": "mock",
        "EVIDENCE_DIR": str(_TEST_ROOT / "_artifacts" / "evidence"),
        "UPLOAD_DIR": str(_TEST_ROOT / "_artifacts" / "uploads"),
        "LOG_LEVEL": "WARNING",
        "CLIP_ENABLED": "false",
        "TARGET_FPS": "0",
        "EVENT_COOLDOWN_SECONDS": "30",
        "EVENT_MAX_PER_MINUTE": "0",
        # Hard-hat validation is a second opinion on a real photograph of a
        # head. The rest of the suite works on synthetic frames — flat colour
        # and black rectangles — where the honest answer is always "that is not
        # a hard hat", which would turn every PPE fixture into a violation and
        # test the validator instead of the thing under test. It is switched on
        # explicitly in tests/test_hardhat.py, which is where it belongs.
        "HARDHAT_VALIDATION_ENABLED": "false",
    }
)

from backend.config import settings  # noqa: E402
from backend.db.base import Base, SessionLocal, engine  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _prepare_environment():
    """Create tables once and ensure the artifact directories exist."""
    (_TEST_ROOT / "_artifacts").mkdir(parents=True, exist_ok=True)
    settings.ensure_directories()
    import backend.db.models  # noqa: F401

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    # Leave no test database behind.
    engine.dispose()
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{_TEST_ROOT / '_artifacts' / 'test.db'}{suffix}")
        candidate.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def clean_database():
    """Stop any pipelines, then truncate every table between tests.

    Pipelines are stopped first: a running worker writes camera telemetry, and
    truncating underneath it would either resurrect rows or fail the delete.
    """
    yield
    from backend.video.manager import get_manager

    get_manager().stop_all()
    with SessionLocal() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


@pytest.fixture
def session():
    """A database session for direct model work."""
    with SessionLocal() as s:
        yield s


@pytest.fixture
def client():
    """FastAPI test client with lifespan startup skipped.

    The real lifespan resolves a detector and starts cameras; tests exercise
    those separately and do not want threads running underneath them.
    """
    from fastapi.testclient import TestClient

    from backend.main import create_app

    app = create_app()
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as test_client:
        yield test_client


from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def _noop_lifespan(app):  # noqa: ANN001, ANN201
    yield


@pytest.fixture
def mock_detector():
    """A loaded synthetic detector."""
    from backend.inference.backends.mock_backend import MockDetector

    detector = MockDetector(people=3)
    detector.load()
    return detector


@pytest.fixture
def blank_frame():
    """A single black 960x540 BGR frame."""
    import numpy as np

    return np.zeros((540, 960, 3), dtype=np.uint8)


@pytest.fixture
def camera(session):
    """A persisted camera row."""
    import uuid

    from backend.db.models import Camera

    record = Camera(
        camera_id=str(uuid.uuid4()),
        name="Test Camera",
        location="Test Site",
        source_type="file",
        stream_url="sample.mp4",
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record
