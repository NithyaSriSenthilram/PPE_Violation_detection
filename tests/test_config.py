"""Configuration loading and derived values."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.config import PROJECT_ROOT, Settings, settings


def test_paths_resolve_to_absolute():
    """Relative configured paths must resolve against the project root, not cwd.

    Without this, running uvicorn from a different directory silently creates a
    second, empty database.
    """
    assert settings.evidence_root.is_absolute()
    assert settings.upload_root.is_absolute()
    assert settings.snapshot_dir.parent == settings.evidence_root
    assert settings.clip_dir.parent == settings.evidence_root


def test_sqlite_url_is_rewritten_to_absolute():
    candidate = Settings(database_url="sqlite:///./data/x.db")
    resolved = candidate.resolved_database_url
    assert resolved.startswith("sqlite:////") or str(PROJECT_ROOT) in resolved


def test_non_sqlite_url_is_untouched():
    url = "postgresql+psycopg://user:pass@db:5432/sentinel"
    assert Settings(database_url=url).resolved_database_url == url


def test_extension_set_is_normalised():
    candidate = Settings(allowed_video_extensions="mp4, .AVI ,mov")
    assert candidate.allowed_extension_set == {".mp4", ".avi", ".mov"}


def test_cors_origins_split():
    candidate = Settings(cors_origins="http://a.test, http://b.test ,")
    assert candidate.cors_origin_list == ["http://a.test", "http://b.test"]


def test_invalid_log_level_rejected():
    with pytest.raises(ValidationError):
        Settings(log_level="chatty")


def test_threshold_bounds_enforced():
    with pytest.raises(ValidationError):
        Settings(confidence_threshold=1.5)
    with pytest.raises(ValidationError):
        Settings(detect_every_n_frames=0)


def test_upload_bytes_derived():
    assert Settings(max_upload_mb=2).max_upload_bytes == 2 * 1024 * 1024


# ── Deployment: PORT, DATA_ROOT, SQLite location ──────────────────────────


def test_port_env_wins_over_app_port(monkeypatch):
    """Hosting platforms inject PORT; it must be honoured without APP_PORT."""
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.delenv("APP_PORT", raising=False)
    assert Settings().app_port == 10000


def test_app_port_still_works_without_port(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setenv("APP_PORT", "8123")
    assert Settings().app_port == 8123


_RUNTIME_PATH_KEYS = ("UPLOAD_DIR", "PROCESSED_DIR", "EVIDENCE_DIR", "DATABASE_URL")


def _clear_runtime_paths(monkeypatch):
    # conftest pins these for the suite; DATA_ROOT only fills paths left unset.
    for key in _RUNTIME_PATH_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_data_root_derives_every_runtime_path(tmp_path, monkeypatch):
    """DATA_ROOT alone must relocate uploads, renders, evidence and SQLite."""
    _clear_runtime_paths(monkeypatch)
    candidate = Settings(data_root=tmp_path)
    assert candidate.upload_root == tmp_path / "uploads"
    assert candidate.processed_root == tmp_path / "processed"
    assert candidate.evidence_root == tmp_path / "evidence"
    assert candidate.sqlite_file == tmp_path / "db" / "surveillance.db"


def test_explicit_path_overrides_data_root(tmp_path, monkeypatch):
    _clear_runtime_paths(monkeypatch)
    candidate = Settings(data_root=tmp_path, upload_dir=tmp_path / "elsewhere")
    assert candidate.upload_root == tmp_path / "elsewhere"
    assert candidate.processed_root == tmp_path / "processed"


def test_ensure_directories_creates_sqlite_parent(tmp_path, monkeypatch):
    _clear_runtime_paths(monkeypatch)
    candidate = Settings(data_root=tmp_path / "disk")
    candidate.ensure_directories()
    assert (tmp_path / "disk" / "db").is_dir()
    assert (tmp_path / "disk" / "uploads").is_dir()
    assert (tmp_path / "disk" / "processed").is_dir()
    assert (tmp_path / "disk" / "evidence" / "snapshots").is_dir()


def test_sqlite_file_is_none_for_other_engines():
    assert Settings(database_url="postgresql+psycopg://u:p@db/x").sqlite_file is None
