"""Engine, session factory and schema bootstrap.

Design note — portability: the engine is built from `settings.database_url`
and only SQLite gets the special-casing it genuinely requires
(`check_same_thread=False`, because the video pipeline threads share the
engine, and WAL mode so readers do not block the writer). Everything else in
the codebase talks to `Session` and is therefore backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Engine, TypeDecorator, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from backend.config import settings
from backend.logging_conf import get_logger

logger = get_logger(__name__)


def utcnow() -> datetime:
    """Timezone-aware UTC now — the single source of time for the DB layer."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """Always store naive UTC, always return timezone-aware UTC.

    SQLite has no native tz support, so a plain `DateTime` round-trips as
    naive and silently breaks comparisons against `utcnow()`. This decorator
    normalises both directions and behaves identically on PostgreSQL.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


def _build_engine(url: str) -> Engine:
    """Create the engine with dialect-appropriate pooling."""
    if url.startswith("sqlite"):
        is_memory = ":memory:" in url
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            # An in-memory SQLite DB is per-connection; StaticPool keeps the
            # single connection alive so tests see one shared database.
            poolclass=StaticPool if is_memory else QueuePool,
            pool_pre_ping=True,
            future=True,
        )

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                if not is_memory:
                    cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

        return engine

    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20, future=True)


engine: Engine = _build_engine(settings.resolved_database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables and seed defaults. Safe to call repeatedly."""
    from backend.db import models  # noqa: F401  (registers mappers)

    settings.ensure_directories()
    Base.metadata.create_all(bind=engine)
    add_missing_columns()
    logger.info(
        "Database ready: %s",
        settings.resolved_database_url.split("://", 1)[0],
    )


def add_missing_columns() -> None:
    """Add columns present in the models but absent from an existing table.

    `create_all` creates missing *tables* and nothing else, so a new field on
    an existing model is invisible to a database that predates it — the next
    query then fails with "no such column". A full migration tool is more than
    this project needs, but silently requiring operators to drop a database
    full of recorded incidents to pick up one nullable column is worse.

    Deliberately additive only: it never drops, renames, retypes or reorders
    anything, so it cannot lose data. A column that needs any of those is a
    real migration and should be written as one.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # create_all just made it, in full
        present = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            if not column.nullable and column.default is None:
                logger.error(
                    "Column %s.%s is NOT NULL with no default and cannot be "
                    "added automatically — this needs a real migration.",
                    table.name, column.name,
                )
                continue
            ddl = column.type.compile(engine.dialect)
            try:
                with engine.begin() as connection:
                    connection.execute(
                        text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {ddl}")
                    )
            except Exception as exc:
                logger.error(
                    "Could not add column %s.%s: %s", table.name, column.name, exc
                )
                continue
            logger.info("Added missing column %s.%s (%s)", table.name, column.name, ddl)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for non-request code (pipeline threads, jobs)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
