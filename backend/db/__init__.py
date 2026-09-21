"""Persistence layer.

SQLAlchemy 2.x is used throughout so the only change required to move from
SQLite to PostgreSQL is `DATABASE_URL`. No raw SQL, no SQLite-specific types.
"""

from backend.db.base import Base, get_session, init_db, session_scope

__all__ = ["Base", "get_session", "init_db", "session_scope"]
