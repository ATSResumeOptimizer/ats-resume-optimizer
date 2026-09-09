"""
Database session management.

Uses SQLAlchemy with a connection URL from the DATABASE_URL environment variable.
Defaults to a local SQLite file so the app works out of the box without Postgres.

Environment variables
---------------------
DATABASE_URL   SQLAlchemy connection URL.
               SQLite (default) : sqlite:///./ats_optimizer.db
               PostgreSQL       : postgresql+psycopg2://user:pass@host/dbname
               PostgreSQL async : postgresql+asyncpg://user:pass@host/dbname
                                  (not needed here — the worker is synchronous)

Usage
-----
    from db import SessionLocal, init_db

    # Create all tables (call once at startup)
    init_db()

    # Use a session
    db = SessionLocal()
    try:
        ...
    finally:
        db.close()

    # Or use the context manager helper
    with get_db() as db:
        ...
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from models import Base

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ats_optimizer.db")

# SQLite-specific: enable WAL mode and foreign keys for better concurrency
_connect_args: dict = {}
if _DATABASE_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(
    _DATABASE_URL,
    connect_args=_connect_args,
    # Keep a small pool; the RQ worker is synchronous so one connection at a time is typical
    pool_pre_ping=True,
)

# Enable FK enforcement on every SQLite connection
if _DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):  # type: ignore[misc]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # safe to read attributes after commit without re-query
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager that opens a session and closes it on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Create all tables that don't already exist.
    Safe to call on every startup — existing tables are not modified.
    For production schema migrations use Alembic instead.
    """
    Base.metadata.create_all(bind=engine)
