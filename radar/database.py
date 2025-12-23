"""
Database session management and initialization.

Supports both SQLite (local development) and PostgreSQL (Supabase production).
"""
from __future__ import annotations

import os
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Optional, Union

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from radar.models import Base


# Default database path for SQLite
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "radar.db"


def get_database_url(db_path: Optional[Union[Path, str]] = None) -> str:
    """
    Get database URL.
    
    Priority:
    1. DATABASE_URL env var (for Supabase/PostgreSQL)
    2. RADAR_DB_PATH env var (for custom SQLite path)
    3. Default SQLite path
    """
    # Check for Supabase/PostgreSQL connection
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        # Supabase uses postgres:// but SQLAlchemy needs postgresql://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return database_url
    
    # Fall back to SQLite
    if db_path is None:
        db_path = os.environ.get("RADAR_DB_PATH", DEFAULT_DB_PATH)
    return f"sqlite:///{db_path}"


def is_postgres() -> bool:
    """Check if we're using PostgreSQL."""
    return os.environ.get("DATABASE_URL") is not None


def create_db_engine(db_path: Optional[Union[Path, str]] = None, echo: bool = False):
    """Create SQLAlchemy engine with appropriate optimizations."""
    url = get_database_url(db_path)
    
    if url.startswith("postgresql"):
        # PostgreSQL configuration
        engine = create_engine(
            url, 
            echo=echo,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Handle stale connections
        )
    else:
        # SQLite configuration
        engine = create_engine(url, echo=echo)
        
        # Enable SQLite optimizations
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    
    return engine


# Global engine and session factory (lazy initialization)
_engine = None
_SessionFactory = None


def get_engine(db_path: Optional[Union[Path, str]] = None, echo: bool = False):
    """Get or create the global database engine."""
    global _engine
    if _engine is None:
        # Only create directory for SQLite
        if not is_postgres():
            db_file = Path(db_path) if db_path else DEFAULT_DB_PATH
            db_file.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_db_engine(db_path, echo=echo)
    return _engine


def get_session_factory(db_path: Optional[Union[Path, str]] = None) -> sessionmaker:
    """Get or create the global session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        engine = get_engine(db_path)
        _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    return _SessionFactory


def init_database(db_path: Optional[Union[Path, str]] = None, echo: bool = False) -> None:
    """Initialize database schema (create all tables)."""
    engine = get_engine(db_path, echo=echo)
    Base.metadata.create_all(engine)


def reset_database(db_path: Optional[Union[Path, str]] = None) -> None:
    """Drop and recreate all tables. USE WITH CAUTION."""
    engine = get_engine(db_path)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


@contextmanager
def get_session(db_path: Optional[Union[Path, str]] = None) -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Usage:
        with get_session() as session:
            session.add(some_object)
            session.commit()
    """
    factory = get_session_factory(db_path)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_session(db_path: Optional[Union[Path, str]] = None) -> Session:
    """
    Create a new session. Caller is responsible for closing.
    
    For most use cases, prefer get_session() context manager.
    """
    factory = get_session_factory(db_path)
    return factory()
