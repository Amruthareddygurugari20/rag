"""Database engine and session handling (sync SQLAlchemy 2.0 over psycopg 3, see D-007)."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from judge_check.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models. Tables are added stage by stage (D-008)."""


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    # pool_pre_ping: survive a `docker compose restart db` without stale-connection errors.
    return create_engine(url or get_settings().database_url, pool_pre_ping=True)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    with factory() as session:
        yield session


def pgvector_version(session: Session) -> str | None:
    """Installed version of the pgvector extension, or None if it isn't installed."""
    return session.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar_one_or_none()
