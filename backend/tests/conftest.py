"""Shared fixtures.

Tests marked ``db`` run against the *test* database (JUDGE_CHECK_TEST_DATABASE_URL), never
the main one. If it isn't reachable they are skipped, unless JUDGE_CHECK_REQUIRE_DB=1, in
which case they fail: CI sets that so a missing database can never turn into a green run.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from judge_check.api.main import app
from judge_check.config import get_settings
from judge_check.db import get_session

BACKEND_DIR = Path(__file__).resolve().parents[1]


def alembic_config(url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.attributes["url"] = url
    return cfg


@pytest.fixture(scope="session")
def db_url() -> str:
    return get_settings().test_database_url


@pytest.fixture(scope="session")
def db_engine(db_url: str) -> Iterator[Engine]:
    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        engine.dispose()
        msg = f"test database unreachable ({db_url}); run `docker compose up -d`: {exc.orig}"
        if os.environ.get("JUDGE_CHECK_REQUIRE_DB") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    command.upgrade(alembic_config(db_url), "head")
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    """A session on an empty test database (every corpus and its dependents removed)."""
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE corpus CASCADE"))
    with sessionmaker(bind=db_engine)() as session:
        yield session


@pytest.fixture
def client(db_engine: Engine) -> Iterator[TestClient]:
    """API client whose requests use the test database."""

    def _session() -> Iterator[Session]:
        with sessionmaker(bind=db_engine)() as session:
            yield session

    app.dependency_overrides[get_session] = _session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
