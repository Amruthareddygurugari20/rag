from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from judge_check.api.main import app
from judge_check.db import get_session


@pytest.mark.db
def test_health_reports_db_and_pgvector(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    # pgvector versions look like "0.8.0".
    assert body["pgvector"].count(".") == 2


def test_health_returns_503_when_database_unreachable() -> None:
    # Port 1 on localhost: nothing listens there, so the connection is refused immediately.
    engine = create_engine(
        "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none",
        connect_args={"connect_timeout": 2},
    )

    def _session() -> Iterator[Session]:
        with sessionmaker(bind=engine)() as session:
            yield session

    app.dependency_overrides[get_session] = _session
    try:
        resp = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()
        engine.dispose()

    assert resp.status_code == 503
    assert resp.json()["database"] == "unreachable"
