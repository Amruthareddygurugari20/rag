"""Smoke tests that pgvector is installed and its distance operators behave as we will rely on.

`<=>` is pgvector's cosine *distance*: 1 - cosine_similarity. So identical direction -> 0,
orthogonal -> 1, opposite -> 2. Stage 1 builds retrieval on exactly this operator.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from tests.conftest import alembic_config

pytestmark = pytest.mark.db


def cosine_distance(session: Session, a: str, b: str) -> float:
    return session.execute(
        text("SELECT CAST(:a AS vector) <=> CAST(:b AS vector)"), {"a": a, "b": b}
    ).scalar_one()


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("[1, 0]", "[1, 0]", 0.0),  # same direction
        ("[1, 2]", "[2, 4]", 0.0),  # same direction, different length: cosine ignores magnitude
        ("[1, 0]", "[0, 1]", 1.0),  # orthogonal
        ("[1, 0]", "[-1, 0]", 2.0),  # opposite
    ],
)
def test_cosine_distance(db_session: Session, a: str, b: str, expected: float) -> None:
    assert cosine_distance(db_session, a, b) == pytest.approx(expected, abs=1e-6)


def test_migration_downgrade_and_upgrade_roundtrip(
    db_engine: Engine, db_url: str, client: TestClient
) -> None:
    from alembic import command

    cfg = alembic_config(db_url)
    try:
        command.downgrade(cfg, "base")
        resp = client.get("/health")
        assert resp.status_code == 503
        assert "missing" in resp.json()["pgvector"]
    finally:
        command.upgrade(cfg, "head")
    assert client.get("/health").status_code == 200
