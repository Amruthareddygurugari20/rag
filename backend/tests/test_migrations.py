import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine

import judge_check.models  # noqa: F401  (registers tables)
from judge_check.db import Base

pytestmark = pytest.mark.db


def test_migrations_match_orm_models(db_engine: Engine) -> None:
    """Migrations are written by hand, so check they produce exactly what the models say."""
    with db_engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []
