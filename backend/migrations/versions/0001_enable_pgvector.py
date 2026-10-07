"""Enable the pgvector extension.

The extension lives in the database, not the image: the pgvector/pgvector image ships
the compiled extension, but each database must still opt in with CREATE EXTENSION.
Doing it in a migration (not an init script) means it also happens on any Postgres
the user brings themselves, e.g. a managed instance with pgvector available.

Revision ID: 0001
Revises:
Create Date: 2026-10-07
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
