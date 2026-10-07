"""Stage 1 schema: corpora, documents, questions with gold evidence spans, chunk sets,
chunks, embedding runs and chunk embeddings.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table(
        "corpus",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text, nullable=False, unique=True),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
    )
    op.create_table(
        "document",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "corpus_id", sa.Integer, sa.ForeignKey("corpus.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("external_id", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False, server_default=""),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("sentence_spans", JSONB),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.UniqueConstraint("corpus_id", "external_id"),
    )
    op.create_table(
        "question",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "corpus_id", sa.Integer, sa.ForeignKey("corpus.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("external_id", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("reference_answer", sa.Text, nullable=False),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.UniqueConstraint("corpus_id", "external_id"),
    )
    op.create_table(
        "question_evidence",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "question_id",
            sa.BigInteger,
            sa.ForeignKey("question.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "document_id",
            sa.BigInteger,
            sa.ForeignKey("document.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.CheckConstraint("char_start >= 0 AND char_end > char_start"),
    )
    op.create_table(
        "chunk_set",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "corpus_id", sa.Integer, sa.ForeignKey("corpus.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("label", sa.Text, nullable=False),
        sa.Column("config", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.UniqueConstraint("corpus_id", "label"),
    )
    op.create_table(
        "chunk",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "chunk_set_id",
            sa.Integer,
            sa.ForeignKey("chunk_set.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.BigInteger,
            sa.ForeignKey("document.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.UniqueConstraint("chunk_set_id", "document_id", "position"),
        sa.CheckConstraint("char_start >= 0 AND char_end > char_start"),
    )
    op.create_table(
        "embedding_run",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "chunk_set_id",
            sa.Integer,
            sa.ForeignKey("chunk_set.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_name", sa.Text, nullable=False),
        sa.Column("dimension", sa.Integer, nullable=False),
        sa.Column("normalized", sa.Boolean, nullable=False),
        sa.Column("query_instruction", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.UniqueConstraint("chunk_set_id", "model_name"),
    )
    op.create_table(
        "chunk_embedding",
        sa.Column(
            "embedding_run_id",
            sa.Integer,
            sa.ForeignKey("embedding_run.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "chunk_id",
            sa.BigInteger,
            sa.ForeignKey("chunk.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("embedding", Vector(), nullable=False),
    )


def downgrade() -> None:
    for table in [
        "chunk_embedding",
        "embedding_run",
        "chunk",
        "chunk_set",
        "question_evidence",
        "question",
        "document",
        "corpus",
    ]:
        op.drop_table(table)
