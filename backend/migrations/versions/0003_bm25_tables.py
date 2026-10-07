"""BM25 in SQL: per-chunk lengths, per-term statistics, postings, per-chunk-set stats.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _chunk_set_fk() -> sa.Column:
    return sa.Column(
        "chunk_set_id",
        sa.Integer,
        sa.ForeignKey("chunk_set.id", ondelete="CASCADE"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "bm25_stats",
        sa.Column(
            "chunk_set_id",
            sa.Integer,
            sa.ForeignKey("chunk_set.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("tokenizer", sa.Text, nullable=False),
        sa.Column("n_docs", sa.Integer, nullable=False),
        sa.Column("avgdl", sa.Float, nullable=False),
        sa.Column("avg_idf", sa.Float, nullable=False),
        sa.Column("epsilon", sa.Float, nullable=False),
    )
    op.create_table(
        "bm25_doc",
        sa.Column(
            "chunk_id",
            sa.BigInteger,
            sa.ForeignKey("chunk.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        _chunk_set_fk(),
        sa.Column("length", sa.Integer, nullable=False),
        sa.Index("ix_bm25_doc_chunk_set_id", "chunk_set_id"),
    )
    op.create_table(
        "bm25_term",
        _chunk_set_fk(),
        sa.Column("term", sa.Text, nullable=False),
        sa.Column("df", sa.Integer, nullable=False),
        sa.Column("idf", sa.Float, nullable=False),
        sa.PrimaryKeyConstraint("chunk_set_id", "term"),
    )
    op.create_table(
        "bm25_posting",
        _chunk_set_fk(),
        sa.Column("term", sa.Text, nullable=False),
        sa.Column(
            "chunk_id", sa.BigInteger, sa.ForeignKey("chunk.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("tf", sa.Integer, nullable=False),
        sa.PrimaryKeyConstraint("chunk_set_id", "term", "chunk_id"),
    )


def downgrade() -> None:
    for table in ["bm25_posting", "bm25_term", "bm25_doc", "bm25_stats"]:
        op.drop_table(table)
