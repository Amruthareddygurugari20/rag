"""Stage 2: prompt_template and generated_answer, with the attribution columns every
LLM-output table carries (D-030, D-031).

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NOW = sa.text("now()")


def llm_call_columns(table: str) -> list:
    """Mirror of models.LLMCallColumns. Reuse for every LLM-output table (stage 4 too)."""
    return [
        sa.Column("provider", sa.Text, nullable=False),
        sa.Column("model_requested", sa.Text, nullable=False),
        sa.Column("model_reported", sa.Text, nullable=False),
        sa.Column("model_version", sa.Text, nullable=False),
        sa.Column("temperature", sa.Float, nullable=False),
        sa.Column("seed", sa.Integer, nullable=False),
        sa.Column("max_tokens", sa.Integer, nullable=False),
        sa.Column(
            "prompt_sha256", sa.Text, sa.ForeignKey("prompt_template.sha256"), nullable=False
        ),
        sa.Column("messages", JSONB, nullable=False),
        sa.Column("output_text", sa.Text, nullable=False),
        sa.Column("raw_response", JSONB, nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("input_tokens", sa.Integer),
        sa.Column("output_tokens", sa.Integer),
        sa.Column("cost_usd", sa.Float),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.CheckConstraint(
            "provider <> '' AND model_requested <> '' AND model_reported <> '' "
            "AND model_version <> ''",
            name=f"ck_{table}_attribution",
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "prompt_template",
        sa.Column("sha256", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("system", sa.Text, nullable=False),
        sa.Column("user", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.UniqueConstraint("name", "version"),
    )
    op.create_table(
        "generated_answer",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "question_id",
            sa.BigInteger,
            sa.ForeignKey("question.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "embedding_run_id", sa.Integer, sa.ForeignKey("embedding_run.id"), nullable=False
        ),
        sa.Column("retrieval_mode", sa.Text, nullable=False),
        sa.Column("retrieved_chunk_ids", JSONB, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("cited_chunk_ids", JSONB, nullable=False),
        sa.Column("answerable", sa.Boolean),
        sa.Column("parse_error", sa.Text),
        *llm_call_columns("generated_answer"),
    )


def downgrade() -> None:
    op.drop_table("generated_answer")
    op.drop_table("prompt_template")
