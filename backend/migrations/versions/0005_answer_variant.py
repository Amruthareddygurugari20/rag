"""Stage 3: answer_variant, with the label rules enforced as CHECK constraints (D-035).

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen copies: a migration must not change when models.py later does.
ACCEPT = ("correct", "verbose_correct", "terse_correct", "paraphrased_correct")
REJECT = (
    "right_topic_wrong_detail", "hedged_nonanswer", "partially_correct",
    "unsupported_but_plausible", "right_answer_wrong_citation",
)  # fmt: skip


NOW = sa.text("now()")


def q(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def upgrade() -> None:
    op.create_table(
        "answer_variant",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "question_id",
            sa.BigInteger,
            sa.ForeignKey("question.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "chunk_set_id",
            sa.Integer,
            sa.ForeignKey("chunk_set.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seed", sa.BigInteger, nullable=False),
        sa.Column("variant_type", sa.Text, nullable=False),
        sa.Column("sub_kind", sa.Text),
        sa.Column("is_correct", sa.Boolean),
        sa.Column("label_source", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("discard_reason", sa.Text),
        sa.Column("constructor", sa.Text, nullable=False),
        sa.Column("constructor_version", sa.Integer, nullable=False),
        sa.Column("params", JSONB, nullable=False),
        sa.Column("text", sa.Text),
        sa.Column("cited_chunk_ids", JSONB, nullable=False),
        sa.Column("guard_report", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.UniqueConstraint(
            "question_id",
            "chunk_set_id",
            "seed",
            "variant_type",
            "sub_kind",
            postgresql_nulls_not_distinct=True,
        ),  # fmt: skip
        sa.CheckConstraint(f"variant_type IN ({q(ACCEPT + REJECT)})", name="ck_variant_type"),
        sa.CheckConstraint("label_source IN ('construction', 'human')", name="ck_label_source"),
        sa.CheckConstraint(
            "status IN ('labelled', 'discarded', 'needs_human_review')", name="ck_status"
        ),
        sa.CheckConstraint(
            f"label_source <> 'construction' OR is_correct = (variant_type IN ({q(ACCEPT)}))",
            name="ck_constructed_label_matches_type",
        ),
        sa.CheckConstraint(
            "status <> 'needs_human_review' OR (label_source = 'human' AND is_correct IS NULL)",
            name="ck_review_has_no_label",
        ),
        sa.CheckConstraint(
            "(status = 'discarded') = (discard_reason IS NOT NULL)", name="ck_discard_reason"
        ),
        sa.CheckConstraint("(status = 'discarded') = (text IS NULL)", name="ck_text_iff_kept"),
        sa.CheckConstraint("status <> 'labelled' OR is_correct IS NOT NULL", name="ck_labelled"),
    )


def downgrade() -> None:
    op.drop_table("answer_variant")
