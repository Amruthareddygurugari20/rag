"""answer_variant: the label rules hold in the database itself, and the database build
matches the in-memory build exactly (D-035)."""

import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from judge_check.datasets.demo import load_demo
from judge_check.ingest.chunking import DEFAULT_CHUNKING
from judge_check.ingest.pipeline import build_chunk_set, load_corpus
from judge_check.models import AnswerVariant, Question
from judge_check.variants.build import build_variants
from tests.test_ingest_retrieval import DOCS, QUESTIONS
from tests.test_variant_types import demo_in_memory  # same construction, no database

pytestmark = pytest.mark.db
SEED = 20261007


@pytest.fixture
def tiny(db_session: Session):
    corpus = load_corpus(db_session, "tiny-variants", DOCS, QUESTIONS)
    return build_chunk_set(db_session, corpus, DEFAULT_CHUNKING)


@pytest.fixture
def demo(db_session: Session):
    docs, qs = load_demo()
    corpus = load_corpus(db_session, "demo-variants", docs, qs)
    return build_chunk_set(db_session, corpus, DEFAULT_CHUNKING)


def a_row(session: Session, chunk_set_id: int, **over) -> dict:
    qid = session.scalar(select(Question.id).limit(1))
    row = {
        "question_id": qid, "chunk_set_id": chunk_set_id, "seed": 1,
        "variant_type": "correct", "sub_kind": None, "is_correct": True,
        "label_source": "construction", "status": "labelled", "discard_reason": None,
        "constructor": "correct", "constructor_version": 1, "params": {},
        "text": "Bergen.", "cited_chunk_ids": [], "guard_report": [],
    }  # fmt: skip
    return {**row, **over}


def test_valid_rows_are_accepted(db_session: Session, tiny) -> None:
    db_session.execute(insert(AnswerVariant), a_row(db_session, tiny.id))
    db_session.execute(
        insert(AnswerVariant),
        a_row(db_session, tiny.id, variant_type="paraphrased_correct", is_correct=None,
              label_source="human", status="needs_human_review"),
    )  # fmt: skip
    db_session.flush()


@pytest.mark.parametrize(
    ("over", "constraint"),
    [
        # Each row breaks exactly one rule, so the named constraint is the one that fires.
        ({"variant_type": "correct_ish", "label_source": "human"}, "ck_variant_type"),
        ({"is_correct": False}, "ck_constructed_label_matches_type"),  # accept type, wrong
        ({"variant_type": "partially_correct", "is_correct": True},
         "ck_constructed_label_matches_type"),  # reject type labelled correct
        ({"status": "needs_human_review"}, "ck_review_has_no_label"),  # construction label
        ({"status": "needs_human_review", "label_source": "human"}, "ck_review_has_no_label"),
        ({"status": "discarded", "text": None}, "ck_discard_reason"),  # no reason
        ({"status": "discarded", "discard_reason": "x"}, "ck_text_iff_kept"),  # text kept
        ({"text": None}, "ck_text_iff_kept"),  # labelled without text
        ({"label_source": "human", "is_correct": None}, "ck_labelled"),
        ({"label_source": "model"}, "ck_label_source"),
        ({"status": "pending"}, "ck_status"),
    ],
)  # fmt: skip
def test_impossible_rows_are_refused(db_session: Session, tiny, over, constraint) -> None:
    with pytest.raises(IntegrityError, match=constraint):
        db_session.execute(insert(AnswerVariant), a_row(db_session, tiny.id, **over))
        db_session.flush()
    db_session.rollback()


def test_db_build_matches_in_memory_build_and_is_idempotent(db_session: Session, demo) -> None:
    report = build_variants(db_session, demo, seed=SEED)
    rows = db_session.execute(
        select(Question.external_id, AnswerVariant.variant_type, AnswerVariant.sub_kind,
               AnswerVariant.status, AnswerVariant.discard_reason, AnswerVariant.text)
        .join(Question, Question.id == AnswerVariant.question_id)
    ).all()  # fmt: skip
    expected = {(m.key, o.variant_type, o.sub_kind, o.discard_reason, o.text)
                for m, o in demo_in_memory()}  # fmt: skip
    got = {(q, t, sk, reason, text) for q, t, sk, _, reason, text in rows}
    assert got == expected
    assert len(rows) == 7 * 200
    # Rebuild at the same seed: replaced, not duplicated, and identical.
    again = build_variants(db_session, demo, seed=SEED)
    assert again == report
    assert db_session.scalar(select(func.count()).select_from(AnswerVariant)) == 7 * 200
