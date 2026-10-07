"""D-030: no LLM-output row can be stored without the exact model version, temperature,
seed, max_tokens and prompt hash. Enforced by the database, tested for EVERY table that
uses LLMCallColumns.

When stage 4 adds the judgement table, it must use LLMCallColumns and get a row builder
below. `test_every_llm_output_table_is_covered` fails until it does.
"""

import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from judge_check.db import Base
from judge_check.ingest.chunking import SentenceWindow
from judge_check.ingest.pipeline import build_chunk_set, embed_chunk_set
from judge_check.llm.base import Completion, Message
from judge_check.llm.records import call_columns
from judge_check.llm.runner import CallRecord
from judge_check.models import GeneratedAnswer, LLMCallColumns, Question
from judge_check.prompts import load_prompt
from tests.fakes import HashingEmbedder
from tests.test_ingest_retrieval import DOCS, QUESTIONS

pytestmark = pytest.mark.db

LLM_OUTPUT_TABLES = sorted(
    (m.class_ for m in Base.registry.mappers if issubclass(m.class_, LLMCallColumns)),
    key=lambda c: c.__name__,
)


def a_call() -> CallRecord:
    prompt = load_prompt("grounded_answer", 1)
    return CallRecord(
        prompt,
        [Message("user", "q")],
        Completion(
            text="{}", provider="ollama", model_requested="qwen2.5:3b",
            model_reported="qwen2.5:3b", model_version="qwen2.5:3b@sha256:abc",
            temperature=0.0, seed=0, max_tokens=256, latency_ms=5, input_tokens=10,
            output_tokens=2, cost_usd=0.0, raw={"done": True},
        ),
    )  # fmt: skip


def generated_answer_row(session: Session) -> dict:
    from judge_check.ingest.pipeline import load_corpus

    corpus = load_corpus(session, "llmcols", DOCS, QUESTIONS)
    cs = build_chunk_set(session, corpus, SentenceWindow(sentences_per_chunk=1))
    run = embed_chunk_set(session, cs, HashingEmbedder())
    qid = session.query(Question.id).filter_by(corpus_id=corpus.id).first()[0]
    return {
        "question_id": qid,
        "embedding_run_id": run.id,
        "retrieval_mode": "dense",
        "retrieved_chunk_ids": [],
        "answer": "x",
        "cited_chunk_ids": [],
        "answerable": True,
        **call_columns(session, a_call()),
    }


ROW_BUILDERS = {GeneratedAnswer: generated_answer_row}


def test_every_llm_output_table_is_covered() -> None:
    assert LLM_OUTPUT_TABLES, "expected at least one LLM-output table"
    missing = [t.__name__ for t in LLM_OUTPUT_TABLES if t not in ROW_BUILDERS]
    assert missing == [], f"add a row builder for {missing} so D-030 is tested on it"


@pytest.mark.parametrize("table", LLM_OUTPUT_TABLES, ids=lambda t: t.__tablename__)
def test_valid_row_is_accepted(db_session: Session, table) -> None:
    db_session.execute(insert(table), ROW_BUILDERS[table](db_session))
    db_session.flush()


@pytest.mark.parametrize("column", LLMCallColumns.REQUIRED)
@pytest.mark.parametrize("table", LLM_OUTPUT_TABLES, ids=lambda t: t.__tablename__)
def test_row_without_required_attribution_is_rejected(
    db_session: Session, table, column: str
) -> None:
    row = ROW_BUILDERS[table](db_session) | {column: None}
    with pytest.raises(IntegrityError, match="not-null|violates"):
        db_session.execute(insert(table), row)
        db_session.flush()


@pytest.mark.parametrize("column", ["provider", "model_requested", "model_reported",
                                    "model_version"])  # fmt: skip
@pytest.mark.parametrize("table", LLM_OUTPUT_TABLES, ids=lambda t: t.__tablename__)
def test_empty_model_identity_is_rejected(db_session: Session, table, column: str) -> None:
    row = ROW_BUILDERS[table](db_session) | {column: ""}
    with pytest.raises(IntegrityError, match="attribution"):
        db_session.execute(insert(table), row)
        db_session.flush()
