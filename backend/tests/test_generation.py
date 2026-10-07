"""Stage 2 generation: strict parsing, label mapping, and the stored row."""

import json

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from judge_check.generation import (
    ANSWER_SCHEMA,
    generate_answer,
    label_chunks,
    parse_answer,
    render_context,
)
from judge_check.ingest.chunking import SentenceWindow
from judge_check.ingest.pipeline import build_chunk_set, embed_chunk_set, load_corpus
from judge_check.llm.base import Completion
from judge_check.models import GeneratedAnswer, PromptTemplateRow, Question
from judge_check.prompts import load_prompt
from tests.fakes import HashingEmbedder
from tests.test_ingest_retrieval import DOCS, QUESTIONS

LABELLED = label_chunks([(101, "Alpha", "Alpha was founded."), (205, "Beta", "Ann was born.")])


def out(**kw) -> str:
    base = {"answer": "Bergen.", "citations": ["C2"], "answerable": True}
    return json.dumps(base | kw)


def test_labels_follow_retrieval_order() -> None:
    assert [(c.label, c.chunk_id) for c in LABELLED] == [("C1", 101), ("C2", 205)]
    assert (
        render_context(LABELLED) == "[C1] (Alpha) Alpha was founded.\n\n[C2] (Beta) Ann was born."
    )


@pytest.mark.parametrize(
    ("citations", "expected_ids"),
    [
        (["C2"], [205]),
        (["C2", "C1"], [205, 101]),  # cited order kept
        (["[C1]", "c2", " C1 "], [101, 205]),  # brackets/case/space normalised; repeat dropped
    ],
)
def test_valid_answers_map_labels_to_chunk_ids(citations, expected_ids) -> None:
    p = parse_answer(out(citations=citations), LABELLED)
    assert p.error is None and p.cited_chunk_ids == expected_ids and p.answer == "Bergen."


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (out(citations=["C2", "C7"]), "not shown"),  # C7 was never shown
        (out(citations=["chunk 205"]), "not shown"),  # raw ids aren't labels
        ("Bergen [C2]", "not JSON"),
        (json.dumps(["Bergen"]), "not an object"),
        (out(answer=""), "'answer'"),
        (out(citations="C2"), "'citations'"),
        (out(answerable="yes"), "'answerable'"),
        (out(citations=[]), "ungrounded"),  # answerable but cites nothing
    ],
)
def test_parse_failures_are_reported_never_repaired(output, reason) -> None:
    p = parse_answer(output, LABELLED)
    assert reason in p.error
    assert p.answer is None and p.cited_chunk_ids == [] and p.answerable is None


def test_unanswerable_without_citations_is_valid() -> None:
    p = parse_answer(out(answer="Not in the passages.", citations=[], answerable=False), LABELLED)
    assert p.error is None and p.answerable is False


class ScriptedClient:
    """Answers by citing whichever label holds the word 'Bergen' (or none)."""

    provider, model = "fake", "fake-llm"

    def __init__(self, raw_output: str | None = None) -> None:
        self.raw_output = raw_output
        self.seen: dict = {}

    def complete(self, messages, **kw):
        self.seen = {"messages": messages, **kw}
        ctx = messages[1].content
        label = next((ln[1:3] for ln in ctx.split("\n\n") if "Bergen" in ln), None)
        text = self.raw_output or json.dumps(
            {"answer": "Bergen.", "citations": [label] if label else [],
             "answerable": bool(label)}
        )  # fmt: skip
        return Completion(
            text=text, provider="fake", model_requested="fake-llm", model_reported="fake-llm",
            model_version="fake-llm@sha256:feed", temperature=kw["temperature"],
            seed=kw["seed"], max_tokens=kw["max_tokens"], latency_ms=3, input_tokens=50,
            output_tokens=9, cost_usd=0.0, raw={"ok": True},
        )  # fmt: skip


@pytest.fixture
def setup(db_session: Session):
    corpus = load_corpus(db_session, "gen", DOCS, QUESTIONS)
    cs = build_chunk_set(db_session, corpus, SentenceWindow(sentences_per_chunk=1))
    emb = HashingEmbedder()
    embed_chunk_set(db_session, cs, emb)
    q1 = db_session.scalar(select(Question).where(Question.external_id == "q1"))
    return cs, emb, q1


@pytest.mark.db
def test_generate_answer_stores_a_grounded_attributed_row(db_session: Session, setup) -> None:
    cs, emb, q1 = setup
    client = ScriptedClient()
    row = generate_answer(db_session, client, q1.id, cs.id, emb, k=7, seed=11)
    assert client.seen["json_schema"] == ANSWER_SCHEMA and client.seen["seed"] == 11
    assert row.parse_error is None and row.answer == "Bergen." and row.answerable is True
    # The cited id is the retrieved chunk whose text contains "Bergen".
    assert len(row.cited_chunk_ids) == 1 and row.cited_chunk_ids[0] in row.retrieved_chunk_ids
    # Attribution and the exact prompt, recorded on the row (D-030, D-031).
    assert row.model_version == "fake-llm@sha256:feed" and row.temperature == 0.0
    prompt = load_prompt("grounded_answer", 1)
    assert row.prompt_sha256 == prompt.sha256
    assert db_session.get(PromptTemplateRow, prompt.sha256).version == 1
    assert row.messages[1]["content"].startswith("Context passages:\n[C1]")


@pytest.mark.db
def test_unparseable_output_is_stored_as_a_failure(db_session: Session, setup) -> None:
    cs, emb, q1 = setup
    bad = json.dumps({"answer": "x", "citations": ["C99"], "answerable": True})
    row = generate_answer(db_session, ScriptedClient(bad), q1.id, cs.id, emb, k=3)
    assert "not shown" in row.parse_error and row.answer is None
    assert row.output_text == bad  # raw output kept for audit
    assert db_session.scalar(select(GeneratedAnswer.id).where(GeneratedAnswer.id == row.id))


@pytest.mark.db
def test_exclusion_report_counts_per_model_version(db_session: Session, setup) -> None:
    from judge_check.generation import exclusion_report

    cs, emb, q1 = setup
    generate_answer(db_session, ScriptedClient(), q1.id, cs.id, emb, k=7)  # ok
    bad = json.dumps({"answer": "x", "citations": ["C42"], "answerable": True})
    generate_answer(db_session, ScriptedClient(bad), q1.id, cs.id, emb, k=3)  # failure
    unans = json.dumps({"answer": "Not stated.", "citations": [], "answerable": False})
    generate_answer(db_session, ScriptedClient(unans), q1.id, cs.id, emb, k=3)
    [row] = exclusion_report(db_session, cs.corpus_id)
    assert (row.model_version, row.total, row.parse_failures, row.unanswerable) == (
        "fake-llm@sha256:feed",
        3,
        1,
        1,
    )
    assert row.excluded_rate == pytest.approx(1 / 3)
    assert row.model_requested_tag == "fake-llm"
