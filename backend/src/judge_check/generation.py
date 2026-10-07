"""Stage 2: grounded answer generation over retrieved chunks, with citations.

    question -> retrieve top-k chunks -> label them C1..Ck -> prompt (grounded_answer.v1)
             -> JSON {answer, citations, answerable} -> map labels back to chunk ids -> store

What a generated answer is NOT: a labelled example. A model wrote it, so its correctness is
unknown (D-000). It becomes part of the evaluation set only through human adjudication in
stage 5, as label_source = "human" (D-032). The diagnostics computed here (citation overlap
with gold chunks, reference-answer containment) are descriptive heuristics, never labels.

Parsing is strict. A citation to a label the model was not shown, malformed JSON, wrong
types, or an "answerable" answer with no citations all make the answer a *parse failure*:
`answer` is stored as NULL with the reason in `parse_error`, and the raw output stays in
`output_text`. Nothing is silently repaired. A silently dropped bad citation would make the
model look better grounded than it was.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from judge_check.embeddings import Embedder
from judge_check.llm.base import DEFAULT_SEED, LLMClient
from judge_check.llm.records import call_columns
from judge_check.llm.runner import run_prompt
from judge_check.models import Chunk, Document, GeneratedAnswer, Question
from judge_check.prompts import load_prompt
from judge_check.retrieval.search import Mode, get_run, search

PROMPT_NAME, PROMPT_VERSION = "grounded_answer", 1

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "answerable": {"type": "boolean"},
    },
    "required": ["answer", "citations", "answerable"],
    "additionalProperties": False,
}

_LABEL = re.compile(r"^\[?\s*([Cc])\s*(\d+)\s*\]?$")


@dataclass(frozen=True)
class LabelledChunk:
    label: str  # "C1".."Ck", in retrieval order
    chunk_id: int
    title: str
    text: str


def label_chunks(chunks: list[tuple[int, str, str]]) -> list[LabelledChunk]:
    """[(chunk_id, title, text)] in retrieval order -> C1..Ck. Short labels, because models
    copy "C3" reliably and mangle "chunk 1849203"."""
    return [LabelledChunk(f"C{i}", cid, title, text) for i, (cid, title, text) in
            enumerate(chunks, start=1)]  # fmt: skip


def render_context(labelled: list[LabelledChunk]) -> str:
    return "\n\n".join(f"[{c.label}] ({c.title}) {c.text}" for c in labelled)


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str | None
    cited_chunk_ids: list[int]
    answerable: bool | None
    error: str | None  # None means valid


def _fail(reason: str) -> ParsedAnswer:
    return ParsedAnswer(None, [], None, reason)


def parse_answer(output: str, labelled: list[LabelledChunk]) -> ParsedAnswer:
    by_label = {c.label: c.chunk_id for c in labelled}
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        return _fail(f"not JSON: {exc.msg}")
    if not isinstance(data, dict):
        return _fail("JSON is not an object")
    answer, citations, answerable = (data.get(k) for k in ("answer", "citations", "answerable"))
    if not isinstance(answer, str) or not answer.strip():
        return _fail("'answer' missing or not a non-empty string")
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        return _fail("'citations' missing or not a list of strings")
    if not isinstance(answerable, bool):
        return _fail("'answerable' missing or not a boolean")

    labels, unshown = [], []
    for raw in citations:
        m = _LABEL.match(raw.strip())
        label = f"C{int(m.group(2))}" if m else raw.strip()
        if label not in by_label:
            unshown.append(raw)
        elif label not in labels:  # keep first-cited order, drop exact repeats
            labels.append(label)
    if unshown:
        return _fail(f"cited passages that were not shown: {unshown}")
    if answerable and not labels:
        return _fail("answerable=true but no citations: the answer is ungrounded")
    return ParsedAnswer(answer.strip(), [by_label[lb] for lb in labels], answerable, None)


def generate_answer(
    session: Session,
    client: LLMClient,
    question_id: int,
    chunk_set_id: int,
    embedder: Embedder,
    *,
    mode: Mode = "dense",
    k: int = 5,
    temperature: float = 0.0,
    max_tokens: int = 256,
    seed: int = DEFAULT_SEED,
) -> GeneratedAnswer:
    """Retrieve, prompt, parse, and store one grounded answer (always stored, even when
    the output fails to parse: failures are data)."""
    question = session.get(Question, question_id)
    if question is None:
        raise LookupError(f"no question {question_id}")
    run = get_run(session, chunk_set_id, embedder.spec.name)
    hits = search(session, chunk_set_id, question.text, mode=mode, k=k, embedder=embedder)
    ids = [h.chunk_id for h in hits]
    rows = {
        r.id: r
        for r in session.execute(
            select(Chunk.id, Chunk.text, Document.title)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(ids))
        )
    }
    labelled = label_chunks([(cid, rows[cid].title, rows[cid].text) for cid in ids])

    prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
    record = run_prompt(
        client,
        prompt,
        {"context": render_context(labelled), "question": question.text},
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        json_schema=ANSWER_SCHEMA,
    )
    parsed = parse_answer(record.completion.text, labelled)
    row = GeneratedAnswer(
        question_id=question.id,
        embedding_run_id=run.id,
        retrieval_mode=mode,
        retrieved_chunk_ids=ids,
        answer=parsed.answer,
        cited_chunk_ids=parsed.cited_chunk_ids,
        answerable=parsed.answerable,
        parse_error=parsed.error,
        **call_columns(session, record),
    )
    session.add(row)
    session.flush()
    return row


@dataclass(frozen=True)
class ExclusionRow:
    """How many generations per exact model version can't become variants (D-032).

    Parse failures are unusable outputs, not wrong answers: they're excluded from the
    variant pool, and this rate is reported per model as a selection-bias statement. If a
    generator fails more on hard questions, the surviving pool skews easy. `unanswerable`
    is reported alongside: it's a valid output, but if a later stage keeps only answerable
    generations, that's a second filter of the same kind.
    """

    model_version: str
    total: int
    parse_failures: int
    unanswerable: int

    @property
    def model_requested_tag(self) -> str:
        # model_version is "tag@digest" for Ollama, "model+fingerprint" for Azure.
        return self.model_version.split("@", 1)[0].split("+", 1)[0]

    @property
    def excluded_rate(self) -> float:
        return self.parse_failures / self.total if self.total else 0.0


def exclusion_report(session: Session, corpus_id: int | None = None) -> list[ExclusionRow]:
    from sqlalchemy import case, func

    q = select(
        GeneratedAnswer.model_version,
        func.count(),
        func.count(GeneratedAnswer.parse_error),
        func.sum(case((GeneratedAnswer.answerable.is_(False), 1), else_=0)),
    ).group_by(GeneratedAnswer.model_version)
    if corpus_id is not None:
        q = q.join(Question, Question.id == GeneratedAnswer.question_id).where(
            Question.corpus_id == corpus_id
        )
    return [
        ExclusionRow(mv, total, failures, int(unans or 0))
        for mv, total, failures, unans in session.execute(q.order_by(GeneratedAnswer.model_version))
    ]
