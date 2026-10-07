"""Build Materials from a corpus and a chunking, run the constructors, store the variants.

`material_for` works on plain values, so the same code serves the database build and the
in-memory one the tests (and `discard_report`) use.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from judge_check.models import AnswerVariant, Chunk, ChunkSet, Document, Question
from judge_check.variants.constructors import VERSIONS, Outcome, deterministic_variants
from judge_check.variants.material import Material, Paragraph, Sentence
from judge_check.variants.mutate import strip_disambiguation

ChunkRow = tuple[int, int, int, str]  # (chunk id, char_start, char_end, text)


@dataclass(frozen=True)
class DocView:
    title: str
    text: str
    sentence_spans: Sequence[Sequence[int]]


def _sentence(doc: DocView, chunks: Sequence[ChunkRow], start: int, end: int) -> Sentence:
    ids = tuple(c for c, s, e, _ in chunks if s < end and start < e)
    return Sentence(doc.text[start:end], ids)


def material_for(
    key: str,
    answer: str,
    evidence: Iterable[tuple[str, int, int]],  # (document id, start, end), in evidence order
    distractor_ids: Iterable[str],
    docs: Mapping[str, DocView],
    chunks: Mapping[str, Sequence[ChunkRow]],
) -> Material:
    evidence = list(evidence)
    gold = tuple(_sentence(docs[d], chunks.get(d, ()), s, e) for d, s, e in evidence)
    paragraphs = []
    for d in dict.fromkeys(d for d, _, _ in evidence):
        doc = docs[d]
        spans = [tuple(sp) for sp in doc.sentence_spans]
        gold_spans = {(s, e) for dd, s, e in evidence if dd == d}
        sents = tuple(_sentence(doc, chunks.get(d, ()), s, e) for s, e in spans)
        paragraphs.append(
            Paragraph(
                doc.title, sents, frozenset(i for i, sp in enumerate(spans) if sp in gold_spans)
            )  # fmt: skip
        )
    distractors = [d for d in distractor_ids if d in docs]
    return Material(
        key=key,
        answer=answer,
        gold=gold,
        gold_paragraphs=tuple(paragraphs),
        distractor_chunks=tuple((c, t) for d in distractors for c, _, _, t in chunks.get(d, ())),
        entity_candidates=tuple(strip_disambiguation(docs[d].title) for d in distractors),
    )


def discard_report(outcomes: Iterable[Outcome]) -> dict[str, Counter]:
    """Per type (and type 7 sub-kind): kept count and each discard reason's count."""
    out: dict[str, Counter] = {}
    for o in outcomes:
        name = o.variant_type + (f"/{o.sub_kind}" if o.variant_type == "partially_correct" else "")
        out.setdefault(name, Counter())[o.discard_reason or "kept"] += 1
    return out


# --- database ---------------------------------------------------------------------------------


def materials_from_db(session: Session, chunk_set: ChunkSet) -> list[tuple[int, Material]]:
    docs = {
        d.external_id: (d.id, DocView(d.title, d.text, d.sentence_spans or []))
        for d in session.scalars(select(Document).where(Document.corpus_id == chunk_set.corpus_id))
    }
    by_pk = {pk: ext for ext, (pk, _) in docs.items()}
    chunks: dict[str, list[ChunkRow]] = {}
    for c in session.scalars(
        select(Chunk).where(Chunk.chunk_set_id == chunk_set.id).order_by(Chunk.id)
    ):
        chunks.setdefault(by_pk[c.document_id], []).append((c.id, c.char_start, c.char_end, c.text))
    views = {ext: v for ext, (_, v) in docs.items()}
    out = []
    questions = session.scalars(
        select(Question).where(Question.corpus_id == chunk_set.corpus_id).order_by(Question.id)
    )
    for q in questions:
        evidence = [(by_pk[e.document_id], e.char_start, e.char_end) for e in q.evidence]
        distractors = q.meta.get("distractor_titles", [])
        out.append(
            (
                q.id,
                material_for(
                    q.external_id, q.reference_answer, evidence, distractors, views, chunks
                ),
            )  # fmt: skip
        )
    return out


def build_variants(session: Session, chunk_set: ChunkSet, *, seed: int) -> dict[str, Counter]:
    """(Re)build every deterministic variant for a chunk set at this seed. Idempotent: rows
    for the same (chunk set, seed) and these types are replaced, so a rebuild is identical."""
    session.execute(
        delete(AnswerVariant).where(
            AnswerVariant.chunk_set_id == chunk_set.id,
            AnswerVariant.seed == seed,
            AnswerVariant.variant_type.in_(list(VERSIONS)),
        )
    )
    outcomes = []
    for question_id, m in materials_from_db(session, chunk_set):
        for o in deterministic_variants(m, seed=seed):
            outcomes.append(o)
            session.add(row_for(o, question_id, chunk_set.id, seed))
    session.flush()
    return discard_report(outcomes)


def row_for(o: Outcome, question_id: int, chunk_set_id: int, seed: int) -> AnswerVariant:
    return AnswerVariant(
        question_id=question_id,
        chunk_set_id=chunk_set_id,
        seed=seed,
        variant_type=o.variant_type,
        sub_kind=o.sub_kind,
        is_correct=o.is_correct,
        label_source="construction",
        status="discarded" if o.discarded else "labelled",
        discard_reason=o.discard_reason,
        constructor=o.variant_type,
        constructor_version=VERSIONS[o.variant_type],
        params=o.params,
        text=o.text,
        cited_chunk_ids=list(o.cited),
        guard_report=o.guard_report,
    )
