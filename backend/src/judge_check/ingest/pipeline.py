"""Load an evaluation set into Postgres, chunk it, and embed the chunks.

Three steps, each idempotent and separately callable:

    corpus    = load_corpus(session, "hotpotqa_demo", documents, questions)
    chunk_set = build_chunk_set(session, corpus, SentenceWindow(sentences_per_chunk=2))  # + BM25
    run       = embed_chunk_set(session, chunk_set, embedder)
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import delete, insert, select, text
from sqlalchemy.orm import Session

from judge_check.embeddings import Embedder, check_unit_norm
from judge_check.ingest.chunking import ChunkingConfig, chunk_spans
from judge_check.ingest.formats import DocumentIn, QuestionIn
from judge_check.models import (
    Chunk,
    ChunkEmbedding,
    ChunkSet,
    Corpus,
    Document,
    EmbeddingRun,
    Question,
    QuestionEvidence,
)
from judge_check.retrieval import bm25


def load_corpus(
    session: Session,
    name: str,
    documents: list[DocumentIn],
    questions: list[QuestionIn],
    *,
    description: str = "",
    replace: bool = False,
) -> Corpus:
    existing = session.scalar(select(Corpus).where(Corpus.name == name))
    if existing is not None:
        if not replace:
            raise ValueError(f"corpus {name!r} already exists (pass replace=True to reload)")
        session.delete(existing)
        session.flush()

    corpus = Corpus(name=name, description=description)
    session.add(corpus)
    session.flush()

    doc_ids = dict(
        session.execute(
            insert(Document).returning(Document.external_id, Document.id),
            [
                {
                    "corpus_id": corpus.id,
                    "external_id": d.id,
                    "title": d.title,
                    "text": d.text,
                    "sentence_spans": [list(s) for s in d.sentence_spans]
                    if d.sentence_spans is not None
                    else None,
                    "meta": d.metadata,
                }
                for d in documents
            ],
        ).all()
    )
    for q in questions:
        session.add(
            Question(
                corpus_id=corpus.id,
                external_id=q.id,
                text=q.question,
                reference_answer=q.reference_answer,
                meta=q.metadata,
                evidence=[
                    QuestionEvidence(
                        document_id=doc_ids[ev.document_id], char_start=ev.start, char_end=ev.end
                    )
                    for ev in q.gold_evidence
                ],
            )
        )
    session.flush()
    return corpus


def build_chunk_set(session: Session, corpus: Corpus, config: ChunkingConfig) -> ChunkSet:
    label = config.label()
    existing = session.scalar(
        select(ChunkSet).where(ChunkSet.corpus_id == corpus.id, ChunkSet.label == label)
    )
    if existing is not None:
        return existing

    chunk_set = ChunkSet(corpus_id=corpus.id, label=label, config=config.model_dump())
    session.add(chunk_set)
    session.flush()

    rows = []
    for doc in session.scalars(
        select(Document).where(Document.corpus_id == corpus.id).order_by(Document.id)
    ):
        spans = [tuple(s) for s in doc.sentence_spans] if doc.sentence_spans else None
        for position, (start, end) in enumerate(chunk_spans(doc.text, config, spans)):
            rows.append(
                {
                    "chunk_set_id": chunk_set.id,
                    "document_id": doc.id,
                    "position": position,
                    "char_start": start,
                    "char_end": end,
                    "text": doc.text[start:end],
                }
            )
    if rows:
        session.execute(insert(Chunk), rows)
    session.flush()
    session.execute(text("ANALYZE chunk"))  # fresh bulk load: give the planner statistics
    # The lexical index belongs to the chunking, so it is built with it.
    bm25.build_index(session, chunk_set)
    return chunk_set


def embed_chunk_set(
    session: Session, chunk_set: ChunkSet, embedder: Embedder, *, batch_size: int = 256
) -> EmbeddingRun:
    spec = embedder.spec
    existing = session.scalar(
        select(EmbeddingRun).where(
            EmbeddingRun.chunk_set_id == chunk_set.id, EmbeddingRun.model_name == spec.name
        )
    )
    if existing is not None:
        return existing

    run = EmbeddingRun(
        chunk_set_id=chunk_set.id,
        model_name=spec.name,
        dimension=spec.dimension,
        normalized=True,
        query_instruction=spec.query_instruction,
    )
    session.add(run)
    session.flush()

    chunks = session.execute(
        select(Chunk.id, Chunk.text).where(Chunk.chunk_set_id == chunk_set.id).order_by(Chunk.id)
    ).all()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = check_unit_norm(embedder.embed_passages([c.text for c in batch]))
        if vectors.shape != (len(batch), spec.dimension):
            raise ValueError(f"expected shape {(len(batch), spec.dimension)}, got {vectors.shape}")
        session.execute(
            insert(ChunkEmbedding),
            [
                {"embedding_run_id": run.id, "chunk_id": c.id, "embedding": np.asarray(v)}
                for c, v in zip(batch, vectors, strict=True)
            ],
        )
    session.flush()
    session.execute(text("ANALYZE chunk_embedding"))
    return run


def delete_corpus(session: Session, name: str) -> None:
    session.execute(delete(Corpus).where(Corpus.name == name))
