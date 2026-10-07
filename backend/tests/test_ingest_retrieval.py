"""Ingestion and retrieval end to end on a tiny corpus, with the hashing embedder."""

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from judge_check.ingest.chunking import SentenceWindow, WholeDocument
from judge_check.ingest.formats import DocumentIn, EvidenceSpan, QuestionIn
from judge_check.ingest.pipeline import build_chunk_set, embed_chunk_set, load_corpus
from judge_check.models import Chunk, ChunkEmbedding, Document, Question
from judge_check.retrieval.evaluate import gold_chunk_ids
from judge_check.retrieval.fusion import reciprocal_rank_fusion
from judge_check.retrieval.search import search
from tests.fakes import HashingEmbedder, UnnormalisedEmbedder

pytestmark = pytest.mark.db

ALPHA = "Alpha Corp was founded by Ann Lee in 1912. It makes bicycles. Its office is in Oslo."
BETA = "Ann Lee was born in Bergen. She studied engineering."
GAMMA = "Gamma is a river in Peru. It is long."

DOCS = [
    DocumentIn(id="Alpha", title="Alpha", text=ALPHA),
    DocumentIn(id="Beta", title="Beta", text=BETA),
    DocumentIn(id="Gamma", title="Gamma", text=GAMMA),
]


def span(doc: str, sentence: str) -> EvidenceSpan:
    start = {"Alpha": ALPHA, "Beta": BETA, "Gamma": GAMMA}[doc].index(sentence)
    return EvidenceSpan(document_id=doc, start=start, end=start + len(sentence))


QUESTIONS = [
    QuestionIn(
        id="q1",
        question="Where was the founder of Alpha Corp born?",
        reference_answer="Bergen",
        gold_evidence=[
            span("Alpha", "Alpha Corp was founded by Ann Lee in 1912."),
            span("Beta", "Ann Lee was born in Bergen."),
        ],
    ),
    QuestionIn(
        id="q2",
        question="Which river is in Peru?",
        reference_answer="Gamma",
        gold_evidence=[span("Gamma", "Gamma is a river in Peru.")],
    ),
]


@pytest.fixture
def corpus(db_session: Session):
    return load_corpus(db_session, "tiny", DOCS, QUESTIONS)


def test_load_corpus_round_trip(db_session: Session, corpus) -> None:
    assert db_session.scalar(select(func.count(Document.id))) == 3
    q1 = db_session.scalar(select(Question).where(Question.external_id == "q1"))
    assert q1.reference_answer == "Bergen"
    assert len(q1.evidence) == 2
    with pytest.raises(ValueError, match="already exists"):
        load_corpus(db_session, "tiny", DOCS, QUESTIONS)
    load_corpus(db_session, "tiny", DOCS, QUESTIONS, replace=True)
    assert db_session.scalar(select(func.count(Document.id))) == 3


def test_chunks_are_exact_slices_and_idempotent(db_session: Session, corpus) -> None:
    cs = build_chunk_set(db_session, corpus, SentenceWindow(sentences_per_chunk=1))
    assert build_chunk_set(db_session, corpus, SentenceWindow(sentences_per_chunk=1)).id == cs.id
    chunks = db_session.execute(
        select(Chunk, Document.text)
        .join(Document)
        .where(Chunk.chunk_set_id == cs.id)
        .order_by(Chunk.id)
    ).all()
    assert len(chunks) == 3 + 2 + 2
    for chunk, doc_text in chunks:
        assert chunk.text == doc_text[chunk.char_start : chunk.char_end]


def test_gold_chunks_follow_the_chunking(db_session: Session, corpus) -> None:
    q1 = db_session.scalar(select(Question.id).where(Question.external_id == "q1"))
    texts = lambda ids: sorted(db_session.get(Chunk, i).text for i in ids)  # noqa: E731

    fine = build_chunk_set(db_session, corpus, SentenceWindow(sentences_per_chunk=1))
    assert texts(gold_chunk_ids(db_session, fine.id)[q1]) == [
        "Alpha Corp was founded by Ann Lee in 1912.",
        "Ann Lee was born in Bergen.",
    ]
    # Same questions, coarser chunking: gold is now the two whole paragraphs.
    coarse = build_chunk_set(db_session, corpus, WholeDocument())
    assert texts(gold_chunk_ids(db_session, coarse.id)[q1]) == sorted([ALPHA, BETA])


def test_embeddings_stored_normalised_and_idempotent(db_session: Session, corpus) -> None:
    cs = build_chunk_set(db_session, corpus, SentenceWindow(sentences_per_chunk=1))
    emb = HashingEmbedder()
    run = embed_chunk_set(db_session, cs, emb)
    assert embed_chunk_set(db_session, cs, emb).id == run.id
    vectors = db_session.scalars(
        select(ChunkEmbedding.embedding).where(ChunkEmbedding.embedding_run_id == run.id)
    ).all()
    assert len(vectors) == 7
    assert np.allclose([np.linalg.norm(v) for v in vectors], 1.0, atol=1e-6)
    assert (run.dimension, run.normalized, run.query_instruction) == (64, True, "query: ")


def test_unnormalised_embedder_is_refused(db_session: Session, corpus) -> None:
    cs = build_chunk_set(db_session, corpus, SentenceWindow(sentences_per_chunk=1))
    with pytest.raises(ValueError, match="not unit-normalised"):
        embed_chunk_set(db_session, cs, UnnormalisedEmbedder())


def test_dense_similarity_equals_numpy_cosine(db_session: Session, corpus) -> None:
    cs = build_chunk_set(db_session, corpus, SentenceWindow(sentences_per_chunk=1))
    emb = HashingEmbedder()
    embed_chunk_set(db_session, cs, emb)
    hits = search(db_session, cs.id, "river in Peru", mode="dense", k=7, embedder=emb)
    q = emb.embed_queries(["river in Peru"])[0]
    for h in hits:
        p = emb.embed_passages([db_session.get(Chunk, h.chunk_id).text])[0]
        cosine = float(q @ p / (np.linalg.norm(q) * np.linalg.norm(p)))
        assert h.score == pytest.approx(cosine, abs=1e-5)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_query_instruction_is_applied_to_queries_only(db_session: Session, corpus) -> None:
    cs = build_chunk_set(db_session, corpus, SentenceWindow(sentences_per_chunk=1))
    emb = HashingEmbedder()
    embed_chunk_set(db_session, cs, emb)
    search(db_session, cs.id, "river", mode="dense", embedder=emb)
    search(db_session, cs.id, "river", mode="dense", embedder=emb, use_query_instruction=False)
    assert emb.query_calls == ["query: river", "river"]


def test_bm25_finds_lexical_match(db_session: Session, corpus) -> None:
    cs = build_chunk_set(db_session, corpus, SentenceWindow(sentences_per_chunk=1))
    hits = search(db_session, cs.id, "Bergen engineering", mode="bm25", k=5)
    top = [db_session.get(Chunk, h.chunk_id).text for h in hits]
    assert set(top) == {"Ann Lee was born in Bergen.", "She studied engineering."}
    assert all(h.score > 0 for h in hits)


def test_hybrid_is_rrf_of_dense_and_bm25(db_session: Session, corpus) -> None:
    cs = build_chunk_set(db_session, corpus, SentenceWindow(sentences_per_chunk=1))
    emb = HashingEmbedder()
    embed_chunk_set(db_session, cs, emb)
    query = "Where was Ann Lee born?"
    dense = search(db_session, cs.id, query, mode="dense", k=100, embedder=emb)
    lexical = search(db_session, cs.id, query, mode="bm25", k=100)
    hybrid = search(db_session, cs.id, query, mode="hybrid", k=100, embedder=emb)
    expected = reciprocal_rank_fusion([[h.chunk_id for h in dense], [h.chunk_id for h in lexical]])
    assert [(h.chunk_id, h.score) for h in hybrid] == expected
    for h in hybrid:
        assert h.dense_rank == next(
            (i for i, d in enumerate(dense, 1) if d.chunk_id == h.chunk_id), None
        )
        assert h.bm25_rank == next(
            (i for i, d in enumerate(lexical, 1) if d.chunk_id == h.chunk_id), None
        )
