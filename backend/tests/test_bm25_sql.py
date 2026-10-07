"""The SQL BM25 must agree with rank_bm25.BM25Okapi, score for score.

rank_bm25 is an independent, widely used reference implementation. If our SQL agrees with
it on every (query, chunk) pair of a real corpus, and on the edge cases where BM25 variants
differ, then the SQL implements the formula in judge_check/retrieval/bm25.py.
"""

import numpy as np
import pytest
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from judge_check.datasets.demo import load_demo
from judge_check.ingest.chunking import DEFAULT_CHUNKING, WholeDocument
from judge_check.ingest.formats import DocumentIn, EvidenceSpan, QuestionIn
from judge_check.ingest.pipeline import build_chunk_set, load_corpus
from judge_check.models import Bm25Stats, Bm25Term, Chunk
from judge_check.retrieval import bm25
from judge_check.retrieval.tokenize import tokenize

pytestmark = pytest.mark.db

TOL = 1e-9


def compare(session: Session, chunk_set_id: int, queries: list[str]) -> int:
    """Assert SQL == rank_bm25 for every chunk and query. Returns pairs compared."""
    chunks = session.execute(
        select(Chunk.id, Chunk.text).where(Chunk.chunk_set_id == chunk_set_id).order_by(Chunk.id)
    ).all()
    ids = [c.id for c in chunks]
    reference = BM25Okapi([tokenize(c.text) for c in chunks], k1=bm25.K1, b=bm25.B,
                          epsilon=bm25.EPSILON)  # fmt: skip
    for query in queries:
        expected = reference.get_scores(tokenize(query))
        sql = bm25.score_all(session, chunk_set_id, query)
        got = np.array([sql.get(i, 0.0) for i in ids])
        np.testing.assert_allclose(got, expected, rtol=TOL, atol=TOL, err_msg=query)
    return len(ids) * len(queries)


# --- edge cases where BM25 variants disagree ----------------------------------------------

EDGE_DOCS = [
    "The river flows. The river is long and the river is wide.",  # repeated term, long
    "The city is old.",
    "The cat sat.",
    "The dog and the cat.",
    "!!! ???",  # zero tokens: counts in N and avgdl, can never match
]


@pytest.fixture
def edge_chunk_set(db_session: Session):
    docs = [DocumentIn(id=f"d{i}", text=t) for i, t in enumerate(EDGE_DOCS)]
    q = QuestionIn(
        id="q",
        question="x?",
        reference_answer="x",
        gold_evidence=[EvidenceSpan(document_id="d0", start=0, end=3)],
    )
    corpus = load_corpus(db_session, "edge", docs, [q])
    return build_chunk_set(db_session, corpus, WholeDocument())


@pytest.mark.parametrize(
    "query",
    [
        "river",
        "river river",  # duplicate query terms count twice (no query-tf saturation)
        "the",  # in 4/5 chunks: raw IDF < 0, so the epsilon floor applies
        "the cat",
        "cat dog city",
        "submarine",  # unknown term: contributes nothing
        "submarine cat",
        "!!!",  # tokenizes to nothing
    ],
)
def test_edge_cases_match_rank_bm25(db_session: Session, edge_chunk_set, query: str) -> None:
    compare(db_session, edge_chunk_set.id, [query])


def test_negative_idf_is_floored(db_session: Session, edge_chunk_set) -> None:
    stats = db_session.get(Bm25Stats, edge_chunk_set.id)
    the = db_session.get(Bm25Term, (edge_chunk_set.id, "the"))
    raw = np.log((stats.n_docs - the.df + 0.5) / (the.df + 0.5))
    assert raw < 0
    assert the.idf == pytest.approx(stats.epsilon * stats.avg_idf)
    assert stats.n_docs == 5  # the zero-token chunk still counts


def test_explain_sums_to_score(db_session: Session, edge_chunk_set) -> None:
    query = "the river river cat submarine"
    scores = bm25.score_all(db_session, edge_chunk_set.id, query)
    for chunk_id, score in scores.items():
        parts = bm25.explain(db_session, edge_chunk_set.id, query, chunk_id)
        assert [p.term for p in parts] == tokenize(query)
        assert sum(p.contribution for p in parts) == pytest.approx(score, rel=TOL)


def test_top_k_is_ordered_and_positive(db_session: Session, edge_chunk_set) -> None:
    hits = bm25.search(db_session, edge_chunk_set.id, "the cat", k=10)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)
    assert bm25.search(db_session, edge_chunk_set.id, "!!!", k=10) == []


# --- the real test: the whole demo corpus -------------------------------------------------


def test_sql_bm25_matches_rank_bm25_on_demo_corpus(db_session: Session) -> None:
    documents, questions = load_demo()
    corpus = load_corpus(db_session, "hotpotqa_demo", documents, questions)
    cs = build_chunk_set(db_session, corpus, DEFAULT_CHUNKING)
    # Every question, and every reference answer as a second, shorter query.
    queries = [q.question for q in questions] + [q.reference_answer for q in questions]
    pairs = compare(db_session, cs.id, queries)
    print(f"\nSQL BM25 == rank_bm25 on {len(queries)} queries x chunks = {pairs:,} scores")
