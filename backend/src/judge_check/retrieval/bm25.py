"""BM25 (Okapi) implemented in SQL over term-statistics tables (D-021).

For a query Q = q1..qn (tokens, duplicates kept) and a chunk D:

    score(D, Q) = Σ_i  IDF(q_i) · f(q_i, D) · (k1 + 1)
                        ─────────────────────────────────────────────
                        f(q_i, D) + k1 · (1 − b + b · |D| / avgdl)

    IDF(t) = ln( (N − n(t) + 0.5) / (n(t) + 0.5) )      (Robertson–Spärck Jones)
             floored: if IDF(t) < 0, use ε · mean_t IDF(t)

    f(t, D)  occurrences of t in D            → bm25_posting.tf
    |D|      tokens in D                      → bm25_doc.length
    avgdl    mean |D| over the chunk set      → bm25_stats.avgdl
    N        number of chunks                 → bm25_stats.n_docs
    n(t)     chunks containing t              → bm25_term.df

This is exactly what rank_bm25.BM25Okapi computes, including its negative-IDF floor. That
is deliberate: tests/test_bm25_sql.py checks the two agree on the demo corpus.

Index building splits the work: Python tokenizes (one tokenizer for everything, D-020), SQL
aggregates the statistics and computes IDF, and SQL does all query-time scoring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sqlalchemy import insert, select, text
from sqlalchemy.orm import Session

from judge_check.models import Bm25Doc, Bm25Posting, Bm25Stats, Chunk, ChunkSet
from judge_check.retrieval.tokenize import tokenize

K1 = 1.5  # term-frequency saturation
B = 0.75  # length normalisation strength
EPSILON = 0.25  # negative-IDF floor, as a fraction of the mean IDF
TOKENIZER = "lower+unicode_word_v1"


def build_index(session: Session, chunk_set: ChunkSet, epsilon: float = EPSILON) -> Bm25Stats:
    """Tokenize every chunk once and store lengths, postings and term statistics."""
    existing = session.get(Bm25Stats, chunk_set.id)
    if existing is not None:
        return existing
    cs = chunk_set.id

    docs, postings = [], []
    for chunk_id, chunk_text in session.execute(
        select(Chunk.id, Chunk.text).where(Chunk.chunk_set_id == cs).order_by(Chunk.id)
    ):
        tokens = tokenize(chunk_text)
        docs.append({"chunk_id": chunk_id, "chunk_set_id": cs, "length": len(tokens)})
        postings.extend(
            {"chunk_set_id": cs, "term": term, "chunk_id": chunk_id, "tf": tf}
            for term, tf in Counter(tokens).items()
        )
    if not docs:
        raise ValueError(f"chunk set {cs} has no chunks")
    session.execute(insert(Bm25Doc), docs)
    if postings:
        session.execute(insert(Bm25Posting), postings)

    params = {"cs": cs, "eps": epsilon}
    # Document frequency and raw IDF for every term.
    session.execute(
        text(
            """
            INSERT INTO bm25_term (chunk_set_id, term, df, idf)
            SELECT p.chunk_set_id, p.term, count(*) AS df,
                   ln((n.n_docs - count(*) + 0.5) / (count(*) + 0.5)) AS idf
            FROM bm25_posting p
            CROSS JOIN (SELECT count(*) AS n_docs FROM bm25_doc WHERE chunk_set_id = :cs) n
            WHERE p.chunk_set_id = :cs
            GROUP BY p.chunk_set_id, p.term, n.n_docs
            """
        ),
        params,
    )
    # Corpus statistics. avg_idf is over the *raw* IDFs, before flooring.
    session.execute(
        text(
            """
            INSERT INTO bm25_stats (chunk_set_id, tokenizer, n_docs, avgdl, avg_idf, epsilon)
            SELECT :cs, :tok, d.n_docs, d.avgdl, coalesce(t.avg_idf, 0), :eps
            FROM (SELECT count(*) AS n_docs, avg(length)::float AS avgdl
                  FROM bm25_doc WHERE chunk_set_id = :cs) d
            CROSS JOIN (SELECT avg(idf) AS avg_idf FROM bm25_term WHERE chunk_set_id = :cs) t
            """
        ),
        {**params, "tok": TOKENIZER},
    )
    # The floor: a term in more than half the chunks has negative raw IDF. Let it count a
    # little instead of penalising chunks that contain it.
    session.execute(
        text(
            """
            UPDATE bm25_term t SET idf = s.epsilon * s.avg_idf
            FROM bm25_stats s
            WHERE t.chunk_set_id = :cs AND s.chunk_set_id = :cs AND t.idf < 0
            """
        ),
        params,
    )
    session.flush()
    return session.get(Bm25Stats, cs)


_SCORE_SQL = """
    WITH q(term) AS (SELECT unnest(CAST(:terms AS text[])))  -- duplicates kept on purpose
    SELECT p.chunk_id,
           SUM(t.idf * p.tf * (:k1 + 1)
               / (p.tf + :k1 * (1 - :b + :b * d.length / NULLIF(s.avgdl, 0)))) AS score
    FROM q
    JOIN bm25_term    t ON t.chunk_set_id = :cs AND t.term = q.term
    JOIN bm25_posting p ON p.chunk_set_id = :cs AND p.term = q.term
    JOIN bm25_doc     d ON d.chunk_id = p.chunk_id
    JOIN bm25_stats   s ON s.chunk_set_id = :cs
    GROUP BY p.chunk_id
"""


def search(
    session: Session, chunk_set_id: int, query: str, k: int, *, k1: float = K1, b: float = B
) -> list[tuple[int, float]]:
    """Top-k chunks by BM25, best first. Chunks sharing no term with the query are not
    returned (their score is 0). Ties break on chunk id."""
    terms = tokenize(query)
    if not terms:
        return []
    if session.get(Bm25Stats, chunk_set_id) is None:
        raise LookupError(f"chunk set {chunk_set_id} has no BM25 index; run build_index")
    rows = session.execute(
        text(_SCORE_SQL + " ORDER BY score DESC, p.chunk_id LIMIT :lim"),
        {"terms": terms, "cs": chunk_set_id, "k1": k1, "b": b, "lim": k},
    ).all()
    return [(r.chunk_id, float(r.score)) for r in rows if r.score > 0]


def score_all(
    session: Session, chunk_set_id: int, query: str, *, k1: float = K1, b: float = B
) -> dict[int, float]:
    """Every chunk's score that shares at least one term with the query (for tests)."""
    terms = tokenize(query)
    if not terms:
        return {}
    rows = session.execute(
        text(_SCORE_SQL), {"terms": terms, "cs": chunk_set_id, "k1": k1, "b": b}
    ).all()
    return {r.chunk_id: float(r.score) for r in rows}


@dataclass(frozen=True)
class TermContribution:
    term: str
    df: int
    idf: float
    tf: int
    length: int
    avgdl: float
    contribution: float


def explain(
    session: Session, chunk_set_id: int, query: str, chunk_id: int, *, k1: float = K1, b: float = B
) -> list[TermContribution]:
    """Per-query-term breakdown of one chunk's score: what each part of the formula did."""
    rows = session.execute(
        text(
            """
            WITH q(term) AS (SELECT unnest(CAST(:terms AS text[])))
            SELECT q.term, t.df, t.idf, coalesce(p.tf, 0) AS tf, d.length, s.avgdl,
                   coalesce(t.idf * p.tf * (:k1 + 1)
                            / (p.tf + :k1 * (1 - :b + :b * d.length / s.avgdl)), 0) AS c
            FROM q
            JOIN bm25_stats s ON s.chunk_set_id = :cs
            JOIN bm25_doc d ON d.chunk_id = :chunk
            LEFT JOIN bm25_term t ON t.chunk_set_id = :cs AND t.term = q.term
            LEFT JOIN bm25_posting p
                   ON p.chunk_set_id = :cs AND p.term = q.term AND p.chunk_id = :chunk
            """
        ),
        {"terms": tokenize(query), "cs": chunk_set_id, "chunk": chunk_id, "k1": k1, "b": b},
    ).all()
    return [
        TermContribution(r.term, r.df or 0, r.idf or 0.0, r.tf, r.length, r.avgdl, r.c)
        for r in rows
    ]
