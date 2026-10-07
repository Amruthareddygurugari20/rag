"""Gold chunks per question, and retrieval metrics against them.

A chunk is gold for a question if it overlaps one of the question's gold evidence spans
by at least one character. Spans are half-open [start, end), so two intervals overlap iff
each starts before the other ends (D-014).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session


def gold_chunk_ids(session: Session, chunk_set_id: int) -> dict[int, set[int]]:
    """question_id -> ids of the chunks in this chunk set that overlap its gold evidence."""
    rows = session.execute(
        text(
            """
            SELECT DISTINCT e.question_id, c.id AS chunk_id
            FROM question_evidence e
            JOIN chunk c
              ON c.document_id = e.document_id
             AND c.char_start < e.char_end
             AND e.char_start < c.char_end
            WHERE c.chunk_set_id = :cs
            """
        ),
        {"cs": chunk_set_id},
    ).all()
    gold: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        gold[r.question_id].add(r.chunk_id)
    return dict(gold)


@dataclass(frozen=True)
class RetrievalMetrics:
    k: int
    n_questions: int
    recall: float  # mean fraction of a question's gold chunks found in the top k
    complete: float  # fraction of questions with *all* gold chunks in the top k (multi-hop)
    mrr: float  # mean reciprocal rank of the first gold chunk (0 if none in the ranking)


@dataclass(frozen=True)
class QueryScores:
    recall: float
    complete: float
    rr: float  # reciprocal rank of the first gold chunk, 0 if none retrieved


def per_query_scores(
    rankings: dict[int, list[int]], gold: dict[int, set[int]], k: int
) -> dict[int, QueryScores]:
    """`rankings` maps question id -> retrieved chunk ids, best first."""
    out = {}
    for qid, ranked in rankings.items():
        g = gold.get(qid)
        if not g:
            raise ValueError(f"question {qid} has no gold chunks in this chunk set")
        top = set(ranked[:k])
        rr = next((1.0 / i for i, c in enumerate(ranked, 1) if c in g), 0.0)
        out[qid] = QueryScores(len(g & top) / len(g), float(g <= top), rr)
    return out


def score_rankings(
    rankings: dict[int, list[int]], gold: dict[int, set[int]], k: int
) -> RetrievalMetrics:
    if not rankings:
        raise ValueError("no rankings to score")
    per_q = per_query_scores(rankings, gold, k).values()
    n = len(per_q)
    return RetrievalMetrics(
        k,
        n,
        sum(q.recall for q in per_q) / n,
        sum(q.complete for q in per_q) / n,
        sum(q.rr for q in per_q) / n,
    )


def paired_difference(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mean of (b - a) over paired per-question values, and its standard error.

    Two retrieval configurations run on the *same* questions. Pairing removes the
    question-difficulty variance both share, so the SE of the difference is usually much
    smaller than either configuration's own SE. That is why comparisons use this and not
    two independent confidence intervals.
    """
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("need two equally long lists of at least 2 values")
    d = [y - x for x, y in zip(a, b, strict=True)]
    mean = sum(d) / len(d)
    var = sum((x - mean) ** 2 for x in d) / (len(d) - 1)
    return mean, (var / len(d)) ** 0.5
