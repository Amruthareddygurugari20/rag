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


def score_rankings(
    rankings: dict[int, list[int]], gold: dict[int, set[int]], k: int
) -> RetrievalMetrics:
    """`rankings` maps question id -> retrieved chunk ids, best first."""
    if not rankings:
        raise ValueError("no rankings to score")
    recall = complete = rr = 0.0
    for qid, ranked in rankings.items():
        g = gold.get(qid)
        if not g:
            raise ValueError(f"question {qid} has no gold chunks in this chunk set")
        top = set(ranked[:k])
        recall += len(g & top) / len(g)
        complete += float(g <= top)
        rr += next((1.0 / i for i, c in enumerate(ranked, 1) if c in g), 0.0)
    n = len(rankings)
    return RetrievalMetrics(k, n, recall / n, complete / n, rr / n)
