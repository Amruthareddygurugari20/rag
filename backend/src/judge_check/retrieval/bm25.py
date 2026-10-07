"""BM25 lexical retrieval.

Current implementation: rank_bm25's BM25Okapi, built in memory from the chunk set and
cached per process. This is a stepping stone to get hybrid fusion working end to end. It is
replaced by a SQL implementation over a term-statistics table (D-021).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from judge_check.models import Chunk
from judge_check.retrieval.tokenize import tokenize

K1 = 1.5
B = 0.75


@lru_cache(maxsize=8)
def _index(chunk_set_id: int, chunk_ids: tuple[int, ...], texts: tuple[str, ...]) -> BM25Okapi:
    return BM25Okapi([tokenize(t) for t in texts], k1=K1, b=B)


def search(session: Session, chunk_set_id: int, query: str, k: int) -> list[tuple[int, float]]:
    rows = session.execute(
        select(Chunk.id, Chunk.text).where(Chunk.chunk_set_id == chunk_set_id).order_by(Chunk.id)
    ).all()
    ids = tuple(r.id for r in rows)
    index = _index(chunk_set_id, ids, tuple(r.text for r in rows))
    scores = index.get_scores(tokenize(query))
    order = np.argsort(-scores, kind="stable")[:k]
    # A chunk sharing no term with the query scores 0: it was not retrieved, so drop it.
    return [(ids[i], float(scores[i])) for i in order if scores[i] > 0]
