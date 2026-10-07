"""Dense, lexical (BM25) and hybrid retrieval over one chunk set.

dense   pgvector, inner product on unit vectors (= cosine), exact scan (D-018, D-019)
bm25    lexical scoring over the shared tokenizer (D-020, D-021)
hybrid  Reciprocal Rank Fusion of the two candidate lists (D-022)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from judge_check.embeddings import Embedder
from judge_check.models import EmbeddingRun
from judge_check.retrieval import bm25
from judge_check.retrieval.fusion import reciprocal_rank_fusion

Mode = Literal["dense", "bm25", "hybrid"]
MODES: tuple[Mode, ...] = ("dense", "bm25", "hybrid")

# How many candidates each retriever contributes to fusion. Larger than k so a chunk ranked
# 30th by one retriever and 2nd by the other can still win.
CANDIDATES = 100


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    score: float  # cosine similarity, BM25 score, or RRF score, depending on mode
    dense_rank: int | None = None  # 1-based rank in the dense list (None: not in top N)
    bm25_rank: int | None = None


def dense_search(
    session: Session, run_id: int, query_vector: np.ndarray, k: int
) -> list[tuple[int, float]]:
    """Top-k chunks by cosine similarity.

    `<#>` is pgvector's *negative* inner product (negated so that ascending order means
    most similar first, like the other distance operators). Our vectors are unit length,
    so -(a <#> b) = a·b = cos(a, b). See D-018 for why not `<=>`.
    """
    rows = session.execute(
        text(
            """
            SELECT chunk_id, -(embedding <#> CAST(:q AS vector)) AS similarity
            FROM chunk_embedding
            WHERE embedding_run_id = :run
            ORDER BY embedding <#> CAST(:q AS vector)
            LIMIT :k
            """
        ),
        {"q": str(query_vector.tolist()), "run": run_id, "k": k},
    ).all()
    return [(r.chunk_id, float(r.similarity)) for r in rows]


def get_run(session: Session, chunk_set_id: int, model_name: str) -> EmbeddingRun:
    run = session.scalar(
        select(EmbeddingRun).where(
            EmbeddingRun.chunk_set_id == chunk_set_id, EmbeddingRun.model_name == model_name
        )
    )
    if run is None:
        raise LookupError(f"chunk set {chunk_set_id} has no embeddings for {model_name!r}")
    return run


def search(
    session: Session,
    chunk_set_id: int,
    query: str,
    *,
    mode: Mode = "hybrid",
    k: int = 10,
    embedder: Embedder | None = None,
    use_query_instruction: bool = True,
    candidates: int = CANDIDATES,
) -> list[Hit]:
    dense: list[tuple[int, float]] = []
    lexical: list[tuple[int, float]] = []
    if mode in ("dense", "hybrid"):
        if embedder is None:
            raise ValueError(f"mode {mode!r} needs an embedder")
        run = get_run(session, chunk_set_id, embedder.spec.name)
        qvec = embedder.embed_queries([query], use_instruction=use_query_instruction)[0]
        dense = dense_search(session, run.id, qvec, candidates if mode == "hybrid" else k)
    if mode in ("bm25", "hybrid"):
        lexical = bm25.search(session, chunk_set_id, query, candidates if mode == "hybrid" else k)

    dense_rank = {cid: r for r, (cid, _) in enumerate(dense, 1)}
    bm25_rank = {cid: r for r, (cid, _) in enumerate(lexical, 1)}
    if mode == "dense":
        scored = dense
    elif mode == "bm25":
        scored = lexical
    else:
        scored = reciprocal_rank_fusion([[c for c, _ in dense], [c for c, _ in lexical]])
    return [Hit(cid, score, dense_rank.get(cid), bm25_rank.get(cid)) for cid, score in scored[:k]]
