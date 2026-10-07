"""Reproduce DECISIONS.md D-025: BM25's b vs chunk length on the demo corpus.

    cd backend && uv run python scripts/bm25_b_sweep.py

BM25 only (no embedding model). Loads the demo set into a throwaway transaction on the
configured database and rolls it back at the end. Prints, per chunking: length
statistics, recall@5 and MRR (over the top 100) for b in {0, .25, .5, .75, 1}, paired
differences with SEs, and the mean length of the top-5 chunks (length bias).
"""

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from judge_check.datasets.demo import load_demo
from judge_check.db import get_engine
from judge_check.ingest.chunking import DEFAULT_CHUNKING, WholeDocument
from judge_check.ingest.pipeline import build_chunk_set, load_corpus
from judge_check.models import Bm25Doc, Bm25Stats, Question
from judge_check.retrieval import bm25
from judge_check.retrieval.evaluate import gold_chunk_ids, paired_difference

BS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    session = sessionmaker(bind=get_engine())()
    try:
        docs, qs = load_demo()
        corpus = load_corpus(session, "__bm25_b_sweep__", docs, qs, replace=True)
        questions = session.execute(
            select(Question.id, Question.text).where(Question.corpus_id == corpus.id)
        ).all()
        for cfg in [DEFAULT_CHUNKING, WholeDocument()]:
            cs = build_chunk_set(session, corpus, cfg)
            stats = session.get(Bm25Stats, cs.id)
            length = dict(
                session.execute(
                    select(Bm25Doc.chunk_id, Bm25Doc.length).where(Bm25Doc.chunk_set_id == cs.id)
                ).all()
            )
            L = np.array(list(length.values()))
            p10, p90 = np.percentile(L, [10, 90])
            print(f"\n{cs.label}: N={stats.n_docs} avgdl={stats.avgdl:.1f} "
                  f"p10/p90 of |D|/avgdl={p10 / stats.avgdl:.2f}/{p90 / stats.avgdl:.2f} "
                  f"CV={L.std() / L.mean():.2f} max={L.max()}")  # fmt: skip
            gold = gold_chunk_ids(session, cs.id)
            rr, rec = {}, {}
            print(f"  {'b':>5} {'recall@5':>9} {'MRR':>7} {'top5 len/avgdl':>15}")
            for b in BS:
                rr[b], rec[b], lens = [], [], []
                for qid, text in questions:
                    ranked = [c for c, _ in bm25.search(session, cs.id, text, 100, b=b)]
                    g = gold[qid]
                    rr[b].append(next((1 / i for i, c in enumerate(ranked, 1) if c in g), 0.0))
                    rec[b].append(len(g & set(ranked[:5])) / len(g))
                    lens += [length[c] for c in ranked[:5]]
                print(f"  {b:>5} {np.mean(rec[b]):>9.3f} {np.mean(rr[b]):>7.3f} "
                      f"{np.mean(lens) / stats.avgdl:>15.2f}")  # fmt: skip
            for lo, hi in [(0.0, 0.75), (0.75, 1.0)]:
                d, se = paired_difference(rr[lo], rr[hi])
                print(f"  paired MRR b={hi} - b={lo}: {d:+.3f} (SE {se:.3f}, z {d / se:+.1f})")
    finally:
        session.rollback()


if __name__ == "__main__":
    main()
