"""judge-check command line.

judge-check ingest ../data/hotpotqa_demo --name hotpotqa_demo
judge-check search "Where was the founder of X born?" --corpus hotpotqa_demo --compare
judge-check eval-retrieval --corpus hotpotqa_demo -k 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from judge_check.db import get_engine
from judge_check.embeddings import DEFAULT_MODEL, get_embedder
from judge_check.ingest.chunking import DEFAULT_CHUNKING, ChunkingConfig, parse_chunking
from judge_check.ingest.formats import load_eval_set
from judge_check.ingest.pipeline import build_chunk_set, embed_chunk_set, load_corpus
from judge_check.models import Chunk, ChunkSet, Corpus, Document, Question
from judge_check.retrieval import bm25
from judge_check.retrieval.evaluate import (
    gold_chunk_ids,
    paired_difference,
    per_query_scores,
    score_rankings,
)
from judge_check.retrieval.search import MODES, search


def _session() -> Session:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)()


def _chunking(arg: str | None) -> ChunkingConfig:
    return parse_chunking(arg) if arg else DEFAULT_CHUNKING


def _chunk_set(session: Session, corpus_name: str, chunking: ChunkingConfig) -> ChunkSet:
    cs = session.scalar(
        select(ChunkSet)
        .join(Corpus)
        .where(Corpus.name == corpus_name, ChunkSet.label == chunking.label())
    )
    if cs is None:
        sys.exit(f"no chunk set {chunking.label()} for corpus {corpus_name!r}; run ingest first")
    return cs


def cmd_ingest(args: argparse.Namespace) -> None:
    documents, questions = load_eval_set(args.directory)
    chunking = _chunking(args.chunking)
    with _session() as session:
        t0 = time.perf_counter()
        corpus = load_corpus(session, args.name, documents, questions, replace=args.replace)
        print(f"loaded {len(documents)} documents, {len(questions)} questions")
        cs = build_chunk_set(session, corpus, chunking)
        n_chunks = len(session.scalars(select(Chunk.id).where(Chunk.chunk_set_id == cs.id)).all())
        print(f"chunked with {cs.label}: {n_chunks} chunks")
        run = embed_chunk_set(session, cs, get_embedder(args.model))
        print(f"embedded with {run.model_name} (dim {run.dimension})")
        session.commit()
        print(f"done in {time.perf_counter() - t0:.1f}s")


def _describe(session: Session, chunk_id: int, width: int = 90) -> str:
    chunk = session.get(Chunk, chunk_id)
    title = session.get(Document, chunk.document_id).title
    text = chunk.text if len(chunk.text) <= width else chunk.text[: width - 1] + "…"
    return f"[{title}] {text}"


def cmd_search(args: argparse.Namespace) -> None:
    embedder = get_embedder(args.model)
    with _session() as session:
        cs = _chunk_set(session, args.corpus, _chunking(args.chunking))
        modes = MODES if args.compare else (args.mode,)
        for mode in modes:
            hits = search(
                session,
                cs.id,
                args.query,
                mode=mode,
                k=args.k,
                embedder=embedder,
                use_query_instruction=not args.no_query_instruction,
            )
            print(f"\n== {mode} ==")
            print(f"{'#':>2} {'score':>8} {'dense#':>6} {'bm25#':>6}  chunk")
            for i, h in enumerate(hits, 1):
                dr = h.dense_rank or "-"
                br = h.bm25_rank or "-"
                print(
                    f"{i:>2} {h.score:>8.4f} {dr!s:>6} {br!s:>6}  {_describe(session, h.chunk_id)}"
                )


def cmd_eval_retrieval(args: argparse.Namespace) -> None:
    # Each run: (label, mode, use_query_instruction). The first run is the baseline that
    # the others are compared against, question by question (paired differences).
    runs = [(m, m, not args.no_query_instruction) for m in args.modes.split(",")]
    if args.instruction_ablation:
        runs.append(("dense-noinstr", "dense", False))
    # BM25-only runs don't need (or download) the embedding model.
    needs_model = any(mode != "bm25" for _, mode, _ in runs)
    embedder = get_embedder(args.model) if needs_model else None
    with _session() as session:
        cs = _chunk_set(session, args.corpus, _chunking(args.chunking))
        gold = gold_chunk_ids(session, cs.id)
        questions = session.execute(
            select(Question.id, Question.text)
            .where(Question.corpus_id == cs.corpus_id)
            .order_by(Question.id)
        ).all()
        if args.limit:
            questions = questions[: args.limit]
        print(f"{len(questions)} questions, chunk set {cs.label}, k={args.k}")
        print(f"{'run':<14} {'recall@k':>9} {'complete@k':>11} {'MRR@k':>7} {'s/query':>8}")
        results, per_run = {}, {}
        for label, mode, use_instr in runs:
            t0 = time.perf_counter()
            rankings = {
                qid: [
                    h.chunk_id
                    for h in search(
                        session,
                        cs.id,
                        text,
                        mode=mode,  # type: ignore[arg-type]
                        k=args.k,
                        embedder=embedder,
                        use_query_instruction=use_instr,
                        bm25_k1=args.k1,
                        bm25_b=args.b,
                    )
                ]
                for qid, text in questions
            }
            dt = (time.perf_counter() - t0) / len(questions)
            m = score_rankings(rankings, gold, args.k)
            per_run[label] = per_query_scores(rankings, gold, args.k)
            results[label] = m.__dict__
            print(f"{label:<14} {m.recall:>9.3f} {m.complete:>11.3f} {m.mrr:>7.3f} {dt:>8.3f}")

        base = runs[0][0]
        if len(runs) > 1:
            print(f"\npaired differences vs {base} (same {len(questions)} questions; z = diff/SE)")
            qids = [qid for qid, _ in questions]
            for label, _, _ in runs[1:]:
                cells = []
                for metric in ("recall", "complete", "rr"):
                    a = [getattr(per_run[base][q], metric) for q in qids]
                    b = [getattr(per_run[label][q], metric) for q in qids]
                    d, se = paired_difference(a, b)
                    z = d / se if se else float("nan")
                    cells.append(f"{metric:>8} {d:+.3f} (SE {se:.3f}, z {z:+.1f})")
                print(f"  {label:<14}" + "  ".join(cells))
        if args.json:
            print(json.dumps(results, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="judge-check")
    sub = p.add_subparsers(required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL)
    common.add_argument("--chunking", help='JSON chunking config, e.g. \'{"strategy":"document"}\'')

    ing = sub.add_parser("ingest", parents=[common], help="load, chunk and embed an eval set")
    ing.add_argument("directory", type=Path, help="directory with corpus.jsonl, questions.jsonl")
    ing.add_argument("--name", required=True)
    ing.add_argument("--replace", action="store_true")
    ing.set_defaults(func=cmd_ingest)

    s = sub.add_parser("search", parents=[common], help="run one query")
    s.add_argument("query")
    s.add_argument("--corpus", required=True)
    s.add_argument("--mode", choices=MODES, default="hybrid")
    s.add_argument("--compare", action="store_true", help="show dense, bm25 and hybrid")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--no-query-instruction", action="store_true")
    s.set_defaults(func=cmd_search)

    e = sub.add_parser("eval-retrieval", parents=[common], help="recall/MRR against gold")
    e.add_argument("--corpus", required=True)
    e.add_argument("--modes", default=",".join(MODES))
    e.add_argument("-k", type=int, default=10)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--no-query-instruction", action="store_true")
    e.add_argument("--k1", type=float, default=bm25.K1, help="BM25 tf saturation")
    e.add_argument("--b", type=float, default=bm25.B, help="BM25 length normalisation")
    e.add_argument(
        "--instruction-ablation",
        action="store_true",
        help="also run dense without the query instruction, paired against the first run",
    )
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_eval_retrieval)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
