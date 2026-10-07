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

from judge_check.config import get_settings
from judge_check.db import get_engine
from judge_check.embeddings import DEFAULT_MODEL, get_embedder
from judge_check.ingest.chunking import DEFAULT_CHUNKING, ChunkingConfig, parse_chunking
from judge_check.ingest.formats import load_eval_set
from judge_check.ingest.pipeline import build_chunk_set, embed_chunk_set, load_corpus
from judge_check.models import Chunk, ChunkSet, Corpus, Document, Question
from judge_check.retrieval import bm25, sweep
from judge_check.retrieval.evaluate import (
    gold_chunk_ids,
    paired_difference,
    per_query_scores,
    score_rankings,
)
from judge_check.retrieval.search import MODES, search
from judge_check.retrieval.sweep import Candidates


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


def cmd_fusion_sweep(args: argparse.Namespace) -> None:
    """Dense/BM25 fusion weight sweep, with a cross-fitted (held-out) result (D-028)."""
    embedder = get_embedder(args.model)
    with _session() as session:
        cs = _chunk_set(session, args.corpus, _chunking(args.chunking))
        gold = gold_chunk_ids(session, cs.id)
        questions = session.execute(
            select(Question.id, Question.text)
            .where(Question.corpus_id == cs.corpus_id)
            .order_by(Question.id)
        ).all()
        cands = {}
        for qid, text in questions:
            dense = search(session, cs.id, text, mode="dense", k=args.candidates, embedder=embedder)
            lexical = search(session, cs.id, text, mode="bm25", k=args.candidates)
            cands[qid] = Candidates(
                dense=[(h.chunk_id, h.score) for h in dense],
                bm25=[(h.chunk_id, h.score) for h in lexical],
            )
    qids = [q for q, _ in questions]
    k = args.k
    dense_only = sweep.scores_at(cands, gold, "rrf", 1.0, k)

    print(f"{len(qids)} questions, chunk set {cs.label}, k={k}, {args.candidates} candidates")
    print("alpha = weight on dense (BM25 gets 1 - alpha); 1.0 = dense only, 0.5 = equal RRF")
    print("Descriptive curve over all questions. Do NOT pick the best row from it (see below).")
    for method in ("rrf", "linear"):
        print(f"\n{method:>6}  alpha  recall@k  complete@k  MRR@k   dMRR vs dense (SE)")
        for a in sweep.ALPHAS:
            sc = sweep.scores_at(cands, gold, method, a, k)
            d, se = paired_difference([dense_only[q].rr for q in qids], [sc[q].rr for q in qids])
            print(
                f"{'':>6}  {a:>5.1f}  {sweep.mean(sc, 'recall', qids):>8.3f}  "
                f"{sweep.mean(sc, 'complete', qids):>10.3f}  {sweep.mean(sc, 'rr', qids):>5.3f}"
                f"   {d:+.3f} ({se:.3f})"
            )

    print("\nCross-fitted (alpha chosen on one half by MRR@k, scored on the other half):")
    for method in ("rrf", "linear"):
        cf = sweep.crossfit(cands, gold, method, k)
        cells = []
        for metric in ("recall", "complete", "rr"):
            d, se = paired_difference(
                [getattr(dense_only[q], metric) for q in qids],
                [getattr(cf.held_out[q], metric) for q in qids],
            )
            held = sweep.mean(cf.held_out, metric, qids)
            z = d / se if se else 0.0
            cells.append(f"{metric} {held:.3f} (vs dense {d:+.3f}, SE {se:.3f}, z {z:+.1f})")
        print(
            f"  {method:>6}: alpha chosen {cf.alpha_chosen_on_a} / {cf.alpha_chosen_on_b}  "
            + "  ".join(cells)
        )

    print(
        f"\nRepeated cross-fitting: {args.repeats} random 2-fold partitions (seed {args.seed})."
        "\nThe spread shows how stable the alpha *choice* is; it is not a population CI"
        " (every partition reuses the same questions). Paired SE above is the sampling error."
    )
    for method in ("rrf", "linear"):
        rc = sweep.repeated_crossfit(cands, gold, method, k, args.repeats, args.seed)
        d = rc.delta_vs_dense
        share_better = sum(x > 0 for x in d) / len(d)
        alpha_counts = {a: rc.alphas_chosen.count(a) for a in sorted(set(rc.alphas_chosen))}
        print(
            f"  {method:>6}: dMRR vs dense median {sweep.percentile(d, 50):+.3f}, "
            f"2.5-97.5% [{sweep.percentile(d, 2.5):+.3f}, {sweep.percentile(d, 97.5):+.3f}], "
            f"partitions better than dense {share_better:.0%}"
        )
        print(f"          alpha chosen (count over {len(rc.alphas_chosen)} folds): {alpha_counts}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate grounded answers for a corpus's questions and summarise them.

    The summary's 'reference in answer' and 'cites gold' columns are descriptive
    heuristics. They are NOT correctness labels and are never used as such (D-000, D-032).
    """
    from judge_check.datasets.hotpotqa import contains_answer
    from judge_check.generation import generate_answer
    from judge_check.llm.factory import make_client

    client = make_client(args.provider, args.model or get_settings().generation_model)
    embedder = get_embedder(args.embedding_model)
    with _session() as session:
        cs = _chunk_set(session, args.corpus, _chunking(args.chunking))
        gold = gold_chunk_ids(session, cs.id)
        questions = session.execute(
            select(Question.id, Question.reference_answer)
            .where(Question.corpus_id == cs.corpus_id)
            .order_by(Question.id)
        ).all()[: args.limit or None]
        n_ok = n_contains = n_cites_gold = n_unanswerable = 0
        versions = set()
        for i, (qid, ref) in enumerate(questions, 1):
            row = generate_answer(
                session, client, qid, cs.id, embedder, mode=args.mode, k=args.k,
                temperature=args.temperature, seed=args.seed, max_tokens=args.max_tokens,
            )  # fmt: skip
            session.commit()
            versions.add(row.model_version)
            status = row.parse_error or ("unanswerable" if not row.answerable else "ok")
            if row.parse_error is None:
                n_ok += 1
                n_unanswerable += not row.answerable
                n_contains += contains_answer(ref, row.answer)
                n_cites_gold += bool(set(row.cited_chunk_ids) & gold.get(qid, set()))
            if args.verbose:
                print(f"[{i}/{len(questions)}] {status:<14} {row.latency_ms:>6} ms  "
                      f"{(row.answer or row.output_text)[:90]!r}")  # fmt: skip
        n = len(questions)
        print(f"\n{n} answers from {', '.join(sorted(versions))}")
        print(f"  parsed OK              {n_ok}/{n}  (failures are stored with their reason)")
        print(f"  declared unanswerable  {n_unanswerable}/{n_ok}")
        print("  Heuristics only, NOT correctness labels:")
        print(f"    reference string in answer   {n_contains}/{n_ok}")
        print(f"    cites >= 1 gold chunk        {n_cites_gold}/{n_ok}")


def cmd_prompts(args: argparse.Namespace) -> None:
    from judge_check.prompts import add_new_prompts_to_lock, all_prompts, lock_problems

    if args.action == "lock":
        added = add_new_prompts_to_lock()
        print("locked: " + (", ".join(added) if added else "nothing new"))
    problems = lock_problems()
    for key, p in sorted(all_prompts().items()):
        print(f"{key:<28} {p.kind:<11} {p.sha256[:12]}")
    if problems:
        print("\n".join(["", "PROBLEMS:", *problems]))
        sys.exit(1)


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

    f = sub.add_parser(
        "fusion-sweep", parents=[common], help="dense/BM25 fusion weight sweep, cross-fitted"
    )
    f.add_argument("--corpus", required=True)
    f.add_argument("-k", type=int, default=5)
    f.add_argument("--candidates", type=int, default=100)
    f.add_argument("--repeats", type=int, default=200, help="random 2-fold partitions")
    f.add_argument("--seed", type=int, default=20261007)
    f.set_defaults(func=cmd_fusion_sweep)

    g = sub.add_parser("generate", help="grounded answers with citations (stage 2)")
    g.add_argument("--corpus", required=True)
    g.add_argument("--provider", choices=["ollama", "azure_openai"], default="ollama")
    g.add_argument("--model", help="Ollama tag or Azure deployment (default from settings)")
    g.add_argument("--embedding-model", default=DEFAULT_MODEL)
    g.add_argument("--chunking", help="JSON chunking config")
    g.add_argument("--mode", choices=MODES, default="dense", help="retrieval mode (D-028)")
    g.add_argument("-k", type=int, default=5)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("-v", "--verbose", action="store_true")
    g.set_defaults(func=cmd_generate)

    pr = sub.add_parser("prompts", help="list prompts; `lock` releases new versions")
    pr.add_argument("action", choices=["check", "lock"], nargs="?", default="check")
    pr.set_defaults(func=cmd_prompts)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
