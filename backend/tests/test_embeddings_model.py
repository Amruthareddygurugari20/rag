"""Tests with the real BGE model. Skipped where the model can't be downloaded, required in CI.

The ablation test is the one to read. It measures what the BGE query instruction is worth
on our own demo corpus, instead of trusting the model card.
"""

import json
import os

import numpy as np
import pytest

from judge_check.datasets.demo import chunk_in_memory, gold_indices, load_demo
from judge_check.embeddings import BGE_QUERY_INSTRUCTION, DEFAULT_MODEL
from judge_check.retrieval.evaluate import paired_difference

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder():
    try:
        from judge_check.embeddings import get_embedder

        return get_embedder(DEFAULT_MODEL)
    except Exception as exc:  # no torch, no network, no cached model...
        msg = f"embedding model unavailable: {type(exc).__name__}: {exc}"
        if os.environ.get("JUDGE_CHECK_REQUIRE_MODEL") == "1":
            pytest.fail(msg)
        pytest.skip(msg)


def test_shape_and_unit_norm(embedder) -> None:
    v = embedder.embed_passages(["Ann Lee was born in Bergen.", "Gamma is a river."])
    assert v.shape == (2, 384)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_instruction_changes_queries_not_passages(embedder) -> None:
    text = "Where was Ann Lee born?"
    with_instr = embedder.embed_queries([text])[0]
    without = embedder.embed_queries([text], use_instruction=False)[0]
    # The instruction moves the query vector...
    assert float(with_instr @ without) < 0.999
    # ...and is exactly "prefix + text", nothing else.
    prefixed = embedder.embed_passages([BGE_QUERY_INSTRUCTION + text])[0]
    assert np.allclose(with_instr, prefixed, atol=1e-5)
    # Passages never get it: same text, same vector as a no-instruction query.
    assert np.allclose(embedder.embed_passages([text])[0], without, atol=1e-5)


def _metrics(sims: np.ndarray, gold: list[set[int]], k: int) -> dict[str, float]:
    order = np.argsort(-sims, axis=1, kind="stable")
    recall = complete = mrr = 0.0
    for row, g in zip(order, gold, strict=True):
        top = set(row[:k].tolist())
        recall += len(g & top) / len(g)
        complete += float(g <= top)
        first = next(i for i, c in enumerate(row.tolist(), 1) if c in g)
        mrr += 1.0 / first
    n = len(gold)
    return {f"recall@{k}": recall / n, f"complete@{k}": complete / n, "mrr": mrr / n}


def test_query_instruction_ablation_on_demo_corpus(embedder) -> None:
    documents, questions = load_demo()
    chunks = chunk_in_memory(documents)
    passages = embedder.embed_passages([c[3] for c in chunks])
    gold = [gold_indices(q, chunks) for q in questions]
    texts = [q.question for q in questions]

    results = {}
    for label, use in [("with_instruction", True), ("without_instruction", False)]:
        queries = embedder.embed_queries(texts, use_instruction=use)
        sims = queries @ passages.T  # unit vectors: inner product == cosine
        results[label] = _metrics(sims, gold, k=5)
        # Mean similarity of a query to its gold chunks vs to everything else.
        gold_mask = np.zeros_like(sims, dtype=bool)
        for i, g in enumerate(gold):
            gold_mask[i, list(g)] = True
        results[label]["mean_sim_gold"] = float(sims[gold_mask].mean())
        results[label]["mean_sim_other"] = float(sims[~gold_mask].mean())

    # Paired over the same questions: per-question reciprocal ranks with vs without.
    rr = {}
    for label, use in [("with", True), ("without", False)]:
        sims = embedder.embed_queries(texts, use_instruction=use) @ passages.T
        order = np.argsort(-sims, axis=1, kind="stable")
        rr[label] = [
            1.0 / next(i for i, c in enumerate(row.tolist(), 1) if c in g)
            for row, g in zip(order, gold, strict=True)
        ]
    d, se = paired_difference(rr["without"], rr["with"])
    results["paired_mrr_with_minus_without"] = {"diff": d, "se": se, "z": d / se}
    print("\nquery-instruction ablation:", json.dumps(results, indent=2))
    w, wo = results["with_instruction"], results["without_instruction"]
    # Sanity: retrieval works at all in both settings.
    assert w["recall@5"] > 0.5 and wo["recall@5"] > 0.5
    # Gold chunks are closer than other chunks in both settings.
    assert w["mean_sim_gold"] > w["mean_sim_other"]
    assert wo["mean_sim_gold"] > wo["mean_sim_other"]
