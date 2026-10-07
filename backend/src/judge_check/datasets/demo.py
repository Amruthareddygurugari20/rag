"""Locate and load the built-in HotpotQA demo set, and compute gold chunks without a DB."""

from __future__ import annotations

from pathlib import Path

from judge_check.ingest.chunking import DEFAULT_CHUNKING, ChunkingConfig, chunk_spans
from judge_check.ingest.formats import DocumentIn, QuestionIn, load_eval_set

DEMO_DIR = Path(__file__).resolve().parents[4] / "data" / "hotpotqa_demo"


def load_demo() -> tuple[list[DocumentIn], list[QuestionIn]]:
    if not (DEMO_DIR / "corpus.jsonl").exists():
        raise FileNotFoundError(f"demo data not found in {DEMO_DIR}")
    return load_eval_set(DEMO_DIR)


def chunk_in_memory(
    documents: list[DocumentIn], config: ChunkingConfig = DEFAULT_CHUNKING
) -> list[tuple[str, int, int, str]]:
    """[(document_id, start, end, text)] for every chunk, in a stable order."""
    out = []
    for d in documents:
        spans = [tuple(s) for s in d.sentence_spans] if d.sentence_spans else None
        for s, e in chunk_spans(d.text, config, spans):
            out.append((d.id, s, e, d.text[s:e]))
    return out


def gold_indices(question: QuestionIn, chunks: list[tuple[str, int, int, str]]) -> set[int]:
    """Indices into `chunks` that overlap the question's gold evidence (same rule as SQL)."""
    return {
        i
        for i, (doc, s, e, _) in enumerate(chunks)
        for ev in question.gold_evidence
        if doc == ev.document_id and s < ev.end and ev.start < e
    }
