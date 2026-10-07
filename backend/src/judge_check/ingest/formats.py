"""The generic input format: two JSONL files, so any evaluation set can be loaded.

corpus.jsonl, one document per line:
    {"id": "Ann Arbor", "title": "Ann Arbor", "text": "...",
     "sentence_spans": [[0, 57], [58, 120]]}      # optional; character [start, end)

questions.jsonl, one question per line:
    {"id": "q1", "question": "...", "reference_answer": "...",
     "gold_evidence": [{"document_id": "Ann Arbor", "start": 0, "end": 57}],
     "metadata": {...}}                              # optional, stored as-is

Gold evidence is a character span in a document, not a chunk id. Chunk ids only exist after
chunking, and chunking is configurable. Gold chunks are derived per chunking (D-014).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class DocumentIn(BaseModel):
    id: str = Field(min_length=1)
    title: str = ""
    text: str = Field(min_length=1)
    sentence_spans: list[tuple[int, int]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _spans_valid(self) -> DocumentIn:
        if self.sentence_spans is not None:
            prev_end = 0
            for start, end in self.sentence_spans:
                if not (prev_end <= start < end <= len(self.text)):
                    raise ValueError(
                        f"document {self.id!r}: sentence span ({start}, {end}) must be "
                        f"non-empty, in range, sorted and non-overlapping"
                    )
                prev_end = end
        return self


class EvidenceSpan(BaseModel):
    document_id: str
    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def _non_empty(self) -> EvidenceSpan:
        if self.end <= self.start:
            raise ValueError(f"evidence span ({self.start}, {self.end}) is empty")
        return self


class QuestionIn(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    gold_evidence: list[EvidenceSpan] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                yield lineno, json.loads(line)


def load_eval_set(directory: Path) -> tuple[list[DocumentIn], list[QuestionIn]]:
    """Load and cross-validate corpus.jsonl + questions.jsonl from a directory."""
    documents = []
    for lineno, row in _read_jsonl(directory / "corpus.jsonl"):
        try:
            documents.append(DocumentIn.model_validate(row))
        except ValueError as exc:
            raise ValueError(f"corpus.jsonl line {lineno}: {exc}") from exc
    questions = []
    for lineno, row in _read_jsonl(directory / "questions.jsonl"):
        try:
            questions.append(QuestionIn.model_validate(row))
        except ValueError as exc:
            raise ValueError(f"questions.jsonl line {lineno}: {exc}") from exc

    by_id: dict[str, DocumentIn] = {}
    for doc in documents:
        if doc.id in by_id:
            raise ValueError(f"duplicate document id {doc.id!r}")
        by_id[doc.id] = doc
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise ValueError(f"duplicate question id {q.id!r}")
        seen.add(q.id)
        for ev in q.gold_evidence:
            doc = by_id.get(ev.document_id)
            if doc is None:
                raise ValueError(f"question {q.id!r}: unknown document {ev.document_id!r}")
            if ev.end > len(doc.text):
                raise ValueError(f"question {q.id!r}: evidence span past end of document")
    return documents, questions
