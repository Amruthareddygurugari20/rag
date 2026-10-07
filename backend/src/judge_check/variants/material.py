"""What a question gives the constructors, and the base text built from it (D-035).

Everything here is plain data, so the constructors can be tested without a database.
`variants.build` fills it from the database for one chunk set.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sentence:
    text: str
    chunk_ids: tuple[int, ...]  # chunks of the chunk set overlapping this sentence


@dataclass(frozen=True)
class Paragraph:
    title: str
    sentences: tuple[Sentence, ...]  # every sentence, in document order
    gold: frozenset[int]  # indices of the supporting (gold) sentences


@dataclass(frozen=True)
class Material:
    key: str  # question external id: part of every seed
    answer: str
    gold: tuple[Sentence, ...]  # gold sentences, in evidence order
    gold_paragraphs: tuple[Paragraph, ...]
    distractor_chunks: tuple[tuple[int, str], ...]  # (chunk id, text), distractor paragraphs
    entity_candidates: tuple[str, ...]  # distractor titles, disambiguation stripped

    @property
    def gold_texts(self) -> list[str]:
        return [s.text for s in self.gold]

    @property
    def gold_chunk_ids(self) -> tuple[int, ...]:
        return cite(self.gold)


def cite(sentences: tuple[Sentence, ...] | list[Sentence]) -> tuple[int, ...]:
    """Chunk ids covering the sentences, first-seen order, no repeats."""
    return tuple(dict.fromkeys(c for s in sentences for c in s.chunk_ids))


def lead(answer: str) -> str:
    """The answer as an opening sentence: "Bergen." (no doubled full stop)."""
    a = answer.strip()
    return a if a.endswith((".", "!", "?")) else f"{a}."


def base_text(m: Material) -> str:
    """Type 1, and the source of types 5, 7b, 9: the reference answer, then the gold
    sentences verbatim. Correct exactly to the extent HotpotQA's labels are."""
    return " ".join([lead(m.answer), *m.gold_texts])
