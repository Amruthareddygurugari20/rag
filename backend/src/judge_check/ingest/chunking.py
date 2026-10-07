"""Configurable chunking. A chunk is a character span ``[start, end)`` of its document.

Chunks are spans, not copies of the text with their own whitespace rules. So
``document.text[start:end] == chunk.text`` always holds, and a gold-evidence span can be
mapped onto any chunking by interval overlap (D-014).

Strategies (choose with a config object, e.g. from JSON
``{"strategy": "sentence_window", "sentences_per_chunk": 3}``):

- ``document``: one chunk per document.
- ``sentence_window``: N consecutive sentences, optionally overlapping by M sentences.
  Uses the document's own sentence spans if it has them (HotpotQA does), else a regex
  sentence splitter.
- ``fixed_words``: N whitespace-delimited words, overlapping by M words. Ignores sentences.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

Span = tuple[int, int]


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def label(self) -> str:
        params = ",".join(f"{k}={v}" for k, v in self.model_dump().items() if k != "strategy")
        return f"{self.strategy}({params})"  # type: ignore[attr-defined]


class WholeDocument(_Config):
    strategy: Literal["document"] = "document"


class SentenceWindow(_Config):
    strategy: Literal["sentence_window"] = "sentence_window"
    sentences_per_chunk: int = Field(default=3, ge=1)
    overlap: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> SentenceWindow:
        if self.overlap >= self.sentences_per_chunk:
            raise ValueError("overlap must be smaller than sentences_per_chunk")
        return self


class FixedWords(_Config):
    strategy: Literal["fixed_words"] = "fixed_words"
    words_per_chunk: int = Field(default=128, ge=1)
    overlap: int = Field(default=32, ge=0)

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> FixedWords:
        if self.overlap >= self.words_per_chunk:
            raise ValueError("overlap must be smaller than words_per_chunk")
        return self


ChunkingConfig = Annotated[
    WholeDocument | SentenceWindow | FixedWords, Field(discriminator="strategy")
]
_adapter: TypeAdapter[ChunkingConfig] = TypeAdapter(ChunkingConfig)

# The default for the demo corpus. HotpotQA paragraphs are ~4 sentences, so this usually
# makes 1-2 chunks per paragraph. Fine-grained enough that "cites the wrong chunk" (stage 3,
# variant 9) means something, coarse enough that a chunk is readable on its own.
DEFAULT_CHUNKING: ChunkingConfig = SentenceWindow(sentences_per_chunk=2, overlap=0)


def parse_chunking(data: dict | str) -> ChunkingConfig:
    if isinstance(data, str):
        return _adapter.validate_json(data)
    return _adapter.validate_python(data)


# --- sentence splitting (fallback when a document brings no sentence spans) -------------

# A candidate boundary: terminal punctuation, optional closing quotes/brackets, whitespace.
_BOUNDARY = re.compile(r"[.!?][\"'”’)\]]*\s+")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "inc", "ltd",
    "co", "corp", "no", "vol", "fig", "e.g", "i.e", "approx", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}  # fmt: skip
# Single letters with optional dotted continuation: "J" (J. K. Rowling), "U.S", "D.C".
_INITIALS = re.compile(r"[A-Za-z](?:\.[A-Za-z])*")


def _strip(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def split_sentences(text: str) -> list[Span]:
    """Rule-based splitter. Deliberately simple and deterministic, with known limits.

    Splits after . ! ? (plus closing quotes/brackets) followed by whitespace, unless the
    word before the period is a common abbreviation or a single letter (initials such as
    "J. K. Rowling", "U.S."), or the next character is lower-case.
    """
    spans: list[Span] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        nxt = text[m.end() : m.end() + 1]
        if nxt and nxt.islower():
            continue
        if text[m.start()] == ".":
            # The token before the period, keeping internal periods: "U.S", "e.g", "Smith".
            word = re.search(r"[\w.]+$", text[start : m.start()])
            w = word.group(0) if word else ""
            if w.lower() in _ABBREVIATIONS or _INITIALS.fullmatch(w):
                continue
        span = _strip(text, start, m.end())
        if span:
            spans.append(span)
        start = m.end()
    tail = _strip(text, start, len(text))
    if tail:
        spans.append(tail)
    return spans


# --- strategies ----------------------------------------------------------------------------


def _windows(units: list[Span], size: int, overlap: int) -> list[Span]:
    """Group consecutive units into windows of `size`, stepping by size - overlap.

    Stops at the first window that reaches the last unit, so the tail is never a window
    entirely contained in the previous one.
    """
    out: list[Span] = []
    step = size - overlap
    i = 0
    while i < len(units):
        window = units[i : i + size]
        out.append((window[0][0], window[-1][1]))
        if i + size >= len(units):
            break
        i += step
    return out


def chunk_spans(
    text: str, config: ChunkingConfig, sentence_spans: list[Span] | None = None
) -> list[Span]:
    """Character spans of the chunks of one document, in document order."""
    if isinstance(config, WholeDocument):
        span = _strip(text, 0, len(text))
        return [span] if span else []
    if isinstance(config, SentenceWindow):
        sents = sentence_spans if sentence_spans is not None else split_sentences(text)
        return _windows(list(sents), config.sentences_per_chunk, config.overlap)
    if isinstance(config, FixedWords):
        words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return _windows(words, config.words_per_chunk, config.overlap)
    raise TypeError(f"unknown chunking config {config!r}")
