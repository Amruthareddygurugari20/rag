"""Chunking invariants, checked on hand-picked cases and on random text (Hypothesis)."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from judge_check.ingest.chunking import (
    DEFAULT_CHUNKING,
    FixedWords,
    SentenceWindow,
    WholeDocument,
    chunk_spans,
    parse_chunking,
    split_sentences,
)


def texts(text: str, spans: list[tuple[int, int]]) -> list[str]:
    return [text[s:e] for s, e in spans]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two! Three?", ["One.", "Two!", "Three?"]),
        ("He was born in 1912. He died.", ["He was born in 1912.", "He died."]),
        ("J. K. Rowling wrote it. Yes.", ["J. K. Rowling wrote it.", "Yes."]),
        ("The U.S. Army fought. Then.", ["The U.S. Army fought.", "Then."]),
        ("Dr. Smith arrived. Mr. Jones left.", ["Dr. Smith arrived.", "Mr. Jones left."]),
        ('She said "Go." Then left.', ['She said "Go."', "Then left."]),
        ("approx. ten people came.", ["approx. ten people came."]),
        ("ends with i.e. lower case", ["ends with i.e. lower case"]),
        ("  padded.   Text  ", ["padded.", "Text"]),
        ("", []),
        ("   ", []),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert texts(text, split_sentences(text)) == expected


def test_sentence_window_uses_document_spans_when_given() -> None:
    text = "A b. C d. E f."
    # Deliberately unusual segmentation: the document's own spans win over the regex.
    own = [(0, 9), (10, 14)]
    assert texts(text, chunk_spans(text, SentenceWindow(sentences_per_chunk=1), own)) == [
        "A b. C d.",
        "E f.",
    ]


def test_sentence_window_overlap_and_tail() -> None:
    text = "S1. S2. S3. S4. S5."
    cfg = SentenceWindow(sentences_per_chunk=2, overlap=1)
    assert texts(text, chunk_spans(text, cfg)) == ["S1. S2.", "S2. S3.", "S3. S4.", "S4. S5."]
    cfg = SentenceWindow(sentences_per_chunk=3, overlap=0)
    assert texts(text, chunk_spans(text, cfg)) == ["S1. S2. S3.", "S4. S5."]


def test_fixed_words() -> None:
    text = "a b c d e f g"
    cfg = FixedWords(words_per_chunk=3, overlap=1)
    assert texts(text, chunk_spans(text, cfg)) == ["a b c", "c d e", "e f g"]


def test_whole_document_strips_outer_whitespace() -> None:
    assert texts("  hello world \n", chunk_spans("  hello world \n", WholeDocument())) == [
        "hello world"
    ]


def test_parse_chunking_round_trip_and_label() -> None:
    cfg = parse_chunking('{"strategy": "fixed_words", "words_per_chunk": 50, "overlap": 10}')
    assert cfg == FixedWords(words_per_chunk=50, overlap=10)
    assert cfg.label() == "fixed_words(words_per_chunk=50,overlap=10)"
    assert parse_chunking(DEFAULT_CHUNKING.model_dump()) == DEFAULT_CHUNKING


@pytest.mark.parametrize(
    "bad",
    [
        {"strategy": "sentence_window", "sentences_per_chunk": 2, "overlap": 2},
        {"strategy": "fixed_words", "words_per_chunk": 0},
        {"strategy": "nope"},
        {"strategy": "document", "unexpected": 1},
    ],
)
def test_invalid_configs_rejected(bad: dict) -> None:
    with pytest.raises(ValidationError):
        parse_chunking(bad)


# --- properties on random text ------------------------------------------------------------

# Random text built from characters plus multi-character fragments that hit the splitter's
# special cases (abbreviations, initials, sentence starts).
random_text = st.lists(
    st.sampled_from([*"abcXYZ019 .!?\n\"')", "Mr. ", "U.S. ", ". A", "\t"]), max_size=150
).map("".join)
configs = st.one_of(
    st.just(WholeDocument()),
    st.integers(1, 5).flatmap(
        lambda n: st.integers(0, n - 1).map(
            lambda o: SentenceWindow(sentences_per_chunk=n, overlap=o)
        )
    ),
    st.integers(1, 8).flatmap(
        lambda n: st.integers(0, n - 1).map(lambda o: FixedWords(words_per_chunk=n, overlap=o))
    ),
)


@settings(max_examples=400, deadline=None)
@given(text=random_text, cfg=configs)
def test_chunk_invariants(text: str, cfg) -> None:
    spans = chunk_spans(text, cfg)
    for s, e in spans:
        # Non-empty, in range, no leading/trailing whitespace.
        assert 0 <= s < e <= len(text)
        assert not text[s].isspace() and not text[e - 1].isspace()
    # Document order, strictly advancing (no duplicate or contained-in-previous chunk).
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(spans, spans[1:], strict=False))
    if getattr(cfg, "overlap", 0) == 0:
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))
    # Coverage: every non-whitespace character lands in some chunk.
    covered = set()
    for s, e in spans:
        covered.update(range(s, e))
    assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())


@settings(max_examples=300, deadline=None)
@given(text=random_text)
def test_split_sentences_partitions_non_whitespace(text: str) -> None:
    spans = split_sentences(text)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))
    non_ws = [i for i, ch in enumerate(text) if not ch.isspace()]
    covered = {i for s, e in spans for i in range(s, e)}
    assert set(non_ws) <= covered
