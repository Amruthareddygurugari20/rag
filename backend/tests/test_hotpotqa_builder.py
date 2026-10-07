"""The demo subset is ground truth, so its builder is tested like the mutators will be."""

import json
from collections import Counter

import pytest

from judge_check.datasets import hotpotqa as hp


def example(
    qid: str,
    answer: str = "1912",
    facts: list | None = None,
    context: list | None = None,
    qtype: str = "bridge",
) -> dict:
    return {
        "_id": qid,
        "question": f"question {qid}?",
        "answer": answer,
        "type": qtype,
        "level": "hard",
        "supporting_facts": facts if facts is not None else [["A", 0], ["B", 1]],
        "context": context
        if context is not None
        else [
            ["A", ["Alpha was founded by Ann.", " It is old."]],
            ["B", ["Beta is a city.", " Ann was born in 1912."]],
            ["C", ["Gamma is a distractor."]],
        ],
    }


def test_build_paragraph_spans_slice_back_to_sentences() -> None:
    sents = ["First one.", " Second one.", "Third without space.", "   ", " Fifth."]
    text, spans = hp.build_paragraph(sents)
    assert text == "First one. Second one. Third without space.    Fifth."
    assert [text[s:e] for s, e in spans] == [
        "First one.",
        "Second one.",
        "Third without space.",
        "",
        "Fifth.",
    ]
    assert len(spans) == len(sents)  # index-aligned with HotpotQA sentence ids


@pytest.mark.parametrize(
    ("answer", "evidence", "expected"),
    [
        ("1912", "Ann was born in 1912.", True),
        ("The Beatles", "a song by beatles", True),  # articles and case are normalised
        ("art", "a party was held", False),  # whole words only
        ("1,500", "about 1500 people", True),  # punctuation removed on both sides
        ("", "anything", False),
    ],
)
def test_contains_answer(answer: str, evidence: str, expected: bool) -> None:
    assert hp.contains_answer(answer, evidence) is expected


@pytest.mark.parametrize(
    ("ex", "reason"),
    [
        (example("q", answer="yes"), hp.YES_NO),
        (example("q", facts=[["A", 0], ["B", 7]]), hp.BAD_SUPPORTING_FACT),
        (example("q", facts=[["A", 0], ["Z", 0]]), hp.BAD_SUPPORTING_FACT),
        (example("q", facts=[["A", 0], ["A", 1]]), hp.NOT_TWO_GOLD),
        (example("q", answer="1913"), hp.ANSWER_NOT_IN_EVIDENCE),
        (
            example("q", context=[["A", ["x."]], ["A", ["y."]], ["B", ["z", "1912"]]]),
            hp.TITLE_CONFLICT,
        ),
        (example("q"), None),
    ],
)
def test_check_example_reasons(ex: dict, reason: str | None) -> None:
    assert hp.check_example(ex)[0] == reason


def test_gold_evidence_spans_point_at_supporting_sentences() -> None:
    docs, questions, _ = hp.build_subset([example("q1")], n=1)
    texts = {d["id"]: d["text"] for d in docs}
    ev = questions[0]["gold_evidence"]
    assert [texts[e["document_id"]][e["start"] : e["end"]] for e in ev] == [
        "Alpha was founded by Ann.",
        "Ann was born in 1912.",
    ]
    assert questions[0]["metadata"]["distractor_titles"] == ["C"]


def test_title_conflict_across_examples_is_never_selected_together() -> None:
    clash = example(
        "q2",
        context=[["A", ["Different text 1912."]], ["B", ["b", "1912"]]],
        facts=[["A", 0], ["B", 1]],
    )
    exs = [example("q1"), clash, *(example(f"q{i}") for i in range(3, 6))]
    # Five eligible questions, but q1 and q2 disagree on what "A" says: at most four fit.
    with pytest.raises(ValueError, match="only 4 eligible"):
        hp.build_subset(exs, n=5)
    docs, questions, _ = hp.build_subset(exs, n=4)
    assert not {"q1", "q2"} <= {q["id"] for q in questions}
    assert len({d["id"] for d in docs}) == len(docs)


def test_build_is_deterministic_and_order_independent() -> None:
    exs = [example(f"q{i}") for i in range(20)]
    a = hp.build_subset(exs, n=5)
    b = hp.build_subset(list(reversed(exs)), n=5)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_too_few_eligible_raises() -> None:
    with pytest.raises(ValueError, match="only 1 eligible"):
        hp.build_subset([example("q1")], n=2)


def test_seeded_permutation_is_a_reproducible_permutation() -> None:
    items = list(range(50))
    a = hp.seeded_permutation(items, seed=1)
    assert sorted(a) == items
    assert a == hp.seeded_permutation(items, seed=1)
    assert a != hp.seeded_permutation(items, seed=2)
    assert items == list(range(50))  # input not mutated


def test_seeded_permutation_is_uniform_over_first_position() -> None:
    # Over many seeds, each of 4 items should come first ~25% of the time.
    counts = Counter(hp.seeded_permutation(["a", "b", "c", "d"], seed=s)[0] for s in range(8000))
    assert all(0.22 < c / 8000 < 0.28 for c in counts.values()), counts


def test_sample_depends_on_seed_and_records_it() -> None:
    exs = [example(f"q{i:02d}") for i in range(40)]
    _, q1, stats = hp.build_subset(exs, n=10, seed=1)
    _, q2, _ = hp.build_subset(exs, n=10, seed=2)
    assert {q["id"] for q in q1} != {q["id"] for q in q2}
    assert stats["sampling"]["seed"] == 1
    assert stats["eligible_population"] == 40
    assert stats["eligible_by_type"] == {"bridge": 40}
