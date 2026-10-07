"""Deterministic mutators and their guards (D-035).

Oracles are written independently of the implementation:
- expected mutation results are rebuilt here from known insertion offsets (not find_spans);
- "is the original gone?" is checked by a token-sequence scan written here (not
  contains_answer);
- replacement types are checked by regexes written here (not valuetypes).
"""

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from judge_check.datasets.demo import load_demo
from judge_check.variants.guards import GuardFailure, mutation_guards, require
from judge_check.variants.mutate import (
    find_spans,
    replacement_for,
    rng_for,
    strip_disambiguation,
    substitute,
)
from judge_check.variants.valuetypes import VType, vtype

# --- typing: the real problem cases from the demo answers, plus the specification ---------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1912", VType.YEAR),
        ("867", VType.NUMBER),  # 3 digits: a number, not a year
        ("1,693", VType.NUMBER),
        ("25 million", VType.NUMBER),
        ("400 members", VType.NUMBER),
        ("June 4, 1931", VType.DATE),
        ("4 June 1931", VType.DATE),
        ("Umberto II", VType.ENTITY),
        ("The Monster", VType.ENTITY),
        ("Isabella II of Jerusalem", VType.ENTITY),
        # Untyped on purpose: a loose parser would mis-type these.
        ("2016 United States elections", None),
        ("575 acres (2.08 km²)", None),
        ("729 at the 2010 census", None),
        ("1861–65", None),
        ("early 1970s", None),
        ("(foaled February 1, 1999)", None),
        ("Los Angeles Xtreme, San Francisco Demons and Memphis Maniax", None),
        ("musician", None),
        ("1,69", None),  # malformed thousands separator
        ("June 31, 1931 and later", None),
    ],
)
def test_vtype_is_strict(value: str, expected) -> None:
    assert vtype(value) is expected


def test_demo_answer_types_are_reported() -> None:
    _, qs = load_demo()
    counts: dict = {}
    for q in qs:
        counts[vtype(q.reference_answer)] = counts.get(vtype(q.reference_answer), 0) + 1
    # Not a target, a record: how many answers each mutation can even attempt.
    print("\ndemo answer types:", {str(k): v for k, v in counts.items()})
    assert sum(counts.values()) == 200


# --- independent oracles --------------------------------------------------------------------

_NORM = re.compile(r"[^\w\s]")


def tokens(s: str) -> list[str]:
    """Oracle normalisation: lowercase, drop punctuation and articles, split."""
    return [t for t in _NORM.sub("", s.lower()).split() if t not in {"a", "an", "the"}]


def contains_seq(haystack: str, needle: str) -> bool:
    h, n = tokens(haystack), tokens(needle)
    return bool(n) and any(h[i : i + len(n)] == n for i in range(len(h) - len(n) + 1))


ORACLE_TYPE = {
    VType.YEAR: re.compile(r"^(1\d\d\d|20\d\d)$"),
    VType.NUMBER: re.compile(r"^[\d,]+(\.\d+)?( \S+)?$"),
    VType.DATE: re.compile(r"^(\w+ \d{1,2}, \d{4}|\d{1,2} \w+ \d{4})$"),
}


# --- mutation properties on generated sentences ---------------------------------------------

words = st.lists(st.sampled_from(["the", "team", "was", "in", "won", "city", "a", "of"]),
                 min_size=0, max_size=6).map(" ".join)  # fmt: skip
values = st.one_of(
    st.integers(1000, 2099).map(lambda y: (str(y), VType.YEAR)),
    st.integers(1, 999_999).map(lambda n: (f"{n:,}", VType.NUMBER)),
    st.tuples(st.sampled_from(["March", "July", "October"]), st.integers(1, 28),
              st.integers(1900, 2020)).map(lambda t: (f"{t[0]} {t[1]}, {t[2]}", VType.DATE)),
)  # fmt: skip


@settings(max_examples=300, deadline=None)
@given(prefix=words, suffix=words, value=values, seed=st.integers(0, 10_000))
def test_mutation_replaces_exactly_the_value(prefix, suffix, value, seed) -> None:
    original, kind = value
    text = f"{prefix} {original} {suffix}".strip() + "."
    start = text.index(original)  # oracle: the known insertion point
    rep = replacement_for(original, kind, rng_for(seed, "q", "t5"), [])
    sub = substitute(text, find_spans(text, original), rep)
    # Independently rebuilt expectation.
    assert sub.result == text[:start] + rep + text[start + len(original) :]
    assert ORACLE_TYPE[kind].match(rep), rep
    assert tokens(rep) != tokens(original)
    assert not contains_seq(sub.result, original) or contains_seq(rep, original)


@pytest.mark.parametrize("year", ["1000", "1001", "2099", "2098"])
def test_year_at_range_edge_still_changes(year: str) -> None:
    for seed in range(60):  # covers every offset sign
        rep = replacement_for(year, VType.YEAR, rng_for(seed, "edge"), [])
        assert rep != year and 1000 <= int(rep) <= 2099


@pytest.mark.parametrize("n", ["1", "2", "0", "3"])
def test_small_numbers_still_change_or_refuse(n: str) -> None:
    for seed in range(30):
        try:
            rep = replacement_for(n, VType.NUMBER, rng_for(seed, "small"), [])
        except LookupError:
            assert n == "0"  # 0 x anything is 0: correctly refused, never a fake mutation
            continue
        assert rep != n


def test_replacements_are_seeded_and_reproducible() -> None:
    a = replacement_for("1912", VType.YEAR, rng_for(1, "q1", "t5"), [])
    b = replacement_for("1912", VType.YEAR, rng_for(1, "q1", "t5"), [])
    c = [replacement_for("1912", VType.YEAR, rng_for(s, "q1", "t5"), []) for s in range(30)]
    assert a == b and len(set(c)) > 5


def test_entity_replacement_comes_from_typed_candidates() -> None:
    cands = ["Derek Dooley", "musician", "1984", "Zach Azzanni"]
    rep = replacement_for("Jimmy Robinson", VType.ENTITY, rng_for(0, "q", "t5"), cands)
    assert rep in {"Derek Dooley", "Zach Azzanni"}  # untyped/other-typed candidates excluded
    with pytest.raises(LookupError):
        replacement_for("Jimmy Robinson", VType.ENTITY, rng_for(0, "q"), ["musician"])


def test_strip_disambiguation() -> None:
    assert strip_disambiguation("Derek Dooley (American football)") == "Derek Dooley"
    assert strip_disambiguation("Slade") == "Slade"


# --- guards: each one must catch a planted violation ----------------------------------------

GOLD = ["Ann Lee was born in 1912 in Bergen.", "Alpha Corp was founded in 1930."]
BASE = "1912. Ann Lee was born in 1912 in Bergen."


def failed(results) -> list[str]:
    return [r.name for r in results if not r.passed]


def test_clean_mutation_passes_all_guards() -> None:
    sub = substitute(BASE, find_spans(BASE, "1912"), "1907")
    report = require(mutation_guards(sub, "1912", VType.YEAR, GOLD))
    assert [r["name"] for r in report] == [
        "different", "same_type", "absent_from_gold", "original_removed", "rest_unchanged",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("replacement", "spans_from", "tamper", "expected_failure"),
    [
        ("1912", "1912", None, "different"),
        ("Bergen", "1912", None, "same_type"),
        # 1930 is another TRUE year in the gold evidence: the "wrong" answer would be supported.
        ("1930", "1912", None, "absent_from_gold"),
        # Only the first occurrence replaced: the text still asserts 1912.
        ("1907", "first_only", None, "original_removed"),
        ("1907", "1912", "Bergen->Oslo", "rest_unchanged"),
    ],
)
def test_each_guard_catches_its_violation(replacement, spans_from, tamper, expected_failure):
    spans = find_spans(BASE, "1912")
    if spans_from == "first_only":
        spans = spans[:1]
    sub = substitute(BASE, spans, replacement)
    if tamper:
        from dataclasses import replace

        sub = replace(sub, result=sub.result.replace("Bergen", "Oslo"))
    results = mutation_guards(sub, "1912", VType.YEAR, GOLD)
    assert expected_failure in failed(results)
    with pytest.raises(GuardFailure) as exc:
        require(results)
    assert exc.value.first_failed == failed(results)[0]


def test_no_spans_is_not_a_mutation() -> None:
    sub = substitute(BASE, [], "1907")
    assert "rest_unchanged" in failed(mutation_guards(sub, "1912", VType.YEAR, GOLD))
