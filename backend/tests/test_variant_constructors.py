"""Type-5 construction and type-level balancing (D-035 and addenda A-C)."""

from collections import Counter
from dataclasses import dataclass

import pytest

from judge_check.variants.balance import balance_by_type
from judge_check.variants.constructors import wrong_detail
from judge_check.variants.valuetypes import vtype

GOLD = ["Ann Lee was born in 1912 in Bergen.", "Alpha Corp was founded by Ann Lee in 1930."]


def build(base: str, answer: str, gold=GOLD, cands=None, seed=0):
    return wrong_detail(base, answer, gold, cands or [], seed=seed, question_key="q1")


# --- D-035 addendum C ------------------------------------------------------------------------


def test_untyped_answer_is_never_partially_mutated() -> None:
    answer = "2016 United States elections"
    assert vtype(answer) is None
    base = "2016 United States elections. Clinton ran in the 2016 United States elections."
    out = build(base, answer, gold=[base])
    assert out.discarded and out.discard_reason == "answer_type_unparseable"
    assert out.text is None and out.sub_kind is None  # no "2017 United States elections"


# --- type 5 outcomes -------------------------------------------------------------------------


def test_year_mutation_is_labelled_wrong_with_full_guard_report() -> None:
    out = build("1912. Ann Lee was born in 1912 in Bergen.", "1912")
    assert not out.discarded and out.is_correct is False and out.sub_kind == "year"
    rep = out.params["replacement"]
    assert rep != "1912" and "1912" not in out.text and rep in out.text
    assert rep not in " ".join(GOLD)  # oracle: plain substring, stricter than the guard
    assert {g["name"] for g in out.guard_report} == {
        "different", "same_type", "absent_from_gold", "original_removed", "rest_unchanged",
    }  # fmt: skip


def test_entity_mutation_uses_on_topic_candidates() -> None:
    out = build("Bergen. Ann Lee was born in Bergen.", "Bergen", cands=["Oslo", "Trondheim"])
    assert out.sub_kind == "entity" and out.params["replacement"] in {"Oslo", "Trondheim"}


def test_candidate_that_is_true_in_gold_is_skipped_or_discarded() -> None:
    # The only candidate, "Ann Lee", appears in the gold evidence: swapping it in could make
    # the "wrong" answer supported. Every attempt must fail absent_from_gold -> discard.
    out = build("Bergen. Ann Lee was born in Bergen.", "Bergen", cands=["Ann Lee"])
    assert out.discarded and out.discard_reason == "guard_failed:absent_from_gold"
    assert out.guard_report  # the rejected attempts are recorded


def test_answer_missing_from_base_is_discarded() -> None:
    out = build("Some unrelated text.", "1912")
    assert out.discard_reason == "answer_not_in_base_text"


def test_construction_is_deterministic() -> None:
    a = build("1912. Born in 1912.", "1912", gold=["Born in 1912."], seed=7)
    b = build("1912. Born in 1912.", "1912", gold=["Born in 1912."], seed=7)
    assert a == b


# --- D-035 addendum B: balance by TYPE only --------------------------------------------------


@dataclass(frozen=True)
class V:
    vid: int
    variant_type: str
    sub_kind: str | None = None


def test_balancing_does_not_floor_on_the_rarest_subkind() -> None:
    pool = (
        [V(0, "right_topic_wrong_detail", "date")]
        + [V(i, "right_topic_wrong_detail", "entity") for i in range(1, 91)]
        + [V(100 + i, "right_answer_wrong_citation") for i in range(91)]
    )
    out = balance_by_type(pool, lambda v: v.variant_type, seed=1)
    kept_t5 = [v for v in out.kept if v.variant_type == "right_topic_wrong_detail"]
    # Floor is the smallest TYPE (91), not the single date sub-kind (1).
    assert len(kept_t5) == 91
    assert Counter(v.sub_kind for v in kept_t5) == {"entity": 90, "date": 1}


def test_balancing_floors_each_group_at_its_smallest_type() -> None:
    pool = (
        [V(i, "correct") for i in range(50)]
        + [V(100 + i, "terse_correct") for i in range(20)]
        + [V(200 + i, "right_answer_wrong_citation") for i in range(30)]
        + [V(300 + i, "partially_correct") for i in range(40)]
    )
    out = balance_by_type(pool, lambda v: v.variant_type, seed=3)
    assert out.after == {"correct": 20, "terse_correct": 20,
                         "right_answer_wrong_citation": 30, "partially_correct": 30}  # fmt: skip
    assert "verbose_correct" in out.missing_types["accept"]
    again = balance_by_type(pool, lambda v: v.variant_type, seed=3)
    assert [v.vid for v in again.kept] == [v.vid for v in out.kept]  # seeded


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown variant types"):
        balance_by_type([V(1, "correct_ish")], lambda v: v.variant_type, seed=0)
